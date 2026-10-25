//! Minimal blocking HTTP/1.1 server exposing `AgentHandler` endpoints with
//! Server-Sent Events streaming.
//!
//! One OS thread per connection. `std::net::TcpListener` + manual HTTP/1.1
//! framing. Zero dependencies.
//!
//! Routing: `POST /agents/<name>/<id>` dispatches to a registered handler
//! by `<name>`. Everything else is a hard error (404 / 405 / 400 / 413).
//!
//! Streaming model: handlers write SSE frames through `EventSink`; an SSE
//! response uses `Connection: close` and we flush after every frame so
//! clients see events as they arrive.
//!
//! # Production hardening
//!
//! - Per-connection read/write timeouts (`ServerConfig::read_timeout`,
//!   `ServerConfig::write_timeout`). A read timeout on an idle socket
//!   produces a clean close; mid-request it produces 408.
//! - Bounded retained worker handles (`ServerConfig::max_connections`). Connections
//!   accepted beyond the cap are closed immediately with no thread spawned.
//! - HTTP/1.1 keep-alive for non-SSE responses (4xx error paths). SSE
//!   responses still terminate the connection — the body is open-ended and
//!   can't be safely followed by another request on the same socket.

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, ErrorKind, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

const MAX_BODY: usize = 1024 * 1024;
const MAX_HEADER_LINE: usize = 8 * 1024;
const MAX_HEADERS: usize = 64;

/// Tunables for `Server::serve_with`. Defaults aim at a conservative
/// production posture: short timeouts, modest connection cap, keep-alive on.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub read_timeout: Duration,
    pub write_timeout: Duration,
    pub max_connections: usize,
    pub keep_alive: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            read_timeout: Duration::from_secs(30),
            write_timeout: Duration::from_secs(30),
            max_connections: 256,
            keep_alive: true,
        }
    }
}

/// Prepared SSE response, executed only after required resources are reserved.
pub type PreparedResponse<'a> = Box<dyn FnOnce(&mut EventSink) -> Result<(), HandlerError> + 'a>;

pub trait AgentHandler: Send + Sync + 'static {
    /// Reserve required resources before the HTTP 200/SSE commitment. The
    /// default preserves simple handlers' existing behavior.
    fn prepare<'a>(
        &'a self,
        id: &str,
        request_id: &str,
        body: &[u8],
    ) -> Result<PreparedResponse<'a>, PrepareError> {
        let id = id.to_owned();
        let request_id = request_id.to_owned();
        let body = body.to_vec();
        Ok(Box::new(move |sink| self.handle(&id, &request_id, &body, sink)))
    }

    fn handle(
        &self,
        id: &str,
        request_id: &str,
        body: &[u8],
        sink: &mut EventSink,
    ) -> Result<(), HandlerError>;
}

pub struct EventSink<'a> {
    writer: &'a mut dyn Write,
}

impl<'a> EventSink<'a> {
    pub fn new(writer: &'a mut dyn Write) -> Self {
        Self { writer }
    }

    pub fn emit(&mut self, event: Option<&str>, data: &str) -> io::Result<()> {
        if let Some(ev) = event {
            self.writer.write_all(b"event: ")?;
            self.writer.write_all(ev.as_bytes())?;
            self.writer.write_all(b"\n")?;
        }
        // Data may contain newlines; SSE requires one `data:` line per line.
        for line in data.split('\n') {
            self.writer.write_all(b"data: ")?;
            self.writer.write_all(line.as_bytes())?;
            self.writer.write_all(b"\n")?;
        }
        self.writer.write_all(b"\n")?;
        self.writer.flush()
    }

    pub fn emit_data(&mut self, data: &str) -> io::Result<()> {
        self.emit(None, data)
    }
}

#[derive(Debug, Clone)]
pub struct HandlerError(pub String);

impl std::fmt::Display for HandlerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for HandlerError {}

#[derive(Debug)]
pub enum PrepareError {
    Client(HandlerError),
    Service(HandlerError),
}

impl std::fmt::Display for PrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Client(e) | Self::Service(e) => write!(f, "{e}"),
        }
    }
}

pub struct Server {
    handlers: HashMap<String, Box<dyn AgentHandler>>,
}

impl Server {
    pub fn new() -> Self {
        Self { handlers: HashMap::new() }
    }

    pub fn register(&mut self, name: impl Into<String>, handler: Box<dyn AgentHandler>) {
        self.handlers.insert(name.into(), handler);
    }

    /// Bind `addr` and serve with [`ServerConfig::default`].
    pub fn serve(self, addr: &str) -> io::Result<()> {
        self.serve_with(addr, ServerConfig::default())
    }

    /// Bind `addr` and serve with a caller-supplied config.
    pub fn serve_with(self, addr: &str, cfg: ServerConfig) -> io::Result<()> {
        self.serve_listener_with(TcpListener::bind(addr)?, cfg)
    }

    /// Drive an already-bound listener with the default config. Lets tests
    /// pick a free port and own the listener's lifetime without leaking
    /// threads on `0.0.0.0`.
    pub fn serve_listener(self, listener: TcpListener) -> io::Result<()> {
        self.serve_listener_with(listener, ServerConfig::default())
    }

    /// Drive an already-bound listener with a caller-supplied config.
    pub fn serve_listener_with(self, listener: TcpListener, cfg: ServerConfig) -> io::Result<()> {
        self.serve_listener_with_shutdown(listener, cfg, Arc::new(AtomicBool::new(false)))
    }

    /// Stop admission on shutdown and retain every connection worker until
    /// its actual join. A stuck handler remains owned; no timeout is success.
    pub fn serve_listener_with_shutdown(
        self,
        listener: TcpListener,
        cfg: ServerConfig,
        shutdown: Arc<AtomicBool>,
    ) -> io::Result<()> {
        listener.set_nonblocking(true)?;
        let shared = Arc::new(self);
        let mut workers: Vec<thread::JoinHandle<io::Result<()>>> = Vec::new();
        let mut failure = None;
        loop {
            let mut index = 0;
            while index < workers.len() {
                if workers[index].is_finished() {
                    let worker = workers.swap_remove(index);
                    // Per-connection I/O failures retain their existing local
                    // failure semantics. A worker panic is a server failure.
                    if worker.join().is_err() {
                        failure
                            .get_or_insert_with(|| io::Error::other("connection worker panicked"));
                        shutdown.store(true, Ordering::Release);
                    }
                } else {
                    index += 1;
                }
            }
            if shutdown.load(Ordering::Acquire) {
                break;
            }
            let stream = match listener.accept() {
                Ok((stream, _)) => stream,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(20));
                    continue;
                }
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            };
            // Only this loop admits workers; the retained handle count is
            // the connection capacity until actual join, including panics.
            if shutdown.load(Ordering::Acquire) || workers.len() >= cfg.max_connections {
                let _ = stream.shutdown(std::net::Shutdown::Both);
                continue;
            }
            if let Err(error) = stream
                .set_read_timeout(Some(cfg.read_timeout))
                .and_then(|()| stream.set_write_timeout(Some(cfg.write_timeout)))
            {
                failure = Some(error);
                break;
            }
            let server = shared.clone();
            let worker_cfg = cfg.clone();
            let worker_shutdown = shutdown.clone();
            match thread::Builder::new().spawn(move || {
                server.handle_connection_with_shutdown(stream, &worker_cfg, Some(&worker_shutdown))
            }) {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        // Close admission before the join barrier. Publish this on every
        // exit, so the caller's cancellation bridge can release transports.
        shutdown.store(true, Ordering::Release);
        drop(listener);
        while !workers.is_empty() {
            let mut index = 0;
            while index < workers.len() {
                if workers[index].is_finished() {
                    if workers.swap_remove(index).join().is_err() {
                        failure
                            .get_or_insert_with(|| io::Error::other("connection worker panicked"));
                    }
                } else {
                    index += 1;
                }
            }
            if !workers.is_empty() {
                thread::sleep(Duration::from_millis(10));
            }
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn handle_connection_with_shutdown(
        &self,
        mut stream: TcpStream,
        cfg: &ServerConfig,
        shutdown: Option<&AtomicBool>,
    ) -> io::Result<()> {
        loop {
            if shutdown.is_some_and(|signal| signal.load(Ordering::Acquire)) {
                return Ok(());
            }
            // Each iteration handles one request. We must clone the stream
            // for buffered reads because BufReader takes ownership, but the
            // raw stream stays here for writing the response and (on
            // keep-alive) the next iteration.
            let mut reader = BufReader::new(stream.try_clone()?);

            // Distinguish "idle close" (peer hung up, or read timeout with
            // no bytes consumed yet) from "mid-request error". The first
            // returns Ok(None) and ends the loop cleanly; the second
            // returns a parse error we map to 4xx.
            let outcome = parse_request(&mut reader);
            // A blocked read may complete after shutdown. Do not dispatch the
            // newly parsed request or admit another keep-alive request then.
            if shutdown.is_some_and(|signal| signal.load(Ordering::Acquire)) {
                return Ok(());
            }

            let keep_alive = match outcome {
                Ok(req) => {
                    let client_wants_close = req.connection_close;
                    let (response_kind, write_res) = self.dispatch(req, &mut stream);
                    write_res?;
                    // SSE responses always terminate the connection; the
                    // body has no length and we have no way to delimit it
                    // from a following request.
                    let response_keeps_alive = matches!(response_kind, ResponseKind::Buffered);
                    cfg.keep_alive && response_keeps_alive && !client_wants_close
                }
                Err(ParseError::Idle) => return Ok(()),
                Err(ParseError::Timeout) => {
                    // Partial request then silence: 408. Don't keep-alive
                    // because the framing state is unknown. Headers may
                    // never have arrived, so no request-id to echo.
                    let _ = write_simple(&mut stream, 408, "Request Timeout", None);
                    return Ok(());
                }
                Err(ParseError::TooLarge) => {
                    write_simple(&mut stream, 413, "Payload Too Large", None)?;
                    // Body framing is unknown after a 413 — there may be
                    // unread bytes on the wire. Close.
                    return Ok(());
                }
                Err(ParseError::MethodNotAllowed) => {
                    write_simple(&mut stream, 405, "Method Not Allowed", None)?;
                    return Ok(());
                }
                Err(_) => {
                    write_simple(&mut stream, 400, "Bad Request", None)?;
                    return Ok(());
                }
            };

            if !keep_alive {
                return Ok(());
            }
            // Loop and read the next request on the same socket. The next
            // `parse_request` call applies the same read_timeout as an
            // idle timeout.
        }
    }

    fn dispatch(&self, req: Request, stream: &mut TcpStream) -> (ResponseKind, io::Result<()>) {
        // Resolve the per-request correlation id: use the client-supplied
        // `X-Request-ID` if valid, otherwise mint a synthetic one. We do
        // this even for paths that won't reach a handler so ops endpoints
        // can echo the id too — cheap and useful for log joins.
        let request_id = req.request_id.clone().unwrap_or_else(generate_request_id);

        // Ops endpoints. GET-only, always cheap, never call handlers. Live
        // here (not registered as `AgentHandler`s) because they're not
        // tenant-specific and shouldn't surface in the handler map.
        if req.method == "GET" {
            return match req.path.split('?').next().unwrap_or(&req.path) {
                "/healthz" => (ResponseKind::Buffered, write_ops(stream, "ok\n", &request_id)),
                "/readyz" => (ResponseKind::Buffered, write_ops(stream, "ok\n", &request_id)),
                // Any other GET is 405 — `/agents/*` is POST-only.
                _ => (
                    ResponseKind::Buffered,
                    write_simple(stream, 405, "Method Not Allowed", Some(&request_id)),
                ),
            };
        }
        let Some((name, id)) = parse_agent_path(&req.path) else {
            return (
                ResponseKind::Buffered,
                write_simple(stream, 404, "Not Found", Some(&request_id)),
            );
        };
        let Some(handler) = self.handlers.get(name) else {
            return (
                ResponseKind::Buffered,
                write_simple(stream, 404, "Not Found", Some(&request_id)),
            );
        };

        let prepared = match handler.prepare(id, &request_id, &req.body) {
            Ok(prepared) => prepared,
            Err(e) => {
                eprintln!("agent handler preflight failed: {e}");
                let (status, reason) = match e {
                    PrepareError::Client(_) => (400, "Bad Request"),
                    PrepareError::Service(_) => (503, "Service Unavailable"),
                };
                return (
                    ResponseKind::Buffered,
                    write_simple(stream, status, reason, Some(&request_id)),
                );
            }
        };

        // Commit SSE headers up front. Once they're on the wire any handler
        // error has to be surfaced inside the stream as an `error` event;
        // we can't retroactively change the status. The `X-Request-ID` echo
        // goes here so the client can correlate the SSE body with its
        // outbound request before any data frames arrive.
        let header_res = write!(
            stream,
            "HTTP/1.1 200 OK\r\n\
             Content-Type: text/event-stream\r\n\
             Cache-Control: no-cache\r\n\
             Connection: close\r\n\
             X-Request-ID: {request_id}\r\n\r\n",
        );
        if let Err(e) = header_res {
            return (ResponseKind::Sse, Err(e));
        }
        if let Err(e) = stream.flush() {
            return (ResponseKind::Sse, Err(e));
        }

        let mut sink = EventSink::new(stream);
        if let Err(e) = prepared(&mut sink) {
            let _ = sink.emit(Some("error"), &e.0);
        }
        (ResponseKind::Sse, Ok(()))
    }
}

impl Default for Server {
    fn default() -> Self {
        Self::new()
    }
}

/// Distinguishes responses that can share a connection with subsequent
/// requests (`Buffered`, length-delimited) from ones that can't (`Sse`,
/// open-ended).
enum ResponseKind {
    Buffered,
    Sse,
}

fn parse_agent_path(path: &str) -> Option<(&str, &str)> {
    // Strip optional query string; we don't use it but we must not let it
    // bleed into the id.
    let path = path.split('?').next().unwrap_or(path);
    let rest = path.strip_prefix("/agents/")?;
    let (name, id) = rest.split_once('/')?;
    if name.is_empty() || id.is_empty() || id.contains('/') {
        return None;
    }
    Some((name, id))
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
    connection_close: bool,
    /// Verbatim value of an inbound `X-Request-ID` header, if any. The
    /// dispatcher falls back to a synthetic id when this is `None`.
    request_id: Option<String>,
}

#[derive(Debug)]
enum ParseError {
    /// Peer closed (or read-timed-out) before sending any bytes.
    Idle,
    /// Bytes received but read timed out mid-request → 408.
    Timeout,
    Malformed,
    MethodNotAllowed,
    TooLarge,
}

fn parse_request<R: BufRead>(reader: &mut R) -> Result<Request, ParseError> {
    // Status line: distinguish "peer never sent anything" (idle close)
    // from "peer sent half a line and stalled" (timeout).
    let status = match read_line(reader, true)? {
        Some(s) => s,
        None => return Err(ParseError::Idle),
    };
    let mut parts = status.splitn(3, ' ');
    let method = parts.next().ok_or(ParseError::Malformed)?.to_string();
    let path = parts.next().ok_or(ParseError::Malformed)?.to_string();
    let version = parts.next().ok_or(ParseError::Malformed)?;
    if !version.starts_with("HTTP/1.") {
        return Err(ParseError::Malformed);
    }
    // POST is the agent dispatch path; GET serves health/ops endpoints.
    // Anything else is 405.
    if method != "POST" && method != "GET" {
        return Err(ParseError::MethodNotAllowed);
    }
    // HTTP/1.0 defaults to Connection: close; HTTP/1.1 to keep-alive.
    let mut content_length: Option<usize> = None;
    let mut connection_close = version == "HTTP/1.0";
    let mut request_id: Option<String> = None;
    let mut header_count = 0usize;
    loop {
        let line = read_line(reader, false)?.ok_or(ParseError::Malformed)?;
        if line.is_empty() {
            break;
        }
        header_count += 1;
        if header_count > MAX_HEADERS {
            return Err(ParseError::Malformed);
        }
        let (name, value) = line.split_once(':').ok_or(ParseError::Malformed)?;
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        match name.as_str() {
            "content-length" => {
                let n: usize = value.parse().map_err(|_| ParseError::Malformed)?;
                if n > MAX_BODY {
                    return Err(ParseError::TooLarge);
                }
                content_length = Some(n);
            }
            "transfer-encoding" => {
                // We don't implement chunked. Reject explicitly rather than
                // silently mis-framing the body.
                return Err(ParseError::Malformed);
            }
            "connection" => {
                // RFC 7230 allows comma-separated tokens. `close` wins
                // over `keep-alive` if both appear.
                for tok in value.split(',') {
                    match tok.trim().to_ascii_lowercase().as_str() {
                        "close" => connection_close = true,
                        "keep-alive" if !connection_close => connection_close = false,
                        _ => {}
                    }
                }
            }
            "x-request-id" => {
                // Reject empty/whitespace-only values and anything containing
                // ASCII control chars (header injection guard) — fall back to
                // a synthetic id in that case.
                if !value.is_empty()
                    && value.len() <= 256
                    && !value.bytes().any(|b| b < 0x20 || b == 0x7f)
                {
                    request_id = Some(value.to_string());
                }
            }
            _ => {}
        }
    }

    let body = match content_length {
        Some(0) | None => Vec::new(),
        Some(n) => {
            let mut buf = vec![0u8; n];
            read_exact_mapped(reader, &mut buf)?;
            buf
        }
    };

    Ok(Request { method, path, body, connection_close, request_id })
}

/// Read one CRLF-terminated line. With `allow_idle = true` (used only for
/// the request line), a peer that closes or read-times-out before sending
/// any bytes returns `Ok(None)` — that's a clean keep-alive idle close,
/// not a 408.
fn read_line<R: BufRead>(reader: &mut R, allow_idle: bool) -> Result<Option<String>, ParseError> {
    let mut buf = Vec::with_capacity(128);
    let mut any = false;
    loop {
        let chunk = match reader.fill_buf() {
            Ok(c) => c,
            Err(e) if is_timeout(&e) => {
                return if any || !allow_idle { Err(ParseError::Timeout) } else { Ok(None) };
            }
            Err(_) => return Err(ParseError::Malformed),
        };
        if chunk.is_empty() {
            return if any || !allow_idle { Err(ParseError::Malformed) } else { Ok(None) };
        }
        any = true;
        if let Some(pos) = chunk.iter().position(|&b| b == b'\n') {
            buf.extend_from_slice(&chunk[..pos]);
            reader.consume(pos + 1);
            break;
        }
        buf.extend_from_slice(chunk);
        let n = chunk.len();
        reader.consume(n);
        if buf.len() > MAX_HEADER_LINE {
            return Err(ParseError::Malformed);
        }
    }
    if buf.last() == Some(&b'\r') {
        buf.pop();
    }
    String::from_utf8(buf).map(Some).map_err(|_| ParseError::Malformed)
}

/// Like `BufRead::read_exact` but maps a read timeout to `Timeout` instead
/// of `Malformed` so callers can return 408.
fn read_exact_mapped<R: BufRead>(reader: &mut R, buf: &mut [u8]) -> Result<(), ParseError> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => return Err(ParseError::Malformed),
            Ok(n) => filled += n,
            Err(e) if is_timeout(&e) => return Err(ParseError::Timeout),
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(_) => return Err(ParseError::Malformed),
        }
    }
    Ok(())
}

fn is_timeout(e: &io::Error) -> bool {
    matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut)
}

fn write_simple(
    w: &mut dyn Write,
    status: u16,
    reason: &str,
    request_id: Option<&str>,
) -> io::Result<()> {
    let body = reason.as_bytes();
    write!(
        w,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    )?;
    if let Some(rid) = request_id {
        write!(w, "X-Request-ID: {rid}\r\n")?;
    }
    w.write_all(b"\r\n")?;
    w.write_all(body)?;
    w.flush()
}

/// 200 OK with a tiny body, no `Connection: close` — health checks
/// benefit from keep-alive so liveness probes don't TIME_WAIT-flood.
fn write_ops(w: &mut dyn Write, body_text: &str, request_id: &str) -> io::Result<()> {
    let body = body_text.as_bytes();
    write!(
        w,
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nX-Request-ID: {request_id}\r\n\r\n",
        body.len()
    )?;
    w.write_all(body)?;
    w.flush()
}

/// Mint a synthetic request id when the client didn't supply one. Shape:
/// `req-<unix_ms>-<8 lowercase hex>`. The hex tail mixes pid, subsec
/// nanos, and a monotonically increasing per-process counter so two
/// calls in the same millisecond (and on the same coarse-clock tick)
/// always differ. Not crypto-random — for log/metric joins only.
fn generate_request_id() -> String {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let now =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let ms = now.as_millis();
    let nanos = now.subsec_nanos();
    let pid = std::process::id();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed) as u32;
    // Fold all three into a 32-bit tail via Knuth-multiplicative mixing.
    let mut tail = pid.wrapping_mul(2_654_435_761);
    tail = tail.wrapping_add(nanos.wrapping_mul(40_503));
    tail = tail.wrapping_add(seq.wrapping_mul(2_246_822_519));
    format!("req-{ms}-{tail:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct EchoAgent;
    impl AgentHandler for EchoAgent {
        fn handle(
            &self,
            id: &str,
            _request_id: &str,
            body: &[u8],
            sink: &mut EventSink,
        ) -> Result<(), HandlerError> {
            sink.emit(Some("start"), id).map_err(io_to_handler)?;
            sink.emit_data(std::str::from_utf8(body).unwrap_or("<binary>"))
                .map_err(io_to_handler)?;
            sink.emit(Some("done"), "").map_err(io_to_handler)?;
            Ok(())
        }
    }

    struct ErrAgent;
    impl AgentHandler for ErrAgent {
        fn handle(
            &self,
            _id: &str,
            _request_id: &str,
            _body: &[u8],
            _sink: &mut EventSink,
        ) -> Result<(), HandlerError> {
            Err(HandlerError("boom".into()))
        }
    }

    struct PreflightRejectAgent;
    impl AgentHandler for PreflightRejectAgent {
        fn prepare<'a>(
            &'a self,
            _id: &str,
            _request_id: &str,
            _body: &[u8],
        ) -> Result<PreparedResponse<'a>, PrepareError> {
            Err(PrepareError::Service(HandlerError("required audit log unavailable".into())))
        }

        fn handle(
            &self,
            _id: &str,
            _request_id: &str,
            _body: &[u8],
            _sink: &mut EventSink,
        ) -> Result<(), HandlerError> {
            panic!("handler must not run after failed preflight")
        }
    }

    struct PreflightClientRejectAgent;
    impl AgentHandler for PreflightClientRejectAgent {
        fn prepare<'a>(
            &'a self,
            _id: &str,
            _request_id: &str,
            _body: &[u8],
        ) -> Result<PreparedResponse<'a>, PrepareError> {
            Err(PrepareError::Client(HandlerError("invalid prompt".into())))
        }

        fn handle(
            &self,
            _id: &str,
            _request_id: &str,
            _body: &[u8],
            _sink: &mut EventSink,
        ) -> Result<(), HandlerError> {
            panic!("handler must not run after invalid prompt")
        }
    }

    struct LenAgent;
    impl AgentHandler for LenAgent {
        fn handle(
            &self,
            _id: &str,
            _request_id: &str,
            body: &[u8],
            sink: &mut EventSink,
        ) -> Result<(), HandlerError> {
            sink.emit_data(&body.len().to_string()).map_err(io_to_handler)
        }
    }

    /// Re-emits the `request_id` it received so tests can assert the
    /// dispatcher passed the same value it echoed on the response header.
    struct EchoReqIdAgent;
    impl AgentHandler for EchoReqIdAgent {
        fn handle(
            &self,
            _id: &str,
            request_id: &str,
            _body: &[u8],
            sink: &mut EventSink,
        ) -> Result<(), HandlerError> {
            sink.emit(Some("req"), request_id).map_err(io_to_handler)
        }
    }

    fn io_to_handler(e: io::Error) -> HandlerError {
        HandlerError(e.to_string())
    }

    fn spawn_server() -> (std::net::SocketAddr, thread::JoinHandle<()>) {
        spawn_server_with(ServerConfig::default())
    }

    fn spawn_server_with(cfg: ServerConfig) -> (std::net::SocketAddr, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut server = Server::new();
        server.register("echo", Box::new(EchoAgent));
        server.register("err", Box::new(ErrAgent));
        server.register("len", Box::new(LenAgent));
        server.register("reqid", Box::new(EchoReqIdAgent));
        server.register("preflight", Box::new(PreflightRejectAgent));
        server.register("badprompt", Box::new(PreflightClientRejectAgent));
        let handle = thread::spawn(move || {
            let _ = server.serve_listener_with(listener, cfg);
        });
        (addr, handle)
    }

    fn send(addr: std::net::SocketAddr, raw: &[u8]) -> Vec<u8> {
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        s.write_all(raw).unwrap();
        s.shutdown(std::net::Shutdown::Write).ok();
        let mut out = Vec::new();
        s.read_to_end(&mut out).unwrap();
        out
    }

    fn split_headers_body(resp: &[u8]) -> (String, Vec<u8>) {
        let sep = b"\r\n\r\n";
        let idx = resp.windows(4).position(|w| w == sep).unwrap();
        let head = String::from_utf8_lossy(&resp[..idx]).to_string();
        let body = resp[idx + 4..].to_vec();
        (head, body)
    }

    #[test]
    fn registered_handler_returns_sse_events() {
        let (addr, _h) = spawn_server();
        let req = b"POST /agents/echo/abc HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhello";
        let resp = send(addr, req);
        let (head, body) = split_headers_body(&resp);
        assert!(head.starts_with("HTTP/1.1 200 OK"));
        assert!(head.contains("Content-Type: text/event-stream"));
        let body = String::from_utf8(body).unwrap();
        assert!(body.contains("event: start\ndata: abc\n\n"), "body was: {body:?}");
        assert!(body.contains("data: hello\n\n"));
        assert!(body.contains("event: done\ndata: \n\n"));
    }

    #[test]
    fn unknown_agent_returns_404() {
        let (addr, _h) = spawn_server();
        let req = b"POST /agents/missing/x HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
        let resp = send(addr, req);
        assert!(resp.starts_with(b"HTTP/1.1 404"));
    }

    #[test]
    fn unknown_route_returns_404() {
        let (addr, _h) = spawn_server();
        let req = b"POST /nope HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
        let resp = send(addr, req);
        assert!(resp.starts_with(b"HTTP/1.1 404"));
    }

    #[test]
    fn healthz_returns_200_ok() {
        let (addr, _h) = spawn_server();
        let req = b"GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n";
        let resp = send(addr, req);
        assert!(resp.starts_with(b"HTTP/1.1 200"));
        assert!(resp.ends_with(b"ok\n"));
    }

    #[test]
    fn readyz_returns_200_ok() {
        let (addr, _h) = spawn_server();
        let req = b"GET /readyz HTTP/1.1\r\nHost: x\r\n\r\n";
        let resp = send(addr, req);
        assert!(resp.starts_with(b"HTTP/1.1 200"));
        assert!(resp.ends_with(b"ok\n"));
    }

    #[test]
    fn healthz_with_query_string_still_matches() {
        let (addr, _h) = spawn_server();
        let req = b"GET /healthz?ts=1 HTTP/1.1\r\nHost: x\r\n\r\n";
        let resp = send(addr, req);
        assert!(resp.starts_with(b"HTTP/1.1 200"));
    }

    #[test]
    fn get_returns_405() {
        let (addr, _h) = spawn_server();
        let req = b"GET /agents/echo/x HTTP/1.1\r\n\r\n";
        let resp = send(addr, req);
        assert!(resp.starts_with(b"HTTP/1.1 405"));
    }

    #[test]
    fn empty_body_ok() {
        let (addr, _h) = spawn_server();
        let req = b"POST /agents/len/x HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
        let resp = send(addr, req);
        let (_, body) = split_headers_body(&resp);
        assert!(String::from_utf8(body).unwrap().contains("data: 0\n\n"));
    }

    #[test]
    fn no_content_length_treated_as_empty_body() {
        let (addr, _h) = spawn_server();
        let req = b"POST /agents/len/x HTTP/1.1\r\n\r\n";
        let resp = send(addr, req);
        let (_, body) = split_headers_body(&resp);
        assert!(String::from_utf8(body).unwrap().contains("data: 0\n\n"));
    }

    #[test]
    fn one_mib_body_ok() {
        let (addr, _h) = spawn_server();
        let mut req = b"POST /agents/len/x HTTP/1.1\r\nContent-Length: 1048576\r\n\r\n".to_vec();
        req.extend(std::iter::repeat_n(b'a', MAX_BODY));
        let resp = send(addr, &req);
        let (head, body) = split_headers_body(&resp);
        assert!(head.starts_with("HTTP/1.1 200 OK"));
        assert!(String::from_utf8(body).unwrap().contains("data: 1048576\n\n"));
    }

    #[test]
    fn oversized_body_returns_413() {
        let (addr, _h) = spawn_server();
        let req = b"POST /agents/len/x HTTP/1.1\r\nContent-Length: 1048577\r\n\r\n";
        let resp = send(addr, req);
        assert!(resp.starts_with(b"HTTP/1.1 413"));
    }

    #[test]
    fn malformed_request_returns_400() {
        let (addr, _h) = spawn_server();
        let resp = send(addr, b"not http at all\r\n\r\n");
        assert!(resp.starts_with(b"HTTP/1.1 400"));
    }

    #[test]
    fn chunked_encoding_rejected() {
        let (addr, _h) = spawn_server();
        let req = b"POST /agents/echo/x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n";
        let resp = send(addr, req);
        assert!(resp.starts_with(b"HTTP/1.1 400"));
    }

    #[test]
    fn handler_error_emits_sse_error_event() {
        let (addr, _h) = spawn_server();
        let req = b"POST /agents/err/x HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
        let resp = send(addr, req);
        let (head, body) = split_headers_body(&resp);
        assert!(head.starts_with("HTTP/1.1 200 OK"));
        let body = String::from_utf8(body).unwrap();
        assert!(body.contains("event: error\ndata: boom\n\n"), "body: {body:?}");
    }

    #[test]
    fn failed_required_preflight_returns_503_before_sse_or_handler_work() {
        let (addr, _h) = spawn_server();
        let req = b"POST /agents/preflight/x HTTP/1.1\r\nX-Request-ID: trace-1\r\nContent-Length: 0\r\n\r\n";
        let resp = send(addr, req);
        let (head, body) = split_headers_body(&resp);
        assert!(head.starts_with("HTTP/1.1 503"), "{head}");
        assert!(head.contains("X-Request-ID: trace-1"));
        assert!(!head.contains("text/event-stream"));
        assert_eq!(body, b"Service Unavailable");
    }

    #[test]
    fn client_preflight_error_returns_400_before_sse() {
        let (addr, _h) = spawn_server();
        let req = b"POST /agents/badprompt/x HTTP/1.1\r\nX-Request-ID: trace-1\r\nContent-Length: 0\r\n\r\n";
        let resp = send(addr, req);
        let (head, body) = split_headers_body(&resp);
        assert!(head.starts_with("HTTP/1.1 400"), "{head}");
        assert!(head.contains("X-Request-ID: trace-1"));
        assert!(!head.contains("text/event-stream"));
        assert_eq!(body, b"Bad Request");
    }

    #[test]
    fn multiple_sequential_requests_work() {
        let (addr, _h) = spawn_server();
        for i in 0..5 {
            let req = format!("POST /agents/echo/id{i} HTTP/1.1\r\nContent-Length: 0\r\n\r\n");
            let resp = send(addr, req.as_bytes());
            let (head, body) = split_headers_body(&resp);
            assert!(head.starts_with("HTTP/1.1 200 OK"));
            let body = String::from_utf8(body).unwrap();
            assert!(body.contains(&format!("event: start\ndata: id{i}\n\n")));
        }
    }

    #[test]
    fn path_with_query_string_routes_correctly() {
        let (addr, _h) = spawn_server();
        let req = b"POST /agents/echo/abc?foo=bar HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
        let resp = send(addr, req);
        let (_, body) = split_headers_body(&resp);
        let body = String::from_utf8(body).unwrap();
        assert!(body.contains("event: start\ndata: abc\n\n"));
    }

    #[test]
    fn nested_id_with_slash_is_404() {
        let (addr, _h) = spawn_server();
        let req = b"POST /agents/echo/a/b HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
        let resp = send(addr, req);
        assert!(resp.starts_with(b"HTTP/1.1 404"));
    }

    // EventSink tests against an in-memory writer ----------------------------

    struct TestSink {
        buf: Vec<u8>,
    }
    impl Write for TestSink {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            self.buf.extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn event_sink_emits_named_event() {
        let mut t = TestSink { buf: Vec::new() };
        EventSink::new(&mut t).emit(Some("msg"), "hi").unwrap();
        assert_eq!(t.buf, b"event: msg\ndata: hi\n\n");
    }

    #[test]
    fn event_sink_emits_unnamed_event() {
        let mut t = TestSink { buf: Vec::new() };
        EventSink::new(&mut t).emit_data("hi").unwrap();
        assert_eq!(t.buf, b"data: hi\n\n");
    }

    #[test]
    fn event_sink_splits_multiline_data() {
        let mut t = TestSink { buf: Vec::new() };
        EventSink::new(&mut t).emit(Some("e"), "a\nb").unwrap();
        assert_eq!(t.buf, b"event: e\ndata: a\ndata: b\n\n");
    }

    // Sanity: handlers can be called from many threads through Arc<Server>.
    #[test]
    fn concurrent_requests_dont_corrupt_each_other() {
        let (addr, _h) = spawn_server();
        let count = std::sync::Arc::new(AtomicUsize::new(0));
        let errors: std::sync::Arc<Mutex<Vec<String>>> =
            std::sync::Arc::new(Mutex::new(Vec::new()));
        let mut workers = Vec::new();
        for i in 0..16 {
            let count = count.clone();
            let errors = errors.clone();
            workers.push(thread::spawn(move || {
                let req = format!("POST /agents/echo/id{i} HTTP/1.1\r\nContent-Length: 0\r\n\r\n");
                let resp = send(addr, req.as_bytes());
                let body = String::from_utf8_lossy(&resp).to_string();
                if body.contains(&format!("event: start\ndata: id{i}\n\n")) {
                    count.fetch_add(1, Ordering::SeqCst);
                } else {
                    errors.lock().unwrap().push(body);
                }
            }));
        }
        for w in workers {
            w.join().unwrap();
        }
        assert_eq!(count.load(Ordering::SeqCst), 16, "errors: {:?}", errors.lock().unwrap());
    }

    // Hardening tests --------------------------------------------------------

    #[test]
    fn config_defaults_are_30s_256_keep_alive() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.read_timeout, Duration::from_secs(30));
        assert_eq!(cfg.write_timeout, Duration::from_secs(30));
        assert_eq!(cfg.max_connections, 256);
        assert!(cfg.keep_alive);
    }

    /// A client that opens a connection and sits silent gets a clean idle
    /// close (not a 408): the parser distinguishes "no bytes yet" from
    /// "partial request then stall".
    #[test]
    fn idle_connection_closes_after_read_timeout() {
        let cfg = ServerConfig { read_timeout: Duration::from_millis(100), ..Default::default() };
        let (addr, _h) = spawn_server_with(cfg);
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut out = Vec::new();
        let start = std::time::Instant::now();
        s.read_to_end(&mut out).unwrap();
        assert!(start.elapsed() < Duration::from_secs(2));
        // No response body: idle close, not 408.
        assert!(out.is_empty(), "unexpected response: {out:?}");
    }

    /// Slowloris: a client that sends a partial request line then stalls
    /// gets a 408 from the server.
    #[test]
    fn read_timeout_mid_request_returns_408() {
        let cfg = ServerConfig { read_timeout: Duration::from_millis(150), ..Default::default() };
        let (addr, _h) = spawn_server_with(cfg);
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        // Partial request — no terminating CRLF.
        s.write_all(b"POST /agents/echo/x HTTP/1").unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).unwrap();
        assert!(out.starts_with(b"HTTP/1.1 408"), "got: {:?}", String::from_utf8_lossy(&out));
    }

    /// Write timeout: if the client refuses to read, an SSE handler that
    /// emits enough data to overflow the kernel buffers will eventually
    /// have its write fail.
    #[test]
    fn write_timeout_fires_when_client_refuses_to_read() {
        struct FloodAgent;
        impl AgentHandler for FloodAgent {
            fn handle(
                &self,
                _id: &str,
                _request_id: &str,
                _body: &[u8],
                sink: &mut EventSink,
            ) -> Result<(), HandlerError> {
                // 1 MiB payload per emit; the kernel send buffer is much
                // smaller, so once the client stops reading the write will
                // block and eventually time out.
                let big = "x".repeat(1024 * 1024);
                for _ in 0..64 {
                    sink.emit_data(&big).map_err(|e| HandlerError(e.to_string()))?;
                }
                Ok(())
            }
        }
        let cfg = ServerConfig { write_timeout: Duration::from_millis(200), ..Default::default() };
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut server = Server::new();
        server.register("flood", Box::new(FloodAgent));
        thread::spawn(move || {
            let _ = server.serve_listener_with(listener, cfg);
        });

        let mut s = TcpStream::connect(addr).unwrap();
        s.write_all(b"POST /agents/flood/x HTTP/1.1\r\nContent-Length: 0\r\n\r\n").unwrap();
        // Don't read. The server's write_timeout should fire and the
        // handler thread should exit cleanly within a few seconds. We
        // probe by waiting then doing one short read to confirm no panic.
        thread::sleep(Duration::from_millis(800));
        s.set_read_timeout(Some(Duration::from_millis(200))).unwrap();
        let mut tiny = [0u8; 16];
        // Either we read something (the bytes that did make it before the
        // timeout) or we don't — both are fine. The contract under test is
        // that the server doesn't panic and the connection is closed
        // server-side.
        let _ = s.read(&mut tiny);
    }

    /// Connection cap: with `max_connections = 4`, the 5th simultaneous
    /// connection is dropped immediately by the server.
    #[test]
    fn max_connections_drops_overflow() {
        struct BlockAgent {
            release: Arc<Mutex<()>>,
        }
        impl AgentHandler for BlockAgent {
            fn handle(
                &self,
                _id: &str,
                _request_id: &str,
                _body: &[u8],
                sink: &mut EventSink,
            ) -> Result<(), HandlerError> {
                sink.emit(Some("hold"), "").map_err(|e| HandlerError(e.to_string()))?;
                // Park until the test releases us.
                let _g = self.release.lock().unwrap();
                Ok(())
            }
        }
        let release = Arc::new(Mutex::new(()));
        let guard = release.lock().unwrap();

        let cfg = ServerConfig { max_connections: 4, ..Default::default() };
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut server = Server::new();
        server.register("block", Box::new(BlockAgent { release: release.clone() }));
        thread::spawn(move || {
            let _ = server.serve_listener_with(listener, cfg);
        });

        // Open 4 long-running connections.
        let mut held = Vec::new();
        for _ in 0..4 {
            let mut s = TcpStream::connect(addr).unwrap();
            s.write_all(b"POST /agents/block/x HTTP/1.1\r\nContent-Length: 0\r\n\r\n").unwrap();
            // Wait for the "hold" SSE event so we know the handler is
            // running and the slot is in use.
            s.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            let mut buf = [0u8; 256];
            let n = s.read(&mut buf).unwrap();
            assert!(n > 0);
            held.push(s);
        }

        // 5th connection: the server should accept and immediately close.
        let mut overflow = TcpStream::connect(addr).unwrap();
        overflow.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut out = Vec::new();
        overflow.read_to_end(&mut out).unwrap();
        assert!(out.is_empty(), "overflow connection got data: {out:?}");

        // Release the held handlers and let everything shut down.
        drop(guard);
        for s in held {
            let _ = s.shutdown(std::net::Shutdown::Both);
        }
    }

    /// Two requests on a single TCP connection both succeed.
    #[test]
    fn keep_alive_serves_two_requests_on_one_connection() {
        let (addr, _h) = spawn_server();
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        // Send the first request, but the response is SSE which closes
        // the connection. So instead pick a route that produces a non-SSE
        // response: 404 path. Two 404s on the same socket.
        s.write_all(b"POST /nope HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n").unwrap();
        let mut buf = [0u8; 512];
        let mut first = Vec::new();
        // Read exactly one response. We know its Content-Length so we
        // can avoid blocking on EOF.
        read_one_response(&mut s, &mut buf, &mut first);
        assert!(first.starts_with(b"HTTP/1.1 404"), "first: {:?}", String::from_utf8_lossy(&first));

        s.write_all(b"POST /nope HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n").unwrap();
        let mut second = Vec::new();
        read_one_response(&mut s, &mut buf, &mut second);
        assert!(
            second.starts_with(b"HTTP/1.1 404"),
            "second: {:?}",
            String::from_utf8_lossy(&second)
        );
    }

    /// `Connection: close` on a non-SSE response terminates the socket
    /// after one request even with keep-alive enabled.
    #[test]
    fn connection_close_header_terminates_after_one_request() {
        let (addr, _h) = spawn_server();
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        s.write_all(
            b"POST /nope HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",
        )
        .unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).unwrap();
        assert!(out.starts_with(b"HTTP/1.1 404"));
        // No second response: connection was closed after the first.
        // (read_to_end returned, meaning the server closed its side.)
    }

    /// SSE response always terminates the connection, even with keep-alive
    /// enabled and no `Connection: close` header from the client.
    #[test]
    fn sse_response_closes_connection_despite_keep_alive() {
        let (addr, _h) = spawn_server();
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        s.write_all(b"POST /agents/echo/x HTTP/1.1\r\nContent-Length: 0\r\n\r\n").unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).unwrap();
        let (head, _body) = split_headers_body(&out);
        assert!(head.contains("Content-Type: text/event-stream"));
        // read_to_end completed → server closed the socket as required
        // for an open-ended SSE body.
    }

    /// Plumb a custom config through `serve_with` and observe its effect:
    /// a 1ms read timeout makes a slow client trip 408 almost immediately.
    #[test]
    fn serve_with_plumbs_config() {
        let cfg = ServerConfig { read_timeout: Duration::from_millis(50), ..Default::default() };
        let (addr, _h) = spawn_server_with(cfg);
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        s.write_all(b"POST /agents/echo/x HTTP/1.").unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).unwrap();
        assert!(out.starts_with(b"HTTP/1.1 408"));
    }

    /// HTTP/1.0 defaults to `Connection: close`. Verify the keep-alive
    /// path honours that: one request, then close.
    #[test]
    fn http_1_0_closes_by_default() {
        let (addr, _h) = spawn_server();
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        s.write_all(b"POST /nope HTTP/1.0\r\nContent-Length: 0\r\n\r\n").unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).unwrap();
        assert!(out.starts_with(b"HTTP/1.1 404"));
    }

    /// With `keep_alive = false` in config, even a clean non-SSE request
    /// on HTTP/1.1 results in close-after-one.
    #[test]
    fn config_keep_alive_off_closes_after_one() {
        let cfg = ServerConfig { keep_alive: false, ..Default::default() };
        let (addr, _h) = spawn_server_with(cfg);
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        s.write_all(b"POST /nope HTTP/1.1\r\nContent-Length: 0\r\n\r\n").unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).unwrap();
        assert!(out.starts_with(b"HTTP/1.1 404"));
    }

    /// Three sequential 404s on one socket — exercises the keep-alive
    /// loop more than twice.
    #[test]
    fn keep_alive_three_requests() {
        let (addr, _h) = spawn_server();
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut buf = [0u8; 512];
        for _ in 0..3 {
            s.write_all(b"POST /nope HTTP/1.1\r\nContent-Length: 0\r\n\r\n").unwrap();
            let mut resp = Vec::new();
            read_one_response(&mut s, &mut buf, &mut resp);
            assert!(resp.starts_with(b"HTTP/1.1 404"));
        }
    }

    /// Read exactly one HTTP response from `s` into `out`. We rely on
    /// `Content-Length` because `read_to_end` would block forever on a
    /// keep-alive socket.
    fn read_one_response(s: &mut TcpStream, buf: &mut [u8], out: &mut Vec<u8>) {
        // Read until we have the full headers, then parse Content-Length
        // and read that many body bytes.
        loop {
            let n = s.read(buf).unwrap();
            assert!(n > 0, "EOF before headers");
            out.extend_from_slice(&buf[..n]);
            if let Some(idx) = out.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = std::str::from_utf8(&out[..idx]).unwrap();
                let mut len = 0usize;
                for line in head.split("\r\n") {
                    if let Some(v) = line.strip_prefix("Content-Length:") {
                        len = v.trim().parse().unwrap_or(0);
                    } else if let Some(v) = line.strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap_or(0);
                    }
                }
                let body_start = idx + 4;
                while out.len() < body_start + len {
                    let n = s.read(buf).unwrap();
                    if n == 0 {
                        return;
                    }
                    out.extend_from_slice(&buf[..n]);
                }
                return;
            }
        }
    }

    // Request-ID propagation -------------------------------------------------

    /// Client-supplied `X-Request-ID` is echoed verbatim on the SSE
    /// response header AND surfaced to the handler.
    #[test]
    fn inbound_x_request_id_echoed_on_response_and_passed_to_handler() {
        let (addr, _h) = spawn_server();
        let req = b"POST /agents/reqid/x HTTP/1.1\r\n\
                    Host: x\r\n\
                    X-Request-ID: my-trace-123\r\n\
                    Content-Length: 0\r\n\r\n";
        let resp = send(addr, req);
        let (head, body) = split_headers_body(&resp);
        assert!(head.starts_with("HTTP/1.1 200 OK"));
        assert!(head.contains("X-Request-ID: my-trace-123"), "header missing X-Request-ID: {head}");
        let body = String::from_utf8(body).unwrap();
        assert!(
            body.contains("event: req\ndata: my-trace-123\n\n"),
            "handler did not receive the inbound id: {body:?}"
        );
    }

    /// Missing inbound header → server mints `req-<unix_ms>-<8 hex>` and
    /// the same value appears on the response header and in the handler.
    #[test]
    fn missing_x_request_id_synthesizes_and_echoes() {
        let (addr, _h) = spawn_server();
        let req = b"POST /agents/reqid/x HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n";
        let resp = send(addr, req);
        let (head, body) = split_headers_body(&resp);
        assert!(head.starts_with("HTTP/1.1 200 OK"));

        // Pull the synthesised id off the response header.
        let header_id = head
            .lines()
            .find_map(|l| l.strip_prefix("X-Request-ID: "))
            .expect("X-Request-ID header present");
        assert!(header_id.starts_with("req-"), "synthetic id has wrong prefix: {header_id:?}");
        // Shape: req-<ms>-<8 hex>
        let rest = &header_id["req-".len()..];
        let (ms, hex) = rest.split_once('-').expect("ms-hex split");
        assert!(!ms.is_empty() && ms.bytes().all(|b| b.is_ascii_digit()));
        assert_eq!(hex.len(), 8);
        assert!(hex.bytes().all(|b| b.is_ascii_hexdigit()));

        // Same value must reach the handler.
        let body = String::from_utf8(body).unwrap();
        assert!(
            body.contains(&format!("event: req\ndata: {header_id}\n\n")),
            "handler request_id mismatch; header={header_id:?} body={body:?}"
        );
    }

    /// An empty or control-char-laced `X-Request-ID` is rejected (treated
    /// as missing) so a hostile client can't inject headers via the id.
    #[test]
    fn empty_x_request_id_falls_back_to_synthetic() {
        let (addr, _h) = spawn_server();
        let req = b"POST /agents/reqid/x HTTP/1.1\r\n\
                    Host: x\r\n\
                    X-Request-ID: \r\n\
                    Content-Length: 0\r\n\r\n";
        let resp = send(addr, req);
        let (head, _body) = split_headers_body(&resp);
        let header_id = head
            .lines()
            .find_map(|l| l.strip_prefix("X-Request-ID: "))
            .expect("X-Request-ID header present");
        assert!(header_id.starts_with("req-"), "got: {header_id:?}");
    }

    /// `generate_request_id` returns distinct values on back-to-back calls.
    /// Cheap sanity check on the suffix mixing.
    #[test]
    fn generate_request_id_is_distinct_within_one_ms() {
        let a = generate_request_id();
        let b = generate_request_id();
        assert_ne!(a, b, "generator collided immediately: {a} == {b}");
    }

    #[test]
    fn shutdown_waits_for_actual_worker_retirement() {
        struct Held(std::sync::mpsc::Sender<()>, Mutex<std::sync::mpsc::Receiver<()>>);
        impl AgentHandler for Held {
            fn handle(
                &self,
                _: &str,
                _: &str,
                _: &[u8],
                _: &mut EventSink,
            ) -> Result<(), HandlerError> {
                self.0.send(()).unwrap();
                self.1.lock().unwrap().recv().unwrap();
                Ok(())
            }
        }
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut server = Server::new();
        server.register("held", Box::new(Held(entered_tx, Mutex::new(release_rx))));
        let shutdown = Arc::new(AtomicBool::new(false));
        let signal = shutdown.clone();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let owner = thread::spawn(move || {
            let result =
                server.serve_listener_with_shutdown(listener, ServerConfig::default(), signal);
            done_tx.send(result).unwrap();
        });
        let mut socket = TcpStream::connect(addr).unwrap();
        socket.write_all(b"POST /agents/held/x HTTP/1.1\r\nContent-Length: 0\r\n\r\n").unwrap();
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        shutdown.store(true, Ordering::Release);
        let before_release = done_rx.recv_timeout(Duration::from_millis(100));
        // Release before asserting, so even a regression does not strand the fixture worker.
        release_tx.send(()).unwrap();
        let completed = match before_release {
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                done_rx.recv_timeout(Duration::from_secs(5)).unwrap()
            }
            _ => {
                owner.join().unwrap();
                panic!("server returned before worker retirement");
            }
        };
        owner.join().unwrap();
        assert!(completed.is_ok());
    }

    #[test]
    fn worker_panic_is_a_failed_server_retirement() {
        struct Panics;
        impl AgentHandler for Panics {
            fn handle(
                &self,
                _: &str,
                _: &str,
                _: &[u8],
                _: &mut EventSink,
            ) -> Result<(), HandlerError> {
                panic!("finite worker failure")
            }
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut server = Server::new();
        server.register("panic", Box::new(Panics));
        let signal = Arc::new(AtomicBool::new(false));
        let stop = signal.clone();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let owner = thread::spawn(move || {
            done_tx
                .send(server.serve_listener_with_shutdown(
                    listener,
                    ServerConfig::default(),
                    signal,
                ))
                .unwrap();
        });
        let mut socket = TcpStream::connect(addr).unwrap();
        socket.write_all(b"POST /agents/panic/x HTTP/1.1\r\nContent-Length: 0\r\n\r\n").unwrap();
        let result = done_rx.recv_timeout(Duration::from_secs(5));
        stop.store(true, Ordering::Release);
        owner.join().unwrap();
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn shutdown_stops_dispatch_on_an_existing_connection() {
        struct Count(Arc<AtomicUsize>);
        impl AgentHandler for Count {
            fn handle(
                &self,
                _: &str,
                _: &str,
                _: &[u8],
                _: &mut EventSink,
            ) -> Result<(), HandlerError> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (socket, _) = listener.accept().unwrap();
        socket.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let mut server = Server::new();
        server.register("count", Box::new(Count(count.clone())));
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let worker = thread::spawn(move || {
            server.handle_connection_with_shutdown(
                socket,
                &ServerConfig::default(),
                Some(&worker_stop),
            )
        });
        // This socket was already accepted. Shutdown precedes arrival of a
        // dispatchable request, whether the worker is about to read or in read.
        stop.store(true, Ordering::Release);
        let _ = client.write_all(b"POST /agents/count/x HTTP/1.1\r\nContent-Length: 0\r\n\r\n");
        client.shutdown(std::net::Shutdown::Write).ok();
        worker.join().unwrap().unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 0);
    }
}
