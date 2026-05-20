//! Minimal blocking HTTP/1.1 server exposing `AgentHandler` endpoints with
//! Server-Sent Events streaming.
//!
//! One OS thread per connection. `std::net::TcpListener` + manual HTTP/1.1
//! framing. Zero dependencies.
//!
//! Routing: `POST /agents/<name>/<id>` dispatches to a registered handler
//! by `<name>`. Everything else is a hard error (404 / 405 / 400 / 413).
//!
//! Streaming model: handlers write SSE frames through `EventSink`; the
//! transport is `Connection: close` and we flush after every frame so
//! clients see events as they arrive.

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

const MAX_BODY: usize = 1024 * 1024;
const MAX_HEADER_LINE: usize = 8 * 1024;
const MAX_HEADERS: usize = 64;

pub trait AgentHandler: Send + Sync + 'static {
    fn handle(&self, id: &str, body: &[u8], sink: &mut EventSink) -> Result<(), HandlerError>;
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

    pub fn serve(self, addr: &str) -> io::Result<()> {
        self.serve_listener(TcpListener::bind(addr)?)
    }

    /// Drive an already-bound listener. Lets tests pick a free port and own
    /// the listener's lifetime without leaking threads on `0.0.0.0`.
    pub fn serve_listener(self, listener: TcpListener) -> io::Result<()> {
        let shared = std::sync::Arc::new(self);
        for conn in listener.incoming() {
            let stream = conn?;
            let server = shared.clone();
            thread::spawn(move || {
                let _ = server.handle_connection(stream);
            });
        }
        Ok(())
    }

    fn handle_connection(&self, mut stream: TcpStream) -> io::Result<()> {
        let mut reader = BufReader::new(stream.try_clone()?);
        match parse_request(&mut reader) {
            Ok(req) => self.dispatch(req, &mut stream),
            Err(ParseError::TooLarge) => write_simple(&mut stream, 413, "Payload Too Large"),
            Err(ParseError::MethodNotAllowed) => {
                write_simple(&mut stream, 405, "Method Not Allowed")
            }
            Err(_) => write_simple(&mut stream, 400, "Bad Request"),
        }
    }

    fn dispatch(&self, req: Request, stream: &mut TcpStream) -> io::Result<()> {
        let Some((name, id)) = parse_agent_path(&req.path) else {
            return write_simple(stream, 404, "Not Found");
        };
        let Some(handler) = self.handlers.get(name) else {
            return write_simple(stream, 404, "Not Found");
        };

        // SSE headers first; once sent, errors become 500-in-body-too-late
        // territory, so we capture handler errors into a buffer first and
        // only commit headers when we know the outcome of the very first
        // write. To keep this simple we send headers up front and, on
        // handler error, emit a final SSE `error` event then close.
        stream.write_all(
            b"HTTP/1.1 200 OK\r\n\
              Content-Type: text/event-stream\r\n\
              Cache-Control: no-cache\r\n\
              Connection: close\r\n\r\n",
        )?;
        stream.flush()?;

        let mut sink = EventSink::new(stream);
        if let Err(e) = handler.handle(id, &req.body, &mut sink) {
            // Headers are already on the wire; we can't change status. Surface
            // the failure as a structured SSE event so clients can react.
            let _ = sink.emit(Some("error"), &e.0);
        }
        Ok(())
    }
}

impl Default for Server {
    fn default() -> Self {
        Self::new()
    }
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
    path: String,
    body: Vec<u8>,
}

#[derive(Debug)]
enum ParseError {
    Malformed,
    MethodNotAllowed,
    TooLarge,
}

impl From<io::Error> for ParseError {
    fn from(_: io::Error) -> Self {
        // Treat truncated / dropped connections as malformed input; we have
        // no way to send a partial response with a different status anyway.
        ParseError::Malformed
    }
}

fn parse_request<R: BufRead>(reader: &mut R) -> Result<Request, ParseError> {
    let status = read_line(reader)?;
    let mut parts = status.splitn(3, ' ');
    let method = parts.next().ok_or(ParseError::Malformed)?;
    let path = parts.next().ok_or(ParseError::Malformed)?.to_string();
    let version = parts.next().ok_or(ParseError::Malformed)?;
    if !version.starts_with("HTTP/1.") {
        return Err(ParseError::Malformed);
    }
    if method != "POST" {
        return Err(ParseError::MethodNotAllowed);
    }

    let mut content_length: Option<usize> = None;
    let mut header_count = 0usize;
    loop {
        let line = read_line(reader)?;
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
            _ => {}
        }
    }

    let body = match content_length {
        Some(0) | None => Vec::new(),
        Some(n) => {
            let mut buf = vec![0u8; n];
            reader.read_exact(&mut buf)?;
            buf
        }
    };

    Ok(Request { path, body })
}

fn read_line<R: BufRead>(reader: &mut R) -> Result<String, ParseError> {
    let mut buf = Vec::with_capacity(128);
    loop {
        let chunk = reader.fill_buf()?;
        if chunk.is_empty() {
            return Err(ParseError::Malformed);
        }
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
    String::from_utf8(buf).map_err(|_| ParseError::Malformed)
}

fn write_simple(w: &mut dyn Write, status: u16, reason: &str) -> io::Result<()> {
    let body = reason.as_bytes();
    write!(
        w,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    w.write_all(body)?;
    w.flush()
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
        fn handle(&self, id: &str, body: &[u8], sink: &mut EventSink) -> Result<(), HandlerError> {
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
            _body: &[u8],
            _sink: &mut EventSink,
        ) -> Result<(), HandlerError> {
            Err(HandlerError("boom".into()))
        }
    }

    struct LenAgent;
    impl AgentHandler for LenAgent {
        fn handle(
            &self,
            _id: &str,
            body: &[u8],
            sink: &mut EventSink,
        ) -> Result<(), HandlerError> {
            sink.emit_data(&body.len().to_string()).map_err(io_to_handler)
        }
    }

    fn io_to_handler(e: io::Error) -> HandlerError {
        HandlerError(e.to_string())
    }

    fn spawn_server() -> (std::net::SocketAddr, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut server = Server::new();
        server.register("echo", Box::new(EchoAgent));
        server.register("err", Box::new(ErrAgent));
        server.register("len", Box::new(LenAgent));
        let handle = thread::spawn(move || {
            let _ = server.serve_listener(listener);
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
        let mut req =
            b"POST /agents/len/x HTTP/1.1\r\nContent-Length: 1048576\r\n\r\n".to_vec();
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
        let req =
            b"POST /agents/echo/x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n";
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
    fn multiple_sequential_requests_work() {
        let (addr, _h) = spawn_server();
        for i in 0..5 {
            let req = format!(
                "POST /agents/echo/id{i} HTTP/1.1\r\nContent-Length: 0\r\n\r\n"
            );
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
                let req = format!(
                    "POST /agents/echo/id{i} HTTP/1.1\r\nContent-Length: 0\r\n\r\n"
                );
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
}
