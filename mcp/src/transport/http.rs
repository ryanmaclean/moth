//! Streamable-HTTP transport for the MCP client.
//!
//! Per the MCP spec: a single POST endpoint accepting one JSON-RPC frame
//! per request. The server may answer with either:
//!
//! - `Content-Type: application/json` — one JSON-RPC frame in the body.
//! - `Content-Type: text/event-stream` — an SSE stream of `data:` frames,
//!   each carrying one JSON-RPC frame.
//!
//! Session continuity: an `Mcp-Session-Id` response header (typically set
//! on the `initialize` response) is echoed back on every subsequent
//! request.
//!
//! The [`Transport`] trait is line-oriented, but HTTP is request-paired.
//! We bridge by POST-ing inside `send_line` (which the inner client calls
//! before every `recv_line`) and pushing each parsed JSON-RPC frame into
//! an inbound FIFO. `recv_line` drains the FIFO. Notifications — which
//! the inner client sends without a matching `recv_line` — are POSTed
//! identically; the server typically replies with `202 Accepted` and an
//! empty body, so nothing lands in the FIFO and the protocol keeps
//! flowing.
//!
//! HTTPS is delegated to `curl-sys` with the same `static-curl` /
//! `static-ssl` build the `anthropic` crate uses. One libcurl easy handle
//! per request — no pooling. `CURLOPT_FOLLOWLOCATION` is on, matching the
//! `anthropic` client.

use std::collections::VecDeque;
use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;
use std::sync::Mutex;

use curl_sys as c;
use wire::SseFramer;

use super::Transport;

/// Streamable-HTTP transport. Construct with [`HttpTransport::new`] and
/// optionally attach a bearer token via [`HttpTransport::with_bearer`].
pub struct HttpTransport {
    url: String,
    bearer: Option<String>,
    /// Updated whenever the server sets `Mcp-Session-Id`; echoed on
    /// subsequent requests. `Mutex` because `Transport`'s methods take
    /// `&self`.
    session_id: Mutex<Option<String>>,
    /// JSON-RPC frames the server has sent us but the inner client hasn't
    /// dequeued yet. One entry per logical frame; SSE responses can push
    /// several at once.
    inbound: Mutex<VecDeque<Vec<u8>>>,
}

impl HttpTransport {
    /// Build a transport pointing at `url`. No network I/O happens until
    /// the first `send_line` call.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            bearer: None,
            session_id: Mutex::new(None),
            inbound: Mutex::new(VecDeque::new()),
        }
    }

    /// Attach `Authorization: Bearer <token>` to every request.
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.bearer = Some(token.into());
        self
    }

    /// Current session id, if the server has set one. Exposed for tests.
    #[cfg(test)]
    pub fn session_id(&self) -> Option<String> {
        self.session_id.lock().unwrap().clone()
    }
}

impl Transport for HttpTransport {
    fn send_line(&self, line: &[u8]) -> Result<(), String> {
        let session = self.session_id.lock().unwrap().clone();
        let resp = post(&self.url, line, self.bearer.as_deref(), session.as_deref())?;

        // Mcp-Session-Id is persisted before status check so a server that
        // returns one *and* an error still leaves the session usable.
        if let Some(new_session) = resp.session_id {
            *self.session_id.lock().unwrap() = Some(new_session);
        }

        if !(200..300).contains(&resp.status) {
            // Surface the body in the error (servers typically include a
            // JSON error message that's useful upstream).
            let body = String::from_utf8_lossy(&resp.body).into_owned();
            return Err(format!("HTTP {} {}: {}", resp.status, http_class(resp.status), body));
        }

        // 202 with empty body is the spec'd "notification accepted" reply;
        // nothing to enqueue.
        if resp.body.is_empty() {
            return Ok(());
        }

        let frames = if resp.content_type_sse {
            parse_sse_data_frames(&resp.body)
                .map_err(|e| format!("SSE response exceeds 4 MiB frame cap: {e}"))?
        } else {
            // Single JSON frame.
            vec![resp.body]
        };

        let mut inbound = self.inbound.lock().unwrap();
        for f in frames {
            // Skip empty frames (server keep-alive comments etc.).
            if f.iter().all(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n')) {
                continue;
            }
            inbound.push_back(f);
        }
        Ok(())
    }

    fn recv_line(&self) -> Result<Vec<u8>, String> {
        let mut inbound = self.inbound.lock().unwrap();
        inbound
            .pop_front()
            .ok_or_else(|| "no buffered response (server returned no body)".to_string())
    }
}

/// Crude HTTP status class label for error messages. Avoids carrying a
/// table of every status code — the number is already in the message.
fn http_class(status: u16) -> &'static str {
    match status / 100 {
        4 => "client error",
        5 => "server error",
        _ => "non-2xx",
    }
}

/// Split an SSE response body into one JSON-RPC frame per event. Per the
/// SSE spec an event can have multiple `data:` lines whose values are
/// joined with `\n`; that's what the MCP server emits when a single
/// response straddles lines. Non-`data:` lines (`event:`, `id:`, `:`
/// comments) are ignored — v1 doesn't need them.
fn parse_sse_data_frames(body: &[u8]) -> Result<Vec<Vec<u8>>, wire::FramerError> {
    // SseFramer scans for the literal pair `\n\n`. Some servers (and our
    // test fixtures) emit CRLF line endings, in which case the separator
    // is `\r\n\r\n`. Normalise by dropping CRs before framing — the SSE
    // spec treats them as line-ending noise anyway.
    let normalised: Vec<u8> = body.iter().copied().filter(|b| *b != b'\r').collect();
    let mut framer = SseFramer::new();
    framer.push(&normalised)?;
    // Some servers don't terminate the final event with a second \n.
    // Pad so `pop_frame` can drain the tail.
    framer.push(b"\n\n")?;

    let mut out = Vec::new();
    while let Some(frame) = framer.pop_frame() {
        let mut joined: Vec<u8> = Vec::new();
        for line in frame.split(|b| *b == b'\n') {
            // Strip optional CR (CRLF line endings).
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if let Some(rest) = line.strip_prefix(b"data:") {
                // Trim one leading space (per SSE spec).
                let rest = rest.strip_prefix(b" ").unwrap_or(rest);
                if !joined.is_empty() {
                    joined.push(b'\n');
                }
                joined.extend_from_slice(rest);
            }
        }
        if !joined.is_empty() {
            out.push(joined);
        }
    }
    Ok(out)
}

// ---- curl plumbing ---------------------------------------------------------

struct Response {
    status: u16,
    body: Vec<u8>,
    content_type_sse: bool,
    session_id: Option<String>,
}

/// One-shot POST. Synchronous; reads the entire body into memory. SSE
/// responses are bounded in practice — one JSON-RPC reply plus optional
/// progress notifications — so this is fine for v1.
fn post(
    url: &str,
    body: &[u8],
    bearer: Option<&str>,
    session: Option<&str>,
) -> Result<Response, String> {
    let url_c = CString::new(url).map_err(|_| "URL contains NUL".to_string())?;

    // Pre-build headers we'll always send.
    let mut header_strings: Vec<CString> = Vec::new();
    header_strings.push(CString::new("content-type: application/json").unwrap());
    // Per spec the client should accept both response shapes.
    header_strings.push(CString::new("accept: application/json, text/event-stream").unwrap());
    if let Some(tok) = bearer {
        header_strings.push(
            CString::new(format!("authorization: Bearer {tok}"))
                .map_err(|_| "bearer token contains NUL".to_string())?,
        );
    }
    if let Some(sid) = session {
        header_strings.push(
            CString::new(format!("mcp-session-id: {sid}"))
                .map_err(|_| "session id contains NUL".to_string())?,
        );
    }

    let mut body_buf: Vec<u8> = Vec::new();
    let mut header_buf: Vec<u8> = Vec::new();

    let perform_status: i64;

    unsafe {
        curl_global_init_once();
        let easy = c::curl_easy_init();
        if easy.is_null() {
            return Err("curl_easy_init failed".into());
        }

        let mut headers: *mut c::curl_slist = ptr::null_mut();
        for h in &header_strings {
            headers = c::curl_slist_append(headers, h.as_ptr());
        }
        let guard = Handle { easy, headers };

        setopt_ptr(easy, c::CURLOPT_URL, url_c.as_ptr() as *const c_void, "URL")?;
        setopt_long(easy, c::CURLOPT_POST, 1, "POST")?;
        setopt_ptr(easy, c::CURLOPT_POSTFIELDS, body.as_ptr() as *const c_void, "POSTFIELDS")?;
        setopt_long(easy, c::CURLOPT_POSTFIELDSIZE, body.len() as i64, "POSTFIELDSIZE")?;
        setopt_ptr(easy, c::CURLOPT_HTTPHEADER, headers as *const c_void, "HTTPHEADER")?;
        setopt_ptr(easy, c::CURLOPT_WRITEFUNCTION, write_cb as *const c_void, "WRITEFUNCTION")?;
        setopt_ptr(
            easy,
            c::CURLOPT_WRITEDATA,
            (&mut body_buf as *mut Vec<u8>) as *const c_void,
            "WRITEDATA",
        )?;
        setopt_ptr(easy, c::CURLOPT_HEADERFUNCTION, header_cb as *const c_void, "HEADERFUNCTION")?;
        setopt_ptr(
            easy,
            c::CURLOPT_HEADERDATA,
            (&mut header_buf as *mut Vec<u8>) as *const c_void,
            "HEADERDATA",
        )?;
        setopt_long(easy, c::CURLOPT_NOPROGRESS, 1, "NOPROGRESS")?;
        setopt_long(easy, c::CURLOPT_FOLLOWLOCATION, 1, "FOLLOWLOCATION")?;
        // Hard timeouts: MCP HTTP is request/response (not streaming), so a
        // half-open / silent-after-handshake socket would otherwise pin the
        // calling thread forever. 10s connect, 30s total.
        setopt_long(easy, c::CURLOPT_CONNECTTIMEOUT, 10, "CONNECTTIMEOUT")?;
        setopt_long(easy, c::CURLOPT_TIMEOUT, 30, "TIMEOUT")?;

        let rc = c::curl_easy_perform(easy);
        let mut status: i64 = 0;
        c::curl_easy_getinfo(easy, c::CURLINFO_RESPONSE_CODE, &mut status);

        drop(guard);

        if rc != c::CURLE_OK {
            return Err(curl_err(rc, "perform"));
        }
        perform_status = status;
    }

    let (content_type_sse, session_id) = parse_headers(&header_buf);

    Ok(Response {
        status: perform_status.clamp(0, u16::MAX as i64) as u16,
        body: body_buf,
        content_type_sse,
        session_id,
    })
}

/// Scan the captured response headers for Content-Type and
/// Mcp-Session-Id. Header buffer holds raw bytes including CRLFs and
/// status lines (curl forwards one call per line, all concatenated by
/// `write_cb`). Status lines like `HTTP/1.1 200 OK` are skipped — they
/// don't contain a colon followed by a value we care about.
fn parse_headers(buf: &[u8]) -> (bool, Option<String>) {
    let mut sse = false;
    let mut session: Option<String> = None;
    for raw_line in buf.split(|b| *b == b'\n') {
        let line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        let Some(colon) = line.iter().position(|b| *b == b':') else {
            continue;
        };
        let name = &line[..colon];
        let value =
            line[colon + 1..].iter().copied().skip_while(|b| *b == b' ').collect::<Vec<_>>();
        if eq_ignore_ascii_case(name, b"content-type") {
            // Content-Type may carry params (charset=utf-8 etc.); we only
            // need the bare type.
            let bare = value.split(|b| *b == b';').next().unwrap_or(&[]);
            let trimmed = trim_ascii(bare);
            if eq_ignore_ascii_case(trimmed, b"text/event-stream") {
                sse = true;
            }
        } else if eq_ignore_ascii_case(name, b"mcp-session-id") {
            let trimmed = trim_ascii(&value);
            if !trimmed.is_empty() {
                session = Some(String::from_utf8_lossy(trimmed).into_owned());
            }
        }
    }
    (sse, session)
}

fn eq_ignore_ascii_case(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.eq_ignore_ascii_case(y))
}

fn trim_ascii(s: &[u8]) -> &[u8] {
    let start = s.iter().position(|b| !b.is_ascii_whitespace()).unwrap_or(s.len());
    let end = s.iter().rposition(|b| !b.is_ascii_whitespace()).map(|i| i + 1).unwrap_or(start);
    &s[start..end]
}

extern "C" fn write_cb(
    ptr: *mut std::os::raw::c_char,
    size: usize,
    nmemb: usize,
    user: *mut c_void,
) -> usize {
    let total = size.saturating_mul(nmemb);
    if total == 0 {
        return 0;
    }
    let buf = unsafe { &mut *(user as *mut Vec<u8>) };
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, total) };
    buf.extend_from_slice(slice);
    total
}

extern "C" fn header_cb(
    ptr: *mut std::os::raw::c_char,
    size: usize,
    nmemb: usize,
    user: *mut c_void,
) -> usize {
    write_cb(ptr, size, nmemb, user)
}

struct Handle {
    easy: *mut c::CURL,
    headers: *mut c::curl_slist,
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            if !self.headers.is_null() {
                c::curl_slist_free_all(self.headers);
            }
            if !self.easy.is_null() {
                c::curl_easy_cleanup(self.easy);
            }
        }
    }
}

unsafe fn setopt_ptr(
    easy: *mut c::CURL,
    opt: c::CURLoption,
    val: *const c_void,
    name: &str,
) -> Result<(), String> {
    let rc = unsafe { c::curl_easy_setopt(easy, opt, val) };
    if rc != c::CURLE_OK {
        return Err(curl_err(rc, name));
    }
    Ok(())
}

unsafe fn setopt_long(
    easy: *mut c::CURL,
    opt: c::CURLoption,
    val: i64,
    name: &str,
) -> Result<(), String> {
    let rc = unsafe { c::curl_easy_setopt(easy, opt, val) };
    if rc != c::CURLE_OK {
        return Err(curl_err(rc, name));
    }
    Ok(())
}

fn curl_err(code: c::CURLcode, where_: &str) -> String {
    let msg = unsafe {
        let p = c::curl_easy_strerror(code);
        if p.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    format!("curl {where_}: {msg} (code {code})")
}

/// Idempotent process-wide `curl_global_init`. Required-once and not
/// thread-safe per libcurl docs; sync::Once makes it safe across parallel
/// callers (tests, multiple McpClient instances).
unsafe fn curl_global_init_once() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| unsafe {
        c::curl_global_init(c::CURL_GLOBAL_DEFAULT);
    });
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests boot a tiny in-process `TcpListener`-based HTTP/1.1 server,
    //! point an `HttpTransport` at it, and assert on the JSON-RPC traffic
    //! flowing through. The mock server is scripted with a list of
    //! responses (and optional per-request body validators) so each test
    //! is self-contained.

    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::Arc;
    use std::sync::Mutex as StdMutex;
    use std::thread;

    use crate::McpClient;

    /// One scripted server reply. `status`, `content_type`, optional
    /// `session_id`, body bytes.
    #[derive(Clone)]
    struct Reply {
        status: u16,
        status_text: &'static str,
        content_type: &'static str,
        session_id: Option<String>,
        body: Vec<u8>,
    }

    impl Reply {
        fn json(status: u16, body: impl Into<Vec<u8>>) -> Self {
            Reply {
                status,
                status_text: status_text(status),
                content_type: "application/json",
                session_id: None,
                body: body.into(),
            }
        }
        fn with_session(mut self, sid: &str) -> Self {
            self.session_id = Some(sid.to_string());
            self
        }
        fn sse(body: impl Into<Vec<u8>>) -> Self {
            Reply {
                status: 200,
                status_text: "OK",
                content_type: "text/event-stream",
                session_id: None,
                body: body.into(),
            }
        }
    }

    fn status_text(code: u16) -> &'static str {
        match code {
            200 => "OK",
            202 => "Accepted",
            400 => "Bad Request",
            404 => "Not Found",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            _ => "Status",
        }
    }

    /// Recorded request captured by the mock server.
    #[derive(Debug, Clone, Default)]
    struct Captured {
        body: Vec<u8>,
        session_id: Option<String>,
        authorization: Option<String>,
        accept: Option<String>,
    }

    /// Single-port mock: takes a scripted Vec<Reply>; serves them in
    /// order; records each captured request. Closes after the script is
    /// exhausted.
    struct Mock {
        addr: String,
        captured: Arc<StdMutex<Vec<Captured>>>,
        _handle: thread::JoinHandle<()>,
        running: Arc<std::sync::atomic::AtomicBool>,
    }

    impl Drop for Mock {
        fn drop(&mut self) {
            self.running.store(false, std::sync::atomic::Ordering::SeqCst);
        }
    }

    fn start_mock(replies: Vec<Reply>) -> Mock {
        use std::sync::atomic::{AtomicBool, Ordering};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();

        let captured = Arc::new(StdMutex::new(Vec::<Captured>::new()));
        let cap_clone = Arc::clone(&captured);
        let total = replies.len();
        let replies = Arc::new(replies);
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);

        let handle = thread::spawn(move || {
            let mut idx = 0usize;
            while running_clone.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        if idx < total {
                            let reply = replies[idx].clone();
                            idx += 1;
                            handle_one(stream, &reply, &cap_clone);
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(std::time::Duration::from_millis(1));
                    }
                    Err(_) => break,
                }
            }
        });

        Mock { addr: format!("http://{addr}/mcp"), captured, _handle: handle, running }
    }

    fn handle_one(
        mut stream: TcpStream,
        reply: &Reply,
        captured_sink: &Arc<StdMutex<Vec<Captured>>>,
    ) {
        let mut reader = match stream.try_clone() {
            Ok(s) => BufReader::new(s),
            Err(_) => return,
        };
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).is_err() {
            return;
        }

        let mut captured = Captured::default();
        let mut content_length: usize = 0;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).is_err() {
                return;
            }
            if line == "\r\n" || line == "\n" || line.is_empty() {
                break;
            }
            let line = line.trim_end_matches(['\r', '\n']);
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim();
            let name_lc = name.to_ascii_lowercase();
            match name_lc.as_str() {
                "content-length" => content_length = value.parse().unwrap_or(0),
                "mcp-session-id" => captured.session_id = Some(value.to_string()),
                "authorization" => captured.authorization = Some(value.to_string()),
                "accept" => captured.accept = Some(value.to_string()),
                _ => {}
            }
        }

        let mut body = vec![0u8; content_length];
        if content_length > 0 && reader.read_exact(&mut body).is_err() {
            return;
        }
        captured.body = body;

        // Record BEFORE responding so the test thread can't read an empty
        // captured Vec between perform completing and the mock loop pushing.
        captured_sink.lock().unwrap().push(captured);

        let mut resp = format!("HTTP/1.1 {} {}\r\n", reply.status, reply.status_text);
        resp.push_str(&format!("content-type: {}\r\n", reply.content_type));
        resp.push_str(&format!("content-length: {}\r\n", reply.body.len()));
        if let Some(sid) = &reply.session_id {
            resp.push_str(&format!("mcp-session-id: {sid}\r\n"));
        }
        resp.push_str("connection: close\r\n\r\n");
        let _ = stream.write_all(resp.as_bytes());
        let _ = stream.write_all(&reply.body);
        let _ = stream.flush();
    }

    fn init_reply(id: u64) -> Vec<u8> {
        format!(
            r#"{{"jsonrpc":"2.0","id":{id},"result":{{"protocolVersion":"2024-11-05","capabilities":{{}},"serverInfo":{{"name":"mock","version":"0"}}}}}}"#
        )
        .into_bytes()
    }

    fn tools_list_reply(id: u64) -> Vec<u8> {
        format!(
            r#"{{"jsonrpc":"2.0","id":{id},"result":{{"tools":[{{"name":"echo","description":"e","inputSchema":{{"type":"object"}}}}]}}}}"#
        )
        .into_bytes()
    }

    /// Stock scripted handshake: initialize → 202 for the notification →
    /// tools/list. Three replies total.
    fn handshake_replies() -> Vec<Reply> {
        vec![
            Reply::json(200, init_reply(1)),
            Reply {
                status: 202,
                status_text: "Accepted",
                content_type: "application/json",
                session_id: None,
                body: Vec::new(),
            },
            Reply::json(200, tools_list_reply(2)),
        ]
    }

    #[test]
    fn initialize_round_trip() {
        let mock = start_mock(handshake_replies());
        let client = McpClient::http(&mock.addr).unwrap();
        let tools = client.tools();
        assert_eq!(tools.len(), 1);
        let cap = mock.captured.lock().unwrap();
        // 3 requests: initialize, notifications/initialized, tools/list.
        assert_eq!(cap.len(), 3);
        let body0 = std::str::from_utf8(&cap[0].body).unwrap();
        assert!(body0.contains(r#""method":"initialize""#));
        let body1 = std::str::from_utf8(&cap[1].body).unwrap();
        assert!(body1.contains(r#""method":"notifications/initialized""#));
        let body2 = std::str::from_utf8(&cap[2].body).unwrap();
        assert!(body2.contains(r#""method":"tools/list""#));
    }

    #[test]
    fn tools_list_populates_catalog() {
        let mock = start_mock(handshake_replies());
        let client = McpClient::http(&mock.addr).unwrap();
        let tools = client.tools();
        use harness::Tool;
        assert_eq!(tools[0].name(), "echo");
        assert_eq!(tools[0].input_schema(), r#"{"type":"object"}"#);
    }

    #[test]
    fn tools_call_round_trip() {
        let mut replies = handshake_replies();
        replies.push(Reply::json(
            200,
            br#"{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"hi"}]}}"#
                .to_vec(),
        ));
        let mock = start_mock(replies);
        let client = McpClient::http(&mock.addr).unwrap();
        let out = client.call_tool("echo", r#"{"x":1}"#).unwrap();
        assert_eq!(out, "hi");
        let cap = mock.captured.lock().unwrap();
        let body = std::str::from_utf8(&cap[3].body).unwrap();
        assert!(body.contains(r#""method":"tools/call""#));
        assert!(body.contains(r#""name":"echo""#));
        assert!(body.contains(r#""arguments":{"x":1}"#));
    }

    #[test]
    fn tools_call_server_error_propagates() {
        let mut replies = handshake_replies();
        replies.push(Reply::json(
            200,
            br#"{"jsonrpc":"2.0","id":3,"error":{"code":-32602,"message":"bad arg"}}"#.to_vec(),
        ));
        let mock = start_mock(replies);
        let client = McpClient::http(&mock.addr).unwrap();
        let err = client.call_tool("echo", "{}").unwrap_err();
        let s = err.to_string();
        assert!(s.contains("-32602"));
        assert!(s.contains("bad arg"));
    }

    #[test]
    fn tools_call_is_error_block_surfaces() {
        let mut replies = handshake_replies();
        replies.push(Reply::json(
            200,
            br#"{"jsonrpc":"2.0","id":3,"result":{"isError":true,"content":[{"type":"text","text":"boom"}]}}"#.to_vec(),
        ));
        let mock = start_mock(replies);
        let client = McpClient::http(&mock.addr).unwrap();
        let err = client.call_tool("echo", "{}").unwrap_err();
        assert!(err.to_string().contains("boom"));
    }

    #[test]
    fn session_id_is_echoed_on_subsequent_requests() {
        let mut replies = handshake_replies();
        replies[0] = Reply::json(200, init_reply(1)).with_session("sess-abc");
        replies.push(Reply::json(
            200,
            br#"{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"ok"}]}}"#
                .to_vec(),
        ));
        let mock = start_mock(replies);
        let client = McpClient::http(&mock.addr).unwrap();
        let _ = client.call_tool("echo", "{}").unwrap();
        let cap = mock.captured.lock().unwrap();
        // First request (initialize) had no session id.
        assert!(cap[0].session_id.is_none());
        // Every subsequent request must carry it.
        assert_eq!(cap[1].session_id.as_deref(), Some("sess-abc"));
        assert_eq!(cap[2].session_id.as_deref(), Some("sess-abc"));
        assert_eq!(cap[3].session_id.as_deref(), Some("sess-abc"));
    }

    #[test]
    fn sse_response_body_is_decoded() {
        // Initialize answers with SSE: one event per data: block carrying
        // the JSON-RPC reply.
        let mut replies = handshake_replies();
        replies[0] = Reply::sse(
            br#"data: {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"sse","version":"0"}}}

"#
            .to_vec(),
        );
        let mock = start_mock(replies);
        let client = McpClient::http(&mock.addr).unwrap();
        assert_eq!(client.tools().len(), 1);
    }

    #[test]
    fn sse_multi_data_line_event_is_joined() {
        // A single event with two data: lines. Per SSE spec the values
        // get \n-joined into one logical message. Used by servers that
        // pretty-print JSON across lines (rare but legal).
        let json_lines = b"data: {\"jsonrpc\":\"2.0\",\"id\":1,\ndata: \"result\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"serverInfo\":{\"name\":\"x\",\"version\":\"0\"}}}\n\n";
        let mut replies = handshake_replies();
        replies[0] = Reply::sse(json_lines.to_vec());
        let mock = start_mock(replies);
        let client = McpClient::http(&mock.addr).unwrap();
        assert_eq!(client.tools().len(), 1);
    }

    #[test]
    fn http_5xx_becomes_transport_error() {
        let replies = vec![Reply::json(503, b"upstream down".to_vec())];
        let mock = start_mock(replies);
        let err = match McpClient::http(&mock.addr) {
            Err(e) => e,
            Ok(_) => panic!("expected error"),
        };
        let s = err.to_string();
        assert!(s.contains("503"));
        assert!(s.contains("server error"));
        assert!(s.contains("upstream down"));
    }

    #[test]
    fn http_4xx_includes_body() {
        let replies = vec![Reply::json(404, b"no such endpoint".to_vec())];
        let mock = start_mock(replies);
        let err = match McpClient::http(&mock.addr) {
            Err(e) => e,
            Ok(_) => panic!("expected error"),
        };
        let s = err.to_string();
        assert!(s.contains("404"));
        assert!(s.contains("client error"));
        assert!(s.contains("no such endpoint"));
    }

    #[test]
    fn bearer_token_is_attached() {
        let replies = handshake_replies();
        let mock = start_mock(replies);
        let _ = McpClient::http_with_bearer(&mock.addr, "tok-123").unwrap();
        let cap = mock.captured.lock().unwrap();
        for c in cap.iter() {
            assert_eq!(c.authorization.as_deref(), Some("Bearer tok-123"));
        }
    }

    #[test]
    fn accept_header_advertises_both_response_types() {
        let mock = start_mock(handshake_replies());
        let _ = McpClient::http(&mock.addr).unwrap();
        let cap = mock.captured.lock().unwrap();
        let accept = cap[0].accept.as_deref().unwrap_or("");
        assert!(accept.contains("application/json"));
        assert!(accept.contains("text/event-stream"));
    }

    #[test]
    fn empty_body_for_notification_is_tolerated() {
        // The notifications/initialized POST returns 202 with no body
        // (reply[1] in handshake_replies). If we mis-handle that, the
        // whole handshake fails. Covered indirectly by other tests, but
        // pin it down explicitly.
        let mock = start_mock(handshake_replies());
        let client = McpClient::http(&mock.addr).unwrap();
        assert_eq!(client.tools().len(), 1);
    }

    #[test]
    fn session_id_persists_across_transport_after_initialize() {
        let mut replies = handshake_replies();
        replies[0] = Reply::json(200, init_reply(1)).with_session("S1");
        let mock = start_mock(replies);
        let transport = HttpTransport::new(&mock.addr);
        let client = McpClient::with_transport(Box::new(transport)).unwrap();
        // We can't peek at the transport directly (Box<dyn Transport>),
        // but we can verify via the wire: any further request sees S1.
        // Pull a frame to ensure the handshake completed.
        assert_eq!(client.tools().len(), 1);
        let cap = mock.captured.lock().unwrap();
        assert_eq!(cap[1].session_id.as_deref(), Some("S1"));
    }

    // ---- direct transport unit tests (no McpClient) ----

    /// A silent peer: TcpListener that `accept`s once and never sends a
    /// reply. Models a half-open / LB-dropped socket. Without
    /// CURLOPT_TIMEOUT the `post` call would block until the OS finally
    /// gives up (often hours). With the 30s hard cap, `send_line` returns
    /// Err well inside that bound.
    #[test]
    fn silent_peer_trips_hard_timeout() {
        use std::sync::atomic::AtomicBool;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_c = Arc::clone(&stop);
        let parked = thread::spawn(move || {
            listener.set_nonblocking(true).ok();
            let mut held: Option<TcpStream> = None;
            while !stop_c.load(std::sync::atomic::Ordering::Relaxed) {
                if held.is_none()
                    && let Ok((s, _)) = listener.accept()
                {
                    held = Some(s);
                }
                thread::sleep(std::time::Duration::from_millis(50));
            }
            drop(held);
        });

        let url = format!("http://{addr}/mcp");
        let transport = HttpTransport::new(&url);

        let started = std::time::Instant::now();
        let err = transport.send_line(b"{}").unwrap_err();
        let elapsed = started.elapsed();

        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = parked.join();

        // CURLOPT_TIMEOUT=30s; allow 35s upper bound for scheduler jitter.
        // Load-bearing: the call returns at all, instead of blocking until
        // the OS gives up.
        assert!(
            elapsed < std::time::Duration::from_secs(35),
            "send_line must abort within ~30s; took {elapsed:?}"
        );
        // curl_easy_perform yields CURLE_OPERATION_TIMEDOUT (28); the post()
        // helper packages it as "curl perform: ... (code 28)".
        assert!(err.contains("code 28"), "expected CURLE_OPERATION_TIMEDOUT, got {err}");
    }

    #[test]
    fn parse_sse_data_frames_handles_crlf() {
        let body = b"data: hello\r\n\r\ndata: world\r\n\r\n";
        let frames = parse_sse_data_frames(body).unwrap();
        assert_eq!(frames, vec![b"hello".to_vec(), b"world".to_vec()]);
    }

    #[test]
    fn parse_sse_data_frames_ignores_comments_and_event_lines() {
        let body = b": keep-alive\nevent: progress\ndata: {\"x\":1}\n\n";
        let frames = parse_sse_data_frames(body).unwrap();
        assert_eq!(frames, vec![br#"{"x":1}"#.to_vec()]);
    }

    #[test]
    fn parse_sse_data_frames_rejects_oversize_body() {
        // 5 MiB body with no boundaries — exceeds the 4 MiB default cap.
        let body = vec![b'x'; 5 * 1024 * 1024];
        assert!(parse_sse_data_frames(&body).is_err());
    }
}
