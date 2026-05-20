//! Model Context Protocol client.
//!
//! Speaks JSON-RPC 2.0 over a transport. v1 ships **stdio only**: spawn a
//! child process, write requests as newline-delimited JSON to stdin, read
//! responses as newline-delimited JSON from stdout. Used by most local
//! MCP tool packages. Streamable HTTP transport is deferred to v2.
//!
//! The handshake (`initialize` → `notifications/initialized` → `tools/list`)
//! runs in [`McpClient::stdio`]; the returned client exposes a catalog of
//! [`McpTool`]s, each implementing [`harness::Tool`] so they slot directly
//! into the harness's tool registry alongside built-ins.

mod jsonrpc;
mod transport;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anthropic::json::{Json, parse as parse_json};
use harness::{Tool, ToolCtx, ToolError};

pub use transport::{StdioTransport, Transport};

/// Errors surfaced by the MCP client. Compact on purpose: most failures
/// upstream just want a message to log, not a typed taxonomy.
#[derive(Debug)]
pub enum McpError {
    /// Transport-level failure: spawn, pipe close, broken stdout, etc.
    Transport(String),
    /// Protocol-level failure: malformed JSON-RPC, missing fields, unexpected
    /// notification while awaiting a response.
    Protocol(String),
    /// The server returned an `error` object in response to a request.
    Server { code: i64, message: String },
}

impl std::fmt::Display for McpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpError::Transport(m) => write!(f, "transport: {m}"),
            McpError::Protocol(m) => write!(f, "protocol: {m}"),
            McpError::Server { code, message } => write!(f, "server error {code}: {message}"),
        }
    }
}

impl std::error::Error for McpError {}

/// Description of one tool advertised by the server. The schema is kept as
/// a raw JSON object string (same shape as `harness::Tool::input_schema`).
#[derive(Clone, Debug)]
struct ToolMeta {
    name: String,
    description: String,
    input_schema: String,
}

/// Shared state between the client and the [`McpTool`]s it produces. Holds
/// the transport, a monotonic request-id counter, and the cached tool
/// catalog. Behind `Arc` so tools outlive the explicit client handle.
pub(crate) struct McpInner {
    transport: Box<dyn Transport>,
    next_id: AtomicU64,
    tools: Vec<ToolMeta>,
}

impl McpInner {
    /// Allocate the next JSON-RPC request id. Monotonic per-client.
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Send a request and read the matching response. Skips intervening
    /// notifications (no id) — servers may emit `notifications/...` at any
    /// time and we don't subscribe to any yet, so we discard them.
    ///
    /// Returns the parsed top-level frame whose `result` field is what the
    /// caller will inspect. We re-parse rather than thread lifetimes
    /// through the request flow.
    fn request(&self, method: &str, params: Option<&str>) -> Result<Json, McpError> {
        let id = self.next_id();
        let body = jsonrpc::build_request(id, method, params);
        self.transport
            .send_line(body.as_bytes())
            .map_err(McpError::Transport)?;
        loop {
            let line = self
                .transport
                .recv_line()
                .map_err(McpError::Transport)?;
            // Servers may emit blank keep-alives; ignore.
            if line.iter().all(|b| matches!(b, b' ' | b'\t' | b'\r')) {
                continue;
            }
            let v = parse_json(&line)
                .map_err(|e| McpError::Protocol(format!("invalid JSON-RPC frame: {e}")))?;
            match jsonrpc::classify(&v) {
                jsonrpc::FrameKind::Response { id: rid, outcome } => {
                    if rid != id {
                        // Stale response from a prior request — drop it. In
                        // practice this only happens if a caller errors mid-
                        // request and reuses the client; we'd rather skip
                        // than wedge.
                        continue;
                    }
                    match outcome {
                        jsonrpc::Outcome::Error(e) => return Err(e),
                        jsonrpc::Outcome::Result => {
                            // The result subtree lives inside `v`; rather than
                            // clone it (Json doesn't implement Clone), return
                            // the whole envelope. The caller looks it up by
                            // key.
                            return Ok(v);
                        }
                    }
                }
                jsonrpc::FrameKind::Notification => continue,
                jsonrpc::FrameKind::Unknown => {
                    return Err(McpError::Protocol(format!(
                        "unrecognised frame: {}",
                        String::from_utf8_lossy(&line)
                    )));
                }
            }
        }
    }

    /// Send a notification (no id, no response expected).
    fn notify(&self, method: &str, params: Option<&str>) -> Result<(), McpError> {
        let body = jsonrpc::build_notification(method, params);
        self.transport
            .send_line(body.as_bytes())
            .map_err(McpError::Transport)
    }
}

/// MCP client. Owns the transport and a cached tool catalog.
pub struct McpClient {
    inner: Arc<McpInner>,
}

impl McpClient {
    /// Spawn `command args...`, perform the MCP handshake, and fetch the
    /// tool catalog. The child process is killed on drop.
    pub fn stdio(command: &str, args: &[&str]) -> Result<Self, McpError> {
        let transport = StdioTransport::spawn(command, args).map_err(McpError::Transport)?;
        Self::with_transport(Box::new(transport))
    }

    /// Build a client from an arbitrary transport. Public for tests; in
    /// production, prefer [`McpClient::stdio`].
    pub fn with_transport(transport: Box<dyn Transport>) -> Result<Self, McpError> {
        let mut inner = McpInner {
            transport,
            next_id: AtomicU64::new(1),
            tools: Vec::new(),
        };

        // ---- initialize ----
        // Per spec, params include protocolVersion, capabilities, and
        // clientInfo. We declare empty capabilities — we only consume tools,
        // and don't subscribe to resources/prompts/sampling.
        let init_params = concat!(
            r#"{"protocolVersion":"2024-11-05","#,
            r#""capabilities":{},"#,
            r#""clientInfo":{"name":"sandcastle-mcp","version":"0.0.1"}}"#
        );
        let init_envelope = inner.request("initialize", Some(init_params))?;
        let init_resp = init_envelope.get("result").ok_or_else(|| {
            McpError::Protocol("initialize response missing 'result'".into())
        })?;
        // We don't strictly need to inspect serverInfo, but a missing
        // protocolVersion is a clear sign of a non-MCP peer.
        if init_resp.get("protocolVersion").is_none() {
            return Err(McpError::Protocol(
                "initialize response missing protocolVersion".into(),
            ));
        }

        // ---- initialized notification ----
        inner.notify("notifications/initialized", None)?;

        // ---- tools/list ----
        let list_envelope = inner.request("tools/list", None)?;
        let list = list_envelope.get("result").ok_or_else(|| {
            McpError::Protocol("tools/list response missing 'result'".into())
        })?;
        inner.tools = parse_tools_list(list)?;

        Ok(Self { inner: Arc::new(inner) })
    }

    /// Snapshot of the tools advertised by the server, each ready to be
    /// registered with the harness. Cheap to call repeatedly — clones a
    /// handful of small strings per tool.
    pub fn tools(&self) -> Vec<McpTool> {
        self.inner
            .tools
            .iter()
            .map(|t| McpTool {
                inner: Arc::clone(&self.inner),
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.input_schema.clone(),
            })
            .collect()
    }

    /// Dispatch a `tools/call` request without going through [`McpTool`].
    /// Intended for tests and for advanced callers that want to invoke
    /// the server by name without registering the tool with the harness.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn call_tool(&self, name: &str, args: &str) -> Result<String, McpError> {
        self.inner.call_tool(name, args)
    }
}

impl McpInner {
    fn call_tool(&self, name: &str, args: &str) -> Result<String, McpError> {
        // arguments must be a JSON object; the spec says "object" with
        // unspecified shape. Empty input -> {}.
        let args = args.trim();
        let args = if args.is_empty() { "{}" } else { args };
        let mut params = String::with_capacity(32 + name.len() + args.len());
        params.push_str(r#"{"name":""#);
        anthropic::json::escape_into(&mut params, name);
        params.push_str(r#"","arguments":"#);
        params.push_str(args);
        params.push('}');
        let envelope = self.request("tools/call", Some(&params))?;
        let result = envelope.get("result").ok_or_else(|| {
            McpError::Protocol("tools/call response missing 'result'".into())
        })?;
        // The spec returns `{ content: [...], isError?: bool }`. We
        // concatenate all `text` blocks and ignore others (images,
        // resources). If `isError` is true, surface as a ToolError —
        // matching the harness's tool-error contract.
        let is_error = matches!(result.get("isError"), Some(Json::Bool(true)));
        let text = extract_text(result)?;
        if is_error {
            Err(McpError::Server { code: 0, message: text })
        } else {
            Ok(text)
        }
    }
}

/// A single MCP tool, bound to its client. Implements [`harness::Tool`] so
/// the harness can register it next to built-ins.
pub struct McpTool {
    inner: Arc<McpInner>,
    name: String,
    description: String,
    input_schema: String,
}

impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> &str {
        &self.input_schema
    }

    fn call(&self, input: &str, _ctx: &ToolCtx) -> Result<String, ToolError> {
        self.inner
            .call_tool(&self.name, input)
            .map_err(|e| ToolError(e.to_string()))
    }
}

fn parse_tools_list(v: &Json) -> Result<Vec<ToolMeta>, McpError> {
    let arr = match v.get("tools") {
        Some(Json::Arr(a)) => a,
        _ => return Err(McpError::Protocol("tools/list: no 'tools' array".into())),
    };
    let mut out = Vec::with_capacity(arr.len());
    for t in arr {
        let name = t
            .get("name")
            .and_then(Json::as_str)
            .ok_or_else(|| McpError::Protocol("tool entry missing 'name'".into()))?
            .to_string();
        let description = t
            .get("description")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string();
        // inputSchema is a JSON Schema object. Re-serialise the parsed value
        // so we don't depend on string-level round-tripping from the wire.
        let schema = t
            .get("inputSchema")
            .ok_or_else(|| McpError::Protocol(format!("tool '{name}' missing inputSchema")))?;
        let mut schema_str = String::new();
        write_json(&mut schema_str, schema);
        out.push(ToolMeta { name, description, input_schema: schema_str });
    }
    Ok(out)
}

fn extract_text(result: &Json) -> Result<String, McpError> {
    let arr = match result.get("content") {
        Some(Json::Arr(a)) => a,
        Some(_) => return Err(McpError::Protocol("tools/call: content not an array".into())),
        // Some servers return no content for void tools; treat as empty.
        None => return Ok(String::new()),
    };
    let mut out = String::new();
    for block in arr {
        let ty = block.get("type").and_then(Json::as_str).unwrap_or("");
        if ty == "text"
            && let Some(t) = block.get("text").and_then(Json::as_str)
        {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(t);
        }
        // Image / resource blocks are silently skipped in v1.
    }
    Ok(out)
}

/// Re-serialise a parsed JSON value. We need this so we can hand the schema
/// to the harness as a raw JSON string regardless of the source formatting.
/// Numbers are stored as their source bytes by the parser, so we pass them
/// through verbatim.
fn write_json(out: &mut String, v: &Json) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::Num(n) => out.push_str(n),
        Json::Str(s) => {
            out.push('"');
            anthropic::json::escape_into(out, s);
            out.push('"');
        }
        Json::Arr(items) => {
            out.push('[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(out, it);
            }
            out.push(']');
        }
        Json::Obj(kv) => {
            out.push('{');
            for (i, (k, val)) in kv.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('"');
                anthropic::json::escape_into(out, k);
                out.push_str("\":");
                write_json(out, val);
            }
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::mpsc::{Receiver, Sender, channel};

    /// In-memory transport: tests push canned response lines onto one
    /// channel and pull sent request lines off another. Lets us exercise
    /// the full handshake + dispatch without spawning processes.
    struct MockTransport {
        sent: Mutex<Sender<Vec<u8>>>,
        sent_rx: Mutex<Receiver<Vec<u8>>>,
        incoming_rx: Mutex<Receiver<Result<Vec<u8>, String>>>,
        incoming_tx: Mutex<Sender<Result<Vec<u8>, String>>>,
    }

    impl MockTransport {
        fn new() -> Arc<Self> {
            let (sent_tx, sent_rx) = channel();
            let (in_tx, in_rx) = channel();
            Arc::new(Self {
                sent: Mutex::new(sent_tx),
                sent_rx: Mutex::new(sent_rx),
                incoming_rx: Mutex::new(in_rx),
                incoming_tx: Mutex::new(in_tx),
            })
        }

        fn push_response(&self, line: &str) {
            self.incoming_tx
                .lock()
                .unwrap()
                .send(Ok(line.as_bytes().to_vec()))
                .unwrap();
        }

        fn push_error(&self, msg: &str) {
            self.incoming_tx
                .lock()
                .unwrap()
                .send(Err(msg.to_string()))
                .unwrap();
        }

        fn pop_sent(&self) -> Vec<u8> {
            self.sent_rx.lock().unwrap().recv().unwrap()
        }

        fn try_pop_sent(&self) -> Option<Vec<u8>> {
            self.sent_rx.lock().unwrap().try_recv().ok()
        }
    }

    /// Wrapper newtype because `Transport` requires `Box<dyn Transport>`
    /// and we want to keep the `Arc<MockTransport>` around in the test.
    struct MockHandle(Arc<MockTransport>);

    impl Transport for MockHandle {
        fn send_line(&self, line: &[u8]) -> Result<(), String> {
            self.0.sent.lock().unwrap().send(line.to_vec()).unwrap();
            Ok(())
        }

        fn recv_line(&self) -> Result<Vec<u8>, String> {
            match self.0.incoming_rx.lock().unwrap().recv() {
                Ok(Ok(v)) => Ok(v),
                Ok(Err(e)) => Err(e),
                Err(_) => Err("channel closed".into()),
            }
        }
    }

    fn init_response(id: u64) -> String {
        format!(
            r#"{{"jsonrpc":"2.0","id":{id},"result":{{"protocolVersion":"2024-11-05","capabilities":{{}},"serverInfo":{{"name":"mock","version":"0"}}}}}}"#
        )
    }

    fn tools_list_response(id: u64) -> String {
        format!(
            r#"{{"jsonrpc":"2.0","id":{id},"result":{{"tools":[{{"name":"echo","description":"Echo input.","inputSchema":{{"type":"object","properties":{{"text":{{"type":"string"}}}},"required":["text"]}}}},{{"name":"add","description":"Adds two numbers.","inputSchema":{{"type":"object","properties":{{"a":{{"type":"number"}},"b":{{"type":"number"}}}}}}}}]}}}}"#
        )
    }

    fn build_handshake(mock: &MockTransport) {
        mock.push_response(&init_response(1));
        mock.push_response(&tools_list_response(2));
    }

    fn build_client(mock: Arc<MockTransport>) -> McpClient {
        let handle = MockHandle(Arc::clone(&mock));
        McpClient::with_transport(Box::new(handle)).unwrap()
    }

    fn ctx_dummy() -> harness::ToolCtx<'static> {
        // We never use the instance for MCP tools, but the trait demands
        // one. Construct a leaked Spawned so we get a valid ActorRef
        // pointer for the lifetime of the test. Cheap; tests are short.
        use actor::spawn;
        use harness::{Instance, MockSandbox, Sandbox};
        let sb: Box<dyn Sandbox> = Box::new(MockSandbox::new(vec![]));
        let inst = Box::leak(Box::new(spawn(Instance::new("t", sb))));
        ToolCtx { instance: &inst.addr }
    }

    // ---- handshake ----

    #[test]
    fn handshake_sends_initialize_and_initialized_notification() {
        let mock = MockTransport::new();
        build_handshake(&mock);
        let _client = build_client(Arc::clone(&mock));

        let init = mock.pop_sent();
        let init_str = std::str::from_utf8(&init).unwrap();
        assert!(init_str.contains(r#""method":"initialize""#));
        assert!(init_str.contains(r#""protocolVersion":"2024-11-05""#));
        assert!(init_str.contains(r#""id":1"#));

        let notified = mock.pop_sent();
        let notif = std::str::from_utf8(&notified).unwrap();
        assert!(notif.contains(r#""method":"notifications/initialized""#));
        // notifications must not carry an id
        assert!(!notif.contains(r#""id":"#));

        let list = mock.pop_sent();
        assert!(std::str::from_utf8(&list).unwrap().contains(r#""method":"tools/list""#));
    }

    #[test]
    fn handshake_populates_tool_catalog() {
        let mock = MockTransport::new();
        build_handshake(&mock);
        let client = build_client(mock);
        let tools = client.tools();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name(), "echo");
        assert_eq!(tools[0].description(), "Echo input.");
        assert!(tools[0].input_schema().contains(r#""required":["text"]"#));
        assert_eq!(tools[1].name(), "add");
    }

    #[test]
    fn handshake_rejects_response_missing_protocol_version() {
        let mock = MockTransport::new();
        mock.push_response(r#"{"jsonrpc":"2.0","id":1,"result":{"capabilities":{}}}"#);
        let handle = MockHandle(Arc::clone(&mock));
        let err = match McpClient::with_transport(Box::new(handle)) {
            Err(e) => e,
            Ok(_) => panic!("expected error"),
        };
        assert!(matches!(err, McpError::Protocol(m) if m.contains("protocolVersion")));
    }

    #[test]
    fn handshake_propagates_server_error() {
        let mock = MockTransport::new();
        mock.push_response(
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"bad request"}}"#,
        );
        let handle = MockHandle(Arc::clone(&mock));
        let err = match McpClient::with_transport(Box::new(handle)) {
            Err(e) => e,
            Ok(_) => panic!("expected error"),
        };
        match err {
            McpError::Server { code, message } => {
                assert_eq!(code, -32600);
                assert!(message.contains("bad request"));
            }
            other => panic!("expected Server, got {other:?}"),
        }
    }

    #[test]
    fn handshake_transport_failure_surfaces() {
        let mock = MockTransport::new();
        mock.push_error("eof");
        let handle = MockHandle(Arc::clone(&mock));
        let err = match McpClient::with_transport(Box::new(handle)) {
            Err(e) => e,
            Ok(_) => panic!("expected error"),
        };
        assert!(matches!(err, McpError::Transport(m) if m.contains("eof")));
    }

    // ---- tools/call ----

    #[test]
    fn tools_call_concatenates_text_blocks() {
        let mock = MockTransport::new();
        build_handshake(&mock);
        let client = build_client(Arc::clone(&mock));
        // drain the three handshake sends so the next pop_sent matches the
        // call we're about to make.
        for _ in 0..3 {
            let _ = mock.pop_sent();
        }
        mock.push_response(
            r#"{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"hello "},{"type":"text","text":"world"}]}}"#,
        );
        let tools = client.tools();
        let out = tools[0].call(r#"{"text":"hi"}"#, &ctx_dummy()).unwrap();
        // text blocks are joined with \n in v1.
        assert_eq!(out, "hello \nworld");
        // Verify the sent payload contained the tool name and arguments.
        let sent = mock.pop_sent();
        let s = std::str::from_utf8(&sent).unwrap();
        assert!(s.contains(r#""method":"tools/call""#));
        assert!(s.contains(r#""name":"echo""#));
        assert!(s.contains(r#""arguments":{"text":"hi"}"#));
    }

    #[test]
    fn tools_call_ignores_non_text_blocks() {
        let mock = MockTransport::new();
        build_handshake(&mock);
        let client = build_client(Arc::clone(&mock));
        for _ in 0..3 {
            let _ = mock.pop_sent();
        }
        mock.push_response(
            r#"{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"image","data":"..."},{"type":"text","text":"only this"}]}}"#,
        );
        let out = client.tools()[0]
            .call("{}", &ctx_dummy())
            .unwrap();
        assert_eq!(out, "only this");
    }

    #[test]
    fn tools_call_is_error_surfaces_as_tool_error() {
        let mock = MockTransport::new();
        build_handshake(&mock);
        let client = build_client(Arc::clone(&mock));
        for _ in 0..3 {
            let _ = mock.pop_sent();
        }
        mock.push_response(
            r#"{"jsonrpc":"2.0","id":3,"result":{"isError":true,"content":[{"type":"text","text":"boom"}]}}"#,
        );
        let err = client.tools()[0]
            .call("{}", &ctx_dummy())
            .unwrap_err();
        assert!(err.0.contains("boom"));
    }

    #[test]
    fn tools_call_protocol_error_via_jsonrpc_error() {
        let mock = MockTransport::new();
        build_handshake(&mock);
        let client = build_client(Arc::clone(&mock));
        for _ in 0..3 {
            let _ = mock.pop_sent();
        }
        mock.push_response(
            r#"{"jsonrpc":"2.0","id":3,"error":{"code":-32602,"message":"missing argument"}}"#,
        );
        let err = client.tools()[0]
            .call("{}", &ctx_dummy())
            .unwrap_err();
        assert!(err.0.contains("-32602"));
        assert!(err.0.contains("missing argument"));
    }

    #[test]
    fn tools_call_empty_input_becomes_empty_object() {
        let mock = MockTransport::new();
        build_handshake(&mock);
        let client = build_client(Arc::clone(&mock));
        for _ in 0..3 {
            let _ = mock.pop_sent();
        }
        mock.push_response(
            r#"{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"ok"}]}}"#,
        );
        client.tools()[0].call("", &ctx_dummy()).unwrap();
        let sent = mock.pop_sent();
        let s = std::str::from_utf8(&sent).unwrap();
        assert!(s.contains(r#""arguments":{}"#));
    }

    #[test]
    fn intervening_notification_is_skipped_before_response() {
        let mock = MockTransport::new();
        build_handshake(&mock);
        let client = build_client(Arc::clone(&mock));
        for _ in 0..3 {
            let _ = mock.pop_sent();
        }
        // Server emits a notification, then the actual response. The
        // client should silently skip the notification.
        mock.push_response(r#"{"jsonrpc":"2.0","method":"notifications/log","params":{"msg":"x"}}"#);
        mock.push_response(
            r#"{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"ok"}]}}"#,
        );
        let out = client.tools()[0].call("{}", &ctx_dummy()).unwrap();
        assert_eq!(out, "ok");
    }

    #[test]
    fn transport_closed_mid_call() {
        let mock = MockTransport::new();
        build_handshake(&mock);
        let client = build_client(Arc::clone(&mock));
        for _ in 0..3 {
            let _ = mock.pop_sent();
        }
        mock.push_error("broken pipe");
        let err = client.tools()[0]
            .call("{}", &ctx_dummy())
            .unwrap_err();
        assert!(err.0.contains("broken pipe"));
    }

    #[test]
    fn tools_call_no_content_field_returns_empty_string() {
        let mock = MockTransport::new();
        build_handshake(&mock);
        let client = build_client(Arc::clone(&mock));
        for _ in 0..3 {
            let _ = mock.pop_sent();
        }
        mock.push_response(r#"{"jsonrpc":"2.0","id":3,"result":{}}"#);
        let out = client.tools()[0].call("{}", &ctx_dummy()).unwrap();
        assert_eq!(out, "");
    }

    // ---- catalog edge cases ----

    #[test]
    fn tools_list_with_no_tools_yields_empty_catalog() {
        let mock = MockTransport::new();
        mock.push_response(&init_response(1));
        mock.push_response(r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[]}}"#);
        let client = build_client(mock);
        assert!(client.tools().is_empty());
    }

    #[test]
    fn tool_with_no_description_uses_empty_string() {
        let mock = MockTransport::new();
        mock.push_response(&init_response(1));
        mock.push_response(
            r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"t","inputSchema":{"type":"object"}}]}}"#,
        );
        let client = build_client(mock);
        let tools = client.tools();
        assert_eq!(tools[0].name(), "t");
        assert_eq!(tools[0].description(), "");
        assert_eq!(tools[0].input_schema(), r#"{"type":"object"}"#);
    }

    #[test]
    fn tool_entry_missing_input_schema_is_rejected() {
        let mock = MockTransport::new();
        mock.push_response(&init_response(1));
        mock.push_response(
            r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"t","description":"d"}]}}"#,
        );
        let handle = MockHandle(Arc::clone(&mock));
        let err = match McpClient::with_transport(Box::new(handle)) {
            Err(e) => e,
            Ok(_) => panic!("expected error"),
        };
        assert!(matches!(err, McpError::Protocol(m) if m.contains("inputSchema")));
    }

    // ---- stdio integration ----

    #[test]
    fn stdio_transport_handshakes_against_scripted_server() {
        // A POSIX shell script that fakes an MCP server: reads one
        // request, prints the initialize response; reads the initialized
        // notification; reads tools/list, prints an empty catalog. We
        // never call any tools — just want to prove StdioTransport wires
        // up correctly.
        let script = r#"
            read line; printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"sh","version":"0"}}}'
            read line
            read line; printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"tools":[]}}'
        "#;
        let client = McpClient::stdio("sh", &["-c", script]).unwrap();
        assert!(client.tools().is_empty());
    }

    #[test]
    fn stdio_transport_round_trip_tool_call() {
        // Same idea as above but with one tool that echoes whatever
        // argument it gets.
        let script = r#"
            read line; printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"sh","version":"0"}}}'
            read line
            read line; printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"echo","description":"e","inputSchema":{"type":"object"}}]}}'
            read line; printf '%s\n' '{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"hi"}]}}'
        "#;
        let client = McpClient::stdio("sh", &["-c", script]).unwrap();
        let tools = client.tools();
        assert_eq!(tools.len(), 1);
        let out = tools[0].call(r#"{"x":1}"#, &ctx_dummy()).unwrap();
        assert_eq!(out, "hi");
    }

    #[test]
    fn stdio_transport_drop_kills_child() {
        // Spawn a script that would otherwise block forever, drop the
        // transport, and verify the child got reaped. `kill(pid, 0)`
        // returns 0 while the process is alive and -1 (ESRCH) once it's
        // gone. Drop calls both kill() and wait(), so after drop returns
        // the process should be fully reaped — but the kernel can take a
        // few microseconds; poll briefly to avoid flakes.
        let script = "sleep 30";
        let t = transport::StdioTransport::spawn("sh", &["-c", script]).unwrap();
        let pid = t.child_id();
        drop(t);
        let dead = (0..50).any(|_| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            unsafe { libc_kill(pid as i32, 0) != 0 }
        });
        assert!(dead, "child {pid} still alive after drop");
    }

    unsafe extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }

    // Aliased so the body above reads naturally; clippy::missing_safety
    // doesn't fire because the use site is already in `unsafe`.
    unsafe fn libc_kill(pid: i32, sig: i32) -> i32 {
        unsafe { kill(pid, sig) }
    }

    // ---- misc ----

    #[test]
    fn json_round_trip_preserves_schema_shape() {
        // Smoke-test write_json against a moderately tangled schema.
        let src = r#"{"type":"object","properties":{"x":{"type":"array","items":{"type":"string"}}},"required":["x"]}"#;
        let parsed = parse_json(src.as_bytes()).unwrap();
        let mut buf = String::new();
        write_json(&mut buf, &parsed);
        // Re-parse + compare structurally — key order is preserved by our
        // parser, so a string compare is also valid here.
        assert_eq!(buf, src);
    }

    #[test]
    fn direct_call_tool_bypasses_mctool_wrapper() {
        // The pub(crate) `call_tool` lets the harness or wiring code
        // invoke a server tool by name without first materialising an
        // McpTool. Exercise it so it doesn't bit-rot.
        let mock = MockTransport::new();
        build_handshake(&mock);
        let client = build_client(Arc::clone(&mock));
        for _ in 0..3 {
            let _ = mock.pop_sent();
        }
        mock.push_response(
            r#"{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"direct"}]}}"#,
        );
        let out = client.call_tool("echo", r#"{"text":"x"}"#).unwrap();
        assert_eq!(out, "direct");
    }

    #[test]
    fn try_pop_sent_is_idle_after_handshake_drained() {
        let mock = MockTransport::new();
        build_handshake(&mock);
        let _client = build_client(Arc::clone(&mock));
        for _ in 0..3 {
            let _ = mock.pop_sent();
        }
        // Nothing else should have been written.
        assert!(mock.try_pop_sent().is_none());
    }
}
