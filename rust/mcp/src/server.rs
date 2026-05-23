//! Model Context Protocol server.
//!
//! Complement to the `mcp` client. Speaks JSON-RPC 2.0 over newline-
//! delimited JSON on stdio (or any [`BufRead`] + [`Write`] pair). Exposes
//! a [`harness::Tool`] registry: clients call `initialize`, `tools/list`,
//! and `tools/call`; we dispatch each `tools/call` through the tool's
//! `call` method using the Instance actor the caller registers.
//!
//! Single-client by design — stdio MCP is point-to-point. Requests are
//! processed sequentially; tools may block (bash exec) but the protocol
//! has no concurrency.

use std::io::{BufRead, Write};
use std::sync::Arc;

use actor::ActorRef;
use anthropic::json::{Json, escape_into, parse as parse_json};
use harness::{InstanceMsg, Tool, ToolCtx};
use wire::NdjsonSplitter;

const PROTOCOL_VERSION: &str = "2024-11-05";

/// Server-side errors. IO failures and oversized inbound frames surface
/// here; protocol failures are encoded as JSON-RPC error responses to the
/// peer.
#[derive(Debug)]
pub enum ServerError {
    Io(std::io::Error),
    /// Inbound stream tried to buffer more than `wire::DEFAULT_MAX` bytes
    /// without emitting a newline. We refuse to grow the splitter past
    /// the cap and tear down the loop instead.
    OversizedFrame(wire::FramerError),
}

impl std::fmt::Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServerError::Io(e) => write!(f, "io: {e}"),
            ServerError::OversizedFrame(e) => write!(f, "oversized frame: {e}"),
        }
    }
}

impl std::error::Error for ServerError {}

impl From<std::io::Error> for ServerError {
    fn from(e: std::io::Error) -> Self {
        ServerError::Io(e)
    }
}

impl From<wire::FramerError> for ServerError {
    fn from(e: wire::FramerError) -> Self {
        ServerError::OversizedFrame(e)
    }
}

/// MCP server. Owns a tool registry plus the Instance actor handle that
/// gets passed to each tool's [`ToolCtx`].
pub struct Server {
    tools: Vec<Arc<dyn Tool>>,
    instance: ActorRef<InstanceMsg>,
    name: String,
    version: String,
}

impl Server {
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        instance: ActorRef<InstanceMsg>,
    ) -> Self {
        Self {
            tools: Vec::new(),
            instance,
            name: name.into(),
            version: version.into(),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.register(tool);
        self
    }

    /// Run on real stdin/stdout. Blocks until stdin closes.
    pub fn serve_stdio(&self) -> Result<(), ServerError> {
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        // Lock once for the lifetime of the server; we're single-client.
        let input = stdin.lock();
        let output = stdout.lock();
        self.serve(input, output)
    }

    /// Run on arbitrary IO. Reads frames from `input`, writes responses
    /// to `output`. Stops when the input reaches EOF.
    pub fn serve<R: BufRead, W: Write>(
        &self,
        mut input: R,
        mut output: W,
    ) -> Result<(), ServerError> {
        let mut splitter = NdjsonSplitter::new();
        let mut chunk = [0u8; 4096];
        loop {
            // Drain any complete lines we already have buffered.
            while let Some(line) = splitter.pop_line() {
                if line.iter().all(|b| matches!(b, b' ' | b'\t' | b'\r')) {
                    continue;
                }
                if let Some(resp) = self.handle_line(&line) {
                    output.write_all(resp.as_bytes())?;
                    output.write_all(b"\n")?;
                    output.flush()?;
                }
            }
            let n = match input.read(&mut chunk) {
                Ok(0) => return Ok(()),
                Ok(n) => n,
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(ServerError::Io(e)),
            };
            splitter.push(&chunk[..n])?;
        }
    }

    /// Handle one inbound JSON-RPC frame. Returns `Some(serialised
    /// response)` for requests, `None` for notifications and for frames
    /// that don't merit a reply (e.g. malformed notifications, since
    /// they have no id to echo).
    fn handle_line(&self, line: &[u8]) -> Option<String> {
        let v = match parse_json(line) {
            Ok(v) => v,
            Err(e) => {
                // Parse error: per JSON-RPC, the response id is null since
                // we couldn't recover it from the request.
                return Some(error_response_null(-32700, &format!("parse error: {e}")));
            }
        };
        let method = v.get("method").and_then(Json::as_str);
        let id = v.get("id").and_then(json_id_as_i64);

        match (method, id) {
            // Notification: method present, id absent (or null).
            (Some(m), None) => {
                // notifications/initialized and any other notification are
                // silently consumed.
                let _ = m;
                None
            }
            // Request: method + integer id.
            (Some(m), Some(id)) => Some(self.handle_request(m, id, &v)),
            // Response (no method, has id) or otherwise malformed: ignore
            // if we can't even build an addressable error. Servers don't
            // receive responses — the client does — so this is dead data.
            (None, Some(id)) => Some(error_response(id, -32600, "invalid request: no method")),
            (None, None) => Some(error_response_null(-32600, "invalid request")),
        }
    }

    fn handle_request(&self, method: &str, id: i64, v: &Json) -> String {
        match method {
            "initialize" => self.handle_initialize(id),
            "tools/list" => self.handle_tools_list(id),
            "tools/call" => self.handle_tools_call(id, v),
            other => error_response(id, -32601, &format!("method not found: {other}")),
        }
    }

    fn handle_initialize(&self, id: i64) -> String {
        let mut s = String::with_capacity(160 + self.name.len() + self.version.len());
        s.push_str(r#"{"jsonrpc":"2.0","id":"#);
        push_id(&mut s, id);
        s.push_str(r#","result":{"protocolVersion":""#);
        escape_into(&mut s, PROTOCOL_VERSION);
        s.push_str(r#"","capabilities":{"tools":{}},"serverInfo":{"name":""#);
        escape_into(&mut s, &self.name);
        s.push_str(r#"","version":""#);
        escape_into(&mut s, &self.version);
        s.push_str(r#""}}}"#);
        s
    }

    fn handle_tools_list(&self, id: i64) -> String {
        let mut s = String::with_capacity(64 + self.tools.len() * 96);
        s.push_str(r#"{"jsonrpc":"2.0","id":"#);
        push_id(&mut s, id);
        s.push_str(r#","result":{"tools":["#);
        for (i, t) in self.tools.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(r#"{"name":""#);
            escape_into(&mut s, t.name());
            s.push_str(r#"","description":""#);
            escape_into(&mut s, t.description());
            s.push_str(r#"","inputSchema":"#);
            // input_schema is a raw JSON object string — splice verbatim.
            s.push_str(t.input_schema());
            s.push('}');
        }
        s.push_str("]}}");
        s
    }

    fn handle_tools_call(&self, id: i64, v: &Json) -> String {
        let params = match v.get("params") {
            Some(p) => p,
            None => return error_response(id, -32602, "missing params"),
        };
        let name = match params.get("name").and_then(Json::as_str) {
            Some(n) => n,
            None => return error_response(id, -32602, "missing tool name"),
        };
        let tool = match self.tools.iter().find(|t| t.name() == name) {
            Some(t) => t,
            None => return tool_error_response(id, &format!("unknown tool: {name}")),
        };
        // Re-serialise arguments. The MCP spec says it's a JSON object;
        // if it's missing, default to {}.
        let args = match params.get("arguments") {
            Some(a) => {
                let mut buf = String::new();
                write_json(&mut buf, a);
                buf
            }
            None => "{}".to_string(),
        };
        let ctx = ToolCtx { instance: &self.instance };
        match tool.call(&args, &ctx) {
            Ok(text) => success_text_response(id, &text),
            Err(e) => tool_error_response(id, &e.0),
        }
    }
}

/// Build `{"jsonrpc":"2.0","id":N,"result":{"content":[{"type":"text","text":"..."}]}}`.
fn success_text_response(id: i64, text: &str) -> String {
    let mut s = String::with_capacity(80 + text.len());
    s.push_str(r#"{"jsonrpc":"2.0","id":"#);
    push_id(&mut s, id);
    s.push_str(r#","result":{"content":[{"type":"text","text":""#);
    escape_into(&mut s, text);
    s.push_str(r#""}]}}"#);
    s
}

/// Tool-level error: still a JSON-RPC `result` (not `error`), but with
/// `isError: true` per the MCP spec. The client surfaces it as a
/// `ToolError`.
fn tool_error_response(id: i64, msg: &str) -> String {
    let mut s = String::with_capacity(96 + msg.len());
    s.push_str(r#"{"jsonrpc":"2.0","id":"#);
    push_id(&mut s, id);
    s.push_str(r#","result":{"content":[{"type":"text","text":""#);
    escape_into(&mut s, msg);
    s.push_str(r#""}],"isError":true}}"#);
    s
}

/// Build a JSON-RPC error response with a numeric id.
fn error_response(id: i64, code: i64, message: &str) -> String {
    let mut s = String::with_capacity(64 + message.len());
    s.push_str(r#"{"jsonrpc":"2.0","id":"#);
    push_id(&mut s, id);
    s.push_str(r#","error":{"code":"#);
    s.push_str(&code.to_string());
    s.push_str(r#","message":""#);
    escape_into(&mut s, message);
    s.push_str(r#""}}"#);
    s
}

/// Same, but with `id: null` — used when we couldn't recover the id from
/// the request (parse error, completely malformed frame).
fn error_response_null(code: i64, message: &str) -> String {
    let mut s = String::with_capacity(64 + message.len());
    s.push_str(r#"{"jsonrpc":"2.0","id":null,"error":{"code":"#);
    s.push_str(&code.to_string());
    s.push_str(r#","message":""#);
    escape_into(&mut s, message);
    s.push_str(r#""}}"#);
    s
}

fn push_id(out: &mut String, id: i64) {
    out.push_str(&id.to_string());
}

/// Extract the request id as an integer. JSON-RPC permits string ids,
/// but the mcp client only emits integers and this server-side handler
/// mirrors that.
fn json_id_as_i64(v: &Json) -> Option<i64> {
    match v {
        Json::Num(n) => n.parse::<i64>().ok(),
        _ => None,
    }
}

/// Re-serialise a parsed JSON value (matches the `mcp` client's helper).
fn write_json(out: &mut String, v: &Json) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::Num(n) => out.push_str(n),
        Json::Str(s) => {
            out.push('"');
            escape_into(out, s);
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
                escape_into(out, k);
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
    use std::io::Cursor;
    use std::sync::Mutex;

    use actor::{Spawned, spawn};
    use harness::{Instance, MockSandbox, Sandbox, ShellResult, ToolError};

    // ---- test fixtures ----

    fn make_instance() -> Spawned<InstanceMsg> {
        let sb: Box<dyn Sandbox> = Box::new(MockSandbox::new(vec![ShellResult {
            exit_code: 0,
            stdout: b"hello\n".to_vec(),
            stderr: Vec::new(),
        }]));
        spawn(Instance::new("t", sb))
    }

    /// Echoes its `text` argument back. Useful for round-trip assertions.
    struct EchoTool;

    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echo the input text back."
        }
        fn input_schema(&self) -> &str {
            r#"{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}"#
        }
        fn call(&self, input: &str, _ctx: &ToolCtx) -> Result<String, ToolError> {
            let v = parse_json(input.as_bytes())
                .map_err(|e| ToolError(format!("bad json: {e}")))?;
            let t = v
                .get("text")
                .and_then(Json::as_str)
                .ok_or_else(|| ToolError("missing 'text'".into()))?;
            Ok(t.to_string())
        }
    }

    /// Always errors. Used to assert the `isError: true` path.
    struct BoomTool;

    impl Tool for BoomTool {
        fn name(&self) -> &str {
            "boom"
        }
        fn description(&self) -> &str {
            "Always fails."
        }
        fn input_schema(&self) -> &str {
            r#"{"type":"object"}"#
        }
        fn call(&self, _input: &str, _ctx: &ToolCtx) -> Result<String, ToolError> {
            Err(ToolError("kaboom".into()))
        }
    }

    /// Records the input it received. Lets us verify that arguments
    /// re-serialise faithfully across the wire boundary.
    struct RecorderTool {
        seen: Arc<Mutex<Vec<String>>>,
    }

    impl Tool for RecorderTool {
        fn name(&self) -> &str {
            "record"
        }
        fn description(&self) -> &str {
            "Record input."
        }
        fn input_schema(&self) -> &str {
            r#"{"type":"object"}"#
        }
        fn call(&self, input: &str, _ctx: &ToolCtx) -> Result<String, ToolError> {
            self.seen.lock().unwrap().push(input.to_string());
            Ok("ok".into())
        }
    }

    /// Drive `Server::serve` against an in-memory request buffer.
    /// Consumes the server so the Instance's ActorRef clone drops at the
    /// end, letting the test's `inst.join()` actually terminate.
    fn drive(server: Server, input: &str) -> String {
        let mut output = Vec::new();
        server
            .serve(Cursor::new(input.as_bytes().to_vec()), &mut output)
            .unwrap();
        String::from_utf8(output).unwrap()
    }

    /// Split `serve`'s output into one JSON value per non-empty line.
    fn split_responses(s: &str) -> Vec<Json> {
        s.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| parse_json(l.as_bytes()).unwrap())
            .collect()
    }

    // ---- initialize ----

    #[test]
    fn initialize_returns_protocol_version_capabilities_and_server_info() {
        let inst = make_instance();
        let server = Server::new("test-srv", "9.9.9", inst.addr.clone());
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"c","version":"0"}}}
"#;
        let out = drive(server, req);
        let v = parse_json(out.trim().as_bytes()).unwrap();
        assert_eq!(v.get("jsonrpc").and_then(Json::as_str), Some("2.0"));
        let id = v.get("id").unwrap();
        assert!(matches!(id, Json::Num(n) if n == "1"));
        let result = v.get("result").unwrap();
        assert_eq!(
            result.get("protocolVersion").and_then(Json::as_str),
            Some("2024-11-05")
        );
        assert!(result.get("capabilities").unwrap().get("tools").is_some());
        let info = result.get("serverInfo").unwrap();
        assert_eq!(info.get("name").and_then(Json::as_str), Some("test-srv"));
        assert_eq!(info.get("version").and_then(Json::as_str), Some("9.9.9"));
        inst.join().unwrap();
    }

    // ---- tools/list ----

    #[test]
    fn tools_list_advertises_registered_tools_with_schemas() {
        let inst = make_instance();
        let server = Server::new("s", "1", inst.addr.clone())
            .with_tool(Arc::new(harness::BashTool))
            .with_tool(Arc::new(EchoTool));
        let req = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
"#;
        let out = drive(server, req);
        let v = parse_json(out.trim().as_bytes()).unwrap();
        let tools = match v.get("result").unwrap().get("tools").unwrap() {
            Json::Arr(a) => a,
            _ => panic!("tools not array"),
        };
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].get("name").and_then(Json::as_str), Some("bash"));
        // inputSchema must be a JSON object (not a string).
        assert!(matches!(tools[0].get("inputSchema"), Some(Json::Obj(_))));
        assert!(
            tools[0]
                .get("inputSchema")
                .unwrap()
                .get("properties")
                .unwrap()
                .get("command")
                .is_some()
        );
        assert_eq!(tools[1].get("name").and_then(Json::as_str), Some("echo"));
        assert!(matches!(tools[1].get("inputSchema"), Some(Json::Obj(_))));
        inst.join().unwrap();
    }

    #[test]
    fn tools_list_with_no_tools_returns_empty_array() {
        let inst = make_instance();
        let server = Server::new("s", "1", inst.addr.clone());
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
"#;
        let out = drive(server, req);
        let v = parse_json(out.trim().as_bytes()).unwrap();
        match v.get("result").unwrap().get("tools").unwrap() {
            Json::Arr(a) => assert!(a.is_empty()),
            _ => panic!("not array"),
        }
        inst.join().unwrap();
    }

    // ---- tools/call ----

    #[test]
    fn tools_call_success_returns_text_content() {
        let inst = make_instance();
        let server = Server::new("s", "1", inst.addr.clone()).with_tool(Arc::new(EchoTool));
        let req = r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hi"}}}
"#;
        let out = drive(server, req);
        let v = parse_json(out.trim().as_bytes()).unwrap();
        assert!(matches!(v.get("id"), Some(Json::Num(n)) if n == "7"));
        let result = v.get("result").unwrap();
        // No isError on success.
        assert!(result.get("isError").is_none());
        let content = match result.get("content").unwrap() {
            Json::Arr(a) => a,
            _ => panic!(),
        };
        assert_eq!(content[0].get("type").and_then(Json::as_str), Some("text"));
        assert_eq!(content[0].get("text").and_then(Json::as_str), Some("hi"));
        inst.join().unwrap();
    }

    #[test]
    fn tools_call_error_sets_is_error_true() {
        let inst = make_instance();
        let server = Server::new("s", "1", inst.addr.clone()).with_tool(Arc::new(BoomTool));
        let req = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"boom","arguments":{}}}
"#;
        let out = drive(server, req);
        let v = parse_json(out.trim().as_bytes()).unwrap();
        let result = v.get("result").unwrap();
        assert_eq!(result.get("isError"), Some(&Json::Bool(true)));
        let content = match result.get("content").unwrap() {
            Json::Arr(a) => a,
            _ => panic!(),
        };
        assert_eq!(
            content[0].get("text").and_then(Json::as_str),
            Some("kaboom")
        );
        inst.join().unwrap();
    }

    #[test]
    fn tools_call_unknown_tool_name_returns_is_error_result() {
        let inst = make_instance();
        let server = Server::new("s", "1", inst.addr.clone()).with_tool(Arc::new(EchoTool));
        let req = r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"nope","arguments":{}}}
"#;
        let out = drive(server, req);
        let v = parse_json(out.trim().as_bytes()).unwrap();
        let result = v.get("result").unwrap();
        assert_eq!(result.get("isError"), Some(&Json::Bool(true)));
        let text = match result.get("content").unwrap() {
            Json::Arr(a) => a[0].get("text").and_then(Json::as_str).unwrap().to_string(),
            _ => panic!(),
        };
        assert!(text.contains("nope"));
        inst.join().unwrap();
    }

    #[test]
    fn tools_call_serialises_complex_arguments_back_to_tool() {
        let inst = make_instance();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let server = Server::new("s", "1", inst.addr.clone())
            .with_tool(Arc::new(RecorderTool { seen: Arc::clone(&seen) }));
        let req = r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"record","arguments":{"a":1,"b":[true,null],"c":{"k":"v"}}}}
"#;
        let _ = drive(server, req);
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        // Re-parse the recorded arguments and verify the structure was
        // preserved across the spoke/respoke round trip.
        let parsed = parse_json(seen[0].as_bytes()).unwrap();
        assert!(matches!(parsed.get("a"), Some(Json::Num(n)) if n == "1"));
        assert!(matches!(parsed.get("b"), Some(Json::Arr(_))));
        assert_eq!(
            parsed.get("c").and_then(|v| v.get("k")).and_then(Json::as_str),
            Some("v")
        );
        inst.join().unwrap();
    }

    #[test]
    fn tools_call_missing_arguments_defaults_to_empty_object() {
        let inst = make_instance();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let server = Server::new("s", "1", inst.addr.clone())
            .with_tool(Arc::new(RecorderTool { seen: Arc::clone(&seen) }));
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"record"}}
"#;
        let _ = drive(server, req);
        assert_eq!(seen.lock().unwrap()[0], "{}");
        inst.join().unwrap();
    }

    // ---- error paths ----

    #[test]
    fn method_not_found_returns_minus_32601() {
        let inst = make_instance();
        let server = Server::new("s", "1", inst.addr.clone());
        let req = r#"{"jsonrpc":"2.0","id":8,"method":"weird/method","params":{}}
"#;
        let out = drive(server, req);
        let v = parse_json(out.trim().as_bytes()).unwrap();
        let err = v.get("error").unwrap();
        assert!(matches!(err.get("code"), Some(Json::Num(n)) if n == "-32601"));
        assert!(
            err.get("message")
                .and_then(Json::as_str)
                .unwrap()
                .contains("weird/method")
        );
        assert!(v.get("result").is_none());
        inst.join().unwrap();
    }

    #[test]
    fn parse_error_returns_minus_32700_with_null_id() {
        let inst = make_instance();
        let server = Server::new("s", "1", inst.addr.clone());
        // Note: the splitter strips the trailing newline; the line itself
        // is intentionally invalid JSON.
        let req = "this is not json\n";
        let out = drive(server, req);
        let v = parse_json(out.trim().as_bytes()).unwrap();
        assert_eq!(v.get("id"), Some(&Json::Null));
        let err = v.get("error").unwrap();
        assert!(matches!(err.get("code"), Some(Json::Num(n)) if n == "-32700"));
        inst.join().unwrap();
    }

    // ---- notifications ----

    #[test]
    fn notifications_initialized_produces_no_response() {
        let inst = make_instance();
        let server = Server::new("s", "1", inst.addr.clone());
        let req = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}
"#;
        let out = drive(server, req);
        assert!(out.is_empty(), "expected no output, got {out:?}");
        inst.join().unwrap();
    }

    #[test]
    fn arbitrary_notification_is_silently_consumed() {
        let inst = make_instance();
        let server = Server::new("s", "1", inst.addr.clone());
        let req = r#"{"jsonrpc":"2.0","method":"notifications/something","params":{"x":1}}
"#;
        let out = drive(server, req);
        assert!(out.is_empty());
        inst.join().unwrap();
    }

    // ---- multi-request / id handling ----

    #[test]
    fn multiple_requests_each_get_their_own_response() {
        let inst = make_instance();
        let server = Server::new("s", "1", inst.addr.clone()).with_tool(Arc::new(EchoTool));
        let req = concat!(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#, "\n",
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#, "\n",
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#, "\n",
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":{"text":"yo"}}}"#, "\n",
        );
        let out = drive(server, req);
        let resps = split_responses(&out);
        // 3 requests → 3 responses, notification swallowed.
        assert_eq!(resps.len(), 3);
        assert!(matches!(resps[0].get("id"), Some(Json::Num(n)) if n == "1"));
        assert!(matches!(resps[1].get("id"), Some(Json::Num(n)) if n == "2"));
        assert!(matches!(resps[2].get("id"), Some(Json::Num(n)) if n == "3"));
        assert!(
            resps[2]
                .get("result")
                .and_then(|r| r.get("content"))
                .map(|c| matches!(c, Json::Arr(_)))
                .unwrap_or(false)
        );
        inst.join().unwrap();
    }

    #[test]
    fn out_of_order_ids_are_echoed_verbatim() {
        let inst = make_instance();
        let server = Server::new("s", "1", inst.addr.clone());
        let req = concat!(
            r#"{"jsonrpc":"2.0","id":42,"method":"tools/list","params":{}}"#, "\n",
            r#"{"jsonrpc":"2.0","id":7,"method":"tools/list","params":{}}"#, "\n",
        );
        let out = drive(server, req);
        let resps = split_responses(&out);
        assert!(matches!(resps[0].get("id"), Some(Json::Num(n)) if n == "42"));
        assert!(matches!(resps[1].get("id"), Some(Json::Num(n)) if n == "7"));
        inst.join().unwrap();
    }

    #[test]
    fn blank_lines_between_requests_are_skipped() {
        let inst = make_instance();
        let server = Server::new("s", "1", inst.addr.clone());
        let req = "\n   \n\r\n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\",\"params\":{}}\n\n";
        let out = drive(server, req);
        let resps = split_responses(&out);
        assert_eq!(resps.len(), 1);
        assert!(matches!(resps[0].get("id"), Some(Json::Num(n)) if n == "1"));
        inst.join().unwrap();
    }

    // ---- end-to-end via BashTool through Instance ----

    #[test]
    fn bash_tool_dispatch_runs_through_instance_actor() {
        // Drives a `tools/call` for the real BashTool — proves the
        // ToolCtx wiring reaches the Sandbox.
        let inst = make_instance();
        let server =
            Server::new("s", "1", inst.addr.clone()).with_tool(Arc::new(harness::BashTool));
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"bash","arguments":{"command":"echo hello"}}}
"#;
        let out = drive(server, req);
        let v = parse_json(out.trim().as_bytes()).unwrap();
        let content = match v.get("result").unwrap().get("content").unwrap() {
            Json::Arr(a) => a,
            _ => panic!(),
        };
        let text = content[0].get("text").and_then(Json::as_str).unwrap();
        assert!(text.contains("hello"));
        inst.join().unwrap();
    }

    // ---- compatibility round-trip with the mcp client's framing ----

    #[test]
    fn responses_match_mcp_client_initialize_shape() {
        // Hard-coded assertion that the on-wire bytes match what the mcp
        // client expects to parse (it checks for protocolVersion).
        let inst = make_instance();
        let server = Server::new("srv", "0.1", inst.addr.clone());
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
"#;
        let out = drive(server, req);
        let line = out.lines().next().unwrap();
        assert!(line.starts_with(r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05""#));
        assert!(line.contains(r#""capabilities":{"tools":{}}"#));
        assert!(line.contains(r#""serverInfo":{"name":"srv","version":"0.1"}"#));
        inst.join().unwrap();
    }
}
