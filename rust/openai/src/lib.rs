//! Streaming client for OpenAI-compatible chat-completions endpoints.
//!
//! Mirrors the structural shape of the `anthropic` crate: `Client::stream`
//! issues a POST, frames the response through `wire::SseFramer`, and yields
//! parsed `Event` values via an `Iterator`. Tools are advertised by raw JSON
//! schema so the caller can stay serde-free.
//!
//! Works with any endpoint that speaks the OpenAI chat-completions wire
//! format: OpenAI itself, OpenRouter, LM Studio, Ollama's compat shim, etc.
//! Set the base URL via `Client::with_base_url`; default is
//! `https://api.openai.com`.
//!
//! ## JSON parser
//!
//! `json.rs` is a verbatim duplicate of `anthropic::json` with one extra
//! helper (`Json::as_arr`). The protocol shapes differ enough that pulling
//! the anthropic crate in as a dep just to share ~150 LOC of JSON would
//! couple two otherwise independent provider crates — the cost of keeping
//! the duplicate is lower than the cost of that coupling. If a third
//! provider lands we'll hoist this into `wire::json` instead.
//!
//! libcurl + OpenSSL via FFI is the only network dep. No async runtime.

mod http;
pub mod json;
mod parse;

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::mpsc::TryRecvError;

use wire::SseFramer;

use http::{Chunk, Stream, post_stream};
use parse::{Parsed, parse_frame};

pub struct Client {
    api_key: String,
    base_url: String,
}

impl Client {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: "https://api.openai.com".into(),
        }
    }

    /// Override the API host. Useful for OpenRouter, LM Studio, Ollama, etc.
    /// The path (`/v1/chat/completions`) is appended automatically.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn stream(&self, req: Request) -> Result<EventStream, Error> {
        let body = serialize_request(&req);
        let stream = post_stream(&self.api_key, &self.base_url, body)?;
        Ok(EventStream {
            stream,
            framer: SseFramer::new(),
            pending: VecDeque::new(),
            done: false,
            terminated: false,
            terminal: None,
        })
    }
}

pub struct Request {
    pub model: String,
    /// Omitted from the wire payload when `0`.
    pub max_tokens: u32,
    pub messages: Vec<Message>,
    /// Emitted as a leading `{role:"system",content:...}` message.
    pub system: Option<String>,
    pub tools: Vec<Tool>,
}

pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user_text(s: impl Into<Arc<str>>) -> Self {
        Self { role: Role::User, content: vec![ContentBlock::Text(s.into())] }
    }

    pub fn assistant_text(s: impl Into<Arc<str>>) -> Self {
        Self { role: Role::Assistant, content: vec![ContentBlock::Text(s.into())] }
    }
}

/// `ContentBlock::ToolUse::input` is a raw JSON value (typically the args
/// JSON the model produced) and is spliced into the payload verbatim — the
/// caller is responsible for it being valid JSON.
///
/// Payloads are `Arc<str>` so callers can clone a request's `Vec<Message>`
/// cheaply — each block clone is an atomic refcount bump rather than a deep
/// String copy.
pub enum ContentBlock {
    Text(Arc<str>),
    ToolUse { id: Arc<str>, name: Arc<str>, input: Arc<str> },
    ToolResult { tool_use_id: Arc<str>, content: Arc<str> },
}

#[derive(Clone, Copy)]
pub enum Role {
    User,
    Assistant,
    /// Wire-format role for tool-result messages. The serializer also emits
    /// this automatically when an otherwise-user message contains only
    /// `ToolResult` blocks.
    Tool,
}

#[derive(Debug, PartialEq)]
pub enum Event {
    TextDelta(String),
    ToolUseStart { id: String, name: String },
    ToolUseInputDelta(String),
    ContentBlockStop,
    /// `reason` is canonicalised to the anthropic vocabulary where possible
    /// (`stop`→`end_turn`, `tool_calls`→`tool_use`) so a caller driving both
    /// providers can share the loop termination logic.
    Stop { reason: Option<String> },
    Other(String),
}

pub struct Tool {
    pub name: String,
    pub description: String,
    /// Raw JSON Schema for `function.parameters`. Spliced verbatim.
    pub input_schema: String,
}

#[derive(Debug)]
pub enum Error {
    Http(String),
    InvalidResponse(String),
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Http(m) => write!(f, "http: {m}"),
            Error::InvalidResponse(m) => write!(f, "invalid response: {m}"),
            Error::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub struct EventStream {
    stream: Stream,
    framer: SseFramer,
    /// One SSE frame can yield several semantic events (text + tool-call +
    /// finish_reason all in the same `delta`); we buffer here between
    /// `Iterator::next` calls.
    pending: VecDeque<Event>,
    done: bool,
    terminated: bool,
    terminal: Option<Error>,
}

impl Iterator for EventStream {
    type Item = Result<Event, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ev) = self.pending.pop_front() {
                return Some(Ok(ev));
            }
            if self.terminated {
                return self.terminal.take().map(Err);
            }
            if let Some(frame) = self.framer.pop_frame() {
                match parse_frame(&frame) {
                    Ok(Parsed::Events(evs)) => self.pending.extend(evs),
                    Ok(Parsed::Terminate) => {
                        self.terminated = true;
                    }
                    Err(e) => return Some(Err(e)),
                }
                continue;
            }
            if self.done {
                self.terminated = true;
                continue;
            }
            match self.stream.rx.recv() {
                Ok(Chunk::Data(b)) => self.framer.push(&b),
                Ok(Chunk::End(Ok(_))) => {
                    self.done = true;
                }
                Ok(Chunk::End(Err(e))) => {
                    self.done = true;
                    self.terminal = Some(e);
                }
                Err(_) => {
                    self.done = true;
                }
            }
        }
    }
}

impl EventStream {
    /// Non-blocking peek. Returns `None` if there's no event yet but the
    /// stream is still open.
    pub fn try_next(&mut self) -> Option<Result<Event, Error>> {
        loop {
            if let Some(ev) = self.pending.pop_front() {
                return Some(Ok(ev));
            }
            if self.terminated {
                return self.terminal.take().map(Err);
            }
            if let Some(frame) = self.framer.pop_frame() {
                match parse_frame(&frame) {
                    Ok(Parsed::Events(evs)) => self.pending.extend(evs),
                    Ok(Parsed::Terminate) => {
                        self.terminated = true;
                    }
                    Err(e) => return Some(Err(e)),
                }
                continue;
            }
            if self.done {
                self.terminated = true;
                continue;
            }
            match self.stream.rx.try_recv() {
                Ok(Chunk::Data(b)) => self.framer.push(&b),
                Ok(Chunk::End(Ok(_))) => {
                    self.done = true;
                }
                Ok(Chunk::End(Err(e))) => {
                    self.done = true;
                    self.terminal = Some(e);
                }
                Err(TryRecvError::Empty) => return None,
                Err(TryRecvError::Disconnected) => {
                    self.done = true;
                }
            }
        }
    }
}

// ---------- request serialization ----------
//
// The OpenAI wire shape is flatter than Anthropic's:
//
//   * `system` is a leading message with role "system", not a top-level field.
//   * An assistant message's `ToolUse` blocks become a `tool_calls` array on
//     the same message; the message's `content` is `null` when the assistant
//     emitted only tool calls.
//   * A user-side `ToolResult` block becomes its own `{role:"tool",
//     tool_call_id, content}` message.
//
// So we walk the user-facing `Vec<Message>` and may split each message into
// one or more wire-level messages.

fn serialize_request(req: &Request) -> String {
    let mut s = String::with_capacity(256);
    s.push('{');
    s.push_str("\"model\":\"");
    json::escape_into(&mut s, &req.model);
    s.push_str("\",\"stream\":true");

    if req.max_tokens > 0 {
        s.push_str(",\"max_tokens\":");
        s.push_str(&req.max_tokens.to_string());
    }

    s.push_str(",\"messages\":[");
    let mut first = true;
    if let Some(sys) = &req.system {
        push_system(&mut s, sys);
        first = false;
    }
    for m in &req.messages {
        write_message(&mut s, m, &mut first);
    }
    s.push(']');

    if !req.tools.is_empty() {
        s.push_str(",\"tools\":[");
        for (i, t) in req.tools.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            write_tool(&mut s, t);
        }
        s.push(']');
    }

    s.push('}');
    s
}

fn push_system(s: &mut String, sys: &str) {
    s.push_str("{\"role\":\"system\",\"content\":\"");
    json::escape_into(s, sys);
    s.push_str("\"}");
}

fn write_tool(s: &mut String, t: &Tool) {
    s.push_str("{\"type\":\"function\",\"function\":{\"name\":\"");
    json::escape_into(s, &t.name);
    s.push_str("\",\"description\":\"");
    json::escape_into(s, &t.description);
    // input_schema is a raw JSON value, not a string.
    s.push_str("\",\"parameters\":");
    s.push_str(&t.input_schema);
    s.push_str("}}");
}

fn write_message(s: &mut String, m: &Message, first: &mut bool) {
    // A single user-side message containing only ToolResult blocks expands
    // into N separate `{role:"tool"}` messages — one per result.
    if matches!(m.role, Role::User | Role::Tool)
        && m.content.iter().all(|b| matches!(b, ContentBlock::ToolResult { .. }))
        && !m.content.is_empty()
    {
        for block in &m.content {
            if let ContentBlock::ToolResult { tool_use_id, content } = block {
                if !*first {
                    s.push(',');
                }
                *first = false;
                s.push_str("{\"role\":\"tool\",\"tool_call_id\":\"");
                json::escape_into(s, tool_use_id);
                s.push_str("\",\"content\":\"");
                json::escape_into(s, content);
                s.push_str("\"}");
            }
        }
        return;
    }

    if !*first {
        s.push(',');
    }
    *first = false;

    let role_str = match m.role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    s.push_str("{\"role\":\"");
    s.push_str(role_str);
    s.push('"');

    // Partition into text/tool_use. Mixed user messages (text + ToolResult)
    // are unusual but technically expressible — we serialize text into
    // `content` and silently drop ToolResult here; well-formed callers
    // shouldn't mix.
    let texts: Vec<&Arc<str>> = m
        .content
        .iter()
        .filter_map(|b| if let ContentBlock::Text(t) = b { Some(t) } else { None })
        .collect();
    let tool_calls: Vec<(&Arc<str>, &Arc<str>, &Arc<str>)> = m
        .content
        .iter()
        .filter_map(|b| {
            if let ContentBlock::ToolUse { id, name, input } = b {
                Some((id, name, input))
            } else {
                None
            }
        })
        .collect();

    s.push_str(",\"content\":");
    match texts.len() {
        0 => {
            // Either pure tool_calls assistant message, or no content at all.
            // Both render as null on the wire.
            s.push_str("null");
        }
        1 => {
            // Single text block (with or without accompanying tool_calls):
            // send as a plain string, matching what most callers see in
            // examples.
            s.push('"');
            json::escape_into(s, texts[0]);
            s.push('"');
        }
        _ => {
            // Multiple text blocks: send as an array of typed content parts.
            // OpenAI accepts this for assistant + user roles.
            s.push('[');
            for (i, t) in texts.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str("{\"type\":\"text\",\"text\":\"");
                json::escape_into(s, t);
                s.push_str("\"}");
            }
            s.push(']');
        }
    }

    if !tool_calls.is_empty() {
        s.push_str(",\"tool_calls\":[");
        for (i, (id, name, input)) in tool_calls.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str("{\"id\":\"");
            json::escape_into(s, id);
            s.push_str("\",\"type\":\"function\",\"function\":{\"name\":\"");
            json::escape_into(s, name);
            // `arguments` is a JSON-encoded string on the wire — the model's
            // arguments JSON object, stringified. Our `input` is the raw JSON
            // value (e.g. `{"command":"ls"}`), so we have to escape it.
            s.push_str("\",\"arguments\":\"");
            json::escape_into(s, input);
            s.push_str("\"}}");
        }
        s.push(']');
    }

    s.push('}');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_minimal_user_message() {
        let req = Request {
            model: "gpt-4o-mini".into(),
            max_tokens: 0,
            messages: vec![Message::user_text("hi")],
            system: None,
            tools: vec![],
        };
        let body = serialize_request(&req);
        assert_eq!(
            body,
            r#"{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"hi"}]}"#
        );
    }

    #[test]
    fn serialize_max_tokens_only_when_nonzero() {
        let req = Request {
            model: "m".into(),
            max_tokens: 64,
            messages: vec![Message::user_text("hi")],
            system: None,
            tools: vec![],
        };
        assert!(serialize_request(&req).contains(r#""max_tokens":64"#));
    }

    #[test]
    fn serialize_system_becomes_leading_message() {
        let req = Request {
            model: "m".into(),
            max_tokens: 0,
            messages: vec![Message::user_text("hello")],
            system: Some("be terse".into()),
            tools: vec![],
        };
        let body = serialize_request(&req);
        assert!(body.contains(
            r#""messages":[{"role":"system","content":"be terse"},{"role":"user","content":"hello"}]"#
        ));
    }

    #[test]
    fn serialize_assistant_with_tool_use() {
        let req = Request {
            model: "m".into(),
            max_tokens: 0,
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "bash".into(),
                    input: r#"{"command":"ls"}"#.into(),
                }],
            }],
            system: None,
            tools: vec![],
        };
        let body = serialize_request(&req);
        assert!(body.contains(
            r#"{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]}"#
        ));
    }

    #[test]
    fn serialize_assistant_with_text_and_tool_use() {
        let req = Request {
            model: "m".into(),
            max_tokens: 0,
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Text("looking…".into()),
                    ContentBlock::ToolUse {
                        id: "call_1".into(),
                        name: "bash".into(),
                        input: r#"{"command":"ls"}"#.into(),
                    },
                ],
            }],
            system: None,
            tools: vec![],
        };
        let body = serialize_request(&req);
        assert!(body.contains(r#""role":"assistant","content":"looking…""#));
        assert!(body.contains(r#""tool_calls":[{"id":"call_1""#));
    }

    #[test]
    fn serialize_tool_result_becomes_tool_role_message() {
        let req = Request {
            model: "m".into(),
            max_tokens: 0,
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: "file1\nfile2\n".into(),
                }],
            }],
            system: None,
            tools: vec![],
        };
        let body = serialize_request(&req);
        assert!(body.contains(
            r#"{"role":"tool","tool_call_id":"call_1","content":"file1\nfile2\n"}"#
        ));
        // No user wrapper, no `tool_calls` field.
        assert!(!body.contains(r#""role":"user""#));
    }

    #[test]
    fn serialize_multiple_tool_results_split_into_multiple_messages() {
        let req = Request {
            model: "m".into(),
            max_tokens: 0,
            messages: vec![Message {
                role: Role::User,
                content: vec![
                    ContentBlock::ToolResult {
                        tool_use_id: "call_1".into(),
                        content: "a".into(),
                    },
                    ContentBlock::ToolResult {
                        tool_use_id: "call_2".into(),
                        content: "b".into(),
                    },
                ],
            }],
            system: None,
            tools: vec![],
        };
        let body = serialize_request(&req);
        assert!(body.contains(
            r#"{"role":"tool","tool_call_id":"call_1","content":"a"},{"role":"tool","tool_call_id":"call_2","content":"b"}"#
        ));
    }

    #[test]
    fn serialize_multiple_tools_registered() {
        let req = Request {
            model: "m".into(),
            max_tokens: 0,
            messages: vec![Message::user_text("hi")],
            system: None,
            tools: vec![
                Tool {
                    name: "bash".into(),
                    description: "run a shell command".into(),
                    input_schema: r#"{"type":"object","properties":{"command":{"type":"string"}}}"#
                        .into(),
                },
                Tool {
                    name: "read_file".into(),
                    description: "read a file".into(),
                    input_schema: r#"{"type":"object"}"#.into(),
                },
            ],
        };
        let body = serialize_request(&req);
        assert!(body.contains(
            r#""tools":[{"type":"function","function":{"name":"bash","description":"run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}},{"type":"function","function":{"name":"read_file","description":"read a file","parameters":{"type":"object"}}}]"#
        ));
    }

    #[test]
    fn serialize_escapes_dangerous_chars() {
        let req = Request {
            model: "m".into(),
            max_tokens: 0,
            messages: vec![Message::user_text("a\"b\\c\n")],
            system: None,
            tools: vec![],
        };
        let body = serialize_request(&req);
        assert!(body.contains(r#""content":"a\"b\\c\n""#));
    }

    #[test]
    fn client_default_base_url() {
        let c = Client::new("k".into());
        assert_eq!(c.base_url, "https://api.openai.com");
    }

    #[test]
    fn client_with_base_url() {
        let c = Client::new("k".into()).with_base_url("http://localhost:1234");
        assert_eq!(c.base_url, "http://localhost:1234");
    }
}
