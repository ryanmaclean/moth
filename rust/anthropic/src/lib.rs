//! Streaming client for the Anthropic Messages API.
//!
//! `Client::stream` issues a POST, streams the response through `wire::SseFramer`,
//! and yields parsed `Event` values via an `Iterator`. Tools are advertised by
//! raw JSON schema — the caller chose to skip serde, so we don't re-parse.
//!
//! libcurl + OpenSSL via FFI is the only network dep. No async runtime.

mod http;
mod json;
mod parse;

use std::sync::mpsc::TryRecvError;

use wire::SseFramer;

use http::{Chunk, Stream, post_stream};

pub struct Client {
    api_key: String,
}

impl Client {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }

    pub fn stream(&self, req: Request) -> Result<EventStream, Error> {
        let body = serialize_request(&req);
        let stream = post_stream(&self.api_key, body)?;
        Ok(EventStream {
            stream,
            framer: SseFramer::new(),
            done: false,
            terminal: None,
        })
    }
}

pub struct Request {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<Message>,
    pub system: Option<String>,
    pub tools: Vec<Tool>,
}

pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Clone, Copy)]
pub enum Role {
    User,
    Assistant,
}

pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: String,
}

#[derive(Debug, PartialEq)]
pub enum Event {
    MessageStart { id: String, model: String },
    TextDelta(String),
    ToolUseStart { id: String, name: String },
    ToolUseInputDelta(String),
    ContentBlockStop,
    MessageDelta { stop_reason: Option<String> },
    MessageStop,
    Ping,
    Other(String),
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
    done: bool,
    terminal: Option<Error>,
}

impl Iterator for EventStream {
    type Item = Result<Event, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(frame) = self.framer.pop_frame() {
                return Some(parse::parse_frame(&frame));
            }
            if self.done {
                return self.terminal.take().map(Err);
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
            if let Some(frame) = self.framer.pop_frame() {
                return Some(parse::parse_frame(&frame));
            }
            if self.done {
                return self.terminal.take().map(Err);
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

fn serialize_request(req: &Request) -> String {
    let mut s = String::with_capacity(256);
    s.push('{');
    s.push_str("\"model\":\"");
    json::escape_into(&mut s, &req.model);
    s.push_str("\",\"max_tokens\":");
    s.push_str(&req.max_tokens.to_string());
    s.push_str(",\"stream\":true,\"messages\":[");
    for (i, m) in req.messages.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("{\"role\":\"");
        s.push_str(match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        });
        s.push_str("\",\"content\":\"");
        json::escape_into(&mut s, &m.content);
        s.push_str("\"}");
    }
    s.push(']');
    if let Some(sys) = &req.system {
        s.push_str(",\"system\":\"");
        json::escape_into(&mut s, sys);
        s.push('"');
    }
    if !req.tools.is_empty() {
        s.push_str(",\"tools\":[");
        for (i, t) in req.tools.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str("{\"name\":\"");
            json::escape_into(&mut s, &t.name);
            s.push_str("\",\"description\":\"");
            json::escape_into(&mut s, &t.description);
            // input_schema is a raw JSON value, not a JSON string.
            s.push_str("\",\"input_schema\":");
            s.push_str(&t.input_schema);
            s.push('}');
        }
        s.push(']');
    }
    s.push('}');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_minimal() {
        let req = Request {
            model: "claude-opus-4-7".into(),
            max_tokens: 64,
            messages: vec![Message { role: Role::User, content: "hi".into() }],
            system: None,
            tools: vec![],
        };
        let body = serialize_request(&req);
        assert_eq!(
            body,
            r#"{"model":"claude-opus-4-7","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#
        );
    }

    #[test]
    fn serialize_with_system_and_tools() {
        let req = Request {
            model: "claude-opus-4-7".into(),
            max_tokens: 1,
            messages: vec![Message {
                role: Role::Assistant,
                content: "ok\nthen".into(),
            }],
            system: Some("be terse".into()),
            tools: vec![Tool {
                name: "get_weather".into(),
                description: "weather lookup".into(),
                input_schema: r#"{"type":"object","properties":{}}"#.into(),
            }],
        };
        let body = serialize_request(&req);
        assert!(body.contains(r#""system":"be terse""#));
        assert!(body.contains(r#""role":"assistant""#));
        assert!(body.contains(r#""content":"ok\nthen""#));
        assert!(body.contains(r#""tools":[{"name":"get_weather""#));
        assert!(body.contains(r#""input_schema":{"type":"object""#));
    }

    #[test]
    fn serialize_escapes_dangerous_chars() {
        let req = Request {
            model: "m".into(),
            max_tokens: 1,
            messages: vec![Message {
                role: Role::User,
                content: "a\"b\\c".into(),
            }],
            system: None,
            tools: vec![],
        };
        let body = serialize_request(&req);
        assert!(body.contains(r#""content":"a\"b\\c""#));
    }
}
