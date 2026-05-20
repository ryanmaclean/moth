//! Model trait + supporting types. `Model::stream` returns a boxed iterator
//! because providers differ in how they buffer/parse a stream — letting each
//! adapter pick its own iterator type is worth the one heap alloc per call.

use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// One block in a `ChatMessage::content` list. Mirrors Anthropic's content
/// block shape so the adapter can pass blocks through with minimal mapping.
/// `ToolUse::input` is a raw JSON value (e.g. `{"command":"ls"}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentBlock {
    Text(String),
    ToolUse { id: String, name: String, input: String },
    ToolResult { tool_use_id: String, content: String, is_error: bool },
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl ChatMessage {
    pub fn user(text: impl Into<String>) -> Self {
        Self { role: Role::User, content: vec![ContentBlock::Text(text.into())] }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: vec![ContentBlock::Text(text.into())] }
    }
}

/// Tool definition exposed to the model. `input_schema` is a raw JSON object
/// (not a JSON string); it's spliced into the request payload verbatim.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: String,
}

#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub system: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: u32,
    pub tools: Vec<ToolDef>,
}

#[derive(Debug, Clone)]
pub enum ModelEvent {
    TextDelta(String),
    ToolUseStart { id: String, name: String },
    ToolUseInputDelta(String),
    BlockStop,
    Stop { reason: Option<String> },
}

#[derive(Debug, Clone)]
pub struct ModelError(pub String);

pub trait Model: Send + Sync + 'static {
    fn stream(
        &self,
        req: ModelRequest,
    ) -> Box<dyn Iterator<Item = Result<ModelEvent, ModelError>> + Send>;
}

/// Mock model. Each call to `stream` drains the next batch from `scripts`
/// (one Vec per expected turn), letting tests assert multi-turn behaviour.
pub struct MockModel {
    scripts: Mutex<Vec<Vec<ModelEvent>>>,
    pub seen: Mutex<Vec<ModelRequest>>,
}

impl MockModel {
    pub fn new(scripts: Vec<Vec<ModelEvent>>) -> Self {
        Self { scripts: Mutex::new(scripts), seen: Mutex::new(Vec::new()) }
    }

    pub fn single(events: Vec<ModelEvent>) -> Self {
        Self::new(vec![events])
    }
}

impl Model for MockModel {
    fn stream(
        &self,
        req: ModelRequest,
    ) -> Box<dyn Iterator<Item = Result<ModelEvent, ModelError>> + Send> {
        self.seen.lock().unwrap().push(req);
        let mut scripts = self.scripts.lock().unwrap();
        let next = if scripts.is_empty() { Vec::new() } else { scripts.remove(0) };
        Box::new(next.into_iter().map(Ok))
    }
}
