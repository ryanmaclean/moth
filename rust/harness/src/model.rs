//! Model trait + supporting types. `Model::stream` returns a boxed iterator
//! because providers differ in how they buffer/parse a stream — letting each
//! adapter pick its own iterator type is worth the one heap alloc per call.

use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub system: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: u32,
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
