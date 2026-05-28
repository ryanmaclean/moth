//! Model trait + supporting types. `Model::stream` returns a boxed iterator
//! because providers differ in how they buffer/parse a stream — letting each
//! adapter pick its own iterator type is worth the one heap alloc per call.

use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// One block in a `ChatMessage::content` list. Mirrors Anthropic's content
/// block shape so the adapter can pass blocks through with minimal mapping.
/// `ToolUse::input` is a raw JSON value (e.g. `{"command":"ls"}`).
///
/// Payloads are `Arc<str>` so cloning a `ChatMessage` is one atomic refcount
/// bump per block instead of a deep String copy. The Session clones its
/// history into every per-turn `ModelRequest`; over a long session that's a
/// lot of bytes saved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentBlock {
    Text(Arc<str>),
    ToolUse { id: Arc<str>, name: Arc<str>, input: Arc<str> },
    ToolResult { tool_use_id: Arc<str>, content: Arc<str>, is_error: bool },
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl ChatMessage {
    pub fn user(text: impl Into<Arc<str>>) -> Self {
        Self { role: Role::User, content: vec![ContentBlock::Text(text.into())] }
    }

    pub fn assistant(text: impl Into<Arc<str>>) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Cloning a `Text` block must not deep-copy the payload — the cloned
    /// block's `Arc` must point at the same allocation as the original.
    #[test]
    fn cloning_text_block_bumps_refcount() {
        let m = ChatMessage::user("hello there");
        let inner = match &m.content[0] {
            ContentBlock::Text(s) => s.clone(),
            _ => panic!("expected text"),
        };
        // m holds 1 ref + we just cloned it = 2.
        assert_eq!(Arc::strong_count(&inner), 2);

        // Cloning the message itself must produce a third reference, not a
        // fresh allocation.
        let clone = m.clone();
        assert_eq!(Arc::strong_count(&inner), 3);

        // Same pointer.
        if let ContentBlock::Text(s) = &clone.content[0] {
            assert!(Arc::ptr_eq(&inner, s));
        } else {
            panic!("expected text after clone");
        }
    }

    #[test]
    fn cloning_tool_use_block_bumps_each_field_refcount() {
        let block = ContentBlock::ToolUse {
            id: "toolu_1".into(),
            name: "bash".into(),
            input: r#"{"command":"ls"}"#.into(),
        };
        let (id0, name0, input0) = match &block {
            ContentBlock::ToolUse { id, name, input } => (id.clone(), name.clone(), input.clone()),
            _ => panic!(),
        };
        // After our clones above: block holds 1 + we hold 1 each = 2.
        assert_eq!(Arc::strong_count(&id0), 2);
        assert_eq!(Arc::strong_count(&name0), 2);
        assert_eq!(Arc::strong_count(&input0), 2);

        let copies: Vec<_> = (0..5).map(|_| block.clone()).collect();
        // 2 (original + our captured) + 5 fresh clones = 7.
        assert_eq!(Arc::strong_count(&id0), 7);
        assert_eq!(Arc::strong_count(&name0), 7);
        assert_eq!(Arc::strong_count(&input0), 7);
        drop(copies);
        assert_eq!(Arc::strong_count(&id0), 2);
    }

    #[test]
    fn cloning_history_vec_shares_underlying_arcs() {
        // The hot path: cloning the entire history Vec for ModelRequest.
        // The clones of each ChatMessage's ContentBlocks must share Arcs.
        let history = vec![
            ChatMessage::user("first user prompt"),
            ChatMessage {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Text("ok".into()),
                    ContentBlock::ToolUse {
                        id: "t1".into(),
                        name: "bash".into(),
                        input: r#"{"x":1}"#.into(),
                    },
                ],
            },
        ];
        let text_arc = match &history[0].content[0] {
            ContentBlock::Text(s) => s.clone(),
            _ => panic!(),
        };
        let tool_id_arc = match &history[1].content[1] {
            ContentBlock::ToolUse { id, .. } => id.clone(),
            _ => panic!(),
        };
        // history holds 1 + we hold 1 = 2 each.
        assert_eq!(Arc::strong_count(&text_arc), 2);
        assert_eq!(Arc::strong_count(&tool_id_arc), 2);

        // Clone the entire history N times. Each clone must bump every
        // payload's refcount by exactly 1 — no String copies.
        const N: usize = 10;
        let clones: Vec<_> = (0..N).map(|_| history.clone()).collect();
        assert_eq!(Arc::strong_count(&text_arc), 2 + N);
        assert_eq!(Arc::strong_count(&tool_id_arc), 2 + N);
        drop(clones);
        assert_eq!(Arc::strong_count(&text_arc), 2);
        assert_eq!(Arc::strong_count(&tool_id_arc), 2);
    }

    #[test]
    fn builders_accept_string_and_str() {
        // Both &str and String should still work as input to the builders.
        let from_str = ChatMessage::user("a literal");
        let from_owned = ChatMessage::assistant(String::from("an owned String"));
        match &from_str.content[0] {
            ContentBlock::Text(s) => assert_eq!(&**s, "a literal"),
            _ => panic!(),
        }
        match &from_owned.content[0] {
            ContentBlock::Text(s) => assert_eq!(&**s, "an owned String"),
            _ => panic!(),
        }
    }
}
