//! Cost of `messages: self.history.clone()` from `Session::run_prompt_inner`.
//!
//! Constructs a 100-turn history with mixed text + tool blocks (the kind a
//! real agentic session accumulates) and times one full `Vec<ChatMessage>`
//! clone — the operation the session performs per turn to build each
//! `ModelRequest`. Before the `Arc<str>` refactor every block clone copied
//! its String contents; after it's an atomic refcount bump per block.

use harness::{ChatMessage, ContentBlock, Role};

use crate::bench_helper::bench;

fn build_history(turns: usize) -> Vec<ChatMessage> {
    // Each "turn" is three messages: a user prompt, an assistant turn with
    // text + a tool_use, and a user turn with a tool_result. The strings are
    // sized to resemble a real tool-using transcript (a couple hundred
    // characters per content block).
    let mut h = Vec::with_capacity(turns * 3);
    let text = "x".repeat(200);
    let tool_input = r#"{"command":"echo hello world && ls -la /tmp/foo"}"#;
    let tool_output = "stdout: ".to_string() + &"y".repeat(400);
    for i in 0..turns {
        h.push(ChatMessage::user(format!("user prompt {i}: {text}")));
        h.push(ChatMessage {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text(format!("thinking about {i}: {text}").into()),
                ContentBlock::ToolUse {
                    id: format!("toolu_{i:04}").into(),
                    name: "bash".into(),
                    input: tool_input.into(),
                },
            ],
        });
        h.push(ChatMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: format!("toolu_{i:04}").into(),
                content: tool_output.clone().into(),
                is_error: false,
            }],
        });
    }
    h
}

#[test]
fn history_clone_100_turns() {
    let history = build_history(100);
    bench("history.clone 100 turns (300 messages)", || {
        let cloned = std::hint::black_box(history.clone());
        std::hint::black_box(cloned);
    });
}

#[test]
fn history_clone_50_turns() {
    let history = build_history(50);
    bench("history.clone 50 turns (150 messages)", || {
        let cloned = std::hint::black_box(history.clone());
        std::hint::black_box(cloned);
    });
}
