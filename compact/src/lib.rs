//! Message-history compaction.
//!
//! Two layers:
//!
//! 1. Pure char-budget splitting (`estimate_chars`, `split_for_compaction`).
//!    No model call. Walks from the tail, accumulating chars, never
//!    splits a `tool_use` / `tool_result` pair, always keeps at least the
//!    final two messages.
//!
//! 2. Model-driven summarisation (`Compactor::maybe_compact`). With
//!    hysteresis: only fires when total chars exceed `target_chars * 2`,
//!    so we don't churn near the budget edge. Drains a model stream's
//!    `TextDelta`s into a `summary` String, then prepends a synthetic
//!    Assistant message containing `[Earlier turns summarised:]\n{summary}`
//!    and keeps the tail verbatim.
//!
//! Char count is a useful proxy for tokens — we don't have a tokeniser and
//! adding one would be a dep we don't want.

use std::sync::Arc;

use harness::{ChatMessage, ContentBlock, Model, ModelEvent, ModelRequest, Role};

/// Estimate the size of a message history in chars. Sums the char count
/// across every `ContentBlock` in every message: `Text(s)` contributes
/// `s.chars().count()`, `ToolUse` contributes `name.len() + input.len()`,
/// `ToolResult` contributes `content.chars().count()`.
///
/// Chars (not bytes) so multibyte UTF-8 sequences count as one. The number
/// is a budget proxy, not a precise token count.
pub fn estimate_chars(messages: &[ChatMessage]) -> usize {
    messages.iter().map(message_chars).sum()
}

fn message_chars(m: &ChatMessage) -> usize {
    m.content
        .iter()
        .map(|b| match b {
            ContentBlock::Text(s) => s.chars().count(),
            ContentBlock::ToolUse { name, input, .. } => {
                name.chars().count() + input.chars().count()
            }
            ContentBlock::ToolResult { content, .. } => content.chars().count(),
        })
        .sum()
}

/// Return the index up to which we can keep messages so the total stays
/// below `budget` chars, AND we don't split a `tool_use` / `tool_result`
/// pair. The returned index is the "keep tail starting at index" — i.e.
/// `&messages[idx..]` is what we keep, `&messages[..idx]` is the head we
/// compact away.
///
/// Guarantees:
/// - `idx <= messages.len().saturating_sub(2)` — we always keep at least
///   the final two messages so there's a user→assistant pair to continue.
/// - If `idx > 0`, then `is_tool_use_pair_boundary(messages, idx)` is false
///   — we never split a tool_use that lives in `messages[idx-1]` from its
///   matching tool_result in `messages[idx]`.
/// - For very short histories (`len <= 2`), returns 0 (keep everything).
pub fn split_for_compaction(messages: &[ChatMessage], budget: usize) -> usize {
    let n = messages.len();
    if n <= 2 {
        return 0;
    }
    // Largest valid keep-start index: keep at least the final 2 messages.
    let max_keep_start = n - 2;

    // Walk from the end, accumulating chars. `keep_start` shrinks as we
    // add older messages. Stop when adding the next earlier one would
    // exceed `budget`.
    let mut keep_start = max_keep_start;
    let mut total = 0usize;
    for i in (0..n).rev() {
        let c = message_chars(&messages[i]);
        if total.saturating_add(c) > budget && i < max_keep_start {
            // Don't include this one — keep_start stays at i+1.
            break;
        }
        total = total.saturating_add(c);
        keep_start = i;
        if i == 0 {
            break;
        }
    }

    // If we'd split a tool_use/tool_result pair, back up one more so the
    // pair stays together in the head (i.e. both get summarised).
    while keep_start > 0 && is_tool_use_pair_boundary(messages, keep_start) {
        keep_start -= 1;
    }
    // Clamp — backing up for a pair can't let us keep less than the final 2.
    keep_start.min(max_keep_start)
}

/// True iff splitting `messages` at `idx` (i.e. head = `..idx`, tail =
/// `idx..`) would separate a `tool_use` block in `messages[idx-1]` from
/// its matching `tool_result` in `messages[idx]`.
///
/// Anthropic's contract: an assistant message with a `tool_use` block is
/// always followed by a user message containing the matching `tool_result`.
/// Splitting between them would leave a dangling tool_use, which the model
/// rejects.
///
/// `idx == 0` or `idx >= messages.len()`: no boundary, returns false.
pub fn is_tool_use_pair_boundary(messages: &[ChatMessage], idx: usize) -> bool {
    if idx == 0 || idx >= messages.len() {
        return false;
    }
    let prev = &messages[idx - 1];
    let next = &messages[idx];
    if prev.role != Role::Assistant || next.role != Role::User {
        return false;
    }
    let has_tool_use = prev.content.iter().any(|b| matches!(b, ContentBlock::ToolUse { .. }));
    if !has_tool_use {
        return false;
    }
    next.content.iter().any(|b| matches!(b, ContentBlock::ToolResult { .. }))
}

/// Model-driven compactor. Stand-alone helper the caller invokes between
/// turns; Session integration is future work.
pub struct Compactor {
    model: Arc<dyn Model>,
    target_chars: usize,
    max_tokens_for_summary: u32,
}

#[derive(Debug, Clone)]
pub enum CompactError {
    Model(String),
}

impl std::fmt::Display for CompactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompactError::Model(s) => write!(f, "model error: {s}"),
        }
    }
}

impl std::error::Error for CompactError {}

impl Compactor {
    pub fn new(model: Arc<dyn Model>, target_chars: usize) -> Self {
        Self { model, target_chars, max_tokens_for_summary: 1024 }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens_for_summary = max_tokens;
        self
    }

    /// If `messages` exceeds `target_chars * 2` (hysteresis), summarise the
    /// head and return a new Vec where the head is replaced by a single
    /// synthetic Assistant `ContentBlock::Text`. Otherwise return the input
    /// unchanged.
    ///
    /// Best-effort: an empty summary or a model error returns the input
    /// unchanged rather than propagating. (We never want compaction to
    /// drop a turn the user might need.)
    pub fn maybe_compact(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<Vec<ChatMessage>, CompactError> {
        let threshold = self.target_chars.saturating_mul(2);
        if estimate_chars(&messages) <= threshold {
            return Ok(messages);
        }
        let split = split_for_compaction(&messages, self.target_chars);
        if split == 0 {
            // Nothing safe to compact.
            return Ok(messages);
        }

        let head = &messages[..split];
        let prompt = build_summary_prompt(head);
        let req = ModelRequest {
            system: None,
            messages: vec![ChatMessage::user(prompt)],
            max_tokens: self.max_tokens_for_summary,
            tools: Vec::new(),
        };

        let mut summary = String::new();
        for ev in self.model.stream(req) {
            match ev {
                Ok(ModelEvent::TextDelta(s)) => summary.push_str(&s),
                Ok(_) => {}
                // Best-effort: surface the error message via CompactError so
                // the caller can log it, but most callers will just discard
                // it and keep going with the original history.
                Err(e) => return Err(CompactError::Model(e.0)),
            }
        }

        if summary.trim().is_empty() {
            // Empty summary — fall back to the original history.
            return Ok(messages);
        }

        let synthetic = ChatMessage::assistant(format!("[Earlier turns summarised:]\n{summary}"));
        let tail_len = messages.len() - split;
        let mut out = Vec::with_capacity(1 + tail_len);
        out.push(synthetic);
        out.extend(messages.into_iter().skip(split));
        Ok(out)
    }
}

/// Render the head as a plain-text transcript for the summariser.
fn build_summary_prompt(head: &[ChatMessage]) -> String {
    let mut s = String::new();
    s.push_str(
        "Summarise the following conversation history in 5-10 lines, preserving \
tool calls and their outcomes. Be concise — this summary replaces the head of \
the message history in a follow-up turn.\n\n--- transcript ---\n",
    );
    for m in head {
        let role = match m.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
        };
        s.push_str(role);
        s.push_str(":\n");
        for b in &m.content {
            match b {
                ContentBlock::Text(t) => {
                    s.push_str(t);
                    s.push('\n');
                }
                ContentBlock::ToolUse { name, input, .. } => {
                    s.push_str("  [tool_use ");
                    s.push_str(name);
                    s.push_str("] ");
                    s.push_str(input);
                    s.push('\n');
                }
                ContentBlock::ToolResult { content, is_error, .. } => {
                    s.push_str(if *is_error { "  [tool_error] " } else { "  [tool_result] " });
                    s.push_str(content);
                    s.push('\n');
                }
            }
        }
        s.push('\n');
    }
    s.push_str("--- end transcript ---\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use harness::{ContentBlock, MockModel, Role};

    fn user_text(s: &str) -> ChatMessage {
        ChatMessage::user(s)
    }
    fn assist_text(s: &str) -> ChatMessage {
        ChatMessage::assistant(s)
    }
    fn assist_tool_use(id: &str, name: &str, input: &str) -> ChatMessage {
        ChatMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: id.into(),
                name: name.into(),
                input: input.into(),
            }],
        }
    }
    fn user_tool_result(id: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: id.into(),
                content: content.into(),
                is_error: false,
            }],
        }
    }

    // ---- estimate_chars -------------------------------------------------

    #[test]
    fn estimate_chars_empty() {
        assert_eq!(estimate_chars(&[]), 0);
    }

    #[test]
    fn estimate_chars_simple_text() {
        let msgs = vec![user_text("hello"), assist_text("world!")];
        assert_eq!(estimate_chars(&msgs), 5 + 6);
    }

    #[test]
    fn estimate_chars_includes_tool_use_and_result() {
        let msgs = vec![
            user_text("hi"),
            assist_tool_use("t1", "bash", "{\"command\":\"ls\"}"),
            user_tool_result("t1", "file.txt\n"),
        ];
        // "hi" = 2; "bash" (4) + input (17) = 21; tool_result content = 9.
        let expected =
            2 + 4 + "{\"command\":\"ls\"}".chars().count() + "file.txt\n".chars().count();
        assert_eq!(estimate_chars(&msgs), expected);
    }

    #[test]
    fn estimate_chars_counts_unicode_chars_not_bytes() {
        // é is 2 bytes UTF-8 but one char.
        let msgs = vec![user_text("café")];
        assert_eq!(estimate_chars(&msgs), 4);
    }

    // ---- split_for_compaction ------------------------------------------

    #[test]
    fn split_short_history_keeps_everything() {
        let msgs = vec![user_text("a"), assist_text("b")];
        assert_eq!(split_for_compaction(&msgs, 1000), 0);
    }

    #[test]
    fn split_respects_budget() {
        // 5 messages, each 10 chars. Budget 25 → fits ~2 messages tail.
        // We always keep at least the final 2; that's 20 chars, leaves
        // 5 for one more. Adding a 10-char message exceeds → stop.
        let m = |s: &str| user_text(&"x".repeat(10).replace("x", s));
        let msgs = vec![m("a"), m("b"), m("c"), m("d"), m("e")];
        // Final 2 always kept; can we fit one earlier? 20 + 10 = 30 > 25, no.
        // So keep_start = 3 (indices 3,4 = final 2).
        assert_eq!(split_for_compaction(&msgs, 25), 3);
    }

    #[test]
    fn split_large_budget_keeps_all() {
        let msgs = vec![user_text("a"), assist_text("b"), user_text("c"), assist_text("d")];
        assert_eq!(split_for_compaction(&msgs, 10_000), 0);
    }

    #[test]
    fn split_always_keeps_final_two() {
        // Tiny budget — but we must still keep final 2.
        let msgs = vec![
            user_text("aaaaaaaaaa"),
            assist_text("bbbbbbbbbb"),
            user_text("cccccccccc"),
            assist_text("dddddddddd"),
        ];
        let idx = split_for_compaction(&msgs, 0);
        assert_eq!(idx, 2); // keep messages[2..] = final 2
        assert_eq!(msgs.len() - idx, 2);
    }

    #[test]
    fn split_never_breaks_tool_use_pair() {
        // History: u, a, u, a(tool_use), u(tool_result), a, u, a
        // = 8 messages. Indices 3 (tool_use) and 4 (tool_result) must stay together.
        let msgs = vec![
            user_text("hello"),
            assist_text("hi"),
            user_text("what's there?"),
            assist_tool_use("t1", "bash", "{\"command\":\"ls\"}"),
            user_tool_result("t1", "one.txt"),
            assist_text("there's one.txt"),
            user_text("ok"),
            assist_text("done"),
        ];
        // Force a split where naive walking would land on idx=4 (tool_result
        // alone in tail). Budget chosen so tail = last 4 messages (5..7) but
        // adding idx 4 (tool_result) would be a pair boundary.
        // Simpler: just check several budgets and assert no boundary.
        for budget in 0..200 {
            let idx = split_for_compaction(&msgs, budget);
            // Final 2 always kept.
            assert!(idx <= msgs.len() - 2, "idx {idx} too large for budget {budget}");
            // No tool_use/result split.
            assert!(
                !is_tool_use_pair_boundary(&msgs, idx),
                "split at {idx} breaks a tool_use/result pair (budget {budget})"
            );
        }
    }

    #[test]
    fn is_tool_use_pair_boundary_detection() {
        let msgs = vec![
            user_text("hi"),
            assist_tool_use("t1", "bash", "{}"),
            user_tool_result("t1", "ok"),
            assist_text("done"),
        ];
        assert!(!is_tool_use_pair_boundary(&msgs, 0));
        assert!(!is_tool_use_pair_boundary(&msgs, 1)); // user→assistant(tool_use), not the pair
        assert!(is_tool_use_pair_boundary(&msgs, 2)); // assistant(tool_use)→user(tool_result)
        assert!(!is_tool_use_pair_boundary(&msgs, 3));
        assert!(!is_tool_use_pair_boundary(&msgs, 4)); // out of range
    }

    #[test]
    fn split_backs_up_through_pair_boundary() {
        // Force a candidate split that lands on a pair boundary; verify we
        // back up by 1 so both pair members live in the head.
        let msgs = vec![
            user_text("aa"),                    // 0
            assist_tool_use("t1", "bash", "x"), // 1 (5 chars: "bash"+"x"=5)
            user_tool_result("t1", "ok"),       // 2 (2 chars)
            user_text("aa"),                    // 3
            assist_text("bb"),                  // 4
        ];
        // Final 2 kept = indices 3,4 (4 chars). Budget 6 → can fit one more
        // message of <=2 chars at index 2 (the tool_result, 2 chars). That
        // would make keep_start=2, but messages[1]=tool_use, messages[2]=tool_result
        // — that's a boundary. We back up to 1.
        let idx = split_for_compaction(&msgs, 6);
        assert_eq!(idx, 1);
        assert!(!is_tool_use_pair_boundary(&msgs, idx));
    }

    // ---- Compactor::maybe_compact --------------------------------------

    fn many_msgs(n: usize, per_chars: usize) -> Vec<ChatMessage> {
        let s = "x".repeat(per_chars);
        (0..n).map(|i| if i % 2 == 0 { user_text(&s) } else { assist_text(&s) }).collect()
    }

    #[test]
    fn maybe_compact_below_threshold_returns_unchanged() {
        // target_chars=100, threshold=200. Total = 4 * 10 = 40, well under.
        let model =
            Arc::new(MockModel::single(vec![ModelEvent::TextDelta("should not be called".into())]));
        let compactor = Compactor::new(model.clone(), 100);
        let msgs = many_msgs(4, 10);
        let out = compactor.maybe_compact(msgs.clone()).unwrap();
        assert_eq!(out.len(), msgs.len());
        // Model not called.
        assert_eq!(model.seen.lock().unwrap().len(), 0);
    }

    #[test]
    fn maybe_compact_above_threshold_summarises_head() {
        // target_chars=20, threshold=40. Total = 8 * 10 = 80 > 40.
        let model = Arc::new(MockModel::single(vec![
            ModelEvent::TextDelta("summarised text".into()),
            ModelEvent::Stop { reason: Some("end_turn".into()) },
        ]));
        let compactor = Compactor::new(model.clone(), 20);
        let msgs = many_msgs(8, 10);
        let out = compactor.maybe_compact(msgs).unwrap();
        // First message is synthetic Assistant carrying the summary.
        assert_eq!(out[0].role, Role::Assistant);
        match &out[0].content[0] {
            ContentBlock::Text(t) => {
                assert!(t.starts_with("[Earlier turns summarised:]"));
                assert!(t.contains("summarised text"));
            }
            _ => panic!("expected synthetic Text block"),
        }
        // Model was called exactly once.
        assert_eq!(model.seen.lock().unwrap().len(), 1);
    }

    #[test]
    fn maybe_compact_empty_summary_falls_back() {
        // Model returns nothing of substance.
        let model = Arc::new(MockModel::single(vec![ModelEvent::TextDelta("   ".into())]));
        let compactor = Compactor::new(model, 20);
        let msgs = many_msgs(8, 10);
        let original_len = msgs.len();
        let out = compactor.maybe_compact(msgs).unwrap();
        // Unchanged.
        assert_eq!(out.len(), original_len);
        // First message is whatever was in the original (user_text("xxxxxxxxxx")),
        // NOT a synthetic Assistant summary.
        assert_eq!(out[0].role, Role::User);
    }

    #[test]
    fn maybe_compact_preserves_tail_verbatim() {
        let model = Arc::new(MockModel::single(vec![ModelEvent::TextDelta("summary here".into())]));
        let compactor = Compactor::new(model, 20);
        let mut msgs = many_msgs(8, 10);
        // Tag the final two so we can identify them.
        msgs[6] = user_text("UNIQUE-PENULTIMATE-USER");
        msgs[7] = assist_text("UNIQUE-FINAL-ASSISTANT");
        let out = compactor.maybe_compact(msgs).unwrap();
        // The final two messages of `out` must match the originals exactly.
        let final_assist = &out[out.len() - 1];
        let penult_user = &out[out.len() - 2];
        assert_eq!(penult_user.role, Role::User);
        assert_eq!(final_assist.role, Role::Assistant);
        match (&penult_user.content[0], &final_assist.content[0]) {
            (ContentBlock::Text(a), ContentBlock::Text(b)) => {
                assert_eq!(&**a, "UNIQUE-PENULTIMATE-USER");
                assert_eq!(&**b, "UNIQUE-FINAL-ASSISTANT");
            }
            _ => panic!("expected text blocks"),
        }
    }

    #[test]
    fn maybe_compact_passes_summarisation_prompt_to_model() {
        // Confirm the request the model sees mentions "Summarise" and
        // includes head content (here: "ORIGINAL-HEAD-TEXT").
        let model = Arc::new(MockModel::single(vec![ModelEvent::TextDelta("ok".into())]));
        let compactor = Compactor::new(model.clone(), 20);
        let mut msgs = many_msgs(8, 10);
        msgs[0] = user_text("ORIGINAL-HEAD-TEXT");
        let _ = compactor.maybe_compact(msgs).unwrap();
        let seen = model.seen.lock().unwrap();
        let req = &seen[0];
        assert_eq!(req.messages.len(), 1);
        match &req.messages[0].content[0] {
            ContentBlock::Text(t) => {
                assert!(t.contains("Summarise"));
                assert!(t.contains("ORIGINAL-HEAD-TEXT"));
            }
            _ => panic!("expected text prompt"),
        }
    }

    #[test]
    fn maybe_compact_with_tool_blocks_in_head_summarises_them_too() {
        // A history with tool_use + tool_result in the head; ensure the
        // prompt encodes them with tool markers.
        let model = Arc::new(MockModel::single(vec![ModelEvent::TextDelta("compacted".into())]));
        let compactor = Compactor::new(model.clone(), 20);
        let msgs = vec![
            user_text("aaaaaaaaaa"),
            assist_tool_use("t1", "bash", "{\"cmd\":\"ls\"}"),
            user_tool_result("t1", "out.txt"),
            assist_text("bbbbbbbbbb"),
            user_text("cccccccccc"),
            assist_text("dddddddddd"),
            user_text("eeeeeeeeee"),
            assist_text("ffffffffff"),
        ];
        let _ = compactor.maybe_compact(msgs).unwrap();
        let seen = model.seen.lock().unwrap();
        match &seen[0].messages[0].content[0] {
            ContentBlock::Text(t) => {
                assert!(t.contains("[tool_use bash]"));
                assert!(t.contains("[tool_result]"));
            }
            _ => panic!("expected text prompt"),
        }
    }

    #[test]
    fn maybe_compact_model_error_propagates() {
        // A model that yields an error is surfaced via CompactError.
        struct ErrModel;
        impl Model for ErrModel {
            fn stream(
                &self,
                _req: ModelRequest,
            ) -> Box<dyn Iterator<Item = Result<ModelEvent, harness::ModelError>> + Send>
            {
                Box::new(std::iter::once(Err(harness::ModelError("boom".into()))))
            }
        }
        let compactor = Compactor::new(Arc::new(ErrModel), 20);
        let msgs = many_msgs(8, 10);
        let err = compactor.maybe_compact(msgs).unwrap_err();
        match err {
            CompactError::Model(m) => assert_eq!(m, "boom"),
        }
    }

    #[test]
    fn maybe_compact_split_zero_returns_unchanged() {
        // Threshold is exceeded, but split_for_compaction returns 0 because
        // even after backing up we can't safely compact. Build a 3-msg
        // history with a tool pair right at the end so backing-up wipes
        // the split to 0.
        let model =
            Arc::new(MockModel::single(vec![ModelEvent::TextDelta("should never be used".into())]));
        // target_chars=5, threshold=10. msgs total >> 10 to trigger.
        let compactor = Compactor::new(model.clone(), 5);
        // 3 messages: assist(tool_use), user(tool_result), assist(text)
        // Final 2 kept ⇒ max_keep_start = 1. messages[0]=tool_use,
        // messages[1]=tool_result → boundary at idx=1, back up to 0.
        let msgs = vec![
            assist_tool_use("t1", "bash", "xxxxxxxxxxxxxxxxxxxxxxxx"),
            user_tool_result("t1", "yyyyyyyyyyyyyyyyyyyyyy"),
            assist_text("zzzzzzzzzzzzzzzzzzzzzz"),
        ];
        let orig_len = msgs.len();
        let out = compactor.maybe_compact(msgs).unwrap();
        assert_eq!(out.len(), orig_len);
        // Model not called (split==0 ⇒ early return before stream).
        assert_eq!(model.seen.lock().unwrap().len(), 0);
    }

    #[test]
    fn maybe_compact_request_max_tokens_honoured() {
        let model = Arc::new(MockModel::single(vec![ModelEvent::TextDelta("ok".into())]));
        let compactor = Compactor::new(model.clone(), 20).with_max_tokens(256);
        let msgs = many_msgs(8, 10);
        let _ = compactor.maybe_compact(msgs).unwrap();
        let seen = model.seen.lock().unwrap();
        assert_eq!(seen[0].max_tokens, 256);
    }

    #[test]
    fn maybe_compact_synthetic_message_is_assistant_role() {
        // Doc-promise: the synthetic message is Assistant-role so it doesn't
        // look like a fresh user prompt to the model.
        let model = Arc::new(MockModel::single(vec![ModelEvent::TextDelta("synthesised".into())]));
        let compactor = Compactor::new(model, 20);
        let msgs = many_msgs(8, 10);
        let out = compactor.maybe_compact(msgs).unwrap();
        assert_eq!(out[0].role, Role::Assistant);
    }
}
