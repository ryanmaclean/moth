//! Session actor: owns message history and drives the iteration loop.
//!
//! `HarnessState` is the passive "harness" in Flue's hierarchy — model
//! defaults plus the Instance handle. It lives behind `Arc` so multiple
//! sessions on one instance share it without cloning the model.
//!
//! The iteration loop:
//!   1. Append user message to history.
//!   2. Stream a model turn, accumulating text and tool-use blocks.
//!   3. After each text delta, scan for `<promise>COMPLETE</promise>`;
//!      break early on hit.
//!   4. Append the assistant turn (text + tool_use blocks) to history.
//!   5. If the turn contains tool_use blocks, execute them via the
//!      Instance, append a user message with tool_result blocks, and
//!      loop back to (2). Otherwise return.

use std::sync::Arc;
use std::sync::mpsc::SyncSender;

use actor::{Actor, ActorRef};
use wire::find_tag;

use crate::instance::InstanceMsg;
use crate::model::{
    ChatMessage, ContentBlock, Model, ModelEvent, ModelRequest, Role, ToolDef,
};
use crate::persist::SessionStore;
use crate::sandbox::{SandboxError, ShellResult};
use crate::tools::{Tool, ToolCtx, default_tools};

const DEFAULT_MAX_TOKENS: u32 = 4096;
const MAX_TURNS_PER_PROMPT: usize = 16;

/// Callback that may shorten `messages` (typically by summarising older
/// turns). Called by the iteration loop BEFORE each turn. Implementations
/// should be best-effort: return the input unchanged on error.
pub type CompactFn =
    dyn Fn(Vec<ChatMessage>) -> Vec<ChatMessage> + Send + Sync;

pub struct HarnessState {
    pub name: String,
    pub model: Arc<dyn Model>,
    pub default_system: Option<String>,
    pub default_max_tokens: u32,
    pub instance: ActorRef<InstanceMsg>,
    pub tools: Vec<Arc<dyn Tool>>,
    /// Optional compaction callback. The Session invokes it between turns
    /// so a long history can be summarised before the next ModelRequest.
    pub compactor: Option<Arc<CompactFn>>,
}

impl HarnessState {
    pub fn new(
        name: impl Into<String>,
        model: Arc<dyn Model>,
        instance: ActorRef<InstanceMsg>,
    ) -> Self {
        Self {
            name: name.into(),
            model,
            default_system: None,
            default_max_tokens: DEFAULT_MAX_TOKENS,
            instance,
            tools: default_tools(),
            compactor: None,
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.default_system = Some(system.into());
        self
    }

    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_compactor(mut self, c: Arc<CompactFn>) -> Self {
        self.compactor = Some(c);
        self
    }

    pub fn tool_defs(&self) -> Vec<ToolDef> {
        self.tools.iter().map(|t| t.definition()).collect()
    }
}

pub struct Session {
    pub name: String,
    pub history: Vec<ChatMessage>,
    pub harness: Arc<HarnessState>,
    pub store: Option<Arc<dyn SessionStore>>,
}

impl Session {
    pub fn new(name: impl Into<String>, harness: Arc<HarnessState>) -> Self {
        Self { name: name.into(), history: Vec::new(), harness, store: None }
    }

    /// Attach a persistence store and load any previous history under
    /// `self.name`. Call this before `spawn`-ing. Errors during load are
    /// logged to stderr and ignored — a corrupt store shouldn't prevent
    /// fresh conversations.
    pub fn with_store(mut self, store: Arc<dyn SessionStore>) -> Self {
        match store.load(&self.name) {
            Ok(Some(h)) => self.history = h,
            Ok(None) => {}
            Err(e) => eprintln!("session store load failed for {}: {}", self.name, e),
        }
        self.store = Some(store);
        self
    }
}

#[derive(Debug, Clone)]
pub struct PromptResult {
    pub text: String,
    pub structured: Option<Vec<u8>>,
    pub completed: bool,
    pub turns: usize,
}

#[derive(Debug, Clone)]
pub enum SessionError {
    Model(String),
    Mailbox,
    TurnLimitExceeded,
}

pub enum SessionMsg {
    Prompt {
        text: String,
        structured_output_tag: Option<String>,
        reply: SyncSender<Result<PromptResult, SessionError>>,
    },
    /// Same iteration loop as `Prompt`, but emits `StreamEvent`s through
    /// `events` as the turn progresses. Drop the receiver to cancel: the
    /// next `events.send()` returns Err, the iteration aborts cleanly,
    /// the partial assistant turn is appended to history, and the final
    /// `Done(_)` / `Cancelled` is the last event sent (best-effort —
    /// if the receiver is gone it's just dropped).
    PromptStream {
        text: String,
        structured_output_tag: Option<String>,
        /// Bounded sender — callers should use `sync_channel(cap)` so a
        /// slow consumer applies backpressure to the iteration loop
        /// instead of letting events queue without bound. Unbounded
        /// callers can use `sync_channel(usize::MAX / 2)` but it's
        /// almost always wrong.
        events: SyncSender<StreamEvent>,
    },
    Shell {
        cmd: String,
        reply: SyncSender<Result<ShellResult, SandboxError>>,
    },
}

/// Live events emitted while a prompt streams. Order matches the wire
/// order of the underlying model + tool execution. `Done` or `Cancelled`
/// is always the terminal event (best-effort: if the receiver dropped
/// mid-stream we won't get to send it).
#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    ToolUseStart { id: String, name: String },
    ToolUseInputDelta(String),
    BlockStop,
    ToolResult { tool_use_id: String, content: String, is_error: bool },
    TurnComplete { turn: usize, stop_reason: Option<String> },
    Done(PromptResult),
    Cancelled,
    Error(SessionError),
}

impl Actor for Session {
    type Msg = SessionMsg;

    fn handle(&mut self, msg: SessionMsg) {
        match msg {
            SessionMsg::Prompt { text, structured_output_tag, reply } => {
                let _ = reply.send(self.run_prompt(text, structured_output_tag));
            }
            SessionMsg::PromptStream { text, structured_output_tag, events } => {
                let result = self.run_prompt_inner(text, structured_output_tag, Some(&events));
                // Best-effort terminal event. Cancellation already emitted
                // its own terminal so we skip emitting Done after Cancelled.
                if result.is_err() {
                    let _ = events.send(StreamEvent::Error(result.err().unwrap()));
                } else if let Ok(Some(pr)) = result {
                    let _ = events.send(StreamEvent::Done(pr));
                }
                // result == Ok(None) means the receiver dropped — Cancelled
                // was already sent (or attempted). Nothing to do.
            }
            SessionMsg::Shell { cmd, reply } => {
                if self
                    .harness
                    .instance
                    .send(InstanceMsg::Shell { cmd, reply: reply.clone() })
                    .is_err()
                {
                    let _ = reply.send(Err(SandboxError("instance mailbox closed".into())));
                }
            }
        }
    }
}

impl Session {
    fn run_prompt(
        &mut self,
        text: String,
        structured_output_tag: Option<String>,
    ) -> Result<PromptResult, SessionError> {
        match self.run_prompt_inner(text, structured_output_tag, None)? {
            Some(pr) => Ok(pr),
            // events is None → cancellation impossible → always Some
            None => unreachable!("non-streaming run can't be cancelled"),
        }
    }

    /// Inner iteration loop shared by `run_prompt` (blocking) and
    /// `SessionMsg::PromptStream`. `Ok(Some(pr))` on completion;
    /// `Ok(None)` if `events` is set and the receiver dropped (cancellation
    /// — partial assistant turn is still appended to history). `Err(_)` for
    /// real failures (model error, turn-limit, etc.).
    fn run_prompt_inner(
        &mut self,
        text: String,
        structured_output_tag: Option<String>,
        events: Option<&SyncSender<StreamEvent>>,
    ) -> Result<Option<PromptResult>, SessionError> {
        self.history.push(ChatMessage::user(text));

        let mut response_text = String::new();
        let mut completed = false;
        let mut turns = 0;

        // Closure: emit + detect cancellation. Returns false if the receiver
        // dropped, true to keep going. No-op if there's no receiver.
        let emit = |evs: Option<&SyncSender<StreamEvent>>, ev: StreamEvent| -> bool {
            match evs {
                Some(tx) => tx.send(ev).is_ok(),
                None => true,
            }
        };

        loop {
            turns += 1;
            if turns > MAX_TURNS_PER_PROMPT {
                return Err(SessionError::TurnLimitExceeded);
            }

            // Best-effort compaction before the model sees the next request.
            // The callback either summarises older turns or returns the
            // history unchanged. Implementations must not panic.
            if let Some(c) = &self.harness.compactor {
                self.history = (c)(std::mem::take(&mut self.history));
            }

            let req = ModelRequest {
                system: self.harness.default_system.clone(),
                messages: self.history.clone(),
                max_tokens: self.harness.default_max_tokens,
                tools: self.harness.tool_defs(),
            };

            let mut blocks: Vec<ContentBlock> = Vec::new();
            let mut current_text = String::new();
            let mut current_tool: Option<(String, String, String)> = None;
            let mut stop_reason: Option<String> = None;
            let mut cancelled = false;

            for ev in self.harness.model.stream(req) {
                match ev {
                    Ok(ModelEvent::TextDelta(s)) => {
                        current_text.push_str(&s);
                        response_text.push_str(&s);
                        if !emit(events, StreamEvent::TextDelta(s)) {
                            cancelled = true;
                            break;
                        }
                        if let Some(body) = find_tag(response_text.as_bytes(), b"promise")
                            && body == b"COMPLETE"
                        {
                            completed = true;
                            break;
                        }
                    }
                    Ok(ModelEvent::ToolUseStart { id, name }) => {
                        if !current_text.is_empty() {
                            blocks.push(ContentBlock::Text(std::mem::take(&mut current_text)));
                        }
                        if !emit(events, StreamEvent::ToolUseStart {
                            id: id.clone(),
                            name: name.clone(),
                        }) {
                            cancelled = true;
                            break;
                        }
                        current_tool = Some((id, name, String::new()));
                    }
                    Ok(ModelEvent::ToolUseInputDelta(s)) => {
                        if let Some((_, _, input)) = current_tool.as_mut() {
                            input.push_str(&s);
                        }
                        if !emit(events, StreamEvent::ToolUseInputDelta(s)) {
                            cancelled = true;
                            break;
                        }
                    }
                    Ok(ModelEvent::BlockStop) => {
                        if let Some((id, name, input)) = current_tool.take() {
                            blocks.push(ContentBlock::ToolUse { id, name, input });
                        }
                        if !emit(events, StreamEvent::BlockStop) {
                            cancelled = true;
                            break;
                        }
                    }
                    Ok(ModelEvent::Stop { reason }) => {
                        stop_reason = reason;
                    }
                    Err(e) => return Err(SessionError::Model(e.0)),
                }
            }

            // Flush any pending blocks the stream didn't terminate cleanly.
            if !current_text.is_empty() {
                blocks.push(ContentBlock::Text(std::mem::take(&mut current_text)));
            }
            if let Some((id, name, input)) = current_tool.take() {
                blocks.push(ContentBlock::ToolUse { id, name, input });
            }

            let tool_uses: Vec<(String, String, String)> = blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolUse { id, name, input } => {
                        Some((id.clone(), name.clone(), input.clone()))
                    }
                    _ => None,
                })
                .collect();

            self.history.push(ChatMessage { role: Role::Assistant, content: blocks });

            if cancelled {
                // Persist partial history then surface Cancelled to the caller.
                if let Some(store) = &self.store
                    && let Err(e) = store.save(&self.name, &self.history)
                {
                    eprintln!("session store save failed for {}: {}", self.name, e);
                }
                let _ = emit(events, StreamEvent::Cancelled);
                return Ok(None);
            }

            // Emit per-turn summary BEFORE we possibly execute tools — this
            // lets observers (runlog, telemetry) checkpoint per turn.
            if !emit(events, StreamEvent::TurnComplete { turn: turns, stop_reason: stop_reason.clone() }) {
                let _ = emit(events, StreamEvent::Cancelled);
                if let Some(store) = &self.store
                    && let Err(e) = store.save(&self.name, &self.history)
                {
                    eprintln!("session store save failed for {}: {}", self.name, e);
                }
                return Ok(None);
            }

            if completed
                || tool_uses.is_empty()
                || stop_reason.as_deref() == Some("end_turn")
            {
                break;
            }

            let mut results: Vec<ContentBlock> = Vec::with_capacity(tool_uses.len());
            for (id, name, input) in tool_uses {
                let block = self.execute_tool(&id, &name, &input);
                if let ContentBlock::ToolResult { tool_use_id, content, is_error } = &block
                    && !emit(events, StreamEvent::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: content.clone(),
                        is_error: *is_error,
                    })
                {
                    // Cancellation between tool calls: append what we have
                    // and bail.
                    results.push(block);
                    self.history.push(ChatMessage { role: Role::User, content: results });
                    if let Some(store) = &self.store
                        && let Err(e) = store.save(&self.name, &self.history)
                    {
                        eprintln!("session store save failed for {}: {}", self.name, e);
                    }
                    let _ = emit(events, StreamEvent::Cancelled);
                    return Ok(None);
                }
                results.push(block);
            }
            self.history.push(ChatMessage { role: Role::User, content: results });
        }

        let structured = structured_output_tag
            .as_deref()
            .and_then(|tag| find_tag(response_text.as_bytes(), tag.as_bytes()).map(<[u8]>::to_vec));

        if let Some(store) = &self.store
            && let Err(e) = store.save(&self.name, &self.history)
        {
            eprintln!("session store save failed for {}: {}", self.name, e);
        }

        Ok(Some(PromptResult { text: response_text, structured, completed, turns }))
    }

    fn execute_tool(&self, id: &str, name: &str, input: &str) -> ContentBlock {
        let tool = self.harness.tools.iter().find(|t| t.name() == name);
        let Some(tool) = tool else {
            return ContentBlock::ToolResult {
                tool_use_id: id.into(),
                content: format!("unknown tool: {name}"),
                is_error: true,
            };
        };
        let ctx = ToolCtx { instance: &self.harness.instance };
        match tool.call(input, &ctx) {
            Ok(content) => ContentBlock::ToolResult {
                tool_use_id: id.into(),
                content,
                is_error: false,
            },
            Err(e) => ContentBlock::ToolResult {
                tool_use_id: id.into(),
                content: e.0,
                is_error: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::Instance;
    use crate::model::MockModel;
    use crate::sandbox::MockSandbox;
    use actor::spawn;

    fn rig(
        model_scripts: Vec<Vec<ModelEvent>>,
        sandbox_responses: Vec<ShellResult>,
    ) -> (
        actor::Spawned<InstanceMsg>,
        actor::Spawned<SessionMsg>,
        Arc<MockModel>,
        Arc<MockSandbox>,
    ) {
        let sandbox = Arc::new(MockSandbox::new(sandbox_responses));
        let instance = spawn(Instance::new("inst-1", Box::new(SandboxRef(sandbox.clone()))));
        let model = Arc::new(MockModel::new(model_scripts));
        let harness = Arc::new(HarnessState::new(
            "harness-1",
            model.clone() as Arc<dyn Model>,
            instance.addr.clone(),
        ));
        let session = spawn(Session::new("sess-1", harness));
        (instance, session, model, sandbox)
    }

    /// Thin wrapper that lets the test rig share a MockSandbox with the
    /// Instance actor (the Instance owns a Box<dyn Sandbox>; we want to
    /// peek at recorded commands from the test thread).
    struct SandboxRef(Arc<MockSandbox>);
    impl crate::sandbox::Sandbox for SandboxRef {
        fn execute(&mut self, cmd: &str) -> Result<ShellResult, SandboxError> {
            self.0.recorded.lock().unwrap().push(cmd.to_string());
            let mut r = self.0.responses.lock().unwrap();
            Ok(if r.is_empty() {
                ShellResult { exit_code: 0, stdout: Vec::new(), stderr: Vec::new() }
            } else {
                r.remove(0)
            })
        }
    }

    #[test]
    fn prompt_roundtrip() {
        let (instance, session, _model, _sb) = rig(
            vec![vec![
                ModelEvent::TextDelta("hello ".into()),
                ModelEvent::TextDelta("world".into()),
                ModelEvent::Stop { reason: Some("end_turn".into()) },
            ]],
            vec![],
        );

        let res = session
            .addr
            .ask(|reply| SessionMsg::Prompt {
                text: "hi".into(),
                structured_output_tag: None,
                reply,
            })
            .unwrap()
            .unwrap();

        assert_eq!(res.text, "hello world");
        assert_eq!(res.turns, 1);
        assert!(!res.completed);
        assert!(res.structured.is_none());

        session.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn multi_turn_history_accumulates() {
        let (instance, session, model, _sb) = rig(
            vec![
                vec![
                    ModelEvent::TextDelta("first reply".into()),
                    ModelEvent::Stop { reason: Some("end_turn".into()) },
                ],
                vec![
                    ModelEvent::TextDelta("second reply".into()),
                    ModelEvent::Stop { reason: Some("end_turn".into()) },
                ],
            ],
            vec![],
        );

        let _ = session
            .addr
            .ask(|reply| SessionMsg::Prompt {
                text: "one".into(),
                structured_output_tag: None,
                reply,
            })
            .unwrap()
            .unwrap();

        let _ = session
            .addr
            .ask(|reply| SessionMsg::Prompt {
                text: "two".into(),
                structured_output_tag: None,
                reply,
            })
            .unwrap()
            .unwrap();

        let seen = model.seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        let second = &seen[1];
        assert_eq!(second.messages.len(), 3);
        assert_eq!(second.messages[0].role, Role::User);
        assert_eq!(
            second.messages[0].content,
            vec![ContentBlock::Text("one".into())]
        );
        assert_eq!(second.messages[1].role, Role::Assistant);
        assert_eq!(
            second.messages[1].content,
            vec![ContentBlock::Text("first reply".into())]
        );
        assert_eq!(second.messages[2].role, Role::User);
        assert_eq!(
            second.messages[2].content,
            vec![ContentBlock::Text("two".into())]
        );

        session.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn completion_signal_marks_completed() {
        let (instance, session, _model, _sb) = rig(
            vec![vec![
                ModelEvent::TextDelta("thinking... ".into()),
                ModelEvent::TextDelta("<promise>COMPLETE</promise>".into()),
                ModelEvent::TextDelta(" extra".into()),
            ]],
            vec![],
        );

        let res = session
            .addr
            .ask(|reply| SessionMsg::Prompt {
                text: "go".into(),
                structured_output_tag: None,
                reply,
            })
            .unwrap()
            .unwrap();

        assert!(res.completed);
        assert!(!res.text.contains("extra"));

        session.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn structured_output_extracted() {
        let (instance, session, _model, _sb) = rig(
            vec![vec![
                ModelEvent::TextDelta("preamble ".into()),
                ModelEvent::TextDelta("<output>{\"x\":1}</output>".into()),
                ModelEvent::TextDelta(" trailing".into()),
                ModelEvent::Stop { reason: Some("end_turn".into()) },
            ]],
            vec![],
        );

        let res = session
            .addr
            .ask(|reply| SessionMsg::Prompt {
                text: "extract".into(),
                structured_output_tag: Some("output".into()),
                reply,
            })
            .unwrap()
            .unwrap();

        assert_eq!(res.structured.as_deref(), Some(&b"{\"x\":1}"[..]));

        session.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn shell_routes_through_instance() {
        let recorded_response = ShellResult {
            exit_code: 0,
            stdout: b"ok\n".to_vec(),
            stderr: Vec::new(),
        };
        let (instance, session, _model, _sb) = rig(vec![], vec![recorded_response]);

        let res = session
            .addr
            .ask(|reply| SessionMsg::Shell {
                cmd: "echo ok".into(),
                reply,
            })
            .unwrap()
            .unwrap();

        assert_eq!(res.exit_code, 0);
        assert_eq!(res.stdout, b"ok\n");

        session.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn graceful_shutdown_on_drop() {
        let (instance, session, _model, _sb) = rig(vec![], vec![]);
        session.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn tool_use_loop_executes_bash_and_continues() {
        // Turn 1: model asks to run `echo hi`.
        // Turn 2: model produces final text after seeing the tool result.
        let (instance, session, model, sandbox) = rig(
            vec![
                vec![
                    ModelEvent::ToolUseStart {
                        id: "toolu_1".into(),
                        name: "bash".into(),
                    },
                    ModelEvent::ToolUseInputDelta(r#"{"command":"echo hi"}"#.into()),
                    ModelEvent::BlockStop,
                    ModelEvent::Stop { reason: Some("tool_use".into()) },
                ],
                vec![
                    ModelEvent::TextDelta("ran it, output was hi".into()),
                    ModelEvent::Stop { reason: Some("end_turn".into()) },
                ],
            ],
            vec![ShellResult {
                exit_code: 0,
                stdout: b"hi\n".to_vec(),
                stderr: Vec::new(),
            }],
        );

        let res = session
            .addr
            .ask(|reply| SessionMsg::Prompt {
                text: "run echo hi".into(),
                structured_output_tag: None,
                reply,
            })
            .unwrap()
            .unwrap();

        assert_eq!(res.text, "ran it, output was hi");
        assert_eq!(res.turns, 2);

        // The sandbox saw the bash invocation.
        let recorded = sandbox.recorded.lock().unwrap();
        assert_eq!(*recorded, vec!["echo hi".to_string()]);

        // The second model request contained the tool_use + tool_result blocks.
        let seen = model.seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        let second = &seen[1];
        // user(prompt), assistant(tool_use), user(tool_result)
        assert_eq!(second.messages.len(), 3);
        assert!(matches!(
            &second.messages[1].content[0],
            ContentBlock::ToolUse { name, .. } if name == "bash"
        ));
        assert!(matches!(
            &second.messages[2].content[0],
            ContentBlock::ToolResult { is_error: false, .. }
        ));

        session.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn tool_loop_terminates_at_turn_limit() {
        // Model keeps requesting bash forever; harness must stop.
        let scripts: Vec<Vec<ModelEvent>> = (0..MAX_TURNS_PER_PROMPT + 2)
            .map(|_| {
                vec![
                    ModelEvent::ToolUseStart {
                        id: "toolu_x".into(),
                        name: "bash".into(),
                    },
                    ModelEvent::ToolUseInputDelta(r#"{"command":"true"}"#.into()),
                    ModelEvent::BlockStop,
                    ModelEvent::Stop { reason: Some("tool_use".into()) },
                ]
            })
            .collect();
        let responses: Vec<ShellResult> = (0..MAX_TURNS_PER_PROMPT + 2)
            .map(|_| ShellResult {
                exit_code: 0,
                stdout: Vec::new(),
                stderr: Vec::new(),
            })
            .collect();

        let (instance, session, _model, _sb) = rig(scripts, responses);

        let res = session
            .addr
            .ask(|reply| SessionMsg::Prompt {
                text: "loop".into(),
                structured_output_tag: None,
                reply,
            })
            .unwrap();
        assert!(matches!(res, Err(SessionError::TurnLimitExceeded)));

        session.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn store_loads_history_at_construction_and_saves_after_prompt() {
        use crate::persist::test_store::InMemoryStore;

        let store = Arc::new(InMemoryStore::new());
        // Pre-seed: a prior conversation under name "resumed".
        store
            .save(
                "resumed",
                &[
                    ChatMessage::user("hello?"),
                    ChatMessage::assistant("earlier reply"),
                ],
            )
            .unwrap();

        let sandbox: Box<dyn crate::sandbox::Sandbox> =
            Box::new(MockSandbox::new(vec![]));
        let instance = spawn(Instance::new("inst", sandbox));
        let model = Arc::new(MockModel::single(vec![
            ModelEvent::TextDelta("new reply".into()),
            ModelEvent::Stop { reason: Some("end_turn".into()) },
        ]));
        let h = Arc::new(HarnessState::new(
            "h",
            model.clone() as Arc<dyn Model>,
            instance.addr.clone(),
        ));
        let session = spawn(Session::new("resumed", h).with_store(store.clone()));

        // Verify the model sees the resumed history when we send a new turn.
        let _ = session
            .addr
            .ask(|reply| SessionMsg::Prompt {
                text: "follow up".into(),
                structured_output_tag: None,
                reply,
            })
            .unwrap()
            .unwrap();

        let seen = model.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        // Resumed user, resumed assistant, new user — 3 messages.
        assert_eq!(seen[0].messages.len(), 3);
        assert_eq!(
            seen[0].messages[0].content,
            vec![ContentBlock::Text("hello?".into())]
        );

        // Store saw a save call after the prompt.
        assert!(*store.save_calls.lock().unwrap() >= 1);
        let persisted = store.data.lock().unwrap().get("resumed").cloned().unwrap();
        // resumed 2 + new user + new assistant = 4
        assert_eq!(persisted.len(), 4);

        session.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn invalid_tool_input_returns_error_block() {
        let (instance, session, model, _sb) = rig(
            vec![
                vec![
                    ModelEvent::ToolUseStart {
                        id: "toolu_1".into(),
                        name: "bash".into(),
                    },
                    ModelEvent::ToolUseInputDelta("not json".into()),
                    ModelEvent::BlockStop,
                    ModelEvent::Stop { reason: Some("tool_use".into()) },
                ],
                vec![
                    ModelEvent::TextDelta("ack".into()),
                    ModelEvent::Stop { reason: Some("end_turn".into()) },
                ],
            ],
            vec![],
        );

        let _ = session
            .addr
            .ask(|reply| SessionMsg::Prompt {
                text: "x".into(),
                structured_output_tag: None,
                reply,
            })
            .unwrap()
            .unwrap();

        let seen = model.seen.lock().unwrap();
        let second = &seen[1];
        assert!(matches!(
            &second.messages[2].content[0],
            ContentBlock::ToolResult { is_error: true, .. }
        ));

        session.join().unwrap();
        instance.join().unwrap();
    }

    // ----- streaming + cancellation ---------------------------------------

    #[test]
    fn prompt_stream_emits_text_deltas_then_done() {
        use std::sync::mpsc::sync_channel;

        let (instance, session, _model, _sb) = rig(
            vec![vec![
                ModelEvent::TextDelta("hel".into()),
                ModelEvent::TextDelta("lo".into()),
                ModelEvent::Stop { reason: Some("end_turn".into()) },
            ]],
            vec![],
        );

        let (tx, rx) = sync_channel::<StreamEvent>(256);
        session
            .addr
            .send(SessionMsg::PromptStream {
                text: "hi".into(),
                structured_output_tag: None,
                events: tx,
            })
            .unwrap();

        let mut text = String::new();
        let mut done: Option<PromptResult> = None;
        for ev in rx.iter() {
            match ev {
                StreamEvent::TextDelta(s) => text.push_str(&s),
                StreamEvent::Done(pr) => {
                    done = Some(pr);
                    break;
                }
                _ => {}
            }
        }

        assert_eq!(text, "hello");
        let pr = done.expect("Done event");
        assert_eq!(pr.text, "hello");
        assert_eq!(pr.turns, 1);

        session.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn prompt_stream_cancellation_on_receiver_drop() {
        use std::sync::mpsc::sync_channel;

        // Long script — caller will drop the receiver after the first delta.
        let (instance, session, _model, _sb) = rig(
            vec![vec![
                ModelEvent::TextDelta("first".into()),
                ModelEvent::TextDelta("second".into()),
                ModelEvent::TextDelta("third".into()),
                ModelEvent::Stop { reason: Some("end_turn".into()) },
            ]],
            vec![],
        );

        let (tx, rx) = sync_channel::<StreamEvent>(256);
        session
            .addr
            .send(SessionMsg::PromptStream {
                text: "go".into(),
                structured_output_tag: None,
                events: tx,
            })
            .unwrap();

        let first = rx.recv().unwrap();
        assert!(matches!(first, StreamEvent::TextDelta(ref s) if s == "first"));
        drop(rx); // signal cancellation

        // Session shouldn't deadlock; subsequent message should still get through.
        let echo: PromptResult = session
            .addr
            .ask(|reply| SessionMsg::Prompt {
                text: "afterwards".into(),
                structured_output_tag: None,
                reply,
            })
            .unwrap()
            .unwrap();
        assert_eq!(echo.turns, 1);

        session.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn prompt_stream_emits_tool_lifecycle_events() {
        use std::sync::mpsc::sync_channel;

        let (instance, session, _model, sandbox) = rig(
            vec![
                vec![
                    ModelEvent::ToolUseStart {
                        id: "toolu_1".into(),
                        name: "bash".into(),
                    },
                    ModelEvent::ToolUseInputDelta(r#"{"command":"echo hi"}"#.into()),
                    ModelEvent::BlockStop,
                    ModelEvent::Stop { reason: Some("tool_use".into()) },
                ],
                vec![
                    ModelEvent::TextDelta("done".into()),
                    ModelEvent::Stop { reason: Some("end_turn".into()) },
                ],
            ],
            vec![ShellResult {
                exit_code: 0,
                stdout: b"hi\n".to_vec(),
                stderr: Vec::new(),
            }],
        );

        let (tx, rx) = sync_channel::<StreamEvent>(256);
        session
            .addr
            .send(SessionMsg::PromptStream {
                text: "run".into(),
                structured_output_tag: None,
                events: tx,
            })
            .unwrap();

        let mut saw_tool_start = false;
        let mut saw_tool_input = false;
        let mut saw_tool_result = false;
        let mut saw_done = false;
        for ev in rx.iter() {
            match ev {
                StreamEvent::ToolUseStart { name, .. } if name == "bash" => saw_tool_start = true,
                StreamEvent::ToolUseInputDelta(_) => saw_tool_input = true,
                StreamEvent::ToolResult { content, is_error, .. } => {
                    saw_tool_result = true;
                    assert!(!is_error);
                    assert!(content.contains("hi"));
                }
                StreamEvent::Done(_) => {
                    saw_done = true;
                    break;
                }
                _ => {}
            }
        }
        assert!(saw_tool_start && saw_tool_input && saw_tool_result && saw_done);
        assert_eq!(sandbox.recorded.lock().unwrap().len(), 1);

        session.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn compactor_fires_before_each_turn_after_the_first() {
        // The callback receives the history just before the model is called.
        // It records what it saw and returns the input unchanged so the
        // iteration loop's other invariants stay verifiable.
        let calls: Arc<std::sync::Mutex<Vec<Vec<ChatMessage>>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let calls_clone = calls.clone();
        let compactor: Arc<crate::session::CompactFn> = Arc::new(move |msgs| {
            calls_clone.lock().unwrap().push(msgs.clone());
            msgs
        });

        let sandbox: Box<dyn crate::sandbox::Sandbox> =
            Box::new(MockSandbox::new(vec![]));
        let inst = spawn(Instance::new("ci", sandbox));
        let model = Arc::new(MockModel::new(vec![
            vec![
                ModelEvent::TextDelta("first".into()),
                ModelEvent::Stop { reason: Some("end_turn".into()) },
            ],
            vec![
                ModelEvent::TextDelta("second".into()),
                ModelEvent::Stop { reason: Some("end_turn".into()) },
            ],
        ]));
        let state = Arc::new(
            HarnessState::new("h", model as Arc<dyn Model>, inst.addr.clone())
                .with_compactor(compactor),
        );
        let sess = spawn(Session::new("s", state));

        let _ = sess
            .addr
            .ask(|reply| SessionMsg::Prompt {
                text: "one".into(),
                structured_output_tag: None,
                reply,
            })
            .unwrap()
            .unwrap();
        let _ = sess
            .addr
            .ask(|reply| SessionMsg::Prompt {
                text: "two".into(),
                structured_output_tag: None,
                reply,
            })
            .unwrap()
            .unwrap();

        // Two prompts → two turns → compactor invoked twice.
        let seen = calls.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].len(), 1);     // just `one`
        assert_eq!(seen[1].len(), 3);     // user(one), assistant(first), user(two)

        sess.join().unwrap();
        inst.join().unwrap();
    }
}
