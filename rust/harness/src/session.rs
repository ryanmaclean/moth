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
use std::sync::mpsc::{SyncSender, sync_channel};

use actor::{Actor, ActorRef};
use wire::find_tag;

use crate::instance::InstanceMsg;
use crate::model::{
    ChatMessage, ContentBlock, Model, ModelEvent, ModelRequest, Role, ToolDef,
};
use crate::sandbox::{SandboxError, ShellResult};

const DEFAULT_MAX_TOKENS: u32 = 4096;
const MAX_TURNS_PER_PROMPT: usize = 16;

pub struct HarnessState {
    pub name: String,
    pub model: Arc<dyn Model>,
    pub default_system: Option<String>,
    pub default_max_tokens: u32,
    pub instance: ActorRef<InstanceMsg>,
    pub tools: Vec<ToolDef>,
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
            tools: vec![bash_tool()],
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.default_system = Some(system.into());
        self
    }

    pub fn with_tools(mut self, tools: Vec<ToolDef>) -> Self {
        self.tools = tools;
        self
    }
}

pub fn bash_tool() -> ToolDef {
    ToolDef {
        name: "bash".into(),
        description: "Execute a bash command in the agent's sandbox. Returns stdout, stderr, and exit code.".into(),
        input_schema: r#"{"type":"object","properties":{"command":{"type":"string","description":"The shell command to execute."}},"required":["command"]}"#.into(),
    }
}

pub struct Session {
    pub name: String,
    pub history: Vec<ChatMessage>,
    pub harness: Arc<HarnessState>,
}

impl Session {
    pub fn new(name: impl Into<String>, harness: Arc<HarnessState>) -> Self {
        Self { name: name.into(), history: Vec::new(), harness }
    }
}

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
    Shell {
        cmd: String,
        reply: SyncSender<Result<ShellResult, SandboxError>>,
    },
}

impl Actor for Session {
    type Msg = SessionMsg;

    fn handle(&mut self, msg: SessionMsg) {
        match msg {
            SessionMsg::Prompt { text, structured_output_tag, reply } => {
                let _ = reply.send(self.run_prompt(text, structured_output_tag));
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
        self.history.push(ChatMessage::user(text));

        let mut response_text = String::new();
        let mut completed = false;
        let mut turns = 0;

        loop {
            turns += 1;
            if turns > MAX_TURNS_PER_PROMPT {
                return Err(SessionError::TurnLimitExceeded);
            }

            let req = ModelRequest {
                system: self.harness.default_system.clone(),
                messages: self.history.clone(),
                max_tokens: self.harness.default_max_tokens,
                tools: self.harness.tools.clone(),
            };

            let mut blocks: Vec<ContentBlock> = Vec::new();
            let mut current_text = String::new();
            let mut current_tool: Option<(String, String, String)> = None;
            let mut stop_reason: Option<String> = None;

            for ev in self.harness.model.stream(req) {
                match ev {
                    Ok(ModelEvent::TextDelta(s)) => {
                        current_text.push_str(&s);
                        response_text.push_str(&s);
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
                        current_tool = Some((id, name, String::new()));
                    }
                    Ok(ModelEvent::ToolUseInputDelta(s)) => {
                        if let Some((_, _, input)) = current_tool.as_mut() {
                            input.push_str(&s);
                        }
                    }
                    Ok(ModelEvent::BlockStop) => {
                        if let Some((id, name, input)) = current_tool.take() {
                            blocks.push(ContentBlock::ToolUse { id, name, input });
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

            // Stop conditions: completion signal, no tool calls, or explicit
            // end_turn from the model.
            if completed
                || tool_uses.is_empty()
                || stop_reason.as_deref() == Some("end_turn")
            {
                break;
            }

            // Execute each tool, build a user message of tool_results.
            let mut results: Vec<ContentBlock> = Vec::with_capacity(tool_uses.len());
            for (id, name, input) in tool_uses {
                results.push(self.execute_tool(&id, &name, &input));
            }
            self.history.push(ChatMessage { role: Role::User, content: results });
        }

        let structured = structured_output_tag
            .as_deref()
            .and_then(|tag| find_tag(response_text.as_bytes(), tag.as_bytes()).map(<[u8]>::to_vec));

        Ok(PromptResult { text: response_text, structured, completed, turns })
    }

    fn execute_tool(&self, id: &str, name: &str, input: &str) -> ContentBlock {
        if name != "bash" {
            return ContentBlock::ToolResult {
                tool_use_id: id.into(),
                content: format!("unknown tool: {name}"),
                is_error: true,
            };
        }
        let cmd = match extract_command(input) {
            Some(c) => c,
            None => {
                return ContentBlock::ToolResult {
                    tool_use_id: id.into(),
                    content: format!("missing or invalid 'command' field in tool input: {input}"),
                    is_error: true,
                };
            }
        };
        let (tx, rx) = sync_channel(1);
        if self
            .harness
            .instance
            .send(InstanceMsg::Shell { cmd, reply: tx })
            .is_err()
        {
            return ContentBlock::ToolResult {
                tool_use_id: id.into(),
                content: "instance mailbox closed".into(),
                is_error: true,
            };
        }
        match rx.recv() {
            Ok(Ok(r)) => {
                let mut content = String::new();
                if !r.stdout.is_empty() {
                    content.push_str(&String::from_utf8_lossy(&r.stdout));
                }
                if !r.stderr.is_empty() {
                    if !content.is_empty() && !content.ends_with('\n') {
                        content.push('\n');
                    }
                    content.push_str("stderr: ");
                    content.push_str(&String::from_utf8_lossy(&r.stderr));
                }
                if r.exit_code != 0 {
                    if !content.is_empty() && !content.ends_with('\n') {
                        content.push('\n');
                    }
                    content.push_str(&format!("(exit {})", r.exit_code));
                }
                ContentBlock::ToolResult {
                    tool_use_id: id.into(),
                    content,
                    is_error: r.exit_code != 0,
                }
            }
            Ok(Err(e)) => ContentBlock::ToolResult {
                tool_use_id: id.into(),
                content: e.0,
                is_error: true,
            },
            Err(_) => ContentBlock::ToolResult {
                tool_use_id: id.into(),
                content: "instance dropped reply channel".into(),
                is_error: true,
            },
        }
    }
}

/// Parse `{"command":"..."}` and return the command. Tolerant of whitespace
/// and ordering. Returns None if the input isn't valid JSON or lacks the
/// field. Uses the anthropic crate's JSON walker — same parser we already
/// vendored for streaming responses.
fn extract_command(input: &str) -> Option<String> {
    let parsed = anthropic::json::parse(input.as_bytes()).ok()?;
    match parsed.get("command")? {
        anthropic::json::Json::Str(s) => Some(s.clone()),
        _ => None,
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
}
