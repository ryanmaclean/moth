//! Session actor: owns message history and drives the iteration loop.
//!
//! `HarnessState` is the passive "harness" in Flue's hierarchy — model
//! defaults plus the Instance handle. It lives behind `Arc` so multiple
//! sessions on one instance share it without cloning the model.

use std::sync::Arc;
use std::sync::mpsc::SyncSender;

use actor::{Actor, ActorRef};
use wire::find_tag;

use crate::instance::InstanceMsg;
use crate::model::{ChatMessage, Model, ModelEvent, ModelRequest, Role};
use crate::sandbox::{SandboxError, ShellResult};

const DEFAULT_MAX_TOKENS: u32 = 4096;

pub struct HarnessState {
    pub name: String,
    pub model: Arc<dyn Model>,
    pub default_system: Option<String>,
    pub default_max_tokens: u32,
    pub instance: ActorRef<InstanceMsg>,
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
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.default_system = Some(system.into());
        self
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
}

#[derive(Debug, Clone)]
pub enum SessionError {
    Model(String),
    Mailbox,
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
                // Forward to the Instance and let it reply directly to the caller.
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
        self.history.push(ChatMessage { role: Role::User, content: text });

        let req = ModelRequest {
            system: self.harness.default_system.clone(),
            messages: self.history.clone(),
            max_tokens: self.harness.default_max_tokens,
        };

        let mut response = String::new();
        let mut completed = false;
        let stream = self.harness.model.stream(req);

        for ev in stream {
            match ev {
                Ok(ModelEvent::TextDelta(s)) => {
                    response.push_str(&s);
                    if let Some(body) = find_tag(response.as_bytes(), b"promise")
                        && body == b"COMPLETE"
                    {
                        completed = true;
                        break;
                    }
                }
                // Tool-use, block-stop, and stop events don't drive text accumulation
                // in v0; they'll matter once we wire real tool calls in M3.
                Ok(_) => {}
                Err(e) => return Err(SessionError::Model(e.0)),
            }
        }

        self.history.push(ChatMessage {
            role: Role::Assistant,
            content: response.clone(),
        });

        let structured = structured_output_tag
            .as_deref()
            .and_then(|tag| find_tag(response.as_bytes(), tag.as_bytes()).map(<[u8]>::to_vec));

        Ok(PromptResult { text: response, structured, completed })
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
    ) {
        let sandbox = Box::new(MockSandbox::new(sandbox_responses));
        let instance = spawn(Instance::new("inst-1", sandbox));
        let model = Arc::new(MockModel::new(model_scripts));
        let harness = Arc::new(HarnessState::new(
            "harness-1",
            model.clone() as Arc<dyn Model>,
            instance.addr.clone(),
        ));
        let session = spawn(Session::new("sess-1", harness));
        (instance, session, model)
    }

    #[test]
    fn prompt_roundtrip() {
        let (instance, session, _model) = rig(
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
        assert!(!res.completed);
        assert!(res.structured.is_none());

        session.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn multi_turn_history_accumulates() {
        let (instance, session, model) = rig(
            vec![
                vec![ModelEvent::TextDelta("first reply".into())],
                vec![ModelEvent::TextDelta("second reply".into())],
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
        // Second request should see: user(one), assistant(first reply), user(two).
        let second = &seen[1];
        assert_eq!(second.messages.len(), 3);
        assert_eq!(second.messages[0].role, Role::User);
        assert_eq!(second.messages[0].content, "one");
        assert_eq!(second.messages[1].role, Role::Assistant);
        assert_eq!(second.messages[1].content, "first reply");
        assert_eq!(second.messages[2].role, Role::User);
        assert_eq!(second.messages[2].content, "two");

        session.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn completion_signal_marks_completed() {
        let (instance, session, _model) = rig(
            vec![vec![
                ModelEvent::TextDelta("thinking... ".into()),
                ModelEvent::TextDelta("<promise>COMPLETE</promise>".into()),
                // Trailing event ignored: loop should have stopped.
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
        let (instance, session, _model) = rig(
            vec![vec![
                ModelEvent::TextDelta("preamble ".into()),
                ModelEvent::TextDelta("<output>{\"x\":1}</output>".into()),
                ModelEvent::TextDelta(" trailing".into()),
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
        let (instance, session, _model) = rig(vec![], vec![recorded_response]);

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
        let (instance, session, _model) = rig(vec![], vec![]);
        // Dropping the Spawned values should join both threads without hanging.
        session.join().unwrap();
        instance.join().unwrap();
    }
}
