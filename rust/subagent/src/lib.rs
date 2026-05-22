//! Subagent: Flue-style `session.task(prompt, opts)`.
//!
//! A subagent is a focused one-shot child Session, detached from the parent's
//! message history but sharing the parent's sandbox/instance and tool registry.
//! Used for parallel research, focused refactors, etc.
//!
//! The subagent IS a Session — same actor, same iteration loop. The only
//! difference from a regular Session is what we hand it at construction:
//!
//! - **Shared** (cloned from parent's `HarnessState`):
//!     - `model: Arc<dyn Model>` — same provider, same defaults.
//!     - `instance: ActorRef<InstanceMsg>` — same sandbox/filesystem.
//!     - `tools: Vec<Arc<dyn Tool>>` — same tool registry.
//!     - `default_max_tokens` — inherited.
//! - **Fresh**:
//!     - `history: Vec<ChatMessage>` — empty.
//!     - `default_system` — overridden by `role_system` if provided,
//!       otherwise inherited.
//! - **Not shared**:
//!     - Parent's history. The subagent never sees it.
//!     - Parent's `default_system` if `role_system: Some(_)`.
//!
//! Because the `instance` ActorRef is cloned (not the Instance itself), any
//! tool call the subagent issues routes through the *same* Instance mailbox
//! the parent uses — so shell commands serialise correctly across both, and
//! the subagent sees the parent's filesystem state.
//!
//! Lifecycle: spawn a worker thread that
//! 1. builds the child `HarnessState` + `Session`,
//! 2. `actor::spawn`s the Session onto its own actor thread,
//! 3. sends one `SessionMsg::Prompt` (or `PromptStream`),
//! 4. waits for the reply, joins the Session, returns the result.
//!
//! The returned `TaskHandle` wraps the worker thread's `JoinHandle`.

use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::thread::{self, JoinHandle};

use actor::spawn;
use harness::{
    HarnessState, PromptResult, Session, SessionError, SessionMsg, StreamEvent,
};

/// Handle to a spawned subagent. Call [`TaskHandle::join`] to block until the
/// subagent finishes its single prompt and return its [`PromptResult`].
#[must_use = "dropping a TaskHandle detaches the subagent's worker thread"]
pub struct TaskHandle {
    handle: JoinHandle<Result<PromptResult, SessionError>>,
}

impl TaskHandle {
    /// Block until the subagent finishes. Returns the final `PromptResult`
    /// or a `SessionError`. Panics propagated from the worker thread are
    /// surfaced as `SessionError::Mailbox` (the actor's mailbox closed
    /// without a reply).
    pub fn join(self) -> Result<PromptResult, SessionError> {
        match self.handle.join() {
            Ok(res) => res,
            Err(_) => Err(SessionError::Mailbox),
        }
    }
}

/// Build a child `HarnessState` cloning every field from `parent` except
/// `default_system`, which is overridden by `role_system` when `Some`.
fn child_state(
    parent: &Arc<HarnessState>,
    role_system: Option<String>,
) -> Arc<HarnessState> {
    let default_system = match role_system {
        Some(s) => Some(s),
        None => parent.default_system.clone(),
    };
    Arc::new(HarnessState {
        name: parent.name.clone(),
        model: parent.model.clone(),
        default_system,
        default_max_tokens: parent.default_max_tokens,
        instance: parent.instance.clone(),
        tools: parent.tools.clone(),
    })
}

/// Spawn a subagent. The returned `TaskHandle` resolves when the subagent's
/// single `Prompt` turn completes.
///
/// The subagent:
/// - shares `parent_state.instance` (same sandbox),
/// - shares `parent_state.model` and `parent_state.tools`,
/// - starts with an empty message history,
/// - uses `role_system` as its system prompt if `Some`, otherwise inherits
///   `parent_state.default_system`.
pub fn spawn_task(
    parent_state: Arc<HarnessState>,
    prompt: String,
    role_system: Option<String>,
) -> TaskHandle {
    let handle = thread::spawn(move || {
        let harness = child_state(&parent_state, role_system);
        let session = spawn(Session::new("subagent", harness));
        let result = session
            .addr
            .ask(|reply| SessionMsg::Prompt {
                text: prompt,
                structured_output_tag: None,
                reply,
            })
            .map_err(|_| SessionError::Mailbox)?;
        // Drop the Spawned so the actor thread exits cleanly. We don't
        // surface its join error — the prompt result is the contract.
        let _ = session.join();
        result
    });
    TaskHandle { handle }
}

/// Streaming variant. Same construction as [`spawn_task`] but the subagent's
/// turn is driven via `SessionMsg::PromptStream`; live events flow to `events`
/// in wire order, with `StreamEvent::Done` (or `Error`/`Cancelled`) as the
/// terminal event.
///
/// The returned `JoinHandle<()>` resolves when the subagent's actor thread
/// has exited. Callers typically just consume `events` and let the handle
/// detach, but joining is available for clean shutdown.
pub fn spawn_task_streaming(
    parent_state: Arc<HarnessState>,
    prompt: String,
    role_system: Option<String>,
    events: Sender<StreamEvent>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let harness = child_state(&parent_state, role_system);
        let session = spawn(Session::new("subagent", harness));
        // Best-effort send: if it fails the actor stays idle and we fall
        // through to join. The streaming consumer will see no events.
        let _ = session.addr.send(SessionMsg::PromptStream {
            text: prompt,
            structured_output_tag: None,
            events,
        });
        // Drop the local ActorRef + wait for the actor thread to drain.
        let _ = session.join();
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    use actor::Spawned;
    use harness::{
        ContentBlock, Instance, InstanceMsg, MockModel, MockSandbox, Model, ModelEvent,
        Role, Sandbox, SandboxError, Session, ShellResult,
    };

    /// Thin wrapper letting the test rig peek at recorded commands while the
    /// Instance owns a `Box<dyn Sandbox>` — same trick `session.rs` uses.
    struct SandboxRef(Arc<MockSandbox>);
    impl Sandbox for SandboxRef {
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

    /// Build a parent HarnessState backed by a MockModel + MockSandbox. The
    /// returned `Spawned<InstanceMsg>` must be joined at end of test.
    fn parent_rig(
        scripts: Vec<Vec<ModelEvent>>,
        responses: Vec<ShellResult>,
    ) -> (Spawned<InstanceMsg>, Arc<HarnessState>, Arc<MockModel>, Arc<MockSandbox>) {
        let sandbox = Arc::new(MockSandbox::new(responses));
        let instance = spawn(Instance::new(
            "inst-parent",
            Box::new(SandboxRef(sandbox.clone())) as Box<dyn Sandbox>,
        ));
        let model = Arc::new(MockModel::new(scripts));
        let state = Arc::new(HarnessState::new(
            "parent",
            model.clone() as Arc<dyn Model>,
            instance.addr.clone(),
        ));
        (instance, state, model, sandbox)
    }

    #[test]
    fn spawn_task_returns_prompt_result() {
        let (instance, state, _model, _sb) = parent_rig(
            vec![vec![
                ModelEvent::TextDelta("done".into()),
                ModelEvent::Stop { reason: Some("end_turn".into()) },
            ]],
            vec![],
        );

        let task = spawn_task(state, "do thing".into(), None);
        let res = task.join().unwrap();

        assert_eq!(res.text, "done");
        assert_eq!(res.turns, 1);
        assert!(!res.completed);

        instance.join().unwrap();
    }

    #[test]
    fn subagent_history_does_not_pollute_parent_session() {
        // Shared MockModel between the parent Session and the subagent.
        // After both run, `model.seen` contains BOTH calls — but parent's
        // own `Session.history` must NOT carry the subagent's messages.
        let (instance, state, model, _sb) = parent_rig(
            vec![
                // Parent's first prompt response.
                vec![
                    ModelEvent::TextDelta("parent reply".into()),
                    ModelEvent::Stop { reason: Some("end_turn".into()) },
                ],
                // Subagent's response.
                vec![
                    ModelEvent::TextDelta("child reply".into()),
                    ModelEvent::Stop { reason: Some("end_turn".into()) },
                ],
                // Parent's second prompt — should NOT see "child reply" in history.
                vec![
                    ModelEvent::TextDelta("parent again".into()),
                    ModelEvent::Stop { reason: Some("end_turn".into()) },
                ],
            ],
            vec![],
        );

        let parent_session = spawn(Session::new("parent-sess", state.clone()));
        // Turn 1 on parent.
        let _ = parent_session
            .addr
            .ask(|reply| SessionMsg::Prompt {
                text: "first".into(),
                structured_output_tag: None,
                reply,
            })
            .unwrap()
            .unwrap();

        // Subagent runs in between.
        let task = spawn_task(state.clone(), "child prompt".into(), None);
        let _child_res = task.join().unwrap();

        // Turn 2 on parent. The request the model sees must contain only the
        // parent's own messages — never the subagent's prompt or reply.
        let _ = parent_session
            .addr
            .ask(|reply| SessionMsg::Prompt {
                text: "second".into(),
                structured_output_tag: None,
                reply,
            })
            .unwrap()
            .unwrap();

        let seen = model.seen.lock().unwrap();
        // Three calls: parent#1, child, parent#2.
        assert_eq!(seen.len(), 3);

        // Subagent's request (seen[1]) was a fresh history of just the prompt.
        assert_eq!(seen[1].messages.len(), 1);
        assert_eq!(
            seen[1].messages[0].content,
            vec![ContentBlock::Text("child prompt".into())],
        );

        // Parent#2's request must contain ONLY parent#1 transcript + "second".
        // Specifically: user("first"), assistant("parent reply"), user("second").
        let p2 = &seen[2];
        assert_eq!(p2.messages.len(), 3);
        assert_eq!(p2.messages[0].role, Role::User);
        assert_eq!(
            p2.messages[0].content,
            vec![ContentBlock::Text("first".into())],
        );
        assert_eq!(p2.messages[1].role, Role::Assistant);
        assert_eq!(
            p2.messages[1].content,
            vec![ContentBlock::Text("parent reply".into())],
        );
        assert_eq!(p2.messages[2].role, Role::User);
        assert_eq!(
            p2.messages[2].content,
            vec![ContentBlock::Text("second".into())],
        );
        // And — critically — no "child prompt" or "child reply" anywhere.
        for m in &p2.messages {
            for c in &m.content {
                if let ContentBlock::Text(t) = c {
                    assert!(
                        !t.contains("child"),
                        "parent history leaked subagent content: {t:?}",
                    );
                }
            }
        }
        drop(seen);

        parent_session.join().unwrap();
        // Drop the test's residual Arc<HarnessState> so the Instance actor's
        // last sender is released and `instance.join()` can return.
        drop(state);
        instance.join().unwrap();
    }

    #[test]
    fn role_system_override_is_passed_to_model() {
        let (instance, state, model, _sb) = parent_rig(
            vec![vec![
                ModelEvent::TextDelta("ok".into()),
                ModelEvent::Stop { reason: Some("end_turn".into()) },
            ]],
            vec![],
        );

        let task = spawn_task(
            state,
            "task".into(),
            Some("you are a triage bot".into()),
        );
        task.join().unwrap();

        let seen = model.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].system.as_deref(), Some("you are a triage bot"));

        instance.join().unwrap();
    }

    #[test]
    fn no_role_system_inherits_parent_default_system() {
        // Build state in a sub-scope so every Arc<HarnessState> + cloned
        // Instance ActorRef drops before we try to `instance.join()`.
        // (Shadowing a `let with_sys = ...; let with_sys = ...;` keeps the
        // first binding alive until the enclosing fn ends, which silently
        // blocks the join.)
        let model: Arc<MockModel> = Arc::new(MockModel::single(vec![
            ModelEvent::TextDelta("ok".into()),
            ModelEvent::Stop { reason: Some("end_turn".into()) },
        ]));
        let (instance, state_no_sys, _model, _sb) = parent_rig(vec![], vec![]);
        let with_sys = Arc::new(HarnessState {
            name: state_no_sys.name.clone(),
            model: model.clone() as Arc<dyn Model>,
            default_system: Some("parent system".into()),
            default_max_tokens: state_no_sys.default_max_tokens,
            instance: state_no_sys.instance.clone(),
            tools: state_no_sys.tools.clone(),
        });
        drop(state_no_sys); // release its Instance ActorRef clone.

        let task = spawn_task(with_sys, "task".into(), None);
        task.join().unwrap();

        let seen = model.seen.lock().unwrap();
        assert_eq!(seen[0].system.as_deref(), Some("parent system"));
        drop(seen);

        instance.join().unwrap();
    }

    #[test]
    fn role_system_does_not_leak_back_to_parent_state() {
        // After spawning a subagent with a role override, the parent's
        // `HarnessState.default_system` is unchanged (HarnessState is Arc'd
        // immutably — but we re-verify by inspecting the model the parent
        // would see if it ran).
        let (instance, state_no_sys, _model, _sb) = parent_rig(vec![], vec![]);
        let parent_state = Arc::new(HarnessState {
            name: state_no_sys.name.clone(),
            model: state_no_sys.model.clone(),
            default_system: Some("parent system".into()),
            default_max_tokens: state_no_sys.default_max_tokens,
            instance: state_no_sys.instance.clone(),
            tools: state_no_sys.tools.clone(),
        });

        // Spawn a sub with an override.
        let sub_model = Arc::new(MockModel::single(vec![
            ModelEvent::TextDelta("ok".into()),
            ModelEvent::Stop { reason: Some("end_turn".into()) },
        ]));
        let sub_state = Arc::new(HarnessState {
            name: parent_state.name.clone(),
            model: sub_model.clone() as Arc<dyn Model>,
            default_system: parent_state.default_system.clone(),
            default_max_tokens: parent_state.default_max_tokens,
            instance: parent_state.instance.clone(),
            tools: parent_state.tools.clone(),
        });
        let task = spawn_task(sub_state, "x".into(), Some("ROLE".into()));
        task.join().unwrap();

        // Parent's default_system is still "parent system".
        assert_eq!(parent_state.default_system.as_deref(), Some("parent system"));

        // Drop leftover Arcs holding clones of the Instance ActorRef so
        // `instance.join()` can return.
        drop(state_no_sys);
        drop(parent_state);
        instance.join().unwrap();
    }

    #[test]
    fn tool_calls_in_subagent_route_through_parent_instance() {
        // Sub asks for `bash`, gets a result, then finishes.
        let scripts = vec![
            vec![
                ModelEvent::ToolUseStart {
                    id: "toolu_1".into(),
                    name: "bash".into(),
                },
                ModelEvent::ToolUseInputDelta(r#"{"command":"echo sub"}"#.into()),
                ModelEvent::BlockStop,
                ModelEvent::Stop { reason: Some("tool_use".into()) },
            ],
            vec![
                ModelEvent::TextDelta("did it".into()),
                ModelEvent::Stop { reason: Some("end_turn".into()) },
            ],
        ];
        let responses = vec![ShellResult {
            exit_code: 0,
            stdout: b"sub\n".to_vec(),
            stderr: Vec::new(),
        }];
        let (instance, state, _model, sandbox) = parent_rig(scripts, responses);

        let task = spawn_task(state, "run echo sub".into(), None);
        let res = task.join().unwrap();
        assert_eq!(res.text, "did it");

        // The parent's MockSandbox recorded the subagent's shell call.
        let recorded = sandbox.recorded.lock().unwrap();
        assert_eq!(*recorded, vec!["echo sub".to_string()]);

        instance.join().unwrap();
    }

    #[test]
    fn spawn_task_streaming_emits_text_then_done() {
        let (instance, state, _model, _sb) = parent_rig(
            vec![vec![
                ModelEvent::TextDelta("strea".into()),
                ModelEvent::TextDelta("med".into()),
                ModelEvent::Stop { reason: Some("end_turn".into()) },
            ]],
            vec![],
        );

        let (tx, rx) = channel::<StreamEvent>();
        let h = spawn_task_streaming(state, "go".into(), None, tx);

        let mut text = String::new();
        let mut done = None;
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
        assert_eq!(text, "streamed");
        let pr = done.expect("Done event");
        assert_eq!(pr.text, "streamed");

        h.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn spawn_task_streaming_emits_tool_events() {
        let scripts = vec![
            vec![
                ModelEvent::ToolUseStart {
                    id: "t1".into(),
                    name: "bash".into(),
                },
                ModelEvent::ToolUseInputDelta(r#"{"command":"echo sub"}"#.into()),
                ModelEvent::BlockStop,
                ModelEvent::Stop { reason: Some("tool_use".into()) },
            ],
            vec![
                ModelEvent::TextDelta("ok".into()),
                ModelEvent::Stop { reason: Some("end_turn".into()) },
            ],
        ];
        let responses = vec![ShellResult {
            exit_code: 0,
            stdout: b"sub\n".to_vec(),
            stderr: Vec::new(),
        }];
        let (instance, state, _model, _sb) = parent_rig(scripts, responses);

        let (tx, rx) = channel::<StreamEvent>();
        let h = spawn_task_streaming(state, "go".into(), None, tx);

        let mut saw_start = false;
        let mut saw_input = false;
        let mut saw_result = false;
        let mut saw_done = false;
        for ev in rx.iter() {
            match ev {
                StreamEvent::ToolUseStart { name, .. } if name == "bash" => saw_start = true,
                StreamEvent::ToolUseInputDelta(_) => saw_input = true,
                StreamEvent::ToolResult { is_error, .. } => {
                    assert!(!is_error);
                    saw_result = true;
                }
                StreamEvent::Done(_) => {
                    saw_done = true;
                    break;
                }
                _ => {}
            }
        }
        assert!(saw_start && saw_input && saw_result && saw_done);

        h.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn three_subagents_in_parallel_each_get_their_own_result() {
        // Each subagent gets its own Model + Instance so they really run
        // concurrently with no cross-talk. (Sharing a MockModel would
        // serialise on its Mutex and not test parallelism.)
        fn rig_with_text(text: &str) -> (Spawned<InstanceMsg>, Arc<HarnessState>) {
            let sandbox = Arc::new(MockSandbox::new(vec![]));
            let instance = spawn(Instance::new(
                "inst",
                Box::new(SandboxRef(sandbox)) as Box<dyn Sandbox>,
            ));
            let model = Arc::new(MockModel::single(vec![
                ModelEvent::TextDelta(text.into()),
                ModelEvent::Stop { reason: Some("end_turn".into()) },
            ]));
            let state = Arc::new(HarnessState::new(
                "p",
                model as Arc<dyn Model>,
                instance.addr.clone(),
            ));
            (instance, state)
        }

        let (i1, s1) = rig_with_text("alpha");
        let (i2, s2) = rig_with_text("beta");
        let (i3, s3) = rig_with_text("gamma");

        let t1 = spawn_task(s1, "p1".into(), None);
        let t2 = spawn_task(s2, "p2".into(), None);
        let t3 = spawn_task(s3, "p3".into(), None);

        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();
        let r3 = t3.join().unwrap();
        assert_eq!(r1.text, "alpha");
        assert_eq!(r2.text, "beta");
        assert_eq!(r3.text, "gamma");

        i1.join().unwrap();
        i2.join().unwrap();
        i3.join().unwrap();
    }

    #[test]
    fn failure_in_subagent_returns_err() {
        // Exhaust the MockModel's scripts so we hit the turn-limit branch:
        // the script asks for a tool_use forever; harness caps at MAX_TURNS.
        let mut scripts = Vec::new();
        for _ in 0..20 {
            scripts.push(vec![
                ModelEvent::ToolUseStart {
                    id: "t".into(),
                    name: "bash".into(),
                },
                ModelEvent::ToolUseInputDelta(r#"{"command":"true"}"#.into()),
                ModelEvent::BlockStop,
                ModelEvent::Stop { reason: Some("tool_use".into()) },
            ]);
        }
        let responses: Vec<ShellResult> = (0..20)
            .map(|_| ShellResult { exit_code: 0, stdout: Vec::new(), stderr: Vec::new() })
            .collect();

        let (instance, state, _model, _sb) = parent_rig(scripts, responses);

        let task = spawn_task(state, "loop forever".into(), None);
        let err = task.join().unwrap_err();
        assert!(matches!(err, SessionError::TurnLimitExceeded));

        instance.join().unwrap();
    }

    #[test]
    fn streaming_subagent_emits_error_on_failure() {
        // Same setup as above but via the streaming API — we should receive
        // a final `StreamEvent::Error(TurnLimitExceeded)`.
        let mut scripts = Vec::new();
        for _ in 0..20 {
            scripts.push(vec![
                ModelEvent::ToolUseStart {
                    id: "t".into(),
                    name: "bash".into(),
                },
                ModelEvent::ToolUseInputDelta(r#"{"command":"true"}"#.into()),
                ModelEvent::BlockStop,
                ModelEvent::Stop { reason: Some("tool_use".into()) },
            ]);
        }
        let responses: Vec<ShellResult> = (0..20)
            .map(|_| ShellResult { exit_code: 0, stdout: Vec::new(), stderr: Vec::new() })
            .collect();
        let (instance, state, _model, _sb) = parent_rig(scripts, responses);

        let (tx, rx) = channel::<StreamEvent>();
        let h = spawn_task_streaming(state, "loop forever".into(), None, tx);

        let mut got_err = false;
        for ev in rx.iter() {
            if let StreamEvent::Error(SessionError::TurnLimitExceeded) = ev {
                got_err = true;
                break;
            }
        }
        assert!(got_err, "expected a terminal Error event");

        h.join().unwrap();
        instance.join().unwrap();
    }

    #[test]
    fn subagent_shares_tools_with_parent() {
        // Cloning Arc<dyn Tool> means both parent and child point at the
        // same tool instance. We assert it by checking object identity on
        // the cloned Vec.
        let (instance, state, _model, _sb) = parent_rig(vec![], vec![]);
        let parent_tool_ptrs: Vec<*const ()> = state
            .tools
            .iter()
            .map(|t| Arc::as_ptr(t) as *const ())
            .collect();

        let child = child_state(&state, None);
        let child_tool_ptrs: Vec<*const ()> = child
            .tools
            .iter()
            .map(|t| Arc::as_ptr(t) as *const ())
            .collect();

        assert_eq!(parent_tool_ptrs, child_tool_ptrs);
        drop(state);
        drop(child);
        instance.join().unwrap();
    }

    #[test]
    fn subagent_shares_instance_addr() {
        // Two child states from the same parent share the same Instance
        // ActorRef, which both will send to.
        let (instance, state, _model, sandbox) = parent_rig(
            vec![
                // Two parallel single-tool prompts — but we'll only run one
                // streaming child; the other branch is unused for this test.
                vec![
                    ModelEvent::ToolUseStart {
                        id: "t1".into(),
                        name: "bash".into(),
                    },
                    ModelEvent::ToolUseInputDelta(r#"{"command":"a"}"#.into()),
                    ModelEvent::BlockStop,
                    ModelEvent::Stop { reason: Some("tool_use".into()) },
                ],
                vec![
                    ModelEvent::TextDelta("a done".into()),
                    ModelEvent::Stop { reason: Some("end_turn".into()) },
                ],
            ],
            vec![ShellResult {
                exit_code: 0,
                stdout: b"a\n".to_vec(),
                stderr: Vec::new(),
            }],
        );

        let task = spawn_task(state.clone(), "run".into(), None);
        task.join().unwrap();

        let recorded = sandbox.recorded.lock().unwrap();
        assert_eq!(*recorded, vec!["a".to_string()]);
        drop(recorded);

        drop(state);
        instance.join().unwrap();
    }

    #[test]
    fn subagent_inherits_max_tokens() {
        let (instance, state_no_sys, _m, _sb) = parent_rig(
            vec![vec![
                ModelEvent::TextDelta("ok".into()),
                ModelEvent::Stop { reason: Some("end_turn".into()) },
            ]],
            vec![],
        );
        // Build a parent state with a non-default max_tokens.
        let model = Arc::new(MockModel::single(vec![
            ModelEvent::TextDelta("ok".into()),
            ModelEvent::Stop { reason: Some("end_turn".into()) },
        ]));
        let parent = Arc::new(HarnessState {
            name: state_no_sys.name.clone(),
            model: model.clone() as Arc<dyn Model>,
            default_system: None,
            default_max_tokens: 12345,
            instance: state_no_sys.instance.clone(),
            tools: state_no_sys.tools.clone(),
        });

        let task = spawn_task(parent, "go".into(), None);
        task.join().unwrap();

        let seen = model.seen.lock().unwrap();
        assert_eq!(seen[0].max_tokens, 12345);

        drop(seen);
        drop(state_no_sys);
        instance.join().unwrap();
    }

    #[test]
    fn subagent_empty_history_at_start() {
        // First (and only) request the model sees from a fresh subagent
        // contains exactly one message: the prompt.
        let (instance, state, model, _sb) = parent_rig(
            vec![vec![
                ModelEvent::TextDelta("ok".into()),
                ModelEvent::Stop { reason: Some("end_turn".into()) },
            ]],
            vec![],
        );

        let task = spawn_task(state, "the only prompt".into(), None);
        task.join().unwrap();

        let seen = model.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].messages.len(), 1);
        assert_eq!(seen[0].messages[0].role, Role::User);
        assert_eq!(
            seen[0].messages[0].content,
            vec![ContentBlock::Text("the only prompt".into())],
        );

        instance.join().unwrap();
    }
}
