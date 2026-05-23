//! Agent CLI.
//!
//! `agent run [opts] <prompt>` — one-shot. Routes the prompt through a
//! Session with bash + read_file + write_file + edit_file tools, the
//! AuditedShell decorator, and the chosen model provider. Streams the
//! final response to stdout.
//!
//! `agent serve [opts]` — long-running HTTP/1.1 + SSE server. Each
//! POST /agents/chat/<id> opens a Session per id (in-memory) and
//! streams model events back as SSE frames.
//!
//! Env:
//!   ANTHROPIC_API_KEY     anthropic provider (default)
//!   OPENAI_API_KEY        openai provider
//!   MODEL                 model name; default depends on provider
//!   OPENAI_BASE_URL       override OpenAI base URL (OpenRouter, LM Studio, etc)
//!   AGENTS_ROOT           dir for .agents/skills/<name>.md + .agents/roles/<name>.md
//!                         (defaults to current dir)
//!   SESSIONS_DIR          enable file-backed session persistence at this dir
//!                         (or pass --sessions DIR)

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Process-wide cancellation flag set by SIGINT. The `run` event loop
/// checks it every event; flipping it causes the CLI to drop the
/// streaming receiver, which propagates as a cancellation through the
/// session iteration loop.
static CANCEL: AtomicBool = AtomicBool::new(false);
/// Set by SIGTERM (`agent serve`). The server loop polls this; when true
/// it stops accepting new connections, drains in-flight, and returns.
pub(crate) static SHUTDOWN: AtomicBool = AtomicBool::new(false);

extern "C" fn sigint_handler(_: libc::c_int) {
    CANCEL.store(true, Ordering::SeqCst);
}

extern "C" fn sigterm_handler(_: libc::c_int) {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

/// Install signal handlers. SIGINT (Ctrl-C) flips `CANCEL` for in-flight
/// prompts; SIGTERM (K8s preStop, supervisord, systemd) flips `SHUTDOWN`
/// so the server drains gracefully. Both are one-shot — a second signal
/// hits the default handler and force-quits.
fn install_signals() {
    unsafe {
        libc::signal(libc::SIGINT, sigint_handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, sigterm_handler as *const () as libc::sighandler_t);
    }
}

use actor::spawn;
use git::{AgentStatus, Branch, BranchStrategy, HeadStrategy, MergeToHeadStrategy};
use harness::{
    AnthropicModel, AuditedShell, BashTool, HarnessState, Instance, Model, OpenAiModel,
    Sandbox, Session, SessionMsg, SessionStore, Tool,
};
use server::{AgentHandler, EventSink, HandlerError, Server, ServerConfig};

const DEFAULT_ANTHROPIC_MODEL: &str = "claude-haiku-4-5";
const DEFAULT_OPENAI_MODEL: &str = "gpt-4o-mini";

fn main() -> ExitCode {
    install_signals();
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return usage_and_exit(2);
    }
    match args.remove(0).as_str() {
        "run" => run_cmd(args),
        "serve" => serve_cmd(args),
        "mcp-serve" => mcp_serve_cmd(args),
        "-h" | "--help" | "help" => usage_and_exit(0),
        "-V" | "--version" | "version" => {
            println!("agent {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            usage_and_exit(2)
        }
    }
}

fn usage_and_exit(code: u8) -> ExitCode {
    let m = "usage:\n  \
        agent run [opts] <prompt>\n  \
        agent serve [opts]\n  \
        agent mcp-serve [opts]            speak MCP over stdio\n\n\
        opts:\n  \
        --openai                     use OpenAI-compatible provider\n  \
        --model NAME                 model id\n  \
        --skill NAME                 load .agents/skills/NAME.md\n  \
        --arg KEY=VAL                substitute {{KEY}} in the skill (repeatable)\n  \
        --sessions DIR               persist message history to DIR (or SESSIONS_DIR env)\n  \
        --mcp 'CMD ARGS'             spawn MCP server, register its tools (repeatable)\n  \
        --runlog DIR                 tee every StreamEvent to <DIR>/<run_id>.jsonl (or RUNLOG_DIR env)\n  \
        --task-tool                  expose a 'task' tool to the LLM (Flue-style session.task)\n  \
        --branch-strategy STRATEGY   [run] head | merge-to-head | branch:NAME\n  \
        --addr HOST:PORT             [serve] bind address (default 0.0.0.0:3583)";
    eprintln!("{m}");
    ExitCode::from(code)
}

#[derive(Default)]
struct CommonOpts {
    openai: bool,
    model: Option<String>,
    sessions_dir: Option<PathBuf>,
    mcp_specs: Vec<String>,
    runlog_dir: Option<PathBuf>,
    enable_task_tool: bool,
    compact_budget: Option<usize>,
}

fn parse_common(args: &mut Vec<String>) -> CommonOpts {
    let mut opts = CommonOpts::default();
    if let Ok(d) = std::env::var("SESSIONS_DIR") {
        opts.sessions_dir = Some(PathBuf::from(d));
    }
    if let Ok(d) = std::env::var("RUNLOG_DIR") {
        opts.runlog_dir = Some(PathBuf::from(d));
    }
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--openai" => {
                opts.openai = true;
                args.remove(i);
            }
            "--model" => {
                args.remove(i);
                if i < args.len() {
                    opts.model = Some(args.remove(i));
                }
            }
            "--sessions" => {
                args.remove(i);
                if i < args.len() {
                    opts.sessions_dir = Some(PathBuf::from(args.remove(i)));
                }
            }
            "--mcp" => {
                args.remove(i);
                if i < args.len() {
                    opts.mcp_specs.push(args.remove(i));
                }
            }
            "--runlog" => {
                args.remove(i);
                if i < args.len() {
                    opts.runlog_dir = Some(PathBuf::from(args.remove(i)));
                }
            }
            "--task-tool" => {
                opts.enable_task_tool = true;
                args.remove(i);
            }
            "--compact-budget" => {
                args.remove(i);
                if i < args.len() {
                    opts.compact_budget = args.remove(i).parse().ok();
                }
            }
            _ => i += 1,
        }
    }
    opts
}

type McpConn = (Vec<mcp::McpClient>, Vec<Arc<dyn Tool>>);

/// Returns owned clients so they live the duration of the run, plus the
/// flattened tool list to register with the harness.
fn connect_mcp(specs: &[String]) -> Result<McpConn, String> {
    let mut clients = Vec::new();
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
    for spec in specs {
        let parts: Vec<&str> = spec.split_whitespace().collect();
        let (cmd, args) = parts.split_first().ok_or("empty --mcp value")?;
        let client = mcp::McpClient::stdio(cmd, args)
            .map_err(|e| format!("mcp {cmd}: {e:?}"))?;
        for t in client.tools() {
            tools.push(Arc::new(t));
        }
        clients.push(client);
    }
    Ok((clients, tools))
}

fn open_store(dir: Option<&PathBuf>) -> Result<Option<Arc<dyn SessionStore>>, String> {
    let Some(d) = dir else { return Ok(None) };
    let store = persist::FileStore::open(d).map_err(|e| format!("sessions dir {d:?}: {e}"))?;
    Ok(Some(Arc::new(store)))
}

fn build_model(opts: &CommonOpts) -> Result<Arc<dyn Model>, String> {
    if opts.openai {
        let key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY required".to_string())?;
        let name = opts
            .model
            .clone()
            .or_else(|| std::env::var("MODEL").ok())
            .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.into());
        let mut m = OpenAiModel::new(key, name);
        if let Ok(url) = std::env::var("OPENAI_BASE_URL") {
            m = m.with_base_url(url);
        }
        Ok(Arc::new(m))
    } else {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY required".to_string())?;
        let name = opts
            .model
            .clone()
            .or_else(|| std::env::var("MODEL").ok())
            .unwrap_or_else(|| DEFAULT_ANTHROPIC_MODEL.into());
        Ok(Arc::new(AnthropicModel::new(key, name)))
    }
}

fn default_tools(root: Option<PathBuf>) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(BashTool),
        Arc::new(fstools::ReadTool { root: root.clone() }),
        Arc::new(fstools::WriteTool { root: root.clone() }),
        Arc::new(fstools::EditTool { root }),
    ]
}

/// LLM-facing tool that spawns a focused subagent and returns its final
/// text. Exposes Flue-style `session.task()` semantics through the model
/// interface so the LLM can fan out exploratory work to a child it
/// describes in a JSON `prompt` argument.
struct TaskTool {
    parent_state: std::sync::RwLock<Option<Arc<HarnessState>>>,
}

impl TaskTool {
    fn new() -> Self {
        Self { parent_state: std::sync::RwLock::new(None) }
    }
    fn set_state(&self, state: Arc<HarnessState>) {
        *self.parent_state.write().unwrap() = Some(state);
    }
}

impl Tool for TaskTool {
    fn name(&self) -> &str {
        "task"
    }
    fn description(&self) -> &str {
        "Run a focused subagent on a self-contained prompt. The child shares the parent's sandbox and tools but starts with empty conversation history. Returns the child's final answer."
    }
    fn input_schema(&self) -> &str {
        r#"{"type":"object","properties":{"prompt":{"type":"string","description":"The prompt to give the child agent."},"role":{"type":"string","description":"Optional system prompt for the child."}},"required":["prompt"]}"#
    }
    fn call(
        &self,
        input: &str,
        _ctx: &harness::ToolCtx,
    ) -> Result<String, harness::ToolError> {
        let v = anthropic::json::parse(input.as_bytes())
            .map_err(|e| harness::ToolError(format!("invalid task input: {e:?}")))?;
        let prompt = match v.get("prompt") {
            Some(anthropic::json::Json::Str(s)) => s.clone(),
            _ => return Err(harness::ToolError("task: missing 'prompt'".into())),
        };
        let role = match v.get("role") {
            Some(anthropic::json::Json::Str(s)) => Some(s.clone()),
            _ => None,
        };
        let parent = self
            .parent_state
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| harness::ToolError("task tool: parent state not bound".into()))?;
        let handle = subagent::spawn_task(parent, prompt, role);
        match handle.join() {
            Ok(pr) => Ok(pr.text),
            Err(e) => Err(harness::ToolError(format!("subagent: {e:?}"))),
        }
    }
}

/// Spawn a `RunLog` drainer if `--runlog DIR` was given, returning a
/// channel-tee handle that mirrors every StreamEvent into the log file
/// AND forwards to the caller's downstream receiver.
fn open_runlog(dir: Option<&PathBuf>, run_id: &str) -> Option<Arc<runlog::RunLog>> {
    let dir = dir?;
    match runlog::RunLog::open(dir, run_id) {
        Ok(r) => Some(Arc::new(r)),
        Err(e) => {
            eprintln!("runlog: open {dir:?} failed: {e:?}; continuing without log");
            None
        }
    }
}

fn run_cmd(mut args: Vec<String>) -> ExitCode {
    let common = parse_common(&mut args);

    let mut skill_name: Option<String> = None;
    let mut skill_args: HashMap<String, String> = HashMap::new();
    let mut strategy_spec: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--skill" => {
                args.remove(i);
                if i < args.len() {
                    skill_name = Some(args.remove(i));
                }
            }
            "--arg" => {
                args.remove(i);
                if i < args.len() {
                    let kv = args.remove(i);
                    if let Some((k, v)) = kv.split_once('=') {
                        skill_args.insert(k.to_string(), v.to_string());
                    } else {
                        eprintln!("--arg expects KEY=VALUE, got: {kv}");
                        return ExitCode::from(2);
                    }
                }
            }
            "--branch-strategy" => {
                args.remove(i);
                if i < args.len() {
                    strategy_spec = Some(args.remove(i));
                }
            }
            _ => i += 1,
        }
    }

    let strategy = match parse_strategy(strategy_spec.as_deref()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };

    let model = match build_model(&common) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };

    let prompt = match (skill_name.as_deref(), args.join(" ")) {
        (Some(name), _) => match render_skill(name, &skill_args) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("skill {name}: {e}");
                return ExitCode::from(2);
            }
        },
        (None, p) if !p.is_empty() => p,
        (None, _) => {
            return usage_and_exit(2);
        }
    };

    let (mcp_clients, mcp_tools) = match connect_mcp(&common.mcp_specs) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };
    let store = match open_store(common.sessions_dir.as_ref()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };

    // If a branch strategy was given, prepare the workspace and constrain
    // fstools + vshell to it. Caller's cwd is moved to the workspace path
    // for the duration of the run.
    let (workspace, strategy) = match strategy {
        Some(s) => {
            let repo_root = match std::env::current_dir() {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("cwd: {e}");
                    return ExitCode::from(1);
                }
            };
            match s.prepare(&repo_root) {
                Ok(ws) => {
                    if let Err(e) = std::env::set_current_dir(&ws.path) {
                        eprintln!("cd {ws_path:?}: {e}", ws_path = ws.path);
                        return ExitCode::from(1);
                    }
                    (Some(ws), Some(s))
                }
                Err(e) => {
                    eprintln!("branch strategy prepare: {e:?}");
                    return ExitCode::from(1);
                }
            }
        }
        None => (None, None),
    };

    let fstools_root = workspace.as_ref().map(|w| w.path.clone());
    let mut tools = default_tools(fstools_root);
    tools.extend(mcp_tools);

    // Optional `task` tool — the LLM can spawn subagents at will.
    let task_tool = if common.enable_task_tool {
        let t = Arc::new(TaskTool::new());
        tools.push(t.clone() as Arc<dyn Tool>);
        Some(t)
    } else {
        None
    };

    let sandbox: Box<dyn Sandbox> = Box::new(AuditedShell::new(vshell::VShell::new()));
    let inst = spawn(Instance::new("cli", sandbox));
    let metrics_client = Arc::new(
        metrics::Client::from_env()
            .with_prefix("agent")
            .with_tag("provider", if common.openai { "openai" } else { "anthropic" }),
    );
    let mut state_build = HarnessState::new("default", model.clone(), inst.addr.clone())
        .with_tools(tools)
        .with_metrics(metrics_client);
    if let Some(budget) = common.compact_budget {
        let compactor = compact::Compactor::new(model.clone(), budget);
        let cb: Arc<harness::CompactFn> = Arc::new(move |msgs| {
            compactor.maybe_compact(msgs.clone()).unwrap_or(msgs)
        });
        state_build = state_build.with_compactor(cb);
    }
    let state = Arc::new(state_build);
    if let Some(tt) = &task_tool {
        tt.set_state(state.clone());
    }
    let mut sess_build = Session::new("default", state.clone());
    if let Some(s) = store {
        sess_build = sess_build.with_store(s);
    }
    let sess = spawn(sess_build);

    // Optional run log: tee every StreamEvent into <dir>/<run_id>.jsonl.
    let run_id = format!("run-{}", unix_ms_now());
    let runlog_handle = open_runlog(common.runlog_dir.as_ref(), &run_id);
    let (log_tx, log_rx) = std::sync::mpsc::sync_channel::<harness::StreamEvent>(1024);
    let log_thread = runlog_handle.as_ref().map(|r| {
        let r = r.clone();
        std::thread::spawn(move || {
            let _ = r.drain(log_rx);
        })
    });

    // Use streaming so text deltas print to stdout as the model generates them.
    let (tx, rx) = std::sync::mpsc::sync_channel::<harness::StreamEvent>(256);
    if let Err(e) = sess.addr.send(SessionMsg::PromptStream {
        text: prompt,
        structured_output_tag: None,
        events: tx,
    }) {
        eprintln!("send failed: {e}");
        drop(mcp_clients);
        if let (Some(s), Some(ws)) = (strategy, workspace) {
            let _ = s.finish(ws, AgentStatus::Failure(e.to_string()));
        }
        return ExitCode::from(1);
    }

    use std::io::Write as _;
    let mut completed = false;
    let mut turns = 0usize;
    let mut last_was_newline = true;
    let (exit, agent_status) = loop {
        // SIGINT? Drop the receiver to signal cancellation to the session.
        if CANCEL.load(Ordering::SeqCst) {
            drop(rx);
            eprintln!("[cancelling on SIGINT]");
            break (
                ExitCode::from(130),
                AgentStatus::Failure("cancelled by signal".into()),
            );
        }
        let ev = match rx.recv() {
            Ok(ev) => ev,
            Err(_) => {
                eprintln!("session dropped event channel");
                break (
                    ExitCode::from(1),
                    AgentStatus::Failure("session dropped event channel".into()),
                );
            }
        };
        // Mirror to the run log first; channel-closed there is a warning,
        // not fatal — the run keeps going.
        let _ = log_tx.send(ev.clone());
        match ev {
            harness::StreamEvent::TextDelta(s) => {
                last_was_newline = s.ends_with('\n');
                print!("{s}");
                let _ = std::io::stdout().flush();
            }
            harness::StreamEvent::ToolUseStart { name, .. } => {
                if !last_was_newline {
                    println!();
                }
                eprintln!("[tool: {name}]");
                last_was_newline = true;
            }
            harness::StreamEvent::ToolResult { is_error, .. } => {
                if is_error {
                    eprintln!("[tool error]");
                }
            }
            harness::StreamEvent::TurnComplete { turn, .. } => {
                turns = turn;
            }
            harness::StreamEvent::Done(pr) => {
                if !last_was_newline {
                    println!();
                }
                completed = pr.completed;
                turns = pr.turns;
                break (ExitCode::SUCCESS, AgentStatus::Success);
            }
            harness::StreamEvent::Cancelled => {
                eprintln!("[cancelled]");
                break (
                    ExitCode::from(130),
                    AgentStatus::Failure("cancelled".into()),
                );
            }
            harness::StreamEvent::Error(e) => {
                eprintln!("session error: {e:?}");
                break (ExitCode::from(1), AgentStatus::Failure(format!("{e:?}")));
            }
            _ => {}
        }
    };
    if completed {
        eprintln!("[completion signal fired after {turns} turn(s)]");
    } else if turns > 1 {
        eprintln!("[{turns} turn(s)]");
    }

    let _ = sess.join();
    let _ = inst.join();
    // Close the runlog channel so the drainer thread exits.
    drop(log_tx);
    if let Some(t) = log_thread {
        let _ = t.join();
    }
    drop(mcp_clients);
    if let (Some(s), Some(ws)) = (strategy, workspace)
        && let Err(e) = s.finish(ws, agent_status)
    {
        eprintln!("branch strategy finish: {e:?}");
    }
    exit
}

fn parse_strategy(spec: Option<&str>) -> Result<Option<Box<dyn BranchStrategy>>, String> {
    let Some(s) = spec else { return Ok(None) };
    match s {
        "head" => Ok(Some(Box::new(HeadStrategy))),
        "merge-to-head" => Ok(Some(Box::new(MergeToHeadStrategy))),
        other if other.starts_with("branch:") => {
            let name = other[7..].trim();
            if name.is_empty() {
                return Err("--branch-strategy branch:<name> requires a name".into());
            }
            Ok(Some(Box::new(Branch { name: name.to_string() })))
        }
        other => Err(format!(
            "unknown --branch-strategy '{other}' (try: head | merge-to-head | branch:NAME)"
        )),
    }
}

fn render_skill(name: &str, args: &HashMap<String, String>) -> Result<String, String> {
    let root = std::env::var("AGENTS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let skill = tmpl::load_skill(&root, name).map_err(|e| format!("{e:?}"))?;
    let args_ref: HashMap<&str, String> = args.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    skill.render(&args_ref).map_err(|e| format!("{e:?}"))
}

fn serve_cmd(mut args: Vec<String>) -> ExitCode {
    let common = parse_common(&mut args);
    let mut addr: String = "0.0.0.0:3583".into();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--addr" => {
                args.remove(i);
                if i < args.len() {
                    addr = args.remove(i);
                }
            }
            _ => i += 1,
        }
    }
    if !args.is_empty() {
        eprintln!("unexpected args: {args:?}");
        return usage_and_exit(2);
    }

    let model = match build_model(&common) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };

    let (mcp_clients, mcp_tools) = match connect_mcp(&common.mcp_specs) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };
    let store = match open_store(common.sessions_dir.as_ref()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };

    let mut tools = default_tools(None);
    tools.extend(mcp_tools);

    let mut srv = Server::new();
    srv.register(
        "chat",
        Box::new(ChatHandler {
            model,
            tools: Arc::new(tools),
            store,
            runlog_dir: common.runlog_dir.clone(),
        }),
    );

    eprintln!("serving on {addr}");
    // Bind first so binding errors don't get mixed up with shutdown state.
    let listener = match std::net::TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("bind {addr}: {e}");
            drop(mcp_clients);
            return ExitCode::from(1);
        }
    };
    // Bridge the process-wide SHUTDOWN flag (set by SIGTERM) into the
    // server's typed Arc<AtomicBool>. The poll thread is cheap (one
    // 50ms tick) and exits when the server returns.
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    let bridge = std::thread::spawn(move || {
        while !shutdown_clone.load(Ordering::Acquire) {
            if SHUTDOWN.load(Ordering::Acquire) {
                shutdown_clone.store(true, Ordering::Release);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    });
    let result = srv.serve_listener_with_shutdown(
        listener,
        ServerConfig::default(),
        shutdown.clone(),
    );
    shutdown.store(true, Ordering::Release); // stop the bridge thread
    let _ = bridge.join();
    drop(mcp_clients);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("serve failed: {e}");
            ExitCode::from(1)
        }
    }
}

/// One `ChatHandler` per server; one fresh Instance + Session per request.
/// We don't persist Sessions across requests in v1 — `<id>` is recorded
/// for log correlation but the in-memory state is per-connection.
struct ChatHandler {
    model: Arc<dyn Model>,
    tools: Arc<Vec<Arc<dyn Tool>>>,
    store: Option<Arc<dyn SessionStore>>,
    /// Where per-request RunLog jsonl files land. `None` disables runlog.
    runlog_dir: Option<PathBuf>,
}

impl AgentHandler for ChatHandler {
    fn handle(
        &self,
        id: &str,
        request_id: &str,
        body: &[u8],
        sink: &mut EventSink,
    ) -> Result<(), HandlerError> {
        let prompt = parse_prompt(body).map_err(HandlerError)?;
        let _ = sink.emit(Some("start"), id);

        let sandbox: Box<dyn Sandbox> = Box::new(AuditedShell::new(vshell::VShell::new()));
        let inst = spawn(Instance::new(id, sandbox));
        let state = HarnessState::new("chat", self.model.clone(), inst.addr.clone())
            .with_tools((*self.tools).clone());
        let mut sess_build = Session::new(id, Arc::new(state));
        if let Some(s) = &self.store {
            sess_build = sess_build.with_store(s.clone());
        }
        let sess = spawn(sess_build);

        // Optional RunLog: use the per-request correlation key as the
        // `run_id` so the jsonl filename matches the `X-Request-ID`
        // surfaced on the response. Every emitted record carries the
        // same id in a dedicated field for log/metric joins.
        let runlog = self.runlog_dir.as_ref().and_then(|dir| {
            match runlog::RunLog::open_with_request_id(dir, request_id, request_id) {
                Ok(r) => Some(Arc::new(r)),
                Err(e) => {
                    eprintln!("runlog: open {dir:?} failed: {e:?}; continuing without log");
                    None
                }
            }
        });
        let (log_tx, log_rx) =
            std::sync::mpsc::sync_channel::<harness::StreamEvent>(1024);
        let log_thread = runlog.as_ref().map(|r| {
            let r = r.clone();
            std::thread::spawn(move || {
                let _ = r.drain(log_rx);
            })
        });

        // Streaming: forward each Session StreamEvent to the SSE client as
        // it arrives. Client-disconnect (sink.emit fails) propagates to the
        // session as a closed receiver, which triggers in-loop cancellation.
        let (tx, rx) = std::sync::mpsc::sync_channel::<harness::StreamEvent>(256);
        sess.addr
            .send(SessionMsg::PromptStream {
                text: prompt,
                structured_output_tag: None,
                events: tx,
            })
            .map_err(|e| HandlerError(format!("session send: {e}")))?;

        let mut final_err: Option<HandlerError> = None;
        for ev in rx.iter() {
            // Tee to the runlog drainer first. A closed/full channel is a
            // warning only; the request continues.
            let _ = log_tx.send(ev.clone());
            let write_result = match &ev {
                harness::StreamEvent::TextDelta(s) => sink.emit(Some("text_delta"), s),
                harness::StreamEvent::ToolUseStart { id: tid, name } => {
                    sink.emit(Some("tool_use_start"), &format!("{tid}\t{name}"))
                }
                harness::StreamEvent::ToolUseInputDelta(s) => {
                    sink.emit(Some("tool_use_input_delta"), s)
                }
                harness::StreamEvent::BlockStop => sink.emit(Some("block_stop"), ""),
                harness::StreamEvent::ToolResult { tool_use_id, content, is_error } => {
                    sink.emit(
                        Some("tool_result"),
                        &format!("{tool_use_id}\t{is_error}\t{content}"),
                    )
                }
                harness::StreamEvent::TurnComplete { turn, stop_reason } => sink.emit(
                    Some("turn_complete"),
                    &format!("{}\t{}", turn, stop_reason.as_deref().unwrap_or("")),
                ),
                harness::StreamEvent::Done(pr) => sink.emit(
                    Some("done"),
                    &format!("turns={}\tcompleted={}", pr.turns, pr.completed),
                ),
                harness::StreamEvent::Cancelled => sink.emit(Some("cancelled"), ""),
                harness::StreamEvent::Error(e) => {
                    final_err = Some(HandlerError(format!("{e:?}")));
                    Ok(())
                }
            };
            if write_result.is_err() {
                // Client disconnected; dropping rx will cancel the in-flight
                // turn on the next event boundary.
                break;
            }
            if matches!(&ev, harness::StreamEvent::Done(_) | harness::StreamEvent::Cancelled) {
                break;
            }
        }

        drop(rx); // signal cancellation to the still-running session if any
        let _ = sess.join();
        let _ = inst.join();
        drop(log_tx);
        if let Some(t) = log_thread {
            let _ = t.join();
        }

        match final_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Parse a request body. Either a raw string (text/plain), or a JSON object
/// `{"prompt":"..."}` (text starts with `{`).
fn parse_prompt(body: &[u8]) -> Result<String, String> {
    let s = std::str::from_utf8(body).map_err(|e| format!("body not utf-8: {e}"))?;
    let trimmed = s.trim();
    if trimmed.starts_with('{') {
        let parsed = anthropic::json::parse(trimmed.as_bytes())
            .map_err(|e| format!("body not valid JSON: {e:?}"))?;
        match parsed.get("prompt") {
            Some(anthropic::json::Json::Str(p)) => Ok(p.clone()),
            _ => Err("body missing 'prompt' string field".into()),
        }
    } else {
        Ok(trimmed.to_string())
    }
}

fn unix_ms_now() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// `agent mcp-serve` — speak MCP JSON-RPC over stdio so other agents can
/// call our tool registry (bash + read/write/edit + audit). Useful for
/// chaining agents: parent agent A connects to child agent B's mcp server.
fn mcp_serve_cmd(mut args: Vec<String>) -> ExitCode {
    let common = parse_common(&mut args);
    if !args.is_empty() {
        eprintln!("unexpected args: {args:?}");
        return usage_and_exit(2);
    }

    let (mcp_clients, mcp_tools) = match connect_mcp(&common.mcp_specs) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };
    let mut tools = default_tools(None);
    tools.extend(mcp_tools);

    let sandbox: Box<dyn Sandbox> = Box::new(AuditedShell::new(vshell::VShell::new()));
    let inst = spawn(Instance::new("mcp-serve", sandbox));

    let mut srv = mcp_server::Server::new("sandcastle", env!("CARGO_PKG_VERSION"), inst.addr.clone());
    for t in tools {
        srv.register(t);
    }

    eprintln!("mcp server: stdio (waiting for handshake)");
    let result = srv.serve_stdio();
    drop(mcp_clients);
    let _ = inst.join();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("mcp serve: {e:?}");
            ExitCode::from(1)
        }
    }
}
