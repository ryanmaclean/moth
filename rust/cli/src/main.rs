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
use std::sync::mpsc::sync_channel;

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
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return usage_and_exit(2);
    }
    match args.remove(0).as_str() {
        "run" => run_cmd(args),
        "serve" => serve_cmd(args),
        "-h" | "--help" | "help" => usage_and_exit(0),
        other => {
            eprintln!("unknown subcommand: {other}");
            usage_and_exit(2)
        }
    }
}

fn usage_and_exit(code: u8) -> ExitCode {
    let m = "usage:\n  \
        agent run [opts] <prompt>\n  \
        agent serve [opts]\n\n\
        opts:\n  \
        --openai                     use OpenAI-compatible provider\n  \
        --model NAME                 model id\n  \
        --skill NAME                 load .agents/skills/NAME.md\n  \
        --arg KEY=VAL                substitute {{KEY}} in the skill (repeatable)\n  \
        --sessions DIR               persist message history to DIR (or SESSIONS_DIR env)\n  \
        --mcp 'CMD ARGS'             spawn MCP server, register its tools (repeatable)\n  \
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
}

fn parse_common(args: &mut Vec<String>) -> CommonOpts {
    let mut opts = CommonOpts::default();
    if let Ok(d) = std::env::var("SESSIONS_DIR") {
        opts.sessions_dir = Some(PathBuf::from(d));
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

    let sandbox: Box<dyn Sandbox> = Box::new(AuditedShell::new(vshell::VShell::new()));
    let inst = spawn(Instance::new("cli", sandbox));
    let state = HarnessState::new("default", model, inst.addr.clone()).with_tools(tools);
    let mut sess_build = Session::new("default", Arc::new(state));
    if let Some(s) = store {
        sess_build = sess_build.with_store(s);
    }
    let sess = spawn(sess_build);

    let (tx, rx) = sync_channel(1);
    if let Err(e) = sess.addr.send(SessionMsg::Prompt {
        text: prompt,
        structured_output_tag: None,
        reply: tx,
    }) {
        eprintln!("send failed: {e}");
        drop(mcp_clients);
        if let (Some(s), Some(ws)) = (strategy, workspace) {
            let _ = s.finish(ws, AgentStatus::Failure(e.to_string()));
        }
        return ExitCode::from(1);
    }

    let (exit, agent_status) = match rx.recv() {
        Ok(Ok(pr)) => {
            print!("{}", pr.text);
            if !pr.text.ends_with('\n') {
                println!();
            }
            if pr.completed {
                eprintln!("[completion signal fired after {} turn(s)]", pr.turns);
            } else if pr.turns > 1 {
                eprintln!("[{} turn(s)]", pr.turns);
            }
            (ExitCode::SUCCESS, AgentStatus::Success)
        }
        Ok(Err(e)) => {
            eprintln!("session error: {e:?}");
            (ExitCode::from(1), AgentStatus::Failure(format!("{e:?}")))
        }
        Err(_) => {
            eprintln!("session dropped reply channel");
            (
                ExitCode::from(1),
                AgentStatus::Failure("session dropped reply channel".into()),
            )
        }
    };

    let _ = sess.join();
    let _ = inst.join();
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
        Box::new(ChatHandler { model, tools: Arc::new(tools), store }),
    );

    eprintln!("serving on {addr}");
    let result = srv.serve_with(&addr, ServerConfig::default());
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
}

impl AgentHandler for ChatHandler {
    fn handle(&self, id: &str, body: &[u8], sink: &mut EventSink) -> Result<(), HandlerError> {
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

        let (tx, rx) = sync_channel(1);
        sess.addr
            .send(SessionMsg::Prompt {
                text: prompt,
                structured_output_tag: None,
                reply: tx,
            })
            .map_err(|e| HandlerError(format!("session send: {e}")))?;

        let result = rx.recv().map_err(|_| HandlerError("session dropped".into()))?;
        let _ = sess.join();
        let _ = inst.join();

        match result {
            Ok(pr) => {
                let _ = sink.emit(Some("text"), &pr.text);
                let _ = sink.emit(Some("done"), &format!("turns={}", pr.turns));
                Ok(())
            }
            Err(e) => Err(HandlerError(format!("{e:?}"))),
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
