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

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::mpsc::sync_channel;

use actor::spawn;
use harness::{
    AnthropicModel, AuditedShell, BashTool, HarnessState, Instance, Model, OpenAiModel,
    Sandbox, Session, SessionMsg, Tool,
};
use server::{AgentHandler, EventSink, HandlerError, Server};

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
        agent run [--openai] [--model NAME] [--skill NAME] [--arg KEY=VALUE]... <prompt>\n  \
        agent serve [--addr 0.0.0.0:3583] [--openai] [--model NAME]";
    eprintln!("{m}");
    ExitCode::from(code)
}

#[derive(Default)]
struct CommonOpts {
    openai: bool,
    model: Option<String>,
}

fn parse_common(args: &mut Vec<String>) -> CommonOpts {
    let mut opts = CommonOpts::default();
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
            _ => i += 1,
        }
    }
    opts
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
            _ => i += 1,
        }
    }

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

    let sandbox: Box<dyn Sandbox> = Box::new(AuditedShell::new(vshell::VShell::new()));
    let inst = spawn(Instance::new("cli", sandbox));
    let state = HarnessState::new("default", model, inst.addr.clone())
        .with_tools(default_tools(None));
    let sess = spawn(Session::new("default", Arc::new(state)));

    let (tx, rx) = sync_channel(1);
    if let Err(e) = sess.addr.send(SessionMsg::Prompt {
        text: prompt,
        structured_output_tag: None,
        reply: tx,
    }) {
        eprintln!("send failed: {e}");
        return ExitCode::from(1);
    }

    let exit = match rx.recv() {
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
            ExitCode::SUCCESS
        }
        Ok(Err(e)) => {
            eprintln!("session error: {e:?}");
            ExitCode::from(1)
        }
        Err(_) => {
            eprintln!("session dropped reply channel");
            ExitCode::from(1)
        }
    };

    let _ = sess.join();
    let _ = inst.join();
    exit
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

    let mut srv = Server::new();
    srv.register("chat", Box::new(ChatHandler { model }));

    eprintln!("serving on {addr}");
    if let Err(e) = srv.serve(&addr) {
        eprintln!("serve failed: {e}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// One `ChatHandler` per server; one fresh Instance + Session per request.
/// We don't persist Sessions across requests in v1 — `<id>` is recorded
/// for log correlation but the in-memory state is per-connection.
struct ChatHandler {
    model: Arc<dyn Model>,
}

impl AgentHandler for ChatHandler {
    fn handle(&self, id: &str, body: &[u8], sink: &mut EventSink) -> Result<(), HandlerError> {
        let prompt = parse_prompt(body).map_err(HandlerError)?;
        let _ = sink.emit(Some("start"), id);

        let sandbox: Box<dyn Sandbox> = Box::new(AuditedShell::new(vshell::VShell::new()));
        let inst = spawn(Instance::new(id, sandbox));
        let state = HarnessState::new("chat", self.model.clone(), inst.addr.clone())
            .with_tools(default_tools(None));
        let sess = spawn(Session::new(id, Arc::new(state)));

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
