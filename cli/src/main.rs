//! Agent CLI.
//!
//! `agent run [opts] <prompt>` or `agent run [opts] --prompt-stdin` — one-shot.
//! Routes the prompt through a
//! Session with bash + read_file + write_file + edit_file tools, the
//! AuditedShell decorator, and the chosen model provider. Streams the
//! final response to stdout.
//!
//! `agent serve [opts]` — long-running HTTP/1.1 + SSE server. Each
//! POST /agents/chat/<id> opens a Session per id (in-memory) and
//! streams model events back as SSE frames.
//!
//! `agent doctor` — smoke-check provider config, env paths, and network
//! reachability before a real run. Prints a small report and exits with
//! 0 if a provider key is set and its host is reachable, 1 if no key is
//! set, 2 if a key is set but the network probe failed.
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

mod doctor;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
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

#[cfg(test)]
use actor::spawn;
use git::{AgentStatus, Branch, BranchStrategy, HeadStrategy, MergeToHeadStrategy};
use harness::{
    AnthropicModel, AuditedShell, BashTool, HarnessState, Instance, MockModel, Model, ModelEvent,
    OpenAiModel, Sandbox, Session, SessionMsg, SessionStore, Tool,
};
use server::{AgentHandler, EventSink, HandlerError, PrepareError, Server, ServerConfig};

const DEFAULT_ANTHROPIC_MODEL: &str = "claude-haiku-4-5";
const DEFAULT_OPENAI_MODEL: &str = "gpt-4o-mini";

fn main() -> ExitCode {
    let mut private_args = std::env::args_os().skip(1);
    let private_command = private_args.next();
    if private_command.as_deref()
        == Some(std::ffi::OsStr::new(vshell::ownership::PRIVATE_LAUNCH_COMMAND))
    {
        return ExitCode::from(vshell::ownership::private_launch(private_args.collect()) as u8);
    }
    if private_command.as_deref() == Some(std::ffi::OsStr::new(vshell::ownership::PRIVATE_COMMAND))
    {
        return ExitCode::from(vshell::ownership::private_main(private_args.collect()) as u8);
    }
    install_signals();
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return usage_and_exit(2);
    }
    match args.remove(0).as_str() {
        "run" => run_cmd(args),
        "serve" => serve_cmd(args),
        "mcp-serve" => mcp_serve_cmd(args),
        "doctor" => doctor::doctor_cmd(args),
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

/// Returns true if the arg list contains a help flag anywhere. Used by
/// each subcommand handler to intercept `--help`/`-h` before any other
/// parsing — otherwise the help flag would be consumed as a prompt
/// fragment (notably by `agent run`).
fn wants_help(args: &[String]) -> bool {
    args.iter().any(|a| a == "--help" || a == "-h")
}

fn usage_and_exit(code: u8) -> ExitCode {
    let m = "usage:\n  \
        agent run [opts] <prompt>          one-shot prompt; streams response to stdout\n    \
          example: agent run \"summarize README.md\"\n  \
        agent serve [opts]                 long-running HTTP/1.1 + SSE server\n    \
          example: agent serve --addr 0.0.0.0:3583\n  \
        agent mcp-serve [opts]             speak MCP over stdio\n    \
          example: agent mcp-serve\n  \
        agent doctor [--mcp 'CMD ARGS']    smoke-check config + network\n    \
          example: agent doctor\n\n\
        opts:\n  \
        --openai                     use OpenAI-compatible provider\n  \
        --model NAME                 model id\n  \
        --skill NAME                 load .agents/skills/NAME.md\n  \
        --arg KEY=VAL                substitute {{KEY}} in the skill (repeatable)\n  \
        --prompt-stdin               [run] read the complete prompt from stdin to EOF\n  \
        --sessions DIR               persist message history to DIR (or SESSIONS_DIR env)\n  \
        --mcp 'CMD ARGS'             spawn MCP server, register its tools (repeatable)\n  \
        --runlog DIR                 tee every StreamEvent to <DIR>/<run_id>.jsonl (or RUNLOG_DIR env)\n  \
        --run-id ID                  [run] external run identity (or AGENT_RUN_ID / BOP_RUN_ID env)\n  \
        --task-tool                  expose a 'task' tool to the LLM (Flue-style session.task)\n  \
        --metrics HOST:PORT          send DogStatsD metrics here (overrides DOGSTATSD_ADDR)\n  \
        --branch-strategy STRATEGY   [run] head | merge-to-head | branch:NAME\n  \
        --mock                       [run] stub model with a canned response; no API key required\n  \
        --mock-script PATH           [run] load scripted turns from a JSON file (implies --mock)\n  \
        --addr HOST:PORT             [serve] bind address (default 0.0.0.0:3583)\n\n\
        Run `agent <subcommand> --help` for subcommand-specific help.";
    eprintln!("{m}");
    ExitCode::from(code)
}

/// Detailed help for `agent run`. Printed on `agent run --help` / `-h`.
fn print_run_help() {
    let m = "agent run \u{2014} one-shot prompt; streams the model response to stdout.\n\n\
        usage:\n  \
        agent run [opts] <prompt>\n  \
        agent run [opts] --skill NAME [--arg KEY=VAL ...]\n\n\
        agent run [opts] --prompt-stdin\n\n\
        required:\n  \
        <prompt>                     free-text prompt, OR use --skill or --prompt-stdin\n\n\
        flags:\n  \
        --openai                     use OpenAI-compatible provider (default: Anthropic)\n  \
        --model NAME                 model id (defaults: claude-haiku-4-5 / gpt-4o-mini)\n  \
        --skill NAME                 load .agents/skills/NAME.md as the prompt\n  \
        --arg KEY=VAL                substitute {{KEY}} in the skill (repeatable)\n  \
        --prompt-stdin               read the complete UTF-8 prompt from stdin to EOF;\n                               \
                              cannot combine with <prompt>, --skill, or --arg\n  \
        --sessions DIR               persist message history to DIR\n  \
        --mcp 'CMD ARGS'             spawn MCP server, register its tools (repeatable)\n  \
        --runlog DIR                 audit to <DIR>/<run_id>.jsonl; relative DIR uses the\n                               \
                              caller's cwd before branch-strategy preparation\n  \
        --run-id ID                  run identity owned by the caller (e.g. BOP); names the\n                               \
                              runlog file. [A-Za-z0-9._:-], <=128 bytes. Default:\n                               \
                              run-<unix-ms>.\n  \
        --task-tool                  expose a 'task' tool so the LLM can spawn subagents\n  \
        --compact-budget N           auto-compact history when over N tokens\n  \
        --metrics HOST:PORT          send DogStatsD metrics here (overrides DOGSTATSD_ADDR)\n  \
        --branch-strategy STRATEGY   head | merge-to-head | branch:NAME\n  \
        --mock                       stub the model with a canned response; no API key\n                               \
                              required. Useful for CI, demos, and verifying the\n                               \
                              tool/sandbox wiring without burning quota.\n  \
        --mock-script PATH           load scripted turns from a JSON file (implies --mock).\n                               \
                              Format: {\"turns\":[[{\"type\":\"text_delta\",\"delta\":\"hi\"},\n                               \
                                       {\"type\":\"stop\",\"reason\":\"end_turn\"}]]}\n\n\
        examples:\n  \
        agent run \"explain this repo in 3 bullets\"\n  \
        agent run --openai --model gpt-4o-mini \"write a haiku\"\n  \
        agent run --skill review --arg PR=123\n  \
        agent run --prompt-stdin --runlog ./logs --run-id card-42.r1\n  \
        agent run --mock \"hello\"                  # no API key required\n  \
        agent run --mock-script ./script.json \"hi\"\n\n\
        env:\n  \
        ANTHROPIC_API_KEY            required for the default Anthropic provider\n  \
        OPENAI_API_KEY               required with --openai\n  \
        OPENAI_BASE_URL              override OpenAI base URL (LM Studio, Ollama, OpenRouter)\n  \
        MODEL                        model id (lower priority than --model)\n  \
        AGENTS_ROOT                  dir holding .agents/skills/<name>.md (default: cwd)\n  \
        SESSIONS_DIR                 same as --sessions\n  \
        RUNLOG_DIR                   same as --runlog\n  \
        AGENT_RUN_ID                 run id when BOP_RUN_ID is unset\n  \
        BOP_RUN_ID                   BOP dispatcher run id (above AGENT_RUN_ID)\n  \
        DOGSTATSD_ADDR               DogStatsD sink HOST:PORT (lower priority than --metrics)";
    eprintln!("{m}");
}

/// Detailed help for `agent serve`. Printed on `agent serve --help` / `-h`.
fn print_serve_help() {
    let m = "agent serve \u{2014} long-running HTTP/1.1 + SSE server.\n\n\
        Each POST /agents/chat/<id> opens a Session per id (in-memory) and\n\
        streams model events back as SSE frames.\n\n\
        usage:\n  \
        agent serve [opts]\n\n\
        flags:\n  \
        --addr HOST:PORT             bind address (default 0.0.0.0:3583)\n  \
        --openai                     use OpenAI-compatible provider (default: Anthropic)\n  \
        --model NAME                 model id\n  \
        --sessions DIR               persist message history to DIR\n  \
        --mcp 'CMD ARGS'             spawn MCP server, register its tools (repeatable)\n  \
        --runlog DIR                 audit each HTTP execution in a distinct <DIR>/http-*.jsonl\n  \
        --metrics HOST:PORT          send DogStatsD metrics here (overrides DOGSTATSD_ADDR)\n\n\
        examples:\n  \
        agent serve\n  \
        agent serve --addr 127.0.0.1:8080\n  \
        agent serve --openai --model gpt-4o-mini --runlog ./logs\n  \
        agent serve --metrics 127.0.0.1:8125\n\n\
        env:\n  \
        ANTHROPIC_API_KEY            required for the default Anthropic provider\n  \
        OPENAI_API_KEY               required with --openai\n  \
        OPENAI_BASE_URL              override OpenAI base URL\n  \
        MODEL                        model id (lower priority than --model)\n  \
        SESSIONS_DIR                 same as --sessions\n  \
        RUNLOG_DIR                   same as --runlog\n  \
        DOGSTATSD_ADDR               DogStatsD sink HOST:PORT (lower priority than --metrics)";
    eprintln!("{m}");
}

/// Detailed help for `agent mcp-serve`. Printed on `--help` / `-h`.
fn print_mcp_serve_help() {
    let m = "agent mcp-serve \u{2014} speak MCP JSON-RPC over stdio.\n\n\
        Exposes this agent's tool registry (bash + read/write/edit + audit,\n\
        plus any --mcp child tools) to a parent agent over stdin/stdout.\n\n\
        usage:\n  \
        agent mcp-serve [opts]\n\n\
        flags:\n  \
        --mcp 'CMD ARGS'             chain another MCP server's tools (repeatable)\n\n\
        examples:\n  \
        agent mcp-serve\n  \
        agent mcp-serve --mcp 'other-mcp-server --flag'\n\n\
        env:\n  \
        (none required \u{2014} no model is invoked; this is a tool server only)";
    eprintln!("{m}");
}

/// Print a friendly error when the provider's API key env var is missing.
/// `openai` selects which provider's hint to surface: Anthropic by default,
/// OpenAI when the user already passed `--openai`.
fn print_missing_key_error(openai: bool) {
    if openai {
        eprintln!(
            "error: OPENAI_API_KEY is not set.\n\n\
            Set it from your OpenAI account, then:\n    \
            export OPENAI_API_KEY=sk-...\n\n\
            Or point at a local OpenAI-compatible server (LM Studio, Ollama):\n    \
            export OPENAI_BASE_URL=http://localhost:1234/v1\n    \
            export OPENAI_API_KEY=dummy   # most local servers ignore the key\n\n\
            Or use Anthropic instead:\n    \
            agent run \"<prompt>\"\n    \
            # with ANTHROPIC_API_KEY set\n\n\
            See QUICKSTART.md for more."
        );
    } else {
        eprintln!(
            "error: ANTHROPIC_API_KEY is not set.\n\n\
            Set it from https://console.anthropic.com/settings/keys, then:\n    \
            export ANTHROPIC_API_KEY=sk-ant-...\n\n\
            Or use an OpenAI-compatible provider:\n    \
            agent run --openai --model gpt-4o-mini \"<prompt>\"\n    \
            # with OPENAI_API_KEY set, or OPENAI_BASE_URL pointing at LM Studio/Ollama\n\n\
            See QUICKSTART.md for more."
        );
    }
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
    /// Explicit DogStatsD sink address from `--metrics HOST:PORT`. Overrides
    /// the `DOGSTATSD_ADDR` env var; `None` falls back to env then disabled.
    metrics_addr: Option<String>,
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
            "--metrics" => {
                args.remove(i);
                if i < args.len() {
                    opts.metrics_addr = Some(args.remove(i));
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
fn connect_mcp(
    specs: &[String],
    registry: Option<&Arc<vshell::ownership::OwnerRegistry>>,
) -> Result<McpConn, String> {
    let mut clients = Vec::new();
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
    for spec in specs {
        let parts: Vec<&str> = spec.split_whitespace().collect();
        let (cmd, args) = parts.split_first().ok_or("empty --mcp value")?;
        let client = match registry {
            Some(registry) => mcp::McpClient::stdio_owned(registry.clone(), cmd, args),
            None => mcp::McpClient::stdio(cmd, args),
        }
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

/// Cap on the user prompt echoed back by `--mock` (without a script). 80
/// chars matches the documented behaviour; truncated input is suffixed
/// with `...` so the cut is visible in logs.
const MOCK_ECHO_LIMIT: usize = 80;

/// Truncate `s` to at most `MOCK_ECHO_LIMIT` Unicode chars (not bytes —
/// we don't want to slice mid-codepoint), appending `...` if anything
/// was dropped.
fn truncate_for_mock(s: &str) -> String {
    let mut out = String::new();
    let mut truncated = false;
    for (n, c) in s.chars().enumerate() {
        if n == MOCK_ECHO_LIMIT {
            truncated = true;
            break;
        }
        out.push(c);
    }
    if truncated {
        out.push_str("...");
    }
    out
}

/// Choose the normal capped mock echo or an exact-match debug fixture.
fn mock_echo_text(
    prompt: &str,
    expected: Option<&str>,
    debug_assertions: bool,
) -> Result<String, String> {
    match expected {
        None => Ok(truncate_for_mock(prompt)),
        Some(_) if !debug_assertions => {
            Err("MOTH_TEST_ECHO_FULL_PROMPT requires a debug build".into())
        }
        Some(expected) if expected == prompt => Ok(prompt.to_owned()),
        Some(_) => Err("MOTH_TEST_ECHO_FULL_PROMPT does not match stdin".into()),
    }
}

/// Test-only full echo is enabled solely when a debug mock run receives the
/// exact synthetic prompt supplied in MOTH_TEST_ECHO_FULL_PROMPT. Normal mock
/// output keeps its 80-character cap, and mismatches never print prompt bytes.
fn build_mock_canned(prompt: &str) -> Result<MockModel, String> {
    let expected = match std::env::var("MOTH_TEST_ECHO_FULL_PROMPT") {
        Ok(expected) => Some(expected),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err("MOTH_TEST_ECHO_FULL_PROMPT must be UTF-8".into());
        }
    };
    let echoed = mock_echo_text(prompt, expected.as_deref(), cfg!(debug_assertions))?;
    let echo = format!("[mock] received: {echoed}\n");
    Ok(MockModel::single(vec![
        ModelEvent::TextDelta(echo),
        ModelEvent::Stop { reason: Some("end_turn".into()) },
    ]))
}

#[cfg(test)]
mod mock_echo_fixture_tests {
    use super::{mock_echo_text, truncate_for_mock};

    #[test]
    fn default_echo_still_truncates() {
        let prompt = "x".repeat(160);
        assert!(mock_echo_text(&prompt, None, true).unwrap() == truncate_for_mock(&prompt));
    }

    #[test]
    fn full_echo_needs_an_exact_debug_fixture() {
        let prompt = format!("fixture 雪🙂\n{}\nend", "x".repeat(160));
        assert!(mock_echo_text(&prompt, Some(prompt.as_str()), true).unwrap() == prompt);
        assert!(mock_echo_text(&prompt, Some("different"), true).is_err());
        assert!(mock_echo_text(&prompt, Some(prompt.as_str()), false).is_err());
    }
}

/// Compute (line, col), both 1-indexed, for a byte offset into `src`.
/// `pos` is clamped to `src.len()`.
fn line_col(src: &str, pos: usize) -> (usize, usize) {
    let p = pos.min(src.len());
    let mut line = 1usize;
    let mut col = 1usize;
    for b in &src.as_bytes()[..p] {
        if *b == b'\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Extract the trailing `at <pos>` byte offset that `anthropic::json`
/// embeds in its error messages, if any. Returns `None` for errors that
/// don't carry a position.
fn extract_pos(msg: &str) -> Option<usize> {
    let mut last = None;
    for token in msg.split_whitespace() {
        if let Ok(n) = token.trim_end_matches(|c: char| !c.is_ascii_digit()).parse::<usize>() {
            last = Some(n);
        }
    }
    last
}

/// Load a mock script from disk. The file is JSON of the form:
///
/// ```text
/// {"turns":[[{"type":"text_delta","delta":"hi"},
///            {"type":"stop","reason":"end_turn"}]]}
/// ```
///
/// Supported event `type` values: `text_delta`, `tool_use_start`,
/// `tool_use_input_delta`, `block_stop`, `stop`. Errors are formatted as
/// `path:line:col: <message>` so editors can jump straight to the bad
/// token.
fn load_mock_script(path: &std::path::Path) -> Result<Vec<Vec<ModelEvent>>, String> {
    let raw =
        std::fs::read_to_string(path).map_err(|e| format!("{}: read: {e}", path.display()))?;
    let parsed = anthropic::json::parse(raw.as_bytes()).map_err(|e| {
        let msg = format!("{e}");
        let (line, col) = match extract_pos(&msg) {
            Some(p) => line_col(&raw, p),
            None => (1, 1),
        };
        format!("{}:{line}:{col}: {msg}", path.display())
    })?;
    let turns = match parsed.get("turns") {
        Some(anthropic::json::Json::Arr(t)) => t,
        _ => {
            return Err(format!("{}: top-level key 'turns' must be an array", path.display()));
        }
    };
    let mut out: Vec<Vec<ModelEvent>> = Vec::with_capacity(turns.len());
    for (ti, turn_json) in turns.iter().enumerate() {
        let events = match turn_json {
            anthropic::json::Json::Arr(e) => e,
            _ => {
                return Err(format!("{}: turns[{ti}] must be an array of events", path.display()));
            }
        };
        let mut turn = Vec::with_capacity(events.len());
        for (ei, ev) in events.iter().enumerate() {
            turn.push(
                parse_event(ev)
                    .map_err(|e| format!("{}: turns[{ti}][{ei}]: {e}", path.display()))?,
            );
        }
        out.push(turn);
    }
    Ok(out)
}

/// Decode one `{"type":...}` event object into a `ModelEvent`.
fn parse_event(j: &anthropic::json::Json) -> Result<ModelEvent, String> {
    let ty = j
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing 'type' string field".to_string())?;
    match ty {
        "text_delta" => {
            let s = j
                .get("delta")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "text_delta: missing 'delta' string".to_string())?;
            Ok(ModelEvent::TextDelta(s.to_string()))
        }
        "tool_use_start" => {
            let id = j
                .get("id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "tool_use_start: missing 'id'".to_string())?;
            let name = j
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "tool_use_start: missing 'name'".to_string())?;
            Ok(ModelEvent::ToolUseStart { id: id.to_string(), name: name.to_string() })
        }
        "tool_use_input_delta" => {
            let s = j
                .get("delta")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "tool_use_input_delta: missing 'delta'".to_string())?;
            Ok(ModelEvent::ToolUseInputDelta(s.to_string()))
        }
        "block_stop" => Ok(ModelEvent::BlockStop),
        "stop" => {
            let reason = j.get("reason").and_then(|v| v.as_str()).map(|s| s.to_string());
            Ok(ModelEvent::Stop { reason })
        }
        other => Err(format!("unknown event type: {other:?}")),
    }
}

/// Error returned by `build_model` when the provider's API key env var is
/// absent. Callers print the dedicated hint via `print_missing_key_error`
/// instead of treating it as a plain string error so the user gets a
/// formatted, actionable message rather than a single-line gripe.
struct MissingApiKey;

fn build_model(opts: &CommonOpts) -> Result<Arc<dyn Model>, MissingApiKey> {
    if opts.openai {
        let key = std::env::var("OPENAI_API_KEY").map_err(|_| MissingApiKey)?;
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
        let key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| MissingApiKey)?;
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
fn run_cancelled() -> bool {
    cancellation_requested(CANCEL.load(Ordering::Acquire), SHUTDOWN.load(Ordering::Acquire))
}

fn cancellation_requested(interrupt: bool, terminate: bool) -> bool {
    interrupt || terminate
}

// A broken diagnostic sink must not unwind past retained worker ownership.
macro_rules! lifecycle_diagnostic {
    ($($arg:tt)*) => {{
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            use std::io::Write as _;
            let _ = writeln!(std::io::stderr().lock(), $($arg)*);
        }));
    }};
}

struct TaskTool {
    parent_state: std::sync::RwLock<Option<Arc<HarnessState>>>,
    stop: actor::Cancellation,
}

impl TaskTool {
    fn new() -> Self {
        Self { parent_state: std::sync::RwLock::new(None), stop: actor::Cancellation::default() }
    }
    fn set_state(&self, state: Arc<HarnessState>) {
        *self.parent_state.write().unwrap_or_else(|e| e.into_inner()) = Some(state);
    }

    fn request_stop(&self) {
        self.stop.request();
    }

    /// Call after Session retirement: no tool call can acquire a new parent.
    fn clear_state(&self) {
        *self.parent_state.write().unwrap_or_else(|e| e.into_inner()) = None;
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
    fn call(&self, input: &str, _ctx: &harness::ToolCtx) -> Result<String, harness::ToolError> {
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
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .ok_or_else(|| harness::ToolError("task tool: parent state not bound".into()))?;
        if self.stop.is_requested() || run_cancelled() {
            return Err(harness::ToolError("task cancelled before admission".into()));
        }
        let mut handle = subagent::try_spawn_task(parent, prompt, role)
            .map_err(|error| harness::ToolError(format!("subagent startup: {error}")))?;
        loop {
            if self.stop.is_requested() || run_cancelled() {
                handle.request_stop();
            }
            match handle.try_join() {
                actor::JoinPoll::Pending => {
                    std::thread::sleep(std::time::Duration::from_millis(10))
                }
                actor::JoinPoll::Joined(Ok(Ok(pr))) => {
                    if self.stop.is_requested() || run_cancelled() {
                        return Err(harness::ToolError("task cancelled".into()));
                    }
                    return Ok(pr.text);
                }
                actor::JoinPoll::Joined(Ok(Err(e))) => {
                    return Err(harness::ToolError(format!("subagent: {e:?}")));
                }
                actor::JoinPoll::Joined(Err(_)) => {
                    return Err(harness::ToolError("subagent worker panicked".into()));
                }
                actor::JoinPoll::AlreadyJoined => {
                    return Err(harness::ToolError("subagent join already consumed".into()));
                }
            }
        }
    }
}

/// Resolve the run identity for `agent run`.
///
/// Precedence: `--run-id` > `BOP_RUN_ID` > `AGENT_RUN_ID` > minted
/// `run-<unix-ms>`. Empty env values are treated as unset. An explicit id
/// that fails [`runlog::validate_run_id`] is an error — never silently
/// replaced, since the caller owns the identity.
fn resolve_run_id(
    flag: Option<String>,
    agent_env: Option<String>,
    bop_env: Option<String>,
    now_ms: u128,
) -> Result<String, String> {
    let nonempty = |v: Option<String>| v.filter(|s| !s.is_empty());
    let (source, id) = if let Some(id) = flag {
        ("--run-id", id)
    } else if let Some(id) = nonempty(bop_env) {
        ("BOP_RUN_ID", id)
    } else if let Some(id) = nonempty(agent_env) {
        ("AGENT_RUN_ID", id)
    } else {
        return Ok(format!("run-{now_ms}"));
    };
    runlog::validate_run_id(&id).map_err(|e| format!("{source}: {e}"))?;
    Ok(id)
}

/// An explicitly requested audit log must open before model/tool work starts.
fn open_runlog(dir: Option<&PathBuf>, run_id: &str) -> Result<Option<Arc<runlog::RunLog>>, String> {
    dir.map(|dir| {
        runlog::RunLog::open(dir, run_id)
            .map(Arc::new)
            .map_err(|e| format!("runlog: open {dir:?} failed: {e}"))
    })
    .transpose()
}

fn anchor_runlog_dir(caller_cwd: &Path, requested: &Path) -> PathBuf {
    if requested.is_absolute() { requested.to_path_buf() } else { caller_cwd.join(requested) }
}

fn write_newline(out: &mut dyn std::io::Write) -> std::io::Result<()> {
    out.write_all(b"\n")?;
    out.flush()
}

/// Read one complete pipe payload before creating the runlog or model. There
/// is no Moth-specific byte ceiling: BOP currently has no matching prompt
/// size limit. Reserve each incoming chunk fallibly so an allocation failure
/// returns a diagnostic rather than relying on an unchecked Vec growth path.
/// A process-wide OOM or a provider's own context limit remains outside this
/// reader's guarantees.
fn read_prompt_stdin(mut input: impl std::io::Read) -> Result<String, String> {
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        let n = match input.read(&mut chunk) {
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(format!("--prompt-stdin: read failed: {e}")),
        };
        if n == 0 {
            break;
        }
        bytes.try_reserve(n).map_err(|e| format!("--prompt-stdin: allocation failed: {e}"))?;
        bytes.extend_from_slice(&chunk[..n]);
    }
    if bytes.is_empty() {
        return Err("--prompt-stdin: no prompt bytes before EOF".into());
    }
    String::from_utf8(bytes).map_err(|e| format!("--prompt-stdin: invalid UTF-8: {e}"))
}

#[cfg(test)]
mod prompt_stdin_reader_tests {
    use super::read_prompt_stdin;
    use std::io::{self, Read};

    #[test]
    fn whitespace_is_preserved_and_only_zero_bytes_are_empty() {
        assert_eq!(read_prompt_stdin(&b" \n\t"[..]).unwrap(), " \n\t");
        assert!(read_prompt_stdin(&b""[..]).unwrap_err().contains("no prompt bytes"));
    }

    struct ErrorAfterChunk(bool);
    impl Read for ErrorAfterChunk {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if !self.0 {
                self.0 = true;
                out[..4].copy_from_slice(b"part");
                Ok(4)
            } else {
                Err(io::Error::other("injected read failure"))
            }
        }
    }

    #[test]
    fn partial_prompt_followed_by_read_error_is_not_accepted() {
        let err = read_prompt_stdin(ErrorAfterChunk(false)).unwrap_err();
        assert!(err.contains("read failed") && err.contains("injected read failure"), "{err}");
    }
}

fn run_cmd(args: Vec<String>) -> ExitCode {
    let mut stdout = std::io::stdout();
    run_cmd_with_output(args, &mut stdout)
}

fn run_cmd_with_output(mut args: Vec<String>, out: &mut dyn std::io::Write) -> ExitCode {
    if wants_help(&args) {
        print_run_help();
        return ExitCode::SUCCESS;
    }
    let common = parse_common(&mut args);

    let mut skill_name: Option<String> = None;
    let mut skill_args: HashMap<String, String> = HashMap::new();
    let mut saw_skill_flag = false;
    let mut saw_skill_arg = false;
    let mut strategy_spec: Option<String> = None;
    // --mock / --mock-script. Either flag enables mock mode and skips the
    // API-key check; --mock-script additionally loads scripted turns from
    // disk in place of the canned echo response.
    let mut mock: bool = false;
    let mut mock_script_path: Option<PathBuf> = None;
    let mut run_id_flag: Option<String> = None;
    let mut prompt_stdin = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt-stdin" => {
                if prompt_stdin {
                    lifecycle_diagnostic!("--prompt-stdin may be passed only once");
                    return ExitCode::from(2);
                }
                prompt_stdin = true;
                args.remove(i);
            }
            "--skill" => {
                saw_skill_flag = true;
                args.remove(i);
                if i < args.len() {
                    skill_name = Some(args.remove(i));
                }
            }
            "--arg" => {
                saw_skill_arg = true;
                args.remove(i);
                if i < args.len() {
                    let kv = args.remove(i);
                    if let Some((k, v)) = kv.split_once('=') {
                        skill_args.insert(k.to_string(), v.to_string());
                    } else {
                        lifecycle_diagnostic!("--arg expects KEY=VALUE, got: {kv}");
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
            "--run-id" => {
                args.remove(i);
                if i < args.len() {
                    run_id_flag = Some(args.remove(i));
                } else {
                    lifecycle_diagnostic!("--run-id requires an ID argument");
                    return ExitCode::from(2);
                }
            }
            "--mock" => {
                mock = true;
                args.remove(i);
            }
            "--mock-script" => {
                args.remove(i);
                if i < args.len() {
                    mock = true;
                    mock_script_path = Some(PathBuf::from(args.remove(i)));
                } else {
                    lifecycle_diagnostic!("--mock-script requires a PATH argument");
                    return ExitCode::from(2);
                }
            }
            _ => i += 1,
        }
    }

    // Run identity belongs to the caller (BOP). Resolve it before any side
    // effects and refuse an unusable one rather than minting a substitute:
    // a retry must keep the same logical identity.
    let run_id = match resolve_run_id(
        run_id_flag,
        std::env::var("AGENT_RUN_ID").ok(),
        std::env::var("BOP_RUN_ID").ok(),
        unix_ms_now(),
    ) {
        Ok(id) => id,
        Err(e) => {
            lifecycle_diagnostic!("{e}");
            return ExitCode::from(2);
        }
    };

    let strategy = match parse_strategy(strategy_spec.as_deref()) {
        Ok(s) => s,
        Err(e) => {
            lifecycle_diagnostic!("{e}");
            return ExitCode::from(2);
        }
    };

    if prompt_stdin && (saw_skill_flag || saw_skill_arg || !args.is_empty()) {
        lifecycle_diagnostic!("--prompt-stdin cannot be combined with <prompt>, --skill, or --arg");
        return ExitCode::from(2);
    }
    let positional_prompt = args.join(" ");
    let prompt = if prompt_stdin {
        match read_prompt_stdin(std::io::stdin().lock()) {
            Ok(p) => p,
            Err(e) => {
                lifecycle_diagnostic!("{e}");
                return ExitCode::from(2);
            }
        }
    } else if let Some(name) = skill_name.as_deref() {
        match render_skill(name, &skill_args) {
            Ok(p) => p,
            Err(e) => {
                lifecycle_diagnostic!("skill {name}: {e}");
                return ExitCode::from(2);
            }
        }
    } else if !positional_prompt.is_empty() {
        positional_prompt
    } else {
        return usage_and_exit(2);
    };

    // Resolve the audit path against the caller's cwd and open it before
    // model, MCP, store, or branch-strategy side effects.
    let anchored_runlog_dir = match common.runlog_dir.as_ref() {
        Some(dir) => match std::env::current_dir() {
            Ok(cwd) => Some(anchor_runlog_dir(&cwd, dir)),
            Err(e) => {
                lifecycle_diagnostic!("runlog: cannot determine caller cwd: {e}");
                return ExitCode::from(1);
            }
        },
        None => None,
    };
    let runlog_handle = match open_runlog(anchored_runlog_dir.as_ref(), &run_id) {
        Ok(r) => r,
        Err(e) => {
            lifecycle_diagnostic!("{e}");
            return ExitCode::from(1);
        }
    };

    // Model build: --mock bypasses the API-key check entirely. With a
    // script path we load + parse here (so a bad script aborts before any
    // sandbox/MCP wiring happens); otherwise the canned echo turn is
    // synthesised from the user's prompt.
    let model: Arc<dyn Model> = if mock {
        let mm = match mock_script_path.as_deref() {
            Some(p) => match load_mock_script(p) {
                Ok(turns) => MockModel::new(turns),
                Err(e) => {
                    lifecycle_diagnostic!("{e}");
                    return ExitCode::from(2);
                }
            },
            None => match build_mock_canned(&prompt) {
                Ok(mm) => mm,
                Err(e) => {
                    lifecycle_diagnostic!("{e}");
                    return ExitCode::from(2);
                }
            },
        };
        Arc::new(mm)
    } else {
        match build_model(&common) {
            Ok(m) => m,
            Err(MissingApiKey) => {
                print_missing_key_error(common.openai);
                return ExitCode::from(2);
            }
        }
    };

    // Own the native registry before any MCP or shell child can be admitted.
    let owner_registry = match run_owner_registry() {
        Ok(registry) => registry,
        Err(error) => {
            lifecycle_diagnostic!("native owner registry: {error}");
            return ExitCode::from(1);
        }
    };
    let exit = 'owned_run: {
        let (mcp_clients, mcp_tools) = match connect_mcp(&common.mcp_specs, owner_registry.as_ref())
        {
            Ok(t) => t,
            Err(e) => {
                lifecycle_diagnostic!("{e}");
                break 'owned_run ExitCode::from(1);
            }
        };
        let store = match open_store(common.sessions_dir.as_ref()) {
            Ok(s) => s,
            Err(e) => {
                lifecycle_diagnostic!("{e}");
                break 'owned_run ExitCode::from(1);
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
                        lifecycle_diagnostic!("cwd: {e}");
                        break 'owned_run ExitCode::from(1);
                    }
                };
                match s.prepare(&repo_root) {
                    Ok(ws) => {
                        if let Err(e) = std::env::set_current_dir(&ws.path) {
                            lifecycle_diagnostic!("cd {ws_path:?}: {e}", ws_path = ws.path);
                            break 'owned_run ExitCode::from(1);
                        }
                        (Some(ws), Some(s))
                    }
                    Err(e) => {
                        lifecycle_diagnostic!("branch strategy prepare: {e:?}");
                        break 'owned_run ExitCode::from(1);
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

        let mut shell = vshell::VShell::new();
        if let Some(registry) = &owner_registry {
            shell = shell.with_owner_registry(registry.clone());
        }
        let sandbox: Box<dyn Sandbox> = Box::new(AuditedShell::new(shell));
        let inst = match actor::try_spawn(Instance::new("cli", sandbox)) {
            Ok(instance) => instance,
            Err(error) => {
                lifecycle_diagnostic!("instance startup: {error}");
                break 'owned_run ExitCode::from(1);
            }
        };
        let metrics_client = Arc::new(
            metrics::Client::resolve(common.metrics_addr.as_deref())
                .with_tag("provider", if common.openai { "openai" } else { "anthropic" }),
        );
        let mut state_build = HarnessState::new("default", model.clone(), inst.addr.clone())
            .with_tools(tools)
            .with_metrics(metrics_client);
        if let Some(budget) = common.compact_budget {
            let compactor = compact::Compactor::new(model.clone(), budget);
            let cb: Arc<harness::CompactFn> =
                Arc::new(move |msgs| compactor.maybe_compact(msgs.clone()).unwrap_or(msgs));
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
        let sess = match actor::try_spawn(sess_build) {
            Ok(session) => session,
            Err(error) => {
                lifecycle_diagnostic!("session startup: {error}");
                for client in &mcp_clients {
                    client.request_stop();
                }
                if let Some(tool) = &task_tool {
                    tool.request_stop();
                    tool.clear_state();
                }
                drop(state);
                if let Err(error) = retire_startup_instance(inst, owner_registry.as_ref()) {
                    lifecycle_diagnostic!("instance startup rollback: {error}");
                }
                break 'owned_run ExitCode::from(1);
            }
        };

        // Use streaming so text deltas print to stdout as the model generates them.
        let (tx, rx) = std::sync::mpsc::sync_channel::<harness::StreamEvent>(256);
        let (audit_ack_tx, audit_ack_rx) = std::sync::mpsc::channel::<()>();
        let stream_msg = if runlog_handle.is_some() {
            SessionMsg::PromptStreamAudited {
                text: prompt,
                structured_output_tag: None,
                events: tx,
                ack: audit_ack_rx,
            }
        } else {
            SessionMsg::PromptStream { text: prompt, structured_output_tag: None, events: tx }
        };
        // A failed admission still reaches the common retirement barrier.
        let send_error = sess.addr.send(stream_msg).err().map(|e| e.to_string());
        let mut completed = false;
        let mut turns = 0usize;
        let mut last_was_newline = true;
        let mut pending_done: Option<harness::PromptResult> = None;
        let (mut exit, mut agent_status) = loop {
            if let Some(error) = &send_error {
                break (ExitCode::from(1), AgentStatus::Failure(format!("send failed: {error}")));
            }
            if let Some(registry) = &owner_registry {
                let _ = registry.poll();
            }
            // SIGINT and SIGTERM are observed even while the event stream is idle.
            if run_cancelled() {
                lifecycle_diagnostic!("[cancelling on signal]");
                break (ExitCode::from(130), AgentStatus::Failure("cancelled by signal".into()));
            }
            let ev = match rx.recv_timeout(std::time::Duration::from_millis(25)) {
                Ok(ev) => ev,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    lifecycle_diagnostic!("session dropped event channel");
                    break (
                        ExitCode::from(1),
                        AgentStatus::Failure("session dropped event channel".into()),
                    );
                }
            };
            // Commit each event before presenting it. An audit write failure
            // cancels further processing instead of silently dropping records.
            if !matches!(&ev, harness::StreamEvent::Done(_))
                && let Some(log) = &runlog_handle
                && let Err(e) = log.record_event(&ev)
            {
                lifecycle_diagnostic!("runlog: record failed: {e}");
                break (ExitCode::from(1), AgentStatus::Failure(format!("runlog: {e}")));
            }
            match ev {
                harness::StreamEvent::TextDelta(s) => {
                    last_was_newline = s.ends_with('\n');
                    if let Err(e) = out.write_all(s.as_bytes()).and_then(|_| out.flush()) {
                        lifecycle_diagnostic!("stdout: {e}");
                        break (ExitCode::from(1), AgentStatus::Failure(format!("stdout: {e}")));
                    }
                }
                harness::StreamEvent::ToolUseStart { name, .. } => {
                    if !last_was_newline
                        && let Err(e) = write_newline(out)
                    {
                        lifecycle_diagnostic!("stdout: {e}");
                        break (
                            ExitCode::from(1),
                            AgentStatus::Failure(format!("stdout: {e}")),
                        );
                    }
                    lifecycle_diagnostic!("[tool: {name}]");
                    last_was_newline = true;
                }
                harness::StreamEvent::ToolResult { is_error, .. } => {
                    if is_error {
                        lifecycle_diagnostic!("[tool error]");
                    }
                }
                harness::StreamEvent::TurnComplete { turn, .. } => {
                    turns = turn;
                }
                harness::StreamEvent::Done(pr) => {
                    if !last_was_newline
                        && let Err(e) = write_newline(out)
                    {
                        lifecycle_diagnostic!("stdout: {e}");
                        break (
                            ExitCode::from(1),
                            AgentStatus::Failure(format!("stdout: {e}")),
                        );
                    }
                    completed = pr.completed;
                    turns = pr.turns;
                    pending_done = Some(pr);
                    break (ExitCode::SUCCESS, AgentStatus::Success);
                }
                harness::StreamEvent::Cancelled => {
                    lifecycle_diagnostic!("[cancelled]");
                    break (ExitCode::from(130), AgentStatus::Failure("cancelled".into()));
                }
                harness::StreamEvent::Error(e) => {
                    lifecycle_diagnostic!("session error: {e:?}");
                    break (ExitCode::from(1), AgentStatus::Failure(format!("{e:?}")));
                }
                _ => {}
            }
            if runlog_handle.is_some() {
                let _ = audit_ack_tx.send(());
            }
        };
        // Stop/cancel endpoints before waiting, including failed initial send.
        drop(rx);
        drop(audit_ack_tx);
        let session_cancel = sess.cancellation();
        let instance_cancel = inst.cancellation();
        if !matches!(&agent_status, AgentStatus::Success) {
            session_cancel.request();
            instance_cancel.request();
            for client in &mcp_clients {
                client.request_stop();
            }
            if let Some(registry) = &owner_registry {
                registry.close_admission();
                registry.cancel_all();
            }
            if let Some(tool) = &task_tool {
                tool.request_stop();
            }
        }
        let mut observe_signal = || {
            if let Some(registry) = &owner_registry {
                let _ = registry.poll();
            }
            if run_cancelled() {
                session_cancel.request();
                instance_cancel.request();
                for client in &mcp_clients {
                    client.request_stop();
                }
                if let Some(registry) = &owner_registry {
                    registry.close_admission();
                    registry.cancel_all();
                }
                if let Some(tool) = &task_tool {
                    tool.request_stop();
                }
            }
        };
        if let Err(error) = retire_actor(sess.into_retirement(), &mut observe_signal) {
            exit = ExitCode::from(1);
            agent_status = AgentStatus::Failure(format!("session retirement: {error}"));
            instance_cancel.request();
            for client in &mcp_clients {
                client.request_stop();
            }
            if let Some(registry) = &owner_registry {
                registry.close_admission();
                registry.cancel_all();
            }
            if let Some(tool) = &task_tool {
                tool.request_stop();
            }
        }
        // Every TaskTool call retains its child until join, so Session retirement
        // also fences those calls. Break the explicit tools -> state -> tools cycle.
        if let Some(tool) = &task_tool {
            tool.clear_state();
        }
        drop(state);
        if let Err(error) = retire_actor(inst.into_retirement(), &mut observe_signal) {
            exit = ExitCode::from(1);
            agent_status = AgentStatus::Failure(format!("instance retirement: {error}"));
        }
        drop(task_tool);
        for client in &mcp_clients {
            client.request_stop();
        }
        drop(mcp_clients);
        retire_registry(owner_registry.as_ref());
        if run_cancelled() {
            exit = ExitCode::from(130);
            agent_status = AgentStatus::Failure("cancelled by signal".into());
        }
        let mut successful = matches!(&agent_status, AgentStatus::Success);
        if let (Some(s), Some(ws)) = (strategy, workspace)
            && let Err(e) = s.finish(ws, agent_status)
        {
            lifecycle_diagnostic!("branch strategy finish: {e:?}");
            successful = false;
            exit = ExitCode::from(1);
        }
        if run_cancelled() {
            successful = false;
            exit = ExitCode::from(130);
        }
        // No successful terminal audit record before the actual worker barrier
        // and strategy finalization. Native registry retirement is checked above;
        // its source API still requires independent integration/native admission.
        if successful
            && let Err(error) =
                commit_done_after_retirement(runlog_handle.as_deref(), pending_done, Ok(()))
        {
            lifecycle_diagnostic!("{error}");
            successful = false;
            exit = ExitCode::from(1);
        }
        if successful && completed {
            lifecycle_diagnostic!("[completion signal fired after {turns} turn(s)]");
        } else if turns > 1 {
            lifecycle_diagnostic!("[{turns} turn(s)]");
        }
        exit
    };
    // Covers every setup failure after registry construction as well as the
    // normal path. Returning Unknown must never drop the final owner Arc.
    retire_registry(owner_registry.as_ref());
    exit
}

/// Roll back a partially started actor pair. Keep the actual Instance join and
/// native registry authority until both barriers complete; never detach on error.
fn retire_startup_instance<M: Send + 'static>(
    instance: actor::Spawned<M>,
    registry: Option<&Arc<vshell::ownership::OwnerRegistry>>,
) -> Result<(), String> {
    instance.request_stop();
    if let Some(owner) = registry {
        owner.close_admission();
        owner.cancel_all();
    }
    let result = retire_actor(instance.into_retirement(), || {
        if let Some(owner) = registry {
            let _ = owner.poll();
        }
    });
    retire_registry(registry);
    result
}

/// Keep the owner during Pending. This intentionally waits for real retirement;
/// cancellation cannot manufacture a deadline for a blocked provider callback.
fn retire_actor(mut owner: actor::Retirement<()>, mut on_tick: impl FnMut()) -> Result<(), String> {
    loop {
        on_tick();
        match owner.try_join() {
            actor::JoinPoll::Pending => std::thread::sleep(std::time::Duration::from_millis(10)),
            actor::JoinPoll::Joined(Ok(())) => return Ok(()),
            actor::JoinPoll::Joined(Err(_)) => return Err("worker panicked".into()),
            actor::JoinPoll::AlreadyJoined => return Err("join was already consumed".into()),
        }
    }
}

fn run_owner_registry() -> Result<Option<Arc<vshell::ownership::OwnerRegistry>>, String> {
    #[cfg(target_os = "freebsd")]
    {
        let agent = std::env::current_exe().map_err(|error| error.to_string())?;
        vshell::ownership::OwnerRegistry::new(agent).map(Some).map_err(|error| error.to_string())
    }
    #[cfg(not(target_os = "freebsd"))]
    {
        // Existing ordinary route remains available; no native-domain claim.
        Ok(None)
    }
}

fn retire_registry(registry: Option<&Arc<vshell::ownership::OwnerRegistry>>) {
    let Some(registry) = registry else {
        return;
    };
    registry.close_admission();
    registry.cancel_all();
    loop {
        let state = registry.poll();
        if state.active == 0 && state.unconfirmed == 0 {
            return;
        }
        // Unconfirmed remains owned. No deadline converts it into success.
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// A terminal record can only be attempted after the supplied retirement
/// result succeeds. The caller must supply its actual complete owner barrier.
fn commit_done_after_retirement(
    log: Option<&runlog::RunLog>,
    done: Option<harness::PromptResult>,
    retired: Result<(), String>,
) -> Result<(), String> {
    retired?;
    let done = done.ok_or_else(|| "session completed without Done".to_string())?;
    if let Some(log) = log {
        log.record_event(&harness::StreamEvent::Done(done))
            .map_err(|error| format!("runlog: terminal record failed: {error}"))?;
    }
    Ok(())
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
    let args_ref: HashMap<&str, String> =
        args.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    skill.render(&args_ref).map_err(|e| format!("{e:?}"))
}

fn serve_cmd(mut args: Vec<String>) -> ExitCode {
    if wants_help(&args) {
        print_serve_help();
        return ExitCode::SUCCESS;
    }
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
        lifecycle_diagnostic!("unexpected args: {args:?}");
        return usage_and_exit(2);
    }

    let model = match build_model(&common) {
        Ok(m) => m,
        Err(MissingApiKey) => {
            print_missing_key_error(common.openai);
            return ExitCode::from(2);
        }
    };

    let owner_registry = match run_owner_registry() {
        Ok(registry) => registry,
        Err(error) => {
            lifecycle_diagnostic!("native owner registry: {error}");
            return ExitCode::from(1);
        }
    };
    let exit = 'owned_serve: {
        let (mcp_clients, mcp_tools) = match connect_mcp(&common.mcp_specs, owner_registry.as_ref())
        {
            Ok(t) => t,
            Err(e) => {
                lifecycle_diagnostic!("{e}");
                break 'owned_serve ExitCode::from(1);
            }
        };
        let store = match open_store(common.sessions_dir.as_ref()) {
            Ok(s) => s,
            Err(e) => {
                lifecycle_diagnostic!("{e}");
                break 'owned_serve ExitCode::from(1);
            }
        };

        let mut tools = default_tools(None);
        tools.extend(mcp_tools);

        // One shared metrics sink for the whole server: every per-request
        // ChatHandler Session clones this Arc (matching `agent run`'s prefix +
        // provider tag) so served sessions emit to the same place.
        let metrics_client = Arc::new(
            metrics::Client::resolve(common.metrics_addr.as_deref())
                .with_tag("provider", if common.openai { "openai" } else { "anthropic" }),
        );

        let mut srv = Server::new();
        srv.register(
            "chat",
            Box::new(ChatHandler {
                model,
                tools: Arc::new(tools),
                store,
                runlog_dir: common.runlog_dir.clone(),
                metrics: metrics_client,
            }),
        );

        lifecycle_diagnostic!("serving on {addr}");
        // Bind first so binding errors don't get mixed up with shutdown state.
        let listener = match std::net::TcpListener::bind(&addr) {
            Ok(l) => l,
            Err(e) => {
                lifecycle_diagnostic!("bind {addr}: {e}");
                drop(mcp_clients);
                break 'owned_serve ExitCode::from(1);
            }
        };
        // Keep persistent MCP owners separate from each request's shell registry.
        let mcp_clients = Arc::new(mcp_clients);
        let shutdown = Arc::new(AtomicBool::new(false));
        let bridge_shutdown = shutdown.clone();
        let bridge_clients = mcp_clients.clone();
        let bridge_registry = owner_registry.clone();
        let bridge = match std::thread::Builder::new().spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                while !bridge_shutdown.load(Ordering::Acquire) && !run_cancelled() {
                    if let Some(registry) = &bridge_registry {
                        let _ = registry.poll();
                    }
                    std::thread::sleep(std::time::Duration::from_millis(25));
                }
            }));
            // Server errors also publish shutdown before its worker join barrier.
            bridge_shutdown.store(true, Ordering::Release);
            SHUTDOWN.store(true, Ordering::Release);
            for client in bridge_clients.iter() {
                client.request_stop();
            }
            if let Some(registry) = &bridge_registry {
                registry.close_admission();
                registry.cancel_all();
            }
            if let Err(payload) = result {
                std::panic::resume_unwind(payload);
            }
        }) {
            Ok(bridge) => bridge,
            Err(error) => {
                lifecycle_diagnostic!("server cancellation bridge: {error}");
                break 'owned_serve ExitCode::from(1);
            }
        };
        let result =
            srv.serve_listener_with_shutdown(listener, ServerConfig::default(), shutdown.clone());
        shutdown.store(true, Ordering::Release);
        let bridge_result = retire_actor(actor::Retirement::new(bridge), || {
            if let Some(registry) = &owner_registry {
                let _ = registry.poll();
            }
        });
        for client in mcp_clients.iter() {
            client.request_stop();
        }
        drop(mcp_clients);
        retire_registry(owner_registry.as_ref());
        match (result, bridge_result) {
            (Ok(()), Ok(())) => ExitCode::SUCCESS,
            (Err(error), _) => {
                lifecycle_diagnostic!("serve failed: {error}");
                ExitCode::from(1)
            }
            (_, Err(error)) => {
                lifecycle_diagnostic!("server cancellation bridge: {error}");
                ExitCode::from(1)
            }
        }
    };
    retire_registry(owner_registry.as_ref());
    exit
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
    /// Shared metrics sink. Each per-request Session clones this Arc so all
    /// served sessions (and their subagents) emit to the same place.
    metrics: Arc<metrics::Client>,
}

impl ChatHandler {
    fn handle_prepared(
        &self,
        id: &str,
        prompt: String,
        runlog: Option<Arc<runlog::RunLog>>,
        sink: &mut EventSink,
    ) -> Result<(), HandlerError> {
        sink.emit(Some("start"), id)
            .map_err(|error| HandlerError(format!("SSE start: {error}")))?;
        // Request-local shell owners are separate from persistent server MCP
        // owners; ending one request must not kill another request's clients.
        let registry = run_owner_registry().map_err(HandlerError)?;
        let mut shell = vshell::VShell::new();
        if let Some(registry) = &registry {
            shell = shell.with_owner_registry(registry.clone());
        }
        let sandbox: Box<dyn Sandbox> = Box::new(AuditedShell::new(shell));
        let inst = match actor::try_spawn(Instance::new(id, sandbox)) {
            Ok(instance) => instance,
            Err(error) => {
                retire_registry(registry.as_ref());
                return Err(HandlerError(format!("instance startup: {error}")));
            }
        };
        let state = HarnessState::new("chat", self.model.clone(), inst.addr.clone())
            .with_tools((*self.tools).clone())
            .with_metrics(self.metrics.clone());
        let mut sess_build = Session::new(id, Arc::new(state));
        if let Some(store) = &self.store {
            sess_build = sess_build.with_store(store.clone());
        }
        let sess = match actor::try_spawn(sess_build) {
            Ok(session) => session,
            Err(error) => {
                let rollback = retire_startup_instance(inst, registry.as_ref());
                return Err(HandlerError(format!(
                    "session startup: {error}; instance rollback: {rollback:?}"
                )));
            }
        };
        let (tx, rx) = std::sync::mpsc::sync_channel::<harness::StreamEvent>(256);
        let (audit_ack_tx, audit_ack_rx) = std::sync::mpsc::channel::<()>();
        let stream_msg = if runlog.is_some() {
            SessionMsg::PromptStreamAudited {
                text: prompt,
                structured_output_tag: None,
                events: tx,
                ack: audit_ack_rx,
            }
        } else {
            SessionMsg::PromptStream { text: prompt, structured_output_tag: None, events: tx }
        };
        let mut final_err = sess
            .addr
            .send(stream_msg)
            .err()
            .map(|error| HandlerError(format!("session send: {error}")));
        let mut pending_done = None;
        while final_err.is_none() {
            if let Some(registry) = &registry {
                let _ = registry.poll();
            }
            if run_cancelled() {
                final_err = Some(HandlerError("server shutdown cancelled request".into()));
                break;
            }
            let ev = match rx.recv_timeout(std::time::Duration::from_millis(25)) {
                Ok(event) => event,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    final_err = Some(HandlerError("session closed without Done".into()));
                    break;
                }
            };
            if let harness::StreamEvent::Done(done) = ev {
                pending_done = Some(done);
                break;
            }
            if let Some(log) = &runlog
                && let Err(error) = log.record_event(&ev)
            {
                final_err = Some(HandlerError(format!("runlog: {error}")));
                break;
            }
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
                    sink.emit(Some("tool_result"), &format!("{tool_use_id}\t{is_error}\t{content}"))
                }
                harness::StreamEvent::TurnComplete { turn, stop_reason } => sink.emit(
                    Some("turn_complete"),
                    &format!("{}\t{}", turn, stop_reason.as_deref().unwrap_or("")),
                ),
                harness::StreamEvent::Done(_) => unreachable!("Done is held for retirement"),
                harness::StreamEvent::Cancelled => {
                    final_err = Some(HandlerError("session cancelled".into()));
                    sink.emit(Some("cancelled"), "")
                }
                harness::StreamEvent::Error(e) => {
                    final_err = Some(HandlerError(format!("{e:?}")));
                    Ok(())
                }
            };
            if let Err(error) = write_result {
                final_err = Some(HandlerError(format!("SSE sink: {error}")));
                break;
            }
            if final_err.is_some() {
                break;
            }
            if runlog.is_some() {
                let _ = audit_ack_tx.send(());
            }
        }
        drop(rx);
        drop(audit_ack_tx);
        let session_cancel = sess.cancellation();
        let instance_cancel = inst.cancellation();
        if final_err.is_some() {
            session_cancel.request();
            instance_cancel.request();
            if let Some(registry) = &registry {
                registry.close_admission();
                registry.cancel_all();
            }
        }
        let mut tick = || {
            if run_cancelled() {
                session_cancel.request();
                instance_cancel.request();
                if let Some(registry) = &registry {
                    registry.close_admission();
                    registry.cancel_all();
                }
            }
            if let Some(registry) = &registry {
                let _ = registry.poll();
            }
        };
        if let Err(error) = retire_actor(sess.into_retirement(), &mut tick) {
            final_err = Some(HandlerError(format!("session retirement: {error}")));
            instance_cancel.request();
            if let Some(registry) = &registry {
                registry.close_admission();
                registry.cancel_all();
            }
        }
        if let Err(error) = retire_actor(inst.into_retirement(), &mut tick) {
            final_err = Some(HandlerError(format!("instance retirement: {error}")));
        }
        retire_registry(registry.as_ref());
        if run_cancelled() {
            final_err = Some(HandlerError("server shutdown cancelled request".into()));
        }
        if let Some(error) = final_err {
            return Err(error);
        }
        let done =
            pending_done.ok_or_else(|| HandlerError("session completed without Done".into()))?;
        // Preserve audit-before-presentation. SSE delivery can still fail after
        // a durable model-completion record; that delivery error is returned.
        commit_done_after_retirement(runlog.as_deref(), Some(done.clone()), Ok(()))
            .map_err(HandlerError)?;
        sink.emit(Some("done"), &format!("turns={}\tcompleted={}", done.turns, done.completed))
            .map_err(|error| HandlerError(format!("SSE done: {error}")))
    }
}

impl AgentHandler for ChatHandler {
    fn prepare<'a>(
        &'a self,
        id: &str,
        request_id: &str,
        body: &[u8],
    ) -> Result<server::PreparedResponse<'a>, PrepareError> {
        let prompt = parse_prompt(body).map_err(|e| PrepareError::Client(HandlerError(e)))?;
        let runlog = self
            .runlog_dir
            .as_ref()
            .map(|dir| {
                runlog::RunLog::create_http(dir, request_id)
                    .map(Arc::new)
                    .map_err(|e| PrepareError::Service(HandlerError(format!("runlog: {e}"))))
            })
            .transpose()?;
        let id = id.to_owned();
        Ok(Box::new(move |sink| self.handle_prepared(&id, prompt, runlog, sink)))
    }

    fn handle(
        &self,
        id: &str,
        request_id: &str,
        body: &[u8],
        sink: &mut EventSink,
    ) -> Result<(), HandlerError> {
        self.prepare(id, request_id, body).map_err(|e| HandlerError(e.to_string()))?(sink)
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
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

/// `agent mcp-serve` — speak MCP JSON-RPC over stdio so other agents can
/// call our tool registry (bash + read/write/edit + audit). Useful for
/// chaining agents: parent agent A connects to child agent B's mcp server.
fn mcp_serve_cmd(mut args: Vec<String>) -> ExitCode {
    if wants_help(&args) {
        print_mcp_serve_help();
        return ExitCode::SUCCESS;
    }
    let common = parse_common(&mut args);
    if !args.is_empty() {
        eprintln!("unexpected args: {args:?}");
        return usage_and_exit(2);
    }
    let registry = match run_owner_registry() {
        Ok(value) => value,
        Err(error) => {
            lifecycle_diagnostic!("owner registry: {error}");
            return ExitCode::from(1);
        }
    };
    let outcome = 'owned_mcp: {
        let (clients, extra_tools) = match connect_mcp(&common.mcp_specs, registry.as_ref()) {
            Ok(value) => value,
            Err(error) => {
                lifecycle_diagnostic!("{error}");
                break 'owned_mcp ExitCode::from(1);
            }
        };
        let mut tools = default_tools(None);
        tools.extend(extra_tools);
        let mut shell = vshell::VShell::new();
        if let Some(owner) = &registry {
            shell = shell.with_owner_registry(owner.clone());
        }
        let inst = match actor::try_spawn(Instance::new(
            "mcp-serve",
            Box::new(AuditedShell::new(shell)),
        )) {
            Ok(instance) => instance,
            Err(error) => {
                lifecycle_diagnostic!("MCP instance startup: {error}");
                break 'owned_mcp ExitCode::from(1);
            }
        };
        let stop = inst.cancellation();
        let mut srv =
            mcp::server::Server::new("moth", env!("CARGO_PKG_VERSION"), inst.addr.clone());
        for tool in tools {
            srv.register(tool);
        }
        lifecycle_diagnostic!("mcp server: stdio (waiting for handshake)");
        // The worker owns every server-held address. Its actual join precedes Instance join.
        // Cancellation cannot interrupt a blocked stdin/provider operation: retain Pending.
        let worker = std::thread::Builder::new()
            .name("moth-mcp-stdio".into())
            .spawn(move || srv.serve_stdio());
        let mut failed = false;
        match worker {
            Ok(worker) => {
                let mut worker = actor::Retirement::new(worker);
                loop {
                    if let Some(owner) = &registry {
                        let _ = owner.poll();
                    }
                    if run_cancelled() {
                        failed = true;
                        stop.request();
                        for client in &clients {
                            client.request_stop();
                        }
                        if let Some(owner) = &registry {
                            owner.close_admission();
                            owner.cancel_all();
                        }
                    }
                    match worker.try_join() {
                        actor::JoinPoll::Pending => {
                            std::thread::sleep(std::time::Duration::from_millis(10))
                        }
                        actor::JoinPoll::Joined(Ok(Ok(()))) => break,
                        actor::JoinPoll::Joined(Ok(Err(error))) => {
                            lifecycle_diagnostic!("mcp serve: {error:?}");
                            failed = true;
                            break;
                        }
                        actor::JoinPoll::Joined(Err(_)) | actor::JoinPoll::AlreadyJoined => {
                            lifecycle_diagnostic!("mcp server worker failed");
                            failed = true;
                            break;
                        }
                    }
                }
            }
            Err(error) => {
                lifecycle_diagnostic!("mcp server worker: {error}");
                failed = true;
            }
        }
        if failed {
            stop.request();
        }
        for client in &clients {
            client.request_stop();
        }
        // Server was dropped by its joined worker (or failed spawn closure), breaking its
        // Instance address ownership before consuming the remaining actor owner.
        if let Err(error) = retire_actor(inst.into_retirement(), || {
            if let Some(owner) = &registry {
                let _ = owner.poll();
            }
            if run_cancelled() {
                stop.request();
            }
        }) {
            lifecycle_diagnostic!("mcp instance: {error}");
            failed = true;
        }
        drop(clients);
        retire_registry(registry.as_ref());
        if failed || run_cancelled() { ExitCode::from(1) } else { ExitCode::SUCCESS }
    };
    retire_registry(registry.as_ref());
    outcome
}

#[cfg(test)]
mod run_id_tests {
    use super::{
        ChatHandler, MockModel, Model, anchor_runlog_dir, build_mock_canned, resolve_run_id,
        run_cmd_with_output, write_newline,
    };
    use harness::{ModelEvent, Tool, ToolCtx, ToolError};
    use server::AgentHandler;
    use server::EventSink;
    use std::io::{self, Write};
    use std::path::{Path, PathBuf};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct CountingTool(Arc<AtomicUsize>);
    impl Tool for CountingTool {
        fn name(&self) -> &str {
            "probe"
        }
        fn description(&self) -> &str {
            "Count tool dispatches"
        }
        fn input_schema(&self) -> &str {
            r#"{"type":"object"}"#
        }
        fn call(&self, _input: &str, _ctx: &ToolCtx) -> Result<String, ToolError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok("called".into())
        }
    }

    fn tool_turn_handler(counter: Arc<AtomicUsize>) -> ChatHandler {
        let model: Arc<dyn Model> = Arc::new(MockModel::new(vec![
            vec![
                ModelEvent::ToolUseStart { id: "t1".into(), name: "probe".into() },
                ModelEvent::ToolUseInputDelta("{}".into()),
                ModelEvent::BlockStop,
                ModelEvent::Stop { reason: Some("tool_use".into()) },
            ],
            vec![
                ModelEvent::TextDelta("done".into()),
                ModelEvent::Stop { reason: Some("end_turn".into()) },
            ],
        ]));
        ChatHandler {
            model,
            tools: Arc::new(vec![Arc::new(CountingTool(counter)) as Arc<dyn Tool>]),
            store: None,
            runlog_dir: None,
            metrics: Arc::new(metrics::Client::disabled()),
        }
    }

    fn some(s: &str) -> Option<String> {
        Some(s.to_string())
    }

    #[test]
    fn precedence_flag_then_bop_env_then_agent_env_then_minted() {
        assert_eq!(resolve_run_id(some("f"), some("a"), some("b"), 7).unwrap(), "f");
        assert_eq!(resolve_run_id(None, some("a"), some("b"), 7).unwrap(), "b");
        assert_eq!(resolve_run_id(None, some("a"), None, 7).unwrap(), "a");
        assert_eq!(resolve_run_id(None, None, None, 7).unwrap(), "run-7");
    }

    #[test]
    fn empty_env_is_unset_but_empty_flag_is_an_error() {
        assert_eq!(resolve_run_id(None, some(""), some("b"), 7).unwrap(), "b");
        assert_eq!(resolve_run_id(None, some(""), some(""), 7).unwrap(), "run-7");
        let e = resolve_run_id(some(""), None, None, 7).unwrap_err();
        assert!(e.starts_with("--run-id:"), "{e}");
    }

    #[test]
    fn invalid_explicit_id_names_its_source() {
        let e = resolve_run_id(None, None, some("../x"), 7).unwrap_err();
        assert!(e.starts_with("BOP_RUN_ID:"), "{e}");
        let e = resolve_run_id(None, some("a"), some("../x"), 7).unwrap_err();
        assert!(e.starts_with("BOP_RUN_ID:"), "{e}");
        let e = resolve_run_id(None, some("a b"), None, 7).unwrap_err();
        assert!(e.starts_with("AGENT_RUN_ID:"), "{e}");
    }

    #[test]
    fn relative_runlog_directory_is_anchored_before_workspace_change() {
        assert_eq!(
            anchor_runlog_dir(Path::new("/caller"), Path::new("logs")),
            PathBuf::from("/caller/logs"),
        );
        assert_eq!(
            anchor_runlog_dir(Path::new("/caller"), Path::new("/audit/logs")),
            PathBuf::from("/audit/logs"),
        );
    }

    #[test]
    fn checked_newline_propagates_write_and_flush_failures() {
        struct BrokenWrite;
        impl Write for BrokenWrite {
            fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "stdout closed"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        struct BrokenFlush(Vec<u8>);
        impl Write for BrokenFlush {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "stdout flush failed"))
            }
        }
        assert_eq!(write_newline(&mut BrokenWrite).unwrap_err().kind(), io::ErrorKind::BrokenPipe);
        let mut out = BrokenFlush(Vec::new());
        assert_eq!(write_newline(&mut out).unwrap_err().kind(), io::ErrorKind::BrokenPipe);
        assert_eq!(out.0.as_slice(), b"\n");
    }

    #[test]
    fn cli_broken_stdout_withholds_audit_ack_before_tool_dispatch() {
        struct BrokenOutput {
            writes: usize,
        }
        impl Write for BrokenOutput {
            fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
                self.writes += 1;
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed CLI stdout"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let dir = std::env::temp_dir().join(format!(
            "moth-cli-no-ack-{}-{}",
            std::process::id(),
            super::unix_ms_now(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join("tool-must-not-run");
        let script_path = dir.join("script.json");
        let mut escaped_path = String::new();
        anthropic::json::escape_into(&mut escaped_path, marker.to_string_lossy().as_ref());
        let tool_input = format!(r#"{{"path":"{escaped_path}","content":"ran"}}"#);
        let mut escaped_input = String::new();
        anthropic::json::escape_into(&mut escaped_input, &tool_input);
        let script = [
            r#"{"turns":[[{"type":"text_delta","delta":"first"},"#,
            r#"{"type":"tool_use_start","id":"t1","name":"write_file"},"#,
            r#"{"type":"tool_use_input_delta","delta":""#,
            escaped_input.as_str(),
            r#""},{"type":"block_stop"},{"type":"stop","reason":"tool_use"}],[{"type":"text_delta","delta":"done"},{"type":"stop","reason":"end_turn"}]]}"#,
        ]
        .concat();
        std::fs::write(&script_path, script).unwrap();

        let args = |run_id: &str| {
            vec![
                "--mock-script".into(),
                script_path.to_string_lossy().into_owned(),
                "--runlog".into(),
                dir.to_string_lossy().into_owned(),
                "--run-id".into(),
                run_id.into(),
                "prompt".into(),
            ]
        };
        let mut control_output = Vec::new();
        assert_eq!(
            run_cmd_with_output(args("control"), &mut control_output),
            std::process::ExitCode::SUCCESS,
        );
        assert!(marker.exists(), "positive control did not dispatch write_file");
        std::fs::remove_file(&marker).unwrap();

        let mut output = BrokenOutput { writes: 0 };
        let exit = run_cmd_with_output(args("no-ack"), &mut output);
        assert_eq!(exit, std::process::ExitCode::from(1));
        assert_eq!(output.writes, 1, "fault must fire on the first text delta");
        let log = std::fs::read_to_string(dir.join("no-ack.jsonl")).unwrap();
        let lines: Vec<_> = log.lines().collect();
        assert_eq!(lines.len(), 2, "{log}");
        assert!(lines[0].contains(r#""kind":"start""#), "{log}");
        assert!(lines[1].contains(r#""kind":"text_delta""#), "{log}");
        assert!(!marker.exists(), "tool ran after the failed output write");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn two_http_preflights_with_same_client_id_reserve_separate_files() {
        let dir = std::env::temp_dir().join(format!(
            "moth-http-runlog-{}-{}",
            std::process::id(),
            super::unix_ms_now(),
        ));
        let model: Arc<dyn Model> =
            Arc::new(build_mock_canned("test").expect("default mock fixture should be valid"));
        let handler = ChatHandler {
            model,
            tools: Arc::new(Vec::new()),
            store: None,
            runlog_dir: Some(dir.clone()),
            metrics: Arc::new(metrics::Client::resolve(None)),
        };
        let a = handler.prepare("agent", "repeated-id", b"one").unwrap();
        let b = handler.prepare("agent", "repeated-id", b"two").unwrap();
        let files: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
        assert_eq!(files.len(), 2);
        assert_ne!(files[0], files[1]);
        for path in &files {
            let text = std::fs::read_to_string(path).unwrap();
            assert!(text.contains(r#""request_id":"repeated-id""#), "{text}");
            assert!(text.starts_with(r#"{"seq":0,"ts_ms":"#), "{text}");
        }
        drop((a, b));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failed_initial_sse_start_does_not_run_model() {
        struct BrokenSink;
        impl Write for BrokenSink {
            fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "client disconnected"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let model = Arc::new(MockModel::single(vec![]));
        let handler = ChatHandler {
            model: model.clone() as Arc<dyn Model>,
            tools: Arc::new(Vec::new()),
            store: None,
            runlog_dir: None,
            metrics: Arc::new(metrics::Client::disabled()),
        };
        let mut writer = BrokenSink;
        let mut sink = EventSink::new(&mut writer);
        assert!(handler.handle_prepared("agent", "prompt".into(), None, &mut sink).is_err());
        assert!(model.seen.lock().unwrap().is_empty(), "model ran after SSE start failure");
    }

    #[test]
    fn real_handler_failed_turn_complete_sink_never_dispatches_tool() {
        struct BrokenAtTurnComplete;
        impl Write for BrokenAtTurnComplete {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                if buf == b"turn_complete" {
                    return Err(io::Error::new(io::ErrorKind::BrokenPipe, "lost SSE client"));
                }
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let dir = std::env::temp_dir().join(format!(
            "moth-turn-sink-{}-{}",
            std::process::id(),
            super::unix_ms_now(),
        ));
        let log = Arc::new(runlog::RunLog::create_http(&dir, "client-id").unwrap());
        let path = dir.join(format!("{}.jsonl", log.run_id()));
        let counter = Arc::new(AtomicUsize::new(0));
        let handler = tool_turn_handler(counter.clone());
        let mut writer = BrokenAtTurnComplete;
        let mut sink = EventSink::new(&mut writer);
        let err = handler
            .handle_prepared("agent", "prompt".into(), Some(log.clone()), &mut sink)
            .unwrap_err();
        assert!(err.0.contains("SSE sink"), "{err}");
        assert_eq!(counter.load(Ordering::SeqCst), 0, "tool ran after local SSE failure");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains(r#""kind":"turn_complete""#), "{text}");
        assert!(!text.contains(r#""kind":"tool_result""#), "{text}");
        drop(log);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn real_handler_failed_sse_flush_at_turn_boundary_never_dispatches_tool() {
        struct BrokenFlushAtTurnComplete {
            frame: Vec<u8>,
            failed_at_turn: bool,
        }
        impl Write for BrokenFlushAtTurnComplete {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.frame.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                if self
                    .frame
                    .windows(b"event: turn_complete\n".len())
                    .any(|w| w == b"event: turn_complete\n")
                {
                    self.failed_at_turn = true;
                    return Err(io::Error::new(io::ErrorKind::BrokenPipe, "lost SSE flush"));
                }
                self.frame.clear();
                Ok(())
            }
        }

        let dir = std::env::temp_dir().join(format!(
            "moth-turn-flush-{}-{}",
            std::process::id(),
            super::unix_ms_now(),
        ));
        let log = Arc::new(runlog::RunLog::create_http(&dir, "client-id").unwrap());
        let path = dir.join(format!("{}.jsonl", log.run_id()));
        let counter = Arc::new(AtomicUsize::new(0));
        let handler = tool_turn_handler(counter.clone());
        let mut writer = BrokenFlushAtTurnComplete { frame: Vec::new(), failed_at_turn: false };
        let mut sink = EventSink::new(&mut writer);
        let err = handler
            .handle_prepared("agent", "prompt".into(), Some(log.clone()), &mut sink)
            .unwrap_err();
        assert!(err.0.contains("SSE sink"), "{err}");
        assert!(writer.failed_at_turn, "fault did not fire at turn_complete flush");
        assert_eq!(counter.load(Ordering::SeqCst), 0, "tool ran after local SSE flush failure");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains(r#""kind":"turn_complete""#), "{text}");
        assert!(!text.contains(r#""kind":"tool_result""#), "{text}");
        drop(log);
        std::fs::remove_dir_all(dir).unwrap();
    }

    // RLIMIT_FSIZE is process-wide, so this real RunLog write-failure test
    // confines the limit and ignored SIGXFSZ to an exact-name child test.
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    #[test]
    fn real_handler_audit_failure_at_turn_boundary_never_dispatches_tool() {
        const CHILD: &str = "MOTH_RUNLOG_AUDIT_FAILURE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "run_id_tests::real_handler_audit_failure_at_turn_boundary_never_dispatches_tool", "--nocapture"])
                .env(CHILD, "1")
                .output().unwrap();
            assert!(
                output.status.success(),
                "child failed: {} {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                String::from_utf8_lossy(&output.stdout).contains("1 passed"),
                "exact child test did not run: {}",
                String::from_utf8_lossy(&output.stdout)
            );
            return;
        }

        struct LimitOnBlockStop {
            frame: Vec<u8>,
            path: PathBuf,
        }
        impl Write for LimitOnBlockStop {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.frame.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                if self
                    .frame
                    .windows(b"event: block_stop\n".len())
                    .any(|w| w == b"event: block_stop\n")
                {
                    let len = std::fs::metadata(&self.path)?.len();
                    let mut limit = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
                    unsafe {
                        if libc::getrlimit(libc::RLIMIT_FSIZE, &mut limit) != 0 {
                            return Err(io::Error::last_os_error());
                        }
                        limit.rlim_cur = len as libc::rlim_t;
                        if libc::setrlimit(libc::RLIMIT_FSIZE, &limit) != 0 {
                            return Err(io::Error::last_os_error());
                        }
                    }
                }
                self.frame.clear();
                Ok(())
            }
        }

        unsafe {
            libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
        }
        let dir = std::env::temp_dir().join(format!(
            "moth-turn-audit-{}-{}",
            std::process::id(),
            super::unix_ms_now(),
        ));
        let log = Arc::new(runlog::RunLog::create_http(&dir, "client-id").unwrap());
        let path = dir.join(format!("{}.jsonl", log.run_id()));
        let counter = Arc::new(AtomicUsize::new(0));
        let handler = tool_turn_handler(counter.clone());
        let mut writer = LimitOnBlockStop { frame: Vec::new(), path: path.clone() };
        let mut sink = EventSink::new(&mut writer);
        let err = handler
            .handle_prepared("agent", "prompt".into(), Some(log.clone()), &mut sink)
            .unwrap_err();
        assert!(err.0.contains("runlog"), "{err}");
        assert_eq!(counter.load(Ordering::SeqCst), 0, "tool ran after audit write failed");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains(r#""kind":"turn_complete""#), "{text}");
        assert!(!text.contains(r#""kind":"tool_result""#), "{text}");
        drop(log);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[cfg(test)]
mod lifetime_completion_tests {
    use super::*;

    #[test]
    fn clearing_task_tool_state_breaks_instance_reference_cycle() {
        let instance = spawn(Instance::new("cycle", Box::new(harness::MockSandbox::new(vec![]))));
        let task = Arc::new(TaskTool::new());
        let state = Arc::new(
            HarnessState::new("cycle", Arc::new(MockModel::new(vec![])), instance.addr.clone())
                .with_tools(vec![task.clone() as Arc<dyn Tool>]),
        );
        let weak = Arc::downgrade(&state);
        task.set_state(state.clone());
        drop(state);
        assert!(weak.upgrade().is_some(), "fixture must contain the real strong cycle");
        task.clear_state();
        assert!(weak.upgrade().is_none(), "parent state must be released before Instance join");
        drop(task);
        retire_actor(instance.into_retirement(), || {}).unwrap();
    }

    #[test]
    fn actual_actor_retirement_failure_withholds_done_record() {
        struct StopsBadly;
        impl actor::Actor for StopsBadly {
            type Msg = ();
            fn handle(&mut self, _: ()) {}
            fn stopped(&mut self) {
                panic!("synthetic retirement failure");
            }
        }
        static SEQUENCE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "moth-terminal-retirement-{}-{}-{}",
            std::process::id(),
            unix_ms_now(),
            SEQUENCE.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir(&dir).unwrap();
        let log = runlog::RunLog::open(&dir, "retirement-failure").unwrap();
        let result = retire_actor(spawn(StopsBadly).into_retirement(), || {});
        assert!(result.is_err());
        let done = harness::PromptResult {
            text: "synthetic".into(),
            structured: None,
            completed: true,
            turns: 1,
        };
        assert!(commit_done_after_retirement(Some(&log), Some(done), result).is_err());
        drop(log);
        let text = std::fs::read_to_string(dir.join("retirement-failure.jsonl")).unwrap();
        assert!(!text.lines().any(|line| line.contains(r#""kind":"done""#)));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn either_signal_requests_run_cancellation() {
        assert!(!cancellation_requested(false, false));
        assert!(cancellation_requested(true, false));
        assert!(cancellation_requested(false, true));
        assert!(cancellation_requested(true, true));
    }

    #[test]
    fn dropping_mcp_server_releases_its_instance_address() {
        let instance =
            spawn(Instance::new("mcp-retirement", Box::new(harness::MockSandbox::new(vec![]))));
        let server = mcp::server::Server::new("test", "0", instance.addr.clone());
        let mut owner = instance.into_retirement();
        assert!(matches!(owner.try_join(), actor::JoinPoll::Pending));
        drop(server);
        retire_actor(owner, || {}).unwrap();
    }
    #[test]
    fn partial_startup_rollback_retains_inflight_actor_and_revokes_held_address() {
        struct Gated {
            entered: std::sync::mpsc::SyncSender<()>,
            release: std::sync::mpsc::Receiver<()>,
            stopped: Arc<AtomicBool>,
        }
        impl actor::Actor for Gated {
            type Msg = ();
            fn handle(&mut self, _: ()) {
                self.entered.send(()).unwrap();
                self.release.recv().unwrap();
            }
            fn stopped(&mut self) {
                self.stopped.store(true, Ordering::Release);
            }
        }
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let stopped = Arc::new(AtomicBool::new(false));
        let instance = actor::try_spawn(Gated {
            entered: entered_tx,
            release: release_rx,
            stopped: stopped.clone(),
        })
        .unwrap();
        let held = instance.addr.clone();
        instance.addr.send(()).unwrap();
        entered_rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        let (barrier_tx, barrier_rx) = std::sync::mpsc::sync_channel(1);
        let rollback = std::thread::spawn(move || {
            barrier_tx.send(()).unwrap();
            retire_startup_instance(instance, None)
        });
        barrier_rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        assert!(!rollback.is_finished());
        assert!(!stopped.load(Ordering::Acquire));
        release_tx.send(()).unwrap();
        rollback.join().unwrap().unwrap();
        assert!(stopped.load(Ordering::Acquire));
        assert!(held.send(()).is_err(), "kept sender cannot keep a rolled-back actor alive");
    }
}
