//! `agent doctor` — smoke-check provider config, paths, and network
//! reachability before a real run.
//!
//! Exit codes:
//!   0 — at least one provider key set and its host is reachable
//!   1 — no provider key set
//!   2 — a key is set but the network probe failed

use std::io::Write;
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

/// Snapshot of the env we inspect. Constructed once in `doctor_cmd` and
/// threaded through so `run_doctor` is unit-testable without `std::env`
/// mutation.
pub(crate) struct DoctorEnv {
    pub anthropic_key: Option<String>,
    pub openai_key: Option<String>,
    pub openai_base_url: Option<String>,
    pub model: Option<String>,
    pub agents_root: Option<String>,
    pub sessions_dir: Option<String>,
    pub runlog_dir: Option<String>,
    pub dogstatsd_addr: Option<String>,
}

impl DoctorEnv {
    fn from_env() -> Self {
        let g = |k: &str| std::env::var(k).ok();
        Self {
            anthropic_key: g("ANTHROPIC_API_KEY"),
            openai_key: g("OPENAI_API_KEY"),
            openai_base_url: g("OPENAI_BASE_URL"),
            model: g("MODEL"),
            agents_root: g("AGENTS_ROOT"),
            sessions_dir: g("SESSIONS_DIR"),
            runlog_dir: g("RUNLOG_DIR"),
            dogstatsd_addr: g("DOGSTATSD_ADDR"),
        }
    }
}

/// Outcome of the overall check; mapped to an exit code by `doctor_cmd`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DoctorOutcome {
    Ok,
    NoKey,
    ProbeFailed,
}

impl DoctorOutcome {
    fn exit_code(&self) -> ExitCode {
        match self {
            DoctorOutcome::Ok => ExitCode::SUCCESS,
            DoctorOutcome::NoKey => ExitCode::from(1),
            DoctorOutcome::ProbeFailed => ExitCode::from(2),
        }
    }
}

pub fn doctor_cmd(args: Vec<String>) -> ExitCode {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return ExitCode::SUCCESS;
    }

    // Parse `--mcp 'CMD ARGS'` (repeatable). Anything else is ignored
    // for now — keep the surface small.
    let mut mcp_specs: Vec<String> = Vec::new();
    let mut i = 0;
    let mut args = args;
    while i < args.len() {
        if args[i] == "--mcp" && i + 1 < args.len() {
            args.remove(i);
            mcp_specs.push(args.remove(i));
        } else {
            i += 1;
        }
    }

    let env = DoctorEnv::from_env();
    let outcome = run_doctor(&env, &mcp_specs, &mut std::io::stdout(), probe_host);
    outcome.exit_code()
}

fn print_help() {
    let m = "agent doctor \u{2014} smoke-check provider config and network reachability.\n\n\
        usage:\n  \
        agent doctor [--mcp 'CMD ARGS' ...]\n\n\
        Reports the binary version, env vars the agent reads, whether the\n\
        relevant provider host accepts a TCP connection, and (with --mcp)\n\
        whether each MCP server spawns + handshakes cleanly.\n\n\
        Exit:  0 ok  |  1 no provider key set  |  2 key set but host unreachable";
    eprintln!("{m}");
}

/// The bulk of `agent doctor`. Pure-ish — every input is an argument so
/// tests can drive it without `std::env` mutation.
pub(crate) fn run_doctor<W: Write>(
    env: &DoctorEnv,
    mcp_specs: &[String],
    out: &mut W,
    probe: fn(&str, u16, Duration) -> Result<Duration, String>,
) -> DoctorOutcome {
    let _ = writeln!(out, "agent doctor");
    let _ = writeln!(out, "─────────────");
    let _ = writeln!(out, "binary:      agent {}", env!("CARGO_PKG_VERSION"));
    let _ = writeln!(out, "target:      {}", target_triple());
    let _ = writeln!(out);

    let _ = writeln!(out, "provider config:");
    let _ = writeln!(
        out,
        "  ANTHROPIC_API_KEY:  {}",
        fmt_key_line(env.anthropic_key.as_deref())
    );
    let _ = writeln!(
        out,
        "  OPENAI_API_KEY:     {}",
        fmt_key_line(env.openai_key.as_deref())
    );
    let _ = writeln!(
        out,
        "  OPENAI_BASE_URL:    {}",
        env.openai_base_url
            .as_deref()
            .map(|v| format!("set ({v})"))
            .unwrap_or_else(|| "unset (defaults to api.openai.com)".to_string())
    );
    let _ = writeln!(
        out,
        "  MODEL:              {}",
        env.model
            .as_deref()
            .map(|v| format!("set ({v})"))
            .unwrap_or_else(|| "unset (defaults per provider)".to_string())
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "paths:");
    let root = env.agents_root.as_deref().unwrap_or(".");
    let skill_count = count_skills(root);
    let _ = writeln!(
        out,
        "  AGENTS_ROOT:        {root:<12} (.agents/skills found: {skill_count})"
    );
    let _ = writeln!(out, "  SESSIONS_DIR:       {}", or_unset(env.sessions_dir.as_deref()));
    let _ = writeln!(out, "  RUNLOG_DIR:         {}", or_unset(env.runlog_dir.as_deref()));
    let _ = writeln!(out, "  DOGSTATSD_ADDR:     {}", or_unset(env.dogstatsd_addr.as_deref()));
    let _ = writeln!(out);

    let _ = writeln!(out, "network:");
    let mut any_probe_ok = false;
    let mut any_probe_fail = false;
    if env.anthropic_key.is_some() {
        let host = extract_host(None, "api.anthropic.com");
        write_probe_line(out, &host, 443, probe, &mut any_probe_ok, &mut any_probe_fail);
    } else {
        let _ = writeln!(out, "  api.anthropic.com:443    skipped (ANTHROPIC_API_KEY unset)");
    }
    if env.openai_key.is_some() {
        let host = extract_host(env.openai_base_url.as_deref(), "api.openai.com");
        write_probe_line(out, &host, 443, probe, &mut any_probe_ok, &mut any_probe_fail);
    } else {
        let _ = writeln!(out, "  api.openai.com:443       skipped (OPENAI_API_KEY unset)");
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "mcp servers:");
    if mcp_specs.is_empty() {
        let _ = writeln!(
            out,
            "  (none configured via --mcp; pass --mcp 'CMD ARGS' to test specific servers)"
        );
    } else {
        for spec in mcp_specs {
            let _ = writeln!(out, "  {spec}: skipped (MCP probe not yet wired in this build)");
        }
    }
    let _ = writeln!(out);

    let outcome = compute_outcome(env, any_probe_ok, any_probe_fail);
    let _ = match &outcome {
        DoctorOutcome::Ok => writeln!(out, "ok — agent is ready."),
        DoctorOutcome::NoKey => writeln!(
            out,
            "error: no provider key set. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
        ),
        DoctorOutcome::ProbeFailed => writeln!(
            out,
            "error: provider host unreachable. Check DNS / proxy / firewall."
        ),
    };
    outcome
}

fn write_probe_line<W: Write>(
    out: &mut W,
    host: &str,
    port: u16,
    probe: fn(&str, u16, Duration) -> Result<Duration, String>,
    any_ok: &mut bool,
    any_fail: &mut bool,
) {
    match probe(host, port, Duration::from_secs(5)) {
        Ok(d) => {
            *any_ok = true;
            let _ = writeln!(
                out,
                "  {host}:{port:<5}   reachable ({}ms)         [ok]",
                d.as_millis()
            );
        }
        Err(e) => {
            *any_fail = true;
            let _ = writeln!(out, "  {host}:{port:<5}   FAILED ({e})    [--]");
        }
    }
}

fn compute_outcome(env: &DoctorEnv, any_probe_ok: bool, any_probe_fail: bool) -> DoctorOutcome {
    if env.anthropic_key.is_none() && env.openai_key.is_none() {
        return DoctorOutcome::NoKey;
    }
    if any_probe_fail && !any_probe_ok {
        return DoctorOutcome::ProbeFailed;
    }
    DoctorOutcome::Ok
}

fn or_unset(v: Option<&str>) -> String {
    v.map(|s| s.to_string()).unwrap_or_else(|| "unset".to_string())
}

/// Mask an API key for display. Keep prefix up to first `-` if present
/// (e.g. `sk-ant-`) + last 4 chars; everything else becomes `…`.
fn mask_key(k: &str) -> String {
    if k.len() < 8 {
        return "*".repeat(k.len());
    }
    let last4 = &k[k.len() - 4..];
    let prefix_end = k.find('-').map(|i| i + 1).unwrap_or(3).min(k.len() - 4);
    let prefix = &k[..prefix_end];
    format!("{prefix}…{last4}")
}

fn fmt_key_line(k: Option<&str>) -> String {
    match k {
        Some(v) => format!("set ({})        [ok]", mask_key(v)),
        None => "unset                     [--]".to_string(),
    }
}

fn target_triple() -> &'static str {
    // Compile-time constants; cfg-built so we don't need a build script.
    if cfg!(all(target_arch = "x86_64", target_os = "linux")) {
        "x86_64-unknown-linux-gnu"
    } else if cfg!(all(target_arch = "aarch64", target_os = "linux")) {
        "aarch64-unknown-linux-gnu"
    } else if cfg!(all(target_arch = "x86_64", target_os = "macos")) {
        "x86_64-apple-darwin"
    } else if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        "aarch64-apple-darwin"
    } else {
        "unknown"
    }
}

fn count_skills(root: &str) -> usize {
    let p = Path::new(root).join(".agents/skills");
    match std::fs::read_dir(&p) {
        Ok(entries) => entries.filter_map(Result::ok).count(),
        Err(_) => 0,
    }
}

/// Pull the host out of an OpenAI base URL, falling back to a default
/// when no override is set. Handles `https://host[:port]/path` and bare
/// `host`.
fn extract_host(base_url: Option<&str>, default: &str) -> String {
    let Some(u) = base_url else { return default.to_string() };
    let without_scheme = u.split_once("://").map(|(_, rest)| rest).unwrap_or(u);
    let without_path = without_scheme.split('/').next().unwrap_or(without_scheme);
    // Drop port if present; doctor probes :443 always.
    without_path.split(':').next().unwrap_or(without_path).to_string()
}

/// Open a TCP connection to `host:port` with a timeout. Returns the
/// observed connect duration on success. We deliberately do NOT speak
/// HTTP — the goal is "does the network path resolve and answer?" and a
/// real request would show up in the provider's audit log.
fn probe_host(host: &str, port: u16, timeout: Duration) -> Result<Duration, String> {
    let addr = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("resolve: {e}"))?
        .next()
        .ok_or_else(|| "no address".to_string())?;
    let start = Instant::now();
    TcpStream::connect_timeout(&addr, timeout).map_err(|e| format!("connect: {e}"))?;
    Ok(start.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_key_typical() {
        assert_eq!(mask_key("sk-ant-abcdefghijklmnop1234"), "sk-…1234");
        assert_eq!(mask_key("sk-proj-abcdefghABCD"), "sk-…ABCD");
    }

    #[test]
    fn mask_key_short_returns_stars() {
        assert_eq!(mask_key("abc"), "***");
        assert_eq!(mask_key(""), "");
    }

    #[test]
    fn no_key_set_yields_no_key_outcome() {
        let env = DoctorEnv {
            anthropic_key: None,
            openai_key: None,
            openai_base_url: None,
            model: None,
            agents_root: None,
            sessions_dir: None,
            runlog_dir: None,
            dogstatsd_addr: None,
        };
        fn never(_: &str, _: u16, _: Duration) -> Result<Duration, String> {
            unreachable!("probe should not be called when no key set")
        }
        let mut out = Vec::new();
        let outcome = run_doctor(&env, &[], &mut out, never);
        assert_eq!(outcome, DoctorOutcome::NoKey);
    }

    #[test]
    fn key_set_and_probe_ok_yields_ok() {
        let env = DoctorEnv {
            anthropic_key: Some("sk-ant-abcd1234".into()),
            openai_key: None,
            openai_base_url: None,
            model: None,
            agents_root: None,
            sessions_dir: None,
            runlog_dir: None,
            dogstatsd_addr: None,
        };
        fn ok(_: &str, _: u16, _: Duration) -> Result<Duration, String> {
            Ok(Duration::from_millis(42))
        }
        let mut out = Vec::new();
        assert_eq!(run_doctor(&env, &[], &mut out, ok), DoctorOutcome::Ok);
    }

    #[test]
    fn key_set_but_probe_fails_yields_probe_failed() {
        let env = DoctorEnv {
            anthropic_key: Some("sk-ant-abcd1234".into()),
            openai_key: None,
            openai_base_url: None,
            model: None,
            agents_root: None,
            sessions_dir: None,
            runlog_dir: None,
            dogstatsd_addr: None,
        };
        fn err(_: &str, _: u16, _: Duration) -> Result<Duration, String> {
            Err("connect: timed out".to_string())
        }
        let mut out = Vec::new();
        assert_eq!(run_doctor(&env, &[], &mut out, err), DoctorOutcome::ProbeFailed);
    }

    #[test]
    fn extract_host_handles_common_shapes() {
        assert_eq!(extract_host(None, "api.openai.com"), "api.openai.com");
        assert_eq!(
            extract_host(Some("https://openrouter.ai/api/v1"), "api.openai.com"),
            "openrouter.ai"
        );
        assert_eq!(
            extract_host(Some("http://localhost:11434/v1"), "api.openai.com"),
            "localhost"
        );
        assert_eq!(
            extract_host(Some("api.openai.com"), "api.openai.com"),
            "api.openai.com"
        );
    }
}
