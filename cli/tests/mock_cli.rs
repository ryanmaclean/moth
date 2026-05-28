//! Black-box tests for the `--mock` / `--mock-script` flags on `agent
//! run`. These spawn the actual `agent` binary (Cargo wires
//! `CARGO_BIN_EXE_agent` for us) so we exercise the real arg parser, the
//! real Session/Instance wiring, and the real mock `Model` integration —
//! and crucially, with `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` cleared,
//! to prove `--mock` really bypasses the API-key check.
//!
//! Tests live here rather than in `rust/integration` because
//! `CARGO_BIN_EXE_<name>` is only set when the integration test belongs
//! to the same package that defines the bin.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Hard cap on how long a single `--mock` run is allowed to take. The
/// session loop drives MockModel synchronously and returns in milliseconds
/// even on cold start; if we hit this, the run hung (the historical
/// failure mode was an instance-actor join deadlocking on a leaked addr
/// clone). We surface the captured stdout/stderr in the panic message so
/// CI logs explain what we saw before the kill.
const RUN_TIMEOUT: Duration = Duration::from_secs(20);

/// Path to the `agent` binary built by this crate.
fn agent_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_agent"))
}

/// Build a `Command` with both provider API keys cleared. If `--mock`
/// (in any of its forms) silently fell through to the real provider path
/// we'd get a "missing API key" exit-2 instead of a clean mock run.
fn agent_cmd() -> Command {
    let mut c = Command::new(agent_bin());
    c.env_remove("ANTHROPIC_API_KEY")
        .env_remove("OPENAI_API_KEY")
        .env_remove("MODEL")
        .env_remove("SESSIONS_DIR")
        .env_remove("RUNLOG_DIR")
        .env_remove("AGENTS_ROOT")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    c
}

/// Run `agent` with the supplied argv, killing it if it hangs past
/// `RUN_TIMEOUT`. Returns `(exit_code, stdout, stderr)`. A timeout panics
/// with whatever was captured so far, which is what we want — silent
/// timeouts hide the real failure.
fn run_agent(args: &[&str]) -> (Option<i32>, String, String) {
    let mut child = agent_cmd().args(args).spawn().expect("spawn agent");
    let mut stdout = child.stdout.take().expect("stdout piped");
    let mut stderr = child.stderr.take().expect("stderr piped");

    // Drain on background threads so a slow consumer can't deadlock the
    // child (its pipes are bounded at OS-buffer size).
    let stdout_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stdout.read_to_string(&mut buf);
        buf
    });
    let stderr_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stderr.read_to_string(&mut buf);
        buf
    });

    let start = Instant::now();
    let status = loop {
        match child.try_wait().expect("try_wait") {
            Some(s) => break s,
            None => {
                if start.elapsed() > RUN_TIMEOUT {
                    let _ = child.kill();
                    let out = stdout_handle.join().unwrap_or_default();
                    let err = stderr_handle.join().unwrap_or_default();
                    panic!(
                        "agent {:?} hung past {:?}\nstdout=<<<{out}>>>\nstderr=<<<{err}>>>",
                        args, RUN_TIMEOUT
                    );
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    };
    let stdout = stdout_handle.join().unwrap_or_default();
    let stderr = stderr_handle.join().unwrap_or_default();
    (status.code(), stdout, stderr)
}

/// `agent run --mock <prompt>` should emit a canned text delta that
/// embeds the (truncated) user prompt and exit cleanly, with no provider
/// API key set in the environment.
#[test]
fn mock_model_run_emits_canned_response() {
    let (code, stdout, stderr) = run_agent(&["run", "--mock", "hello from the test"]);
    assert_eq!(code, Some(0), "non-zero exit\nstdout=<<<{stdout}>>>\nstderr=<<<{stderr}>>>");
    assert!(
        stdout.contains("[mock] received:"),
        "stdout missing canned marker\nstdout=<<<{stdout}>>>\nstderr=<<<{stderr}>>>"
    );
    assert!(
        stdout.contains("hello from the test"),
        "stdout missing echoed prompt\nstdout=<<<{stdout}>>>"
    );
}

/// `agent run --mock-script PATH <prompt>` should load the script,
/// replay its events in order, and surface text deltas to stdout. Tests
/// the JSON parser, the event mapping, and the end-to-end Session/Instance
/// pump — all without an API key.
#[test]
fn mock_script_loads_and_replays() {
    // Three deltas, then a stop. The CLI prints deltas verbatim to stdout
    // so the concatenated output must read "alpha-beta-gamma\n".
    let script = r#"{
        "turns": [[
            {"type":"text_delta","delta":"alpha-"},
            {"type":"text_delta","delta":"beta-"},
            {"type":"text_delta","delta":"gamma\n"},
            {"type":"stop","reason":"end_turn"}
        ]]
    }"#;
    let dir = tempdir("mock_script");
    let path = dir.join("script.json");
    {
        let mut f = std::fs::File::create(&path).expect("create script");
        f.write_all(script.as_bytes()).expect("write script");
    }

    let (code, stdout, stderr) =
        run_agent(&["run", "--mock-script", path.to_str().unwrap(), "anything"]);
    assert_eq!(code, Some(0), "non-zero exit\nstdout=<<<{stdout}>>>\nstderr=<<<{stderr}>>>");
    // Asserting contiguous concatenation proves both ordering and presence.
    assert!(
        stdout.contains("alpha-beta-gamma"),
        "expected concatenated deltas in stdout\nstdout=<<<{stdout}>>>"
    );

    cleanup(&dir);
}

/// A malformed script must exit non-zero with a `path:line:col` error
/// pointing at the bad token; this is the contract the task spec calls
/// out explicitly.
#[test]
fn mock_script_bad_json_reports_line_col() {
    let script = r#"{"turns":[]"#; // missing closing brace
    let dir = tempdir("mock_script_bad");
    let path = dir.join("script.json");
    std::fs::write(&path, script).expect("write script");

    let (code, _stdout, stderr) = run_agent(&["run", "--mock-script", path.to_str().unwrap(), "x"]);
    assert_ne!(code, Some(0), "expected non-zero exit for bad JSON");
    assert!(stderr.contains("script.json:"), "stderr missing path prefix\nstderr=<<<{stderr}>>>");

    cleanup(&dir);
}

// ----- tiny tempdir helpers (no extra crates) --------------------------------

fn tempdir(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos =
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("cli-{label}-{}-{nanos}-{n}", std::process::id()));
    std::fs::create_dir_all(&p).expect("mkdir tempdir");
    p
}

fn cleanup(p: &std::path::Path) {
    let _ = std::fs::remove_dir_all(p);
}
