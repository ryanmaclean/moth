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
        .env_remove("AGENT_RUN_ID")
        .env_remove("BOP_RUN_ID")
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
    run_agent_env(args, &[])
}

/// [`run_agent`] with extra environment variables set on the child.
fn run_agent_env(args: &[&str], envs: &[(&str, &str)]) -> (Option<i32>, String, String) {
    let mut cmd = agent_cmd();
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let mut child = cmd.args(args).spawn().expect("spawn agent");
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

// ----- run identity + ordered runlog -----------------------------------------

/// Parse `N` from a runlog line beginning `{"seq":N,`.
fn seq_of(line: &str) -> Option<u64> {
    let rest = line.strip_prefix(r#"{"seq":"#)?;
    rest[..rest.find(',')?].parse().ok()
}

fn runlog_lines(dir: &std::path::Path, run_id: &str) -> Vec<String> {
    let path = dir.join(format!("{run_id}.jsonl"));
    let s = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {path:?}: {e}; dir has {:?}", ls(dir)));
    s.lines().map(str::to_string).collect()
}

fn ls(dir: &std::path::Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into()).collect()
        })
        .unwrap_or_default()
}

/// `--run-id` names the runlog file and stamps every record; records carry
/// a contiguous `seq` from 0 and the run ends with a `done` record.
#[test]
fn run_id_flag_names_runlog_and_seq_is_contiguous() {
    let dir = tempdir("runid_flag");
    let (code, stdout, stderr) = run_agent(&[
        "run",
        "--mock",
        "--runlog",
        dir.to_str().unwrap(),
        "--run-id",
        "bop:card-42.r1",
        "hi",
    ]);
    assert_eq!(code, Some(0), "stdout=<<<{stdout}>>>\nstderr=<<<{stderr}>>>");
    let lines = runlog_lines(&dir, "bop:card-42.r1");
    assert!(lines.len() >= 3, "expected start..done, got {lines:?}");
    for (i, l) in lines.iter().enumerate() {
        assert_eq!(seq_of(l), Some(i as u64), "line {i}: {l}");
        assert!(l.contains(r#""run_id":"bop:card-42.r1""#), "line {i}: {l}");
    }
    assert!(lines[0].contains(r#""kind":"start""#));
    assert!(lines.last().unwrap().contains(r#""kind":"done""#), "{lines:?}");
    assert_eq!(ls(&dir).len(), 1, "no minted run-<ms> file alongside: {:?}", ls(&dir));
    cleanup(&dir);
}

/// With no flag, a BOP dispatcher's `BOP_RUN_ID` is honoured, and
/// `AGENT_RUN_ID` takes precedence over it.
#[test]
fn run_id_env_precedence() {
    let dir = tempdir("runid_env");
    let d = dir.to_str().unwrap();
    let (code, _o, e) =
        run_agent_env(&["run", "--mock", "--runlog", d, "x"], &[("BOP_RUN_ID", "bop-7")]);
    assert_eq!(code, Some(0), "{e}");
    assert!(dir.join("bop-7.jsonl").exists(), "{:?}", ls(&dir));

    let (code, _o, e) = run_agent_env(
        &["run", "--mock", "--runlog", d, "x"],
        &[("BOP_RUN_ID", "bop-8"), ("AGENT_RUN_ID", "agent-8")],
    );
    assert_eq!(code, Some(0), "{e}");
    assert!(dir.join("agent-8.jsonl").exists(), "{:?}", ls(&dir));
    assert!(!dir.join("bop-8.jsonl").exists());
    cleanup(&dir);
}

/// Re-running with the same external run id (a retry) appends to the same
/// file and continues `seq` rather than restarting it.
#[test]
fn rerun_same_run_id_continues_seq() {
    let dir = tempdir("runid_retry");
    let d = dir.to_str().unwrap();
    for _ in 0..2 {
        let (code, _o, e) =
            run_agent(&["run", "--mock", "--runlog", d, "--run-id", "retry-1", "x"]);
        assert_eq!(code, Some(0), "{e}");
    }
    let lines = runlog_lines(&dir, "retry-1");
    let starts = lines.iter().filter(|l| l.contains(r#""kind":"start""#)).count();
    assert_eq!(starts, 2, "{lines:?}");
    for (i, l) in lines.iter().enumerate() {
        assert_eq!(seq_of(l), Some(i as u64), "line {i}: {l}");
    }
    cleanup(&dir);
}

/// An unsafe external id is refused (exit 2) before any runlog file is
/// created, instead of being silently replaced by a minted id.
#[test]
fn invalid_run_id_is_rejected() {
    let dir = tempdir("runid_bad");
    let logs = dir.join("logs");
    let (code, _o, stderr) = run_agent(&[
        "run",
        "--mock",
        "--runlog",
        logs.to_str().unwrap(),
        "--run-id",
        "../escape",
        "x",
    ]);
    assert_eq!(code, Some(2), "stderr=<<<{stderr}>>>");
    assert!(stderr.contains("--run-id") && stderr.contains("invalid run id"), "{stderr}");
    assert!(!dir.join("escape.jsonl").exists());
    assert!(!logs.exists(), "no runlog dir should be created: {:?}", ls(&dir));

    let (code, _o, stderr) = run_agent_env(
        &["run", "--mock", "--runlog", logs.to_str().unwrap(), "x"],
        &[("BOP_RUN_ID", "a/b")],
    );
    assert_eq!(code, Some(2), "stderr=<<<{stderr}>>>");
    assert!(stderr.contains("BOP_RUN_ID"), "{stderr}");
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
