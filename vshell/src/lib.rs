//! Tiny in-process POSIX shell subset.
//!
//! Most agent shell calls are simple — `echo`, `cd`, `git status`, `ls`. We
//! parse and dispatch them in-process; for real external binaries we spawn
//! a child. Full POSIX semantics belongs in a separate sandbox crate that
//! shells out to `dash`. This is for the 90% case.
//!
//! Supports: simple commands, quoting, variable expansion, command
//! substitution `$(...)`, redirects, pipelines, `;`/`&&`/`||` sequencing,
//! the common built-ins. Bash-isms (globs, loops, heredocs, subshells,
//! arithmetic, arrays, backticks, job control) deliberately error out.

mod exec;
#[doc(hidden)]
pub mod ownership;
mod lex;
mod parse;

pub use exec::{ExecResult, VShell};

#[derive(Debug)]
pub enum ShellError {
    ParseError { msg: String, pos: usize },
    UnsupportedSyntax(String),
    Io(std::io::Error),
}

impl std::fmt::Display for ShellError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShellError::ParseError { msg, pos } => write!(f, "parse error at {pos}: {msg}"),
            ShellError::UnsupportedSyntax(s) => write!(f, "unsupported syntax: {s}"),
            ShellError::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for ShellError {}

impl From<std::io::Error> for ShellError {
    fn from(e: std::io::Error) -> Self {
        ShellError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SEQ: AtomicUsize = AtomicUsize::new(0);

    fn tmpfile(name: &str) -> PathBuf {
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("vshell-{pid}-{n}-{name}"))
    }

    fn s(b: &[u8]) -> String {
        String::from_utf8_lossy(b).into_owned()
    }

    #[test]
    fn simple_echo() {
        let mut sh = VShell::new();
        let r = sh.execute("echo hello");
        assert_eq!(r.exit_code, 0);
        assert_eq!(s(&r.stdout), "hello\n");
    }

    #[test]
    fn echo_n_flag() {
        let mut sh = VShell::new();
        let r = sh.execute("echo -n hi");
        assert_eq!(s(&r.stdout), "hi");
    }

    #[test]
    fn empty_script() {
        let mut sh = VShell::new();
        let r = sh.execute("");
        assert_eq!(r.exit_code, 0);
    }

    #[test]
    fn comments_ignored() {
        let mut sh = VShell::new();
        let r = sh.execute("# this is a comment\necho ok");
        assert_eq!(s(&r.stdout), "ok\n");
    }

    #[test]
    fn exit_propagates() {
        let mut sh = VShell::new();
        let r = sh.execute("exit 7");
        assert_eq!(r.exit_code, 7);
    }

    #[test]
    fn false_returns_one() {
        let mut sh = VShell::new();
        let r = sh.execute("false");
        assert_eq!(r.exit_code, 1);
    }

    #[test]
    fn true_returns_zero() {
        let mut sh = VShell::new();
        let r = sh.execute("true");
        assert_eq!(r.exit_code, 0);
    }

    #[test]
    fn colon_returns_zero() {
        let mut sh = VShell::new();
        let r = sh.execute(":");
        assert_eq!(r.exit_code, 0);
    }

    #[test]
    fn pwd_prints_cwd() {
        let mut sh = VShell::new();
        let cwd = sh.cwd().to_path_buf();
        let r = sh.execute("pwd");
        assert_eq!(s(&r.stdout).trim_end(), cwd.display().to_string());
    }

    #[test]
    fn cd_changes_dir() {
        let mut sh = VShell::new();
        let r = sh.execute("cd /tmp && pwd");
        assert_eq!(r.exit_code, 0);
        assert!(s(&r.stdout).contains("/tmp"));
    }

    #[test]
    fn cd_bad_dir_fails() {
        let mut sh = VShell::new();
        let r = sh.execute("cd /this/does/not/exist/zzz");
        assert_eq!(r.exit_code, 1);
        assert!(!r.stderr.is_empty());
    }

    #[test]
    fn export_marks_var_for_export() {
        let mut sh = VShell::new();
        sh.execute("export FOO=bar");
        assert_eq!(sh.env("FOO"), Some("bar"));
    }

    #[test]
    fn export_no_args_errors() {
        let mut sh = VShell::new();
        let r = sh.execute("export");
        assert_eq!(r.exit_code, 1);
    }

    #[test]
    fn unset_removes_var() {
        let mut sh = VShell::new();
        sh.set_env("X", "y");
        sh.execute("unset X");
        assert_eq!(sh.env("X"), None);
    }

    #[test]
    fn single_quote_literal() {
        let mut sh = VShell::new();
        sh.set_env("V", "EXPANDED");
        let r = sh.execute("echo '$V'");
        assert_eq!(s(&r.stdout), "$V\n");
    }

    #[test]
    fn double_quote_expands() {
        let mut sh = VShell::new();
        sh.set_env("V", "X");
        let r = sh.execute("echo \"$V-$V\"");
        assert_eq!(s(&r.stdout), "X-X\n");
    }

    #[test]
    fn backslash_escape() {
        let mut sh = VShell::new();
        let r = sh.execute("echo \\$VAR");
        assert_eq!(s(&r.stdout), "$VAR\n");
    }

    #[test]
    fn backslash_newline_continuation() {
        let mut sh = VShell::new();
        let r = sh.execute("echo a\\\nb");
        assert_eq!(s(&r.stdout), "ab\n");
    }

    #[test]
    fn unterminated_single_quote() {
        let mut sh = VShell::new();
        let r = sh.execute("echo 'oops");
        assert_eq!(r.exit_code, 2);
        assert!(s(&r.stderr).contains("unterminated single quote"));
    }

    #[test]
    fn unterminated_double_quote() {
        let mut sh = VShell::new();
        let r = sh.execute("echo \"oops");
        assert_eq!(r.exit_code, 2);
        assert!(s(&r.stderr).contains("unterminated double quote"));
    }

    #[test]
    fn unquoted_var_expansion() {
        let mut sh = VShell::new();
        sh.set_env("X", "world");
        let r = sh.execute("echo hello $X");
        assert_eq!(s(&r.stdout), "hello world\n");
    }

    #[test]
    fn braced_var_expansion() {
        let mut sh = VShell::new();
        sh.set_env("X", "v");
        let r = sh.execute("echo ${X}x");
        assert_eq!(s(&r.stdout), "vx\n");
    }

    #[test]
    fn braced_var_modifier_rejected() {
        let mut sh = VShell::new();
        let r = sh.execute("echo ${X:-default}");
        assert_eq!(r.exit_code, 2);
        assert!(s(&r.stderr).contains("parameter expansion"));
    }

    #[test]
    fn unset_var_expands_to_empty() {
        let mut sh = VShell::new();
        let r = sh.execute("echo a${MISSING}b");
        assert_eq!(s(&r.stdout), "ab\n");
    }

    #[test]
    fn env_prefix_per_command() {
        // POSIX: `$X` is expanded BEFORE the command runs, before the prefix
        // takes effect. The subprocess sees X=1 in its env; the shell does not.
        let mut sh = VShell::new();
        let r = sh.execute("X=1 sh -c 'echo $X'");
        assert_eq!(s(&r.stdout), "1\n");
        assert_eq!(sh.env("X"), None);
    }

    #[test]
    fn bare_assignment_persists() {
        let mut sh = VShell::new();
        sh.execute("X=hello");
        assert_eq!(sh.env("X"), Some("hello"));
    }

    #[test]
    fn sequence_semicolon() {
        let mut sh = VShell::new();
        let r = sh.execute("echo a; echo b");
        assert_eq!(s(&r.stdout), "a\nb\n");
    }

    #[test]
    fn and_if_short_circuits() {
        let mut sh = VShell::new();
        let r = sh.execute("false && echo nope");
        assert_eq!(r.exit_code, 1);
        assert!(s(&r.stdout).is_empty());
    }

    #[test]
    fn and_if_runs_on_success() {
        let mut sh = VShell::new();
        let r = sh.execute("true && echo yes");
        assert_eq!(s(&r.stdout), "yes\n");
    }

    #[test]
    fn or_if_runs_on_failure() {
        let mut sh = VShell::new();
        let r = sh.execute("false || echo recovered");
        assert_eq!(s(&r.stdout), "recovered\n");
    }

    #[test]
    fn or_if_skips_on_success() {
        let mut sh = VShell::new();
        let r = sh.execute("true || echo nope");
        assert!(s(&r.stdout).is_empty());
    }

    #[test]
    fn redirect_stdout_truncate() {
        let p = tmpfile("trunc");
        let mut sh = VShell::new();
        let r = sh.execute(&format!("echo hello > {}", p.display()));
        assert_eq!(r.exit_code, 0);
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "hello\n");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn redirect_stdout_append() {
        let p = tmpfile("append");
        let mut sh = VShell::new();
        sh.execute(&format!("echo a > {}", p.display()));
        sh.execute(&format!("echo b >> {}", p.display()));
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "a\nb\n");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn redirect_stdin() {
        let p = tmpfile("stdin");
        std::fs::write(&p, "from-file\n").unwrap();
        let mut sh = VShell::new();
        let r = sh.execute(&format!("cat < {}", p.display()));
        assert_eq!(s(&r.stdout), "from-file\n");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn redirect_stderr_truncate() {
        let p = tmpfile("err");
        let mut sh = VShell::new();
        let r = sh.execute(&format!("sh -c 'echo err >&2' 2> {}", p.display()));
        assert_eq!(r.exit_code, 0);
        assert!(s(&r.stderr).is_empty());
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "err\n");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn redirect_stderr_append() {
        let p = tmpfile("erra");
        let mut sh = VShell::new();
        sh.execute(&format!("sh -c 'echo a >&2' 2> {}", p.display()));
        sh.execute(&format!("sh -c 'echo b >&2' 2>> {}", p.display()));
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "a\nb\n");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn redirect_stderr_to_stdout() {
        let mut sh = VShell::new();
        let r = sh.execute("sh -c 'echo out; echo err >&2' 2>&1");
        assert!(s(&r.stdout).contains("out"));
        assert!(s(&r.stdout).contains("err"));
        assert!(s(&r.stderr).is_empty());
    }

    #[test]
    fn redirect_missing_target_errors() {
        let mut sh = VShell::new();
        let r = sh.execute("echo x > ");
        assert_eq!(r.exit_code, 2);
    }

    #[test]
    fn pipeline_two_stage() {
        let mut sh = VShell::new();
        let r = sh.execute("echo hello | cat");
        assert_eq!(s(&r.stdout), "hello\n");
    }

    #[test]
    fn pipeline_three_stage() {
        let mut sh = VShell::new();
        let r = sh.execute("echo abc | cat | cat");
        assert_eq!(s(&r.stdout), "abc\n");
    }

    #[test]
    fn pipeline_exit_is_last() {
        let mut sh = VShell::new();
        let r = sh.execute("true | false");
        assert_eq!(r.exit_code, 1);
    }

    #[test]
    fn command_substitution() {
        let mut sh = VShell::new();
        let r = sh.execute("echo $(echo hello)");
        assert_eq!(s(&r.stdout), "hello\n");
    }

    #[test]
    fn command_substitution_nested() {
        let mut sh = VShell::new();
        let r = sh.execute("echo $(echo $(echo deep))");
        assert_eq!(s(&r.stdout), "deep\n");
    }

    #[test]
    fn command_substitution_in_double_quotes() {
        let mut sh = VShell::new();
        let r = sh.execute("echo \"x-$(echo y)-z\"");
        assert_eq!(s(&r.stdout), "x-y-z\n");
    }

    #[test]
    fn command_substitution_trims_trailing_newlines() {
        let mut sh = VShell::new();
        let r = sh.execute("echo -$(printf 'hi\\n\\n\\n')-");
        assert_eq!(s(&r.stdout), "-hi-\n");
    }

    #[test]
    fn unsupported_for_loop() {
        let mut sh = VShell::new();
        let r = sh.execute("for i in a b; do echo $i; done");
        assert_eq!(r.exit_code, 2);
        assert!(s(&r.stderr).contains("unsupported") || s(&r.stderr).contains("parse"));
    }

    #[test]
    fn unsupported_glob() {
        let mut sh = VShell::new();
        let r = sh.execute("echo *.txt");
        assert_eq!(r.exit_code, 2);
        assert!(s(&r.stderr).contains("glob"));
    }

    #[test]
    fn unsupported_heredoc() {
        let mut sh = VShell::new();
        let r = sh.execute("cat <<EOF\nhi\nEOF");
        assert_eq!(r.exit_code, 2);
        assert!(s(&r.stderr).contains("heredoc"));
    }

    #[test]
    fn unsupported_subshell() {
        let mut sh = VShell::new();
        let r = sh.execute("(echo a)");
        assert_eq!(r.exit_code, 2);
    }

    #[test]
    fn unsupported_backtick() {
        let mut sh = VShell::new();
        let r = sh.execute("echo `date`");
        assert_eq!(r.exit_code, 2);
        assert!(s(&r.stderr).contains("backtick"));
    }

    #[test]
    fn unsupported_background_job() {
        let mut sh = VShell::new();
        let r = sh.execute("sleep 1 &");
        assert_eq!(r.exit_code, 2);
        assert!(s(&r.stderr).contains("background"));
    }

    #[test]
    fn external_command_not_found() {
        let mut sh = VShell::new();
        let r = sh.execute("nope_xxx_definitely_no_binary_xyz");
        assert_eq!(r.exit_code, 127);
    }

    #[test]
    fn external_command_propagates_exit() {
        let mut sh = VShell::new();
        let r = sh.execute("sh -c 'exit 42'");
        assert_eq!(r.exit_code, 42);
    }

    #[test]
    fn export_then_subprocess_sees_var() {
        let mut sh = VShell::new();
        sh.execute("export X=visible");
        let r = sh.execute("sh -c 'echo $X'");
        assert_eq!(s(&r.stdout), "visible\n");
    }

    #[test]
    fn unexported_var_invisible_to_subprocess() {
        let mut sh = VShell::new();
        sh.execute("X=hidden");
        // `[]` is part of the inner sh -c body, not parsed by vshell.
        let r = sh.execute("sh -c 'echo @${X}@'");
        assert_eq!(s(&r.stdout), "@@\n");
    }

    #[test]
    fn env_prefix_inherited_by_subprocess() {
        let mut sh = VShell::new();
        let r = sh.execute("X=passed sh -c 'echo $X'");
        assert_eq!(s(&r.stdout), "passed\n");
    }

    #[test]
    fn echo_with_escapes_in_double_quotes() {
        let mut sh = VShell::new();
        let r = sh.execute("echo \"a\\\"b\"");
        assert_eq!(s(&r.stdout), "a\"b\n");
    }

    // --- supervision (timeout / cancel / output cap) ---

    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    #[test]
    fn command_killed_after_timeout() {
        let mut sh = VShell::new().with_timeout(Duration::from_secs(1));
        let start = Instant::now();
        let r = sh.execute("sleep 60");
        let elapsed = start.elapsed();
        // Should kill via SIGTERM within ~1s; SIGKILL grace is 5s but `sleep`
        // exits on SIGTERM immediately.
        assert!(elapsed < Duration::from_secs(3), "expected fast timeout, took {elapsed:?}");
        assert_eq!(r.exit_code, 124, "stderr: {}", s(&r.stderr));
        assert!(
            s(&r.stderr).contains("timeout"),
            "stderr did not mention timeout: {}",
            s(&r.stderr)
        );
    }

    #[test]
    fn stdout_overflow_kills_child_and_errors() {
        // `yes` writes "y\n" forever; 8 MiB cap is hit very quickly.
        let mut sh = VShell::new().with_timeout(Duration::from_secs(30));
        let start = Instant::now();
        let r = sh.execute("yes");
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(10),
            "overflow kill should be prompt, took {elapsed:?}"
        );
        assert_ne!(r.exit_code, 0);
        let err = s(&r.stderr);
        assert!(
            err.contains("cap") || err.contains("exceeded"),
            "stderr did not mention cap: {err}"
        );
        // Captured stdout should be bounded at the cap (allow small slack for
        // the final chunk that triggered overflow).
        assert!(
            r.stdout.len() <= 8 * 1024 * 1024 + 8192,
            "stdout exceeded cap+chunk: {}",
            r.stdout.len()
        );
    }

    #[test]
    fn cancel_token_set_terminates_in_flight_child() {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut sh =
            VShell::new().with_timeout(Duration::from_secs(60)).with_cancel(cancel.clone());

        // Flip the flag after 100ms from another thread.
        let flipper = {
            let cancel = cancel.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(100));
                cancel.store(true, std::sync::atomic::Ordering::SeqCst);
            })
        };

        let start = Instant::now();
        let r = sh.execute("sleep 60");
        let elapsed = start.elapsed();
        let _ = flipper.join();

        assert!(elapsed < Duration::from_secs(2), "cancel should kill promptly, took {elapsed:?}");
        assert_eq!(r.exit_code, 130, "stderr: {}", s(&r.stderr));
        assert!(s(&r.stderr).contains("cancelled"));
    }
}

#[cfg(all(test, target_os = "freebsd"))]
mod native_owner_tests {
    //! Authored native acceptance controls; never executed in this source packet.
    //! Run only on an admitted FreeBSD target, inside the independently owned
    //! outer fixture, with MOTH_OWNER_TEST_AGENT naming the exact selected agent.
    use std::ffi::OsString;
    use std::io::{self, BufRead, Read, Write};
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use crate::ownership::{CommandSpec, Limits, OwnedRun, OwnerRegistry, Retirement, RunState};
    
    fn registry() -> Arc<OwnerRegistry> {
        let agent = PathBuf::from(std::env::var_os("MOTH_OWNER_TEST_AGENT").expect("explicit admitted agent path required"));
        OwnerRegistry::new(agent).unwrap()
    }
    fn spec(mode: &str, input_bytes: u64) -> CommandSpec {
        CommandSpec {
            program: std::env::current_exe().unwrap().into_os_string(),
            args: ["--ignored", "--exact", "native_owner_tests::owner_fixture_entry", "--nocapture", "--test-threads=1"].map(OsString::from).to_vec(),
            cwd: None, env: vec![("MOTH_OWNER_FIXTURE_MODE".into(), mode.into())],
            limits: Limits { deadline: Duration::from_secs(20), capture_bytes: 1024 * 1024, input_bytes },
        }
    }
    fn finish(run: &mut OwnedRun, input: &[u8]) -> (Retirement, Vec<u8>, Vec<u8>) {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut sent = 0; let mut out = Vec::new(); let mut err = Vec::new();
        let mut buffer = [0u8; 16_384];
        loop {
            if sent < input.len() {
                match run.write(&input[sent..]) {
                    Ok(n) => sent += n,
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {},
                    Err(e) => panic!("fixture input failed: {e}"),
                }
            }
            // Exact finite inputs deliberately send no EOF. This exercises the
            // actual paired protocol and the fast-exit closure race.
            for error in [false, true] {
                loop {
                    let result = if error { run.read_stderr(&mut buffer) } else { run.read(&mut buffer) };
                    match result {
                        Ok(0) => break,
                        Ok(n) => if error { err.extend_from_slice(&buffer[..n]); } else { out.extend_from_slice(&buffer[..n]); },
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                        Err(e) => panic!("fixture output failed: {e}"),
                    }
                }
            }
            if let RunState::Retired(result) = run.poll() {
                assert_eq!(sent, input.len());
                // A final poll can have captured new buffered data; drain after
                // retirement too, before comparing the complete payload.
                run.read_to_end(&mut out).unwrap();
                loop { let n = run.read_stderr(&mut buffer).unwrap(); if n == 0 { break; } err.extend_from_slice(&buffer[..n]); }
                return (result, out, err);
            }
            assert!(Instant::now() < deadline, "retirement unconfirmed: {:?}", run.poll());
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    #[test]
    #[ignore = "requires admitted FreeBSD native target and exact agent; unrun"]
    fn shell_prefix_reader_preserves_success_but_default_requires_full_input() {
        let input = vec![0x61; 8 * 1024 * 1024];
        for shell_policy in [false, true] {
            let owner = registry();
            let command = spec("prefix", input.len() as u64);
            let mut run = if shell_policy { owner.spawn_redirected(command, &[]).unwrap() }
                else { owner.spawn_cleared(command).unwrap() };
            let deadline = Instant::now() + Duration::from_secs(30);
            let mut sent = 0; let mut stopped = false; let mut buffer = [0u8; 16_384];
            let outcome = loop {
                if !stopped && sent < input.len() && matches!(run.poll(), RunState::Running) {
                    match run.write(&input[sent..]) {
                        Ok(n) => sent += n,
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {},
                        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => stopped = true,
                        Err(e) => panic!("{e}"),
                    }
                }
                for error in [false, true] {
                    match if error { run.read_stderr(&mut buffer) } else { run.read(&mut buffer) } {
                        Ok(_) => {}, Err(e) if e.kind() == io::ErrorKind::WouldBlock => {},
                        Err(e) => panic!("{e}"),
                    }
                }
                if let RunState::Retired(result) = run.poll() { break result; }
                assert!(Instant::now() < deadline, "unconfirmed: {:?}", run.poll());
                std::thread::sleep(Duration::from_millis(5));
            };
            assert!(sent < input.len(), "fixture failed to establish a prefix consumer");
            if shell_policy { assert!(outcome.success(), "{outcome:?}"); }
            else { assert_eq!(outcome.reason, 4, "strict full-input policy was weakened"); }
            assert!(owner.retire_until(Instant::now() + Duration::from_secs(2)));
        }
    }

    fn wait_running(run: &OwnedRun) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match run.poll() {
                RunState::Running => return,
                RunState::Starting => {},
                other => panic!("did not start: {other:?}"),
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    
    #[test]
    #[ignore = "requires admitted FreeBSD native target and exact agent; unrun"]
    fn zero_input_fast_exit_and_full_unicode_input_preserve_receipts() {
        let owner = registry();
        for _ in 0..16 {
            let mut run = owner.spawn(spec("fast", 0)).unwrap();
            let (result, out, _) = finish(&mut run, &[]);
            assert!(result.success()); assert!(out.windows(10).any(|w| w == b"FAST-OWNED"));
        }
        let input = "full-prompt-♥-🂡\n".repeat(8192).into_bytes();
        let mut run = owner.spawn(spec("echo", input.len() as u64)).unwrap();
        let (result, out, _) = finish(&mut run, &input);
        assert!(result.success());
        assert!(out.windows(input.len()).any(|w| w == input.as_slice()), "full input lost or truncated");
        assert!(owner.retire_until(Instant::now() + Duration::from_secs(2)));
    }
    
    #[test]
    #[ignore = "requires admitted FreeBSD native target and exact agent; unrun"]
    fn separate_session_and_nested_reaper_retire_without_touching_sibling_owner() {
        let sentinel_owner = registry();
        let sentinel = sentinel_owner.spawn(spec("leaf", 0)).unwrap();
        wait_running(&sentinel);
        for mode in ["spawn-session", "spawn-nested"] {
            let owner = registry();
            let mut run = owner.spawn(spec(mode, 0)).unwrap();
            let (result, out, _) = finish(&mut run, &[]);
            assert!(result.success());
            assert!(out.windows(16).any(|w| w == b"DESCENDANT-READY!"));
            assert!(owner.retire_until(Instant::now() + Duration::from_secs(2)));
            assert!(matches!(sentinel.poll(), RunState::Running), "sibling owner was affected");
        }
        sentinel.cancel();
        assert!(sentinel_owner.retire_until(Instant::now() + Duration::from_secs(10)));
    }
    
    #[test]
    #[ignore = "requires admitted FreeBSD native target and exact agent; unrun"]
    fn cancel_stalled_input_does_not_wait_for_credit_or_discard_cancel() {
        let owner = registry();
        let mut run = owner.spawn(spec("leaf", u64::MAX)).unwrap();
        wait_running(&run);
        let bytes = [0u8; 16_384];
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut accepted = 0; let mut blocked = 0;
        loop {
            match run.write(&bytes) {
                Ok(n) => { accepted += n; blocked = 0; },
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => blocked += 1,
                Err(e) => panic!("{e}"),
            }
            if accepted >= 32_768 && blocked >= 20 { break; }
            assert!(Instant::now() < deadline, "did not establish actual input backpressure");
            std::thread::sleep(Duration::from_millis(5));
        }
        // Repetition specifically guards the former unsent-Cancel deletion bug.
        run.cancel(); run.cancel(); owner.cancel_all(); owner.cancel_all();
        let (result, _, _) = finish(&mut run, &[]);
        assert_eq!(result.reason, 1);
        assert!(owner.retire_until(Instant::now() + Duration::from_secs(2)));
    }
    
    #[test]
    #[ignore = "requires admitted FreeBSD native target and exact agent; unrun"]
    fn dropped_lease_and_failed_spawn_remain_owned_until_native_receipt() {
        let owner = registry();
        let run = owner.spawn(spec("leaf", 0)).unwrap();
        wait_running(&run); drop(run);
        assert!(owner.retire_until(Instant::now() + Duration::from_secs(10)));
        assert!(owner.spawn(spec("fast", 0)).is_err(), "closed admission reopened");
        let owner = registry();
        let mut missing = spec("fast", 0);
        missing.program = "/__moth_owner_fixture_missing__/no-program".into();
        let mut run = owner.spawn(missing).unwrap();
        let (result, _, _) = finish(&mut run, &[]);
        assert_eq!(result.reason, 5); assert!(!result.success());
        assert_eq!(result.payload_exit_code, None);
        assert!(owner.retire_until(Instant::now() + Duration::from_secs(2)));
    }
    
    #[test]
    #[ignore = "requires admitted FreeBSD native target and exact agent; unrun"]
    fn output_overflow_is_failure_with_confirmed_cleanup() {
        let owner = registry();
        let mut run = owner.spawn(spec("flood", 0)).unwrap();
        let (result, _, _) = finish(&mut run, &[]);
        assert_eq!(result.reason, 3); assert!(!result.success());
        assert!(owner.retire_until(Instant::now() + Duration::from_secs(2)));
    }
    
    fn spawn_fixture(mode: &str) -> std::process::Child {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "native_owner_tests::owner_fixture_entry", "--nocapture", "--test-threads=1"])
            .env("MOTH_OWNER_FIXTURE_MODE", mode).stdin(Stdio::null())
            .stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().unwrap();
        // Confirm the descendant established its separate session/reaper before
        // allowing the direct payload to exit. Its inherited stderr keeps a
        // captured native endpoint open until actual descendant retirement.
        let mut reader = io::BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        loop {
            line.clear(); assert_ne!(reader.read_line(&mut line).unwrap(), 0);
            if line.contains("OWNED-CHILD-READY") { break; }
        }
        child
    }
    
    #[test]
    #[ignore = "private payload entry; invoke only through admitted owner fixture"]
    fn owner_fixture_entry() {
        let Some(mode) = std::env::var_os("MOTH_OWNER_FIXTURE_MODE") else { return; };
        let mode = mode.to_str().expect("selected fixture mode must be UTF-8");
        match mode {
            "fast" => println!("FAST-OWNED"),
            "announce-echo" => {
            std::fs::write(std::env::var_os("MOTH_OWNER_MARKER").unwrap(), b"started").unwrap();
            let mut input = Vec::new(); io::stdin().read_to_end(&mut input).unwrap(); io::stdout().write_all(&input).unwrap();
        }
        "streams" => { io::stdout().write_all(b"PAYLOAD-STDOUT\n").unwrap(); io::stderr().write_all(b"PAYLOAD-STDERR\n").unwrap(); }
        "prefix" => { let mut byte = [0u8; 1]; io::stdin().read_exact(&mut byte).unwrap(); assert_eq!(byte[0], 0x61); }
        "echo" => { let mut input = Vec::new(); io::stdin().read_to_end(&mut input).unwrap(); io::stdout().write_all(&input).unwrap(); }
        "duplex" => {
            std::thread::scope(|scope| {
                scope.spawn(|| { io::stdout().write_all(&vec![0xff; 2 * 1024 * 1024]).unwrap(); });
                io::stderr().write_all(b"DUPLEX-STDERR-READY\n").unwrap(); io::stderr().flush().unwrap();
                let mut input = Vec::new(); io::stdin().read_to_end(&mut input).unwrap();
                assert_eq!(input, b"duplex-input");
            });
        }
            "spawn-session" | "spawn-nested" => {
                let child = spawn_fixture(if mode == "spawn-session" { "session-leaf" } else { "nested" });
                println!("DESCENDANT-READY!");
                // The private owner, not Child::drop, retains native authority.
                drop(child);
            }
            "session-leaf" | "nested" | "leaf" => {
                unsafe { assert_ne!(libc::signal(libc::SIGTERM, libc::SIG_IGN), libc::SIG_ERR); }
                if mode == "session-leaf" { assert!(unsafe { libc::setsid() } > 0); }
                let child = if mode == "nested" {
                    assert_eq!(unsafe { libc::procctl(libc::P_PID, 0, libc::PROC_REAP_ACQUIRE, std::ptr::null_mut()) }, 0);
                    Some(spawn_fixture("leaf"))
                } else { None };
                println!("OWNED-CHILD-READY"); io::stdout().flush().unwrap();
                std::thread::sleep(Duration::from_secs(60)); drop(child);
            }
            "flood" => { let bytes = [b'x'; 16_384]; loop { if io::stdout().write_all(&bytes).is_err() { break; } } }
            _ => panic!("invalid fixture mode"),
        }
    }
    
    #[test]
    #[ignore = "requires admitted FreeBSD native target and exact agent; unrun"]
    fn owned_redirect_order_append_and_malformed_request_controls() {
        use crate::ownership::OwnedRedirect as R;
        let root = std::env::temp_dir().join(format!("moth-launch-order-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&root).unwrap(); let path = root.join("output"); let untouched = root.join("must-not-exist");
        let owner = registry();
        let mut run = owner.spawn_redirected(spec("streams", 0), &[R::ErrorToOutput, R::Output { path: path.clone(), append: false }]).unwrap();
        let (result, out, err) = finish(&mut run, &[]); assert!(result.success());
        let file = std::fs::read(&path).unwrap();
        assert!(out.windows(14).any(|w| w == b"PAYLOAD-STDERR"));
        assert!(!out.windows(14).any(|w| w == b"PAYLOAD-STDOUT")); assert!(err.is_empty());
        assert!(file.windows(14).any(|w| w == b"PAYLOAD-STDOUT"));
        assert!(!file.windows(14).any(|w| w == b"PAYLOAD-STDERR"));
        let mut run = owner.spawn_redirected(spec("streams", 0), &[R::Output { path: path.clone(), append: false }, R::ErrorToOutput]).unwrap();
        let (result, out, err) = finish(&mut run, &[]); assert!(result.success()); assert!(out.is_empty() && err.is_empty());
        let before = std::fs::read(&path).unwrap(); assert!(before.windows(14).any(|w| w == b"PAYLOAD-STDOUT")); assert!(before.windows(14).any(|w| w == b"PAYLOAD-STDERR"));
        let mut run = owner.spawn_redirected(spec("streams", 0), &[R::Output { path: path.clone(), append: true }, R::ErrorToOutput]).unwrap();
        assert!(finish(&mut run, &[]).0.success()); let after = std::fs::read(&path).unwrap(); assert!(after.starts_with(&before) && after.len() > before.len());
        let mut bad = spec("fast", 0); bad.args.push(OsString::from("bad\0argument"));
        assert!(owner.spawn_redirected(bad, &[R::Output { path: untouched.clone(), append: false }]).is_err());
        assert!(!untouched.exists(), "malformed request caused filesystem effects");
        assert!(owner.retire_until(Instant::now() + Duration::from_secs(2)));
        std::fs::remove_file(path).unwrap(); std::fs::remove_dir(root).unwrap();
    }

}
