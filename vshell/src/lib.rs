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
        assert!(
            elapsed < Duration::from_secs(3),
            "expected fast timeout, took {elapsed:?}"
        );
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
        let mut sh = VShell::new()
            .with_timeout(Duration::from_secs(60))
            .with_cancel(cancel.clone());

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

        assert!(
            elapsed < Duration::from_secs(2),
            "cancel should kill promptly, took {elapsed:?}"
        );
        assert_eq!(r.exit_code, 130, "stderr: {}", s(&r.stderr));
        assert!(s(&r.stderr).contains("cancelled"));
    }
}
