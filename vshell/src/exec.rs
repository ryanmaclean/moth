//! Interpreter.
//!
//! Walks the AST, runs built-ins in-process, spawns children for externals.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::ShellError;
use crate::lex::{Word, WordPart};
use crate::parse::{Item, Pipeline, Redirect, SeqOp, Sequence, SimpleCommand};

/// Per-stream captured-output cap. Matches Anthropic API tool-result limits.
pub const OUTPUT_CAP_BYTES: usize = 8 * 1024 * 1024;
/// Default wall-clock timeout for an external child.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);
/// Time between SIGTERM and SIGKILL when killing a child.
const KILL_GRACE: Duration = Duration::from_secs(5);
/// Polling interval for waitpid + deadline + cancel checks.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

pub struct VShell {
    cwd: PathBuf,
    env: HashMap<String, String>,
    exported: std::collections::HashSet<String>,
    last_exit: i32,
    timeout: Duration,
    cancel: Option<Arc<AtomicBool>>,
}

pub struct ExecResult {
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl Default for VShell {
    fn default() -> Self {
        Self::new()
    }
}

impl VShell {
    pub fn new() -> Self {
        Self {
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
            env: HashMap::new(),
            exported: Default::default(),
            last_exit: 0,
            timeout: DEFAULT_TIMEOUT,
            cancel: None,
        }
    }

    /// Set the wall-clock timeout for external children. Builtins are not
    /// subject to this (they run in-process and complete promptly).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Install a cancel token. Setting the flag terminates any in-flight
    /// external child (SIGTERM, then SIGKILL after a grace period).
    pub fn with_cancel(mut self, cancel: Arc<AtomicBool>) -> Self {
        self.cancel = Some(cancel);
        self
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    pub fn env(&self, key: &str) -> Option<&str> {
        self.env.get(key).map(|s| s.as_str())
    }

    pub fn set_env(&mut self, key: &str, value: &str) {
        self.env.insert(key.to_string(), value.to_string());
    }

    pub fn execute(&mut self, script: &str) -> ExecResult {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let code = match self.run_script(script, &mut stdout, &mut stderr) {
            Ok(c) => c,
            Err(e) => {
                let _ = writeln!(&mut stderr, "vshell: {e}");
                2
            }
        };
        ExecResult { exit_code: code, stdout, stderr }
    }

    fn run_script(
        &mut self,
        script: &str,
        stdout: &mut Vec<u8>,
        stderr: &mut Vec<u8>,
    ) -> Result<i32, ShellError> {
        let items = crate::parse::parse(script)?;
        for item in items {
            match item {
                Item::BareAssign(k, v) => {
                    let val = self.expand_word(&v, stderr)?;
                    self.env.insert(k, val);
                    self.last_exit = 0;
                }
                Item::Seq(seq) => match self.run_sequence(&seq, Io::capture(stdout, stderr))? {
                    Flow::Done(code) => self.last_exit = code,
                    Flow::Exit(code) => return Ok(code),
                },
            }
        }
        Ok(self.last_exit)
    }

    fn run_sequence(&mut self, seq: &Sequence, io: Io<'_>) -> Result<Flow, ShellError> {
        let Io { stdout, stderr, .. } = io;
        let first = self.run_pipeline(&seq.head, Io::capture(stdout, stderr))?;
        let mut last = match first {
            Flow::Exit(c) => return Ok(Flow::Exit(c)),
            Flow::Done(c) => c,
        };
        for (op, p) in &seq.tail {
            let run = match op {
                SeqOp::AndIf => last == 0,
                SeqOp::OrIf => last != 0,
            };
            if !run {
                continue;
            }
            match self.run_pipeline(p, Io::capture(stdout, stderr))? {
                Flow::Exit(c) => return Ok(Flow::Exit(c)),
                Flow::Done(c) => last = c,
            }
        }
        Ok(Flow::Done(last))
    }

    fn run_pipeline(&mut self, p: &Pipeline, io: Io<'_>) -> Result<Flow, ShellError> {
        if p.stages.len() == 1 {
            return self.run_simple(&p.stages[0], io);
        }
        // Multi-stage pipeline.
        //
        // Strategy:
        //  - For each stage build either a Child (external) or a Builtin closure.
        //  - Stages are connected via OS pipes (std::io::pipe was stabilised
        //    too recently; use a Vec<u8> buffer for in-process built-ins and
        //    Stdio::piped() between externals).
        //  - We thread data left to right, collecting captured stdout at the
        //    end into `io.stdout`.

        // We implement pipelines as: each stage reads from the previous stage's
        // output buffer and writes to its own. This is not concurrent but is
        // simple and correct for our use case (small commands, captured I/O).
        let Io { stdout, stderr, stdin } = io;
        let mut buf_in: Vec<u8> = stdin.unwrap_or_default();
        for (idx, stage) in p.stages.iter().enumerate() {
            let last = idx + 1 == p.stages.len();
            let mut buf_out = Vec::<u8>::new();
            let stage_io = if last {
                Io { stdout, stderr, stdin: Some(buf_in.clone()) }
            } else {
                Io { stdout: &mut buf_out, stderr, stdin: Some(buf_in.clone()) }
            };
            let flow = self.run_simple(stage, stage_io)?;
            match flow {
                Flow::Exit(c) => return Ok(Flow::Exit(c)),
                Flow::Done(c) => {
                    if last {
                        return Ok(Flow::Done(c));
                    }
                }
            }
            buf_in = buf_out;
        }
        Ok(Flow::Done(0))
    }

    fn run_simple(&mut self, cmd: &SimpleCommand, mut io: Io<'_>) -> Result<Flow, ShellError> {
        // Expand assignment values; record onetime env if cmd is to run.
        let mut local_env: Vec<(String, String)> = Vec::with_capacity(cmd.assigns.len());
        for (k, v) in &cmd.assigns {
            let val = self.expand_word(v, io.stderr)?;
            local_env.push((k.clone(), val));
        }
        // Expand argv.
        if cmd.argv.is_empty() {
            // Pure assignment-with-command was already handled at item level.
            // But this branch is reachable when a SimpleCommand had only
            // assigns + redirects with no command. Apply assigns to the shell
            // state, open & close redirects (touching files).
            for (k, v) in local_env {
                self.env.insert(k, v);
            }
            for r in &cmd.redirects {
                self.touch_redirect(r, io.stderr)?;
            }
            return Ok(Flow::Done(0));
        }
        let mut argv = Vec::with_capacity(cmd.argv.len());
        for w in &cmd.argv {
            argv.push(self.expand_word(w, io.stderr)?);
        }

        // Resolve redirects: open files now.
        let mut rio = ResolvedIo::default();
        for r in &cmd.redirects {
            self.resolve_redirect(r, &mut rio, io.stderr)?;
        }

        let name = argv[0].clone();
        if let Some(b) = Builtin::lookup(&name) {
            return self.run_builtin(b, &argv, &local_env, &cmd.redirects, &mut io);
        }
        // External — spawn.
        let mut child_cmd = Command::new(&name);
        child_cmd.args(&argv[1..]);
        child_cmd.current_dir(&self.cwd);
        child_cmd.env_clear();
        for k in &self.exported {
            if let Some(v) = self.env.get(k) {
                child_cmd.env(k, v);
            }
        }
        for (k, v) in &local_env {
            child_cmd.env(k, v);
        }

        // Wire stdio.
        let want_stderr_to_stdout = cmd.redirects.iter().any(|r| matches!(r, Redirect::Err2Out));

        if let Some(path) = rio.stdin_file.take() {
            child_cmd.stdin(Stdio::from(File::open(&path).map_err(ShellError::Io)?));
        } else if let Some(ref bytes) = io.stdin {
            if bytes.is_empty() {
                child_cmd.stdin(Stdio::null());
            } else {
                child_cmd.stdin(Stdio::piped());
            }
        } else {
            child_cmd.stdin(Stdio::null());
        }

        let stdout_to_file = rio.stdout_file.is_some();
        if let Some(f) = rio.stdout_file.take() {
            child_cmd.stdout(Stdio::from(f));
        } else {
            child_cmd.stdout(Stdio::piped());
        }

        if let Some(f) = rio.stderr_file.take() {
            child_cmd.stderr(Stdio::from(f));
        } else if want_stderr_to_stdout && !stdout_to_file {
            child_cmd.stderr(Stdio::piped());
        } else if want_stderr_to_stdout && stdout_to_file {
            // stdout went to a file; clone the same fd for stderr.
            // Re-open the file in append mode to match POSIX semantics for `>file 2>&1`.
            // Simplification: reopen the same path with append, which works for our redirects.
            if let Some(path) = redirect_target_path(cmd, self, io.stderr)? {
                let f = OpenOptions::new().append(true).open(path).map_err(ShellError::Io)?;
                child_cmd.stderr(Stdio::from(f));
            } else {
                child_cmd.stderr(Stdio::piped());
            }
        } else {
            child_cmd.stderr(Stdio::piped());
        }

        let mut child: Child = match child_cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let _ = writeln!(io.stderr, "vshell: {}: {}", name, e);
                return Ok(Flow::Done(127));
            }
        };

        // Push stdin from a background thread so a builtin's upstream output
        // larger than the pipe buffer can't deadlock the wait loop.
        let stdin_join = if let Some(bytes) = io.stdin.clone()
            && !bytes.is_empty()
            && let Some(mut sin) = child.stdin.take()
        {
            Some(std::thread::spawn(move || {
                let _ = sin.write_all(&bytes);
                drop(sin);
            }))
        } else {
            None
        };

        // Spawn reader threads with per-stream caps. On overflow they tag the
        // shared `overflow` cell (CAS, first stream wins); the wait loop will
        // kill the child promptly.
        let merging = want_stderr_to_stdout && !stdout_to_file;
        let overflow = Arc::new(AtomicUsize::new(OVERFLOW_NONE));
        let stdout_pipe = child.stdout.take();
        let stderr_pipe = child.stderr.take();
        let out_join =
            spawn_reader(stdout_pipe, OUTPUT_CAP_BYTES, overflow.clone(), OVERFLOW_STDOUT);
        let err_join =
            spawn_reader(stderr_pipe, OUTPUT_CAP_BYTES, overflow.clone(), OVERFLOW_STDERR);

        let deadline = Instant::now().checked_add(self.timeout);
        let outcome = wait_with_supervision(&mut child, deadline, self.cancel.as_ref(), &overflow);

        if let Some(j) = stdin_join {
            let _ = j.join();
        }
        let out_buf = out_join.map(|t| t.join().unwrap_or_default()).unwrap_or_default();
        let err_buf = err_join.map(|t| t.join().unwrap_or_default()).unwrap_or_default();

        match outcome {
            ChildOutcome::Exited(code) => {
                if !out_buf.is_empty() {
                    io.stdout.extend_from_slice(&out_buf);
                }
                if !err_buf.is_empty() {
                    if merging {
                        io.stdout.extend_from_slice(&err_buf);
                    } else {
                        io.stderr.extend_from_slice(&err_buf);
                    }
                }
                Ok(Flow::Done(code))
            }
            ChildOutcome::TimedOut => {
                let _ = writeln!(
                    io.stderr,
                    "vshell: {}: killed after {}s timeout",
                    name,
                    self.timeout.as_secs()
                );
                Ok(Flow::Done(124))
            }
            ChildOutcome::Cancelled => {
                let _ = writeln!(io.stderr, "vshell: {}: cancelled", name);
                Ok(Flow::Done(130))
            }
            ChildOutcome::StdoutOverflow => {
                let _ = writeln!(
                    io.stderr,
                    "vshell: {}: stdout exceeded {} MiB cap",
                    name,
                    OUTPUT_CAP_BYTES / (1024 * 1024)
                );
                Ok(Flow::Done(1))
            }
            ChildOutcome::StderrOverflow => {
                let _ = writeln!(
                    io.stderr,
                    "vshell: {}: stderr exceeded {} MiB cap",
                    name,
                    OUTPUT_CAP_BYTES / (1024 * 1024)
                );
                Ok(Flow::Done(1))
            }
        }
    }

    fn run_builtin(
        &mut self,
        b: Builtin,
        argv: &[String],
        local_env: &[(String, String)],
        redirects: &[Redirect],
        io: &mut Io<'_>,
    ) -> Result<Flow, ShellError> {
        // Resolve redirects (open files), but a builtin lives in our process,
        // so we write to file handles directly instead of dup'ing fds.
        let mut rio = ResolvedIo::default();
        for r in redirects {
            self.resolve_redirect(r, &mut rio, io.stderr)?;
        }
        let want_stderr_to_stdout = redirects.iter().any(|r| matches!(r, Redirect::Err2Out));

        // Builtins that mutate shell state must NOT be hidden by redirects on
        // success — we always apply the local_env to the temporary env table
        // first if the command itself doesn't touch shell state.
        // Per spec: stateful builtins (cd, export, unset, exit) in a pipeline
        // would be subshells in POSIX; we error out only when called inside a
        // pipeline. Detect "in pipeline" via the presence of a captured stdin.
        let in_pipeline = io.stdin.is_some();

        // Builtin output destinations.
        let mut out_local = Vec::<u8>::new();
        let mut err_local = Vec::<u8>::new();
        let stdin_bytes = io.stdin.clone().unwrap_or_default();

        // POSIX: env prefixes don't affect builtin execution (argv was already
        // expanded before this point). We just ignore local_env for built-ins
        // that don't have a child process — except `exit`, which still needs
        // a clean restore. The prefix's only effect on a builtin is being a
        // no-op; that matches dash and bash behaviour.
        let _ = local_env;

        let result = match b {
            Builtin::Echo => builtin_echo(argv, &mut out_local),
            Builtin::Pwd => builtin_pwd(&self.cwd, &mut out_local),
            Builtin::Cd => {
                if in_pipeline {
                    let _ = writeln!(&mut err_local, "cd: not supported inside a pipeline");
                    Ok(1)
                } else {
                    builtin_cd(self, argv, &mut err_local)
                }
            }
            Builtin::Export => {
                if in_pipeline {
                    let _ = writeln!(&mut err_local, "export: not supported inside a pipeline");
                    Ok(1)
                } else {
                    builtin_export(self, argv, &mut err_local)
                }
            }
            Builtin::Unset => {
                if in_pipeline {
                    let _ = writeln!(&mut err_local, "unset: not supported inside a pipeline");
                    Ok(1)
                } else {
                    builtin_unset(self, argv)
                }
            }
            Builtin::Exit => {
                let code =
                    if argv.len() > 1 { argv[1].parse().unwrap_or(2) } else { self.last_exit };
                io.stdout.extend_from_slice(&out_local);
                io.stderr.extend_from_slice(&err_local);
                return Ok(Flow::Exit(code));
            }
            Builtin::True => Ok(0),
            Builtin::False => Ok(1),
            Builtin::Colon => Ok(0),
        };

        let code = match result {
            Ok(c) => c,
            Err(e) => {
                let _ = writeln!(&mut err_local, "{e}");
                1
            }
        };

        // Route outputs to file or to io.
        // stdout
        if let Some(mut f) = rio.stdout_file.take() {
            let _ = f.write_all(&out_local);
            // If merging and stdout went to file, route stderr to same file too.
            if want_stderr_to_stdout {
                let _ = f.write_all(&err_local);
                err_local.clear();
            }
        } else {
            io.stdout.extend_from_slice(&out_local);
            if want_stderr_to_stdout {
                io.stdout.extend_from_slice(&err_local);
                err_local.clear();
            }
        }
        // stderr (if not already routed)
        if let Some(mut f) = rio.stderr_file.take() {
            let _ = f.write_all(&err_local);
        } else if !err_local.is_empty() {
            io.stderr.extend_from_slice(&err_local);
        }

        // The stdin we received gets consumed but builtins don't read it. That
        // matches the simplest correct behaviour from the spec.
        let _ = stdin_bytes;

        Ok(Flow::Done(code))
    }

    fn touch_redirect(&self, r: &Redirect, _stderr: &mut Vec<u8>) -> Result<(), ShellError> {
        match r {
            Redirect::StdoutTrunc(w) | Redirect::StderrTrunc(w) => {
                let p = self.expand_word_to_path(w)?;
                File::create(&p).map_err(ShellError::Io)?;
            }
            Redirect::StdoutAppend(w) | Redirect::StderrAppend(w) => {
                let p = self.expand_word_to_path(w)?;
                OpenOptions::new().create(true).append(true).open(&p).map_err(ShellError::Io)?;
            }
            Redirect::Stdin(_) | Redirect::Err2Out => {}
        }
        Ok(())
    }

    fn resolve_redirect(
        &mut self,
        r: &Redirect,
        rio: &mut ResolvedIo,
        stderr: &mut Vec<u8>,
    ) -> Result<(), ShellError> {
        match r {
            Redirect::StdoutTrunc(w) => {
                let p = self.expand_word(w, stderr)?;
                let p = self.path_for(&p);
                rio.stdout_file = Some(File::create(&p).map_err(ShellError::Io)?);
            }
            Redirect::StdoutAppend(w) => {
                let p = self.expand_word(w, stderr)?;
                let p = self.path_for(&p);
                rio.stdout_file = Some(
                    OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&p)
                        .map_err(ShellError::Io)?,
                );
            }
            Redirect::Stdin(w) => {
                let p = self.expand_word(w, stderr)?;
                let p = self.path_for(&p);
                rio.stdin_file = Some(p);
            }
            Redirect::StderrTrunc(w) => {
                let p = self.expand_word(w, stderr)?;
                let p = self.path_for(&p);
                rio.stderr_file = Some(File::create(&p).map_err(ShellError::Io)?);
            }
            Redirect::StderrAppend(w) => {
                let p = self.expand_word(w, stderr)?;
                let p = self.path_for(&p);
                rio.stderr_file = Some(
                    OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&p)
                        .map_err(ShellError::Io)?,
                );
            }
            Redirect::Err2Out => { /* handled by caller */ }
        }
        Ok(())
    }

    fn path_for(&self, s: &str) -> PathBuf {
        let p = Path::new(s);
        if p.is_absolute() { p.to_path_buf() } else { self.cwd.join(p) }
    }

    fn expand_word_to_path(&self, w: &Word) -> Result<PathBuf, ShellError> {
        // Used only by `touch_redirect`, which fires when a SimpleCommand has
        // redirects but no command. No cmd-sub is allowed in that path.
        let s = self.expand_word_no_subst(w)?;
        Ok(self.path_for(&s))
    }

    fn expand_word(&mut self, w: &Word, stderr: &mut Vec<u8>) -> Result<String, ShellError> {
        let mut out = String::new();
        for part in w {
            match part {
                WordPart::Literal(s) => out.push_str(s),
                WordPart::Var(name) => {
                    if let Some(v) = self.env.get(name) {
                        out.push_str(v);
                    }
                }
                WordPart::CmdSub(body) => {
                    let mut sub_out = Vec::new();
                    let mut sub_err = Vec::new();
                    let code = match self.run_script(body, &mut sub_out, &mut sub_err) {
                        Ok(c) => c,
                        Err(e) => {
                            let _ = writeln!(stderr, "vshell: $(): {e}");
                            return Err(e);
                        }
                    };
                    stderr.extend_from_slice(&sub_err);
                    self.last_exit = code;
                    // Trim trailing newlines.
                    let mut s = String::from_utf8_lossy(&sub_out).into_owned();
                    while s.ends_with('\n') {
                        s.pop();
                    }
                    out.push_str(&s);
                }
            }
        }
        Ok(out)
    }

    fn expand_word_no_subst(&self, w: &Word) -> Result<String, ShellError> {
        let mut out = String::new();
        for part in w {
            match part {
                WordPart::Literal(s) => out.push_str(s),
                WordPart::Var(name) => {
                    if let Some(v) = self.env.get(name) {
                        out.push_str(v);
                    }
                }
                WordPart::CmdSub(_) => {
                    return Err(ShellError::ParseError {
                        msg: "command substitution not allowed here".into(),
                        pos: 0,
                    });
                }
            }
        }
        Ok(out)
    }
}

fn redirect_target_path(
    cmd: &SimpleCommand,
    sh: &mut VShell,
    stderr: &mut Vec<u8>,
) -> Result<Option<PathBuf>, ShellError> {
    for r in &cmd.redirects {
        match r {
            Redirect::StdoutTrunc(w) | Redirect::StdoutAppend(w) => {
                let p = sh.expand_word(w, stderr)?;
                return Ok(Some(sh.path_for(&p)));
            }
            _ => {}
        }
    }
    Ok(None)
}

#[derive(Default)]
struct ResolvedIo {
    stdin_file: Option<PathBuf>,
    stdout_file: Option<File>,
    stderr_file: Option<File>,
}

pub(crate) struct Io<'a> {
    stdout: &'a mut Vec<u8>,
    stderr: &'a mut Vec<u8>,
    stdin: Option<Vec<u8>>,
}

impl<'a> Io<'a> {
    fn capture(stdout: &'a mut Vec<u8>, stderr: &'a mut Vec<u8>) -> Self {
        Io { stdout, stderr, stdin: None }
    }
}

#[derive(Debug, Clone, Copy)]
enum Builtin {
    Echo,
    Pwd,
    Cd,
    Export,
    Unset,
    Exit,
    True,
    False,
    Colon,
}

impl Builtin {
    fn lookup(name: &str) -> Option<Self> {
        Some(match name {
            "echo" => Builtin::Echo,
            "pwd" => Builtin::Pwd,
            "cd" => Builtin::Cd,
            "export" => Builtin::Export,
            "unset" => Builtin::Unset,
            "exit" => Builtin::Exit,
            "true" => Builtin::True,
            "false" => Builtin::False,
            ":" => Builtin::Colon,
            _ => return None,
        })
    }
}

enum Flow {
    Done(i32),
    Exit(i32),
}

fn builtin_echo(argv: &[String], out: &mut Vec<u8>) -> Result<i32, String> {
    let mut newline = true;
    let mut start = 1;
    if argv.get(1).map(|s| s.as_str()) == Some("-n") {
        newline = false;
        start = 2;
    }
    let mut first = true;
    for a in &argv[start..] {
        if !first {
            out.push(b' ');
        }
        first = false;
        out.extend_from_slice(a.as_bytes());
    }
    if newline {
        out.push(b'\n');
    }
    Ok(0)
}

fn builtin_pwd(cwd: &Path, out: &mut Vec<u8>) -> Result<i32, String> {
    out.extend_from_slice(cwd.as_os_str().as_encoded_bytes());
    out.push(b'\n');
    Ok(0)
}

fn builtin_cd(sh: &mut VShell, argv: &[String], err: &mut Vec<u8>) -> Result<i32, String> {
    let target: PathBuf = if argv.len() <= 1 {
        match sh.env.get("HOME") {
            Some(h) => PathBuf::from(h),
            None => {
                let _ = writeln!(err, "cd: HOME not set");
                return Ok(1);
            }
        }
    } else {
        PathBuf::from(&argv[1])
    };
    let resolved = if target.is_absolute() { target } else { sh.cwd.join(target) };
    match std::fs::canonicalize(&resolved) {
        Ok(p) => {
            sh.cwd = p;
            Ok(0)
        }
        Err(e) => {
            let _ = writeln!(err, "cd: {}: {}", resolved.display(), e);
            Ok(1)
        }
    }
}

fn builtin_export(sh: &mut VShell, argv: &[String], err: &mut Vec<u8>) -> Result<i32, String> {
    if argv.len() == 1 {
        let _ = writeln!(err, "export: missing argument");
        return Ok(1);
    }
    for a in &argv[1..] {
        match a.find('=') {
            Some(eq) => {
                let (k, v) = (&a[..eq], &a[eq + 1..]);
                if k.is_empty() {
                    let _ = writeln!(err, "export: invalid name");
                    return Ok(1);
                }
                sh.env.insert(k.to_string(), v.to_string());
                sh.exported.insert(k.to_string());
            }
            None => {
                sh.exported.insert(a.clone());
            }
        }
    }
    Ok(0)
}

fn builtin_unset(sh: &mut VShell, argv: &[String]) -> Result<i32, String> {
    for a in &argv[1..] {
        sh.env.remove(a);
        sh.exported.remove(a);
    }
    Ok(0)
}

// Overflow sentinel values stored in the shared AtomicUsize. The reader
// threads compare-and-swap from `OVERFLOW_NONE`, so the first stream to
// overflow wins and `wait_with_supervision` reports that one.
const OVERFLOW_NONE: usize = 0;
const OVERFLOW_STDOUT: usize = 1;
const OVERFLOW_STDERR: usize = 2;

enum ChildOutcome {
    Exited(i32),
    TimedOut,
    Cancelled,
    StdoutOverflow,
    StderrOverflow,
}

/// Spawn a reader thread that captures up to `cap` bytes. If the child writes
/// more than `cap`, the thread records its tag in `overflow` (CAS, so the
/// first overflow wins) and keeps draining the pipe so the kernel never
/// blocks the child on a full pipe — the supervisor's SIGTERM is what
/// actually stops it. Returns `None` if `stream` was `None`.
fn spawn_reader<R>(
    stream: Option<R>,
    cap: usize,
    overflow: Arc<AtomicUsize>,
    tag: usize,
) -> Option<std::thread::JoinHandle<Vec<u8>>>
where
    R: Read + Send + 'static,
{
    let mut s = stream?;
    Some(std::thread::spawn(move || {
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 8192];
        let mut overflowed = false;
        loop {
            match s.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    if overflowed {
                        // Discard; keep draining so the child doesn't block.
                        continue;
                    }
                    let remaining = cap.saturating_sub(buf.len());
                    if n <= remaining {
                        buf.extend_from_slice(&chunk[..n]);
                    } else {
                        buf.extend_from_slice(&chunk[..remaining]);
                        overflowed = true;
                        let _ = overflow.compare_exchange(
                            OVERFLOW_NONE,
                            tag,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        );
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
        buf
    }))
}

/// Poll for child exit while watching the deadline, cancel token, and output
/// overflow flag. On any trigger, send SIGTERM, then SIGKILL after
/// `KILL_GRACE` if the child hasn't exited.
///
/// Uses `Child::try_wait` (a thin wrapper around `waitpid(WNOHANG)`) rather
/// than calling `libc::waitpid` directly so std's internal `Child` bookkeeping
/// stays consistent — otherwise `Child::drop` could try to reap a pid that
/// we already reaped.
fn wait_with_supervision(
    child: &mut Child,
    deadline: Option<Instant>,
    cancel: Option<&Arc<AtomicBool>>,
    overflow: &Arc<AtomicUsize>,
) -> ChildOutcome {
    let pid = child.id() as libc::pid_t;
    let mut killed_for: Option<KillReason> = None;
    let mut sigkill_after: Option<Instant> = None;

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let code = status.code().unwrap_or_else(|| {
                    use std::os::unix::process::ExitStatusExt;
                    status.signal().map(|s| 128 + s).unwrap_or(128)
                });
                return outcome_for(killed_for, code);
            }
            Ok(None) => { /* still running */ }
            Err(_) => return outcome_for(killed_for, 128),
        }

        if killed_for.is_none() {
            let reason = if deadline.is_some_and(|d| Instant::now() >= d) {
                Some(KillReason::Timeout)
            } else if cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
                Some(KillReason::Cancel)
            } else {
                match overflow.load(Ordering::SeqCst) {
                    OVERFLOW_STDOUT => Some(KillReason::StdoutOverflow),
                    OVERFLOW_STDERR => Some(KillReason::StderrOverflow),
                    _ => None,
                }
            };
            if let Some(r) = reason {
                send_signal(pid, libc::SIGTERM);
                sigkill_after = Instant::now().checked_add(KILL_GRACE);
                killed_for = Some(r);
            }
        } else if let Some(t) = sigkill_after
            && Instant::now() >= t
        {
            send_signal(pid, libc::SIGKILL);
            sigkill_after = None;
        }

        std::thread::sleep(POLL_INTERVAL);
    }
}

#[derive(Clone, Copy)]
enum KillReason {
    Timeout,
    Cancel,
    StdoutOverflow,
    StderrOverflow,
}

fn outcome_for(killed_for: Option<KillReason>, code: i32) -> ChildOutcome {
    match killed_for {
        Some(KillReason::Timeout) => ChildOutcome::TimedOut,
        Some(KillReason::Cancel) => ChildOutcome::Cancelled,
        Some(KillReason::StdoutOverflow) => ChildOutcome::StdoutOverflow,
        Some(KillReason::StderrOverflow) => ChildOutcome::StderrOverflow,
        None => ChildOutcome::Exited(code),
    }
}

fn send_signal(pid: libc::pid_t, sig: libc::c_int) {
    // SAFETY: standard libc call; failure modes (ESRCH if the child already
    // exited and was reaped) are benign here.
    unsafe {
        libc::kill(pid, sig);
    }
}
