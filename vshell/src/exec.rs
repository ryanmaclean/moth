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
    owner_registry: Option<Arc<crate::ownership::OwnerRegistry>>,
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
            owner_registry: None,
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

    /// Select the CLI-owned native retirement barrier for external commands.
    /// The CLI must retain its Arc through explicit retirement; construction
    /// of this registry rejects unsupported targets before strict execution.
    pub fn with_owner_registry(mut self, registry: Arc<crate::ownership::OwnerRegistry>) -> Self {
        self.owner_registry = Some(registry);
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

        let name = argv[0].clone();
        if let Some(b) = Builtin::lookup(&name) {
            return self.run_builtin(b, &argv, &local_env, &cmd.redirects, &mut io);
        }
        if let Some(registry) = self.owner_registry.clone() {
            let redirects = self.resolve_owned_redirects(&cmd.redirects, io.stderr)?;
            return self.run_owned_external(&registry, &argv, &local_env, &redirects, &mut io);
        }
        let mut rio = ResolvedIo::default();
        for r in &cmd.redirects { self.resolve_redirect(r, &mut rio, io.stderr)?; }
        // External — legacy route for callers that did not select strict ownership.
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

    fn redirect_deadline(&self) -> Instant {
        Instant::now().checked_add(self.timeout).unwrap_or_else(Instant::now)
    }
    fn poll_redirect(&self, deadline: Instant) -> std::io::Result<()> {
        if self.cancel.as_ref().is_some_and(|flag| flag.load(Ordering::SeqCst)) {
            return Err(std::io::Error::new(std::io::ErrorKind::Interrupted, "redirect cancelled"));
        }
        if let Some(registry) = &self.owner_registry {
            registry.poll();
            if registry.admission_closed() {
                return Err(std::io::Error::new(std::io::ErrorKind::Interrupted, "owner admission closed during redirect"));
            }
        }
        if Instant::now() >= deadline {
            return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "redirect deadline expired"));
        }
        Ok(())
    }
    fn open_redirect(&self, path: &Path, read: bool, append: bool, deadline: Instant) -> std::io::Result<File> {
        loop {
            if self.owner_registry.is_some() { self.poll_redirect(deadline)?; }
            let mut options = OpenOptions::new();
            options.read(read).write(!read).create(!read).append(append).truncate(!read && !append);
            if self.owner_registry.is_some() {
                #[cfg(target_os = "freebsd")]
                { use std::os::unix::fs::OpenOptionsExt;
                  options.custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC); }
                #[cfg(not(target_os = "freebsd"))]
                { return Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "owned redirects require FreeBSD")); }
            }
            match options.open(path) {
                Ok(file) => return Ok(file),
                Err(error) if self.owner_registry.is_some() && !read && error.raw_os_error() == Some(libc::ENXIO) => {
                    // Nonblocking FIFO writer open waits for an actual reader;
                    // no blocking open can strand the actor before cancellation.
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(error) => return Err(error),
            }
        }
    }
    fn write_redirect(&self, file: &mut File, bytes: &[u8], deadline: Instant) -> std::io::Result<()> {
        if self.owner_registry.is_none() { return file.write_all(bytes); }
        let mut offset = 0;
        while offset < bytes.len() {
            self.poll_redirect(deadline)?;
            match file.write(&bytes[offset..]) {
                Ok(0) => return Err(std::io::ErrorKind::WriteZero.into()),
                Ok(n) => offset += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(10)),
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    fn resolve_owned_redirects(&mut self, redirects: &[Redirect], stderr: &mut Vec<u8>) -> Result<Vec<crate::ownership::OwnedRedirect>, ShellError> {
        use crate::ownership::OwnedRedirect as R;
        let mut result = Vec::new();
        for redirect in redirects {
            if matches!(redirect, Redirect::Err2Out) { result.push(R::ErrorToOutput); continue; }
            let word = match redirect {
                Redirect::Stdin(word) | Redirect::StdoutTrunc(word) | Redirect::StdoutAppend(word)
                | Redirect::StderrTrunc(word) | Redirect::StderrAppend(word) => word,
                Redirect::Err2Out => unreachable!(),
            };
            let expanded = self.expand_word(word, stderr)?;
            let path = self.path_for(&expanded);
            result.push(match redirect {
                Redirect::Stdin(_) => R::Input(path),
                Redirect::StdoutTrunc(_) => R::Output { path, append: false },
                Redirect::StdoutAppend(_) => R::Output { path, append: true },
                Redirect::StderrTrunc(_) => R::Error { path, append: false },
                Redirect::StderrAppend(_) => R::Error { path, append: true },
                Redirect::Err2Out => unreachable!(),
            });
        }
        Ok(result)
    }
    fn run_owned_external(
        &self, registry: &Arc<crate::ownership::OwnerRegistry>, argv: &[String],
        local_env: &[(String, String)], redirects: &[crate::ownership::OwnedRedirect], io: &mut Io<'_>,
    ) -> Result<Flow, ShellError> {
        use crate::ownership::{CommandSpec, Limits, OwnedRedirect, RunState};
        use std::ffi::OsString;
        // File/FIFO redirect setup belongs to a fresh payload-launch child
        // beneath the acquired owner. The shared actor never opens, waits on,
        // or relays these endpoints. A stdin redirect overrides pipeline data.
        let input = if redirects.iter().any(|r| matches!(r, OwnedRedirect::Input(_))) {
            io.stdin.take(); Vec::new()
        } else { io.stdin.take().unwrap_or_default() };
        let mut env = Vec::new();
        for key in &self.exported {
            if let Some(value) = self.env.get(key) { env.push((OsString::from(key), OsString::from(value))); }
        }
        env.extend(local_env.iter().map(|(key, value)| (OsString::from(key), OsString::from(value))));
        let mut run = registry.spawn_redirected(CommandSpec {
            program: OsString::from(&argv[0]), args: argv[1..].iter().map(OsString::from).collect(),
            cwd: Some(self.cwd.clone()), env,
            limits: Limits { deadline: self.timeout, capture_bytes: OUTPUT_CAP_BYTES as u64, input_bytes: input.len() as u64 },
        }, redirects).map_err(ShellError::Io)?;
        let stop_waiting = Instant::now().checked_add(self.timeout.saturating_add(Duration::from_secs(10)));
        let mut sent = 0; let mut input_stopped = false; let mut out_eof = false; let mut err_eof = false;
        let mut output = [0u8; 16_384];
        loop {
            if self.cancel.as_ref().is_some_and(|flag| flag.load(Ordering::SeqCst)) { run.cancel(); }
            if matches!(run.poll(), RunState::Running) && !input_stopped && sent < input.len() {
                match run.write(&input[sent..]) {
                    Ok(n) => sent += n,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {},
                    Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => {
                        // A terminal event can arrive inside write polling.
                        // Stop supplying shell input, then require the actual
                        // retirement receipt and preserve the payload status.
                        input_stopped = true;
                    },
                    Err(error) => return Err(ShellError::Io(error)),
                }
            }
            for error_stream in [false, true] {
                let eof = if error_stream { &mut err_eof } else { &mut out_eof };
                if !*eof {
                    match if error_stream { run.read_stderr(&mut output) } else { run.read(&mut output) } {
                        Ok(0) => *eof = true,
                        Ok(n) => if error_stream { io.stderr.extend_from_slice(&output[..n]); } else { io.stdout.extend_from_slice(&output[..n]); },
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {},
                        Err(error) => return Err(ShellError::Io(error)),
                    }
                }
            }
            if let RunState::Retired(result) = run.poll()
                && out_eof && err_eof {
                return Ok(Flow::Done(match result.reason {
                    0 => result.payload_exit_code.unwrap_or_else(|| 128 + result.payload_signal.unwrap_or(0)),
                    1 => 130, 2 => 124, _ => 1,
                }));
            }
            if stop_waiting.is_none_or(|deadline| Instant::now() >= deadline) {
                run.cancel();
                let _ = writeln!(io.stderr, "vshell: owned completion remains unconfirmed; registry retains owner");
                return Ok(Flow::Done(125));
            }
            std::thread::sleep(Duration::from_millis(10));
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

        let redirect_deadline = self.redirect_deadline();
        // Route outputs to file or to io.
        // stdout
        if let Some(mut f) = rio.stdout_file.take() {
            self.write_redirect(&mut f, &out_local, redirect_deadline).map_err(ShellError::Io)?;
            // If merging and stdout went to file, route stderr to same file too.
            if want_stderr_to_stdout {
                self.write_redirect(&mut f, &err_local, redirect_deadline).map_err(ShellError::Io)?;
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
            self.write_redirect(&mut f, &err_local, redirect_deadline).map_err(ShellError::Io)?;
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
                self.open_redirect(&p, false, false, self.redirect_deadline()).map_err(ShellError::Io)?;
            }
            Redirect::StdoutAppend(w) | Redirect::StderrAppend(w) => {
                let p = self.expand_word_to_path(w)?;
                self.open_redirect(&p, false, true, self.redirect_deadline()).map_err(ShellError::Io)?;
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
                rio.stdout_file = Some(self.open_redirect(&p, false, false, self.redirect_deadline()).map_err(ShellError::Io)?);
            }
            Redirect::StdoutAppend(w) => {
                let p = self.expand_word(w, stderr)?;
                let p = self.path_for(&p);
                rio.stdout_file = Some(
                    self.open_redirect(&p, false, true, self.redirect_deadline()).map_err(ShellError::Io)?,
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
                rio.stderr_file = Some(self.open_redirect(&p, false, false, self.redirect_deadline()).map_err(ShellError::Io)?);
            }
            Redirect::StderrAppend(w) => {
                let p = self.expand_word(w, stderr)?;
                let p = self.path_for(&p);
                rio.stderr_file = Some(
                    self.open_redirect(&p, false, true, self.redirect_deadline()).map_err(ShellError::Io)?,
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

#[cfg(all(test, target_os = "freebsd"))]
mod owned_redirect_tests {
    use super::*;
    use std::ffi::CString;
    use std::os::unix::fs::OpenOptionsExt;
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    fn shell(timeout: Duration) -> VShell {
        let agent = PathBuf::from(std::env::var_os("MOTH_OWNER_TEST_AGENT").expect("explicit admitted agent required"));
        VShell::new().with_timeout(timeout).with_owner_registry(crate::ownership::OwnerRegistry::new(agent).unwrap())
    }
    fn fifo() -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!("moth-owned-fifo-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("channel");
        let name = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        (root, path)
    }
    fn clean(root: PathBuf, path: PathBuf) { std::fs::remove_file(path).unwrap(); std::fs::remove_dir(root).unwrap(); }

    #[test]
    #[ignore = "admitted FreeBSD target and explicit outer owner required; unrun"]
    fn fifo_writer_open_waits_for_reader_but_observes_cancel() {
        let (root, path) = fifo();
        let cancel = Arc::new(AtomicBool::new(false));
        let sh = shell(Duration::from_secs(2)).with_cancel(cancel.clone());
        std::thread::scope(|scope| {
            scope.spawn(|| { std::thread::sleep(Duration::from_millis(50)); cancel.store(true, Ordering::SeqCst); });
            let error = sh.open_redirect(&path, false, false, sh.redirect_deadline()).unwrap_err();
            assert_eq!(error.kind(), std::io::ErrorKind::Interrupted);
        });
        assert!(sh.owner_registry.as_ref().unwrap().retire_until(Instant::now() + Duration::from_secs(1)));
        clean(root, path);
    }

    #[test]
    #[ignore = "admitted FreeBSD target and explicit outer owner required; unrun"]
    fn owned_fifo_open_precedes_real_payload_and_overrides_pipeline_input() {
        let (root, path) = fifo(); let marker = root.join("payload-started");
        let sh = shell(Duration::from_secs(5));
        let registry = sh.owner_registry.as_ref().unwrap().clone();
        std::thread::scope(|scope| {
            let reader = scope.spawn(|| {
                let program = std::env::current_exe().unwrap().to_str().unwrap().to_owned();
                let argv = vec![program, "--ignored".into(), "--exact".into(), "native_owner_tests::owner_fixture_entry".into(), "--nocapture".into(), "--test-threads=1".into()];
                let mut out = Vec::new(); let mut err = Vec::new();
                let mut io = Io { stdout: &mut out, stderr: &mut err, stdin: Some(b"IGNORED-PIPELINE".to_vec()) };
                let result = sh.run_owned_external(&registry, &argv,
                    &[("MOTH_OWNER_FIXTURE_MODE".into(), "announce-echo".into()), ("MOTH_OWNER_MARKER".into(), marker.to_str().unwrap().into())],
                    &[crate::ownership::OwnedRedirect::Input(path.clone())], &mut io).unwrap();
                assert!(matches!(result, Flow::Done(0))); out
            });
            std::thread::sleep(Duration::from_millis(100));
            assert!(!reader.is_finished()); assert!(!marker.exists(), "payload started before FIFO writer connected");
            let mut writer = sh.open_redirect(&path, false, false, sh.redirect_deadline()).unwrap();
            // Keep the connected writer empty: payload startup must not wait
            // for data or writer EOF to prove the ordinary open handshake.
            let marker_deadline = Instant::now() + Duration::from_secs(2);
            while !marker.exists() && Instant::now() < marker_deadline {
                registry.poll(); std::thread::sleep(Duration::from_millis(10));
            }
            assert!(marker.exists(), "empty connected writer did not release payload startup");
            sh.write_redirect(&mut writer, b"complete-fifo-data", sh.redirect_deadline()).unwrap(); drop(writer);
            let out = reader.join().unwrap();
            assert!(out.windows(b"complete-fifo-data".len()).any(|w| w == b"complete-fifo-data"));
            assert!(!out.windows(b"IGNORED-PIPELINE".len()).any(|w| w == b"IGNORED-PIPELINE"));
        });
        assert!(marker.exists()); std::fs::remove_file(marker).unwrap();
        assert!(registry.retire_until(Instant::now() + Duration::from_secs(1)));
        clean(root, path);
    }

    #[test]
    #[ignore = "admitted FreeBSD target and explicit outer owner required; unrun"]
    fn fifo_writer_retains_partial_data_while_reader_progresses() {
        let (root, path) = fifo();
        let sh = shell(Duration::from_secs(10));
        let expected = vec![0x61u8; 2 * 1024 * 1024];
        std::thread::scope(|scope| {
            let reader = scope.spawn(|| {
                let mut file = OpenOptions::new().read(true).custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC).open(&path).unwrap();
                let deadline = sh.redirect_deadline(); let mut received = Vec::new(); let mut bytes = [0u8; 1024];
                while received.len() < expected.len() {
                    sh.poll_redirect(deadline).unwrap();
                    match file.read(&mut bytes) {
                        Ok(n) => received.extend_from_slice(&bytes[..n]),
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {},
                        Err(e) => panic!("{e}"),
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                received
            });
            let mut writer = sh.open_redirect(&path, false, false, sh.redirect_deadline()).unwrap();
            sh.write_redirect(&mut writer, &expected, sh.redirect_deadline()).unwrap(); drop(writer);
            assert_eq!(reader.join().unwrap(), expected);
        });
        assert!(sh.owner_registry.as_ref().unwrap().retire_until(Instant::now() + Duration::from_secs(1)));
        clean(root, path);
    }

    #[test]
    #[ignore = "admitted FreeBSD target and explicit outer owner required; unrun"]
    fn fifo_full_output_pipe_observes_deadline_without_blocking_write_all() {
        let (root, path) = fifo();
        let sh = shell(Duration::from_millis(100));
        let mut reader = OpenOptions::new().read(true).custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC).open(&path).unwrap();
        let mut writer = sh.open_redirect(&path, false, false, sh.redirect_deadline()).unwrap();
        let error = sh.write_redirect(&mut writer, &vec![0x62; 2 * 1024 * 1024], sh.redirect_deadline()).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        let mut prefix = [0u8; 32]; assert!(reader.read(&mut prefix).unwrap() > 0);
        assert!(prefix.iter().all(|b| *b == 0x62));
        drop(writer); drop(reader);
        assert!(sh.owner_registry.as_ref().unwrap().retire_until(Instant::now() + Duration::from_secs(1)));
        clean(root, path);
    }
    #[test]
    #[ignore = "admitted FreeBSD target and explicit outer owner required; unrun"]
    fn external_fifo_stdout_backpressure_does_not_starve_stderr_or_input() {
        let (out_root, out_path) = fifo(); let (err_root, err_path) = fifo();
        let sh = shell(Duration::from_secs(15));
        let registry = sh.owner_registry.as_ref().unwrap().clone();
        std::thread::scope(|scope| {
            let consumer = scope.spawn(|| {
                let mut out = OpenOptions::new().read(true).custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC).open(&out_path).unwrap();
                let mut err = OpenOptions::new().read(true).custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC).open(&err_path).unwrap();
                let deadline = Instant::now() + Duration::from_secs(20);
                let mut errors = Vec::new(); let mut body_bytes = 0; let mut ready = false;
                let mut bytes = [0u8; 16_384];
                loop {
                    assert!(Instant::now() < deadline, "full duplex redirect stalled");
                    match err.read(&mut bytes) {
                        Ok(n) => errors.extend_from_slice(&bytes[..n]),
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {},
                        Err(e) => panic!("{e}"),
                    }
                    ready |= errors.windows(b"DUPLEX-STDERR-READY".len()).any(|w| w == b"DUPLEX-STDERR-READY");
                    // Deliberately withhold stdout reads until stderr proves
                    // readiness. The payload also waits for complete stdin.
                    if ready {
                        match out.read(&mut bytes) {
                            Ok(0) => break,
                            Ok(n) => body_bytes += bytes[..n].iter().filter(|b| **b == 0xff).count(),
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {},
                            Err(e) => panic!("{e}"),
                        }
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                assert!(ready); assert_eq!(body_bytes, 2 * 1024 * 1024);
            });
            let program = std::env::current_exe().unwrap().to_str().unwrap().to_owned();
            let argv = vec![program, "--ignored".into(), "--exact".into(), "native_owner_tests::owner_fixture_entry".into(), "--nocapture".into(), "--test-threads=1".into()];
            let mut captured_out = Vec::new(); let mut captured_err = Vec::new();
            let mut io = Io { stdout: &mut captured_out, stderr: &mut captured_err, stdin: Some(b"duplex-input".to_vec()) };
            let result = sh.run_owned_external(&registry, &argv, &[("MOTH_OWNER_FIXTURE_MODE".into(), "duplex".into())],
                &[crate::ownership::OwnedRedirect::Output { path: out_path.clone(), append: false },
                  crate::ownership::OwnedRedirect::Error { path: err_path.clone(), append: false }], &mut io).unwrap();
            assert!(matches!(result, Flow::Done(0)));
            consumer.join().unwrap();
        });
        assert!(registry.retire_until(Instant::now() + Duration::from_secs(2)));
        clean(out_root, out_path); clean(err_root, err_path);
    }

}
