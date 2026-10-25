//! Stdio transport: spawn a child process and exchange newline-delimited
//! JSON over its stdin/stdout. Used by most local MCP packages.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use wire::NdjsonSplitter;

use super::Transport;

/// Stdio transport: owns a child process, writes to its stdin, reads
/// from its stdout. The child is killed on drop so callers can't leak
/// processes by losing a client handle.
pub struct StdioTransport {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    // `BufReader::read_until` is the natural read primitive but stdout
    // can deliver chunks that don't align to line boundaries (rare in
    // practice for MCP — most servers `println!` — but we already have
    // `NdjsonSplitter` for exactly this, so use it.)
    reader: Mutex<ReaderState>,
}

struct ReaderState {
    reader: BufReader<ChildStdout>,
    splitter: NdjsonSplitter,
}

impl StdioTransport {
    /// Spawn `command args...` with piped stdin/stdout. Inherits stderr
    /// so server diagnostics land in the user's terminal (and aren't
    /// silently dropped on a buffer that no one's reading).
    pub fn spawn(command: &str, args: &[&str]) -> Result<Self, String> {
        let mut cmd = Command::new(command);
        cmd.args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
        let mut child = cmd.spawn().map_err(|e| format!("spawn {command}: {e}"))?;
        let stdin = child.stdin.take().ok_or_else(|| "stdin not piped".to_string())?;
        let stdout = child.stdout.take().ok_or_else(|| "stdout not piped".to_string())?;
        Ok(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            reader: Mutex::new(ReaderState {
                reader: BufReader::new(stdout),
                splitter: NdjsonSplitter::new(),
            }),
        })
    }

    /// PID of the child, for tests that want to verify cleanup.
    #[cfg(test)]
    pub fn child_id(&self) -> u32 {
        self.child.lock().unwrap().id()
    }
}

impl Transport for StdioTransport {
    fn send_line(&self, line: &[u8]) -> Result<(), String> {
        let mut stdin = self.stdin.lock().map_err(|e| e.to_string())?;
        stdin.write_all(line).map_err(|e| format!("write: {e}"))?;
        stdin.write_all(b"\n").map_err(|e| format!("write: {e}"))?;
        stdin.flush().map_err(|e| format!("flush: {e}"))?;
        Ok(())
    }

    fn recv_line(&self) -> Result<Vec<u8>, String> {
        let mut state = self.reader.lock().map_err(|e| e.to_string())?;
        loop {
            if let Some(line) = state.splitter.pop_line() {
                return Ok(line);
            }
            // Pull the next chunk from the BufReader. fill_buf returns an
            // empty slice on EOF. Copy into a local before pushing to the
            // splitter so we don't hold two &mut borrows on `state`.
            let chunk = {
                let buf = state.reader.fill_buf().map_err(|e| format!("read: {e}"))?;
                if buf.is_empty() {
                    return Err("stdout closed".into());
                }
                let owned = buf.to_vec();
                let n = owned.len();
                state.reader.consume(n);
                owned
            };
            state
                .splitter
                .push(&chunk)
                .map_err(|e| format!("stdout line exceeds 4 MiB cap: {e}"))?;
        }
    }
}

impl Drop for StdioTransport {
    fn drop(&mut self) {
        // Best-effort: kill + reap so the child doesn't linger or become a
        // zombie. Errors here mean the child is already dead, which is
        // exactly what we want.
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Native-owned MCP stream. The caller retains the registry until its explicit
/// cleanup barrier; dropping this transport requests cancellation, not proof.
/// This transport intentionally has no blocking Drop or background thread.
pub struct OwnedStdioTransport {
    state: Mutex<OwnedReaderState>,
    stopped: std::sync::atomic::AtomicBool,
    _registry: std::sync::Arc<vshell::ownership::OwnerRegistry>,
}

struct OwnedReaderState {
    run: vshell::ownership::OwnedRun,
    splitter: NdjsonSplitter,
    diagnostics: Vec<u8>,
    stdout_eof: bool,
}

impl OwnedReaderState {
    fn pump(&mut self) -> Result<(), String> {
        use std::io::Read as _;
        // Each tick has a finite read budget. Native ownership controls the
        // per-stream total cap; NdjsonSplitter retains its existing 4 MiB cap.
        let mut bytes = [0_u8; 16384];
        for _ in 0..4 {
            match self.run.read(&mut bytes) {
                Ok(0) => {
                    self.stdout_eof = true;
                    break;
                }
                Ok(n) => self
                    .splitter
                    .push(&bytes[..n])
                    .map_err(|e| format!("MCP stdout buffer exceeds 4 MiB: {e}"))?,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(format!("MCP owned stdout: {e}")),
            }
        }
        for _ in 0..4 {
            match self.run.read_stderr(&mut bytes) {
                Ok(0) => break,
                Ok(n) => {
                    self.diagnostics.extend_from_slice(&bytes[..n]);
                    if self.diagnostics.len() > 8192 {
                        let excess = self.diagnostics.len() - 8192;
                        self.diagnostics.drain(..excess);
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(format!("MCP owned stderr: {e}")),
            }
        }
        match self.run.poll() {
            vshell::ownership::RunState::Unknown(reason) => {
                Err(format!("MCP cleanup unconfirmed: {reason}"))
            }
            vshell::ownership::RunState::Retired(result) if !result.success() => Err(format!(
                "MCP owned server failed (reason {}, exit {:?}, signal {:?})",
                result.reason, result.payload_exit_code, result.payload_signal
            )),
            _ => Ok(()),
        }
    }
}

impl OwnedStdioTransport {
    /// One long-lived server under the selected native owner. Lifetime and
    /// total-capture policy are explicit: one day, 64 MiB per stream. Individual
    /// send/receive waits and the enclosing JSON-RPC request each have 30 s
    /// budgets. Limits cause cancellation/failure, never partial success.
    pub fn spawn(
        registry: std::sync::Arc<vshell::ownership::OwnerRegistry>,
        command: &str,
        args: &[&str],
    ) -> Result<Self, String> {
        let run = registry
            .spawn(vshell::ownership::CommandSpec {
                program: command.into(),
                args: args.iter().map(|arg| (*arg).into()).collect(),
                cwd: None,
                env: Vec::new(),
                limits: vshell::ownership::Limits {
                    deadline: std::time::Duration::from_secs(86400),
                    capture_bytes: 64 * 1024 * 1024,
                    input_bytes: u64::MAX,
                },
            })
            .map_err(|e| format!("owned MCP spawn {command}: {e}"))?;
        Ok(Self {
            state: Mutex::new(OwnedReaderState {
                run,
                splitter: NdjsonSplitter::new(),
                diagnostics: Vec::new(),
                stdout_eof: false,
            }),
            stopped: std::sync::atomic::AtomicBool::new(false),
            _registry: registry,
        })
    }

    fn lock_until(
        &self,
        deadline: std::time::Instant,
    ) -> Result<std::sync::MutexGuard<'_, OwnedReaderState>, String> {
        loop {
            if self.stopped.load(std::sync::atomic::Ordering::Acquire)
                || std::time::Instant::now() >= deadline
            {
                self.request_stop();
                return Err("MCP operation cancelled or timed out while queued".into());
            }
            match self.state.try_lock() {
                Ok(state) => return Ok(state),
                Err(std::sync::TryLockError::WouldBlock) => {
                    std::thread::sleep(std::time::Duration::from_millis(10))
                }
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    self.request_stop();
                    return Err("MCP owner lock poisoned".into());
                }
            }
        }
    }

    fn check_wait(
        &self,
        state: &mut OwnedReaderState,
        deadline: std::time::Instant,
    ) -> Result<(), String> {
        if self.stopped.load(std::sync::atomic::Ordering::Acquire)
            || std::time::Instant::now() >= deadline
        {
            self.stopped.store(true, std::sync::atomic::Ordering::Release);
            state.run.cancel();
            return Err(
                "MCP operation cancelled or timed out; retirement remains registry-owned".into()
            );
        }
        if let Err(error) = state.pump() {
            self.stopped.store(true, std::sync::atomic::Ordering::Release);
            state.run.cancel();
            return Err(error);
        }
        Ok(())
    }
}

impl Transport for OwnedStdioTransport {
    fn send_line(&self, line: &[u8]) -> Result<(), String> {
        self.send_line_until(line, std::time::Instant::now() + std::time::Duration::from_secs(30))
    }
    fn recv_line(&self) -> Result<Vec<u8>, String> {
        self.recv_line_until(std::time::Instant::now() + std::time::Duration::from_secs(30))
    }
    fn send_line_until(&self, line: &[u8], deadline: std::time::Instant) -> Result<(), String> {
        let mut state = self.lock_until(deadline)?;
        // The framing cap includes the newline. Avoid allocating a second
        // full request copy and preserve the exact caller bytes.
        if line.len() >= 4 * 1024 * 1024 {
            self.stopped.store(true, std::sync::atomic::Ordering::Release);
            state.run.cancel();
            return Err("MCP outbound line exceeds 4 MiB framing cap".into());
        }
        for part in [line, &b"\n"[..]] {
            let mut position = 0;
            while position < part.len() {
                self.check_wait(&mut state, deadline)?;
                match state.run.write(&part[position..]) {
                    Ok(0) => {
                        state.run.cancel();
                        self.stopped.store(true, std::sync::atomic::Ordering::Release);
                        return Err("MCP owned input closed".into());
                    }
                    Ok(n) => position += n,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(10))
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        state.run.cancel();
                        self.stopped.store(true, std::sync::atomic::Ordering::Release);
                        return Err(format!("MCP owned input: {e}"));
                    }
                }
            }
        }
        Ok(())
    }

    fn recv_line_until(&self, deadline: std::time::Instant) -> Result<Vec<u8>, String> {
        let mut state = self.lock_until(deadline)?;
        loop {
            self.check_wait(&mut state, deadline)?;
            if let Some(line) = state.splitter.pop_line() {
                return Ok(line);
            }
            if state.stdout_eof {
                state.run.cancel();
                self.stopped.store(true, std::sync::atomic::Ordering::Release);
                return Err("MCP stdout closed before complete response".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    fn request_stop(&self) {
        self.stopped.store(true, std::sync::atomic::Ordering::Release);
        // Never wait behind an operation holding state. That operation checks
        // the atomic request on every bounded polling iteration.
        if let Ok(state) = self.state.try_lock() {
            state.run.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stdio_echo_round_trip() {
        // `cat` echoes whatever we send. Useful for verifying the
        // line-buffering wiring without any MCP-specific framing.
        let t = StdioTransport::spawn("cat", &[]).unwrap();
        t.send_line(b"hello").unwrap();
        let line = t.recv_line().unwrap();
        assert_eq!(line, b"hello");
    }

    #[test]
    fn stdio_multiple_lines_in_one_read() {
        // `printf` writes both lines in one go; the splitter has to peel
        // them apart on the read side.
        let t = StdioTransport::spawn("sh", &["-c", "printf 'one\\ntwo\\n'"]).unwrap();
        assert_eq!(t.recv_line().unwrap(), b"one");
        assert_eq!(t.recv_line().unwrap(), b"two");
        // Third read hits EOF.
        let err = t.recv_line().unwrap_err();
        assert!(err.contains("closed"), "got: {err}");
    }

    #[test]
    fn stdio_spawn_failure_surfaces_inline() {
        let err = match StdioTransport::spawn("definitely-not-a-real-binary-xyzzy", &[]) {
            Err(e) => e,
            Ok(_) => panic!("expected spawn failure"),
        };
        assert!(err.contains("spawn"), "got: {err}");
    }
}
