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
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("spawn {command}: {e}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "stdin not piped".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "stdout not piped".to_string())?;
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
        stdin
            .write_all(line)
            .map_err(|e| format!("write: {e}"))?;
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
