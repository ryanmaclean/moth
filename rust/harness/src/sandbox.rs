//! Sandbox trait: shell execution boundary the Instance actor owns.

use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct ShellResult {
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct SandboxError(pub String);

pub trait Sandbox: Send + 'static {
    fn execute(&mut self, cmd: &str) -> Result<ShellResult, SandboxError>;
}

/// Mock sandbox. Records each command and pops the next pre-canned response.
/// If `responses` runs dry, every subsequent call yields exit 0 with empty
/// stdout/stderr — keeps tests that don't care about output terse.
pub struct MockSandbox {
    pub recorded: Mutex<Vec<String>>,
    pub responses: Mutex<Vec<ShellResult>>,
}

impl MockSandbox {
    pub fn new(responses: Vec<ShellResult>) -> Self {
        Self {
            recorded: Mutex::new(Vec::new()),
            responses: Mutex::new(responses),
        }
    }
}

impl Sandbox for MockSandbox {
    fn execute(&mut self, cmd: &str) -> Result<ShellResult, SandboxError> {
        self.recorded.lock().unwrap().push(cmd.to_string());
        let mut r = self.responses.lock().unwrap();
        Ok(if r.is_empty() {
            ShellResult { exit_code: 0, stdout: Vec::new(), stderr: Vec::new() }
        } else {
            r.remove(0)
        })
    }
}
