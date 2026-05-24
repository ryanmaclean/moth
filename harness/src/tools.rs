//! Tool trait + registry.
//!
//! Each `Tool` declares its name, description, and JSON schema for the
//! model, and a `call` that takes raw JSON input + a context handle to
//! the running session. The Session iteration loop dispatches `tool_use`
//! blocks through the registry instead of hardcoding bash. Built-in
//! tools live next to this module; consumers can register their own.

use std::sync::Arc;
use std::sync::mpsc::sync_channel;

use actor::ActorRef;

use crate::instance::InstanceMsg;
use crate::model::ToolDef;

/// Per-call context handed to a tool. Currently exposes the Instance
/// actor (for shell execution); future hooks (run id, cancellation
/// token, log sink) attach here without changing the trait.
pub struct ToolCtx<'a> {
    pub instance: &'a ActorRef<InstanceMsg>,
}

#[derive(Debug, Clone)]
pub struct ToolError(pub String);

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ToolError {}

pub trait Tool: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON schema as a raw JSON object string.
    fn input_schema(&self) -> &str;
    fn call(&self, input: &str, ctx: &ToolCtx) -> Result<String, ToolError>;

    fn definition(&self) -> ToolDef {
        ToolDef {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema().to_string(),
        }
    }
}

/// Bash tool: routes the `command` field through the Instance actor, which
/// owns the Sandbox (with optional `AuditedShell` wrapping). Returns
/// stdout, then stderr (prefixed), then `(exit N)` if non-zero.
pub struct BashTool;

impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command in the agent's sandbox. Returns stdout, stderr, and exit code."
    }

    fn input_schema(&self) -> &str {
        r#"{"type":"object","properties":{"command":{"type":"string","description":"The shell command to execute."}},"required":["command"]}"#
    }

    fn call(&self, input: &str, ctx: &ToolCtx) -> Result<String, ToolError> {
        let cmd = extract_command(input)
            .ok_or_else(|| ToolError(format!("missing 'command' in tool input: {input}")))?;
        let (tx, rx) = sync_channel(1);
        if ctx.instance.send(InstanceMsg::Shell { cmd, reply: tx }).is_err() {
            return Err(ToolError("instance mailbox closed".into()));
        }
        let r = rx
            .recv()
            .map_err(|_| ToolError("instance dropped reply channel".into()))?
            .map_err(|e| ToolError(e.0))?;
        Ok(format_shell(&r))
    }
}

fn format_shell(r: &crate::sandbox::ShellResult) -> String {
    let mut s = String::new();
    if !r.stdout.is_empty() {
        s.push_str(&String::from_utf8_lossy(&r.stdout));
    }
    if !r.stderr.is_empty() {
        if !s.is_empty() && !s.ends_with('\n') {
            s.push('\n');
        }
        s.push_str("stderr: ");
        s.push_str(&String::from_utf8_lossy(&r.stderr));
    }
    if r.exit_code != 0 {
        if !s.is_empty() && !s.ends_with('\n') {
            s.push('\n');
        }
        s.push_str(&format!("(exit {})", r.exit_code));
    }
    s
}

fn extract_command(input: &str) -> Option<String> {
    let parsed = anthropic::json::parse(input.as_bytes()).ok()?;
    match parsed.get("command")? {
        anthropic::json::Json::Str(s) => Some(s.clone()),
        _ => None,
    }
}

/// Convenience: returns the default tool set (just bash for now; built-in
/// read/write/edit tools live in the `tools` crate so harness doesn't pull
/// in filesystem-specific code unless wanted).
pub fn default_tools() -> Vec<Arc<dyn Tool>> {
    vec![Arc::new(BashTool)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::Instance;
    use crate::sandbox::{MockSandbox, Sandbox};
    use actor::spawn;

    #[test]
    fn bash_tool_executes_via_instance() {
        let sb: Box<dyn Sandbox> = Box::new(MockSandbox::new(vec![crate::sandbox::ShellResult {
            exit_code: 0,
            stdout: b"hello\n".to_vec(),
            stderr: Vec::new(),
        }]));
        let inst = spawn(Instance::new("t", sb));
        let ctx = ToolCtx { instance: &inst.addr };
        let out = BashTool.call(r#"{"command":"echo hello"}"#, &ctx).unwrap();
        assert_eq!(out, "hello\n");
        inst.join().unwrap();
    }

    #[test]
    fn bash_tool_propagates_nonzero_exit_and_stderr() {
        let sb: Box<dyn Sandbox> = Box::new(MockSandbox::new(vec![crate::sandbox::ShellResult {
            exit_code: 2,
            stdout: b"out\n".to_vec(),
            stderr: b"oops\n".to_vec(),
        }]));
        let inst = spawn(Instance::new("t", sb));
        let ctx = ToolCtx { instance: &inst.addr };
        let out = BashTool.call(r#"{"command":"x"}"#, &ctx).unwrap();
        assert!(out.contains("out"));
        assert!(out.contains("stderr: oops"));
        assert!(out.contains("(exit 2)"));
        inst.join().unwrap();
    }

    #[test]
    fn bash_tool_rejects_bad_input() {
        let sb: Box<dyn Sandbox> = Box::new(MockSandbox::new(vec![]));
        let inst = spawn(Instance::new("t", sb));
        let ctx = ToolCtx { instance: &inst.addr };
        let err = BashTool.call("not json", &ctx).unwrap_err();
        assert!(err.0.contains("missing 'command'"));
        inst.join().unwrap();
    }

    #[test]
    fn definition_round_trips() {
        let def = BashTool.definition();
        assert_eq!(def.name, "bash");
        assert!(def.input_schema.contains("command"));
    }
}
