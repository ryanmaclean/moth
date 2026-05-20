//! Tiny in-process POSIX shell subset.
//!
//! Parses and interprets common agent shell calls (`echo`, `cd`, simple
//! pipelines, redirects, `&&`/`||`/`;`, command substitution, env prefixes)
//! without spawning a full bash. External binaries still go through
//! `std::process::Command`; built-ins run in-process and mutate `VShell`
//! state directly.
//!
//! Std-only, no dependencies. Full POSIX semantics live in a separate
//! `extern` sandbox; this crate handles the 90% case.

mod ast;
mod interp;
mod parser;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub struct VShell {
    cwd: PathBuf,
    vars: BTreeMap<String, String>,
    exported: std::collections::BTreeSet<String>,
    last_exit: i32,
}

pub struct ExecResult {
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

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

impl Default for VShell {
    fn default() -> Self {
        Self::new()
    }
}

impl VShell {
    pub fn new() -> Self {
        Self {
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
            vars: BTreeMap::new(),
            exported: std::collections::BTreeSet::new(),
            last_exit: 0,
        }
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    pub fn env(&self, key: &str) -> Option<&str> {
        self.vars.get(key).map(String::as_str)
    }

    pub fn set_env(&mut self, key: &str, value: &str) {
        self.vars.insert(key.to_string(), value.to_string());
    }

    pub fn execute(&mut self, script: &str) -> ExecResult {
        let ast = match parser::parse(script) {
            Ok(a) => a,
            Err(e) => {
                let msg = format!("{e}\n");
                return ExecResult {
                    exit_code: 2,
                    stdout: Vec::new(),
                    stderr: msg.into_bytes(),
                };
            }
        };
        let mut out = Vec::new();
        let mut err = Vec::new();
        let code = interp::run(self, &ast, &mut out, &mut err);
        self.last_exit = code;
        ExecResult { exit_code: code, stdout: out, stderr: err }
    }
}
