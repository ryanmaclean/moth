//! Private, single-run native lifecycle ownership. Not a public task protocol.
use std::ffi::OsString;
use std::time::Duration;

pub const PRIVATE_COMMAND: &str = "__moth_owner_v1";
pub const FRAME_LIMIT: usize = 65_536;
pub const CHUNK: usize = 16_384;
pub const WINDOW: usize = 32_768;
pub const QUEUE_LIMIT: usize = 256 * 1024;
pub const READY: u8 = 0x01;
pub const STARTED: u8 = 0x02;
pub const CREDIT: u8 = 0x03;
pub const START: u8 = 0x10;
pub const INPUT: u8 = 0x11;
pub const INPUT_EOF: u8 = 0x12;
pub const CANCEL: u8 = 0x13;
pub const STDOUT: u8 = 0x20;
pub const STDERR: u8 = 0x21;
pub const PAYLOAD_EXIT: u8 = 0x22;
pub const RETIRED: u8 = 0x23;
pub const UNKNOWN: u8 = 0x24;
pub const REJECT: u8 = 0x7f;

#[derive(Clone, Debug)]
pub struct Limits {
    pub deadline: Duration,
    pub capture_bytes: u64,
    pub input_bytes: u64,
}
#[derive(Debug)]
pub struct Launch {
    pub limits: Limits,
    pub program: OsString,
    pub arguments: Vec<OsString>,
    pub allow_early_input_close: bool,
}
impl Launch {
    pub fn parse(args: Vec<OsString>) -> Result<Self, String> {
        let allow_early_input_close = args.get(6).is_some_and(|arg| arg == "--allow-early-input-close");
        let separator = if allow_early_input_close { 7 } else { 6 };
        if args.len() < separator + 2 || args[0] != "--deadline-ms" || args[2] != "--capture-bytes"
            || args[4] != "--input-bytes" || args[separator] != "--" || args[separator + 1].is_empty() {
            return Err("invalid private owner arguments".into());
        }
        fn decimal(value: &OsString) -> Result<u64, String> {
            let text = value.to_str().ok_or("limit is not ASCII decimal")?;
            if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
                return Err("limit is not ASCII decimal".into());
            }
            text.parse().map_err(|_| "limit exceeds u64".into())
        }
        let deadline = decimal(&args[1])?;
        let capture_bytes = decimal(&args[3])?;
        let input_bytes = decimal(&args[5])?;
        if !(1..=86_400_000).contains(&deadline) || !(1..=67_108_864).contains(&capture_bytes) {
            return Err("owner limit outside supported range".into());
        }
        Ok(Self { limits: Limits { deadline: Duration::from_millis(deadline), capture_bytes, input_bytes },
            program: args[separator + 1].clone(), arguments: args[separator + 2..].to_vec(), allow_early_input_close })
    }
}

#[cfg(target_os = "freebsd")]
mod freebsd;

/// Called only in a fresh private executable role, before the normal agent
/// initializes signals, actors, providers or runlog. This function owns the
/// process lifetime and returns only after cleanup or pre-acquire rejection.
pub fn private_main(args: Vec<OsString>) -> i32 {
    #[cfg(target_os = "freebsd")]
    { freebsd::run(args) }
    #[cfg(not(target_os = "freebsd"))]
    { let _ = args; eprintln!("Moth native ownership is unsupported on this target"); 2 }
}

pub fn frame(kind: u8, bytes: &[u8]) -> Result<Vec<u8>, &'static str> {
    let length = bytes.len().checked_add(1).ok_or("frame overflow")?;
    if length > FRAME_LIMIT { return Err("frame exceeds bound"); }
    let mut result = Vec::with_capacity(length + 4);
    result.extend_from_slice(&(length as u32).to_be_bytes());
    result.push(kind); result.extend_from_slice(bytes); Ok(result)
}

mod client;
pub use client::{CommandSpec, OwnedRun, OwnerRegistry, RegistryState, Retirement, RunState};

mod launch;
pub use launch::{OwnedRedirect, PRIVATE_LAUNCH_COMMAND, private_launch};

#[cfg(test)]
mod launch_policy_tests {
    use super::*;
    #[test]
    fn early_input_close_requires_explicit_exact_private_flag() {
        let args = ["--deadline-ms", "1000", "--capture-bytes", "1024", "--input-bytes", "99", "--", "program"];
        let strict = Launch::parse(args.map(OsString::from).to_vec()).unwrap();
        assert!(!strict.allow_early_input_close);
        let mut shell = args.map(OsString::from).to_vec();
        shell.insert(6, "--allow-early-input-close".into());
        assert!(Launch::parse(shell.clone()).unwrap().allow_early_input_close);
        shell.insert(6, "--allow-early-input-close".into());
        assert!(Launch::parse(shell).is_err());
        let mut unknown = args.map(OsString::from).to_vec(); unknown.insert(6, "--unknown-policy".into());
        assert!(Launch::parse(unknown).is_err());
    }
}
