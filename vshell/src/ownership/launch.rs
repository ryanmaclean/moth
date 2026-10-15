//! Private redirection setup, executed only as a payload of the native owner.
//! Potentially blocking FIFO opens/exec occur here, never in the shared actor.
use super::*;
use std::io;
use std::path::PathBuf;

pub const PRIVATE_LAUNCH_COMMAND: &str = "__moth_payload_v1";
#[derive(Clone, Debug)]
pub enum OwnedRedirect {
    Input(PathBuf), Output { path: PathBuf, append: bool },
    Error { path: PathBuf, append: bool }, ErrorToOutput,
}
#[derive(Debug)]
struct Plan { redirects: Vec<OwnedRedirect>, program: OsString, args: Vec<OsString> }
impl Plan {
    fn parse(args: Vec<OsString>) -> io::Result<Self> {
        let invalid = || io::Error::new(io::ErrorKind::InvalidInput, "invalid private payload request");
        // Validate every byte-bearing argument before any open/truncate/dup.
        // OsString paths remain byte-exact; empty ordinary argv is valid.
        if args.iter().any(|arg| arg.as_encoded_bytes().contains(&0)) { return Err(invalid()); }
        let mut redirects = Vec::new(); let mut i = 0;
        while i < args.len() {
            let op = &args[i]; i += 1;
            if op == "--" {
                let program = args.get(i).filter(|p| !p.is_empty()).ok_or_else(invalid)?.clone();
                return Ok(Self { redirects, program, args: args[i + 1..].to_vec() });
            }
            if op == "--stderr-to-stdout" { redirects.push(OwnedRedirect::ErrorToOutput); continue; }
            let path = args.get(i).filter(|p| !p.is_empty()).ok_or_else(invalid)?;
            let operation = if op == "--stdin" { OwnedRedirect::Input(PathBuf::from(path)) }
                else if op == "--stdout" { OwnedRedirect::Output { path: PathBuf::from(path), append: false } }
                else if op == "--stdout-append" { OwnedRedirect::Output { path: PathBuf::from(path), append: true } }
                else if op == "--stderr" { OwnedRedirect::Error { path: PathBuf::from(path), append: false } }
                else if op == "--stderr-append" { OwnedRedirect::Error { path: PathBuf::from(path), append: true } }
                else { return Err(invalid()); };
            redirects.push(operation); i += 1;
        }
        Err(invalid())
    }
}
pub(super) fn arguments(redirects: &[OwnedRedirect], program: OsString, args: Vec<OsString>) -> io::Result<Vec<OsString>> {
    let mut encoded = Vec::new();
    for redirect in redirects {
        let (op, path) = match redirect {
            OwnedRedirect::Input(path) => ("--stdin", Some(path)),
            OwnedRedirect::Output { path, append } => (if *append { "--stdout-append" } else { "--stdout" }, Some(path)),
            OwnedRedirect::Error { path, append } => (if *append { "--stderr-append" } else { "--stderr" }, Some(path)),
            OwnedRedirect::ErrorToOutput => ("--stderr-to-stdout", None),
        };
        encoded.push(op.into()); if let Some(path) = path { encoded.push(path.as_os_str().to_owned()); }
    }
    encoded.push("--".into()); encoded.push(program); encoded.extend(args);
    Plan::parse(encoded.clone())?;
    Ok(encoded)
}
#[cfg(target_os = "freebsd")]
fn install(file: std::fs::File, target: libc::c_int) -> io::Result<()> {
    use std::os::fd::{AsRawFd, IntoRawFd};
    let fd = file.as_raw_fd();
    if fd == target {
        let fd = file.into_raw_fd();
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) } < 0 {
            return Err(io::Error::last_os_error());
        }
    } else if unsafe { libc::dup2(fd, target) } < 0 { return Err(io::Error::last_os_error()); }
    // Drop closes only a distinct temporary FD; target remains the standard
    // endpoint and dup2 cleared CLOEXEC. Same-FD ownership moved above.
    Ok(())
}
/// Fresh private child role. The independently acquired owner remains alive
/// and responsive while this process waits in an ordinary FIFO open or exec.
pub fn private_launch(args: Vec<OsString>) -> i32 {
    let plan = match Plan::parse(args) { Ok(plan) => plan, Err(_) => return 2 };
    #[cfg(not(target_os = "freebsd"))]
    { let _ = plan; return 2; }
    #[cfg(target_os = "freebsd")]
    {
        use std::fs::OpenOptions;
        use std::os::unix::fs::OpenOptionsExt;
        use std::os::unix::process::CommandExt;
        use std::process::Command;
        for redirect in plan.redirects {
            let result = match redirect {
                OwnedRedirect::Input(path) => OpenOptions::new().read(true).custom_flags(libc::O_CLOEXEC).open(path).and_then(|file| install(file, 0)),
                OwnedRedirect::Output { path, append } => OpenOptions::new().write(true).create(true).append(append).truncate(!append).custom_flags(libc::O_CLOEXEC).open(path).and_then(|file| install(file, 1)),
                OwnedRedirect::Error { path, append } => OpenOptions::new().write(true).create(true).append(append).truncate(!append).custom_flags(libc::O_CLOEXEC).open(path).and_then(|file| install(file, 2)),
                OwnedRedirect::ErrorToOutput => {
                    if unsafe { libc::dup2(1, 2) } < 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
                }
            };
            if let Err(error) = result { diagnostic("redirect", &error); return 2; }
        }
        let mut command = Command::new(&plan.program); command.args(&plan.args);
        unsafe { command.pre_exec(|| {
            let mut action: libc::sigaction = std::mem::zeroed(); action.sa_sigaction = libc::SIG_DFL;
            if libc::sigemptyset(&mut action.sa_mask) != 0
                || libc::sigaction(libc::SIGPIPE, &action, std::ptr::null_mut()) != 0 { return Err(io::Error::last_os_error()); }
            Ok(())
        }); }
        let error = command.exec();
        diagnostic("exec", &error);
        127
    }
}

#[cfg(target_os = "freebsd")]
fn diagnostic(stage: &str, error: &io::Error) {
    use std::io::Write;
    let message = format!("vshell: {stage}: {error}\n");
    let bytes = message.as_bytes();
    // A blocked diagnostic remains inside the native-owned child. Failed
    // diagnostics cannot replace the intended redirect/exec failure status.
    let _ = std::io::stderr().write_all(&bytes[..bytes.len().min(4096)]);
}

#[cfg(test)]
mod request_tests {
    use super::*;
    fn args(values: &[&str]) -> Vec<OsString> { values.iter().map(OsString::from).collect() }
    #[test]
    fn entire_request_is_validated_before_effects() {
        assert!(Plan::parse(args(&["--stdout", "file", "--bogus", "x", "--", "program"])).is_err());
        assert!(Plan::parse(args(&["--stdout", "file", "--"])).is_err());
        assert!(Plan::parse(args(&["--stdout", "file", "--", "program", "x\0y"])).is_err());
        let plan = Plan::parse(args(&["--stderr-to-stdout", "--stdout", "file", "--", "program", ""])).unwrap();
        assert!(matches!(plan.redirects[0], OwnedRedirect::ErrorToOutput));
        assert!(matches!(plan.redirects[1], OwnedRedirect::Output { append: false, .. }));
        assert_eq!(plan.args, vec![OsString::new()]);
    }
    #[cfg(unix)]
    #[test]
    fn paths_and_argv_remain_non_utf8_os_strings() {
        use std::os::unix::ffi::OsStringExt;
        let path = OsString::from_vec(vec![b'f', 0xff]); let value = OsString::from_vec(vec![b'a', 0xfe]);
        let plan = Plan::parse(vec!["--stdin".into(), path.clone(), "--".into(), "program".into(), value.clone()]).unwrap();
        assert!(matches!(&plan.redirects[0], OwnedRedirect::Input(p) if p.as_os_str() == path.as_os_str()));
        assert_eq!(plan.args, vec![value]);
    }
}
