//! FreeBSD private-role backend. All signals use kernel reaper ownership.
//! ABI: FreeBSD releng/15.1 procctl.h at e38a7085f3a8ecd317c947591fe42b5ca9ab317b.
//! libc 0.2.186 exposes procctl but not the two padded argument records.
use super::*;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::process::CommandExt;
use std::process::{ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

// Narrow original ABI declarations matching the full public header, including
// its reserved words. No kernel implementation is copied or vendored.
#[repr(C)]
struct ReaperStatus {
    flags: libc::c_uint, children: libc::c_uint, descendants: libc::c_uint,
    reaper: libc::pid_t, pid: libc::pid_t, reserved: [libc::c_uint; 15],
}
#[repr(C)]
struct ReaperKill {
    signal: libc::c_int, flags: libc::c_uint, subtree: libc::pid_t,
    killed: libc::c_uint, failed_pid: libc::pid_t, reserved: [libc::c_uint; 15],
}
const _: () = {
    assert!(std::mem::size_of::<ReaperStatus>() == 80);
    assert!(std::mem::align_of::<ReaperStatus>() == 4);
    assert!(std::mem::size_of::<ReaperKill>() == 80);
    assert!(std::mem::align_of::<ReaperKill>() == 4);
};
const OWNED: libc::c_uint = 1;
static INTERRUPTED: AtomicBool = AtomicBool::new(false);
extern "C" fn interrupt(_: libc::c_int) { INTERRUPTED.store(true, Ordering::Relaxed); }

fn proc_call<T>(command: libc::c_int, value: &mut T) -> io::Result<()> {
    let result = unsafe { libc::procctl(libc::P_PID, 0, command, (value as *mut T).cast()) };
    if result == 0 { Ok(()) } else { Err(io::Error::last_os_error()) }
}
fn acquire() -> io::Result<()> {
    let result = unsafe { libc::procctl(libc::P_PID, 0, libc::PROC_REAP_ACQUIRE, std::ptr::null_mut()) };
    if result == 0 { Ok(()) } else { Err(io::Error::last_os_error()) }
}
fn status() -> io::Result<ReaperStatus> {
    let mut value: ReaperStatus = unsafe { std::mem::zeroed() };
    proc_call(libc::PROC_REAP_STATUS, &mut value)?;
    if value.flags & OWNED == 0 || value.reaper != unsafe { libc::getpid() } {
        return Err(io::Error::other("self reaper authority unavailable"));
    }
    Ok(value)
}
fn signal_domain(signal: libc::c_int) -> io::Result<()> {
    let mut value: ReaperKill = unsafe { std::mem::zeroed() };
    value.signal = signal;
    // flags=0 means entire owned domain, including nested reapers. The
    // failed PID is diagnostic only; never reuse it as signal authority.
    proc_call(libc::PROC_REAP_KILL, &mut value)?;
    if value.failed_pid != -1 {
        return Err(io::Error::other("partial domain signal failure"));
    }
    Ok(())
}
fn configure(fd: RawFd) -> io::Result<()> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error());
    }
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
fn private_pipe(fd: RawFd) -> io::Result<()> {
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut stat) } != 0 { return Err(io::Error::last_os_error()); }
    if stat.st_mode & libc::S_IFMT != libc::S_IFIFO {
        return Err(io::Error::other("private control must use pipes"));
    }
    configure(fd)
}

struct Owner {
    control_in: File, control_out: File, launch: Launch,
    input_frames: Vec<u8>, output_frames: VecDeque<Vec<u8>>, output_offset: usize, queued: usize,
    controller_alive: bool, ready: bool, admitted: bool, started: bool, retiring: bool,
    direct_pid: Option<libc::pid_t>, direct_status: Option<i32>, exit_queued: bool, wait_unknown: bool,
    stdin: Option<ChildStdin>, stdout: Option<ChildStdout>, stderr: Option<ChildStderr>,
    input: VecDeque<u8>, credit: usize, input_total: u64, input_eof: bool, input_peer_closed: bool,
    stdout_total: u64, stderr_total: u64,
    reason: u8, acquired_at: Instant, started_at: Option<Instant>, retirement_at: Option<Instant>,
    last_signal: Option<Instant>, last_unknown: Option<Instant>, terminal_queued: bool,
}
impl Owner {
    fn fail(&mut self, reason: u8) {
        if self.reason == 0 { self.reason = reason; }
        self.retire();
    }
    fn retire(&mut self) {
        self.retiring = true;
        self.retirement_at.get_or_insert_with(Instant::now);
        self.stdin.take(); self.input.clear();
        // Preserve only previously granted credit after an ordinary shell
        // consumer exits. It bounds in-flight frames; no more credit is issued.
        if !(self.launch.allow_early_input_close && self.reason == 0 && self.direct_status.is_some()) { self.credit = 0; }
    }
    fn disconnected(&mut self) {
        self.controller_alive = false; self.output_frames.clear(); self.queued = 0;
        self.output_offset = 0; self.fail(1);
    }
    fn enqueue(&mut self, kind: u8, bytes: &[u8]) {
        if !self.controller_alive || self.terminal_queued { return; }
        let Ok(packet) = frame(kind, bytes) else { self.fail(6); return; };
        if self.queued.saturating_add(packet.len()) > QUEUE_LIMIT {
            self.fail(4); return;
        }
        self.queued += packet.len(); self.output_frames.push_back(packet);
    }
    fn unknown(&mut self, message: &str) {
        if message.len() <= 4096 && self.queued.saturating_add(message.len() + 5) <= QUEUE_LIMIT - 64
            && self.last_unknown.is_none_or(|at| at.elapsed() >= Duration::from_secs(1)) {
            self.last_unknown = Some(Instant::now()); self.enqueue(UNKNOWN, message.as_bytes());
        }
    }
    fn flush_control(&mut self) {
        // Work budget keeps payload and cancellation observations fair.
        for _ in 0..16 {
            let Some(front) = self.output_frames.front() else { return; };
            match self.control_out.write(&front[self.output_offset..]) {
                Ok(0) => { self.disconnected(); return; }
                Ok(n) => {
                    self.output_offset += n; self.queued -= n;
                    if self.output_offset == front.len() { self.output_frames.pop_front(); self.output_offset = 0; }
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => { self.disconnected(); return; }
            }
        }
    }
    fn command(&mut self, kind: u8, bytes: &[u8]) {
        if kind == CANCEL && bytes.is_empty() { self.fail(1); return; }
        if self.retiring && self.launch.allow_early_input_close && self.reason == 0 && self.direct_status.is_some() {
            match kind {
                INPUT => self.accept_input(bytes, true),
                INPUT_EOF if bytes.is_empty() && !self.input_eof => self.input_eof = true,
                _ => self.fail(6),
            }
            return;
        }
        if self.retiring {
            // Full delivery can race the final EOF after a natural exit.
            if kind == INPUT_EOF && bytes.is_empty() && self.started && !self.input_eof
                && self.direct_status.is_some() && self.input_total == self.launch.limits.input_bytes {
                self.input_eof = true; return;
            }
            self.fail(6); return;
        }
        match kind {
            START if bytes.is_empty() && self.ready && !self.admitted => self.spawn(),
            INPUT => self.accept_input(bytes, self.input_peer_closed),
            INPUT_EOF if bytes.is_empty() && self.started && !self.input_eof => self.input_eof = true,
            _ => self.fail(6),
        }
    }
    fn accept_input(&mut self, bytes: &[u8], discard: bool) {
        if !self.started || self.input_eof || bytes.is_empty() || bytes.len() > CHUNK || bytes.len() > self.credit {
            self.fail(6); return;
        }
        let Some(total) = self.input_total.checked_add(bytes.len() as u64) else { self.fail(6); return; };
        if total > self.launch.limits.input_bytes { self.fail(6); return; }
        self.input_total = total; self.credit -= bytes.len();
        if !discard { self.input.extend(bytes); }
    }
    fn receive_control(&mut self) {
        if !self.controller_alive || self.terminal_queued { return; }
        for _ in 0..16 {
            while self.input_frames.len() >= 4 {
                let len = u32::from_be_bytes(self.input_frames[..4].try_into().unwrap()) as usize;
                if !(1..=FRAME_LIMIT).contains(&len) { self.input_frames.clear(); self.fail(6); return; }
                if self.input_frames.len() < len + 4 { break; }
                let packet: Vec<u8> = self.input_frames.drain(..len + 4).collect();
                self.command(packet[4], &packet[5..]);
            }
            let room = (FRAME_LIMIT + 4).saturating_sub(self.input_frames.len());
            if room == 0 { self.fail(6); return; }
            let mut buffer = [0u8; CHUNK];
            let limit = room.min(buffer.len());
            match self.control_in.read(&mut buffer[..limit]) {
                Ok(0) => { if !self.input_frames.is_empty() { self.fail(6); } self.disconnected(); return; }
                Ok(n) => self.input_frames.extend_from_slice(&buffer[..n]),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => { self.fail(4); self.disconnected(); return; }
            }
        }
    }
    fn spawn(&mut self) {
        // Close admission before the first potentially successful spawn.
        self.admitted = true;
        let mut command = Command::new(&self.launch.program);
        command.args(&self.launch.arguments).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        unsafe { command.pre_exec(|| {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = libc::SIG_DFL;
            if libc::sigemptyset(&mut action.sa_mask) != 0
                || libc::sigaction(libc::SIGPIPE, &action, std::ptr::null_mut()) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }); }
        let result = command.spawn();
        let mut child = match result { Ok(c) => c, Err(_) => { self.fail(5); return; } };
        self.direct_pid = Some(child.id() as libc::pid_t);
        self.stdin = child.stdin.take(); self.stdout = child.stdout.take(); self.stderr = child.stderr.take();
        // Child::drop does not reap. This dedicated process exclusively owns
        // all waitpid observations, including adoption; never mix try_wait.
        drop(child);
        self.started = true; self.started_at = Some(Instant::now());
        self.enqueue(STARTED, &[]);
        let setup = (|| -> io::Result<()> {
            configure(self.stdin.as_ref().ok_or_else(|| io::Error::other("missing input"))?.as_raw_fd())?;
            configure(self.stdout.as_ref().ok_or_else(|| io::Error::other("missing output"))?.as_raw_fd())?;
            configure(self.stderr.as_ref().ok_or_else(|| io::Error::other("missing error"))?.as_raw_fd())?;
            Ok(())
        })();
        if setup.is_err() {
            // Never enter the polling loop with an endpoint whose nonblocking
            // setup failed. Domain cleanup still owns the spawned child.
            self.stdin.take(); self.stdout.take(); self.stderr.take();
            self.fail(4); return;
        }
        if self.launch.limits.input_bytes == 0 {
            self.input_eof = true; self.stdin.take();
        } else {
            self.credit = WINDOW; self.enqueue(CREDIT, &(WINDOW as u32).to_be_bytes());
        }
    }
    fn write_input(&mut self) {
        if self.retiring || self.input_peer_closed { return; }
        for _ in 0..4 {
            if self.input.is_empty() {
                if self.input_eof || self.input_total == self.launch.limits.input_bytes {
                    self.input_eof = true; self.stdin.take();
                }
                return;
            }
            let Some(writer) = self.stdin.as_mut() else { self.fail(4); return; };
            let (front, _) = self.input.as_slices();
            let amount = front.len().min(CHUNK);
            match writer.write(&front[..amount]) {
                Ok(0) => { self.fail(4); return; }
                Ok(n) => {
                    self.input.drain(..n); self.credit += n;
                    self.enqueue(CREDIT, &(n as u32).to_be_bytes());
                    if self.input.is_empty() && self.input_total == self.launch.limits.input_bytes {
                        self.input_eof = true; self.stdin.take(); return;
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) if e.kind() == io::ErrorKind::BrokenPipe && self.launch.allow_early_input_close => {
                    // A prefix reader may close stdin and continue doing work.
                    // Discard undeliverable input, never infer exit or retire.
                    self.input_peer_closed = true; self.stdin.take(); self.input.clear(); return;
                }
                Err(_) => { self.fail(4); return; }
            }
        }
    }
    fn read_output(&mut self, error: bool) {
        for _ in 0..4 {
            // Backpressure payload data, reserving space for outcome/credit
            // frames. Cancellation and deadlines continue while pipes fill.
            // Failed operations discard output but still drain native EOF.
            if self.reason == 0 && self.queued > QUEUE_LIMIT - CHUNK - 5 - 4096 { return; }
            let mut buffer = [0u8; CHUNK];
            let result = if error { self.stderr.as_mut().map(|s| s.read(&mut buffer)) }
                else { self.stdout.as_mut().map(|s| s.read(&mut buffer)) };
            match result {
                None => return,
                Some(Ok(0)) => { if error { self.stderr.take(); } else { self.stdout.take(); } return; }
                Some(Ok(n)) => {
                    let old = if error { self.stderr_total } else { self.stdout_total };
                    let total = old.checked_add(n as u64);
                    if error { self.stderr_total = total.unwrap_or(u64::MAX); }
                    else { self.stdout_total = total.unwrap_or(u64::MAX); }
                    if total.is_none_or(|value| value > self.launch.limits.capture_bytes) {
                        self.fail(3); continue;
                    }
                    if self.reason == 0 { self.enqueue(if error { STDERR } else { STDOUT }, &buffer[..n]); }
                }
                Some(Err(e)) if e.kind() == io::ErrorKind::WouldBlock => return,
                Some(Err(e)) if e.kind() == io::ErrorKind::Interrupted => continue,
                Some(Err(_)) => {
                    if error { self.stderr.take(); } else { self.stdout.take(); }
                    self.fail(4); return;
                }
            }
        }
    }
    fn reap(&mut self) {
        // Bounded work; waitpid(-1) only targets children of this private
        // process. It is an observation, never numeric signal authority.
        for _ in 0..128 {
            let mut raw = 0;
            let pid = unsafe { libc::waitpid(-1, &mut raw, libc::WNOHANG) };
            if pid > 0 {
                if Some(pid) == self.direct_pid && self.direct_status.is_none() {
                    self.direct_status = Some(raw);
                    // An exit cannot certify delivery of queued input.
                    if !self.launch.allow_early_input_close && !input_delivered(self.input.len(), self.input_eof, self.input_total, self.launch.limits.input_bytes) { self.fail(4); }
                    self.retire();
                }
                continue;
            }
            if pid == 0 { return; }
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::EINTR) { continue; }
            if error.raw_os_error() == Some(libc::ECHILD) {
                if self.direct_pid.is_some() && self.direct_status.is_none() {
                    self.wait_unknown = true; self.fail(7); self.unknown("direct child outcome unavailable");
                }
                return;
            }
            self.wait_unknown = true; self.fail(7); self.unknown("native wait observation failed"); return;
        }
    }
    fn queue_payload_exit(&mut self) {
        if !self.exit_queued && self.queued <= QUEUE_LIMIT - 9 {
            if let Some(raw) = self.direct_status {
                // The observation survives queue saturation. Never drop the
                // matching event and then emit an inconsistent Retired frame.
                self.enqueue(PAYLOAD_EXIT, &(raw as u32).to_be_bytes());
                self.exit_queued = true;
            }
        }
    }
    fn cleanup(&mut self) -> bool {
        if !self.retiring { return false; }
        let observed = match status() {
            Ok(value) => value,
            Err(_) => { self.fail(7); self.unknown("native domain status unavailable"); return false; }
        };
        if observed.children != 0 || observed.descendants != 0 {
            let signal = if self.retirement_at.is_some_and(|at| at.elapsed() >= Duration::from_secs(5)) {
                libc::SIGKILL
            } else { libc::SIGTERM };
            if self.last_signal.is_none_or(|at| at.elapsed() >= Duration::from_millis(250)) {
                self.last_signal = Some(Instant::now());
                if signal_domain(signal).is_err() { self.fail(7); self.unknown("native domain signal incomplete"); }
            }
            return false;
        }
        if self.wait_unknown || (self.direct_pid.is_some() && self.direct_status.is_none()) { return false; }
        // Domain empty is necessary but distinct from I/O closure. Read and
        // drain remaining bytes/EOF in later ticks; no false close-only proof.
        self.stdin.take();
        self.stdout.is_none() && self.stderr.is_none()
    }
    fn tick(&mut self) -> Option<i32> {
        if self.terminal_queued {
            self.flush_control();
            return self.output_frames.is_empty().then_some(if self.controller_alive { 0 } else { 2 });
        }
        if INTERRUPTED.load(Ordering::Relaxed) { self.fail(1); }
        self.receive_control();
        if !self.admitted && self.acquired_at.elapsed() >= Duration::from_secs(3) { self.fail(2); }
        if self.started_at.is_some_and(|at| at.elapsed() >= self.launch.limits.deadline) { self.fail(2); }
        self.write_input(); self.read_output(false); self.read_output(true); self.reap();
        self.queue_payload_exit();
        if self.cleanup() && !self.terminal_queued {
            if !self.controller_alive { return Some(2); }
            // Reserve terminal frame only after previously queued data drains.
            if self.output_frames.is_empty() && (self.direct_status.is_none() || self.exit_queued) {
                let mut data = vec![self.reason, u8::from(self.direct_status.is_some())];
                data.extend_from_slice(&(self.direct_status.unwrap_or(0) as u32).to_be_bytes());
                self.enqueue(RETIRED, &data); self.terminal_queued = true;
            }
        }
        self.flush_control();
        if self.terminal_queued && self.output_frames.is_empty() {
            return Some(if self.controller_alive { 0 } else { 2 });
        }
        None
    }
}

pub(super) fn run(args: Vec<OsString>) -> i32 {
    // No potentially blocking diagnostic write on inherited stderr here.
    let launch = match Launch::parse(args) { Ok(v) => v, Err(_) => return 2 };
    if private_pipe(0).is_err() || private_pipe(1).is_err() { return 2; }
    // The default hook can block on inherited stderr before catch_unwind.
    // This private process reports failure through its bounded control path.
    std::panic::set_hook(Box::new(|_| {}));
    unsafe {
        // A selected parent may ignore SIGCHLD or set SA_NOCLDWAIT. This
        // fresh owner must retain real child outcomes before admitting work.
        let mut child_status_action: libc::sigaction = std::mem::zeroed();
        child_status_action.sa_sigaction = libc::SIG_DFL;
        if libc::sigemptyset(&mut child_status_action.sa_mask) != 0
            || libc::sigaction(libc::SIGCHLD, &child_status_action, std::ptr::null_mut()) != 0 { return 2; }
        if libc::signal(libc::SIGPIPE, libc::SIG_IGN) == libc::SIG_ERR
            || libc::signal(libc::SIGINT, interrupt as *const () as libc::sighandler_t) == libc::SIG_ERR
            || libc::signal(libc::SIGTERM, interrupt as *const () as libc::sighandler_t) == libc::SIG_ERR {
            return 2;
        }
    }
    let input = unsafe { File::from_raw_fd(0) };
    let mut output = unsafe { File::from_raw_fd(1) };
    if acquire().is_err() {
        // Best effort only; no child/domain was acquired or admitted.
        if let Ok(message) = frame(REJECT, b"reaper acquisition failed") { let _ = output.write(&message); }
        return 2;
    }
    // All ownership is held outside catch_unwind. A panic after spawn cannot
    // silently transfer native children by unwinding the private role.
    let mut owner = Owner {
        control_in: input, control_out: output, launch,
        input_frames: Vec::new(), output_frames: VecDeque::new(), output_offset: 0, queued: 0,
        controller_alive: true, ready: false, admitted: false, started: false, retiring: false,
        direct_pid: None, direct_status: None, exit_queued: false, wait_unknown: false,
        stdin: None, stdout: None, stderr: None, input: VecDeque::new(), credit: 0,
        input_total: 0, input_eof: false, input_peer_closed: false, stdout_total: 0, stderr_total: 0,
        reason: 0, acquired_at: Instant::now(), started_at: None, retirement_at: None,
        last_signal: None, last_unknown: None, terminal_queued: false,
    };
    match status() {
        Ok(s) if s.children == 0 && s.descendants == 0 => {
            owner.ready = true; owner.enqueue(READY, b"MOTHOWN1");
        }
        _ => { owner.fail(7); owner.unknown("initial owned empty domain not confirmed"); }
    }
    loop {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| owner.tick())) {
            Ok(Some(code)) => return code,
            Ok(None) => {},
            Err(_) => { owner.fail(7); owner.unknown("owner loop panic; retirement retained"); }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

// Delivery means successful writes to payload stdin, never a claim that the
// application consumed those bytes. EOF remains required for short streams.
fn input_delivered(pending: usize, eof: bool, received: u64, declared: u64) -> bool {
    pending == 0 && (eof || received == declared)
}

#[cfg(test)]
mod input_completion_tests {
    use super::input_delivered;
    #[test]
    fn fast_exit_requires_every_declared_byte_written_or_explicit_short_eof() {
        assert!(input_delivered(0, false, 0, 0));
        assert!(input_delivered(0, false, 32_768, 32_768));
        assert!(!input_delivered(1, false, 32_768, 32_768));
        assert!(!input_delivered(0, false, 32_767, 32_768));
        assert!(!input_delivered(0, false, 100, u64::MAX));
        assert!(input_delivered(0, true, 100, u64::MAX));
        assert!(!input_delivered(1, true, 100, u64::MAX));
    }
}
