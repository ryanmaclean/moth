//! Caller-driven retained clients. No background scheduler or detached threads.
use super::*;
use std::collections::{HashMap, VecDeque};
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct CommandSpec {
    pub program: OsString,
    pub args: Vec<OsString>,
    pub cwd: Option<PathBuf>,
    pub env: Vec<(OsString, OsString)>,
    pub limits: Limits,
}
#[derive(Clone, Debug)]
pub struct Retirement {
    pub reason: u8,
    pub payload_exit_code: Option<i32>,
    pub payload_signal: Option<i32>,
}
impl Retirement {
    pub fn success(&self) -> bool {
        self.reason == 0 && self.payload_exit_code == Some(0) && self.payload_signal.is_none()
    }
}
#[derive(Clone, Debug)]
pub enum RunState {
    Starting,
    Running,
    Retiring,
    Unknown(String),
    Retired(Retirement),
}
#[derive(Clone, Copy, Debug)]
pub struct RegistryState {
    pub active: usize,
    pub unconfirmed: usize,
}
struct Entries {
    closed: bool,
    next: u64,
    pending: usize,
    clients: HashMap<u64, Client>,
}
/// Keep the final Arc in the existing CLI/lifetime owner until the explicit
/// retirement barrier. Drop cannot certify cleanup. An unexpected final Drop
/// closes control so the private native owner requests cleanup; it leaves an
/// unconfirmed failure for the outer owner, never a successful receipt.
#[must_use]
pub struct OwnerRegistry {
    agent: PathBuf,
    entries: Mutex<Entries>,
}
#[must_use]
pub struct OwnedRun {
    registry: Arc<OwnerRegistry>,
    id: u64,
}
impl OwnerRegistry {
    pub fn new(agent: PathBuf) -> io::Result<Arc<Self>> {
        if !cfg!(target_os = "freebsd") {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "native owner requires FreeBSD",
            ));
        }
        if !agent.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "owner executable must be absolute",
            ));
        }
        Ok(Arc::new(Self {
            agent,
            entries: Mutex::new(Entries {
                closed: false,
                next: 0,
                pending: 0,
                clients: HashMap::new(),
            }),
        }))
    }
    fn lock(&self) -> MutexGuard<'_, Entries> {
        match self.entries.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                guard.closed = true;
                for client in guard.clients.values_mut() {
                    client.cancel();
                }
                guard
            }
        }
    }
    pub fn spawn(self: &Arc<Self>, spec: CommandSpec) -> io::Result<OwnedRun> {
        self.spawn_with_env(spec, false, false)
    }
    /// Preserve VShell's explicit exported environment without inheriting the CLI environment.
    pub fn spawn_cleared(self: &Arc<Self>, spec: CommandSpec) -> io::Result<OwnedRun> {
        self.spawn_with_env(spec, true, false)
    }
    pub fn spawn_redirected(
        self: &Arc<Self>,
        mut spec: CommandSpec,
        redirects: &[OwnedRedirect],
    ) -> io::Result<OwnedRun> {
        if redirects.iter().any(|r| matches!(r, OwnedRedirect::Input(_))) {
            spec.limits.input_bytes = 0;
        }
        let mut args = vec![OsString::from(PRIVATE_LAUNCH_COMMAND)];
        args.extend(super::launch::arguments(redirects, spec.program, spec.args)?);
        spec.program = self.agent.as_os_str().to_owned();
        spec.args = args;
        self.spawn_with_env(spec, true, true)
    }
    fn spawn_with_env(
        self: &Arc<Self>,
        spec: CommandSpec,
        clear_env: bool,
        allow_early_input_close: bool,
    ) -> io::Result<OwnedRun> {
        let millis = spec.limits.deadline.as_millis();
        if spec.program.is_empty()
            || !(1..=86_400_000).contains(&millis)
            || !(1..=67_108_864).contains(&spec.limits.capture_bytes)
        {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid native owner launch"));
        }
        let id = {
            let mut entries = self.lock();
            if entries.closed {
                return Err(io::Error::new(io::ErrorKind::BrokenPipe, "owner admission closed"));
            }
            let id = entries.next;
            entries.next =
                id.checked_add(1).ok_or_else(|| io::Error::other("owner ID exhausted"))?;
            let pending = entries
                .pending
                .checked_add(1)
                .ok_or_else(|| io::Error::other("owner admission count exhausted"))?;
            entries
                .clients
                .try_reserve(pending)
                .map_err(|_| io::Error::other("owner registry capacity unavailable"))?;
            entries.pending = pending;
            id
        };
        // No registry mutex across potentially blocking process creation.
        let mut command = Command::new(&self.agent);
        if clear_env {
            command.env_clear();
        }
        command
            .arg(PRIVATE_COMMAND)
            .arg("--deadline-ms")
            .arg(millis.to_string())
            .arg("--capture-bytes")
            .arg(spec.limits.capture_bytes.to_string())
            .arg("--input-bytes")
            .arg(spec.limits.input_bytes.to_string());
        if allow_early_input_close {
            command.arg("--allow-early-input-close");
        }
        command
            .arg("--")
            .arg(&spec.program)
            .args(&spec.args)
            .envs(spec.env.iter().cloned())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if let Some(cwd) = &spec.cwd {
            command.current_dir(cwd);
        }
        let prepared = command.spawn().map(|child| Client::new(child, spec.limits));
        let mut entries = self.lock();
        entries.pending -= 1;
        let mut client = prepared?;
        if entries.closed {
            client.cancel();
        }
        // Insert even after pipe/configuration failure: the registry owns the
        // actual child until confirmed retirement, not the transient caller.
        entries.clients.insert(id, client);
        Ok(OwnedRun { registry: self.clone(), id })
    }
    pub fn admission_closed(&self) -> bool {
        self.lock().closed
    }
    pub fn close_admission(&self) {
        self.lock().closed = true;
    }
    pub fn cancel_all(&self) {
        let mut entries = self.lock();
        entries.closed = true;
        for client in entries.clients.values_mut() {
            client.cancel();
        }
    }
    pub fn poll(&self) -> RegistryState {
        let mut entries = self.lock();
        let mut result = RegistryState { active: entries.pending, unconfirmed: entries.pending };
        for client in entries.clients.values_mut() {
            client.tick();
            match client.state() {
                RunState::Retired(_) => {}
                RunState::Unknown(_) => {
                    result.active += 1;
                    result.unconfirmed += 1;
                }
                _ => result.active += 1,
            }
        }
        entries.clients.retain(|_, client| {
            client.lease_live || !matches!(client.state(), RunState::Retired(_))
        });
        result
    }
    pub fn retire_until(&self, deadline: Instant) -> bool {
        self.cancel_all();
        loop {
            if self.poll().active == 0 {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
impl Drop for OwnerRegistry {
    fn drop(&mut self) {
        // No blocking Drop, claimed join or leaked global cleanup registry.
        // The controlled CLI path must already have observed the barrier.
        let entries = match self.entries.get_mut() {
            Ok(e) => e,
            Err(e) => e.into_inner(),
        };
        for client in entries.clients.values_mut() {
            client.cancel();
        }
    }
}
impl OwnedRun {
    fn with<T>(&self, f: impl FnOnce(&mut Client) -> T) -> T {
        self.registry.poll();
        let mut entries = self.registry.lock();
        f(entries.clients.get_mut(&self.id).expect("owned run retained by registry"))
    }
    pub fn poll(&self) -> RunState {
        self.with(|c| c.state())
    }
    pub fn cancel(&self) {
        self.with(|c| c.cancel());
    }
    pub fn close_input(&self) -> io::Result<()> {
        self.with(|c| {
            if c.input_closed {
                return Ok(());
            }
            c.input_closed = true;
            if c.input_total == c.limits.input_bytes {
                return Ok(());
            }
            if c.cancelled || c.terminal.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "run no longer accepts input",
                ));
            }
            c.want_eof = true;
            Ok(())
        })
    }
    pub fn read_stderr(&self, bytes: &mut [u8]) -> io::Result<usize> {
        self.with(|c| c.take_output(true, bytes))
    }
}
impl Drop for OwnedRun {
    fn drop(&mut self) {
        let mut entries = self.registry.lock();
        if let Some(client) = entries.clients.get_mut(&self.id) {
            client.lease_live = false;
            client.stdout.clear();
            client.stderr.clear();
            client.cancel();
        }
    }
}
impl Read for OwnedRun {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.with(|c| c.take_output(false, bytes))
    }
}
impl Write for OwnedRun {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        self.with(|c| {
            if c.input_closed || c.cancelled || c.terminal.is_some() {
                return Err(io::Error::new(io::ErrorKind::BrokenPipe, "owner input closed"));
            }
            if !c.started || c.credit == 0 || c.queued >= QUEUE_LIMIT - CHUNK - 5 {
                return Err(io::ErrorKind::WouldBlock.into());
            }
            let amount = bytes.len().min(CHUNK).min(c.credit);
            let total = c
                .input_total
                .checked_add(amount as u64)
                .ok_or_else(|| io::Error::other("input count overflow"))?;
            if total > c.limits.input_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "input exceeds declared bound",
                ));
            }
            c.queue(INPUT, &bytes[..amount])?;
            c.credit -= amount;
            c.input_total = total;
            if total == c.limits.input_bytes {
                c.input_closed = true;
            }
            Ok(amount)
        })
    }
    fn flush(&mut self) -> io::Result<()> {
        self.with(
            |c| if c.queue.is_empty() { Ok(()) } else { Err(io::ErrorKind::WouldBlock.into()) },
        )
    }
}

struct Client {
    child: Child,
    input: Option<ChildStdin>,
    output: Option<ChildStdout>,
    limits: Limits,
    created: Instant,
    queue: VecDeque<Vec<u8>>,
    offset: usize,
    queued: usize,
    incoming: Vec<u8>,
    ready: bool,
    start_sent: bool,
    started: bool,
    credit: usize,
    credit_total: u64,
    input_total: u64,
    input_closed: bool,
    want_eof: bool,
    eof_sent: bool,
    cancelled: bool,
    cancel_sent: bool,
    control_write_closed: bool,
    protocol_failure: bool,
    diagnostic: Option<String>,
    payload: Option<i32>,
    terminal: Option<Retirement>,
    owner_status: Option<ExitStatus>,
    eof: bool,
    stdout: VecDeque<u8>,
    stderr: VecDeque<u8>,
    stdout_total: u64,
    stderr_total: u64,
    lease_live: bool,
}
fn nonblocking<T>(pipe: &T) -> io::Result<()>
where
    T: PipeDescriptor,
{
    pipe.set_nonblocking()
}
trait PipeDescriptor {
    fn set_nonblocking(&self) -> io::Result<()>;
}
macro_rules! pipe_descriptor {
    ($ty:ty) => {
        impl PipeDescriptor for $ty {
            fn set_nonblocking(&self) -> io::Result<()> {
                #[cfg(target_os = "freebsd")]
                {
                    use std::os::fd::AsRawFd;
                    let fd = self.as_raw_fd();
                    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
                    if flags < 0
                        || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0
                    {
                        return Err(io::Error::last_os_error());
                    }
                    Ok(())
                }
                #[cfg(not(target_os = "freebsd"))]
                {
                    Err(io::Error::new(io::ErrorKind::Unsupported, "native owner requires FreeBSD"))
                }
            }
        }
    };
}
pipe_descriptor!(ChildStdin);
pipe_descriptor!(ChildStdout);
impl Client {
    fn new(mut child: Child, limits: Limits) -> Self {
        let input = child.stdin.take();
        let output = child.stdout.take();
        let setup = input.as_ref().is_some_and(|p| nonblocking(p).is_ok())
            && output.as_ref().is_some_and(|p| nonblocking(p).is_ok());
        let mut result = Self {
            child,
            input,
            output,
            limits,
            created: Instant::now(),
            queue: VecDeque::new(),
            offset: 0,
            queued: 0,
            incoming: Vec::new(),
            ready: false,
            start_sent: false,
            started: false,
            credit: 0,
            credit_total: 0,
            input_total: 0,
            input_closed: false,
            want_eof: false,
            eof_sent: false,
            cancelled: false,
            cancel_sent: false,
            control_write_closed: false,
            protocol_failure: false,
            diagnostic: None,
            payload: None,
            terminal: None,
            owner_status: None,
            eof: false,
            stdout: VecDeque::new(),
            stderr: VecDeque::new(),
            stdout_total: 0,
            stderr_total: 0,
            lease_live: true,
        };
        if !setup {
            // Never poll a pipe whose O_NONBLOCK setup was not confirmed.
            result.input.take();
            result.output.take();
            result.eof = true;
            result.broken("owner control setup unavailable");
        }
        result
    }
    fn queue(&mut self, kind: u8, bytes: &[u8]) -> io::Result<()> {
        let packet = frame(kind, bytes).map_err(io::Error::other)?;
        if self.queued + packet.len() > QUEUE_LIMIT {
            return Err(io::ErrorKind::WouldBlock.into());
        }
        self.queued += packet.len();
        self.queue.push_back(packet);
        Ok(())
    }
    fn cancel(&mut self) {
        if self.terminal.is_some() || self.cancelled {
            return;
        }
        self.cancelled = true;
        self.input_closed = true;
        // Finish at most the partial front frame. Discard unsent input and
        // place cancellation next; never interleave bytes or wait for credit.
        if self.offset > 0 {
            self.queue.truncate(1);
            self.queued = self.queue.front().map_or(0, |p| p.len() - self.offset);
        } else {
            self.queue.clear();
            self.queued = 0;
        }
        if !self.cancel_sent {
            let _ = self.queue(CANCEL, &[]);
            self.cancel_sent = true;
        }
    }
    fn broken(&mut self, message: &str) {
        self.protocol_failure = true;
        self.diagnostic = Some(message.into());
        self.cancel();
    }
    fn state(&self) -> RunState {
        if !self.protocol_failure
            && self.eof
            && self.incoming.is_empty()
            && self.owner_status.is_some_and(|s| s.success())
            && let Some(result) = &self.terminal
        {
            return RunState::Retired(result.clone());
        }
        if self.protocol_failure
            || self.owner_status.is_some_and(|s| !s.success())
            || (self.owner_status.is_some() && self.terminal.is_none())
        {
            return RunState::Unknown(
                self.diagnostic.clone().unwrap_or_else(|| "owner exited without retirement".into()),
            );
        }
        if let Some(message) = &self.diagnostic {
            return RunState::Unknown(message.clone());
        }
        if self.cancelled || self.control_write_closed || self.payload.is_some() {
            RunState::Retiring
        } else if self.started {
            RunState::Running
        } else {
            RunState::Starting
        }
    }
    fn take_output(&mut self, error: bool, bytes: &mut [u8]) -> io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let stream = if error { &mut self.stderr } else { &mut self.stdout };
        let count = stream.len().min(bytes.len());
        for byte in &mut bytes[..count] {
            *byte = stream.pop_front().unwrap();
        }
        if count != 0 {
            return Ok(count);
        }
        if matches!(self.state(), RunState::Retired(_)) {
            return Ok(0);
        }
        if self.protocol_failure {
            return Err(io::Error::other("unconfirmed owner output"));
        }
        Err(io::ErrorKind::WouldBlock.into())
    }
    fn event(&mut self, kind: u8, bytes: &[u8]) {
        if self.terminal.is_some() {
            self.broken("event after retirement");
            return;
        }
        match kind {
            READY if !self.ready && bytes == b"MOTHOWN1" => {
                self.ready = true;
                if !self.cancelled && self.queue(START, &[]).is_err() {
                    self.broken("start queue overflow");
                }
            }
            STARTED if self.ready && self.start_sent && !self.started && bytes.is_empty() => {
                self.started = true;
                if self.limits.input_bytes == 0 {
                    self.input_closed = true;
                    self.eof_sent = true;
                }
            }
            CREDIT if self.started && self.limits.input_bytes != 0 && bytes.len() == 4 => {
                let amount = u32::from_be_bytes(bytes.try_into().unwrap()) as usize;
                let total = self.credit_total.checked_add(amount as u64);
                if amount == 0
                    || amount > WINDOW
                    || self.credit.saturating_add(amount) > WINDOW
                    || total.is_none_or(|n| n > self.input_total.saturating_add(WINDOW as u64))
                {
                    self.broken("invalid input credit");
                    return;
                }
                self.credit_total = total.unwrap();
                self.credit += amount;
            }
            STDOUT | STDERR if self.started && bytes.len() <= CHUNK => {
                let error = kind == STDERR;
                let old = if error { self.stderr_total } else { self.stdout_total };
                let Some(total) = old.checked_add(bytes.len() as u64) else {
                    self.broken("capture count overflow");
                    return;
                };
                let stream = if error { &mut self.stderr } else { &mut self.stdout };
                if total > self.limits.capture_bytes {
                    self.broken("capture bound exceeded");
                    return;
                }
                if self.lease_live {
                    stream.extend(bytes);
                }
                if error {
                    self.stderr_total = total;
                } else {
                    self.stdout_total = total;
                }
            }
            PAYLOAD_EXIT if self.started && self.payload.is_none() && bytes.len() == 4 => {
                let raw = u32::from_be_bytes(bytes.try_into().unwrap()) as i32;
                let (code, signal) = decode_status(raw);
                if code.is_none() && signal.is_none() {
                    self.broken("nonterminal payload wait status");
                    return;
                }
                self.payload = Some(raw);
            }
            RETIRED if self.ready && bytes.len() == 6 && bytes[0] <= 7 && bytes[1] <= 1 => {
                let raw = u32::from_be_bytes(bytes[2..].try_into().unwrap()) as i32;
                if (bytes[1] == 1 && self.payload != Some(raw))
                    || (bytes[1] == 0 && (raw != 0 || self.payload.is_some()))
                    || (bytes[0] == 0 && (bytes[1] != 1 || !self.started))
                {
                    self.broken("inconsistent retirement outcome");
                    return;
                }
                let (code, signal) = if bytes[1] == 1 { decode_status(raw) } else { (None, None) };
                self.terminal = Some(Retirement {
                    reason: bytes[0],
                    payload_exit_code: code,
                    payload_signal: signal,
                });
                self.input.take();
                self.queue.clear();
                self.queued = 0;
                self.offset = 0;
            }
            UNKNOWN if bytes.len() <= 4096 && std::str::from_utf8(bytes).is_ok() => {
                // Diagnostics never supply authority and are not logged with
                // inherited credentials or payload bytes.
                self.diagnostic = Some("native retirement unresolved".into());
            }
            REJECT if bytes.len() <= 4096 => self.broken("owner rejected launch"),
            _ => self.broken("invalid owner event or sequence"),
        }
    }
    fn tick(&mut self) {
        if !self.ready && !self.protocol_failure && self.created.elapsed() >= Duration::from_secs(4)
        {
            self.broken("owner Ready deadline expired");
            // Closing control requests cleanup if acquisition later succeeds;
            // the registry still retains the actual child as Unknown.
            self.input.take();
            self.queue.clear();
            self.queued = 0;
            self.offset = 0;
        }
        if self.started
            && self.want_eof
            && !self.eof_sent
            && !self.cancelled
            && self.queue(INPUT_EOF, &[]).is_ok()
        {
            self.eof_sent = true;
        }
        for _ in 0..8 {
            let Some(front) = self.queue.front() else {
                break;
            };
            let Some(writer) = self.input.as_mut() else {
                break;
            };
            match writer.write(&front[self.offset..]) {
                Ok(0) => {
                    self.broken("owner input write zero");
                    self.input.take();
                    break;
                }
                Ok(n) => {
                    self.offset += n;
                    self.queued -= n;
                    if self.offset == front.len() {
                        if front[4] == START {
                            self.start_sent = true;
                        }
                        self.queue.pop_front();
                        self.offset = 0;
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) if e.kind() == io::ErrorKind::BrokenPipe => {
                    // The owner may have closed control after writing its final
                    // receipt while credited input was still in flight. EPIPE
                    // alone is neither failure proof nor retirement proof.
                    self.control_write_closed = true;
                    self.input_closed = true;
                    self.input.take();
                    self.queue.clear();
                    self.queued = 0;
                    self.offset = 0;
                    break;
                }
                Err(_) => {
                    self.broken("owner input failed");
                    self.input.take();
                    break;
                }
            }
        }
        for _ in 0..16 {
            while self.incoming.len() >= 4 {
                let length = u32::from_be_bytes(self.incoming[..4].try_into().unwrap()) as usize;
                if !(1..=FRAME_LIMIT).contains(&length) {
                    self.broken("owner frame size invalid");
                    self.incoming.clear();
                    break;
                }
                if self.incoming.len() < length + 4 {
                    break;
                }
                let packet: Vec<u8> = self.incoming.drain(..length + 4).collect();
                self.event(packet[4], &packet[5..]);
            }
            let Some(reader) = self.output.as_mut() else {
                break;
            };
            let mut buffer = [0u8; CHUNK];
            let room = (FRAME_LIMIT + 4).saturating_sub(self.incoming.len()).min(CHUNK);
            if room == 0 {
                self.broken("owner frame buffer exhausted");
                break;
            }
            match reader.read(&mut buffer[..room]) {
                Ok(0) => {
                    self.eof = true;
                    self.output.take();
                    if !self.incoming.is_empty() {
                        self.broken("truncated owner frame");
                    }
                    break;
                }
                Ok(n) => self.incoming.extend_from_slice(&buffer[..n]),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    self.broken("owner output read failure");
                    self.output.take();
                    break;
                }
            }
        }
        if self.owner_status.is_none() {
            match self.child.try_wait() {
                Ok(status) => self.owner_status = status,
                Err(_) => self.broken("owner wait unavailable"),
            }
        }
        if self.eof && self.terminal.is_none() {
            self.broken("owner EOF without retirement");
        }
        if self.owner_status.is_some_and(|s| !s.success()) {
            self.broken("owner process failed");
        }
    }
}
fn decode_status(raw: i32) -> (Option<i32>, Option<i32>) {
    #[cfg(target_os = "freebsd")]
    {
        use std::os::unix::process::ExitStatusExt;
        let status = ExitStatus::from_raw(raw);
        (status.code(), status.signal())
    }
    #[cfg(not(target_os = "freebsd"))]
    {
        let _ = raw;
        (None, None)
    }
}
