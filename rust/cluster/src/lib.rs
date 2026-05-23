//! Multi-node `ActorRef`: TCP-backed remote message passing.
//!
//! A `Node` listens on a TCP socket and holds a registry of local actors by
//! string id. Peers hold `RemoteActorRef<M>` handles that dial the node and
//! ship a length-prefixed frame containing `actor_id\0encoded_message`.
//!
//! Wire format (big-endian u32 length includes the actor_id, the NUL, and
//! the encoded payload; cap at 4 MiB):
//!
//! ```text
//! [ 4 bytes: payload length ]
//! [ N bytes: actor_id\0encoded_message ]
//! ```
//!
//! Fire-and-forget: senders block on the write but never wait for an ack.
//! On write failure the cached TCP connection is dropped; the next `send`
//! re-dials. Receivers decode each frame via the registered closure and
//! forward the typed message to the local `ActorRef<M>`. Unknown actor ids
//! and codec errors log to stderr and skip the frame; they do not tear down
//! the connection.
//!
//! # Authentication
//!
//! `NodeConfig::auth_token` holds an optional 32-byte shared secret. When
//! set, every accepted connection must open with a fixed-length handshake
//! (`b"AUTH"` magic + 32-byte token) before any frame is read; mismatches
//! are silently dropped (we close without responding so a probing attacker
//! cannot distinguish "wrong token" from "wrong protocol"). The compare is
//! constant-time.
//!
//! **WARNING**: a `Node` without an `auth_token` accepts traffic from any
//! TCP peer that can reach the listener. Binding to anything other than
//! `127.0.0.1` (or another trusted loopback) without configuring
//! `auth_token` is a remote-code-execution-shaped foot-gun: an attacker
//! who reaches the socket can deliver arbitrary `Codec::decode` payloads
//! into any registered actor. Always pair public bind addresses with
//! `auth_token`.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::marker::PhantomData;
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use actor::ActorRef;

/// Maximum frame payload size. Anything larger is rejected on both sender
/// and receiver to bound memory and protect against runaway peers.
pub const MAX_PAYLOAD: usize = 4 * 1024 * 1024;

/// Handshake magic bytes. Sent before the 32-byte token by clients that
/// have a configured `auth_token`.
const AUTH_MAGIC: [u8; 4] = *b"AUTH";

/// Length of the auth token in bytes.
pub const AUTH_TOKEN_LEN: usize = 32;

/// Default per-connection idle timeout. Peers that send nothing within this
/// window have their reader dropped to prevent slowloris-style resource
/// exhaustion.
pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

pub trait Codec: Sized + Send + 'static {
    fn encode(&self) -> Vec<u8>;
    fn decode(bytes: &[u8]) -> Result<Self, CodecError>;
}

#[derive(Debug)]
pub struct CodecError(pub String);

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "codec error: {}", self.0)
    }
}

impl std::error::Error for CodecError {}

#[derive(Debug)]
pub enum ClusterError {
    Io(io::Error),
    Codec(CodecError),
    UnknownActor(String),
    NotConnected,
    /// Frame payload exceeded `MAX_PAYLOAD`.
    Oversize(usize),
    /// Peer rejected the handshake or closed the connection mid-handshake.
    AuthFailed,
}

impl std::fmt::Display for ClusterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClusterError::Io(e) => write!(f, "io: {e}"),
            ClusterError::Codec(e) => write!(f, "{e}"),
            ClusterError::UnknownActor(id) => write!(f, "unknown actor id: {id}"),
            ClusterError::NotConnected => write!(f, "not connected"),
            ClusterError::Oversize(n) => write!(f, "oversize payload: {n} bytes"),
            ClusterError::AuthFailed => write!(f, "auth handshake failed"),
        }
    }
}

impl std::error::Error for ClusterError {}

impl From<io::Error> for ClusterError {
    fn from(e: io::Error) -> Self {
        ClusterError::Io(e)
    }
}

impl From<CodecError> for ClusterError {
    fn from(e: CodecError) -> Self {
        ClusterError::Codec(e)
    }
}

/// Constant-time byte-slice equality. We avoid `==` on `[u8; N]` because
/// the standard comparator short-circuits on the first mismatching byte,
/// which leaks token bytes through a timing side channel.
#[inline]
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ---------------------------------------------------------------------------
// NodeConfig
// ---------------------------------------------------------------------------

/// Tunables for a `Node`.
#[derive(Clone)]
pub struct NodeConfig {
    /// Shared secret expected from inbound peers. `None` disables the
    /// handshake — fine for `127.0.0.1`-only listeners, dangerous on any
    /// reachable address (see module docs).
    pub auth_token: Option<[u8; AUTH_TOKEN_LEN]>,
    /// Per-connection idle timeout. A peer that sends nothing within this
    /// window has its reader thread closed; the listener stays up.
    pub idle_timeout: Duration,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            auth_token: None,
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
        }
    }
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

type Dispatcher = Box<dyn Fn(&[u8]) + Send + Sync>;
type Registry = Arc<Mutex<HashMap<String, Dispatcher>>>;

/// Tracks the listener thread and any per-connection reader threads so
/// `shutdown` can join them cleanly.
struct Inner {
    running: AtomicBool,
    addr: SocketAddr,
    /// Number of live reader threads. Each thread bumps on entry and
    /// decrements on exit, so `shutdown` can spin-wait for the count to
    /// drain instead of holding `JoinHandle`s that never get joined on the
    /// long-lived-node path.
    reader_count: AtomicUsize,
    auth_token: Option<[u8; AUTH_TOKEN_LEN]>,
    idle_timeout: Duration,
}

pub struct Node {
    inner: Arc<Inner>,
    registry: Registry,
    listener_thread: Option<JoinHandle<()>>,
}

impl Node {
    /// Start listening on `addr` with default config (no auth, 60s idle
    /// timeout). Pass `"127.0.0.1:0"` for an ephemeral port.
    pub fn start(addr: &str) -> Result<Self, ClusterError> {
        Self::start_with_config(addr, NodeConfig::default())
    }

    /// Start listening on `addr` with explicit config.
    pub fn start_with_config(addr: &str, config: NodeConfig) -> Result<Self, ClusterError> {
        let listener = TcpListener::bind(addr)?;
        let local_addr = listener.local_addr()?;
        let inner = Arc::new(Inner {
            running: AtomicBool::new(true),
            addr: local_addr,
            reader_count: AtomicUsize::new(0),
            auth_token: config.auth_token,
            idle_timeout: config.idle_timeout,
        });
        let registry: Registry = Arc::new(Mutex::new(HashMap::new()));

        let inner_l = Arc::clone(&inner);
        let registry_l = Arc::clone(&registry);
        let listener_thread = thread::spawn(move || {
            accept_loop(listener, inner_l, registry_l);
        });

        Ok(Node {
            inner,
            registry,
            listener_thread: Some(listener_thread),
        })
    }

    /// Register a local actor under `id`.
    pub fn register<M: Codec>(&self, id: &str, addr: ActorRef<M>) {
        let dispatcher: Dispatcher = Box::new(move |bytes: &[u8]| {
            match M::decode(bytes) {
                Ok(msg) => {
                    let _ = addr.send(msg);
                }
                Err(e) => {
                    eprintln!("cluster: decode error: {}", e.0);
                }
            }
        });
        self.registry
            .lock()
            .expect("registry poisoned")
            .insert(id.to_string(), dispatcher);
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.inner.addr
    }

    /// Number of currently live reader threads. Exposed for tests and ops
    /// dashboards — useful for confirming idle peers are reaped.
    pub fn reader_count(&self) -> usize {
        self.inner.reader_count.load(Ordering::SeqCst)
    }

    /// Stop the listener and wait for all reader threads to drain.
    pub fn shutdown(mut self) {
        self.inner.running.store(false, Ordering::SeqCst);
        // Wake the blocking accept by dialing ourselves once.
        let _ = TcpStream::connect_timeout(&self.inner.addr, Duration::from_millis(200));
        if let Some(h) = self.listener_thread.take() {
            let _ = h.join();
        }
        // Wait for reader threads to drain. They observe EOF/timeout and
        // exit on their own; cap the wait at one idle window plus a small
        // slack so a wedged reader doesn't block shutdown forever.
        let deadline = Instant::now()
            + self
                .inner
                .idle_timeout
                .saturating_add(Duration::from_secs(2));
        while self.inner.reader_count.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
    }
}

fn accept_loop(listener: TcpListener, inner: Arc<Inner>, registry: Registry) {
    for incoming in listener.incoming() {
        if !inner.running.load(Ordering::SeqCst) {
            if let Ok(s) = incoming {
                let _ = s.shutdown(Shutdown::Both);
            }
            break;
        }
        match incoming {
            Ok(stream) => {
                let registry = Arc::clone(&registry);
                let inner_r = Arc::clone(&inner);
                inner.reader_count.fetch_add(1, Ordering::SeqCst);
                let spawned = thread::Builder::new()
                    .name("cluster-reader".into())
                    .spawn(move || {
                        // Decrement on any exit path.
                        struct Guard<'a>(&'a AtomicUsize);
                        impl Drop for Guard<'_> {
                            fn drop(&mut self) {
                                self.0.fetch_sub(1, Ordering::SeqCst);
                            }
                        }
                        let _g = Guard(&inner_r.reader_count);

                        // Bound how long we'll wait for the handshake. Use
                        // the same per-connection idle timeout: a peer that
                        // dials and then sits silent for that long gets
                        // dropped here rather than parking a reader thread.
                        let _ = stream.set_read_timeout(Some(inner_r.idle_timeout));

                        if let Some(expected) = inner_r.auth_token.as_ref()
                            && !server_handshake(&stream, expected)
                        {
                            // Don't respond — silently close. We log so
                            // ops can see brute-force attempts.
                            eprintln!(
                                "cluster: rejecting connection from {:?} (bad auth)",
                                stream.peer_addr().ok()
                            );
                            let _ = stream.shutdown(Shutdown::Both);
                            return;
                        }

                        if let Err(e) = read_loop(stream, &registry, inner_r.idle_timeout)
                            && !is_expected_disconnect(&e)
                        {
                            eprintln!("cluster: reader error: {e}");
                        }
                    });
                if spawned.is_err() {
                    // Thread spawn failed — undo the count we eagerly added.
                    inner.reader_count.fetch_sub(1, Ordering::SeqCst);
                }
            }
            Err(e) => {
                if !inner.running.load(Ordering::SeqCst) {
                    break;
                }
                eprintln!("cluster: accept error: {e}");
            }
        }
    }
}

/// Read the fixed-length handshake from a freshly accepted connection.
/// Returns `true` iff the magic and token match. Reads exactly
/// `4 + AUTH_TOKEN_LEN` bytes (or fails) so a partial/malformed handshake
/// can't shift the framer.
fn server_handshake(mut stream: &TcpStream, expected: &[u8; AUTH_TOKEN_LEN]) -> bool {
    let mut buf = [0u8; 4 + AUTH_TOKEN_LEN];
    if stream.read_exact(&mut buf).is_err() {
        return false;
    }
    let magic_ok = ct_eq(&buf[..4], &AUTH_MAGIC);
    let token_ok = ct_eq(&buf[4..], expected);
    // Compute both halves to keep timing uniform regardless of which side
    // mismatched.
    magic_ok && token_ok
}

fn is_expected_disconnect(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::UnexpectedEof
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::BrokenPipe
            | io::ErrorKind::WouldBlock
            | io::ErrorKind::TimedOut
    )
}

/// Pull frames off the connection until EOF, idle timeout, or
/// unrecoverable error. `idle_timeout` sets a per-read deadline; a peer
/// that goes silent for that long is dropped.
fn read_loop(mut stream: TcpStream, registry: &Registry, idle_timeout: Duration) -> io::Result<()> {
    stream.set_read_timeout(Some(idle_timeout))?;
    loop {
        let mut len_buf = [0u8; 4];
        if let Err(e) = stream.read_exact(&mut len_buf) {
            match e.kind() {
                io::ErrorKind::UnexpectedEof => return Ok(()),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut => {
                    // Idle peer — drop. Treating timeout-as-EOF here keeps
                    // the slowloris bound at `idle_timeout` per connection.
                    return Ok(());
                }
                _ => return Err(e),
            }
        }
        let len = u32::from_be_bytes(len_buf) as usize;
        if len > MAX_PAYLOAD {
            eprintln!("cluster: oversize frame ({len} bytes), closing connection");
            return Ok(());
        }
        let mut payload = vec![0u8; len];
        if let Err(e) = stream.read_exact(&mut payload) {
            match e.kind() {
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut => return Ok(()),
                _ => return Err(e),
            }
        }
        dispatch_frame(&payload, registry);
    }
}

fn dispatch_frame(payload: &[u8], registry: &Registry) {
    let Some(nul) = payload.iter().position(|b| *b == 0) else {
        eprintln!("cluster: malformed frame (no actor_id terminator)");
        return;
    };
    let (id_bytes, rest) = payload.split_at(nul);
    let msg_bytes = &rest[1..]; // skip the NUL
    let id = match std::str::from_utf8(id_bytes) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("cluster: non-utf8 actor_id");
            return;
        }
    };
    let reg = registry.lock().expect("registry poisoned");
    match reg.get(id) {
        Some(dispatcher) => dispatcher(msg_bytes),
        None => {
            eprintln!("cluster: unknown actor id {id:?}, dropping frame");
        }
    }
}

// ---------------------------------------------------------------------------
// RemoteActorRef
// ---------------------------------------------------------------------------

pub struct RemoteActorRef<M: Codec> {
    node_addr: String,
    actor_id: String,
    conn: Mutex<Option<TcpStream>>,
    auth_token: Option<[u8; AUTH_TOKEN_LEN]>,
    _phantom: PhantomData<fn(M)>,
}

impl<M: Codec> RemoteActorRef<M> {
    /// Create a remote ref with no auth token. Use against a `Node` whose
    /// `auth_token` is `None`.
    pub fn new(node_addr: &str, actor_id: impl Into<String>) -> Self {
        Self {
            node_addr: node_addr.to_string(),
            actor_id: actor_id.into(),
            conn: Mutex::new(None),
            auth_token: None,
            _phantom: PhantomData,
        }
    }

    /// Create a remote ref that presents `token` to the server during the
    /// handshake. Required when the target `Node` has `auth_token: Some(_)`.
    pub fn with_auth(
        node_addr: &str,
        actor_id: impl Into<String>,
        token: [u8; AUTH_TOKEN_LEN],
    ) -> Self {
        Self {
            node_addr: node_addr.to_string(),
            actor_id: actor_id.into(),
            conn: Mutex::new(None),
            auth_token: Some(token),
            _phantom: PhantomData,
        }
    }

    /// Connect (or reuse the cached connection), serialise `msg`, send.
    pub fn send(&self, msg: M) -> Result<(), ClusterError> {
        let encoded = msg.encode();
        let payload_len = self.actor_id.len() + 1 + encoded.len();
        if payload_len > MAX_PAYLOAD {
            return Err(ClusterError::Oversize(payload_len));
        }

        let mut frame = Vec::with_capacity(4 + payload_len);
        frame.extend_from_slice(&(payload_len as u32).to_be_bytes());
        frame.extend_from_slice(self.actor_id.as_bytes());
        frame.push(0);
        frame.extend_from_slice(&encoded);

        let mut guard = self.conn.lock().expect("conn poisoned");
        if guard.is_none() {
            let stream = dial(&self.node_addr)?;
            // Send the handshake before any frame, so the server can drop
            // unauthenticated peers without ever seeing a payload.
            if let Some(token) = self.auth_token.as_ref()
                && let Err(e) = client_handshake(&stream, token)
            {
                return Err(ClusterError::Io(e));
            }
            *guard = Some(stream);
        }
        let stream = guard.as_mut().expect("conn populated");
        if let Err(e) = stream.write_all(&frame).and_then(|()| stream.flush()) {
            *guard = None;
            return Err(ClusterError::Io(e));
        }
        Ok(())
    }
}

fn client_handshake(mut stream: &TcpStream, token: &[u8; AUTH_TOKEN_LEN]) -> io::Result<()> {
    let mut buf = [0u8; 4 + AUTH_TOKEN_LEN];
    buf[..4].copy_from_slice(&AUTH_MAGIC);
    buf[4..].copy_from_slice(token);
    stream.write_all(&buf)?;
    stream.flush()
}

fn dial(addr: &str) -> io::Result<TcpStream> {
    let mut last_err: Option<io::Error> = None;
    let resolved: Vec<SocketAddr> = match addr.to_socket_addrs() {
        Ok(it) => it.collect(),
        Err(e) => return Err(e),
    };
    for sa in resolved {
        match TcpStream::connect_timeout(&sa, Duration::from_secs(5)) {
            Ok(s) => {
                let _ = s.set_nodelay(true);
                return Ok(s);
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err
        .unwrap_or_else(|| io::Error::new(io::ErrorKind::AddrNotAvailable, "no addrs resolved")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use actor::{Actor, spawn};

    // ---- TestMsg ----------------------------------------------------------

    #[derive(Debug, Clone, PartialEq)]
    struct TestMsg {
        kind: String,
        n: i64,
    }

    impl Codec for TestMsg {
        fn encode(&self) -> Vec<u8> {
            let mut out = String::from("{\"kind\":\"");
            anthropic::json::escape_into(&mut out, &self.kind);
            out.push_str("\",\"n\":");
            out.push_str(&self.n.to_string());
            out.push('}');
            out.into_bytes()
        }
        fn decode(bytes: &[u8]) -> Result<Self, CodecError> {
            let v = anthropic::json::parse(bytes)
                .map_err(|e| CodecError(format!("{e:?}")))?;
            let kind = v
                .get("kind")
                .and_then(|x| x.as_str())
                .ok_or_else(|| CodecError("missing kind".into()))?
                .to_string();
            let n_str = match v.get("n") {
                Some(anthropic::json::Json::Num(s)) => s.clone(),
                _ => return Err(CodecError("missing n".into())),
            };
            let n: i64 = n_str
                .parse()
                .map_err(|_| CodecError(format!("bad number: {n_str}")))?;
            Ok(TestMsg { kind, n })
        }
    }

    struct Collector {
        observed: Arc<Mutex<Vec<TestMsg>>>,
    }
    impl Actor for Collector {
        type Msg = TestMsg;
        fn handle(&mut self, msg: TestMsg) {
            self.observed.lock().expect("vec poisoned").push(msg);
        }
    }

    fn wait_until<F: FnMut() -> bool>(timeout: Duration, mut f: F) -> bool {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if f() {
                return true;
            }
            thread::sleep(Duration::from_millis(5));
        }
        false
    }

    // ---- 1: round-trip a TestMsg ------------------------------------------

    #[test]
    fn local_to_remote_roundtrip() {
        let node_a = Node::start("127.0.0.1:0").unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let s = spawn(Collector { observed: observed.clone() });
        node_a.register::<TestMsg>("greeter", s.addr.clone());

        let remote: RemoteActorRef<TestMsg> =
            RemoteActorRef::new(&node_a.local_addr().to_string(), "greeter");
        remote
            .send(TestMsg { kind: "hi".into(), n: 7 })
            .unwrap();

        assert!(wait_until(Duration::from_secs(2), || {
            !observed.lock().unwrap().is_empty()
        }));
        assert_eq!(
            observed.lock().unwrap()[0],
            TestMsg { kind: "hi".into(), n: 7 }
        );

        drop(remote);
        drop(s);
        node_a.shutdown();
    }

    // ---- 2: routing by actor_id on one node -------------------------------

    #[test]
    fn routes_by_actor_id() {
        let node = Node::start("127.0.0.1:0").unwrap();
        let a_obs = Arc::new(Mutex::new(Vec::new()));
        let b_obs = Arc::new(Mutex::new(Vec::new()));
        let sa = spawn(Collector { observed: a_obs.clone() });
        let sb = spawn(Collector { observed: b_obs.clone() });
        node.register::<TestMsg>("alpha", sa.addr.clone());
        node.register::<TestMsg>("beta", sb.addr.clone());

        let to_a: RemoteActorRef<TestMsg> =
            RemoteActorRef::new(&node.local_addr().to_string(), "alpha");
        let to_b: RemoteActorRef<TestMsg> =
            RemoteActorRef::new(&node.local_addr().to_string(), "beta");

        to_a.send(TestMsg { kind: "A".into(), n: 1 }).unwrap();
        to_b.send(TestMsg { kind: "B".into(), n: 2 }).unwrap();
        to_a.send(TestMsg { kind: "A".into(), n: 3 }).unwrap();

        assert!(wait_until(Duration::from_secs(2), || {
            a_obs.lock().unwrap().len() == 2 && b_obs.lock().unwrap().len() == 1
        }));
        let a = a_obs.lock().unwrap().clone();
        let b = b_obs.lock().unwrap().clone();
        assert_eq!(a.iter().map(|m| m.n).collect::<Vec<_>>(), vec![1, 3]);
        assert_eq!(b[0].n, 2);

        drop(to_a);
        drop(to_b);
        drop(sa);
        drop(sb);
        node.shutdown();
    }

    // ---- 3: connection reuse ---------------------------------------------

    #[test]
    fn connection_is_reused() {
        let node = Node::start("127.0.0.1:0").unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let s = spawn(Collector { observed: observed.clone() });
        node.register::<TestMsg>("sink", s.addr.clone());

        let remote: RemoteActorRef<TestMsg> =
            RemoteActorRef::new(&node.local_addr().to_string(), "sink");
        for i in 0..25 {
            remote.send(TestMsg { kind: "tick".into(), n: i }).unwrap();
        }

        assert!(wait_until(Duration::from_secs(2), || {
            observed.lock().unwrap().len() == 25
        }));

        // Exactly one TCP connection (one reader thread) should be alive.
        assert_eq!(node.reader_count(), 1, "expected single reused connection");

        drop(remote);
        drop(s);
        node.shutdown();
    }

    // ---- 4: send to unknown actor is fire-and-forget ----------------------

    #[test]
    fn unknown_actor_is_ignored_not_errored() {
        let node = Node::start("127.0.0.1:0").unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let s = spawn(Collector { observed: observed.clone() });
        node.register::<TestMsg>("known", s.addr.clone());

        let nope: RemoteActorRef<TestMsg> =
            RemoteActorRef::new(&node.local_addr().to_string(), "nope");
        let known: RemoteActorRef<TestMsg> =
            RemoteActorRef::new(&node.local_addr().to_string(), "known");

        nope.send(TestMsg { kind: "ignored".into(), n: 0 }).unwrap();
        known.send(TestMsg { kind: "ok".into(), n: 9 }).unwrap();

        assert!(wait_until(Duration::from_secs(2), || {
            !observed.lock().unwrap().is_empty()
        }));
        let got = observed.lock().unwrap().clone();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].n, 9);

        drop(nope);
        drop(known);
        drop(s);
        node.shutdown();
    }

    // ---- 5: concurrent senders -------------------------------------------

    #[test]
    fn concurrent_senders_all_arrive() {
        let node = Node::start("127.0.0.1:0").unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let s = spawn(Collector { observed: observed.clone() });
        node.register::<TestMsg>("hub", s.addr.clone());

        let addr = node.local_addr().to_string();
        let senders: Vec<_> = (0..8)
            .map(|w| {
                let addr = addr.clone();
                thread::spawn(move || {
                    let remote: RemoteActorRef<TestMsg> = RemoteActorRef::new(&addr, "hub");
                    for i in 0..50 {
                        remote
                            .send(TestMsg { kind: format!("w{w}"), n: i })
                            .unwrap();
                    }
                })
            })
            .collect();
        for w in senders {
            w.join().unwrap();
        }

        assert!(wait_until(Duration::from_secs(5), || {
            observed.lock().unwrap().len() == 400
        }));

        drop(s);
        node.shutdown();
    }

    // ---- 6: codec decode error doesn't crash the connection --------------

    #[test]
    fn codec_error_logs_and_continues() {
        struct AlwaysFails;
        impl Codec for AlwaysFails {
            fn encode(&self) -> Vec<u8> {
                vec![0xff]
            }
            fn decode(_: &[u8]) -> Result<Self, CodecError> {
                Err(CodecError("nope".into()))
            }
        }

        let node = Node::start("127.0.0.1:0").unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let s = spawn(Collector { observed: observed.clone() });
        node.register::<TestMsg>("good", s.addr.clone());

        struct Sink;
        impl Actor for Sink {
            type Msg = AlwaysFails;
            fn handle(&mut self, _: AlwaysFails) {
                unreachable!("decode always errs")
            }
        }
        let s_bad = spawn(Sink);
        node.register::<AlwaysFails>("bad", s_bad.addr.clone());

        let to_bad: RemoteActorRef<AlwaysFails> =
            RemoteActorRef::new(&node.local_addr().to_string(), "bad");
        to_bad.send(AlwaysFails).unwrap();

        let to_good: RemoteActorRef<TestMsg> =
            RemoteActorRef::new(&node.local_addr().to_string(), "good");
        to_good.send(TestMsg { kind: "after".into(), n: 1 }).unwrap();

        assert!(wait_until(Duration::from_secs(2), || {
            !observed.lock().unwrap().is_empty()
        }));
        assert_eq!(observed.lock().unwrap()[0].n, 1);

        drop(to_bad);
        drop(to_good);
        drop(s);
        drop(s_bad);
        node.shutdown();
    }

    // ---- 7: oversize payload rejected on send ----------------------------

    #[test]
    fn oversize_payload_is_rejected_on_send() {
        struct Big(Vec<u8>);
        impl Codec for Big {
            fn encode(&self) -> Vec<u8> {
                self.0.clone()
            }
            fn decode(b: &[u8]) -> Result<Self, CodecError> {
                Ok(Big(b.to_vec()))
            }
        }

        let node = Node::start("127.0.0.1:0").unwrap();
        struct Drain;
        impl Actor for Drain {
            type Msg = Big;
            fn handle(&mut self, _: Big) {}
        }
        let s = spawn(Drain);
        node.register::<Big>("drain", s.addr.clone());

        let remote: RemoteActorRef<Big> =
            RemoteActorRef::new(&node.local_addr().to_string(), "drain");
        let too_big = Big(vec![0u8; MAX_PAYLOAD + 1]);
        match remote.send(too_big) {
            Err(ClusterError::Oversize(_)) => {}
            other => panic!("expected Oversize, got {other:?}"),
        }

        drop(remote);
        drop(s);
        node.shutdown();
    }

    // ---- 8: shutdown joins listener cleanly ------------------------------

    #[test]
    fn shutdown_joins_cleanly() {
        let node = Node::start("127.0.0.1:0").unwrap();
        let start = Instant::now();
        node.shutdown();
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "shutdown should be near-instant"
        );
    }

    // ---- 9: shutdown with live readers ------------------------------------

    #[test]
    fn shutdown_with_open_connection() {
        let node = Node::start_with_config(
            "127.0.0.1:0",
            NodeConfig {
                auth_token: None,
                idle_timeout: Duration::from_secs(2),
            },
        )
        .unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let s = spawn(Collector { observed: observed.clone() });
        node.register::<TestMsg>("x", s.addr.clone());

        let remote: RemoteActorRef<TestMsg> =
            RemoteActorRef::new(&node.local_addr().to_string(), "x");
        remote.send(TestMsg { kind: "one".into(), n: 1 }).unwrap();
        assert!(wait_until(Duration::from_secs(2), || {
            !observed.lock().unwrap().is_empty()
        }));

        drop(remote);
        drop(s);
        let t = Instant::now();
        node.shutdown();
        assert!(t.elapsed() < Duration::from_secs(4));
    }

    // ---- 10: write failure drops cached connection, next send recovers ----

    #[test]
    fn write_failure_recovers_on_next_send() {
        let node = Node::start("127.0.0.1:0").unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let s = spawn(Collector { observed: observed.clone() });
        node.register::<TestMsg>("y", s.addr.clone());

        let remote: RemoteActorRef<TestMsg> =
            RemoteActorRef::new(&node.local_addr().to_string(), "y");
        remote.send(TestMsg { kind: "first".into(), n: 1 }).unwrap();
        assert!(wait_until(Duration::from_secs(2), || {
            !observed.lock().unwrap().is_empty()
        }));

        {
            let mut g = remote.conn.lock().unwrap();
            if let Some(stream) = g.as_mut() {
                let _ = stream.shutdown(Shutdown::Both);
            }
        }
        let _ = remote.send(TestMsg { kind: "shed".into(), n: 2 });
        let mut ok = false;
        for _ in 0..3 {
            if remote.send(TestMsg { kind: "retry".into(), n: 3 }).is_ok() {
                ok = true;
                break;
            }
        }
        assert!(ok, "post-failure send should eventually succeed");
        assert!(wait_until(Duration::from_secs(2), || {
            observed.lock().unwrap().iter().any(|m| m.n == 3)
        }));

        drop(remote);
        drop(s);
        node.shutdown();
    }

    // ---- 11: two nodes, A talks to B and vice versa -----------------------

    #[test]
    fn two_nodes_bidirectional() {
        let a = Node::start("127.0.0.1:0").unwrap();
        let b = Node::start("127.0.0.1:0").unwrap();
        let a_obs = Arc::new(Mutex::new(Vec::new()));
        let b_obs = Arc::new(Mutex::new(Vec::new()));
        let sa = spawn(Collector { observed: a_obs.clone() });
        let sb = spawn(Collector { observed: b_obs.clone() });
        a.register::<TestMsg>("on_a", sa.addr.clone());
        b.register::<TestMsg>("on_b", sb.addr.clone());

        let to_b: RemoteActorRef<TestMsg> =
            RemoteActorRef::new(&b.local_addr().to_string(), "on_b");
        let to_a: RemoteActorRef<TestMsg> =
            RemoteActorRef::new(&a.local_addr().to_string(), "on_a");
        to_b.send(TestMsg { kind: "->b".into(), n: 1 }).unwrap();
        to_a.send(TestMsg { kind: "->a".into(), n: 2 }).unwrap();

        assert!(wait_until(Duration::from_secs(2), || {
            !a_obs.lock().unwrap().is_empty() && !b_obs.lock().unwrap().is_empty()
        }));
        assert_eq!(a_obs.lock().unwrap()[0].n, 2);
        assert_eq!(b_obs.lock().unwrap()[0].n, 1);

        drop(to_a);
        drop(to_b);
        drop(sa);
        drop(sb);
        a.shutdown();
        b.shutdown();
    }

    // ---- 12: malformed frame (no NUL) doesn't crash connection -----------

    #[test]
    fn malformed_frame_no_nul_is_dropped() {
        let node = Node::start("127.0.0.1:0").unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let s = spawn(Collector { observed: observed.clone() });
        node.register::<TestMsg>("z", s.addr.clone());

        let mut sock = TcpStream::connect(node.local_addr()).unwrap();
        let bogus = b"not-null-terminated-anywhere";
        let len = (bogus.len() as u32).to_be_bytes();
        sock.write_all(&len).unwrap();
        sock.write_all(bogus).unwrap();

        let real = TestMsg { kind: "ok".into(), n: 42 };
        let encoded = real.encode();
        let id = "z";
        let payload_len = (id.len() + 1 + encoded.len()) as u32;
        sock.write_all(&payload_len.to_be_bytes()).unwrap();
        sock.write_all(id.as_bytes()).unwrap();
        sock.write_all(&[0]).unwrap();
        sock.write_all(&encoded).unwrap();
        sock.flush().unwrap();

        assert!(wait_until(Duration::from_secs(2), || {
            observed.lock().unwrap().iter().any(|m| m.n == 42)
        }));

        drop(sock);
        drop(s);
        node.shutdown();
    }

    // ---- 13: connection without token rejected when auth required --------

    #[test]
    fn connection_without_token_rejected_when_auth_required() {
        let token = [0x42u8; AUTH_TOKEN_LEN];
        let node = Node::start_with_config(
            "127.0.0.1:0",
            NodeConfig {
                auth_token: Some(token),
                idle_timeout: Duration::from_secs(5),
            },
        )
        .unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let s = spawn(Collector { observed: observed.clone() });
        node.register::<TestMsg>("vault", s.addr.clone());

        // No token at all — server reads the first 36 bytes of the data
        // frame as a (wrong) handshake and closes. The kernel may accept
        // a few bytes before the FIN propagates, so loop until a send fails.
        let anon: RemoteActorRef<TestMsg> =
            RemoteActorRef::new(&node.local_addr().to_string(), "vault");
        let mut saw_err = false;
        for _ in 0..40 {
            if anon
                .send(TestMsg { kind: "knock".into(), n: 1 })
                .is_err()
            {
                saw_err = true;
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
        assert!(saw_err, "unauthenticated send must eventually fail");

        // Wrong token — same outcome.
        let wrong: RemoteActorRef<TestMsg> = RemoteActorRef::with_auth(
            &node.local_addr().to_string(),
            "vault",
            [0xAAu8; AUTH_TOKEN_LEN],
        );
        let mut saw_err2 = false;
        for _ in 0..40 {
            if wrong
                .send(TestMsg { kind: "knock".into(), n: 2 })
                .is_err()
            {
                saw_err2 = true;
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
        assert!(saw_err2, "wrong-token send must eventually fail");

        // No frames delivered.
        thread::sleep(Duration::from_millis(200));
        assert!(observed.lock().unwrap().is_empty());

        drop(anon);
        drop(wrong);
        drop(s);
        node.shutdown();
    }

    // ---- 14: connection with correct token succeeds ----------------------

    #[test]
    fn connection_with_correct_token_succeeds() {
        let token = [0x77u8; AUTH_TOKEN_LEN];
        let node = Node::start_with_config(
            "127.0.0.1:0",
            NodeConfig {
                auth_token: Some(token),
                idle_timeout: Duration::from_secs(5),
            },
        )
        .unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let s = spawn(Collector { observed: observed.clone() });
        node.register::<TestMsg>("door", s.addr.clone());

        let remote: RemoteActorRef<TestMsg> = RemoteActorRef::with_auth(
            &node.local_addr().to_string(),
            "door",
            token,
        );
        remote
            .send(TestMsg { kind: "open".into(), n: 12 })
            .unwrap();

        assert!(wait_until(Duration::from_secs(2), || {
            !observed.lock().unwrap().is_empty()
        }));
        assert_eq!(observed.lock().unwrap()[0].n, 12);

        drop(remote);
        drop(s);
        node.shutdown();
    }

    // ---- 15: idle peer disconnected within timeout -----------------------

    #[test]
    fn idle_peer_disconnected_within_timeout() {
        let node = Node::start_with_config(
            "127.0.0.1:0",
            NodeConfig {
                auth_token: None,
                idle_timeout: Duration::from_secs(2),
            },
        )
        .unwrap();

        let sock = TcpStream::connect(node.local_addr()).unwrap();
        // Reader thread should be live almost immediately.
        assert!(wait_until(Duration::from_secs(1), || {
            node.reader_count() == 1
        }));

        // Within `idle_timeout + slack` the reader should drop the silent peer.
        let dropped = wait_until(Duration::from_secs(5), || node.reader_count() == 0);
        assert!(dropped, "idle reader should be reaped");

        drop(sock);
        node.shutdown();
    }

    // ---- 16: reader count returns to zero after peer disconnects ---------

    #[test]
    fn reader_count_returns_to_zero_after_peer_disconnects() {
        let node = Node::start("127.0.0.1:0").unwrap();

        for _ in 0..10 {
            let sock = TcpStream::connect(node.local_addr()).unwrap();
            assert!(wait_until(Duration::from_secs(1), || {
                node.reader_count() >= 1
            }));
            drop(sock);
            assert!(wait_until(Duration::from_secs(1), || {
                node.reader_count() == 0
            }));
        }

        assert_eq!(node.reader_count(), 0);
        node.shutdown();
    }
}
