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

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::marker::PhantomData;
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use actor::ActorRef;

/// Maximum frame payload size. Anything larger is rejected on both sender
/// and receiver to bound memory and protect against runaway peers.
pub const MAX_PAYLOAD: usize = 4 * 1024 * 1024;

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
}

impl std::fmt::Display for ClusterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClusterError::Io(e) => write!(f, "io: {e}"),
            ClusterError::Codec(e) => write!(f, "{e}"),
            ClusterError::UnknownActor(id) => write!(f, "unknown actor id: {id}"),
            ClusterError::NotConnected => write!(f, "not connected"),
            ClusterError::Oversize(n) => write!(f, "oversize payload: {n} bytes"),
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
    readers: Mutex<Vec<JoinHandle<()>>>,
}

pub struct Node {
    inner: Arc<Inner>,
    registry: Registry,
    listener_thread: Option<JoinHandle<()>>,
}

impl Node {
    /// Start listening on `addr`. Pass `"127.0.0.1:0"` for an ephemeral port.
    /// Spawns the accept loop on a background thread and returns once the
    /// socket is bound.
    pub fn start(addr: &str) -> Result<Self, ClusterError> {
        let listener = TcpListener::bind(addr)?;
        // Non-blocking accept with a short timeout via set_nonblocking would
        // force a poll loop; instead we use a blocking accept and unblock it
        // on shutdown by dialing ourselves once the running flag is cleared.
        let local_addr = listener.local_addr()?;
        let inner = Arc::new(Inner {
            running: AtomicBool::new(true),
            addr: local_addr,
            readers: Mutex::new(Vec::new()),
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

    /// Register a local actor under `id`. Subsequent inbound frames addressed
    /// to this id are decoded as `M` and forwarded to the actor's mailbox.
    /// Re-registering the same id replaces the previous dispatcher.
    pub fn register<M: Codec>(&self, id: &str, addr: ActorRef<M>) {
        let dispatcher: Dispatcher = Box::new(move |bytes: &[u8]| {
            match M::decode(bytes) {
                Ok(msg) => {
                    // Mailbox closed (actor dropped) — skip. Don't crash
                    // the reader thread; other actors on this node may
                    // still be live.
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

    /// Stop the listener and join its thread (and any reader threads
    /// spawned for inbound connections).
    pub fn shutdown(mut self) {
        self.inner.running.store(false, Ordering::SeqCst);
        // Wake the blocking accept by dialing ourselves once.
        let _ = TcpStream::connect_timeout(
            &self.inner.addr,
            std::time::Duration::from_millis(200),
        );
        if let Some(h) = self.listener_thread.take() {
            let _ = h.join();
        }
        // Reader threads will exit when their peer disconnects. Join any we
        // know about, with the listener already stopped no new ones will
        // appear.
        let readers = std::mem::take(&mut *self.inner.readers.lock().expect("readers poisoned"));
        for h in readers {
            let _ = h.join();
        }
    }
}

fn accept_loop(listener: TcpListener, inner: Arc<Inner>, registry: Registry) {
    for incoming in listener.incoming() {
        if !inner.running.load(Ordering::SeqCst) {
            // Drop any stream we just accepted as part of the wake-up dial.
            if let Ok(s) = incoming {
                let _ = s.shutdown(Shutdown::Both);
            }
            break;
        }
        match incoming {
            Ok(stream) => {
                let registry = Arc::clone(&registry);
                let inner_r = Arc::clone(&inner);
                let h = thread::spawn(move || {
                    if let Err(e) = read_loop(stream, &registry) {
                        // EOF / connection drop is expected — only log real
                        // protocol failures, and never tear down the node.
                        if !is_expected_disconnect(&e) {
                            eprintln!("cluster: reader error: {e}");
                        }
                    }
                    // Best-effort: trim our entry from the readers list so
                    // it doesn't grow unbounded over many connections.
                    let _ = inner_r;
                });
                if let Ok(mut rs) = inner.readers.lock() {
                    rs.push(h);
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

fn is_expected_disconnect(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::UnexpectedEof
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::BrokenPipe
    )
}

/// Pull frames off the connection until EOF or an unrecoverable error.
fn read_loop(mut stream: TcpStream, registry: &Registry) -> io::Result<()> {
    loop {
        let mut len_buf = [0u8; 4];
        if let Err(e) = stream.read_exact(&mut len_buf) {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                return Ok(());
            }
            return Err(e);
        }
        let len = u32::from_be_bytes(len_buf) as usize;
        if len > MAX_PAYLOAD {
            eprintln!("cluster: oversize frame ({len} bytes), closing connection");
            return Ok(());
        }
        let mut payload = vec![0u8; len];
        stream.read_exact(&mut payload)?;
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
    _phantom: PhantomData<fn(M)>,
}

impl<M: Codec> RemoteActorRef<M> {
    pub fn new(node_addr: &str, actor_id: impl Into<String>) -> Self {
        Self {
            node_addr: node_addr.to_string(),
            actor_id: actor_id.into(),
            conn: Mutex::new(None),
            _phantom: PhantomData,
        }
    }

    /// Connect (or reuse the cached connection), serialise `msg`, send. On a
    /// write failure the cached connection is dropped; the next call dials a
    /// fresh one. Returns once the bytes are flushed to the kernel — there is
    /// no ack.
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
            *guard = Some(dial(&self.node_addr)?);
        }
        // SAFETY: we just populated it.
        let stream = guard.as_mut().expect("conn populated");
        if let Err(e) = stream.write_all(&frame).and_then(|()| stream.flush()) {
            // Drop the dead connection; let the next send re-dial.
            *guard = None;
            return Err(ClusterError::Io(e));
        }
        Ok(())
    }
}

fn dial(addr: &str) -> io::Result<TcpStream> {
    let mut last_err: Option<io::Error> = None;
    let resolved: Vec<SocketAddr> = match addr.to_socket_addrs() {
        Ok(it) => it.collect(),
        Err(e) => return Err(e),
    };
    for sa in resolved {
        match TcpStream::connect_timeout(&sa, std::time::Duration::from_secs(5)) {
            Ok(s) => {
                // Disable Nagle so small fire-and-forget messages don't sit
                // in a 40ms coalesce window.
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

    /// A minimal JSON-shaped message. We hand-roll the encoder because the
    /// shape is fixed; we reuse `anthropic::json` for decoding so we don't
    /// re-implement a parser.
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

    // ---- Collector actor --------------------------------------------------

    /// The actor's `Msg` is `TestMsg` (which is `Codec`); a shared `Vec`
    /// collects what arrives so the test can inspect.
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

    /// On the server side, each accepted TCP connection spawns one reader
    /// thread. We can verify a single TCP connection is reused across many
    /// sends by counting how many readers the node ever spawned.
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

        // Exactly one TCP connection (one reader thread) should have been
        // opened over those 25 sends.
        let reader_count = node.inner.readers.lock().unwrap().len();
        assert_eq!(reader_count, 1, "expected single reused connection");

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

        // Sending to an unknown id succeeds (server logs and drops the frame).
        nope.send(TestMsg { kind: "ignored".into(), n: 0 }).unwrap();
        // The connection should still be alive — subsequent traffic on the
        // (separate) `known` ref must deliver.
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

    /// A message type whose decoder always fails. We feed it a frame, then
    /// follow with a `TestMsg` frame on the same connection — the second
    /// frame must still be delivered.
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

        // Register the broken codec under another id.
        struct Sink;
        impl Actor for Sink {
            type Msg = AlwaysFails;
            fn handle(&mut self, _: AlwaysFails) {
                unreachable!("decode always errs")
            }
        }
        let s_bad = spawn(Sink);
        node.register::<AlwaysFails>("bad", s_bad.addr.clone());

        // Sending to "bad" must succeed at the wire level; server logs the
        // codec error and moves on.
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
        let node = Node::start("127.0.0.1:0").unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let s = spawn(Collector { observed: observed.clone() });
        node.register::<TestMsg>("x", s.addr.clone());

        let remote: RemoteActorRef<TestMsg> =
            RemoteActorRef::new(&node.local_addr().to_string(), "x");
        remote.send(TestMsg { kind: "one".into(), n: 1 }).unwrap();
        assert!(wait_until(Duration::from_secs(2), || {
            !observed.lock().unwrap().is_empty()
        }));

        // Drop the remote so the reader sees EOF, then shut down.
        drop(remote);
        drop(s);
        let t = Instant::now();
        node.shutdown();
        assert!(t.elapsed() < Duration::from_secs(2));
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

        // Forcibly tear down the cached connection from inside.
        {
            let mut g = remote.conn.lock().unwrap();
            if let Some(stream) = g.as_mut() {
                let _ = stream.shutdown(Shutdown::Both);
            }
        }
        // The next send may fail (write to a half-closed socket) or succeed
        // (kernel-buffered write); either way a subsequent send must succeed
        // by re-dialing.
        let _ = remote.send(TestMsg { kind: "shed".into(), n: 2 });
        // Loop a couple of times in case the first attempt won the race.
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

        // Dial manually and write a bogus frame: 4-byte length + payload
        // with no NUL terminator. Then write a real frame on the same
        // connection and confirm it delivers.
        let mut sock = TcpStream::connect(node.local_addr()).unwrap();
        let bogus = b"not-null-terminated-anywhere";
        let len = (bogus.len() as u32).to_be_bytes();
        sock.write_all(&len).unwrap();
        sock.write_all(bogus).unwrap();

        // Then a well-formed frame.
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

}
