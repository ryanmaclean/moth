//! Minimal actor primitive.
//!
//! One OS thread per actor, `std::sync::mpsc` mailboxes, zero dependencies.
//! Actors drain when every ActorRef is dropped, or stop cooperatively after
//! an explicit cancellation request. Keep the join owner until retirement.
//!
//! Not a scheduler. Not a supervision tree. Not a runtime. It's the smallest
//! thing that lets one piece of state own its mailbox and process messages
//! in order. We add more only when a real workload forces it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SendError, Sender, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Cooperative stop request. A request is not a worker-retirement receipt.
#[derive(Clone, Default)]
pub struct Cancellation(Arc<AtomicBool>);

impl Cancellation {
    pub fn request(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_requested(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Polling retains the handle while its main function is running. Keep this
/// owner until Joined. is_finished + join is the standard polling idiom; it
/// is not a hard wall-clock bound on final OS/thread-local cleanup.
#[must_use = "dropping Retirement detaches the still-owned thread"]
pub struct Retirement<T> {
    handle: Option<JoinHandle<T>>,
}

pub enum JoinPoll<T> {
    Pending,
    Joined(thread::Result<T>),
    AlreadyJoined,
}

impl<T> Retirement<T> {
    pub fn new(handle: JoinHandle<T>) -> Self {
        Self { handle: Some(handle) }
    }
    pub fn try_join(&mut self) -> JoinPoll<T> {
        match self.handle.as_ref() {
            None => JoinPoll::AlreadyJoined,
            Some(handle) if !handle.is_finished() => JoinPoll::Pending,
            Some(_) => JoinPoll::Joined(self.handle.take().expect("owned handle").join()),
        }
    }
    /// Explicitly blocking; do not call on a deadline/control-plane path.
    pub fn join(mut self) -> thread::Result<T> {
        self.handle.take().expect("retirement already joined").join()
    }
}

pub trait Actor: Send + 'static {
    type Msg: Send + 'static;

    fn handle(&mut self, msg: Self::Msg);

    /// Installed before spawn. Overrides may observe cancellation inside a
    /// handler; the default stops only between messages, never interrupts it.
    fn set_cancellation(&mut self, _cancel: Cancellation) {}

    /// Called once after normal drain, cancellation or handler failure.
    fn stopped(&mut self) {}
}

pub struct ActorRef<M: Send + 'static> {
    tx: Sender<M>,
}

impl<M: Send + 'static> Clone for ActorRef<M> {
    fn clone(&self) -> Self {
        Self { tx: self.tx.clone() }
    }
}

impl<M: Send + 'static> ActorRef<M> {
    pub fn send(&self, msg: M) -> Result<(), SendError<M>> {
        self.tx.send(msg)
    }

    /// Request/reply. Caller builds the message given a reply channel; we
    /// block on the reply. Returns `Err` if the mailbox closed or the actor
    /// dropped the reply channel without sending.
    pub fn ask<R, F>(&self, make_msg: F) -> Result<R, AskError>
    where
        R: Send + 'static,
        F: FnOnce(SyncSender<R>) -> M,
    {
        let (reply_tx, reply_rx) = sync_channel(1);
        self.tx.send(make_msg(reply_tx)).map_err(|_| AskError::MailboxClosed)?;
        reply_rx.recv().map_err(|_| AskError::NoReply)
    }
}

#[derive(Debug)]
pub enum AskError {
    MailboxClosed,
    NoReply,
}

#[must_use = "dropping Spawned detaches the actor's thread"]
pub struct Spawned<M: Send + 'static> {
    pub addr: ActorRef<M>,
    handle: JoinHandle<()>,
    cancel: Cancellation,
}

impl<M: Send + 'static> Spawned<M> {
    pub fn cancellation(&self) -> Cancellation {
        self.cancel.clone()
    }
    pub fn request_stop(&self) {
        self.cancel.request();
    }
    /// Close this owner's address; other clones remain live until stopped.
    /// This does not request cancellation: queued work normally drains.
    pub fn into_retirement(self) -> Retirement<()> {
        let Self { addr, handle, .. } = self;
        drop(addr);
        Retirement::new(handle)
    }

    /// Drop the internal `ActorRef` and block until the actor's thread exits.
    ///
    /// If the caller cloned `addr` elsewhere, those clones must drop before
    /// this returns, unless cancellation or a handler failure stops dispatch.
    pub fn join(self) -> thread::Result<()> {
        let Spawned { addr, handle, .. } = self;
        drop(addr);
        handle.join()
    }
}

pub fn spawn<A: Actor>(actor: A) -> Spawned<A::Msg> {
    try_spawn(actor).expect("actor thread creation failed")
}

/// Recoverable admission for lifecycle owners that already hold resources.
pub fn try_spawn<A: Actor>(actor: A) -> std::io::Result<Spawned<A::Msg>> {
    try_spawn_with_cancellation(actor, Cancellation::default())
}

/// Compatibility wrapper. Resource-owning callers should use the fallible API.
pub fn spawn_with_cancellation<A: Actor>(actor: A, cancel: Cancellation) -> Spawned<A::Msg> {
    try_spawn_with_cancellation(actor, cancel).expect("actor thread creation failed")
}

pub fn try_spawn_with_cancellation<A: Actor>(
    actor: A,
    cancel: Cancellation,
) -> std::io::Result<Spawned<A::Msg>> {
    try_spawn_using(actor, cancel, |work| thread::Builder::new().spawn(work))
}

fn try_spawn_using<A: Actor>(
    mut actor: A,
    cancel: Cancellation,
    launch: impl FnOnce(Box<dyn FnOnce() + Send>) -> std::io::Result<JoinHandle<()>>,
) -> std::io::Result<Spawned<A::Msg>> {
    actor.set_cancellation(cancel.clone());
    let (tx, rx) = mpsc::channel();
    let worker_cancel = cancel.clone();
    let handle = launch(Box::new(move || run_loop(actor, rx, worker_cancel)))?;
    Ok(Spawned { addr: ActorRef { tx }, handle, cancel })
}

/// Bounded mailbox. A blocked sender wakes once the receiver retires;
/// cancellation cannot force a currently blocked handler to return.
pub fn spawn_bounded<A: Actor>(mut actor: A, capacity: usize) -> SpawnedSync<A::Msg> {
    let cancel = Cancellation::default();
    actor.set_cancellation(cancel.clone());
    let (tx, rx) = sync_channel::<A::Msg>(capacity);
    let worker_cancel = cancel.clone();
    let handle = thread::spawn(move || run_loop(actor, rx, worker_cancel));
    SpawnedSync { addr: SyncActorRef { tx }, handle, cancel }
}

/// Keep the first handler failure observable at join, while still invoking
/// stopped exactly once. A stopped-hook panic also makes join fail.
fn run_loop<A: Actor>(mut actor: A, rx: Receiver<A::Msg>, cancel: Cancellation) {
    let mut failure = None;
    while !cancel.is_requested() {
        let msg = match rx.recv_timeout(Duration::from_millis(10)) {
            Ok(msg) => msg,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        if cancel.is_requested() {
            break;
        }
        if let Err(payload) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| actor.handle(msg)))
        {
            log_panic(&payload);
            failure = Some(payload);
            break;
        }
    }
    let stopped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| actor.stopped()));
    if let Some(payload) = failure {
        std::panic::resume_unwind(payload);
    }
    if let Err(payload) = stopped {
        std::panic::resume_unwind(payload);
    }
}

fn log_panic(payload: &Box<dyn std::any::Any + Send>) {
    // Diagnostics must not skip stopped() or replace the original panic.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = write_panic(&mut std::io::stderr().lock(), payload);
    }));
}

fn write_panic(
    out: &mut dyn std::io::Write,
    payload: &Box<dyn std::any::Any + Send>,
) -> std::io::Result<()> {
    let msg = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic>");
    writeln!(out, "actor: handler panicked: {msg}")
}

/// Bounded-mailbox counterpart of `Spawned`. The `SyncActorRef::send` call
/// applies backpressure: it blocks until either capacity frees up or the
/// receiver is gone.
#[must_use = "dropping SpawnedSync detaches the actor's thread"]
pub struct SpawnedSync<M: Send + 'static> {
    pub addr: SyncActorRef<M>,
    handle: JoinHandle<()>,
    cancel: Cancellation,
}

impl<M: Send + 'static> SpawnedSync<M> {
    pub fn cancellation(&self) -> Cancellation {
        self.cancel.clone()
    }
    pub fn request_stop(&self) {
        self.cancel.request();
    }
    pub fn into_retirement(self) -> Retirement<()> {
        let Self { addr, handle, .. } = self;
        drop(addr);
        Retirement::new(handle)
    }

    pub fn join(self) -> thread::Result<()> {
        let SpawnedSync { addr, handle, .. } = self;
        drop(addr);
        handle.join()
    }
}

pub struct SyncActorRef<M: Send + 'static> {
    tx: SyncSender<M>,
}

impl<M: Send + 'static> Clone for SyncActorRef<M> {
    fn clone(&self) -> Self {
        Self { tx: self.tx.clone() }
    }
}

impl<M: Send + 'static> SyncActorRef<M> {
    /// Send; blocks if the mailbox is at capacity. Returns `Err` if the
    /// receiver thread has exited (panicked or run to completion).
    pub fn send(&self, msg: M) -> Result<(), std::sync::mpsc::SendError<M>> {
        self.tx.send(msg)
    }

    /// Try-send. Returns `Err(Full)` if the mailbox is at capacity; lets
    /// callers shed load instead of blocking when the actor is slow.
    pub fn try_send(&self, msg: M) -> Result<(), std::sync::mpsc::TrySendError<M>> {
        self.tx.try_send(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counter {
        n: usize,
        observed: Arc<AtomicUsize>,
    }
    enum Count {
        Inc,
        Get(SyncSender<usize>),
    }
    impl Actor for Counter {
        type Msg = Count;
        fn handle(&mut self, msg: Count) {
            match msg {
                Count::Inc => self.n += 1,
                Count::Get(reply) => {
                    let _ = reply.send(self.n);
                }
            }
        }
        fn stopped(&mut self) {
            self.observed.store(self.n, Ordering::SeqCst);
        }
    }

    #[test]
    fn send_and_ask() {
        let s = spawn(Counter { n: 0, observed: Arc::new(AtomicUsize::new(0)) });
        for _ in 0..1000 {
            s.addr.send(Count::Inc).unwrap();
        }
        assert_eq!(s.addr.ask(Count::Get).unwrap(), 1000);
        s.join().unwrap();
    }

    #[test]
    fn clone_sends_from_many_threads() {
        let s = spawn(Counter { n: 0, observed: Arc::new(AtomicUsize::new(0)) });
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let addr = s.addr.clone();
                thread::spawn(move || {
                    for _ in 0..1000 {
                        addr.send(Count::Inc).unwrap();
                    }
                })
            })
            .collect();
        for w in workers {
            w.join().unwrap();
        }
        assert_eq!(s.addr.ask(Count::Get).unwrap(), 8000);
        s.join().unwrap();
    }

    #[test]
    fn join_runs_stopped_hook() {
        let observed = Arc::new(AtomicUsize::new(0));
        let s = spawn(Counter { n: 0, observed: observed.clone() });
        s.addr.send(Count::Inc).unwrap();
        s.addr.send(Count::Inc).unwrap();
        s.addr.send(Count::Inc).unwrap();
        s.join().unwrap();
        assert_eq!(observed.load(Ordering::SeqCst), 3);
    }

    // ----- panic isolation -----------------------------------------------

    struct Panicker;
    enum PMsg {
        Ok,
        Boom,
    }
    impl Actor for Panicker {
        type Msg = PMsg;
        fn handle(&mut self, msg: PMsg) {
            match msg {
                PMsg::Ok => {}
                PMsg::Boom => panic!("intentional"),
            }
        }
    }

    #[test]
    fn handler_panic_closes_mailbox_without_killing_caller() {
        let s = spawn(Panicker);
        // First message is fine.
        s.addr.send(PMsg::Ok).unwrap();
        // A handler panic stays on its worker and is observable through join.
        // The send itself succeeds because the channel is still open at
        // the moment we push.
        let _ = s.addr.send(PMsg::Boom);
        // Subsequent sends MAY succeed (race: thread might not have
        // exited yet) or fail (Err once the receiver scope drops).
        // What matters is that the test process doesn't crash.
        std::thread::sleep(std::time::Duration::from_millis(10));
        let _ = s.addr.send(PMsg::Ok);
        // Join is safe; thread already exited.
        let _ = s.join();
    }

    // ----- bounded mailbox -----------------------------------------------

    #[test]
    fn spawn_bounded_applies_backpressure() {
        use std::sync::mpsc::TrySendError;

        // Capacity 2: third try_send returns Full while the actor is busy.
        struct Slow;
        impl Actor for Slow {
            type Msg = ();
            fn handle(&mut self, _: ()) {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
        let s = spawn_bounded(Slow, 2);
        // First two queue immediately.
        s.addr.try_send(()).unwrap();
        s.addr.try_send(()).unwrap();
        // Third hits the capacity wall right away (worker is sleeping on #1).
        let third = s.addr.try_send(());
        assert!(matches!(third, Err(TrySendError::Full(_))));
        s.join().unwrap();
    }

    #[test]
    fn handler_failure_is_observable_after_stopped() {
        struct Fails(Arc<AtomicUsize>);
        impl Actor for Fails {
            type Msg = ();
            fn handle(&mut self, _: ()) {
                panic!("handler failure");
            }
            fn stopped(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let stopped = Arc::new(AtomicUsize::new(0));
        let worker = spawn(Fails(stopped.clone()));
        worker.addr.send(()).unwrap();
        assert!(worker.join().is_err());
        assert_eq!(stopped.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stop_retains_inflight_worker_and_discards_queued_work() {
        struct Gated {
            entered: SyncSender<()>,
            release: Receiver<()>,
            calls: Arc<AtomicUsize>,
        }
        impl Actor for Gated {
            type Msg = ();
            fn handle(&mut self, _: ()) {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.entered.send(()).unwrap();
                self.release.recv().unwrap();
            }
        }
        let (entered_tx, entered_rx) = sync_channel(1);
        let (release_tx, release_rx) = sync_channel(1);
        let calls = Arc::new(AtomicUsize::new(0));
        let worker =
            spawn(Gated { entered: entered_tx, release: release_rx, calls: calls.clone() });
        let kept_address = worker.addr.clone();
        worker.addr.send(()).unwrap();
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.addr.send(()).unwrap();
        worker.request_stop();
        let mut retirement = worker.into_retirement();
        assert!(matches!(retirement.try_join(), JoinPoll::Pending));
        release_tx.send(()).unwrap();
        retirement.join().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(kept_address.send(()).is_err());
    }

    #[test]
    fn bounded_actor_reports_stopped_hook_failure() {
        struct StopsBadly;
        impl Actor for StopsBadly {
            type Msg = ();
            fn handle(&mut self, _: ()) {}
            fn stopped(&mut self) {
                panic!("stopped failure");
            }
        }
        assert!(spawn_bounded(StopsBadly, 1).join().is_err());
    }

    #[test]
    fn failed_panic_diagnostic_is_an_error_not_a_second_panic() {
        struct Broken;
        impl std::io::Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let payload: Box<dyn std::any::Any + Send> = Box::new("original failure");
        assert_eq!(
            write_panic(&mut Broken, &payload).unwrap_err().kind(),
            std::io::ErrorKind::BrokenPipe
        );
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"original failure"));
    }
    #[test]
    fn rejected_thread_creation_drops_unstarted_actor_and_returns_error() {
        struct NeverStarted {
            handled: Arc<AtomicUsize>,
            dropped: Arc<AtomicUsize>,
        }
        impl Actor for NeverStarted {
            type Msg = ();
            fn handle(&mut self, _: ()) {
                self.handled.fetch_add(1, Ordering::SeqCst);
            }
        }
        impl Drop for NeverStarted {
            fn drop(&mut self) {
                self.dropped.fetch_add(1, Ordering::SeqCst);
            }
        }
        let handled = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let result = try_spawn_using(
            NeverStarted { handled: handled.clone(), dropped: dropped.clone() },
            Cancellation::default(),
            |work| {
                drop(work); // The failed native launcher never ran the closure.
                Err(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    "injected thread creation failure",
                ))
            },
        );
        assert!(matches!(result, Err(error) if error.kind() == std::io::ErrorKind::WouldBlock));
        assert_eq!(handled.load(Ordering::SeqCst), 0);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
}
