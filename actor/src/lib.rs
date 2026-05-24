//! Minimal actor primitive.
//!
//! One OS thread per actor, `std::sync::mpsc` mailboxes, zero dependencies.
//! Actors stop when every `ActorRef` is dropped (the channel closes).
//!
//! Not a scheduler. Not a supervision tree. Not a runtime. It's the smallest
//! thing that lets one piece of state own its mailbox and process messages
//! in order. We add more only when a real workload forces it.

use std::sync::mpsc::{self, Receiver, SendError, Sender, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

pub trait Actor: Send + 'static {
    type Msg: Send + 'static;

    fn handle(&mut self, msg: Self::Msg);

    /// Called once after the mailbox closes, before the thread exits.
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
}

impl<M: Send + 'static> Spawned<M> {
    /// Drop the internal `ActorRef` and block until the actor's thread exits.
    ///
    /// If the caller cloned `addr` elsewhere, those clones must drop before
    /// this returns — the actor only stops when every sender is gone.
    pub fn join(self) -> thread::Result<()> {
        let Spawned { addr, handle } = self;
        drop(addr);
        handle.join()
    }
}

pub fn spawn<A: Actor>(actor: A) -> Spawned<A::Msg> {
    let (tx, rx): (Sender<A::Msg>, Receiver<A::Msg>) = mpsc::channel();
    let handle = thread::spawn(move || run_loop(actor, rx));
    Spawned { addr: ActorRef { tx }, handle }
}

/// Bounded variant: `send` blocks once `capacity` messages are queued.
/// Use this in any path where a fast producer can outrun the actor —
/// streaming HTTP, fan-in from many connections, etc. Prevents silent
/// memory growth under backpressure.
pub fn spawn_bounded<A: Actor>(actor: A, capacity: usize) -> SpawnedSync<A::Msg> {
    let (tx, rx) = sync_channel::<A::Msg>(capacity);
    let handle = thread::spawn(move || run_loop_sync(actor, rx));
    SpawnedSync { addr: SyncActorRef { tx }, handle }
}

/// Shared dispatch loop. Wraps every `handle` call in `catch_unwind` so a
/// tool panic doesn't kill the mailbox silently — we log the payload to
/// stderr and break out of the loop so subsequent sends fail-fast
/// (channel half closes when the receiver thread exits its scope).
fn run_loop<A: Actor>(mut actor: A, rx: Receiver<A::Msg>) {
    while let Ok(msg) = rx.recv() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            actor.handle(msg);
        }));
        if let Err(payload) = result {
            log_panic(&payload);
            break;
        }
    }
    actor.stopped();
}

fn run_loop_sync<A: Actor>(mut actor: A, rx: Receiver<A::Msg>) {
    while let Ok(msg) = rx.recv() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            actor.handle(msg);
        }));
        if let Err(payload) = result {
            log_panic(&payload);
            break;
        }
    }
    actor.stopped();
}

fn log_panic(payload: &Box<dyn std::any::Any + Send>) {
    let msg = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic>");
    eprintln!("actor: handler panicked: {msg}");
}

/// Bounded-mailbox counterpart of `Spawned`. The `SyncActorRef::send` call
/// applies backpressure: it blocks until either capacity frees up or the
/// receiver is gone.
#[must_use = "dropping SpawnedSync detaches the actor's thread"]
pub struct SpawnedSync<M: Send + 'static> {
    pub addr: SyncActorRef<M>,
    handle: JoinHandle<()>,
}

impl<M: Send + 'static> SpawnedSync<M> {
    pub fn join(self) -> thread::Result<()> {
        let SpawnedSync { addr, handle } = self;
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
        // This panic must not propagate; the actor thread exits cleanly.
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
}
