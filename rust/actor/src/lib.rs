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

pub fn spawn<A: Actor>(mut actor: A) -> Spawned<A::Msg> {
    let (tx, rx): (Sender<A::Msg>, Receiver<A::Msg>) = mpsc::channel();
    let handle = thread::spawn(move || {
        while let Ok(msg) = rx.recv() {
            actor.handle(msg);
        }
        actor.stopped();
    });
    Spawned { addr: ActorRef { tx }, handle }
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
}
