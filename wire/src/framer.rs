//! Shared types for the stream framers (`SseFramer`, `NdjsonSplitter`).
//!
//! Both framers cap their buffered prefix at `DEFAULT_MAX` (4 MiB,
//! matching `cluster::MAX_PAYLOAD`) so a peer that streams bytes
//! without ever emitting a frame boundary can't grow the internal
//! `Vec<u8>` without bound. The cap and the resulting error type live
//! here so the two framers stay in lockstep.

/// Default ceiling on buffered bytes awaiting a frame boundary. Matches
/// `cluster::MAX_PAYLOAD` so cross-crate sizing stays consistent.
pub const DEFAULT_MAX: usize = 4 * 1024 * 1024;

/// Errors produced by `SseFramer::push` / `NdjsonSplitter::push`.
///
/// `Overflow` carries the cap and the size of the rejected chunk for
/// diagnostics. After the first overflow the framer is poisoned: every
/// subsequent `push` returns the same variant and the buffer is cleared
/// so the framer doesn't keep holding onto memory after the error.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FramerError {
    Overflow { max: usize, attempted: usize },
}

impl std::fmt::Display for FramerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FramerError::Overflow { max, attempted } => {
                write!(f, "framer overflow: attempted {attempted} bytes, cap is {max}")
            }
        }
    }
}

impl std::error::Error for FramerError {}
