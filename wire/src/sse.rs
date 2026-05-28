//! Server-Sent Events frame splitter.
//!
//! Accumulates streamed bytes and yields complete events (delimited by
//! `\n\n`). Does not parse the event body — `data:`/`event:`/`id:` lines
//! are the caller's problem. Uses `scan_for_pair` on every `push`, so the
//! SIMD path is hit for every chunk.
//!
//! Buffered bytes are bounded by a configurable cap (`DEFAULT_MAX` =
//! 4 MiB, matching `cluster::MAX_PAYLOAD`). A peer that streams beyond
//! the cap without emitting a frame boundary trips `FramerError::Overflow`
//! instead of growing the internal `Vec<u8>` without bound. Once a
//! framer overflows it is poisoned: subsequent `push` calls return the
//! same `Overflow` error and the buffer is cleared to release memory.

use crate::framer::{DEFAULT_MAX, FramerError};
use crate::scan::scan_for_pair;

pub struct SseFramer {
    buf: Vec<u8>,
    max: usize,
    poisoned: bool,
}

impl SseFramer {
    /// Construct a framer with the default 4 MiB cap.
    pub fn new() -> Self {
        Self::with_max(DEFAULT_MAX)
    }

    /// Construct a framer with a custom byte cap. Choose this if the
    /// 4 MiB default doesn't match the upstream's expected frame size.
    pub fn with_max(max: usize) -> Self {
        Self { buf: Vec::new(), max, poisoned: false }
    }

    /// Append `chunk` to the internal buffer. Returns `Overflow` (and
    /// poisons the framer, dropping any buffered bytes) when the append
    /// would exceed `max`. After the first overflow every subsequent
    /// `push` returns the same error without doing any work — the buffer
    /// has already been cleared and is no longer a useful container.
    pub fn push(&mut self, chunk: &[u8]) -> Result<(), FramerError> {
        if self.poisoned {
            return Err(FramerError::Overflow { max: self.max, attempted: chunk.len() });
        }
        if self.buf.len().saturating_add(chunk.len()) > self.max {
            self.poisoned = true;
            self.buf.clear();
            self.buf.shrink_to_fit();
            return Err(FramerError::Overflow { max: self.max, attempted: chunk.len() });
        }
        self.buf.extend_from_slice(chunk);
        Ok(())
    }

    /// Pop the next complete frame, if one is buffered. The returned frame
    /// does not include the `\n\n` terminator.
    pub fn pop_frame(&mut self) -> Option<Vec<u8>> {
        let idx = scan_for_pair(&self.buf, b'\n', b'\n')?;
        let frame: Vec<u8> = self.buf.drain(..idx).collect();
        self.buf.drain(..2);
        Some(frame)
    }

    /// Bytes still in the buffer that don't yet form a complete frame.
    /// Useful for inspection after the source closes.
    pub fn remaining(&self) -> &[u8] {
        &self.buf
    }
}

impl Default for SseFramer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(framer: &mut SseFramer) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Some(f) = framer.pop_frame() {
            out.push(f);
        }
        out
    }

    #[test]
    fn empty_buffer_yields_nothing() {
        let mut f = SseFramer::new();
        assert!(f.pop_frame().is_none());
    }

    #[test]
    fn one_complete_frame() {
        let mut f = SseFramer::new();
        f.push(b"data: hello\n\n").unwrap();
        assert_eq!(drain(&mut f), vec![b"data: hello".to_vec()]);
        assert_eq!(f.remaining(), b"");
    }

    #[test]
    fn two_frames_one_chunk() {
        let mut f = SseFramer::new();
        f.push(b"event: a\ndata: 1\n\ndata: 2\n\n").unwrap();
        assert_eq!(drain(&mut f), vec![b"event: a\ndata: 1".to_vec(), b"data: 2".to_vec()]);
    }

    #[test]
    fn frame_split_across_chunks() {
        let mut f = SseFramer::new();
        f.push(b"data: hel").unwrap();
        assert!(f.pop_frame().is_none());
        f.push(b"lo\n").unwrap();
        assert!(f.pop_frame().is_none());
        f.push(b"\nleftover").unwrap();
        assert_eq!(f.pop_frame().unwrap(), b"data: hello");
        assert_eq!(f.remaining(), b"leftover");
    }

    #[test]
    fn empty_frame() {
        let mut f = SseFramer::new();
        f.push(b"\n\n").unwrap();
        assert_eq!(f.pop_frame().unwrap(), b"");
    }

    #[test]
    fn stress_many_frames() {
        let mut f = SseFramer::new();
        for i in 0..1000 {
            f.push(format!("data: {i}\n\n").as_bytes()).unwrap();
        }
        let frames = drain(&mut f);
        assert_eq!(frames.len(), 1000);
        assert_eq!(frames[0], b"data: 0");
        assert_eq!(frames[999], b"data: 999");
    }

    #[test]
    fn long_chunk_forces_simd_path() {
        // A 200-byte preamble of zeros ensures the AVX2/NEON loop bodies run.
        let mut buf = vec![b' '; 200];
        buf.extend_from_slice(b"\n\ntail");
        let mut f = SseFramer::new();
        f.push(&buf).unwrap();
        let frame = f.pop_frame().unwrap();
        assert_eq!(frame.len(), 200);
        assert_eq!(f.remaining(), b"tail");
    }

    #[test]
    fn sse_framer_rejects_oversize_frame() {
        // 5 MiB pushed in one chunk with no frame boundary; default cap is 4 MiB.
        let mut f = SseFramer::new();
        let chunk = vec![b'a'; 5 * 1024 * 1024];
        let err = f.push(&chunk).unwrap_err();
        match err {
            FramerError::Overflow { max, attempted } => {
                assert_eq!(max, DEFAULT_MAX);
                assert_eq!(attempted, 5 * 1024 * 1024);
            }
        }
        // Buffer is cleared after overflow.
        assert_eq!(f.remaining(), b"");
    }

    #[test]
    fn sse_framer_under_cap_works() {
        // 3 MiB fits under the 4 MiB default.
        let mut f = SseFramer::new();
        let mut chunk = vec![b'a'; 3 * 1024 * 1024];
        chunk.extend_from_slice(b"\n\n");
        f.push(&chunk).unwrap();
        let frame = f.pop_frame().unwrap();
        assert_eq!(frame.len(), 3 * 1024 * 1024);
    }

    #[test]
    fn framer_state_after_overflow_stays_errored() {
        let mut f = SseFramer::with_max(8);
        // 9 bytes > cap of 8.
        let err = f.push(b"123456789").unwrap_err();
        assert!(matches!(err, FramerError::Overflow { .. }));
        // Even a tiny follow-up push is rejected — framer is poisoned.
        let err2 = f.push(b"x").unwrap_err();
        assert!(matches!(err2, FramerError::Overflow { .. }));
        // pop_frame on a poisoned (cleared) buffer returns None.
        assert!(f.pop_frame().is_none());
    }

    #[test]
    fn with_max_respects_custom_cap() {
        let mut f = SseFramer::with_max(16);
        f.push(b"abcdefgh").unwrap();
        // 8 + 9 > 16 — should overflow.
        assert!(f.push(b"123456789").is_err());
    }

    #[test]
    fn overflow_only_at_strict_greater_than_cap() {
        // Pushing exactly `max` bytes is allowed; one more is not.
        let mut f = SseFramer::with_max(4);
        f.push(b"abcd").unwrap();
        assert_eq!(f.remaining(), b"abcd");
        assert!(f.push(b"e").is_err());
    }
}
