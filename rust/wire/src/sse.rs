//! Server-Sent Events frame splitter.
//!
//! Accumulates streamed bytes and yields complete events (delimited by
//! `\n\n`). Does not parse the event body — `data:`/`event:`/`id:` lines
//! are the caller's problem. Uses `scan_for_pair` on every `push`, so the
//! SIMD path is hit for every chunk.

use crate::scan::scan_for_pair;

pub struct SseFramer {
    buf: Vec<u8>,
}

impl SseFramer {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
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
        f.push(b"data: hello\n\n");
        assert_eq!(drain(&mut f), vec![b"data: hello".to_vec()]);
        assert_eq!(f.remaining(), b"");
    }

    #[test]
    fn two_frames_one_chunk() {
        let mut f = SseFramer::new();
        f.push(b"event: a\ndata: 1\n\ndata: 2\n\n");
        assert_eq!(
            drain(&mut f),
            vec![b"event: a\ndata: 1".to_vec(), b"data: 2".to_vec()]
        );
    }

    #[test]
    fn frame_split_across_chunks() {
        let mut f = SseFramer::new();
        f.push(b"data: hel");
        assert!(f.pop_frame().is_none());
        f.push(b"lo\n");
        assert!(f.pop_frame().is_none());
        f.push(b"\nleftover");
        assert_eq!(f.pop_frame().unwrap(), b"data: hello");
        assert_eq!(f.remaining(), b"leftover");
    }

    #[test]
    fn empty_frame() {
        let mut f = SseFramer::new();
        f.push(b"\n\n");
        assert_eq!(f.pop_frame().unwrap(), b"");
    }

    #[test]
    fn stress_many_frames() {
        let mut f = SseFramer::new();
        for i in 0..1000 {
            f.push(format!("data: {i}\n\n").as_bytes());
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
        f.push(&buf);
        let frame = f.pop_frame().unwrap();
        assert_eq!(frame.len(), 200);
        assert_eq!(f.remaining(), b"tail");
    }
}
