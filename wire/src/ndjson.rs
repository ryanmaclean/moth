//! Newline-delimited line splitter.
//!
//! Mirror of `SseFramer` but splits on `\n` instead of `\n\n`. Suitable
//! for NDJSON streams and any line-oriented protocol — MCP stdio, log
//! tailing, jsonl LLM responses. Shares the bounded-buffer semantics
//! (see `framer::DEFAULT_MAX` / `FramerError::Overflow`): a peer that
//! never emits a `\n` can't grow the buffer past the cap.

use crate::framer::{DEFAULT_MAX, FramerError};
use crate::scan::scan_for_byte;

pub struct NdjsonSplitter {
    buf: Vec<u8>,
    max: usize,
    poisoned: bool,
}

impl NdjsonSplitter {
    /// Construct a splitter with the default 4 MiB cap.
    pub fn new() -> Self {
        Self::with_max(DEFAULT_MAX)
    }

    /// Construct a splitter with a custom byte cap. Useful when the
    /// upstream's max line size differs from the 4 MiB default.
    pub fn with_max(max: usize) -> Self {
        Self { buf: Vec::new(), max, poisoned: false }
    }

    /// Append `chunk` to the internal buffer. Returns `Overflow` (and
    /// poisons the splitter, dropping any buffered bytes) when the append
    /// would exceed `max`. Once poisoned every subsequent `push` returns
    /// the same error without doing any work.
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

    /// Pop the next complete line, excluding the trailing `\n`.
    pub fn pop_line(&mut self) -> Option<Vec<u8>> {
        let idx = scan_for_byte(&self.buf, b'\n')?;
        let line: Vec<u8> = self.buf.drain(..idx).collect();
        self.buf.drain(..1);
        Some(line)
    }

    pub fn remaining(&self) -> &[u8] {
        &self.buf
    }
}

impl Default for NdjsonSplitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(s: &mut NdjsonSplitter) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Some(l) = s.pop_line() {
            out.push(l);
        }
        out
    }

    #[test]
    fn empty_buffer() {
        let mut s = NdjsonSplitter::new();
        assert!(s.pop_line().is_none());
    }

    #[test]
    fn one_line() {
        let mut s = NdjsonSplitter::new();
        s.push(b"{\"a\":1}\n").unwrap();
        assert_eq!(drain(&mut s), vec![b"{\"a\":1}".to_vec()]);
    }

    #[test]
    fn three_lines_one_chunk() {
        let mut s = NdjsonSplitter::new();
        s.push(b"a\nbb\nccc\n").unwrap();
        assert_eq!(drain(&mut s), vec![b"a".to_vec(), b"bb".to_vec(), b"ccc".to_vec()]);
    }

    #[test]
    fn line_split_across_chunks() {
        let mut s = NdjsonSplitter::new();
        s.push(b"abc").unwrap();
        assert!(s.pop_line().is_none());
        s.push(b"def\nghi").unwrap();
        assert_eq!(s.pop_line().unwrap(), b"abcdef");
        assert_eq!(s.remaining(), b"ghi");
    }

    #[test]
    fn empty_lines() {
        let mut s = NdjsonSplitter::new();
        s.push(b"\n\n\n").unwrap();
        assert_eq!(drain(&mut s), vec![vec![], vec![], vec![]]);
    }

    #[test]
    fn no_trailing_newline_leaves_remainder() {
        let mut s = NdjsonSplitter::new();
        s.push(b"complete\npartial").unwrap();
        assert_eq!(s.pop_line().unwrap(), b"complete");
        assert!(s.pop_line().is_none());
        assert_eq!(s.remaining(), b"partial");
    }

    #[test]
    fn ndjson_splitter_rejects_oversize_line() {
        // 5 MiB chunk with no newline; default cap is 4 MiB.
        let mut s = NdjsonSplitter::new();
        let chunk = vec![b'x'; 5 * 1024 * 1024];
        let err = s.push(&chunk).unwrap_err();
        match err {
            FramerError::Overflow { max, attempted } => {
                assert_eq!(max, DEFAULT_MAX);
                assert_eq!(attempted, 5 * 1024 * 1024);
            }
        }
        assert_eq!(s.remaining(), b"");
    }

    #[test]
    fn ndjson_splitter_under_cap_works() {
        let mut s = NdjsonSplitter::new();
        let mut chunk = vec![b'x'; 3 * 1024 * 1024];
        chunk.push(b'\n');
        s.push(&chunk).unwrap();
        let line = s.pop_line().unwrap();
        assert_eq!(line.len(), 3 * 1024 * 1024);
    }

    #[test]
    fn ndjson_state_after_overflow_stays_errored() {
        let mut s = NdjsonSplitter::with_max(8);
        let err = s.push(b"123456789").unwrap_err();
        assert!(matches!(err, FramerError::Overflow { .. }));
        let err2 = s.push(b"x").unwrap_err();
        assert!(matches!(err2, FramerError::Overflow { .. }));
        assert!(s.pop_line().is_none());
    }
}
