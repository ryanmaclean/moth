//! Newline-delimited line splitter.
//!
//! Mirror of `SseFramer` but splits on `\n` instead of `\n\n`. Suitable
//! for NDJSON streams and any line-oriented protocol — MCP stdio, log
//! tailing, jsonl LLM responses.

use crate::scan::scan_for_byte;

pub struct NdjsonSplitter {
    buf: Vec<u8>,
}

impl NdjsonSplitter {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
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
        s.push(b"{\"a\":1}\n");
        assert_eq!(drain(&mut s), vec![b"{\"a\":1}".to_vec()]);
    }

    #[test]
    fn three_lines_one_chunk() {
        let mut s = NdjsonSplitter::new();
        s.push(b"a\nbb\nccc\n");
        assert_eq!(
            drain(&mut s),
            vec![b"a".to_vec(), b"bb".to_vec(), b"ccc".to_vec()]
        );
    }

    #[test]
    fn line_split_across_chunks() {
        let mut s = NdjsonSplitter::new();
        s.push(b"abc");
        assert!(s.pop_line().is_none());
        s.push(b"def\nghi");
        assert_eq!(s.pop_line().unwrap(), b"abcdef");
        assert_eq!(s.remaining(), b"ghi");
    }

    #[test]
    fn empty_lines() {
        let mut s = NdjsonSplitter::new();
        s.push(b"\n\n\n");
        assert_eq!(drain(&mut s), vec![vec![], vec![], vec![]]);
    }

    #[test]
    fn no_trailing_newline_leaves_remainder() {
        let mut s = NdjsonSplitter::new();
        s.push(b"complete\npartial");
        assert_eq!(s.pop_line().unwrap(), b"complete");
        assert!(s.pop_line().is_none());
        assert_eq!(s.remaining(), b"partial");
    }
}
