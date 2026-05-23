//! Wire parsing primitives.
//!
//! Byte/pair scanners with NEON (aarch64) + AVX2 (x86_64, runtime-detected) +
//! scalar fallback, plus the stream framers and tag extractor built on top.
//! Zero dependencies.

mod ndjson;
pub mod retry;
mod scan;
mod sse;
mod tag;

pub use ndjson::NdjsonSplitter;
pub use scan::{scan_for_byte, scan_for_pair};
pub use sse::SseFramer;
pub use tag::find_tag;
