//! Wire scanner benches.
//!
//! Note on scenario 5: `wire::scalar::scan_for_byte` is `pub(crate)` —
//! not reachable from outside the wire crate. We bench only the public
//! API. On x86_64 this resolves to AVX2 at runtime; on aarch64 NEON
//! unconditionally. The scalar-vs-SIMD comparison would need wire to
//! `pub use scalar`.

use std::hint::black_box;

use wire::{find_tag, scan_for_byte, scan_for_pair};

use crate::bench_helper::{bench_throughput, pseudo_random};

/// 4 KiB buffer, byte not present anywhere. Forces a full scan.
fn worst_case_buffer(len: usize, seed: u32, needle: u8) -> Vec<u8> {
    let mut buf = pseudo_random(len, seed);
    // Scrub every occurrence of needle so the scan walks the whole buffer.
    for b in &mut buf {
        if *b == needle {
            *b = needle.wrapping_add(1);
        }
    }
    buf
}

#[test]
fn bench_wire_scan_byte_4k() {
    let buf = worst_case_buffer(4 * 1024, 0xDEAD_BEEF, b'\n');
    bench_throughput("wire::scan_for_byte 4 KiB (absent)", || {
        let _ = black_box(scan_for_byte(black_box(&buf), black_box(b'\n')));
        buf.len()
    });
}

#[test]
fn bench_wire_scan_byte_64k() {
    let buf = worst_case_buffer(64 * 1024, 0xCAFE_F00D, b'\n');
    bench_throughput("wire::scan_for_byte 64 KiB (absent)", || {
        let _ = black_box(scan_for_byte(black_box(&buf), black_box(b'\n')));
        buf.len()
    });
}

#[test]
fn bench_wire_scan_pair_4k() {
    // Strip out any naturally-occurring pair so the scan walks the buffer.
    let mut buf = pseudo_random(4 * 1024, 0xFEED_FACE);
    let (a, b) = (b'<', b'/');
    let mut i = 0;
    while i + 1 < buf.len() {
        if buf[i] == a && buf[i + 1] == b {
            buf[i + 1] = b.wrapping_add(1);
        }
        i += 1;
    }
    bench_throughput("wire::scan_for_pair 4 KiB (absent)", || {
        let _ = black_box(scan_for_pair(black_box(&buf), black_box(a), black_box(b)));
        buf.len()
    });
}

#[test]
fn bench_wire_find_tag_4k() {
    // 4 KiB buffer with `<output>...</output>` at the very end. find_tag
    // does a scan_for_byte('<') for each candidate; the worst case is the
    // tag living far from the start.
    let mut buf = pseudo_random(4 * 1024 - 32, 0xBADD_CAFE);
    // Scrub any '<' that might cause early false-hits.
    for b in &mut buf {
        if *b == b'<' {
            *b = b'_';
        }
    }
    buf.extend_from_slice(b"<output>hello</output>");
    bench_throughput("wire::find_tag 4 KiB (tag at end)", || {
        let body = black_box(find_tag(black_box(&buf), black_box(b"output")));
        // assert in debug; in release this is folded away by black_box.
        debug_assert_eq!(body, Some(&b"hello"[..]));
        buf.len()
    });
}

#[test]
fn bench_wire_scan_byte_4k_note_scalar() {
    // Scenario 5: the scalar path is `pub(crate)`. We document the limit
    // and run the public API again with a different seed so the result is
    // a useful second sample, not noise.
    let buf = worst_case_buffer(4 * 1024, 0x1234_5678, b'\n');
    eprintln!(
        "wire::scalar::scan_for_byte is pub(crate); cannot bench from outside the crate. \
         Public-API result follows for reference:"
    );
    bench_throughput("wire::scan_for_byte 4 KiB (absent, seed 2)", || {
        let _ = black_box(scan_for_byte(black_box(&buf), black_box(b'\n')));
        buf.len()
    });
}
