//! Micro-bench helpers.
//!
//! Use `std::time::Instant`, batch by 1000 ops per sample, 100 samples,
//! report the median. Warmup is 10 batches. Inputs and outputs go through
//! `std::hint::black_box` to keep LLVM honest.

use std::hint::black_box;
use std::time::{Duration, Instant};

const WARMUP_BATCHES: usize = 10;
const SAMPLES: usize = 100;
const BATCH: usize = 1000;

/// Run `f` repeatedly and print the median per-op latency to stderr.
/// Returns the median per-op `Duration` so callers may assert on it.
pub fn bench<F: FnMut()>(name: &str, mut f: F) -> Duration {
    for _ in 0..WARMUP_BATCHES {
        f();
    }
    let mut samples: Vec<Duration> = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t = Instant::now();
        for _ in 0..BATCH {
            f();
        }
        samples.push(t.elapsed());
    }
    samples.sort();
    let median_batch = samples[SAMPLES / 2];
    let ns_per_op = median_batch.as_nanos() / BATCH as u128;
    eprintln!("{name}: {ns_per_op} ns/op (median over {SAMPLES} batches of {BATCH})");
    Duration::from_nanos(ns_per_op as u64)
}

/// Same as `bench` but `f` returns the number of bytes processed per op.
/// Prints ns/op and MB/s.
pub fn bench_throughput<F: FnMut() -> usize>(name: &str, mut f: F) -> Duration {
    for _ in 0..WARMUP_BATCHES {
        black_box(f());
    }
    let mut samples: Vec<(Duration, usize)> = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let mut bytes = 0usize;
        let t = Instant::now();
        for _ in 0..BATCH {
            bytes = bytes.wrapping_add(black_box(f()));
        }
        let elapsed = t.elapsed();
        samples.push((elapsed, bytes));
    }
    samples.sort_by_key(|(d, _)| *d);
    let (median_batch, bytes_in_batch) = samples[SAMPLES / 2];
    let ns_per_op = median_batch.as_nanos() / BATCH as u128;
    // bytes_in_batch is the sum over BATCH ops; per-op = bytes_in_batch / BATCH.
    let bytes_per_op = bytes_in_batch / BATCH;
    let mb_per_s = if median_batch.as_nanos() > 0 {
        (bytes_in_batch as f64) / (median_batch.as_nanos() as f64) * 1_000.0
    } else {
        0.0
    };
    eprintln!(
        "{name}: {ns_per_op} ns/op, {bytes_per_op} B/op, {mb_per_s:.1} MB/s \
         (median over {SAMPLES} batches of {BATCH})"
    );
    Duration::from_nanos(ns_per_op as u64)
}

/// Deterministic pseudo-random bytes; LCG, good enough for bench inputs.
pub fn pseudo_random(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed | 1;
    let mut buf = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        buf.push((state >> 24) as u8);
    }
    buf
}
