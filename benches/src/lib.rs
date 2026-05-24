//! Microbenchmarks for the hot paths the rest of the workspace was built
//! around: `wire`'s SIMD byte/pair scanners, `audit`'s defensive scanner,
//! `anthropic::json`'s parser, and `vshell`'s parse-and-dispatch cost.
//!
//! No `criterion`, no `divan`. `std::time::Instant`, a warm-up loop, a
//! batch of 1000 ops per timed sample, 100 samples, report the median.
//! Runs as `cargo test -p benches --release -- --nocapture`.

#[cfg(test)]
mod bench_helper;

#[cfg(test)]
mod audit_bench;
#[cfg(test)]
mod history_bench;
#[cfg(test)]
mod json_bench;
#[cfg(test)]
mod shell_bench;
#[cfg(test)]
mod wire_bench;
