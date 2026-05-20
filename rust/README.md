# rust/

Experimental Rust core for a minimal agent harness. Distils ideas from
Sandcastle (sandbox provider × branch strategy, two-phase prompt, completion
signal, structured output) and Flue (instance/harness/session hierarchy,
virtual sandbox default, role precedence, comptime-shaped result schemas)
into a build with a small dependency footprint.

## Status

M0 + M1. Actor primitive live. SIMD byte/pair scanners live.

## Layout

- `actor/` — OS-thread actor primitive. `std` only, zero dependencies.
- `wire/` — `scan_for_byte` / `scan_for_pair` with NEON (aarch64), AVX2
  (x86_64, runtime-detected), scalar fallback. Zero dependencies.

## Dependencies & supply chain

Both crates have zero external dependencies. When we add the first one,
we run `cargo vendor`, commit `vendor/`, and add `.cargo/config.toml`
redirecting crates.io to the vendored sources. CI builds with
`--locked --frozen --offline` after that. No registry pulls at build time.

We don't pre-build the vendor pipeline until there's something to vendor.

## Non-goals (for now)

- No async/await runtime.
- No supervision tree.
- No location-transparent addressing.
- No work-stealing scheduler.
- No benchmark harness — correctness first, benches when we have a
  real consumer that lets us measure end-to-end impact.

Each lands when a workload forces it, not before.

## Test

```
cd actor && cargo test
cd wire  && cargo test
cd wire  && cargo check --target aarch64-unknown-linux-gnu   # cross-check NEON
```
