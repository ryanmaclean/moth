# rust/

Experimental Rust core for a minimal agent harness. Distils ideas from
Sandcastle (sandbox provider × branch strategy, two-phase prompt, completion
signal, structured output) and Flue (instance/harness/session hierarchy,
virtual sandbox default, role precedence) into a build with a small
dependency footprint.

## Status

M0 + M1 + M2. Five crates live; 126 inline tests across the workspace.

## Layout

- `actor/` — OS-thread actor primitive. `Send` / `ask`, `std` only. Zero deps.
- `wire/` — SIMD byte/pair scanners + SSE framer + NDJSON splitter +
  tag extractor. NEON (aarch64) + AVX2 (x86_64, runtime-detected) +
  scalar. Zero deps.
- `vshell/` — in-process POSIX shell subset. Quoting, `$VAR`/`${VAR}`,
  `$(cmd)`, pipelines, redirects, sequencing, env prefix, built-ins.
  std only. Out-of-scope syntax errors clearly rather than silently
  misbehaving. Zero deps.
- `anthropic/` — streaming Messages API client. libcurl + OpenSSL via
  `curl-sys` with static features. Hand-rolled JSON walker (no serde).
  Consumes `wire::SseFramer` for response framing. One direct dep
  (`curl-sys`); transitive set fully vendored.
- `harness/` — Instance / Session actors + iteration loop. Defines
  `Model` + `Sandbox` traits; mock impls inline. Completion-signal
  detection via `wire::find_tag(b"promise")`, structured-output
  extraction via `wire::find_tag(tag)`. Built on `actor` + `wire`.

## Dependencies & supply chain

`anthropic` introduced the first real dependency (`curl-sys 0.4`).
`cargo vendor` was run at the workspace root; the resulting `vendor/`
tree (12 crates including OpenSSL + libcurl source for static builds)
is committed. `.cargo/config.toml` redirects crates.io to
`vendored-sources`. Builds with `--locked --frozen --offline` succeed.

The vendor dir is ~83 MB, almost entirely OpenSSL + libcurl C source.
If git size becomes a problem we'll switch to system libcurl/openssl
(smaller -sys crates, but inherits whatever the host ships).

## Non-goals (for now)

- No async/await runtime.
- No supervision tree.
- No location-transparent addressing.
- No work-stealing scheduler.
- No benchmark harness.

Each lands when a workload forces it.

## Test

```
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo check --workspace --target aarch64-unknown-linux-gnu
```

All three pass on the committed tree.
