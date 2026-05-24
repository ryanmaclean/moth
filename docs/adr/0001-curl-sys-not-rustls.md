# 0001: libcurl (via curl-sys) over rustls + hyper

Status: Accepted (initial workspace design)
Date: 2026-05

## Context

The Rust agent harness needs HTTPS streaming clients for Anthropic + OpenAI
+ MCP HTTP transport + Gitea + GitHub forge calls. Two viable stacks:

1. `rustls` + `hyper` (or `ureq`) — pure-Rust, modern.
2. `curl-sys` — FFI to libcurl, vendored statically with `openssl-src`.

The project's stated threat model is **supply-chain isolation** (motivated
by the shai-hulud worm hitting npm). We `cargo vendor` the entire
dependency tree and commit `vendor/` so builds are reproducible
offline.

## Decision

Use `curl-sys` with `static-curl` + `static-ssl` features. Direct deps:
`curl-sys` (one crate). Transitive: openssl-src + libz-sys + cc/pkg-config.
12 crates vendored, ~83 MB on disk (mostly OpenSSL + libcurl C source).

The rustls + hyper alternative is a much larger Rust dep tree (~50+ crates
for hyper + tokio + rustls + ring + their proc-macro siblings) — every
additional crate is a separate supply-chain risk and a separate
vendor-update cadence. libcurl + OpenSSL are old, audited, well-known,
and their CVE pipelines are well-trodden. The CVE risk is real but
familiar, and `cargo vendor` pins us to an exact version sha until we
explicitly refresh.

## Consequences

- **Build complexity**: one C toolchain required (already universal on
  target platforms). cross-compile to `aarch64-unknown-linux-gnu` works
  out of the box with the workspace's `aarch64-linux-gnu-gcc` setup.
- **Streaming**: libcurl's WRITEFUNCTION callback model works cleanly
  through a synchronous `std::sync::mpsc::sync_channel`. No async runtime
  needed.
- **Cancellation**: harder than rustls (no native `select!` interplay);
  solved via `CURLOPT_XFERINFOFUNCTION` polling an `AtomicBool`.
- **Connection reuse / keep-alive**: not exploited yet — each request
  opens a fresh easy handle. Adding curl-share / multi-handle pooling is
  a separate optimisation when latency budget demands it.
- **Repo size**: vendor/ is ~83 MB. Acceptable; git history is fine.

## Alternatives considered

- **rustls + hyper**: pure-Rust, but supply-chain surface area conflicts
  with the project's stated goal.
- **ureq**: lighter, but still pure-Rust and pulls a moderately large
  dep tree; doesn't solve the streaming SSE consumer ergonomically.
- **OpenSSL directly via openssl-sys**: would require writing our own
  HTTP/1.1 + framing code. Bigger code surface than libcurl.
