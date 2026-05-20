# rust/

Experimental Rust core for a minimal agent harness. Distils ideas from
Sandcastle (sandbox provider × branch strategy, two-phase prompt, completion
signal, structured output) and Flue (instance/harness/session hierarchy,
virtual sandbox default, role precedence) into a build with a small
dependency footprint.

## Status

End-to-end runnable. 147 inline tests across the workspace; one direct
crate dependency (`curl-sys`), 12 transitive crates fully vendored.

## Layout

- `actor/` — OS-thread actor primitive. `Send` / `ask` / `stopped`. `std` only. Zero deps.
- `wire/` — SIMD byte/pair scanners + `SseFramer` + `NdjsonSplitter` +
  `find_tag`. NEON (aarch64) + AVX2 (x86_64, runtime-detected) + scalar.
  Zero deps.
- `vshell/` — in-process POSIX shell subset. Quoting, `$VAR`/`${VAR}`,
  `$(cmd)`, pipelines, redirects, sequencing, env prefix, built-ins.
  Out-of-scope syntax errors clearly. std only. Zero deps.
- `anthropic/` — streaming Messages API client. libcurl + OpenSSL via
  `curl-sys` (static features). Hand-rolled JSON walker and serializer;
  no serde. Consumes `wire::SseFramer` for response framing. Supports
  `tool_use` / `tool_result` content blocks.
- `audit/` — literal-pattern shell scanner. Defensive layer that blocks
  shai-hulud-class payloads (curl-pipe-to-shell, eval-of-curl,
  base64-pipe-to-shell) before exec. Built on `wire::scan_for_byte`.
- `harness/` — Instance + Session actors. `Model` + `Sandbox` traits with
  real adapters for `anthropic::Client` (as `AnthropicModel`) and
  `vshell::VShell`. `AuditedShell<S>` decorator runs `audit::Scanner`
  before every command. Session drives a tool-use iteration loop:
  streams the model, executes `bash` tool calls via the Instance,
  appends `tool_result` blocks, re-prompts, terminates at the completion
  signal / `end_turn` / a turn cap.
- `cli/` — `agent <prompt>` binary. Wires everything via `AuditedShell
  <VShell>`, reads `ANTHROPIC_API_KEY` from env.

## Dependencies & supply chain

`anthropic` introduced the only direct dependency: `curl-sys` with
`static-curl` + `static-ssl` features. `cargo vendor` was run at the
workspace root; the resulting `vendor/` tree (12 crates including
OpenSSL + libcurl source for static builds) is committed.
`.cargo/config.toml` redirects crates.io to `vendored-sources`. Builds
with `--locked --frozen --offline` succeed against the vendored tree.

The vendor dir is ~83 MB, almost entirely OpenSSL + libcurl C source.
If git size becomes a problem we'll switch to system libcurl/openssl
(smaller -sys crates, but inherits whatever the host ships).

## Non-goals (for now)

- No async/await runtime.
- No supervision tree.
- No location-transparent addressing.
- No work-stealing scheduler.
- No HTTP server.
- No MCP client.
- No benchmark harness.

## Run

```bash
# All tests + lints
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo check --workspace --target aarch64-unknown-linux-gnu

# One-shot agent (requires ANTHROPIC_API_KEY)
cargo run --bin agent -- "list the files in /tmp"
```
