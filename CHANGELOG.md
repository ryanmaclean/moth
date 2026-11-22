# Changelog

All notable changes to the workspace. Format: Keep-A-Changelog,
semver pre-1.0 (every release is `0.x.y`; minor bumps may break API).

## Unreleased

### Added
- `agent run --run-id ID` (env `AGENT_RUN_ID`, then `BOP_RUN_ID`): run identity is supplied by the caller (BOP owns it) instead of always minted as `run-<ms>`; the minted id remains the fallback. An invalid explicit id exits 2 rather than being silently replaced.
- runlog records carry a leading `"seq"`: per-run, contiguous from 0, assigned under the write lock (file order == seq order), and resumed from the last committed record when a run file is re-opened. `ts_ms` is now documented as metadata, not an ordering key.
- `runlog::validate_run_id` / `is_valid_run_id`, `RunLog::{sync, next_seq, write_record_seq}`, `TerminalSummary::last_seq`, `RunLogError::InvalidRunId`.
- `agent --version` / `-V` prints `agent <cargo-pkg-version>`.
- `docs/adr/` directory with the first architecture decision records.
- ADR-0002 captures the subtraction-first scope policy.
- ADR-0003 keeps the `metrics` crate (superseding ADR-0002's deferred-cut bullet) and wires it end-to-end.
- `--metrics <HOST:PORT>` flag on `agent run` / `agent serve` (overrides `DOGSTATSD_ADDR`; flag > env > disabled).
- FreeBSD support: builds + runs natively (the vendored OpenSSL/libcurl C sources need `gmake` + `perl5`); `agent doctor` reports FreeBSD/NetBSD/OpenBSD target triples; a `freebsd` CI job builds the binary and tests the platform-sensitive crates inside a native FreeBSD VM (the C deps don't cross-compile from Linux).
- `agent serve --token <TOKEN>` (also `AGENT_API_TOKEN`) enables bearer-token auth on the agent endpoints.
- `dependency-scan` CI workflow: `osv-scanner` on `Cargo.lock` (fails the build on Rust-crate advisories) plus a best-effort OSV query of the bundled C-library versions (libcurl / OpenSSL / zlib) surfaced in the job summary; runs on PR, push to `main`, a weekly cron, and manual dispatch. (audit H5)

### Changed
- runlog record field order is now `seq, ts_ms, run_id, [request_id], kind, data` (consumers matching on a `{"ts_ms":` line prefix must update).
- runlog `drain` `fdatasync`s after each terminal record (`done`/`cancelled`/`error`), so a returned `final_event` implies the terminal record is durable.
- Metrics now emitted across `agent run`, `agent serve`, and subagents (subagents inherit the parent's emitter); opt-in via `--metrics`/`DOGSTATSD_ADDR`, no-op when unset.
- **`agent serve` default bind moved from `0.0.0.0:3583` to `127.0.0.1:3583`**, and it now refuses to start on a non-loopback address unless a token is configured. Behavior change: off-host serving now requires an explicit `--addr` *and* a `--token`/`AGENT_API_TOKEN`. (audit H2)

### Security
- JSON parser (`anthropic`, `openai`): nesting depth is now capped at 128 (`MAX_DEPTH`). A single untrusted SSE/NDJSON frame of deeply nested `[`/`{` within the 4 MiB frame cap could previously recurse until the stack overflowed and the process aborted — an uncatchable DoS reachable from the model API, any OpenAI-compatible endpoint, or an MCP peer. Over-deep input now returns a clean `InvalidResponse` error. Regression tests cover both crates. (audit H1)
- `agent serve` now gates `POST /agents/*` behind an optional bearer token (`--token`/`AGENT_API_TOKEN`) with a constant-time `Authorization: Bearer` check (401 on missing/mismatch); `/healthz` and `/readyz` stay open. When no token is set the endpoints remain open (local-dev default) but `serve` refuses to start on a non-loopback bind. Previously the endpoint — which can execute shell commands — was unauthenticated on the old `0.0.0.0` default. (audit H2)
- HTTP server rejects a duplicate `Content-Length`, and non-numeric / sign-prefixed values, with `400` instead of last-wins overwrite (a request-smuggling primitive when fronted by another proxy). Note: `usize::parse` does *not* reject a leading `+`, so an explicit digits-only guard was added. (audit L1)
- `openai` HTTP client now runs `curl_global_init` behind a `Once` before `curl_easy_init`, matching the `anthropic`/`mcp` clients — removes a first-request/startup data race in libcurl/OpenSSL one-time init. (audit M1)
- `git` ref/path operands are separated from options with `--`, and externally-supplied ref names are validated (rejecting leading `-`, `..`, `@{`, whitespace/control chars, `~^:?*[\`, `/`-edges, `.lock`) before any git process runs — closes an argument-injection vector where a ref like `--upload-pack=…` could be parsed as a git flag. (audit L4)

### Fixed
- `agent serve --runlog`: a client-supplied `X-Request-ID` containing `/` or `..` was used verbatim as the runlog filename, letting a request write `<id>.jsonl` outside the runlog dir. Unsafe ids now get a minted `req-<ms>-<n>` filename; the original id is still recorded in `request_id`.
- runlog re-open after a crash mid-write newline-terminates the torn fragment so later records are not glued onto it.

### Removed (round 7 — staff-eng cut pass)
- `cluster` crate: distributed actor refs with no callers.
- `gitea`, `github`: forge clients with no callers; forge ops belong behind MCP.
- `jj`: second branch-strategy backend alongside `git/`.
- `mcp_server`: merged into `mcp::server`; the two halves share framing.

### Production hardening (round 2)
- `actor::spawn_bounded(actor, capacity)` + `SyncActorRef::try_send`.
- `catch_unwind` around every actor handler call.
- Server `/healthz`, `/readyz`, SIGTERM graceful drain with 30s deadline.
- DogStatsD metrics emission from `harness::Session` and `harness::execute_tool`.
- Per-host circuit breaker (`wire::retry`) wired into anthropic + openai streaming.
- Retry-with-backoff + `CURLOPT_XFERINFOFUNCTION` cancellation in both model HTTP clients.
- Bounded `SyncSender<StreamEvent>` end-to-end (CLI / ChatHandler / runlog tee).
- `audit::LiveScanner` with JSON pattern files + atomic swap.
- `persist::FileStore` append-only log + snapshot (was full-file rewrite per turn).
- `ChatMessage::content` Arc-wrapped for cheap clones across turns.
- `X-Request-ID` propagation: server → handler → runlog records.

### Provider + workflow
- `anthropic` + `openai` streaming clients.
- `mcp` (client) + `mcp_server` (stdio JSON-RPC).
- `gitea` + `github` forge clients.
- `cluster::RemoteActorRef<M: Codec>` over TCP.
- `git` + `jj` branch strategies (`HeadStrategy`, `MergeToHeadStrategy`,
  `BranchStrategy { name }`).
- `subagent::spawn_task` Flue-style child sessions; LLM-callable `task` tool.
- `compact::Compactor` with `HarnessState::with_compactor` hook.
- `runlog` JSONL audit trail.
- `tmpl` skill + role markdown loader; `{{KEY}}` substitution.
- `fstools` (read/write/edit, symlink + hard-link safe).
- `vshell` in-proc POSIX shell subset.
- `audit` Aho-Corasick shai-hulud-class scanner.
- `wire` SIMD scanners + SSE framer + NDJSON splitter + tag finder.
- `benches` microbenchmark suite (cargo test -p benches --release -- --nocapture).

## See also

- `docs/adr/` — rationale for the load-bearing architecture decisions.
- `README.md` — current crate matrix.
