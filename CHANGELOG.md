# Changelog

All notable changes to the workspace. Format: Keep-A-Changelog,
semver pre-1.0 (every release is `0.x.y`; minor bumps may break API).

## Unreleased

### Added
- `agent --version` / `-V` prints `agent <cargo-pkg-version>`.
- `docs/adr/` directory with the first architecture decision records.
- ADR-0002 captures the subtraction-first scope policy.

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
