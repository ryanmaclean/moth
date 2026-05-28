# 0002: Subtraction-first scope policy

Status: Accepted
Date: 2026-05

## Context

Round 6 grew the workspace to 24 crates. A staff-eng pass found that
several crates had no callers in `cli` or `harness` and existed only
as speculative infrastructure. We removed them. This ADR captures the
criteria so the same drift doesn't recur, and lists the candidates
deferred for now.

## Decision

A crate earns its place in the workspace only if at least one of these
is true:

1. **The CLI invokes it on a hot path.** `agent run` or `agent serve`
   must reach the code under reasonable use, not just `--feature-flag`
   gated paths.
2. **Another in-workspace crate that meets (1) depends on it.**
3. **It's a benchmark, integration test, or test-only fixture.**

If none of (1-3) hold, the crate goes. "Might be useful later" is not
a reason. Re-adding a crate is a `git revert` away; carrying dead code
costs review time, build time, vendored-dep churn, and audit surface
on every change.

## What was cut (round 7)

| Crate | LOC | Reason |
|---|---|---|
| `cluster` | ~700 | Distributed actor refs over TCP. Zero callers. Multi-node isn't on the roadmap. We had just added bearer-token auth, read-timeout, and reader reaping — to code no one ran. |
| `gitea` | ~600 | Forge client. Zero callers in cli/harness. Forge ops belong behind an MCP server, not as a first-class crate. |
| `github` | ~600 | Same as gitea. |
| `jj` | ~300 | Second branch-strategy backend alongside `git/`. Pick one. `git` retained. |
| `mcp_server` | — | Merged into `mcp::server`. Client and server share JSON-RPC + `wire` framing; one crate suffices. |

Net: 24 → 19 crates, ~2200 LOC removed, 656 → 592 tests (only the
removed crates' own tests went; nothing else regressed).

## Deferred (candidates for future passes)

These would shrink the tree further but each has a real tradeoff. Don't
cut without an explicit call.

- **`vshell` → `bash -c`.** ~2000 LOC of POSIX shell subset (lex, parse,
  exec). The argument for keeping it: predictable parsing, deterministic
  pipeline semantics, no bash-injection-via-quoting surprises. The
  argument for cutting: bash is always present on the target platforms
  and already does this correctly; we just added timeout/output-cap/cancel
  to vshell that bash gets for free via `Command` + a supervisor thread.
  The `Sandbox` trait is wired through `Session` and `Instance`, so
  every test that spawns a Session would need re-fixturing.

- **`wire::retry` breaker simplification.** Drop the 3-state
  Closed/Open/HalfOpen breaker (~150 LOC) for a flat "N consecutive
  failures → fail-fast for K seconds" guard. We don't have enough
  per-host traffic for half-open probing to matter.

- **`metrics` crate.** _Superseded by ADR-0003 — kept and wired._ 497
  LOC of DogStatsD UDP emission. We have call sites in Session, but the
  deployment isn't wired to a DogStatsD agent yet. Could shrink to a
  30-LOC trait stub with a real impl bolted on when there's a sink to
  hit.

- **`subagent` + `compact` inline into `harness`.** Both are <800 LOC
  features of the Session loop. Folding them in would reduce the
  workspace edge count but the crate boundary documents the feature.
  Low priority.

## Consequences

- **The "what's left" list is shorter.** A future round 8 audit shouldn't
  find any orphans matching the (1-3) check.
- **Forge integration is now external.** If we want GitHub/Gitea support
  later, it ships as an MCP server, not as a workspace crate. This
  matches the "MCP for everything that isn't core agent" principle.
- **Branch strategies live in `git/`** — adding `jj` (or `hg`, `pijul`,
  …) means a new crate, not extending `git/`.
- **`mcp` now has both client and server.** The doc on `mcp::server`
  must make clear that the two halves aren't symmetric — client speaks
  to remote servers; server exposes a local Tool registry over stdio.

## Alternatives considered

- **Keep everything, document "experimental."** Rejected: experimental
  code attracts hardening work it doesn't deserve (cluster auth was the
  proximate example).
- **Workspace cargo features to gate optional crates.** Rejected: same
  build still has to compile them; vendoring + audit surface is
  unchanged. Plus our policy is "no feature flags" elsewhere.
