# 0003: Keep and wire the metrics crate (supersedes ADR-0002's deferred-cut bullet)

Status: Accepted

Date: 2026-05

## Context

ADR-0002 ("Subtraction-first scope policy") listed the `metrics` crate
among the deferred cut candidates: "497 LOC of DogStatsD UDP emission.
We have call sites in Session, but the deployment isn't wired to a
DogStatsD agent yet. Could shrink to a 30-LOC trait stub with a real
impl bolted on when there's a sink to hit."

Under ADR-0002's own criteria, a crate earns its place if the CLI
reaches it on a hot path or an in-workspace crate that does depends on
it. `metrics` had Session call sites but no live emit path from the CLI,
so it sat on the fence — kept tentatively, flagged for a possible cut to
a trait stub.

We've now made the call the other way: rather than shrink it, we're
investing in observability and wiring it end-to-end.

## Decision

Keep the `metrics` crate as-is and treat it as **load-bearing
observability**, not speculative infrastructure. It is now wired end to
end:

- `HarnessState` carries a `metrics::Client` (defaulting to
  `Client::disabled()` so call sites never branch). `agent run` and
  `agent serve` construct a real client and install it via
  `with_metrics`.
- A `--metrics <HOST:PORT>` flag is added to `run` and `serve`,
  overriding `DOGSTATSD_ADDR`. Resolution order is flag > env >
  disabled.
- Subagents spawned through the `task` tool inherit the parent's
  emitter, so child prompt/tool/audit metrics land in the same sink
  with the same prefix and constant tags.
- `agent doctor` reports the configured sink so misconfiguration is
  visible before a run.

The Session loop emits `agent.prompt.started` (counter),
`agent.prompt.turns` (histogram), `agent.prompt.duration_ms` (timer,
tagged by outcome, fired on every exit path including panic),
`agent.tool.calls` (counter), `agent.tool.duration_ms` (timer), and
`agent.audit.blocked` (counter). See QUICKSTART.md "Metrics &
observability" for the full table.

This satisfies ADR-0002's criterion (1): `agent run` / `agent serve`
reach the emit path under normal use, not behind an experimental flag.

## Consequences

- **Observability is a first-class feature, not a stub.** Operators get
  prompt latency, turn count, per-tool success/latency, and a security
  signal (audit blocks) out of the box.
- **Opt-in and zero-cost when off.** With no flag and no env var the
  client is disabled: no socket is bound and every emit returns
  immediately. Leaving metrics off costs nothing on the hot path.
- **DogStatsD is the contract.** Any DogStatsD receiver works — a
  Datadog Agent, or `statsd_exporter` into Prometheus. We commit to the
  metric names and tag schema above as a stable surface; renaming a
  metric is a breaking change for downstream dashboards.
- **The trait-stub option is closed.** We won't replace the UDP impl
  with a 30-LOC stub; the real emitter is the deployed path.

## Alternatives considered

- **Shrink to a 30-LOC trait stub (the ADR-0002 suggestion).** Rejected:
  we have a concrete need for the signal and a standard wire format that
  every metrics backend already speaks. A stub would just defer the same
  work and lose the tested DogStatsD rendering/sanitisation.
- **Keep the crate but leave it unwired.** Rejected: that's the exact
  limbo ADR-0002 flagged — call sites with no live sink invite the "cut
  it" argument on the next audit. Wiring it removes the ambiguity.
- **Pull in a third-party metrics crate.** Rejected: conflicts with the
  supply-chain-isolation goal (ADR-0001). The in-tree emitter is ~497
  LOC with zero external deps and is fully vendored already.
