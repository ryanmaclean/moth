# Cross-project lessons — 2026-09

Moth is the strongest existing candidate for the **tiny reusable agent worker** in the lower-bound runtime work.

## What other repos teach Moth

- **BOP**: filesystem state transitions and run identity should remain external/canonical; Moth should execute work, not become another scheduler DB.
- **smolFire**: target a minimal FreeBSD/microVM profile and measure bytes/RSS, not only desktop/server environments.
- **Genoa**: build/deployment receipts belong outside the harness.
- **skills/quota-gate + Jev/System One experiments**: routing/gating should be pluggable and cheap; do not hardwire expensive model decisions.
- **HAMMER/LFS/FFS work**: Moth's JSONL runlog and SessionStore are useful baselines, but test whether native filesystem history can eliminate or reduce them.

## Avoid duplication

Do not add a second actor primitive, subagent primitive, persistence layer, or metrics emitter in BOP/smolFire when Moth already provides one.

## Agent assignment

Copilot is primary; use `@codex` delegation when Copilot capacity/credits are unavailable.
