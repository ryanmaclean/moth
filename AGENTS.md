# AGENTS.md

## Role

Tiny reusable agent execution harness.

## Owns

- actors
- subagents
- tool/session execution
- minimal persistence/runlog
- metrics emitter

## Do not duplicate

- work scheduler DB
- OS/image builder
- canonical lineage store

## Sibling repos to consult first

- ryanmaclean/bop
- ryanmaclean/smolfire
- ryanmaclean/genoa

## Cross-project context

Read `docs/CROSS-PROJECT-LESSONS-2026-09.md` before making architectural changes.

## Agent delegation

- Primary GitHub coding agent: Copilot when assignable/available.
- Fallback: delegate the issue or PR to Codex with `@codex`.
- Do not treat Copilot/Codex state as canonical project state; keep canonical work in repo issues/BOP/filesystem state.
