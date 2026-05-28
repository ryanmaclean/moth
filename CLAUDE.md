# Agent harness — Rust workspace

Cargo workspace, 19 crates, one direct external dep (`curl-sys`). All
transitive deps live under `vendor/`. Every cargo invocation runs with
`--locked --frozen --offline` — vendored sources are the only source
of truth; network fetches are a build-failure signal, not a fallback.

## Daily commands

```bash
cargo check   --workspace --locked --frozen --offline
cargo clippy  --workspace --all-targets --locked --frozen --offline -- -D warnings
cargo test    --workspace --locked --frozen --offline
cargo fmt --all
```

Cross-target sanity check (must pass before any release tag):

```bash
cargo check --workspace --target aarch64-unknown-linux-gnu --locked --frozen --offline
```

## Install

```bash
cargo install --locked --offline --frozen --path cli --root ~/.local
# binary lands at ~/.local/bin/agent
```

The shipped binary is `agent`. Subcommands: `run`, `serve`, `mcp-serve`,
`doctor`, `--version`.

## Architecture decisions

ADRs live under `docs/adr/`. Read them before proposing structural
changes. `0002-cut-list.md` codifies the round-7 **subtraction rule**:
a crate stays only if (1) the CLI hits it on a hot path, (2) another
qualifying crate depends on it, or (3) it's a bench/integration/test
fixture. "Might be useful later" is not a reason — re-adding is one
`git revert` away.

## Changelog

Root `CHANGELOG.md`, Keep-A-Changelog format, semver pre-1.0. Add
user-visible changes under `## Unreleased` in `Added` / `Changed` /
`Fixed` / `Removed`.
