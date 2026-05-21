# rust/

Experimental Rust core for a minimal agent harness. Distils ideas from
Sandcastle (sandbox provider × branch strategy, two-phase prompt,
completion signal, structured output) and Flue (instance/harness/session
hierarchy, virtual sandbox default, skills + roles, declarative triggers)
into a build with a small dependency footprint.

## Status

Production-shippable v1. 13 crates, 320 inline tests across the workspace;
one direct external dependency (`curl-sys`), all transitive deps fully
vendored. HTTP server hardened (timeouts, connection cap, keep-alive),
sessions persist across requests, MCP-pluggable, fstools symlink-safe.

## Layout

| Crate | Purpose | Direct deps |
|---|---|---|
| `actor/` | OS-thread actor primitive (`Send` / `ask` / `stopped`). | — |
| `wire/` | SIMD byte/pair scanners + `SseFramer` + `NdjsonSplitter` + `find_tag`. NEON (aarch64) + AVX2 (x86_64, runtime-detected) + scalar. | — |
| `vshell/` | In-process POSIX shell subset. Quoting, `$VAR`/`${VAR}`, `$(cmd)`, pipelines, redirects, sequencing, env prefix, built-ins. | — |
| `anthropic/` | Streaming Messages API client. libcurl + OpenSSL via `curl-sys`. Hand-rolled JSON. Tool-use / tool-result content blocks. | `wire`, `curl-sys` |
| `openai/` | Streaming `/v1/chat/completions` client for any OpenAI-compatible endpoint (OpenAI, OpenRouter, LM Studio, Ollama). | `wire`, `curl-sys` |
| `audit/` | Literal-pattern shell scanner. Blocks shai-hulud-class payloads. | `wire` |
| `fstools/` | `read_file` / `write_file` / `edit_file` tools. Per-component canonicalisation + leaf `symlink_metadata` check makes them symlink-safe when a `root` is set. | `harness`, `anthropic` |
| `tmpl/` | `{{KEY}}` substitution + `.agents/skills/<name>.md` + `.agents/roles/<name>.md` markdown loading. | `wire` |
| `server/` | Hand-rolled HTTP/1.1 + SSE. Read/write timeouts, configurable connection cap (atomic counter), HTTP/1.1 keep-alive on non-streaming responses. | — |
| `mcp/` | Model Context Protocol client (stdio transport). Each remote tool implements `harness::Tool`. | `harness`, `wire`, `anthropic` |
| `persist/` | File-backed `SessionStore`. Atomic writes via tmp + rename. Version-tagged JSON; key validation rejects path-traversal. | `harness`, `anthropic` |
| `harness/` | Instance + Session actors. `Model` / `Sandbox` / `Tool` / `SessionStore` traits. `AnthropicModel`, `OpenAiModel`, `AuditedShell<S>`, `BashTool`. Iteration loop with tool-use, completion signal, structured output, 16-turn cap, optional persistence hook. | `actor`, `wire`, `anthropic`, `openai`, `audit`, `vshell` |
| `cli/` | `agent run` / `agent serve` binary. Wires every other crate. | all of the above |

## Dependencies & supply chain

One direct external dep: `curl-sys 0.4` with `static-curl` + `static-ssl`
features. `cargo vendor` was run at the workspace root; the resulting
`vendor/` tree (12 crates including OpenSSL + libcurl source for static
builds) is committed. `.cargo/config.toml` redirects crates.io to
`vendored-sources`. Builds with `--locked --frozen --offline` succeed.

`rust/.gitignore` is anchored so `target/` only matches Cargo build
dirs, not `vendor/**/target/` source subdirectories (cc 1.2 reorganised
its layout — the unanchored rule used to swallow real source files).

## Run

```bash
# All tests + lints
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo check --workspace --target aarch64-unknown-linux-gnu

# One-shot agent
ANTHROPIC_API_KEY=... cargo run --bin agent -- run "list /tmp"

# OpenAI-compatible (with OPENAI_BASE_URL pointed at OpenRouter,
# LM Studio, Ollama, etc.)
OPENAI_API_KEY=... cargo run --bin agent -- run --openai "hello"

# Skill-driven prompt (loads .agents/skills/triage.md, substitutes {{KEY}}s)
cargo run --bin agent -- run --skill triage --arg issue_number=42

# Persist conversations across runs
cargo run --bin agent -- run --sessions ./.sessions "first turn"
cargo run --bin agent -- run --sessions ./.sessions "second turn — remembers the first"

# Register external MCP tools (e.g. an MCP-spec'd filesystem server)
cargo run --bin agent -- run --mcp 'npx -y @modelcontextprotocol/server-filesystem /tmp' "list /tmp"

# HTTP server: POST /agents/chat/<id>, SSE response, sessions persist by id
cargo run --bin agent -- serve --addr 0.0.0.0:3583 --sessions ./.sessions
```

## What's left

- HTTP transport for MCP (stdio only in v1; the same `Transport` trait
  takes an HTTP impl).
- Multi-node `ActorRef` (location-transparent addressing across nodes).
- Sandcastle's worktree + merge-to-head branch strategies.
- Hard-link safety inside `fstools` (path-level defence only — a hard
  link to an outside-root inode isn't distinguishable by path).
- Real-network E2E test (would require an `ANTHROPIC_API_KEY` and is
  sandbox-policy-sensitive).
