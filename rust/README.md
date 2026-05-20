# rust/

Experimental Rust core for a minimal agent harness. Distils ideas from
Sandcastle (sandbox provider × branch strategy, two-phase prompt,
completion signal, structured output) and Flue (instance/harness/session
hierarchy, virtual sandbox default, skills + roles, declarative triggers)
into a build with a small dependency footprint.

## Status

Feature-complete v1. 250 inline tests across the workspace; one direct
crate dependency (`curl-sys`), all transitive deps fully vendored.

## Layout

| Crate | Purpose | Direct deps |
|---|---|---|
| `actor/` | OS-thread actor primitive (`Send` / `ask` / `stopped`). std-only. | — |
| `wire/` | SIMD byte/pair scanners + `SseFramer` + `NdjsonSplitter` + `find_tag`. NEON (aarch64) + AVX2 (x86_64, runtime-detected) + scalar. | — |
| `vshell/` | In-process POSIX shell subset. Quoting, `$VAR`/`${VAR}`, `$(cmd)`, pipelines, redirects, sequencing, env prefix, built-ins. std-only. | — |
| `anthropic/` | Streaming Messages API client. libcurl + OpenSSL via `curl-sys`. Hand-rolled JSON. Tool-use / tool-result content blocks. | `wire`, `curl-sys` |
| `openai/` | Streaming `/v1/chat/completions` client for any OpenAI-compatible endpoint (OpenAI, OpenRouter, LM Studio, Ollama). | `wire`, `curl-sys` |
| `audit/` | Literal-pattern shell scanner. Blocks shai-hulud-class payloads (curl-pipe-to-shell, eval-of-curl, base64-pipe-to-shell). | `wire` |
| `fstools/` | `read_file` / `write_file` / `edit_file` tools implementing `harness::Tool`. Optional `root` sandboxing. | `harness`, `anthropic` |
| `tmpl/` | `{{KEY}}` substitution + `.agents/skills/<name>.md` + `.agents/roles/<name>.md` markdown loading. Built on `wire::scan_for_pair`. | `wire` |
| `server/` | Hand-rolled HTTP/1.1 + SSE server. `AgentHandler` trait, one thread per connection. std-only. | — |
| `harness/` | Instance + Session actors. `Model` + `Sandbox` + `Tool` traits. `AnthropicModel`, `OpenAiModel`, `AuditedShell<S>`, `BashTool`. Iteration loop with tool-use, completion signal, structured output, 16-turn cap. | `actor`, `wire`, `anthropic`, `openai`, `audit`, `vshell` |
| `cli/` | `agent run` / `agent serve` binary. Wires every other crate. | all of the above |

## Dependencies & supply chain

One direct dep: `curl-sys 0.4` with `static-curl` + `static-ssl` features.
`cargo vendor` was run at the workspace root; the resulting `vendor/`
tree (12 crates including OpenSSL + libcurl source for static builds) is
committed. `.cargo/config.toml` redirects crates.io to `vendored-sources`.
Builds with `--locked --frozen --offline` succeed against the vendored
tree.

`rust/.gitignore` is anchored so `target/` only matches Cargo build dirs,
not `vendor/**/target/` source subdirectories (cc 1.2 reorganised its
layout — the unanchored rule used to swallow real source files).

## Run

```bash
# All tests + lints
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo check --workspace --target aarch64-unknown-linux-gnu

# One-shot agent
ANTHROPIC_API_KEY=... cargo run --bin agent -- run "list the files in /tmp"

# OpenAI-compatible (also works with OPENAI_BASE_URL pointed at OpenRouter,
# LM Studio, Ollama, etc.)
OPENAI_API_KEY=... cargo run --bin agent -- run --openai "hello"

# HTTP server: POST /agents/chat/<id>, SSE response
cargo run --bin agent -- serve --addr 0.0.0.0:3583

# Skill-driven prompt (loads .agents/skills/triage.md, substitutes {{KEY}}s)
cargo run --bin agent -- run --skill triage --arg issue_number=42 --arg severity=high
```

## What's left

Deliberately deferred from v1:

- MCP client (streamable HTTP + stdio).
- Session persistence across HTTP requests.
- Multi-node `ActorRef`.
- HTTP keep-alive + request timeouts + connection cap (the server is
  fine for one client at a time; production needs hardening).
- Symlink-safe paths inside fstools (currently best-effort).
- Branch strategies (Sandcastle's worktree+merge-to-head pattern).

Each lands when a workload forces it.
