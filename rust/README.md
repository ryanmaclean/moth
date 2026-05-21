# rust/

Experimental Rust core for a minimal agent harness. Distils ideas from
Sandcastle (sandbox provider × branch strategy, two-phase prompt,
completion signal, structured output) and Flue (instance/harness/session
hierarchy, virtual sandbox default, skills + roles, declarative triggers)
into a build with a small dependency footprint.

## Status

Coding-agent grade. 18 crates, 456 inline tests across the workspace;
one direct external dependency (`curl-sys`), all transitive deps fully
vendored. HTTP server hardened, sessions persist, MCP-pluggable (stdio
+ HTTP), fstools symlink + hard-link safe, git + jj branch strategies,
Gitea forge client, multi-node `cluster::RemoteActorRef`, microbench
suite with measured numbers, cross-crate integration suite.

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
| `mcp/` | Model Context Protocol client (stdio + streamable-HTTP transports). Each remote tool implements `harness::Tool`. Session-id propagation, SSE response framing via `wire`, bearer auth. | `harness`, `wire`, `anthropic`, `curl-sys` |
| `git/` | Branch strategies for code-editing agents: `HeadStrategy` (works in repo_root, refuses dirty tree), `MergeToHeadStrategy` (temp worktree + merge back on success, leave on failure), `Branch{name}` (named persistent branch). Shells out to `git(1)`. | — |
| `integration/` | Cross-crate scenario tests. 12 whole-stack tests covering tool routing, audit blocking, fstools sandboxing, session persistence, MCP wiring, completion signal, turn cap, structured output extraction. | every crate it tests |
| `jj/` | Jujutsu (jj) branch strategies implementing `git::BranchStrategy`. Workspace + bookmark mapping. Tests skip cleanly if `jj` isn't installed. | `git` |
| `gitea/` | Minimal Gitea REST API client (PATs via `Authorization: token`, `/api/v1/...`). Issues, comments, PRs. Works with Forgejo, Codeberg. | `wire`, `anthropic`, `curl-sys` |
| `cluster/` | Multi-node `RemoteActorRef<M: Codec>` over TCP. Length-prefixed frames, per-node registry, fire-and-forget delivery, cached connections with re-dial on failure. | `actor`, `wire`, `anthropic` |
| `benches/` | Microbenchmarks for the SIMD/parsing hot paths. `std::time::Instant`-based, no criterion. `cargo test -p benches --release -- --nocapture`. | `wire`, `audit`, `anthropic`, `vshell` |
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

## Measured numbers

From `cargo test -p benches --release -- --nocapture` on x86_64 + AVX2:

```
wire::scan_for_byte 4 KiB (absent):       59 ns/op   68.9 GB/s
wire::scan_for_byte 64 KiB (absent):    1032 ns/op   63.5 GB/s
wire::scan_for_pair 4 KiB (absent):      146 ns/op   28.0 GB/s
wire::find_tag 4 KiB (tag at end):       100 ns/op   40.7 GB/s
audit::Scanner::scan benign:              25 ns/op  (was 133 — Aho-Corasick win)
audit::Scanner::scan malicious:           94 ns/op  (was 448)
audit::Scanner::scan 1 KiB no patterns: 2359 ns/op  (was 5501)
anthropic::json::parse SSE event:        835 ns/op
vshell::execute("echo $X"):              566 ns/op
```

SIMD byte scan saturates L1 bandwidth (~65 GB/s). Audit replaced its
per-pattern scan with a flat Aho-Corasick automaton; one walk now
finds every match regardless of pattern count.

## What's left

- Real-network E2E test (would require an `ANTHROPIC_API_KEY` and is
  sandbox-policy-sensitive).
- GitHub forge client (parallel to `gitea` — different auth scheme,
  same shape).
- Aho–Corasick for `audit` (per-pattern cost dominates the 1 KiB-no-hit
  case at ~5.5 µs; multi-pattern scan would cut that ~5×).
