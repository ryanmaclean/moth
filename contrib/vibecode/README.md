# contrib/vibecode

Moved from ryanmaclean/vibecode-webgui@129302a73be9719d213561116319d13ed679fd71 on 2026-10-24.

This directory holds agent-harness code that used to live in the VibeCode web IDE repo. It moved
here under the repo-boundary split (vibecode-webgui issues #2125 and #2130), because moth owns agent
execution and VibeCode keeps only the UI and control surface. It is a plain copy without history.
To see the history, run `git log` on the old paths in vibecode-webgui at the commit above.

None of this is part of the moth cargo workspace. moth CI does not build or test it. The code is
unchanged from the source commit. Only the paths changed.

| Path here | Old path in vibecode-webgui | What it is | Licence |
|---|---|---|---|
| `sdk-python/` | `tools/sdk/sdk/python/` (the doubled `sdk/sdk` is flattened) | `vibecode-agents` Python SDK: client, streaming, tool registry, sample agents, unit tests | MIT, own `sdk-python/LICENSE` (Copyright (c) 2025 VibeCode) |
| `ralph_loop/` | `tools/ralph_loop/` | `ralph_loop.py`, a Ralph-loop runner for the Codex or Ollama backends | MIT, `LICENSE` in this directory |
| `ai-cli-tools/` | `src/lib/ai-cli-tools/` | TypeScript wrappers for the claude-code, codex, gemini and opencode CLIs. Nothing in vibecode-webgui imported them. | MIT, `LICENSE` in this directory |

`LICENSE` in this directory is the vibecode-webgui MIT licence (Copyright (c) 2026 VibeCode
Contributors). It covers `ralph_loop/` and `ai-cli-tools/`. The top-level moth `LICENSE` covers the
rest of the repo.

Notes:

- `ralph_loop.py` tries `from vibecode.telemetry import init_telemetry` and skips telemetry if the
  import fails, so it runs without VibeCode.
- `sdk-python/PROJECT_SUMMARY.md` still names the old vibecode-webgui paths. It is kept unchanged.
