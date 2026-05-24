#!/usr/bin/env bash
set -euo pipefail
# 01-hello: simplest invocation. Requires ANTHROPIC_API_KEY set.

: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY before running}"

agent run "Write a haiku about Rust ownership."
