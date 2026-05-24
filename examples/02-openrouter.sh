#!/usr/bin/env bash
set -euo pipefail
# 02-openrouter: route through any OpenAI-compatible endpoint.
# OpenRouter, LM Studio, Ollama, and vLLM all expose the same API shape.

: "${OPENAI_API_KEY:?set OPENAI_API_KEY (or any non-empty string for local models)}"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
MODEL="${MODEL:-anthropic/claude-sonnet-4}"

agent run --openai --model "$MODEL" "List three things Rust does better than C++."
