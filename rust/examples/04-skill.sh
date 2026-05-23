#!/usr/bin/env bash
set -euo pipefail
# 04-skill: load a markdown skill, substitute {{KEY}} args, run the prompt.

: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY before running}"

# Create a throwaway skill in a tmp AGENTS_ROOT so we don't pollute the repo.
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/.agents/skills"

cat > "$WORK/.agents/skills/code-review.md" <<'MD'
You are reviewing a code diff. Be terse.

Diff:
```
{{DIFF}}
```

Output (in order):
1. One sentence: what changed.
2. Bug risks, if any.
3. Suggestions, if any.
MD

DIFF=$(cat <<'DIFF'
- fn add(a: i32, b: i32) -> i32 { a + b }
+ fn add(a: i32, b: i32) -> i32 { a.wrapping_add(b) }
DIFF
)

AGENTS_ROOT="$WORK" agent run --skill code-review --arg "DIFF=$DIFF"
