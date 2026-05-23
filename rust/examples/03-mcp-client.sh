#!/usr/bin/env bash
set -euo pipefail
# 03-mcp-client: spawn an MCP server and use its tools.
#
# Self-hosted: we use `agent mcp-serve` itself as the MCP server so the
# example has no external dependencies. The parent agent inherits its
# child's tools (bash, read_file, write_file, edit_file) as MCP-callable
# tools — useful for nesting agents.

: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY before running}"

# In real use this would be `npx -y @modelcontextprotocol/server-filesystem /tmp`
# or any other MCP-spec server.
agent run \
  --mcp 'agent mcp-serve' \
  "List the files in /tmp older than 7 days, using the bash tool."
