#!/usr/bin/env bash
# cargo-annotate.sh — run a cargo command inside the builder image and
# surface the last ~20 lines of output as ::error:: GitHub annotations
# on non-zero exit. Used by every cargo step in rust.yml, release.yml,
# and cross-check.yml so the failure reason shows on the PR checks
# page without needing log auth.
#
# Usage:
#   cargo-annotate.sh <label> <cargo-args...>
# Example:
#   cargo-annotate.sh clippy clippy --workspace --locked -- -D warnings
#
# Assumes:
#   - $PWD is the workspace root (mounted into the container at /work).
#   - The docker image `agent-builder` exists (built by the calling
#     workflow with `docker build --target builder -t agent-builder .`).
#
# Extra `docker run` flags may be passed via $DOCKER_RUN_EXTRA (e.g.
# `-e CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=...`). Most env is
# already baked into the builder image.

set -u
set -o pipefail

LABEL="${1:?label required}"
shift

LOG="/tmp/cargo-${LABEL}.log"
mkdir -p "$(dirname "$LOG")"

# shellcheck disable=SC2086
docker run --rm \
  -v "$PWD:/work" -w /work \
  ${DOCKER_RUN_EXTRA:-} \
  agent-builder \
  cargo "$@" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}

echo "::notice::cargo $LABEL exit=$RC ($(wc -l <"$LOG") log lines)"

if [ "$RC" -ne 0 ]; then
  echo "::group::tail of cargo $LABEL log"
  tail -120 "$LOG"
  echo "::endgroup::"
  echo "::error::cargo $LABEL failed (exit $RC); see tail above"
  # GitHub annotations are single-line; escape newlines/CRs/percents.
  tail -20 "$LOG" | sed 's/%/%25/g; s/\r/%0D/g' | while IFS= read -r line; do
    [ -n "$line" ] && echo "::error::$line"
  done
fi

exit "$RC"
