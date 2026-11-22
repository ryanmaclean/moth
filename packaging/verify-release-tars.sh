#!/usr/bin/env bash
# Verify the Linux release tarballs of this run and stage one payload per
# architecture under release-payload/<arch>/ for the OCI image assembly.
# Usage: verify-release-tars.sh <dist-dir> <ref-name> <payload-dir>
set -euo pipefail

dist=$1
ref=$2
payload_root=$3
manifest=packaging/native-notice-files.txt

test "$(wc -l < "$manifest")" -eq 14
test "$(sort -u "$manifest" | wc -l)" -eq 14
while IFS= read -r source; do
  case "$source" in vendor/*) ;; *) exit 1 ;; esac
  case "$source" in *..*|*//*) exit 1 ;; esac
  test -f "$source" && test ! -L "$source"
done < "$manifest"

declare -A targets=([amd64]=x86_64-unknown-linux-gnu [arm64]=aarch64-unknown-linux-gnu)

expected=$(for t in "${targets[@]}"; do
  printf '%s\n' "agent-$ref-$t.tar.gz" "agent-$ref-$t.tar.gz.sha256"
done | LC_ALL=C sort)
test "$(ls -A "$dist" | LC_ALL=C sort)" = "$expected"

mkdir "$payload_root"
for arch in amd64 arm64; do
  stage="agent-$ref-${targets[$arch]}"
  archive="$dist/$stage.tar.gz"
  digest=$(sha256sum "$archive" | cut -d' ' -f1)
  test "$(cat "$archive.sha256")" = "$digest  $stage.tar.gz"

  # Only regular files and directories, no absolute or parent paths, and
  # exactly the expected file set.
  ! tar -tvzf "$archive" | cut -c1 | grep -qv '^[-d]$'
  ! tar -tzf "$archive" | grep -qE '(^|/)\.\.(/|$)|^/'
  wanted=$({
    printf '%s\n' "$stage/agent" "$stage/LICENSE" "$stage/README.md" "$stage/CHANGELOG.md" \
      "$stage/THIRD_PARTY_NOTICES/native/FILES.txt"
    sed "s#^#$stage/THIRD_PARTY_NOTICES/native/#" "$manifest"
  } | LC_ALL=C sort)
  test "$(tar -tzf "$archive" | grep -v '/$' | LC_ALL=C sort)" = "$wanted"

  work=$(mktemp -d)
  tar -xzf "$archive" -C "$work" --no-same-owner
  src="$work/$stage"
  test -x "$src/agent"
  cmp "$src/LICENSE" LICENSE
  cmp "$src/THIRD_PARTY_NOTICES/native/FILES.txt" "$manifest"
  while IFS= read -r source; do
    cmp "$src/THIRD_PARTY_NOTICES/native/$source" "$source"
  done < "$manifest"

  out="$payload_root/$arch"
  mkdir -p "$out"
  install -m 0755 "$src/agent" "$out/agent"
  install -m 0644 "$src/LICENSE" "$out/LICENSE"
  cp -R "$src/THIRD_PARTY_NOTICES/native" "$out/native"
  echo "verified $arch tar=$digest agent=$(sha256sum "$out/agent" | cut -d' ' -f1) notices=14"
  rm -rf "$work"
done
