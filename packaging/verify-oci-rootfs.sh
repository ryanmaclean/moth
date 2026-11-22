#!/usr/bin/env bash
# Verify an exported OCI root filesystem tar against the staged release payload.
# Usage: verify-oci-rootfs.sh <arch> <rootfs.tar> <payload-dir>
set -euo pipefail

arch=$1
rootfs=$2
payload=$3/$arch
manifest=packaging/native-notice-files.txt
prefix=usr/share/doc/moth/THIRD_PARTY_NOTICES/native

listing=$(mktemp)
trap 'rm -f "$listing"' EXIT
tar -tvf "$rootfs" | awk '{t=substr($1,1,1); $1=$2=$3=$4=$5=""; sub(/^ +/,""); n=$0; sub(/^\.\//,"",n); print t " " n}' > "$listing"

check() { # <path-in-image> <expected-file>
  test "$(grep -cFx -- "- $1" "$listing")" -eq 1
  cmp <(tar -xOf "$rootfs" "$1" 2>/dev/null || tar -xOf "$rootfs" "./$1") "$2"
}

check usr/local/bin/agent "$payload/agent"
check usr/share/licenses/moth/LICENSE "$payload/LICENSE"
check "$prefix/FILES.txt" "$payload/native/FILES.txt"
while IFS= read -r source; do
  check "$prefix/$source" "$payload/native/$source"
done < "$manifest"
test "$(grep -c "^- $prefix/" "$listing")" -eq 15
echo "verified pushed $arch OCI agent and 15 native files against release tar payload"
