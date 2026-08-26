#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$ROOT/scripts/check_dmg_size_delta.sh"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/dmg-size-delta.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
PASS=0
FAIL=0

ok() { PASS=$((PASS + 1)); printf '  PASS %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf '  FAIL %s\n' "$1" >&2; }

# Sparse files exercise the real byte-counting path without consuming the
# corresponding physical disk space. The historical unit bug treated this
# 60 MiB growth as roughly 36 after mixing MiB and decimal MB.
truncate -s $((560 * 1024 * 1024)) "$TMP/current.dmg"
if bash "$SCRIPT" "$TMP/current.dmg" $((500 * 1024 * 1024)) 50 >/dev/null; then
  bad '60 MiB growth exceeds a 50 MiB cap'
else
  ok '60 MiB growth exceeds a 50 MiB cap'
fi

truncate -s $((550 * 1024 * 1024)) "$TMP/current.dmg"
if bash "$SCRIPT" "$TMP/current.dmg" $((500 * 1024 * 1024)) 50 >/dev/null; then
  ok 'exactly 50 MiB growth is allowed'
else
  bad 'exactly 50 MiB growth is allowed'
fi

truncate -s $((499 * 1024 * 1024)) "$TMP/current.dmg"
if bash "$SCRIPT" "$TMP/current.dmg" $((500 * 1024 * 1024)) 50 >/dev/null; then
  ok 'a smaller DMG is allowed'
else
  bad 'a smaller DMG is allowed'
fi

if bash "$SCRIPT" "$TMP/current.dmg" 999999999999999999999999999 50 >/dev/null; then
  ok 'large API integers cannot overflow into a false rejection or approval'
else
  bad 'large API integers cannot overflow into a false rejection or approval'
fi

for args in \
  "0 50" \
  "not-bytes 50" \
  "$((500 * 1024 * 1024)) not-cap"; do
  if bash "$SCRIPT" "$TMP/current.dmg" $args >/dev/null 2>&1; then
    bad "invalid numeric input fails closed: $args"
  else
    ok "invalid numeric input fails closed: $args"
  fi
done

printf 'passed: %d failed: %d\n' "$PASS" "$FAIL"
[[ "$FAIL" == 0 ]]
