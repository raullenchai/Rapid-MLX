#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$ROOT/scripts/release_asset_size.sh"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/release-asset-size.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
PASS=0
FAIL=0

ok() { PASS=$((PASS + 1)); printf '  PASS %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf '  FAIL %s\n' "$1" >&2; }

cat > "$TMP/gh" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
[[ "$*" == "release view rapid-mac-v0.12.17 --json assets --jq .assets" ]] || exit 9
[[ "${MOCK_FAIL:-0}" == 0 ]] || { echo 'auth/API failed' >&2; exit 1; }
printf '%s\n' "${MOCK_ASSETS:?}"
MOCK
chmod +x "$TMP/gh"

run_case() {
  MOCK_ASSETS="$1" MOCK_FAIL="${2:-0}" GH="$TMP/gh" \
    bash "$SCRIPT" rapid-mac-v0.12.17 rapid-mlx-desktop.dmg 2>&1
}

OUT=$(run_case '[{"name":"rapid-mlx-desktop.dmg","size":123}]') && RC=0 || RC=$?
[[ "$RC" == 0 && "$OUT" == $'present\t123' ]] && ok 'returns one positive canonical asset size' || bad 'returns one positive canonical asset size'

OUT=$(run_case '[{"name":"other.dmg","size":123}]') && RC=0 || RC=$?
[[ "$RC" == 0 && "$OUT" == 'absent' ]] && ok 'verified missing asset is the only skip state' || bad 'verified missing asset is the only skip state'

for fixture in \
  'not-json' \
  '{"assets":[]}' \
  '[{"name":"rapid-mlx-desktop.dmg","size":0}]' \
  '[{"name":"rapid-mlx-desktop.dmg","size":1},{"name":"rapid-mlx-desktop.dmg","size":2}]'; do
  if run_case "$fixture" >/dev/null; then
    bad "malformed/ambiguous fixture fails: $fixture"
  else
    ok "malformed/ambiguous fixture fails: $fixture"
  fi
done

if run_case '[]' 1 >/dev/null; then
  bad 'gh auth/API failure fails closed'
else
  ok 'gh auth/API failure fails closed'
fi

printf 'passed: %d failed: %d\n' "$PASS" "$FAIL"
[[ "$FAIL" == 0 ]]
