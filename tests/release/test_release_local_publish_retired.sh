#!/usr/bin/env bash
#
# Offline regression: release-local.sh --publish is RETIRED fail-closed (#2301).
#
# A local tag push must NOT be able to cut a public Desktop release — it would
# bypass the signed Desktop candidate, the live release-blocker / main-head
# checks, and the protected `rapid-mac-tag` reviewer approval. Provable offline:
#   * --publish exits NON-ZERO;
#   * it performs ZERO git/gh/security mutation (mock stubs record every call);
#   * it does NOT source the operator's RAPID_RELEASE_ENV (a sentinel env file
#     that would print a marker / fail if sourced is untouched);
#   * it names the canonical flow (bump PR → rapid-mac-tag approval).
#   * --check still parses and exits 0 (dogfood/--check preserved).
#
#   ./tests/release/test_release_local_publish_retired.sh
#
# Requires: bash. No network/real gh/git/security needed.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="$REPO_ROOT/apps/rapid-mac/scripts/release-local.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }

# A sentinel env file: if the script sources it, it runs the flagging touch so
# we can assert the operator-owned file was never executed.
SENTINEL="$TMP/sentinel.env"
cat > "$SENTINEL" <<'EOF'
# If this file is ever sourced, touch the marker (only meaningful for --check,
# which legitimately sources it; --publish must never source it). Uses an `if`
# so it is a no-op (exit 0) when the marker var is unset — never trips set -e.
if [[ -n "${SENTINEL_UNLESS_NOT_SOURCED_MARKER:-}" ]]; then
  touch "$SENTINEL_UNLESS_NOT_SOURCED_MARKER"
fi
EOF

# Mock git/gh/security on PATH: every invocation appends to a call log (writes
# nothing, runs nothing real). If the retired --publish honours its ordering, the
# log stays empty.
BIN="$TMP/bin"
mkdir -p "$BIN"
for tool in git gh security; do
  cat > "$BIN/$tool" <<'MOCK'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$CALL_LOG"
exit 0
MOCK
  chmod +x "$BIN/$tool"
done
# A mock that records the process name even when the tool isn't one of the above
# (e.g. PlistBuddy shouldn't run either for --publish).
cat > "$BIN/plutil" <<'MOCK'
#!/usr/bin/env bash
printf 'plutil %s\n' "$*" >> "$CALL_LOG"
exit 0
MOCK
chmod +x "$BIN/plutil"

CALL_LOG="$TMP/calls"
: > "$CALL_LOG"
MARKER="$TMP/sourced-no"

# Run from the apps/rapid-mac dir so relative paths resolve like real usage.
OUT=$(
  cd "$REPO_ROOT/apps/rapid-mac" \
  && CALL_LOG="$CALL_LOG" PATH="$BIN:$PATH" \
     RAPID_RELEASE_ENV="$SENTINEL" \
     SENTINEL_UNLESS_NOT_SOURCED_MARKER="$MARKER" \
     bash scripts/release-local.sh --publish rapid-mac-v9.9.9 2>&1
) && RC=0 || RC=$?

# 1) Non-zero exit.
[ "${RC:-0}" -ne 0 ] && ok "--publish exits non-zero (retired)" \
                     || bad "--publish exits non-zero (retired)"

# 2) Zero git/gh/security/plutil mutation.
containstool() { grep -qE "^(git|gh|security|plutil) " "$CALL_LOG"; }
if containstool; then
  bad "--publish performs NO git/gh/security/plutil call"
  cat "$CALL_LOG"
else
  ok "--publish performs NO git/gh/security/plutil call"
fi

# 3) RAPID_RELEASE_ENV was NOT sourced (sentinel would have touched $MARKER).
if [[ -e "$MARKER" ]]; then
  bad "--publish does not source the operator env file"
else
  ok "--publish does not source the operator env file"
fi

# 4) Message names the canonical flow + the protected gate, and never claims a
#    tag was pushed.
contains() { if grep -qF -- "$2" <<<"$1"; then ok "$3"; else bad "$3"; fi; }
contains "$OUT" "chore: bump version to X.Y.Z" "names the canonical bump-PR flow"
contains "$OUT" "rapid-mac-tag" "names the protected approval gate (rapid-mac-tag)"
contains "$OUT" "retired" "explicitly says --publish is retired"

# 5) --check still parses and exits 0 (dogfood/--check preserved).
OUT2=$(
  cd "$REPO_ROOT/apps/rapid-mac" \
  && CALL_LOG="$TMP/calls2" PATH="$BIN:$PATH" RAPID_RELEASE_ENV="$SENTINEL" \
     bash scripts/release-local.sh --check 2>&1
) && RC2=0 || RC2=$?
[ "${RC2:-0}" -eq 0 ] && ok "--check still exits 0 (preserved)" \
                      || bad "--check still exits 0 (preserved)"

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
