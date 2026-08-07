#!/usr/bin/env bash
#
# Offline tests for scripts/tag_desktop_app.sh.
#
# The real path cannot be exercised without cutting a release, so the tag
# claim, the annotated-tag peel and the refuse-to-move branch run here
# against a mock ``gh`` that records the exact API calls made.
#
#   ./tests/release/test_tag_desktop_app.sh
#
# Requires: bash. No network/gh/git required.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="$REPO_ROOT/scripts/tag_desktop_app.sh"
WORKFLOW="$REPO_ROOT/.github/workflows/auto-release.yml"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
contains() { if grep -qF -- "$2" <<<"$1"; then ok "$3"; else bad "$3"; printf '        want substring: %s\n        got:            %s\n' "$2" "$1"; fi; }
lacks()    { if grep -qF -- "$2" <<<"$1"; then bad "$3"; else ok "$3"; fi; }

SHA_GOOD="1111111111111111111111111111111111111111"
SHA_BAD="2222222222222222222222222222222222222222"
TAG_OBJ="3333333333333333333333333333333333333333"

# ``$MODE`` selects the remote state the mock simulates; every invocation is
# appended to $TMP/calls so the test can assert on what was actually asked.
make_mock_gh() {
  cat > "$TMP/gh" <<'MOCK'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$CALLS"
case "$1 $2" in
  "api -X")  # POST repos/<r>/git/refs
    case "$MODE" in
      free) exit 0 ;;
      *)    echo "HTTP 422: Reference already exists" >&2; exit 1 ;;
    esac
    ;;
esac
# read-back paths
case "$*" in
  *"git/ref/tags/"*)
    case "$MODE" in
      taken_light)      printf 'commit\t%s\n' "$READ_SHA" ;;
      taken_annotated)  printf 'tag\t%s\n' "$TAG_OBJ" ;;
      unreadable)       exit 1 ;;
      *)                exit 1 ;;
    esac
    ;;
  *"git/tags/"*)
    printf 'commit\t%s\n' "$READ_SHA"
    ;;
esac
MOCK
  chmod +x "$TMP/gh"
}
make_mock_gh

# ``${3-…}``, not ``${3:-…}``: an EMPTY VERSION is a case under test, and
# ``:-`` would silently substitute the default for it. (No comments inside the
# assignment chain below — a comment line ends the ``\`` continuation, which
# silently drops every later assignment.)
run() {  # run <MODE> <READ_SHA> [VERSION]
  : > "$TMP/calls"
  MODE="$1" READ_SHA="$2" TAG_OBJ="$TAG_OBJ" CALLS="$TMP/calls" \
  GH="$TMP/gh" GITHUB_REPOSITORY="raullenchai/Rapid-MLX" \
  VERSION="${3-0.12.8}" RELEASE_SHA="$SHA_GOOD" \
    bash "$SCRIPT" 2>&1
}

# ==========================================================================
echo "== 1. free tag: claimed with one atomic POST =="
# ==========================================================================
OUT=$(run free "" ) && RC=0 || RC=$?
[ "${RC:-0}" -eq 0 ] && ok "exit 0 when the tag is free" || bad "exit 0 when the tag is free (got $RC)"
contains "$OUT" "Created rapid-mac-v0.12.8 at $SHA_GOOD" "reports the tag it created"
CALLS=$(cat "$TMP/calls")
contains "$CALLS" "ref=refs/tags/rapid-mac-v0.12.8" "POSTs the rapid-mac-v-prefixed ref"
contains "$CALLS" "sha=$SHA_GOOD" "POSTs the release commit"
lacks "$CALLS" "git/ref/tags/" "does not pre-check — the POST itself is the claim"

# ==========================================================================
echo "== 2. re-run over an ANNOTATED tag at the same commit is a no-op =="
# ==========================================================================
# The regression this guards: /git/ref/tags/X returns the TAG OBJECT's sha for
# an annotated tag. Comparing that raw value to the release commit reports
# "already exists somewhere else" for a tag that is in fact correct — and four
# of this repo's rapid-mac-v tags are annotated, so the very first re-run of
# the release job would have failed.
OUT=$(run taken_annotated "$SHA_GOOD") && RC=0 || RC=$?
[ "${RC:-0}" -eq 0 ] && ok "exit 0 when an annotated tag already points at the release commit" \
                     || bad "exit 0 when an annotated tag already points at the release commit (got $RC)"
contains "$OUT" "already points at $SHA_GOOD" "says it is a no-op"
contains "$(cat "$TMP/calls")" "git/tags/$TAG_OBJ" "peels the annotated tag object"

# ==========================================================================
echo "== 3. re-run over a LIGHTWEIGHT tag at the same commit is a no-op =="
# ==========================================================================
OUT=$(run taken_light "$SHA_GOOD") && RC=0 || RC=$?
[ "${RC:-0}" -eq 0 ] && ok "exit 0 for a lightweight tag at the release commit" \
                     || bad "exit 0 for a lightweight tag at the release commit (got $RC)"

# ==========================================================================
echo "== 4. a tag at a DIFFERENT commit is refused, never moved =="
# ==========================================================================
OUT=$(run taken_annotated "$SHA_BAD") && RC=0 || RC=$?
[ "${RC:-0}" -ne 0 ] && ok "non-zero when the tag points elsewhere" || bad "non-zero when the tag points elsewhere"
contains "$OUT" "already exists at $SHA_BAD" "names the commit the tag actually holds"
lacks "$(cat "$TMP/calls")" "--force" "never force-moves a published tag"

# ==========================================================================
echo "== 5. unreadable after a failed claim fails closed =="
# ==========================================================================
OUT=$(run unreadable "") && RC=0 || RC=$?
[ "${RC:-0}" -ne 0 ] && ok "non-zero when the tag can be neither created nor read" \
                     || bad "non-zero when the tag can be neither created nor read"
contains "$OUT" "refusing to guess" "says why it stopped"

# ==========================================================================
echo "== 6. a version the tag namespace cannot carry is rejected =="
# ==========================================================================
for BADV in "0.12" "0.12.8-rc1" "v0.12.8" ""; do
  OUT=$(run free "" "$BADV") && RC=0 || RC=$?
  [ "${RC:-0}" -ne 0 ] && ok "rejects VERSION='$BADV'" || bad "rejects VERSION='$BADV'"
done

# ==========================================================================
echo "== 7. workflow wiring =="
# ==========================================================================
# The app tag MUST be created through the API under the PAT. A ``git push``
# uses the credential actions/checkout persisted (GITHUB_TOKEN), and GitHub
# suppresses workflow runs caused by GITHUB_TOKEN pushes — the step would go
# green with no app build, signing, notarisation or DMG.
APP_STEP=$(sed -n '/Tag the desktop app at the same version/,/^      - name:/p' "$WORKFLOW")
contains "$APP_STEP" "secrets.RELEASE_PAT" "app tag step runs under the PAT"
contains "$APP_STEP" "scripts/tag_desktop_app.sh" "app tag step calls the tested script"
lacks "$APP_STEP" "git push" "app tag step does not git push the tag"

# The CHANGELOG check has to precede the irreversible engine publication, or a
# missing section ships the engine and then fails — the half-release this whole
# change exists to prevent.
CHANGELOG_LINE=$(grep -n "Pre-check the desktop app CHANGELOG" "$WORKFLOW" | head -1 | cut -d: -f1)
CREATE_LINE=$(grep -n "name: Create tag and release" "$WORKFLOW" | head -1 | cut -d: -f1)
if [ -n "$CHANGELOG_LINE" ] && [ -n "$CREATE_LINE" ] && [ "$CHANGELOG_LINE" -lt "$CREATE_LINE" ]; then
  ok "app CHANGELOG is checked before the engine release is created"
else
  bad "app CHANGELOG is checked before the engine release is created (changelog=$CHANGELOG_LINE create=$CREATE_LINE)"
fi

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
