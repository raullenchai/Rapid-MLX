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
# The claim is asserted precisely — verb AND endpoint — so a rewrite that
# POSTs somewhere else, or GETs where it should POST, fails here instead of
# passing against a mock that accepts anything.
if [ "$1 $2 $3" = "api -X POST" ]; then
  [ "$4" = "repos/$GITHUB_REPOSITORY/git/refs" ] || {
    echo "mock: claim POSTed to '$4', expected repos/$GITHUB_REPOSITORY/git/refs" >&2
    exit 64
  }
  case "$MODE" in
    free) exit 0 ;;
    *)    echo "HTTP 422: Reference already exists" >&2; exit 1 ;;
  esac
fi
# read-back paths
case "$*" in
  *"git/ref/tags/"*)
    case "$MODE" in
      taken_light)      printf 'commit\t%s\n' "$READ_SHA" ;;
      taken_annotated)  printf 'tag\t%s\n' "$TAG_OBJ" ;;
      # A chain of nested tag objects: each /git/tags/<sha> hop below returns
      # another tag until the hop count runs out.
      taken_chain)      printf 'tag\t%s\n' "$TAG_OBJ" ;;
      unreadable)       exit 1 ;;
      *)                exit 1 ;;
    esac
    ;;
  *"git/tags/"*)
    if [ "$MODE" = "taken_chain" ]; then
      HOPS=$(( $(cat "$TMP_HOPS" 2>/dev/null || echo 0) + 1 ))
      printf '%s' "$HOPS" > "$TMP_HOPS"
      if [ "$HOPS" -lt "$CHAIN_LEN" ]; then
        printf 'tag\t%s\n' "$TAG_OBJ"
      else
        printf 'commit\t%s\n' "$READ_SHA"
      fi
      exit 0
    fi
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
  : > "$TMP/hops"
  MODE="$1" READ_SHA="$2" TAG_OBJ="$TAG_OBJ" CALLS="$TMP/calls" \
  TMP_HOPS="$TMP/hops" CHAIN_LEN="${CHAIN_LEN:-1}" HAVE_PAT="${HAVE_PAT-true}" \
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
for BADV in "0.12" "0.12.8-rc0" "0.12.8-beta1" "v0.12.8" ""; do
  OUT=$(run free "" "$BADV") && RC=0 || RC=$?
  [ "${RC:-0}" -ne 0 ] && ok "rejects VERSION='$BADV'" || bad "rejects VERSION='$BADV'"
done

OUT=$(run free "" "0.13.0-rc1") && RC=0 || RC=$?
[ "${RC:-0}" -eq 0 ] && ok "accepts VERSION='0.13.0-rc1'" || bad "accepts VERSION='0.13.0-rc1'"
contains "$(cat "$TMP/calls")" "ref=refs/tags/rapid-mac-v0.13.0-rc1" \
  "RC desktop tag keeps the prerelease suffix"

# ==========================================================================
echo "== 6b. a deep annotated-tag chain still peels to its commit =="
# ==========================================================================
# The bound is on FOLLOWS, not iterations. A loop bounded by iterations
# rejects a chain whose last hop lands on a commit — it had the answer and
# threw it away, reporting "unreadable" AFTER the engine has published.
CHAIN_LEN=6 OUT=$(CHAIN_LEN=6 run taken_chain "$SHA_GOOD") && RC=0 || RC=$?
[ "${RC:-0}" -eq 0 ] && ok "peels a 6-deep annotated chain to the release commit" \
                     || bad "peels a 6-deep annotated chain to the release commit (got $RC)"
contains "$OUT" "already points at $SHA_GOOD" "6-deep chain reads as a no-op"

# ...and a chain past the bound is still refused rather than looping forever.
OUT=$(CHAIN_LEN=99 run taken_chain "$SHA_GOOD") && RC=0 || RC=$?
[ "${RC:-0}" -ne 0 ] && ok "refuses a chain deeper than the peel bound" \
                     || bad "refuses a chain deeper than the peel bound"

# ==========================================================================
echo "== 6c. no PAT: refuse to run at all =="
# ==========================================================================
# A tag written with GITHUB_TOKEN fires no workflow AND blocks every retry: the
# re-run finds it already at the right commit and exits green, so the DMG can
# never be built for that version without hand-deleting a published tag.
# Skipping quietly is no better — the run would end green with only the engine
# released, and the next re-run sees the published engine Release and decides
# there is nothing left to release. So this fails, and the workflow asks the
# same question before anything irreversible happens.
#
# The EMPTY case is the one that bites in production: a workflow expression
# that evaluates to nothing would otherwise hit the "unset means a human is
# running this" default and recreate the dead-tag failure.
for BAD_PAT in false ""; do
  OUT=$(HAVE_PAT="$BAD_PAT" run free "") && RC=0 || RC=$?
  [ "${RC:-0}" -ne 0 ] && ok "non-zero with HAVE_PAT='$BAD_PAT'" || bad "non-zero with HAVE_PAT='$BAD_PAT'"
  lacks "$(cat "$TMP/calls")" "git/refs" "creates NO tag with HAVE_PAT='$BAD_PAT'"
  contains "$OUT" "RELEASE_PAT is not available" "says which secret is missing (HAVE_PAT='$BAD_PAT')"
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
# Finding RELEASE_PAT in the env is not enough — the step falls back to
# GITHUB_TOKEN, so it must also be TOLD whether the token it got can trigger a
# workflow. Without this wiring the script's own guard defaults to "true" and
# the fallback silently creates a dead tag.
contains "$APP_STEP" "HAVE_PAT: \${{ steps.appcheck.outputs.have_pat }}" \
  "app tag step is told whether the PAT is actually present"
# Assert the EXACT expression and the EXACT guard. A bare "have_pat=" check
# passed even when the published value was empty — which the script's
# "unset means a human" default then read as consent, recreating the very
# dead-tag failure this wiring exists to prevent. Pin what is written, not
# that something is.
PREFLIGHT=$(sed -n '/Pre-check the desktop app CHANGELOG/,/Build release notes/p' "$WORKFLOW")
contains "$PREFLIGHT" "HAVE_PAT: \${{ secrets.RELEASE_PAT != '' }}" \
  "pre-flight derives have_pat from the secret's presence"
contains "$PREFLIGHT" 'echo "have_pat=${HAVE_PAT}" >> "$GITHUB_OUTPUT"' \
  "pre-flight publishes that value verbatim, not a literal"
contains "$PREFLIGHT" 'if [ "$HAVE_PAT" != "true" ]; then' \
  "pre-flight refuses before anything is published"

# Everything that can refuse the app half has to happen BEFORE the engine
# Release is published. Publishing is the irreversible step: once the Release
# exists, `detect` sets should_release=false and no re-run reaches these steps
# again, so a failure after it strands the version as engine-only forever.
CREATE_LINE=$(grep -n "name: Create tag and release" "$WORKFLOW" | head -1 | cut -d: -f1)
before_publish() {  # before_publish <step name> <label>
  local line
  line=$(grep -n "$1" "$WORKFLOW" | head -1 | cut -d: -f1)
  if [ -n "$line" ] && [ -n "$CREATE_LINE" ] && [ "$line" -lt "$CREATE_LINE" ]; then
    ok "$2"
  else
    bad "$2 (step=$line publish=$CREATE_LINE)"
  fi
}
before_publish "Pre-check the desktop app CHANGELOG" \
  "app CHANGELOG is checked before the engine release is published"
before_publish "name: Tag the desktop app at the same version" \
  "app tag is claimed before the engine release is published"

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
