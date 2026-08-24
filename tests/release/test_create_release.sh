#!/usr/bin/env bash
#
# Offline tests for scripts/create_release.sh (issue #1462).
#
# The real workflow cannot be run without cutting a release, so the tag-claim
# and tag-without-Release recovery logic is exercised here against a mock
# ``gh`` that records the exact API calls made and tracks the tag's state.
#
#   ./tests/release/test_create_release.sh
#
# Requires: bash. No network/gh/git required.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="$REPO_ROOT/scripts/create_release.sh"
WORKFLOW="$REPO_ROOT/.github/workflows/auto-release.yml"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0

ok()   { PASS=$((PASS + 1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad()  { FAIL=$((FAIL + 1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
contains() { if grep -qF -- "$2" <<<"$1"; then ok "$3"; else bad "$3"; printf '        want substring: %s\n        got:            %s\n' "$2" "$1"; fi; }
lacks()    { if grep -qF -- "$2" <<<"$1"; then bad "$3"; else ok "$3"; fi; }

SHA_GOOD="1111111111111111111111111111111111111111"
SHA_BAD="2222222222222222222222222222222222222222"

# ==========================================================================
echo "== 0. workflow wiring keeps tag-without-Release recovery reachable =="
# ==========================================================================
NORMAL_DETECT=$(sed -n '/# --- Normal push path/,/# 2) The gate/p' "$WORKFLOW")
lacks "$NORMAL_DETECT" "git ls-remote --tags" \
  "normal detect does not skip merely because a tag exists"
contains "$NORMAL_DETECT" 'RELEASE_DRAFT=$(gh release view "v$VERSION"' \
  "normal detect checks published-vs-draft state"
contains "$NORMAL_DETECT" 'if [ "$RELEASE_DRAFT" = "false" ]; then' \
  "only a published Release enters the idempotent skip branch"
FALSE_OUTPUT_LINE=$(grep -n 'echo "should_release=false"' <<<"$NORMAL_DETECT" | head -1 | cut -d: -f1)
TRUE_OUTPUT_LINE=$(grep -n 'echo "should_release=true"' <<<"$NORMAL_DETECT" | tail -1 | cut -d: -f1)
[ -n "$FALSE_OUTPUT_LINE" ] && [ -n "$TRUE_OUTPUT_LINE" ] && \
  [ "$FALSE_OUTPUT_LINE" -lt "$TRUE_OUTPUT_LINE" ] && \
  ok "draft/missing Release falls through to should_release=true" || \
  bad "draft/missing Release falls through to should_release=true"
contains "$(cat "$WORKFLOW")" "Tag v\$VERSION exists without a published Release — will recover" \
  "forced dispatch keeps bare-tag recovery reachable"
contains "$(cat "$WORKFLOW")" "bash scripts/create_release.sh" \
  "release job invokes the atomic helper"

# Run the script with a stateful mock ``gh``. Env vars:
#   MOCK_RELEASE_VIEW   "yes" -> release view succeeds (Release exists)
#   MOCK_RELEASE_DRAFT  "yes" -> existing Release is an unpublished draft
#   MOCK_RELEASE_MARKED "yes" -> draft carries the workflow ownership marker
#   MOCK_MARKER_NOT_FIRST "yes" -> marker appears only inside manual notes
#   MOCK_WRONG_TAG_MARKER "yes" -> marker belongs to another tag at same SHA
#   MOCK_TAG_REF        SHA the tag already points at ("" = absent)
#   MOCK_CLAIM_FAILS    "yes" -> the atomic refs POST fails (tag exists)
#   MOCK_CREATE_FAILS   "yes" -> gh release create fails
#   MOCK_REF_UNREADABLE "yes" -> tag cannot be peeled to a commit
#   MOCK_TAG_TYPE       "tag" models an annotated tag object
#   MOCK_CONCURRENT_RELEASE "yes" -> losing tag claim observes a new Release
#   MOCK_CREATE_RACE    "yes" -> create loses to a concurrent Release writer
run_script() {
  local release_tag="${TEST_TAG:-v1.2.3}"
  export GH="$TMP/mock-gh"
  : > "$TMP/calls"
  cat > "$GH" <<MOCK
#!/usr/bin/env bash
set -euo pipefail
cmd="\$1"; shift
printf '%s %s\n' "\$cmd" "\$*" >> "$TMP/calls"
case "\$cmd" in
  release)
    sub="\$1"; shift
    if [ "\$sub" = "view" ]; then
      if [ "${MOCK_RELEASE_VIEW:-}" = "yes" ] || [ -s "$TMP/release-state" ]; then
        if [[ "\$*" == *"--json body"* ]]; then
          if [ "${MOCK_WRONG_TAG_MARKER:-}" = "yes" ]; then
            echo '<!-- rapid-mlx-auto-release:v9.9.9:$SHA_GOOD -->'
          elif [ "${MOCK_MARKER_NOT_FIRST:-}" = "yes" ]; then
            printf 'manual preface\n<!-- rapid-mlx-auto-release:$release_tag:$SHA_GOOD -->\n'
          elif [ "${MOCK_RELEASE_MARKED:-}" = "yes" ] || \
             [ "\$(cat "$TMP/release-state" 2>/dev/null)" = "draft-marked" ]; then
            echo '<!-- rapid-mlx-auto-release:$release_tag:$SHA_GOOD -->'
          else
            echo 'manually staged draft notes'
          fi
          exit 0
        fi
        if [ "${MOCK_RELEASE_DRAFT:-}" = "yes" ] || \
           [[ "\$(cat "$TMP/release-state" 2>/dev/null)" == draft* ]]; then
          printf '$release_tag\ttrue\n'
        else
          printf '$release_tag\tfalse\n'
        fi
      else
        exit 1
      fi
    elif [ "\$sub" = "create" ]; then
      if [ "${MOCK_CREATE_RACE:-}" = "yes" ]; then
        echo yes > "$TMP/release-state"
        exit 1
      fi
      if [ "${MOCK_CREATE_FAILS:-}" = "yes" ]; then exit 1; fi
      NOTES_ARG=""
      while [ "\$#" -gt 0 ]; do
        if [ "\$1" = "--notes-file" ] && [ "\$#" -ge 2 ]; then
          NOTES_ARG="\$2"
          break
        fi
        shift
      done
      if [ -n "\$NOTES_ARG" ] && grep -qF \
        '<!-- rapid-mlx-auto-release:$release_tag:$SHA_GOOD -->' "\$NOTES_ARG"; then
        echo draft-marked > "$TMP/release-state"
      else
        echo draft > "$TMP/release-state"
      fi
      echo "created-$release_tag" >&2
    elif [ "\$sub" = "edit" ]; then
      echo published > "$TMP/release-state"
    else
      exit 2
    fi
    ;;
  api)
    if [ "\$1" = "-X" ] && [ "\$2" = "POST" ]; then
      # Atomic claim: on success the tag now exists at the claimed SHA.
      if [ "${MOCK_CLAIM_FAILS:-}" = "yes" ]; then
        if [ "${MOCK_CONCURRENT_RELEASE:-}" = "yes" ]; then
          echo yes > "$TMP/release-state"
        fi
        exit 1
      fi
      SUBMITTED_REF=""
      SUBMITTED_SHA=""
      shift 3
      while [ "\$#" -gt 0 ]; do
        if [ "\$1" = "-f" ] && [ "\$#" -ge 2 ]; then
          case "\$2" in
            ref=*) SUBMITTED_REF="\${2#ref=}" ;;
            sha=*) SUBMITTED_SHA="\${2#sha=}" ;;
          esac
          shift 2
        else
          shift
        fi
      done
      [ "\$SUBMITTED_REF" = "refs/tags/$release_tag" ] || exit 3
      [ "\$SUBMITTED_SHA" = "$SHA_GOOD" ] || exit 3
      echo "\$SUBMITTED_SHA" > "$TMP/ref-state"
      echo '{"ref":"refs/tags/$release_tag"}'
    elif [[ "\$1" == repos/o/r/git/tags/* ]]; then
      if [ "${MOCK_REF_UNREADABLE:-}" = "yes" ]; then exit 1; fi
      printf 'commit\t%s\n' "${MOCK_TAG_REF:-$SHA_GOOD}"
    else
      # Resolve the explicit tag ref. Annotated tags are peeled by a second
      # git/tags/<object> request above.
      if [ "${MOCK_REF_UNREADABLE:-}" = "yes" ]; then exit 1; fi
      if [ -s "$TMP/ref-state" ]; then
        printf 'commit\t%s\n' "\$(cat "$TMP/ref-state")"
        exit 0
      fi
      if [ -n "${MOCK_TAG_REF:-}" ]; then
        if [ "${MOCK_TAG_TYPE:-commit}" = "tag" ]; then
          printf 'tag\tannotated-object-sha\n'
        else
          printf 'commit\t%s\n' "\$MOCK_TAG_REF"
        fi
        exit 0
      fi
      exit 1
    fi
    ;;
  *)
    exit 2
    ;;
esac
MOCK
  chmod +x "$GH"
  : > "$TMP/ref-state"
  : > "$TMP/release-state"
  printf 'notes for v1.2.3\n' > "$TMP/notes.md"
  # Env-prefix vars on the run_script call are function-scoped, not exported,
  # so push the mock knobs into the child explicitly.
  export MOCK_RELEASE_VIEW MOCK_CLAIM_FAILS MOCK_TAG_REF MOCK_CREATE_FAILS
  export MOCK_RELEASE_DRAFT MOCK_RELEASE_MARKED MOCK_REF_UNREADABLE MOCK_TAG_TYPE
  export MOCK_CONCURRENT_RELEASE MOCK_CREATE_RACE
  if GH_TOKEN=test TAG="$release_tag" RELEASE_SHA="$SHA_GOOD" \
     NOTES_FILE="$TMP/notes.md" GITHUB_REPOSITORY="o/r" \
     bash "$SCRIPT" >"$TMP/out" 2>"$TMP/err"; then
    echo "0"
  else
    echo "1"
  fi
}

run_case() (
  export MOCK_RELEASE_VIEW=no MOCK_RELEASE_DRAFT=no MOCK_RELEASE_MARKED=no
  export MOCK_MARKER_NOT_FIRST=no
  export MOCK_WRONG_TAG_MARKER=no
  export MOCK_CLAIM_FAILS=no MOCK_TAG_REF="" MOCK_TAG_TYPE=commit
  export MOCK_CREATE_FAILS=no MOCK_CREATE_RACE=no MOCK_REF_UNREADABLE=no
  export MOCK_CONCURRENT_RELEASE=no
  export TEST_TAG=v1.2.3
  for setting in "$@"; do
    export "$setting"
  done
  run_script
)

call_order() { grep '^' "$TMP/calls" 2>/dev/null || true; }

# ==========================================================================
echo "== 1. TOCTOU closed: tag is claimed atomically BEFORE release create =="
# ==========================================================================
RC=$(run_case)
[ "$RC" = "0" ] && ok "fresh release succeeds" || bad "fresh release succeeds (rc=$RC)"
POS=$(grep -n '^api -X POST repos/o/r/git/refs' "$TMP/calls" | head -1 | cut -d: -f1)
CRT=$(grep -n '^release create' "$TMP/calls" | head -1 | cut -d: -f1)
[ -n "$POS" ] && [ -n "$CRT" ] && [ "$POS" -lt "$CRT" ] && \
  ok "refs POST precedes gh release create" || bad "refs POST precedes gh release create"
contains "$(cat "$TMP/out")" "released v1.2.3 at $SHA_GOOD" "success banner printed"
contains "$(call_order)" "release create v1.2.3 --title v1.2.3" \
  "Release is created only after the tag claim"
CREATE_CALL=$(grep '^release create' "$TMP/calls" | head -1)
contains "$CREATE_CALL" "--draft" "release create starts as a draft"
contains "$(call_order)" \
  "release edit v1.2.3 --title v1.2.3 --notes-file $TMP/notes.md --draft=false" \
  "verified draft is explicitly published"

RC=$(run_case TEST_TAG=v1.2.3-rc1)
[ "$RC" = "0" ] && ok "fresh RC release succeeds" || bad "fresh RC release succeeds (rc=$RC)"
contains "$(grep '^release create' "$TMP/calls" | head -1)" "--prerelease" \
  "RC Release draft is explicitly marked prerelease"
contains "$(grep '^release edit' "$TMP/calls" | tail -1)" "--prerelease" \
  "RC Release remains prerelease when published"

# Negative control: prove the mock fails if production submits the wrong SHA,
# rather than hard-coding a successful claim independently of request args.
GOOD_SCRIPT="$SCRIPT"
BROKEN_SCRIPT="$TMP/create-release-wrong-sha.sh"
sed 's/-f "sha=$RELEASE_SHA"/-f "sha=wrong-sha"/' "$GOOD_SCRIPT" > "$BROKEN_SCRIPT"
SCRIPT="$BROKEN_SCRIPT"
RC=$(run_case)
SCRIPT="$GOOD_SCRIPT"
[ "$RC" = "1" ] && ok "mock rejects an incorrect claimed SHA" \
  || bad "mock rejects an incorrect claimed SHA (rc=$RC)"
lacks "$(call_order)" "release create" "wrong claim never reaches Release creation"

# ==========================================================================
echo "== 2. tag-without-Release recovery: claim fails, tag matches -> recover =="
# ==========================================================================
RC=$(run_case MOCK_CLAIM_FAILS=yes MOCK_TAG_REF="$SHA_GOOD")
[ "$RC" = "0" ] && ok "recovery succeeds when existing tag matches RELEASE_SHA" \
  || bad "recovery succeeds when existing tag matches RELEASE_SHA (rc=$RC)"
contains "$(cat "$TMP/out")" "already exists and points at the release commit" "reuse banner printed"
contains "$(cat "$TMP/out")" "released v1.2.3" "release still created"

RC=$(run_case MOCK_CLAIM_FAILS=yes MOCK_TAG_REF="$SHA_GOOD" MOCK_TAG_TYPE=tag)
[ "$RC" = "0" ] && ok "annotated tag is peeled to its commit" \
  || bad "annotated tag is peeled to its commit (rc=$RC)"
contains "$(call_order)" "git/ref/tags/v1.2.3" \
  "verification starts from the explicit tag namespace"

RC=$(run_case MOCK_CLAIM_FAILS=yes MOCK_TAG_REF="$SHA_GOOD" \
  MOCK_CONCURRENT_RELEASE=yes)
[ "$RC" = "0" ] && ok "concurrent Release after lost claim is idempotent" \
  || bad "concurrent Release after lost claim is idempotent (rc=$RC)"
lacks "$(call_order)" "release create" \
  "lost-claim recovery does not race a concurrent Release create"

# ==========================================================================
echo "== 3. stale tag mismatch: claim fails, tag points elsewhere -> hard fail =="
# ==========================================================================
RC=$(run_case MOCK_CLAIM_FAILS=yes MOCK_TAG_REF="$SHA_BAD")
[ "$RC" = "1" ] && ok "mismatched tag fails loudly" || bad "mismatched tag fails loudly (rc=$RC)"
lacks "$(call_order)" "release create" "no gh release create on a stale tag"

# ==========================================================================
echo "== 4. idempotent: Release already exists -> done, no claim/create =="
# ==========================================================================
RC=$(run_case MOCK_RELEASE_VIEW=yes MOCK_TAG_REF="$SHA_GOOD")
[ "$RC" = "0" ] && ok "existing Release is idempotent-done" || bad "existing Release is idempotent-done (rc=$RC)"
lacks "$(call_order)" "api -X POST" "no tag claim when Release already exists"
lacks "$(call_order)" "release create" "no release create when Release already exists"

RC=$(run_case MOCK_RELEASE_VIEW=yes MOCK_RELEASE_DRAFT=yes \
  MOCK_RELEASE_MARKED=yes MOCK_TAG_REF="$SHA_GOOD")
[ "$RC" = "0" ] && ok "matching draft Release is resumed" \
  || bad "matching draft Release is resumed (rc=$RC)"
contains "$(call_order)" \
  "release edit v1.2.3 --title v1.2.3 --notes-file $TMP/notes.md --draft=false" \
  "recovery replaces draft title/notes before publishing"
contains "$(cat "$TMP/out")" "released v1.2.3 at $SHA_GOOD" \
  "resumed draft reports published success"

RC=$(run_case MOCK_RELEASE_VIEW=yes MOCK_RELEASE_DRAFT=yes \
  MOCK_RELEASE_MARKED=no MOCK_TAG_REF="$SHA_GOOD")
[ "$RC" = "1" ] && ok "manual draft is never auto-published" \
  || bad "manual draft is never auto-published (rc=$RC)"
lacks "$(call_order)" "release edit" "manual draft remains untouched"

RC=$(run_case MOCK_RELEASE_VIEW=yes MOCK_RELEASE_DRAFT=yes \
  MOCK_MARKER_NOT_FIRST=yes MOCK_TAG_REF="$SHA_GOOD")
[ "$RC" = "1" ] && ok "marker outside the first line does not claim ownership" \
  || bad "marker outside the first line does not claim ownership (rc=$RC)"
lacks "$(call_order)" "release edit" "embedded marker cannot publish manual draft"

RC=$(run_case MOCK_RELEASE_VIEW=yes MOCK_RELEASE_DRAFT=yes \
  MOCK_WRONG_TAG_MARKER=yes MOCK_TAG_REF="$SHA_GOOD")
[ "$RC" = "1" ] && ok "same-SHA marker copied from another tag is rejected" \
  || bad "same-SHA marker copied from another tag is rejected (rc=$RC)"
lacks "$(call_order)" "release edit" "cross-tag copied draft remains untouched"

# Existing Release idempotency must remain fail-closed when its tag cannot be
# resolved.  Otherwise a transient API/auth failure can falsely bless a
# Release whose shipped tree was never verified.
RC=$(run_case MOCK_RELEASE_VIEW=yes MOCK_TAG_REF="$SHA_GOOD" \
  MOCK_REF_UNREADABLE=yes)
[ "$RC" = "1" ] && ok "existing Release with unreadable tag fails closed" \
  || bad "existing Release with unreadable tag fails closed (rc=$RC)"
lacks "$(call_order)" "release create" "unverifiable existing Release is not recreated"

# ==========================================================================
echo "== 5. release create failure leaves tag claimed (recovery reaches it) =="
# ==========================================================================
RC=$(run_case MOCK_CREATE_FAILS=yes)
[ "$RC" = "1" ] && ok "release create failure propagates" || bad "release create failure propagates (rc=$RC)"
POS=$(grep -n '^api -X POST' "$TMP/calls" | head -1 | cut -d: -f1)
CRT=$(grep -n '^release create' "$TMP/calls" | head -1 | cut -d: -f1)
[ -n "$POS" ] && [ -n "$CRT" ] && [ "$POS" -lt "$CRT" ] && \
  ok "tag still claimed before the failed create (recoverable next run)" || \
  bad "tag still claimed before the failed create (recoverable next run)"

RC=$(run_case MOCK_CREATE_RACE=yes)
[ "$RC" = "0" ] && ok "concurrent Release that wins create race is idempotent" \
  || bad "concurrent Release that wins create race is idempotent (rc=$RC)"
contains "$(cat "$TMP/out")" "already exists at the release commit" \
  "create-race success is reported after verification"

# ==========================================================================
echo
echo "tests/release/test_create_release.sh: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
