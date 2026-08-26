#!/usr/bin/env bash
#
# Offline promotion contract for the Desktop RC tag (#2301).
#
# Proves, without cutting a release or touching GitHub, that the candidate-
# acceptance → manifest → tag identity chain is bound end-to-end: the tag may
# only be claimed at the exact commit whose desktop artefact the signed/notarised
# lane validated, and that commit must also be the live main head. It asserts
# against the actual workflow wiring (auto-release.yml + rapid-mac-release.yml +
# the shared desktop-releasable action) and reuses a mock ``gh`` to prove the tag
# claim itself never moves/deletes an old RC.
#
#   ./tests/release/test_desktop_release_promotion.sh
#
# Requires: bash, python3, grep. No network/real gh/git mutation.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="$REPO_ROOT/scripts/tag_desktop_app.sh"
AUTO_RELEASE="$REPO_ROOT/.github/workflows/auto-release.yml"
RAPID_RELEASE="$REPO_ROOT/.github/workflows/rapid-mac-release.yml"
ACTION="$REPO_ROOT/.github/actions/desktop-releasable/action.yml"
CHECK_MAIN_HEAD="$REPO_ROOT/scripts/check_main_head.py"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
contains() { if grep -qF -- "$2" <<<"$1"; then ok "$3"; else bad "$3"; printf '        want substring: %s\n        got:            %s\n' "$2" "$1"; fi; }
lacks()    { if grep -qF -- "$2" <<<"$1"; then bad "$3"; else ok "$3"; fi; }

A="1111111111111111111111111111111111111111"  # bump commit (candidate)
B="2222222222222222222222222222222222222222"  # packaging fix that lands on main
TAG_OBJ="3333333333333333333333333333333333333333"

# ---------------------------------------------------------------------------
echo "== 1. candidate gate binds accepted SHA + manifest source + main head =="
# ---------------------------------------------------------------------------
# The candidate gate accepts only when the manifest's source_sha equals the
# built SHA AND the triggering commit (auto-release.yml "Accept the desktop
# candidate"). Pull that binding out of the shared action + workflow text.
ACCEPT=$(sed -n '/name: Accept the desktop candidate/,/^      - name: Upload desktop/p' "$AUTO_RELEASE")
contains "$ACCEPT" 'desktop_manifest.py verify' "candidate re-verifies the manifest against the DMG"
contains "$ACCEPT" 'manifest source ($SRC)' "candidate refuses a source mismatch"
contains "$ACCEPT" 'if [ "$SRC" != "$BUILT_SHA" ] || [ "$SRC" != "$EXPECTED" ]' \
  "candidate requires manifest source == built SHA == triggering commit"
contains "$ACCEPT" 'echo "accepted_sha=$BUILT_SHA"' "accepted SHA is the built/validated commit"

# The manifest itself binds source_sha + embedded version + app tag.
MANIFEST_ACTION=$(sed -n '/name: Create + verify the Desktop release manifest/,/name: Clean up/p' "$ACTION")
contains "$MANIFEST_ACTION" '--source-sha "$SOURCE_SHA"' "manifest records the candidate source SHA"
contains "$MANIFEST_ACTION" '--version "$RELEASE_VERSION"' "manifest records the release version"
contains "$MANIFEST_ACTION" '--signed' "manifest records the signed/delta gate state"

# The shared candidate/tag contract must distinguish a verified missing asset
# (the only safe no-baseline skip) from auth/API/malformed lookup failures.
DELTA_GATE=$(sed -n '/name: Bundle size delta gate/,/name: Validate DMG contents/p' "$ACTION")
contains "$DELTA_GATE" 'scripts/release_asset_size.sh' \
  "DMG delta gate uses the executable fail-closed asset resolver"
contains "$DELTA_GATE" 'scripts/check_dmg_size_delta.sh' \
  "DMG delta gate compares current and previous sizes in one byte unit"
lacks "$DELTA_GATE" 'du -sm "$DMG"' \
  "DMG delta gate does not mix du MiB with release asset bytes"
lacks "$DELTA_GATE" 'b/1000000' \
  "DMG delta gate does not convert only the previous size to decimal MB"
lacks "$DELTA_GATE" 'gh release view' \
  "DMG delta gate does not mask an inline gh lookup"
lacks "$DELTA_GATE" '|| true' \
  "DMG baseline lookup never turns API failure into a skip"
contains "$DELTA_GATE" 'if [[ "$ASSET_STATE" == "absent" ]]' \
  "only a verified absent canonical asset skips the comparison"

# ---------------------------------------------------------------------------
echo "== 2. live main head must equal validated candidate (TOCTOU) =="
# ---------------------------------------------------------------------------
# release-prep re-resolves refs/heads/main and requires main == accepted ==
# release; the protected tag job re-queries immediately before claiming.
PREP=$(sed -n '/name: Verify the live main head still equals/,/name: Release-blocker evidence/p' "$AUTO_RELEASE")
contains "$PREP" 'git/ref/heads/main' "release-prep resolves the live main head"
contains "$PREP" 'check_main_head.py' "release-prep uses the structured main-head gate"
contains "$PREP" '--accepted-sha "$ACCEPTED_SHA"' "release-prep binds accepted SHA to main head"
contains "$PREP" '--release-sha "$RELEASE_SHA"' "release-prep binds release SHA to main head"
contains "$PREP" 'tee main-head-evidence.txt' "pre-approval evidence is captured"

REQUERY=$(sed -n '/name: Re-query live main head immediately before tag/,/name: Tag the desktop app/p' "$AUTO_RELEASE")
contains "$REQUERY" 'git/ref/heads/main' "tag job re-resolves main head immediately before tag"
contains "$REQUERY" 'check_main_head.py' "tag job reuses the same main-head gate"
contains "$REQUERY" '--main-sha "$MAIN_SHA"' "tag job passes the freshly resolved head"
lacks "$AUTO_RELEASE" 'TAG_APPROVED: "true"' "no self-asserted approval boolean anywhere"

# Offline behaviour of the gate itself: unchanged head passes, head advanced
# (the exact #2301 reproduction) fails closed, malformed fails.
if python3 "$CHECK_MAIN_HEAD" --main-sha "$A" --accepted-sha "$A" --release-sha "$A" >/dev/null 2>&1; then
  ok "unchanged main head A == candidate passes"
else
  bad "unchanged main head A == candidate passes"
fi
if python3 "$CHECK_MAIN_HEAD" --main-sha "$B" --accepted-sha "$A" --release-sha "$A" >/dev/null 2>&1; then
  bad "packaging fix B landing on main while A validates -> refuses (behind head)"
else
  ok "packaging fix B landing on main while A validates -> refuses (behind head)"
fi
if python3 "$CHECK_MAIN_HEAD" --main-sha "short" --accepted-sha "$A" --release-sha "$A" >/dev/null 2>&1; then
  bad "malformed main SHA fails closed"
else
  ok "malformed main SHA fails closed"
fi

# ---------------------------------------------------------------------------
echo "== 3. tag only claimed at the validated SHA; rc1 never moved/deleted =="
# ---------------------------------------------------------------------------
# A mock gh that returns nothing (claim fails) then reads back an EXISTING rc1
# tag at a DIFFERENT commit must make the script refuse — never --force/move.
cat > "$TMP/gh" <<'MOCK'
#!/usr/bin/env bash
if [ "$1 $2 $3" = "api -X POST" ]; then
  echo "HTTP 422: Reference already exists" >&2; exit 1
fi
if [[ "$*" == *"git/ref/tags/"* ]]; then
  printf 'commit\t%s\n' "$READ_SHA"; exit 0
fi
exit 0
MOCK
chmod +x "$TMP/gh"
OUT=$(
  HAVE_PAT=true GH="$TMP/gh" GITHUB_REPOSITORY="raullenchai/Rapid-MLX" \
  CALLS="$TMP/calls" VERSION="0.13.0-rc1" READ_SHA="4444444444444444444444444444444444444444" \
  RELEASE_SHA="$B" ACCEPTED_SHA="$B" \
    bash "$SCRIPT" 2>&1
) && RC=0 || RC=$?
[ "${RC:-0}" -ne 0 ] && ok "existing rc1 at a different commit -> nonzero (never moves)" \
                     || bad "existing rc1 at a different commit -> nonzero (never moves)"
contains "$OUT" "NEXT rc" "remediation directs to cut the NEXT rc, not delete/move"
lacks "$OUT" "--force" "never forces a tag move"
lacks "$OUT" "git tag -d" "never recommends deleting a published RC"
lacks "$OUT" "force-push" "never recommends a force push"

# ---------------------------------------------------------------------------
echo "== 4. tag step wiring in auto-release =="
# ---------------------------------------------------------------------------
TAGSTEP=$(sed -n '/name: Tag the desktop app at the exact validated SHA/,/name: Create tag and release/p' "$AUTO_RELEASE")
contains "$TAGSTEP" 'ACCEPTED_SHA: ${{ needs.release-prep.outputs.accepted_sha }}' \
  "tag step claims at the pre-approval accepted SHA"
contains "$TAGSTEP" 'scripts/tag_desktop_app.sh' "tag step invokes the tagged script"
lacks "$TAGSTEP" "git push" "tag step never git pushes"

# A same-SHA tag claim may no-op after a prior failed/missed Desktop workflow.
# The engine must therefore wait for exact tagged Desktop publication evidence,
# not treat ref identity alone as proof that the DMG shipped.
PUBLISH_WAIT=$(sed -n '/name: Wait for exact Desktop tagged publication/,/name: Create tag and release/p' "$AUTO_RELEASE")
contains "$PUBLISH_WAIT" 'scripts/check_desktop_publish.py' \
  "engine release waits on the tested Desktop publication helper"
contains "$PUBLISH_WAIT" 'APP_TAG: rapid-mac-v${{ needs.detect.outputs.version }}' \
  "publication wait binds the exact Desktop tag"
contains "$PUBLISH_WAIT" 'ACCEPTED_SHA: ${{ needs.release-prep.outputs.accepted_sha }}' \
  "publication wait binds the validated candidate SHA"
contains "$PUBLISH_WAIT" '--workflow rapid-mac-release.yml' \
  "publication wait binds the expected Desktop workflow"
contains "$PUBLISH_WAIT" 'timeout-minutes: 360' \
  "publication wait covers the bounded child release critical path"
contains "$PUBLISH_WAIT" '--deadline-min 350' \
  "publication poll leaves a ten-minute parent-job diagnostic margin"
TAG_LINE=$(grep -n 'name: Tag the desktop app at the exact validated SHA' "$AUTO_RELEASE" | cut -d: -f1)
WAIT_LINE=$(grep -n 'name: Wait for exact Desktop tagged publication' "$AUTO_RELEASE" | cut -d: -f1)
ENGINE_LINE=$(grep -n 'name: Create tag and release' "$AUTO_RELEASE" | cut -d: -f1)
if [[ -n "$TAG_LINE" && -n "$WAIT_LINE" && -n "$ENGINE_LINE" \
      && "$TAG_LINE" -lt "$WAIT_LINE" && "$WAIT_LINE" -lt "$ENGINE_LINE" ]]; then
  ok "tag claim -> Desktop publication evidence -> engine release ordering"
else
  bad "tag claim -> Desktop publication evidence -> engine release ordering"
fi
PUBLISH_STEP=$(sed -n '/name: Create the GitHub Release/,/echo "::notice::Published GitHub Release/p' "$RAPID_RELEASE")
contains "$PUBLISH_STEP" 'asset.get("digest") != f"sha256:{digest}"' \
  "tagged rerun requires an existing canonical DMG to be byte-identical"
lacks "$PUBLISH_STEP" '--clobber' \
  "tagged rerun never replaces an already-published Desktop DMG"
UPDATER_STEP=$(sed -n '/name: Publish updater fallback monotonically/,/name: Create the GitHub Release/p' "$RAPID_RELEASE")
contains "$UPDATER_STEP" 'if [[ "$CURRENT_VERSION" == "$TARGET_VERSION" ]]' \
  "equal-version rerun branches before the first mutable updater write"
contains "$UPDATER_STEP" 'scripts/check_equal_version_republish.py' \
  "equal-version rerun verifies pointer, exact DMG and existing Release identity"
contains "$UPDATER_STEP" 'exit 0' \
  "identical equal-version rerun no-ops the mutable updater transaction"
contains "$UPDATER_STEP" 'monotonic ordering is unprovable' \
  "malformed current pointer fails closed after Release identity verification"
IDENTITY_LINE=$(grep -n 'scripts/check_equal_version_republish.py' "$RAPID_RELEASE" | head -1 | cut -d: -f1)
POINTER_GET_LINE=$(grep -n 'r2 object get "${R2_BUCKET}/latest.json"' "$RAPID_RELEASE" | cut -d: -f1)
LATEST_PUT_LINE=$(grep -n 'r2 object put "${R2_BUCKET}/latest.json"' "$RAPID_RELEASE" | tail -1 | cut -d: -f1)
APPCAST_PUT_LINE=$(grep -n 'r2 object put "${R2_BUCKET}/appcast.xml"' "$RAPID_RELEASE" | tail -1 | cut -d: -f1)
COMPAT_PUT_LINE=$(grep -n '"${R2_BUCKET}/rapid-mac/rapid-mlx-desktop.dmg"' "$RAPID_RELEASE" | tail -1 | cut -d: -f1)
if [[ -n "$IDENTITY_LINE" && "$IDENTITY_LINE" -lt "$POINTER_GET_LINE" \
      && "$IDENTITY_LINE" -lt "$LATEST_PUT_LINE" \
      && "$IDENTITY_LINE" -lt "$APPCAST_PUT_LINE" \
      && "$IDENTITY_LINE" -lt "$COMPAT_PUT_LINE" ]]; then
  ok "canonical Release identity gate precedes pointer branching and every mutable write"
else
  bad "canonical Release identity gate precedes pointer branching and every mutable write"
fi

RESOLVER_USES=$(grep -c 'resolve_github_tag_commit "$TAG"' "$RAPID_RELEASE")
[[ "$RESOLVER_USES" == 2 ]] \
  && ok "both tagged-lane tag checks use the bounded recursive resolver" \
  || bad "both tagged-lane tag checks use the bounded recursive resolver (got $RESOLVER_USES)"
lacks "$RAPID_RELEASE" 'git/refs/tags/${TAG}' \
  "tagged lane has no remaining one-level tag resolver"

BUILD_TIMEOUT=$(sed -n '/^  build:/,/^  mirror-dist:/p' "$RAPID_RELEASE" | awk '/timeout-minutes:/ {print $2; exit}')
MIRROR_TIMEOUT=$(sed -n '/^  mirror-dist:/,/^  publish-updater-fallback:/p' "$RAPID_RELEASE" | awk '/timeout-minutes:/ {print $2; exit}')
PUBLISH_TIMEOUT=$(sed -n '/^  publish-updater-fallback:/,$p' "$RAPID_RELEASE" | awk '/timeout-minutes:/ {print $2; exit}')
if [[ "$BUILD_TIMEOUT" == 120 && "$MIRROR_TIMEOUT" == 30 && "$PUBLISH_TIMEOUT" == 120 \
      && $((BUILD_TIMEOUT + MIRROR_TIMEOUT + PUBLISH_TIMEOUT)) -lt 350 ]]; then
  ok "parent poll covers child build, mirror, turnstyle and publication budgets"
else
  bad "parent poll covers child build, mirror, turnstyle and publication budgets"
fi

# ---------------------------------------------------------------------------
echo "== 5. tagged lane verifies tag binding before any publication =="
# ---------------------------------------------------------------------------
BIND=$(sed -n '/name: Verify tag binding before any publication/,/name: Upload workflow artifact/p' "$RAPID_RELEASE")
contains "$BIND" 'tag/checkout/built/manifest source SHAs disagree' \
  "tagged lane fails closed on any SHA disagreement before upload"
contains "$BIND" 'exit 1' "refusal is a hard failure, not a notice"
lacks "$BIND" "actions/upload-artifact" "binding step itself never uploads an artifact"

# ---------------------------------------------------------------------------
echo "== 6. workflow_dispatch dry-run preserved (branch, no tag) =="
# ---------------------------------------------------------------------------
CONS=$(sed -n '/name: Compute app version and previous rapid-mac release tag/,/uses: \.\/\.github\/actions\/desktop-releasable/p' "$RAPID_RELEASE")
contains "$CONS" 'workflow_dispatch dry-run' "branch dispatch is a recognised dry-run"
contains "$CONS" 'CFBundleShortVersionString' "dry-run version derives from the source plist"
contains "$CONS" 'preceding-release' "Sparkle predecessor (RC-inclusive) resolved"
contains "$CONS" 'preceding "${APP_VERSION}"' "DMG stable predecessor resolved"
PUBLISH_SPARKLE=$(sed -n '/uses: \.\/\.github\/actions\/desktop-releasable/,/SPARKLE_ED_PRIVATE_KEY/p' "$RAPID_RELEASE")
contains "$PUBLISH_SPARKLE" 'publish_sparkle: ${{ steps.appmeta.outputs.is_tag == '\''true'\'' }}' \
  "Sparkle generation gated to tag runs only (dry-run builds no Sparkle)"
# The Sparkle UPLOAD step must be tag-gated too: a signed branch dry-run sets
# SIGNED=true but generates no Sparkle files, so an upload keyed on SIGNED alone
# would fail on if-no-files-found. Both generation and upload are tag-only.
SPARKLE_UPLOAD=$(sed -n '/name: Upload Sparkle update artifact/,/retention-days/p' "$RAPID_RELEASE")
contains "$SPARKLE_UPLOAD" "steps.appmeta.outputs.is_tag == 'true' && env.SIGNED == 'true'" \
  "Sparkle upload gated to tag runs only (signed dry-run has no Sparkle files)"
contains "$SPARKLE_UPLOAD" "if-no-files-found: error" "upload still fails loud if a tag run is missing Sparkle"

# #2301: the CFBundleVersion monotonicity check (RC-inclusive predecessor, so
# rc2 beats rc1) must run in the CANDIDATE lane too — gated on signed +
# previous_release_tag, NOT on Sparkle publication. Otherwise the tag-triggered
# lane could discover a build regression only AFTER the immutable tag exists.
MONOTONIC=$(sed -n '/name: Enforce CFBundleVersion monotonicity vs prior release/,/name: Build signed Sparkle archive and appcast/p' "$ACTION")
contains "$MONOTONIC" "inputs.signed == 'true' && inputs.previous_release_tag != ''" \
  "CFBundleVersion monotonicity runs for signed builds with a prior release, independent of Sparkle publication"
contains "$MONOTONIC" "inputs.previous_release_tag" \
  "monotonicity consumes the RC-inclusive predecessor"
lacks "$MONOTONIC" "publish_sparkle" \
  "monotonicity gate does NOT require Sparkle publication (candidate lane enforces it pre-tag)"
contains "$MONOTONIC" "must exceed" \
  "monotonicity fails closed when this build does not exceed the prior release's build"

# #2301: a REAL tag run must be signed (the approved evidence is the signed
# candidate). require_signed is bound to is_tag, so drift to ad-hoc on a tag run
# fails the lane closed before it can mirror/publish; only a branch dispatch
# dry-run stays ad-hoc-capable.
SIGN_REQ=$(sed -n '/name: Build + sign + notarise + DMG-validate (shared contract)/,/SPARKLE_ED_PRIVATE_KEY/p' "$RAPID_RELEASE")
contains "$SIGN_REQ" 'require_signed: ${{ steps.appmeta.outputs.is_tag == '\''true'\'' }}' \
  "tag lane requires signed (require_signed bound to is_tag); dispatch dry-run stays ad-hoc"
contains "$SIGN_REQ" "signed: \${{ secrets.MACOS_DEVID_CERT_P12_BASE64" \
  "tag lane still reads whether Apple secrets are present"

# ---------------------------------------------------------------------------
echo "== 6b. candidate gate runs IN PARALLEL with the Tier-1 gate (dependency) =="
# ---------------------------------------------------------------------------
# The Desktop candidate and the Tier-1 agent smoke share nothing (different
# runners, different checks) and both depend only on detect outputs. Sequenceing
# them (candidate needs tier1) would add the full Studio latency (~125 min) to
# every RC for no safety gain. The candidate gate therefore `needs: detect`
# ONLY — it starts at the same time as tier1 — while release-prep still `needs`
# BOTH and requires BOTH before the tag can be claimed. Parallelism must not let
# a release skip either gate.
CAND_NEEDS=$(sed -n '/^  desktop-candidate-gate:/,/^  # 3)/p' "$AUTO_RELEASE")
contains "$CAND_NEEDS" "needs: detect" \
  "desktop-candidate-gate needs only detect (starts in parallel with tier1)"
lacks "$CAND_NEEDS" "tier1-agent-gate" \
  "desktop-candidate-gate does NOT wait on the Tier-1 gate"
PREP_NEEDS=$(sed -n '/^  release-prep:/,/^    if:/p' "$AUTO_RELEASE")
contains "$PREP_NEEDS" "needs: [detect, tier1-agent-gate, desktop-candidate-gate]" \
  "release-prep needs AND requires BOTH the Tier-1 gate and the desktop candidate gate"
RELPREP_IF=$(sed -n '/^  release-prep:/,/runs-on:/p' "$AUTO_RELEASE")
contains "$RELPREP_IF" "needs.tier1-agent-gate.result == 'success'" \
  "release-prep requires Tier-1 success (or force)"
contains "$RELPREP_IF" "needs.desktop-candidate-gate.result == 'success'" \
  "release-prep requires desktop-candidate success"

# ---------------------------------------------------------------------------
echo "== 7b. normal workflow_dispatch retry after main drift (no bypass) =="
# ---------------------------------------------------------------------------
# A release can stall when main advances past the validated candidate (the pre-
# tag TOCTOU gate aborts). A maintainer needs a NORMAL re-run at the CURRENT
# new main head, separate from the emergency force override. retry_version:
# requires main, must equal pyproject, is mutually exclusive with force_version,
# and sets force=false so Tier-1 + signed candidate + evidence + approval all
# re-run. force=true is reserved for the emergency path only.
RETRY_BLOCK=$(sed -n '/retry_version:$/,/force_version:$/p' "$AUTO_RELEASE")
contains "$RETRY_BLOCK" "NORMAL retry" "retry_version input is a NORMAL (non-emergency) route"
contains "$RETRY_BLOCK" "Mutually exclusive with force_version" "retry_version documented as mutually exclusive with force_version"
DISPATCH=$(sed -n '/--- Manual workflow_dispatch/,/# --- Normal push path/p' "$AUTO_RELEASE")
contains "$DISPATCH" 'if [ -n "$FORCE_VERSION" ] && [ -n "$RETRY_VERSION" ]' \
  "dispatch refuses passing BOTH force_version and retry_version"
contains "$DISPATCH" 'Forced/retry release must be dispatched on main' \
  "retry requires dispatch on refs/heads/main"
contains "$DISPATCH" 'retry_version $VERSION != pyproject.toml' \
  "retry_version must equal pyproject.toml"
contains "$DISPATCH" 'RETRY release requested' "retry is recognised distinctly from force"
contains "$DISPATCH" 'if [ -n "$FORCE_VERSION" ]; then FORCE_BOOL=true; else FORCE_BOOL=false; fi' \
  "force=true is reserved for the emergency override"
lacks "$DISPATCH" 'echo "force=true" >>' "the retry path never emits a hardcoded force=true output"
# The downstream gate contract: force=false (retry) must make Tier-1 a hard
# requirement (never the force bypass).
PREP_IF=$(sed -n '/^  release-prep:/,/runs-on:/p' "$AUTO_RELEASE")
contains "$PREP_IF" "force == 'true' || needs.tier1-agent-gate.result == 'success'" \
  "with retry (force=false) Tier-1 success is REQUIRED, not bypassed"

# ---------------------------------------------------------------------------
echo "== 7. release job re-verifies LIVE environment protection before claim =="
# ---------------------------------------------------------------------------
# PF-3 read the rapid-mac-tag environment back on the bump PR, which is days
# before claim time. #2301 TOCTOU: the approval contract must STILL hold exactly
# when the tag is claimed. The release job live-fetches the environment +
# deployment-branch-policies and runs the SAME check_release_environment.py —
# positioned BEFORE the pre-tag blocker/main re-queries and any tag write — and
# carries the actions:read permission the environments read-back needs.
RELJOB=$(sed -n '/^  release:$/,/^name: Auto-release on version bump/p' "$AUTO_RELEASE")
contains "$RELJOB" 'actions: read' "release job has actions:read (environments read-back)"
RE_ENV=$(sed -n '/Verify live rapid-mac-tag environment protection/,/Download the release evidence/p' "$AUTO_RELEASE")
contains "$RE_ENV" 'repos/${REPO}/environments/${ENV_NAME}' \
  "release job live-fetches the rapid-mac-tag environment"
contains "$RE_ENV" 'deployment-branch-policies' \
  "release job live-fetches the plural deployment-branch-policies endpoint"
contains "$RE_ENV" 'check_release_environment.py' \
  "release job runs the SAME fail-closed environment checker live"
# Ordering: environment read-back step must precede the pre-tag blocker/main
# TOCTOU re-queries and every irreversible write.
ENV_POS=$(grep -n 'Verify live rapid-mac-tag environment protection' "$AUTO_RELEASE" | head -1 | cut -d: -f1)
BLOCKER_POS=$(grep -n 'Re-query release blockers immediately before tag' "$AUTO_RELEASE" | head -1 | cut -d: -f1)
MAINHEAD_POS=$(grep -n 'Re-query live main head immediately before tag' "$AUTO_RELEASE" | head -1 | cut -d: -f1)
TAGPOS=$(grep -n 'Tag the desktop app at the exact validated SHA' "$AUTO_RELEASE" | head -1 | cut -d: -f1)
if [ "${ENV_POS:-0}" -lt "${BLOCKER_POS:-0}" ] && [ "${ENV_POS:-0}" -lt "${MAINHEAD_POS:-0}" ] && [ "${ENV_POS:-0}" -lt "${TAGPOS:-0}" ]; then
  ok "live environment read-back runs BEFORE blocker/main re-queries and the tag claim"
else
  bad "live environment read-back runs BEFORE blocker/main re-queries and the tag claim (env=$ENV_POS blocker=$BLOCKER_POS main=$MAINHEAD_POS tag=$TAGPOS)"
fi
# Query-before-POST freshness ordering: BOTH the blocker and main-head re-queries
# must precede the tag-data POST step (the immutable claim), so the cutoff is
# established immediately before the write.
if [ "${BLOCKER_POS:-0}" -lt "${TAGPOS:-0}" ] && [ "${MAINHEAD_POS:-0}" -lt "${TAGPOS:-0}" ]; then
  ok "blocker + main-head freshness re-queries run BEFORE the tag claim (query-before-POST cutoff)"
else
  bad "blocker + main-head freshness re-queries run BEFORE the tag claim (blocker=$BLOCKER_POS main=$MAINHEAD_POS tag=$TAGPOS)"
fi

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
