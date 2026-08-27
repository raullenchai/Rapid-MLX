#!/usr/bin/env bash
#
# Offline contract for auto-release's maintainer-dispatched dry-run lane.
# The lane must exercise the two real pre-publication gates at the selected
# ref's exact SHA while making release-prep, the protected environment, tags,
# Releases, and updater publication unreachable.
# shellcheck disable=SC2016  # Assertions intentionally match literal ${{ }} / shell text.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKFLOW="$REPO_ROOT/.github/workflows/auto-release.yml"
RELEASE_DOC="$REPO_ROOT/RELEASE.md"
PROJECT_VERSION=$(awk -F'"' '/^version *=/ { print $2; exit }' "$REPO_ROOT/pyproject.toml")
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
contains() {
  if grep -qF -- "$2" <<<"$1"; then
    ok "$3"
  else
    bad "$3"
    printf '        want: %s\n' "$2"
  fi
}
lacks() {
  if grep -qF -- "$2" <<<"$1"; then
    bad "$3"
    printf '        reject: %s\n' "$2"
  else
    ok "$3"
  fi
}

echo "== 1. workflow_dispatch exposes a typed, non-publishing dry_run input =="
INPUTS=$(sed -n '/^  workflow_dispatch:/,/^permissions:/p' "$WORKFLOW")
DRY_INPUT=$(sed -n '/^      dry_run:$/,/^      force_version:$/p' "$WORKFLOW")
contains "$INPUTS" "dry_run:" "workflow_dispatch exposes dry_run"
contains "$DRY_INPUT" "type: boolean" "dry_run preserves GitHub's Boolean input type"
contains "$DRY_INPUT" "default: false" "normal dispatches do not become dry runs implicitly"
contains "$DRY_INPUT" "no tag, Release, updater, or protected environment" \
  "the input advertises its zero-publication boundary"

echo "== 2. detect routes dry_run through the real gates without a release route =="
DETECT_HEAD=$(sed -n '/^  detect:$/,/^    steps:/p' "$WORKFLOW")
DRY_ROUTE=$(sed -n '/if \[ "$DRY_RUN" = "true" \]; then/,/# force_version and retry_version/p' "$WORKFLOW")
contains "$DETECT_HEAD" 'dry_run: ${{ steps.detect.outputs.dry_run }}' \
  "detect exports dry_run for every downstream job"
contains "$(sed -n '/^  detect:$/,/^  # 2)/p' "$WORKFLOW")" \
  'DRY_RUN: ${{ inputs.dry_run }}' \
  "detect consumes the typed inputs context"
contains "$DRY_ROUTE" 'if [ "$DRY_RUN" = "true" ]' \
  "manual detection has an explicit dry-run route"
contains "$DRY_ROUTE" 'dry_run cannot be combined with force_version or retry_version' \
  "dry-run and publishing dispatch routes are mutually exclusive"
contains "$DRY_ROUTE" 'echo "should_release=true"' \
  "dry-run reaches both existing pre-publication gates"
contains "$DRY_ROUTE" 'echo "force=false"' \
  "dry-run never activates the Tier-1 emergency bypass"
contains "$DRY_ROUTE" 'echo "dry_run=true"' \
  "dry-run identity is explicit rather than inferred downstream"

DETECT_SCRIPT=$(awk '
  /- name: Detect version bump \(or forced release\)/ { found = 1 }
  found && $0 == "        run: |" { in_run = 1; next }
  in_run && /^  # 2\)/ { exit }
  in_run { sub(/^          /, ""); print }
' "$WORKFLOW")
DRY_OUTPUT="$TMP_DIR/dry-output"
if (
  cd "$REPO_ROOT"
  EVENT=workflow_dispatch \
  DRY_RUN=true \
  FORCE_VERSION='' \
  RETRY_VERSION='' \
  REASON='' \
  ACTOR=offline-contract \
  GITHUB_SHA=0123456789abcdef0123456789abcdef01234567 \
  GITHUB_REF=refs/heads/test-dry-run \
  GITHUB_OUTPUT="$DRY_OUTPUT" \
  bash -c "$DETECT_SCRIPT"
); then
  if grep -qxF "version=$PROJECT_VERSION" "$DRY_OUTPUT" \
    && grep -qxF 'should_release=true' "$DRY_OUTPUT" \
    && grep -qxF 'force=false' "$DRY_OUTPUT" \
    && grep -qxF 'dry_run=true' "$DRY_OUTPUT"; then
    ok "extracted detect script routes dry_run=true to both non-forced gates"
  else
    bad "extracted detect script routes dry_run=true to both non-forced gates"
    sed 's/^/        output: /' "$DRY_OUTPUT"
  fi
else
  bad "extracted detect script routes dry_run=true to both non-forced gates"
fi

MIXED_OUTPUT="$TMP_DIR/mixed-output"
if (
  cd "$REPO_ROOT"
  EVENT=workflow_dispatch \
  DRY_RUN=true \
  FORCE_VERSION="$PROJECT_VERSION" \
  RETRY_VERSION='' \
  REASON='' \
  ACTOR=offline-contract \
  GITHUB_SHA=0123456789abcdef0123456789abcdef01234567 \
  GITHUB_REF=refs/heads/test-dry-run \
  GITHUB_OUTPUT="$MIXED_OUTPUT" \
  bash -c "$DETECT_SCRIPT" >/dev/null 2>&1
); then
  bad "extracted detect script rejects dry_run plus a publishing input"
else
  ok "extracted detect script rejects dry_run plus a publishing input"
fi

echo "== 3. both production jobs are unreachable in dry-run mode =="
PREP_IF=$(sed -n '/^  release-prep:$/,/runs-on:/p' "$WORKFLOW")
RELEASE_IF=$(sed -n '/^  release:$/,/runs-on:/p' "$WORKFLOW")
contains "$PREP_IF" "needs.detect.outputs.dry_run != 'true'" \
  "dry-run skips release-prep before any publication evidence path"
contains "$RELEASE_IF" "needs.detect.outputs.dry_run != 'true'" \
  "dry-run skips the environment-gated release job"

echo "== 4. the final dry-run summary fails closed on either gate =="
SUMMARY=$(sed -n '/^  dry-run-summary:$/,/^  # 3)/p' "$WORKFLOW")
contains "$SUMMARY" "needs: [detect, tier1-agent-gate, desktop-candidate-gate]" \
  "summary waits for both real gates"
contains "$SUMMARY" "needs.tier1-agent-gate.result" \
  "summary consumes the Tier-1 result"
contains "$SUMMARY" "needs.desktop-candidate-gate.result" \
  "summary consumes the Desktop candidate result"
contains "$SUMMARY" 'if [ "$TIER1_RESULT" != "success" ] || [ "$DESKTOP_RESULT" != "success" ]' \
  "summary fails unless both gates succeeded"
contains "$SUMMARY" 'if [ "$ACCEPTED_SHA" != "$SOURCE_SHA" ]' \
  "summary binds the accepted Desktop candidate to the run SHA"
contains "$SUMMARY" "No tag, GitHub Release, updater pointer, or protected environment was created." \
  "run summary states the observable zero-publication result"
lacks "$SUMMARY" "environment:" "dry-run summary has no protected environment"
lacks "$SUMMARY" "tag_desktop_app.sh" "dry-run summary cannot claim the Desktop tag"
lacks "$SUMMARY" "create_release.sh" "dry-run summary cannot create the engine release"

echo "== 5. operator docs require exact-head evidence before release changes =="
contains "$(cat "$RELEASE_DOC")" \
  'gh workflow run auto-release.yml -R raullenchai/Rapid-MLX --ref "$DRY_RUN_REF" -f dry_run=true' \
  "runbook uses the documented workflow_dispatch ref + Boolean input"
contains "$(cat "$RELEASE_DOC")" \
  '"repos/raullenchai/Rapid-MLX/commits/$DRY_RUN_REF" --jq .sha' \
  "runbook resolves branch, tag, and SHA refs through the generic commit endpoint"
lacks "$(cat "$RELEASE_DOC")" \
  'git/ref/heads/$DRY_RUN_REF' \
  "runbook does not resolve selectable refs through a branch-only endpoint"
contains "$(cat "$RELEASE_DOC")" \
  'DRY_RUN_DISPATCHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"' \
  "runbook records an unambiguous lower bound before dispatch"
contains "$(cat "$RELEASE_DOC")" \
  '--created ">=$DRY_RUN_DISPATCHED_AT"' \
  "run discovery cannot select an older run at the same SHA"
contains "$(cat "$RELEASE_DOC")" \
  'for _ in {1..30}; do' \
  "runbook waits for the newly dispatched run to become visible"
contains "$(cat "$RELEASE_DOC")" \
  "Every PR that changes \`.github/workflows/auto-release.yml\`" \
  "release workflow PRs must link a green dry-run URL"
contains "$(cat "$RELEASE_DOC")" \
  "Before opening the version-bump PR" \
  "the intended bump parent is dry-run before the bump PR opens"

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
