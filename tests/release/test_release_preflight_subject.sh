#!/usr/bin/env bash
# Offline contract for the split bump guard / dispatched release pre-flight.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PREFLIGHT="$REPO_ROOT/.github/workflows/release-preflight.yml"
VERSION_GUARD="$REPO_ROOT/.github/workflows/version-check.yml"
AUTO_RELEASE="$REPO_ROOT/.github/workflows/auto-release.yml"

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
contains() { if grep -qF -- "$2" <<<"$1"; then ok "$3"; else bad "$3"; printf '        want: %s\n' "$2"; fi; }
absent() { if grep -qF -- "$2" <<<"$1"; then bad "$3"; else ok "$3"; fi; }

TRIGGERS=$(sed -n '/^on:/,/^permissions:/p' "$PREFLIGHT")
contains "$TRIGGERS" 'workflow_dispatch:' \
  "release pre-flight is explicitly dispatchable"
contains "$TRIGGERS" 'pr_number:' \
  "dispatch binds an open bump PR number"
contains "$TRIGGERS" 'expected_sha:' \
  "dispatch binds the expected exact head SHA"
absent "$TRIGGERS" 'pull_request:' \
  "release pre-flight never requests privileged context from a PR event"

BIND=$(sed -n '/^  bind-bump-pr:/,/^  pf1-release-contract:/p' "$PREFLIGHT")
contains "$BIND" 'gh api "repos/${REPO}/pulls/${PR_NUMBER}"' \
  "dispatch resolves the live PR through GitHub"
contains "$BIND" '[ "$EXPECTED_SHA" != "$HEAD_SHA" ] || [ "$DISPATCH_SHA" != "$HEAD_SHA" ]' \
  "dispatch refuses stale input, selected ref, or live PR head"
contains "$BIND" 'scripts/release_version.py subject "$TITLE"' \
  "dispatch rejects a noncanonical bump title before privileged jobs"

GUARD=$(sed -n '/Validate the complete bump PR contract/,/Pass — no stray version change/p' "$VERSION_GUARD")
# The guard reads the authoritative commit count from the PR payload (GitHub's
# `pull_request.commits`), NOT a local `git rev-list` that can vary with how the
# source branch was fetched or whether base advanced — so no stale-count drift.
contains "$GUARD" 'COMMIT_COUNT="${{ github.event.pull_request.commits }}"' \
  "required guard enforces one bump commit (authoritative PR payload count)"
contains "$GUARD" 'must contain exactly one commit' \
  "required guard refuses a multi-commit bump PR"
contains "$GUARD" 'scripts/check_release_notes.py' \
  "required guard synchronizes the two release-note inputs"
contains "$GUARD" '--pr-body "$PR_BODY"' \
  "required guard validates the evidence line through the subject SSOT"
contains "$GUARD" 'RUN_EVENT" != "workflow_dispatch"' \
  "required guard accepts only an explicit pre-flight dispatch"
contains "$GUARD" 'RUN_SHA" != "$HEAD_SHA"' \
  "required guard binds successful evidence to the exact PR head"

AUTO=$(sed -n '/Detect version bump/,/pyproject.toml must match/p' "$AUTO_RELEASE")
contains "$AUTO" 'release_version.py subject --allow-pr-suffix "$SUBJECT"' \
  "post-merge detect remains tolerant of GitHub's squash suffix"

if (cd "$REPO_ROOT" && python3 scripts/release_version.py subject --allow-pr-suffix \
     "chore: bump version to 0.13.2 (#2491)" >/dev/null 2>&1); then
  ok "post-merge parser accepts a suffixed bump subject"
else
  bad "post-merge parser accepts a suffixed bump subject"
fi
if (cd "$REPO_ROOT" && python3 scripts/validate_release_subject.py \
     --subject "chore: bump version to 0.13.2 (#2491)" >/dev/null 2>&1); then
  bad "required PR guard rejects a suffixed title"
else
  ok "required PR guard rejects a suffixed title"
fi

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
