#!/usr/bin/env bash
#
# Offline contract: the bump-subject policy is TWO-LEVEL, and the wiring in
# release-preflight.yml + auto-release.yml matches the documented distinction.
#
#   detect-bump-pr (release-preflight.yml)   BROAD r: routes a bump title into the
#                                            gates even when GitHub's default
#                                            squash suffix "(#NN)" is present
#                                            (release_version.py subject --allow-pr-suffix).
#   PF-1 (release-preflight.yml)             STRICT: requires the canonical title
#                                            (validate_release_subject.py) and FAILS a
#                                            suffixed one — so a bump PR title is kept clean.
#   detect (auto-release.yml, post-merge)    TOLERANT: accepts the "(#NN)" suffix on the
#                                            merged commit subject (--allow-pr-suffix),
#                                            so an un-"--subject" squash merge can't strand
#                                            a release.
#
# Provable offline: read the two workflow files and assert the exact wiring, then
# exercise the two code paths (release_version.py --allow-pr-suffix vs the strict
# validate_release_subject.py) against a suffixed subject.
#
#   ./tests/release/test_release_preflight_subject.sh
#
# Requires: bash, python3, grep.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PREFLIGHT="$REPO_ROOT/.github/workflows/release-preflight.yml"
AUTO_RELEASE="$REPO_ROOT/.github/workflows/auto-release.yml"

PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
contains() { if grep -qF -- "$2" <<<"$1"; then ok "$3"; else bad "$3"; printf '        want: %s\n' "$2"; fi; }

# 1) detect-bump-pr is BROAD: recognizes the optional "(#NN)" suffix so a suffixed
#    bump title is still routed into PF-1 (which then fails it). The version
#    derivation must tolerate the suffix.
DETECT=$(sed -n '/id: check$/,/^  pf1-subject-regex/p' "$PREFLIGHT")
contains "$DETECT" 'release_version.py subject --allow-pr-suffix "$TITLE"' \
  "detect-bump-pr uses --allow-pr-suffix (routing is broad, not strict)"

# 2) PF-1 is STRICT: it runs the canonical-title validator, so a suffixed title is
#    still caught. PF-1 runs whenever is_bump is true (which now includes suffixed
#    titles from step 1), so the strict check genuinely applies to them.
PF1=$(sed -n '/^  pf1-subject-regex:/,/^  pf2-release-secrets:/p' "$PREFLIGHT")
contains "$PF1" 'needs.detect-bump-pr.outputs.is_bump == '\''true'\''' \
  "PF-1 runs for every routed bump title (incl. a suffixed one)"
contains "$PF1" 'validate_release_subject.py --subject "$TITLE"' \
  "PF-1 enforces the STRICT canonical validator (no --allow-pr-suffix)"

# 3) Post-merge detect in auto-release.yml is TOLERANT: it derives the version from
#    the merged commit subject with --allow-pr-suffix.
AUTO=$(sed -n '/Detect version bump/,/pyproject.toml must match/p' "$AUTO_RELEASE")
contains "$AUTO" 'release_version.py subject --allow-pr-suffix "$SUBJECT"' \
  "post-merge detect tolerates the squash suffix (--allow-pr-suffix)"

# 4) Code-level behaviour: a suffixed subject is ACCEPTED by --allow-pr-suffix
#    (detect/pf-broad path) and REJECTED by the strict validator (PF-1 path).
if (cd "$REPO_ROOT" && python3 scripts/release_version.py subject --allow-pr-suffix \
     "chore: bump version to 0.6.82 (#518)" >/dev/null 2>&1); then
  ok "release_version.py --allow-pr-suffix accepts a suffixed subject (broad path)"
else
  bad "release_version.py --allow-pr-suffix accepts a suffixed subject (broad path)"
fi
if (cd "$REPO_ROOT" && python3 scripts/validate_release_subject.py \
     --subject "chore: bump version to 0.6.82 (#518)" >/dev/null 2>&1); then
  bad "validate_release_subject.py (PF-1) rejects a suffixed subject (strict path)"
else
  ok "validate_release_subject.py (PF-1) rejects a suffixed subject (strict path)"
fi

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
