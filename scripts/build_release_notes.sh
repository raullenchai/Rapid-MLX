#!/usr/bin/env bash
#
# Build the GitHub Release body for one version and print it on stdout.
#
# Extracted from .github/workflows/auto-release.yml so the logic can be run and
# tested against real repository history without cutting a release
# (see tests/release/test_build_release_notes.sh).
#
# THE CONTRACT
#
#   The notes describe EXACTLY the commit named by $RELEASE_SHA — the same SHA
#   the tag is created at. Nothing in here reads a bare ``HEAD``, and nothing
#   re-derives "the current commit" a second time. A release that lies about
#   its own contents is worse than a release that fails to publish, so the
#   auto-generated commit list is asserted to be reachable from $RELEASE_SHA
#   before it is emitted.
#
# THE SHAPE
#
#   ## What's new in vX.Y.Z
#   [⚠️ emergency-release banner, when FORCE=true]
#   <docs/release-notes/vX.Y.Z.md, verbatim — prose, Highlights, tables>
#   <details><summary>All changes</summary> …auto commit list… </details>
#   ## Community contributors
#   Install: …
#
#   When no highlights file exists the commit list is emitted flat and
#   un-collapsed, i.e. byte-identical to the pre-highlights behaviour. Prose is
#   never required: a release must never be blocked on someone having written
#   it.
#
# Required env:
#   VERSION            X.Y.Z being released
#   RELEASE_SHA        commit being tagged
# Optional env:
#   FORCE              "true" → stamp the emergency (gate-bypassed) banner
#   ACTOR / REASON     audit fields for that banner
#   GITHUB_REPOSITORY  owner/repo, enables the contributor lookup via ``gh``
#   HIGHLIGHTS_DIR     default docs/release-notes
#   SKIP_CONTRIBUTORS  "1" → skip the ``gh`` calls (offline tests)

set -euo pipefail

: "${VERSION:?VERSION is required}"
: "${RELEASE_SHA:?RELEASE_SHA is required}"
FORCE="${FORCE:-false}"
ACTOR="${ACTOR:-unknown}"
REASON="${REASON:-}"
REPO="${GITHUB_REPOSITORY:-}"
HIGHLIGHTS_DIR="${HIGHLIGHTS_DIR:-docs/release-notes}"
SKIP_CONTRIBUTORS="${SKIP_CONTRIBUTORS:-0}"

TAG="v$VERSION"
RELEASE_SHA=$(git rev-parse --verify "${RELEASE_SHA}^{commit}")

# --------------------------------------------------------------------------
# 1. Baseline.
#
# The baseline must be the newest tag that is an ANCESTOR of the commit being
# released — not simply the highest version string that exists anywhere in the
# repository.
#
# The old ``git tag --list 'v*' --sort=-v:refname | head -1`` picked the latter.
# Those coincide on the happy path and diverge whenever a higher tag exists that
# this commit does not descend from:
#   * an emergency force_release of an older version (it skips the gate, so it
#     can overtake a normal release still sitting in the 90-minute gate);
#   * re-running the ``release`` job of an older run after newer tags landed
#     (the re-run reuses ``detect``'s cached "tag is new" answer);
#   * any two version bumps in flight at once.
# In each case ``highest..HEAD`` is empty or truncated, and the release ships
# with "_(no changes recorded)_" or a partial list.
#
# ``git describe`` walks ancestry, which is the property we actually want.
# ``--first-parent`` keeps that walk on the mainline: without it a tag that
# rode in on a merged side branch (fork at v1.0.0, tag v0.9.9, merge just
# before v1.1.0) is "nearer" than the real previous release, and every
# commit between them silently vanishes from the published notes.
# ``--exclude $TAG`` keeps a re-run from selecting the version's own tag (which
# would make the range empty).
# --------------------------------------------------------------------------
PREV_TAG=$(
  git describe --first-parent --tags --abbrev=0 --match 'v[0-9]*' --exclude "$TAG" \
    "$RELEASE_SHA" 2>/dev/null || true
)

if [ -n "$PREV_TAG" ]; then
  if ! git merge-base --is-ancestor "$PREV_TAG" "$RELEASE_SHA"; then
    echo "internal error: $PREV_TAG is not an ancestor of $RELEASE_SHA" >&2
    exit 1
  fi
  RANGE="$PREV_TAG..$RELEASE_SHA"
  COMMITS=$(git log "$RANGE" --pretty='- %s (%h)')
  echo "baseline: $PREV_TAG → $RELEASE_SHA" >&2
else
  # First release in the repository: no ancestor tag to diff against.
  RANGE="$RELEASE_SHA"
  COMMITS=$(git log -50 "$RELEASE_SHA" --pretty='- %s (%h)')
  echo "baseline: <none, first release> → $RELEASE_SHA" >&2
fi

# --------------------------------------------------------------------------
# 2. Assert the generated list really describes the tree being tagged.
#
# Guaranteed by construction today; asserted anyway so a future edit to the
# baseline logic fails the job instead of silently publishing a release whose
# notes belong to a different tree.
# --------------------------------------------------------------------------
cited_shas() {
  # Trailing "(<sha>)" of a list item — the citation format emitted above.
  sed -nE 's/^- .*\(([0-9a-f]{7,40})\)[[:space:]]*$/\1/p'
}

while IFS= read -r sha; do
  [ -n "$sha" ] || continue
  if ! git merge-base --is-ancestor "$sha" "$RELEASE_SHA" 2>/dev/null; then
    echo "❌ generated notes cite $sha, which is not an ancestor of $RELEASE_SHA" >&2
    exit 1
  fi
done < <(printf '%s\n' "$COMMITS" | cited_shas)

# --------------------------------------------------------------------------
# 3. Human-authored highlights (optional).
#
# Read out of the release commit itself, not the working tree, so the prose can
# only ever be the prose that was committed to the tree being tagged.
# --------------------------------------------------------------------------
HIGHLIGHTS=""
HIGHLIGHTS_PATH="$HIGHLIGHTS_DIR/$TAG.md"
if git cat-file -e "$RELEASE_SHA:$HIGHLIGHTS_PATH" 2>/dev/null; then
  HIGHLIGHTS=$(
    git show "$RELEASE_SHA:$HIGHLIGHTS_PATH" |
      # Drop complete whole-line HTML comment blocks so drafting guidance left
      # in the template can never ship. Buffer until the closing marker so an
      # unmatched opener fails safe instead of swallowing the remaining notes.
      awk '
        function flush_pending(    i) {
          for (i = 1; i <= pending_count; i++) print pending[i]
          pending_count = 0
        }
        in_comment {
          pending[++pending_count] = $0
          if ($0 ~ /-->/) {
            in_comment = 0
            if ($0 ~ /-->[[:space:]]*$/) pending_count = 0
            else flush_pending()
          }
          next
        }
        /^[[:space:]]*<!--/ {
          if ($0 ~ /-->/) {
            if ($0 !~ /-->[[:space:]]*$/) print
            next
          }
          in_comment = 1
          pending[++pending_count] = $0
          next
        }
        { print }
        END {
          if (in_comment) flush_pending()
        }
      ' |
      # Trim leading blank lines ($( ) trims trailing ones).
      sed -e '/./,$!d'
  )
  if [ -n "$HIGHLIGHTS" ]; then
    echo "highlights: $HIGHLIGHTS_PATH ($(printf '%s\n' "$HIGHLIGHTS" | wc -l | tr -d ' ') lines)" >&2
  else
    echo "highlights: $HIGHLIGHTS_PATH is empty — falling back to the commit list" >&2
  fi
else
  echo "highlights: no $HIGHLIGHTS_PATH in the release tree — commit list only" >&2
fi

# A wrong SHA in hand-written prose is worth flagging, but it must not block the
# release: notes are allowed to be absent or imperfect, never blocking.
if [ -n "$HIGHLIGHTS" ]; then
  while IFS= read -r sha; do
    [ -n "$sha" ] || continue
    git rev-parse --verify --quiet "${sha}^{commit}" >/dev/null || continue
    if ! git merge-base --is-ancestor "$sha" "$RELEASE_SHA" 2>/dev/null; then
      echo "⚠️  $HIGHLIGHTS_PATH cites $sha, which is not in this release" >&2
    fi
  done < <(printf '%s\n' "$HIGHLIGHTS" | cited_shas)
fi

# --------------------------------------------------------------------------
# 4. Credit every merged PR author other than the repository owner.
# --------------------------------------------------------------------------
CONTRIBUTIONS=()
if [ "$SKIP_CONTRIBUTORS" != "1" ] && [ -n "$REPO" ]; then
  OWNER="${REPO%%/*}"
  PR_NUMBERS=$(
    {
      git log "$RANGE" --pretty='%s' | sed -nE 's/.*\(#([0-9]+)\)$/\1/p'
      git log "$RANGE" --pretty='%s' | sed -nE 's/^Merge pull request #([0-9]+).*/\1/p'
    } | sort -nu
  )
  while IFS= read -r PR_NUMBER; do
    [ -n "$PR_NUMBER" ] || continue
    PR_JSON=$(gh pr view "$PR_NUMBER" --repo "$REPO" --json author,title,url 2>/dev/null) || continue
    AUTHOR=$(jq -r '.author.login // empty' <<<"$PR_JSON")
    if [ -z "$AUTHOR" ] || [ "$AUTHOR" = "$OWNER" ]; then
      continue
    fi
    CONTRIBUTIONS+=("$(
      jq -r '"- [@\(.author.login)](https://github.com/\(.author.login)) — [\(.title)](\(.url))"' \
        <<<"$PR_JSON"
    )")
  done <<<"$PR_NUMBERS"
fi

# --------------------------------------------------------------------------
# 5. Assemble.
# --------------------------------------------------------------------------
echo "## What's new in $TAG"
echo
if [ "$FORCE" = "true" ]; then
  echo "> ⚠️ **Emergency release** — the Tier-1 agent gate was bypassed"
  echo "> (forced by @$ACTOR). Reason: ${REASON:-<none given>}."
  echo
fi

if [ -n "$HIGHLIGHTS" ]; then
  printf '%s\n' "$HIGHLIGHTS"
  echo
  echo "<details>"
  echo "<summary>All changes</summary>"
  echo
  if [ -n "$COMMITS" ]; then
    printf '%s\n' "$COMMITS"
  else
    echo "_(no changes recorded)_"
  fi
  echo
  echo "</details>"
elif [ -n "$COMMITS" ]; then
  printf '%s\n' "$COMMITS"
else
  echo "_(no changes recorded)_"
fi
echo

if [ "${#CONTRIBUTIONS[@]}" -gt 0 ]; then
  echo "## Community contributors"
  echo
  printf '%s\n' "${CONTRIBUTIONS[@]}"
  echo
fi

echo "Install: \`brew upgrade rapid-mlx\` or \`pip install -U rapid-mlx==$VERSION\` (or just \`rapid-mlx upgrade\`)."
