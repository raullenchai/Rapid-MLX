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
        # An HTML comment closes at its FIRST "-->"; anything after that first
        # marker is visible content. Return that trailing text so the caller can
        # judge what a clean whole-line close leaves behind. Checking the whole
        # line for a trailing "-->" instead would treat "<!-- x --> note -->" as
        # a clean close and silently drop "note".
        function tail_after_close(line,    p) {
          p = index(line, "-->")
          return substr(line, p + 3)
        }
        # Would the text trailing a closed comment ship nothing visible? True
        # when it is only whitespace and/or further complete whole-line comments
        # (so "<!-- a --> <!-- b -->" is fully strippable). A trailing UNCLOSED
        # "<!--" is treated as not-clean so the line is kept verbatim rather than
        # swallowing the notes that follow it. Only ever fed the tail of a line
        # already in whole-line-comment context, so column-0-only stays intact.
        function rest_is_clean(s,    o, c) {
          while (1) {
            if (s ~ /^[[:space:]]*$/) return 1
            if (s !~ /^[[:space:]]*<!--/) return 0
            o = index(s, "<!--"); s = substr(s, o + 4)
            c = index(s, "-->"); if (c == 0) return 0
            s = substr(s, c + 3)
          }
        }
        # A line indented four or more spaces (or led by a tab) is a CommonMark
        # indented code block, not prose: fences and drafting comments there are
        # example text and must survive untouched. Kept deliberately simple —
        # the interval form /^ {0,3}/ is avoided for portability with older awks.
        function indented_code(line) {
          return (line ~ /^    / || line ~ /^\t/)
        }
        # The leading code-fence run (0-3 spaces of indent), e.g. "```" or
        # "~~~~", or "" when the line does not open/close a fence. A CommonMark
        # fence is three or more backticks or tildes at 0-3 spaces of indent.
        function fence_marker(line,    s, ch, n) {
          if (indented_code(line)) return ""
          s = line
          sub(/^ */, "", s)
          if (s ~ /^```/) ch = "`"
          else if (s ~ /^~~~/) ch = "~"
          else return ""
          n = 0
          while (substr(s, n + 1, 1) == ch) n++
          return substr(s, 1, n)
        }
        in_comment {
          pending[++pending_count] = $0
          if ($0 ~ /-->/) {
            in_comment = 0
            if (rest_is_clean(tail_after_close($0))) pending_count = 0
            else flush_pending()
          }
          next
        }
        # Inside a fenced code block, "<!--" is example text, not a drafting
        # comment: pass every line through untouched. Track the opener char and
        # length so only a matching bare closer (same char, at least as long, no
        # info string) ends the block — a shorter run or the other delimiter is
        # just fenced content. Checked only when not mid-comment so an open
        # comment still closes on its own "-->".
        !in_fence {
          fm = fence_marker($0)
          if (fm != "") {
            print
            fence_char = substr(fm, 1, 1); fence_len = length(fm); in_fence = 1
            next
          }
        }
        in_fence {
          print
          fm = fence_marker($0)
          if (fm != "" && substr(fm, 1, 1) == fence_char && length(fm) >= fence_len \
              && $0 ~ ("^[[:space:]]*[" fence_char "]+[[:space:]]*$")) in_fence = 0
          next
        }
        # A whole-line drafting comment sits at 0-3 spaces of indent; deeper
        # indentation is an indented code block whose "<!--" is example text.
        !indented_code($0) && /^[[:space:]]*<!--/ {
          if ($0 ~ /-->/) {
            if (!rest_is_clean(tail_after_close($0))) print
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
