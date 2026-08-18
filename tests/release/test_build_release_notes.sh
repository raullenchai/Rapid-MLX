#!/usr/bin/env bash
#
# Offline tests for scripts/build_release_notes.sh.
#
# The real workflow cannot be run without cutting a release, so the notes logic
# is exercised here against (a) this repository's real history and (b) a
# synthetic throwaway repo for the cases real history cannot produce on demand.
#
#   ./tests/release/test_build_release_notes.sh
#
# Requires: git, jq. Network/``gh`` are not required — contributor lookup is
# switched off via SKIP_CONTRIBUTORS=1.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="$REPO_ROOT/scripts/build_release_notes.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0

ok()   { PASS=$((PASS + 1)); printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad()  { FAIL=$((FAIL + 1)); printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
check() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1"; printf '        want: %s\n        got:  %s\n' "$3" "$2"; fi; }
contains() { if grep -qF -- "$2" <<<"$1"; then ok "$3"; else bad "$3"; fi; }
lacks()    { if grep -qF -- "$2" <<<"$1"; then bad "$3"; else ok "$3"; fi; }

# ==========================================================================
echo "== 1. real history: the notes describe exactly the tagged tree =="
# ==========================================================================
# Regression for the reported defect: v0.11.6's body must be the 7 commits that
# are actually reachable from the tag, and every one must be an ancestor of it.
cd "$REPO_ROOT"
if git rev-parse --verify --quiet v0.11.6 >/dev/null && \
   git rev-parse --verify --quiet v0.11.5 >/dev/null; then
  SHA=$(git rev-parse "v0.11.6^{commit}")
  BODY=$(VERSION=0.11.6 RELEASE_SHA="$SHA" SKIP_CONTRIBUTORS=1 bash "$SCRIPT" 2>"$TMP/err")

  contains "$(cat "$TMP/err")" "baseline: v0.11.5" "baseline is the nearest ANCESTOR tag (v0.11.5)"

  WANT=$(git log v0.11.5..v0.11.6 --pretty='- %s (%h)')
  GOT=$(grep -E '^- ' <<<"$BODY" || true)
  check "commit list == git log v0.11.5..v0.11.6" "$GOT" "$WANT"
  check "commit count" "$(grep -c '^- ' <<<"$BODY")" "$(git rev-list --count v0.11.5..v0.11.6)"

  # Every cited SHA reachable from the tag.
  BAD=0
  while IFS= read -r s; do
    [ -n "$s" ] || continue
    git merge-base --is-ancestor "$s" "$SHA" || BAD=$((BAD + 1))
  done < <(sed -nE 's/^- .*\(([0-9a-f]{7,40})\)$/\1/p' <<<"$BODY")
  check "every cited SHA is an ancestor of v0.11.6" "$BAD" "0"

  contains "$BODY" "## What's new in v0.11.6" "heading preserved"
  # shellcheck disable=SC2016  # the backticks are literal markdown, not a subshell
  contains "$BODY" 'Install: `brew upgrade rapid-mlx`' "Install: line preserved"
  lacks "$BODY" "<details>" "no <details> when there are no highlights"
else
  echo "  SKIP (tags v0.11.5/v0.11.6 not present in this clone)"
fi

# Sweep the last N releases: notes must equal the true ancestor range for each.
echo "-- sweep: recent tags --"
SWEPT=0
for TAG in $(git tag --list 'v0.11.*' --sort=-v:refname | head -8); do
  SHA=$(git rev-parse "$TAG^{commit}")
  PREV=$(git describe --tags --abbrev=0 --match 'v[0-9]*' --exclude "$TAG" "$SHA" 2>/dev/null || true)
  [ -n "$PREV" ] || continue
  BODY=$(VERSION="${TAG#v}" RELEASE_SHA="$SHA" SKIP_CONTRIBUTORS=1 bash "$SCRIPT" 2>/dev/null)
  GOT=$(grep -cE '^- ' <<<"$BODY" || true)
  check "$TAG: $GOT commits == git rev-list $PREV..$TAG" "$GOT" "$(git rev-list --count "$PREV..$SHA")"
  SWEPT=$((SWEPT + 1))
done
[ "$SWEPT" -gt 0 ] && echo "  (swept $SWEPT tags)"

# ==========================================================================
echo
echo "== 2. synthetic repo: highlights, degradation, banner, baseline =="
# ==========================================================================
SB="$TMP/sandbox"
mkdir -p "$SB/docs/release-notes"
cd "$SB"
git init -q .
git config user.email t@t; git config user.name t; git config commit.gpgsign false
cp "$SCRIPT" "$SB/build.sh"

commit() { printf '%s\n' "$2" > "$1"; git add -A; git commit -qm "$3"; }

commit a.txt 1 "feat: alpha (#1)"
git tag v1.0.0
commit b.txt 1 "fix: bravo (#2)"
commit c.txt 1 "feat: charlie (#3)"
commit d.txt 1 "chore: bump version to 1.1.0"
V110=$(git rev-parse HEAD)

# --- 2a. degrades to today's behaviour with no highlights file ---
BODY=$(VERSION=1.1.0 RELEASE_SHA="$V110" SKIP_CONTRIBUTORS=1 bash build.sh 2>/dev/null)
check "no highlights: 3 commits listed" "$(grep -c '^- ' <<<"$BODY")" "3"
lacks "$BODY" "<details>" "no highlights: commit list is NOT collapsed"
contains "$BODY" "## What's new in v1.1.0" "no highlights: heading present"
contains "$BODY" "Install:" "no highlights: install line present"

# --- 2b. highlights file present ---
cat > docs/release-notes/v1.2.0.md <<'MD'
<!-- drafting note that must not ship -->
<!-- Scratch space for the next release's notes. Append as you land work; in the
     version-bump PR, `git mv` this to vX.Y.Z.md and recreate this file empty.
     Whole-line HTML comments like this one are stripped before publishing.
     See README.md in this directory for what good notes look like. -->
<!-- inline drafting note --> Visible inline release note.
IMPORTANT RELEASE NOTE
<!-- later whole-line drafting note must not ship -->
<!-- solo open --> KEEP SOLO NOTE -->
<!-- multi open
--> KEEP MULTI NOTE -->
<!-- twin one --> <!-- twin two -->
<!-- lead comment --> KEEP TWIN TAIL <!-- trailing comment -->
This release is about speculative decoding.

Example markup:

```html
<!-- generated marker
FENCED COMMENT BODY stays visible -->
```

Nested-delimiter example:

~~~text
```
<!-- inner marker
NESTED FENCE BODY stays visible -->
```
~~~

Indented-code example:

    <!-- indented marker
    INDENTED CODE COMMENT stays visible -->
    plain indented code line

## Highlights

**DSpark speculation** — 2.1x on code. Prose lands at 32 to 57% acceptance and
does not consistently gain, so the adaptive controller parks speculation there.

| Workload | Off | On | Change | Acceptance |
| -------- | --: | -: | -----: | ---------: |
| Code     |  42 | 88 |  +110% |        71% |
| Prose    |  44 | 45 |    +2% |     32-57% |

<!-- unmatched drafting note stays visible rather than swallowing notes
Visible after unmatched opener.
MD
commit e.txt 1 "feat: delta (#4)"   # picks up the notes file too (git add -A)
commit f.txt 1 "chore: bump version to 1.2.0"
V120=$(git rev-parse HEAD)
git tag v1.1.0 "$V110"

BODY=$(VERSION=1.2.0 RELEASE_SHA="$V120" SKIP_CONTRIBUTORS=1 bash build.sh 2>/dev/null)
contains "$BODY" "This release is about speculative decoding." "highlights: prose prepended"
contains "$BODY" "## Highlights" "highlights: section rendered"
contains "$BODY" "| Prose    |  44 | 45 |    +2% |     32-57% |" "highlights: benchmark table intact"
contains "$BODY" "does not consistently gain" "highlights: negative result intact"
contains "$BODY" "<details>" "highlights: commit list collapsed"
contains "$BODY" "<summary>All changes</summary>" "highlights: <details> summary"
lacks "$BODY" "drafting note that must not ship" "highlights: HTML comments stripped"
lacks "$BODY" "Scratch space for the next release's notes" "highlights: multi-line HTML comments stripped"
lacks "$BODY" "version-bump PR" "highlights: multi-line HTML comment body stripped"
contains "$BODY" "<!-- inline drafting note --> Visible inline release note." "highlights: inline comment with visible suffix is preserved"
contains "$BODY" "IMPORTANT RELEASE NOTE" "highlights: inline comment does not swallow following notes"
contains "$BODY" "KEEP SOLO NOTE" "highlights: note after the first --> on an opener line is preserved"
contains "$BODY" "KEEP MULTI NOTE" "highlights: note after the first --> closing a multi-line comment is preserved"
lacks "$BODY" "twin one" "highlights: an all-comment line with two comments is fully stripped"
lacks "$BODY" "twin two" "highlights: the second comment on an all-comment line is stripped"
contains "$BODY" "KEEP TWIN TAIL" "highlights: visible text between two comments is preserved"
contains "$BODY" "FENCED COMMENT BODY stays visible" "highlights: HTML comments inside a code fence are not stripped"
contains "$BODY" "<!-- generated marker" "highlights: code-fence comment opener is preserved verbatim"
contains "$BODY" "NESTED FENCE BODY stays visible" "highlights: a shorter delimiter inside a tilde fence does not close it, comment survives"
contains "$BODY" "<!-- inner marker" "highlights: nested-fence comment opener is preserved verbatim"
contains "$BODY" "INDENTED CODE COMMENT stays visible" "highlights: HTML comments in a 4-space indented code block are not stripped"
contains "$BODY" "<!-- indented marker" "highlights: indented-code comment opener is preserved verbatim"
lacks "$BODY" "later whole-line drafting note" "highlights: later whole-line comment is still stripped"
contains "$BODY" "<!-- unmatched drafting note" "highlights: unmatched comment opener fails safe"
contains "$BODY" "Visible after unmatched opener." "highlights: unmatched comment does not swallow following notes"
contains "$BODY" "Install:" "highlights: install line still last"
# order: prose before <details> before Install
check "highlights: prose is above All changes" \
  "$([ "$(grep -n 'This release is about' <<<"$BODY" | cut -d: -f1)" -lt \
      "$(grep -n '<details>' <<<"$BODY" | cut -d: -f1)" ] && echo yes || echo no)" "yes"

# --- 2c. empty highlights file degrades ---
: > docs/release-notes/v1.3.0.md
commit g.txt 1 "fix: echo (#5)"
commit h.txt 1 "chore: bump version to 1.3.0"
V130=$(git rev-parse HEAD)
git tag v1.2.0 "$V120"
BODY=$(VERSION=1.3.0 RELEASE_SHA="$V130" SKIP_CONTRIBUTORS=1 bash build.sh 2>/dev/null)
lacks "$BODY" "<details>" "empty highlights file: degrades to flat commit list"
contains "$BODY" "- fix: echo (#5)" "empty highlights file: commits still listed"

# --- 2d. emergency banner ---
BODY=$(VERSION=1.3.0 RELEASE_SHA="$V130" FORCE=true ACTOR=raullenchai \
       REASON="Studio offline" SKIP_CONTRIBUTORS=1 bash build.sh 2>/dev/null)
contains "$BODY" "⚠️ **Emergency release**" "forced: banner rendered"
contains "$BODY" "(forced by @raullenchai). Reason: Studio offline." "forced: actor + reason"
BODY=$(VERSION=1.3.0 RELEASE_SHA="$V130" FORCE=true ACTOR=raullenchai \
       SKIP_CONTRIBUTORS=1 bash build.sh 2>/dev/null)
contains "$BODY" "Reason: <none given>." "forced: missing reason falls back"

# --- 2e. THE BASELINE FIX ---
# Two bumps in flight on linear main: v1.4.0's release job is still sitting in
# the 90-minute Tier-1 gate when v1.5.0 is force-released (a forced release
# skips the gate, so it can overtake). By the time v1.4.0's notes are built,
# v1.5.0 is the highest tag in the repo AND a DESCENDANT of v1.4.0's commit —
# so ``--sort=-v:refname | head -1`` yields a baseline that already contains
# everything, and the range collapses to nothing.
git tag v1.3.0 "$V130"
commit i.txt 1 "fix: foxtrot (#6)"
commit j.txt 1 "chore: bump version to 1.4.0"
V140=$(git rev-parse HEAD)
commit k.txt 1 "feat: golf (#7)"
commit l.txt 1 "chore: bump version to 1.5.0"
git tag v1.5.0 HEAD

OLD_PREV=$(git tag --list 'v*' --sort=-v:refname | head -1)
check "old logic picks the highest tag" "$OLD_PREV" "v1.5.0"
check "old logic's range is EMPTY (would publish 'no changes recorded')" \
  "$(git rev-list --count "$OLD_PREV..$V140")" "0"

BODY=$(VERSION=1.4.0 RELEASE_SHA="$V140" SKIP_CONTRIBUTORS=1 bash build.sh 2>"$TMP/e2")
contains "$(cat "$TMP/e2")" "baseline: v1.3.0" "new logic picks nearest ANCESTOR tag, not highest"
check "new logic lists the 2 real commits" "$(grep -c '^- ' <<<"$BODY")" "2"
lacks "$BODY" "_(no changes recorded)_" "new logic does not emit 'no changes recorded'"
lacks "$BODY" "golf" "new logic excludes commits that came after the release"

# --- 2f. re-run safety: the version's own tag must not become the baseline ---
git tag v1.4.0 "$V140"
BODY=$(VERSION=1.4.0 RELEASE_SHA="$V140" SKIP_CONTRIBUTORS=1 bash build.sh 2>"$TMP/e3")
contains "$(cat "$TMP/e3")" "baseline: v1.3.0" "re-run with tag already present: baseline unchanged"
check "re-run still lists 2 commits" "$(grep -c '^- ' <<<"$BODY")" "2"

# --- 2g. first release ever (no ancestor tag) ---
SB2="$TMP/sandbox2"; mkdir -p "$SB2"; cd "$SB2"
git init -q .; git config user.email t@t; git config user.name t; git config commit.gpgsign false
printf 1 > a; git add -A; git commit -qm "feat: first"
printf 2 > b; git add -A; git commit -qm "chore: bump version to 0.1.0"
BODY=$(VERSION=0.1.0 RELEASE_SHA="$(git rev-parse HEAD)" SKIP_CONTRIBUTORS=1 bash "$SCRIPT" 2>"$TMP/e4")
contains "$(cat "$TMP/e4")" "first release" "no tags at all: falls back to git log -50"
check "no tags at all: lists both commits" "$(grep -c '^- ' <<<"$BODY")" "2"

# --- 2h. a tag that rode in on a merged SIDE BRANCH must not become the
# baseline. Without --first-parent, `git describe` walks into merged parents
# and finds v0.9.9 "nearer" than the real previous mainline release v2.0.0 —
# which would silently drop every commit between them from the notes.
SB3="$TMP/sandbox3"; mkdir -p "$SB3"; cd "$SB3"
git init -q .; git config user.email t@t; git config user.name t; git config commit.gpgsign false
printf 1 > a; git add -A; git commit -qm "feat: base"
printf 2 > b; git add -A; git commit -qm "chore: bump version to 2.0.0"
git tag v2.0.0 HEAD
MAINLINE=$(git rev-parse HEAD)
# real mainline work that MUST appear in the notes
printf 3 > c; git add -A; git commit -qm "feat: mainline work (#42)"
# a side branch forked from before the release, carrying an older tag
git checkout -q -b side "$MAINLINE"
printf 4 > d; git add -A; git commit -qm "feat: side work"
git tag v0.9.9 HEAD
git checkout -q -
git merge -q --no-ff side -m "Merge side"
printf 5 > e; git add -A; git commit -qm "chore: bump version to 2.1.0"
V210=$(git rev-parse HEAD)
BODY=$(VERSION=2.1.0 RELEASE_SHA="$V210" SKIP_CONTRIBUTORS=1 bash "$SCRIPT" 2>"$TMP/e5")
contains "$(cat "$TMP/e5")" "baseline: v2.0.0" \
  "side-branch tag does not hijack the baseline (--first-parent)"
contains "$BODY" "mainline work" \
  "commits after the real previous release still appear in the notes"

echo
printf 'passed %d, failed %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
