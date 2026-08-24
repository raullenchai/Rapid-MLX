#!/usr/bin/env bash
#
# Tag the desktop app at the version the engine just released.
#
# The engine and the app ship ONE version number (enforced on every PR by
# scripts/check_version_sync.py), so a release is one event, not two.
# Cutting them separately is what produced the 0.12.5-vs-0.12.6 split that
# reached users: the engine shipped from auto-release.yml, the
# ``rapid-mac-vX.Y.Z`` tag was a manual step somebody had to remember, and
# for four releases nobody did.
#
# Creating ``refs/tags/rapid-mac-v$VERSION`` hands off to
# .github/workflows/rapid-mac-release.yml, which builds, signs, notarises
# and attaches the DMG.
#
# WHY THE API AND NOT ``git push``
# --------------------------------
# actions/checkout persists the default ``GITHUB_TOKEN`` as the credential
# for ``origin``, and GitHub suppresses workflow runs caused by pushes
# authenticated with ``GITHUB_TOKEN`` (its anti-loop guard). A ``git push``
# of this tag would therefore succeed, the step would go green, and NO app
# build, signing, notarisation or DMG would ever happen — the silent
# half-release this whole change exists to prevent. Going through
# ``gh api`` uses ``GH_TOKEN``, which the workflow sets to the RELEASE_PAT
# user; a ref created by a real user does fire ``push``. This is the same
# reason create_release.sh runs under the PAT.
#
# Idempotency is a POST, not a pre-check: ``POST /git/refs`` fails 422 when
# the ref already exists, so the claim is atomic. Only on that failure do we
# read the tag back — peeling annotated tag objects, because
# ``/git/ref/tags/X`` returns the *tag object's* SHA for an annotated tag
# and comparing that raw value to a commit SHA would report "points
# somewhere else" for a tag that is in fact correct (four of this repo's
# rapid-mac tags are annotated). Same tree → done. Different tree → refuse:
# moving a published tag would ship one version's notes against another's
# build.
#
# WHY A MISSING PAT IS FATAL RATHER THAN A FALLBACK
# -------------------------------------------------
# Creating the tag under GITHUB_TOKEN would be worse than not creating it. The
# ref would exist and this step would go green, but the run it should have
# triggered is suppressed — and the recovery (restore the secret, re-run) then
# finds a tag that already points at the right commit and exits 0 without
# emitting any event. The app could never be built for that version again
# without hand-deleting a published tag.
#
# Skipping quietly is no better: the run ends green with only the engine
# released, and a re-run sees the published engine Release, decides there is
# nothing to release, and never reaches this step again. So this refuses to
# run at all. In the workflow the same condition is checked BEFORE anything is
# published, which is what makes "set the secret and re-run" an actual fix.
#
# Required environment:
#   VERSION            X.Y.Z — the version the engine just released
#   RELEASE_SHA        the commit the engine release was cut from
#   GITHUB_REPOSITORY  owner/repo
#   GH_TOKEN           consumed by ``gh`` (RELEASE_PAT in the workflow)
# Optional:
#   HAVE_PAT           "true" when GH_TOKEN is the RELEASE_PAT. Anything else,
#                      INCLUDING the empty string, refuses to tag (see above).
#                      Only a genuinely unset value defaults to true, for a
#                      hand-run — an empty one means a workflow expression
#                      evaluated to nothing, which must not read as consent.
#   GH                 path to the gh binary (tests point this at a mock)
#
# Exit status: 0 tag created or already correct, 1 anything else.

set -euo pipefail

: "${VERSION:?tag_desktop_app.sh: VERSION is required}"
: "${RELEASE_SHA:?tag_desktop_app.sh: RELEASE_SHA is required}"
: "${GITHUB_REPOSITORY:?tag_desktop_app.sh: GITHUB_REPOSITORY is required}"
GH_BIN="${GH:-gh}"
HAVE_PAT="${HAVE_PAT-true}"

# The tag is built from this string, so a value the tag namespace cannot
# carry has to fail here rather than produce ``rapid-mac-v0.12.7\n``.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if ! python3 "$REPO_ROOT/scripts/release_version.py" validate "$VERSION" >/dev/null; then
  echo "❌ VERSION is '$VERSION', which is not X.Y.Z or X.Y.Z-rcN" >&2
  exit 1
fi

APP_TAG="rapid-mac-v${VERSION}"

if [ "$HAVE_PAT" != "true" ]; then
  echo "::error::refusing to create ${APP_TAG}: RELEASE_PAT is not available, and a tag written with the default GITHUB_TOKEN triggers no workflow — no DMG would ever be built, and the tag's existence would then block every retry." >&2
  echo "   Set the RELEASE_PAT secret and run again." >&2
  exit 1
fi

# Follows at most this many annotated tag objects before giving up. Ordinary
# tags peel in one hop; the bound only exists so a malformed chain cannot spin.
MAX_TAG_PEEL_DEPTH=10

resolve_tag_commit() {
  # Start from the explicit tag namespace so a same-named branch can never
  # satisfy verification, then peel annotated tag objects to a commit.
  #
  # The loop checks the type BEFORE following, and the counter bounds the
  # number of FOLLOWS. Bounding iterations instead would reject a chain whose
  # last hop lands on a commit — the answer was in hand and thrown away.
  local object_line object_type object_sha depth=0
  object_line=$(
    "$GH_BIN" api "repos/$GITHUB_REPOSITORY/git/ref/tags/$APP_TAG" \
      --jq '.object | [.type, .sha] | @tsv' 2>/dev/null
  ) || return 1
  IFS=$'\t' read -r object_type object_sha <<<"$object_line"
  while true; do
    case "$object_type" in
      commit)
        [ -n "$object_sha" ] || return 1
        printf '%s\n' "$object_sha"
        return 0
        ;;
      tag)
        depth=$((depth + 1))
        [ "$depth" -le "$MAX_TAG_PEEL_DEPTH" ] || return 1
        object_line=$(
          "$GH_BIN" api "repos/$GITHUB_REPOSITORY/git/tags/$object_sha" \
            --jq '.object | [.type, .sha] | @tsv' 2>/dev/null
        ) || return 1
        IFS=$'\t' read -r object_type object_sha <<<"$object_line"
        ;;
      *) return 1 ;;
    esac
  done
}

if "$GH_BIN" api -X POST "repos/$GITHUB_REPOSITORY/git/refs" \
    -f "ref=refs/tags/$APP_TAG" -f "sha=$RELEASE_SHA" >/dev/null 2>&1; then
  echo "Created $APP_TAG at $RELEASE_SHA → rapid-mac-release.yml"
  exit 0
fi

EXISTING=$(resolve_tag_commit || true)
if [ -z "$EXISTING" ]; then
  echo "❌ could not create refs/tags/$APP_TAG and could not read it back — refusing to guess." >&2
  echo "   Check the token's contents:write scope, then re-run this workflow." >&2
  exit 1
fi
if [ "$EXISTING" != "$RELEASE_SHA" ]; then
  echo "❌ $APP_TAG already exists at $EXISTING, but this release is $RELEASE_SHA." >&2
  echo "   Refusing to move a published tag: the DMG built from $EXISTING would" >&2
  echo "   ship under $RELEASE_SHA's release notes." >&2
  echo "   Fix by hand: delete the stale tag, or bump to a fresh version." >&2
  exit 1
fi
echo "$APP_TAG already points at $RELEASE_SHA — nothing to do."
