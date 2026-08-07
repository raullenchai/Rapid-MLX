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
# Required environment:
#   VERSION            X.Y.Z — the version the engine just released
#   RELEASE_SHA        the commit the engine release was cut from
#   GITHUB_REPOSITORY  owner/repo
#   GH_TOKEN           consumed by ``gh`` (RELEASE_PAT in the workflow)
# Optional:
#   GH                 path to the gh binary (tests point this at a mock)
#
# Exit status: 0 tag created or already correct, 1 anything else.

set -euo pipefail

: "${VERSION:?tag_desktop_app.sh: VERSION is required}"
: "${RELEASE_SHA:?tag_desktop_app.sh: RELEASE_SHA is required}"
: "${GITHUB_REPOSITORY:?tag_desktop_app.sh: GITHUB_REPOSITORY is required}"
GH_BIN="${GH:-gh}"

# The tag is built from this string, so a value the tag namespace cannot
# carry has to fail here rather than produce ``rapid-mac-v0.12.7\n``.
case "$VERSION" in
  *[!0-9.]* | *..* | .* | *. | "")
    echo "❌ VERSION is '$VERSION', which is not X.Y.Z" >&2
    exit 1
    ;;
esac
if [ "$(printf '%s' "$VERSION" | tr -cd '.' | wc -c | tr -d ' ')" != "2" ]; then
  echo "❌ VERSION is '$VERSION', which is not X.Y.Z" >&2
  exit 1
fi

APP_TAG="rapid-mac-v${VERSION}"

resolve_tag_commit() {
  # Start from the explicit tag namespace so a same-named branch can never
  # satisfy verification, then peel annotated tag objects to a commit.
  local object_line object_type object_sha
  object_line=$(
    "$GH_BIN" api "repos/$GITHUB_REPOSITORY/git/ref/tags/$APP_TAG" \
      --jq '.object | [.type, .sha] | @tsv' 2>/dev/null
  ) || return 1
  IFS=$'\t' read -r object_type object_sha <<<"$object_line"
  for _ in 1 2 3 4 5; do
    case "$object_type" in
      commit)
        [ -n "$object_sha" ] || return 1
        printf '%s\n' "$object_sha"
        return 0
        ;;
      tag)
        object_line=$(
          "$GH_BIN" api "repos/$GITHUB_REPOSITORY/git/tags/$object_sha" \
            --jq '.object | [.type, .sha] | @tsv' 2>/dev/null
        ) || return 1
        IFS=$'\t' read -r object_type object_sha <<<"$object_line"
        ;;
      *) return 1 ;;
    esac
  done
  return 1
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
