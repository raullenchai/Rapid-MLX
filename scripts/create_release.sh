#!/usr/bin/env bash
#
# Create a GitHub Release whose tag we atomically claim at $RELEASE_SHA.
#
# Extracted from .github/workflows/auto-release.yml so the TOCTOU + recovery
# logic can be tested offline against a mock ``gh`` (no real release needed).
#
# WHY THIS EXISTS (issue #1462)
#
#   1. TOCTOU: a separate ``git ls-remote`` pre-check proves the tag is absent,
#      but another writer can push it before ``gh release create`` runs. GitHub
#      reuses an existing tag and silently ignores ``--target``, so the release
#      would publish notes describing $RELEASE_SHA against someone else's tag.
#      We claim the tag atomically FIRST via the git refs API (which fails if
#      the ref exists), and only then create the Release against the tag we own.
#
#   2. Recovery: if the tag was already created (a prior run that died between
#      tag and Release, a hand-pushed tag, a deleted Release), we create the
#      Release against it — but ONLY when it points at exactly $RELEASE_SHA.
#      Otherwise we fail loudly rather than attach these notes to a different
#      tree (GitHub ignores ``--target`` once the tag exists).
#
# Required env:
#   GH_TOKEN            token with contents: write (RELEASE_PAT preferred)
#   TAG                 ``vX.Y.Z``
#   RELEASE_SHA         commit the tag must point at
#   NOTES_FILE          path to the release body
#   GITHUB_REPOSITORY   owner/repo, e.g. raullenchai/Rapid-MLX
# Optional env:
#   GH                  the gh binary (default ``gh``; tests inject a mock)
#
# Exit 0 and prints ``released <TAG> at <SHA>`` on success.

set -euo pipefail

: "${GH_TOKEN:?create_release.sh: GH_TOKEN is required}"
: "${TAG:?create_release.sh: TAG is required}"
: "${RELEASE_SHA:?create_release.sh: RELEASE_SHA is required}"
: "${NOTES_FILE:?create_release.sh: NOTES_FILE is required}"
: "${GITHUB_REPOSITORY:?create_release.sh: GITHUB_REPOSITORY is required}"
GH_BIN="${GH:-gh}"
[ -f "$NOTES_FILE" ] || { echo "❌ notes file missing: $NOTES_FILE" >&2; exit 1; }
AUTOMATION_MARKER="<!-- rapid-mlx-auto-release:$TAG:$RELEASE_SHA -->"
DRAFT_NOTES_FILE=$(mktemp)
trap 'rm -f "$DRAFT_NOTES_FILE"' EXIT
{
  printf '%s\n' "$AUTOMATION_MARKER"
  cat "$NOTES_FILE"
} > "$DRAFT_NOTES_FILE"

resolve_tag_commit() {
  # Start from the explicit tag namespace so a same-named branch can never
  # satisfy verification, then peel annotated tag objects to a commit.
  local object_line object_type object_sha
  object_line=$(
    "$GH_BIN" api "repos/$GITHUB_REPOSITORY/git/ref/tags/$TAG" \
      --jq '.object | [.type, .sha] | @tsv' 2>/dev/null
  ) || return 1
  IFS=$'\t' read -r object_type object_sha <<<"$object_line"
  # Bounds the number of FOLLOWS, not the number of iterations: a bound on
  # iterations rejects a chain whose last hop lands on a commit, having had the
  # answer in hand. Same fix as scripts/tag_desktop_app.sh.
  local depth=0
  while true; do
    case "$object_type" in
      commit)
        [ -n "$object_sha" ] || return 1
        printf '%s\n' "$object_sha"
        return 0
        ;;
      tag)
        depth=$((depth + 1))
        [ "$depth" -le 10 ] || return 1
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

verify_existing_release() {
  local release_line release_sha
  if ! release_line=$(
    "$GH_BIN" release view "$TAG" --json targetCommitish,tagName,isDraft \
      --jq '[.targetCommitish, .isDraft] | @tsv' 2>/dev/null
  ); then
    return 1
  fi
  IFS=$'\t' read -r EXISTING_RELEASE EXISTING_IS_DRAFT <<<"$release_line"
  case "$EXISTING_IS_DRAFT" in true|false) ;; *) return 2 ;; esac
  if [ "$EXISTING_IS_DRAFT" = "true" ]; then
    local release_body
    release_body=$(
      "$GH_BIN" release view "$TAG" --json body -q .body 2>/dev/null
    ) || return 2
    if [ "${release_body%%$'\n'*}" != "$AUTOMATION_MARKER" ]; then
      echo "❌ Draft Release $TAG was not created by this workflow." >&2
      echo "   Refusing to overwrite or publish a manually staged draft." >&2
      return 2
    fi
  fi
  release_sha=$(resolve_tag_commit || true)
  if [ -z "$release_sha" ]; then
    echo "❌ Release $TAG exists, but its tag could not be resolved to a commit." >&2
    echo "   Refusing to report idempotent success without verifying the shipped tree." >&2
    return 2
  fi
  if [ "$release_sha" != "$RELEASE_SHA" ]; then
    echo "❌ Release $TAG already exists at $release_sha, but these notes describe $RELEASE_SHA." >&2
    echo "   Refusing to report success over a release that documents a different tree." >&2
    return 2
  fi
  return 0
}

finish_verified_existing_release() {
  if [ "$EXISTING_IS_DRAFT" = "true" ]; then
    "$GH_BIN" release edit "$TAG" \
      --title "$TAG" \
      --notes-file "$NOTES_FILE" \
      --draft=false
    local published_sha
    published_sha=$(resolve_tag_commit || true)
    if [ "$published_sha" != "$RELEASE_SHA" ]; then
      echo "❌ published tag $TAG points at ${published_sha:-<unknown>}, expected $RELEASE_SHA" >&2
      return 2
    fi
    echo "released $TAG at $RELEASE_SHA"
  else
    echo "Release $TAG already exists at the release commit ($EXISTING_RELEASE) — done"
  fi
}

# --- Idempotency: an existing Release is only "done" if it matches. -----
if verify_existing_release; then
  finish_verified_existing_release
  exit 0
else
  VERIFY_STATUS=$?
  [ "$VERIFY_STATUS" -eq 1 ] || exit "$VERIFY_STATUS"
fi

# --- Atomically claim the tag. -------------------------------------------
# POST /git/refs fails (422) if refs/tags/$TAG already exists, closing the
# TOCTOU that a separate pre-check + ``gh release create --target`` left open.
if ! "$GH_BIN" api -X POST "repos/$GITHUB_REPOSITORY/git/refs" \
    -f "ref=refs/tags/$TAG" -f "sha=$RELEASE_SHA" >/dev/null 2>&1; then
  # Claim lost the race (tag exists) — recover ONLY if it is our tag.
  EXISTING=$(resolve_tag_commit || true)
  if [ -z "$EXISTING" ]; then
    echo "❌ could not create refs/tags/$TAG and could not read it back — refusing to guess." >&2
    exit 1
  fi
  if [ "$EXISTING" != "$RELEASE_SHA" ]; then
    echo "❌ Tag $TAG already exists at $EXISTING, but these notes describe $RELEASE_SHA." >&2
    echo "   GitHub ignores --target when the tag already exists, so publishing now" >&2
    echo "   would attach these notes to a different tree." >&2
    echo "   Fix by hand: delete the stale tag, or bump to a fresh version." >&2
    exit 1
  fi
  echo "Tag $TAG already exists and points at the release commit — reusing it."
  # A concurrent run may have created the Release after our initial check.
  if verify_existing_release; then
    finish_verified_existing_release
    exit 0
  else
    VERIFY_STATUS=$?
    [ "$VERIFY_STATUS" -eq 1 ] || exit "$VERIFY_STATUS"
  fi
fi

# --- Create the Release against the tag we own. --------------------------
if ! CREATE_ERROR=$(
  "$GH_BIN" release create "$TAG" \
    --title "$TAG" \
    --notes-file "$DRAFT_NOTES_FILE" \
    --target "$RELEASE_SHA" \
    --draft 2>&1
); then
  # Close the second race: another run can publish after our re-check but
  # before this create. Treat that as idempotent success only after verifying
  # its tag still names the requested commit.
  if verify_existing_release; then
    finish_verified_existing_release
    exit 0
  else
    VERIFY_STATUS=$?
    [ "$VERIFY_STATUS" -eq 1 ] || exit "$VERIFY_STATUS"
  fi
  printf '%s\n' "$CREATE_ERROR" >&2
  exit 1
fi

# Verify while the Release is still a draft. A bad tag therefore cannot fire
# the downstream ``release: published`` workflow before this guard runs.
PUBLISHED=$(resolve_tag_commit || true)
if [ "$PUBLISHED" != "$RELEASE_SHA" ]; then
  echo "❌ draft release tag $TAG points at ${PUBLISHED:-<unknown>}, expected $RELEASE_SHA" >&2
  exit 1
fi

"$GH_BIN" release edit "$TAG" \
  --title "$TAG" \
  --notes-file "$NOTES_FILE" \
  --draft=false

# Defense in depth for an out-of-policy force-update after the pre-publish
# check. Repository policy should deny updates/deletions of release tags.
PUBLISHED=$(resolve_tag_commit || true)
if [ "$PUBLISHED" != "$RELEASE_SHA" ]; then
  echo "❌ published tag $TAG points at ${PUBLISHED:-<unknown>}, expected $RELEASE_SHA" >&2
  exit 1
fi

echo "released $TAG at $RELEASE_SHA"
