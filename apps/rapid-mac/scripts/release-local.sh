#!/usr/bin/env bash
# release-local.sh — two lanes for shipping the rapid-mac app. See RELEASING.md.
#
# ── DOGFOOD (default) ────────────────────────────────────────────────────
#   Build + sign + (if a notary key is configured) notarise a DMG on THIS
#   Mac, for you or a tester. NO git tag, NO GitHub Release, NO CI, NO
#   R2/latest.json. This is what you run ~all the time; it costs $0 of CI.
#     scripts/release-local.sh            # build a dogfood DMG
#     scripts/release-local.sh --check    # verify signing/notary setup only
#
# ── PUBLIC (--publish) ───────────────────────────────────────────────────
#   The guarded way to cut a USER-FACING release. It does NOT build locally.
#   It preflights (CHANGELOG entry exists, tag == plist version, tag is newer
#   than the latest release, local main == origin/main) then pushes the tag —
#   which hands off to the monorepo-root workflow
#   .github/workflows/rapid-mac-release.yml. CI builds, signs, notarises,
#   enforces size gates and attaches the DMG to the GitHub Release (an
#   required mirror job publishes the DMG and updater fallback to the CDN).
#   MONOREPO: the app release tag is prefixed ``rapid-mac-v`` so it never
#   collides with the engine's own ``v*`` release tags in the same repo.
#   Rare. Stable (non-prerelease) tags only.
#     scripts/release-local.sh --publish rapid-mac-v0.11.0
#
# ─────────────────────────────────────────────────────────────────────────
# KEY HANDLING: this script does NOT create, copy, print, or commit the
# signing/notary private key. For a NOTARISED dogfood DMG the App Store
# Connect .p8 — placed by YOU (see RELEASING.md, Part C) — is read by Apple's
# notarytool during notarisation; the script only passes its path. Public
# releases use the repo's CI secrets, not this file. ~/.rapid-release.env is
# SOURCED as shell, so keep it owner-only (chmod 600) and never commit it.
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${RAPID_RELEASE_ENV:-$HOME/.rapid-release.env}"

# Strict arg parsing — exactly the accepted shapes, nothing else. (A lax
# parser let ``--publish v0.10.4 --check`` still push the public tag.)
MODE=""
TAG=""
if [[ $# -eq 0 ]]; then
    MODE="dogfood"
elif [[ $# -eq 1 && "$1" == "--check" ]]; then
    MODE="check"
elif [[ $# -eq 2 && "$1" == "--publish" ]]; then
    MODE="publish"; TAG="$2"
else
    {
        echo "usage:"
        echo "  $0                              # dogfood build (no args)"
        echo "  $0 --check                      # verify signing/notary setup"
        echo "  $0 --publish rapid-mac-vX.Y.Z   # guarded public release (pushes tag → CI)"
    } >&2
    exit 2
fi

# MONOREPO: app release tags are prefixed so they never collide with the
# engine's own ``v*`` tags. The workflow that fires on this tag lives at
# the repository root, not under apps/rapid-mac.
TAG_PREFIX="rapid-mac-v"
MONOREPO_ROOT="$(cd "$ROOT/../.." && pwd)"
RELEASE_WORKFLOW="$MONOREPO_ROOT/.github/workflows/rapid-mac-release.yml"

note() { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mrelease-local: %s\033[0m\n' "$*" >&2; }
fail() { printf '\033[31mrelease-local: %s\033[0m\n' "$*" >&2; exit 1; }

# Source the operator's env file ONCE, up front — before detect_identity, so a
# blank ``CODESIGN_IDENTITY=""`` in the file doesn't clobber auto-detection
# (codex r2 BLOCKING). Sets AC_API_* (for notarisation) and an optional
# CODESIGN_IDENTITY pin. Owner-only shell; --publish ignores its values.
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

# ── Signing identity ─────────────────────────────────────────────────────
# Developer ID if present → shareable/notarisable. Otherwise ad-hoc, which
# runs on YOUR Mac (right-click → Open) but isn't Gatekeeper-valid elsewhere.
# awk (vs grep|head) so "no identity" yields "" without tripping pipefail.
detect_identity() {
    if [[ -z "${CODESIGN_IDENTITY:-}" ]]; then
        CODESIGN_IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
            | awk '/Developer ID Application/ {print $2; exit}')"
    fi
    : "${CODESIGN_IDENTITY:=}"
}

# True when a usable notary setup is present (does NOT source — env already
# sourced above). Warns + returns 1 so dogfood degrades to un-notarised.
notary_ready() {
    if [[ -z "${AC_API_KEY_ID:-}" || -z "${AC_API_ISSUER_ID:-}" || -z "${AC_API_KEY_PATH:-}" \
          || "${AC_API_ISSUER_ID:-}" == PUT-* ]]; then
        warn "notary not configured in $ENV_FILE (AC_API_* / issuer placeholder) — build will be un-notarised."
        return 1
    fi
    if [[ ! -f "$AC_API_KEY_PATH" ]]; then
        warn "notary key not found at $AC_API_KEY_PATH — place the .p8 yourself (RELEASING.md, Part C). Building un-notarised."
        return 1
    fi
    export AC_API_KEY_ID AC_API_ISSUER_ID AC_API_KEY_PATH
    return 0
}

# Strict SemVer "a > b" for prefixed rapid-mac-vX.Y.Z tags (no prerelease) —
# Bash-3.2-safe, and not fooled by ``sort -V`` ranking an RC after its final
# (codex r2 MAJOR). Strips the ``rapid-mac-v`` prefix (falling back to a bare
# ``v``) before comparing the numeric fields.
version_gt() {
    local a="${1#"$TAG_PREFIX"}" b="${2#"$TAG_PREFIX"}"
    a="${a#v}"; b="${b#v}"
    local IFS=.
    # shellcheck disable=SC2206
    local -a A=($a) B=($b)
    local i ai bi
    for i in 0 1 2; do
        ai=${A[i]:-0}; bi=${B[i]:-0}
        if (( 10#$ai > 10#$bi )); then return 0; fi
        if (( 10#$ai < 10#$bi )); then return 1; fi
    done
    return 1   # equal → not greater
}

# ── --check : report setup, build nothing ────────────────────────────────
if [[ "$MODE" == "check" ]]; then
    detect_identity
    if [[ -n "$CODESIGN_IDENTITY" ]]; then note "signing identity: $CODESIGN_IDENTITY"
    else warn "no 'Developer ID Application' identity — builds will be ad-hoc (local use only)."; fi
    if notary_ready; then note "notary key present at $AC_API_KEY_PATH (path only; read by notarytool, not by this script)"
    else warn "notarisation not configured — dogfood is fine, but shareable builds need it."; fi
    note "check complete."
    exit 0
fi

# ── --publish : guarded tag push → CI does the real release ──────────────
if [[ "$MODE" == "publish" ]]; then
    is_canonical_release_url() {
        [[ "$1" =~ ^https://github\.com/raullenchai/Rapid-MLX(\.git)?$ ]] \
            || [[ "$1" =~ ^git@github\.com:raullenchai/Rapid-MLX(\.git)?$ ]] \
            || [[ "$1" =~ ^ssh://git@github\.com/raullenchai/Rapid-MLX(\.git)?$ ]]
    }

    # Resolve the repository that owns the release workflow. ``origin`` is a
    # convention, not an identity: contributor clones commonly point it at
    # upstream. An explicit override wins; otherwise require exactly one
    # remote whose URL names the canonical release repository.
    RELEASE_REMOTE="${RAPID_RELEASE_REMOTE:-}"
    if [[ -z "$RELEASE_REMOTE" ]]; then
        MATCHING_REMOTES=()
        while read -r remote; do
            fetch_url="$(git remote get-url --all "$remote" 2>/dev/null || true)"
            push_url="$(git remote get-url --push --all "$remote" 2>/dev/null || true)"
            is_canonical_release_url "$fetch_url" \
                && is_canonical_release_url "$push_url" \
                && MATCHING_REMOTES+=("$remote")
        done < <(git remote)
        [[ "${#MATCHING_REMOTES[@]}" -eq 1 ]] \
            || fail "cannot uniquely resolve the raullenchai/Rapid-MLX release remote; set RAPID_RELEASE_REMOTE explicitly."
        RELEASE_REMOTE="${MATCHING_REMOTES[0]}"
    fi
    RELEASE_FETCH_URL="$(git remote get-url --all "$RELEASE_REMOTE" 2>/dev/null || true)"
    RELEASE_PUSH_URL="$(git remote get-url --push --all "$RELEASE_REMOTE" 2>/dev/null || true)"
    [[ "$(printf '%s\n' "$RELEASE_FETCH_URL" | awk 'NF{n++} END{print n+0}')" -eq 1 ]] \
        || fail "release remote '$RELEASE_REMOTE' must have exactly one fetch URL."
    [[ "$(printf '%s\n' "$RELEASE_PUSH_URL" | awk 'NF{n++} END{print n+0}')" -eq 1 ]] \
        || fail "release remote '$RELEASE_REMOTE' must have exactly one push URL."
    is_canonical_release_url "$RELEASE_FETCH_URL" \
        || fail "release remote '$RELEASE_REMOTE' fetches from '$RELEASE_FETCH_URL', not raullenchai/Rapid-MLX."
    is_canonical_release_url "$RELEASE_PUSH_URL" \
        || fail "release remote '$RELEASE_REMOTE' pushes to '$RELEASE_PUSH_URL', not raullenchai/Rapid-MLX."

    # Stable tags only — the CI publish + in-app updater have no prerelease
    # channel, so an RC would be offered to stable users (codex r2 MAJOR).
    [[ "$TAG" =~ ^rapid-mac-v[0-9]+\.[0-9]+\.[0-9]+$ ]] \
        || fail "tag '$TAG' must be rapid-mac-vMAJOR.MINOR.PATCH (no prerelease suffix — the updater has no prerelease channel)."
    VERSION="${TAG#"$TAG_PREFIX"}"

    # CHANGELOG must carry this version as a real heading (blocks stale /
    # placeholder tags; anchored so it can't match mid-line).
    CL_ESC="${VERSION//./\\.}"
    grep -qE "^## \[${CL_ESC}\]" CHANGELOG.md \
        || fail "CHANGELOG.md has no '## [${VERSION}]' heading — bump the version + write the entry first."

    # Tag must equal the built version. No bypass on the public lane.
    PLIST_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' Resources/Info.plist 2>/dev/null || true)"
    [[ "$PLIST_VERSION" == "$VERSION" ]] \
        || fail "tag $TAG (=$VERSION) != Resources/Info.plist CFBundleShortVersionString ($PLIST_VERSION). Bump the plist first."
    PLIST_BUILD="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' Resources/Info.plist 2>/dev/null || true)"
    [[ "$PLIST_BUILD" =~ ^[1-9][0-9]*$ ]] \
        || fail "Resources/Info.plist CFBundleVersion must be a positive integer for Sparkle (got '$PLIST_BUILD')."

    # NOT --quiet, and NOT bare: `git fetch --tags` exits non-zero when ANY tag
    # would clobber a local one, and `set -e` then killed this script with no
    # output at all — the operator saw a release that simply did not happen and
    # no reason why. That is the worst way for a release tool to fail: the
    # natural next step is to cut the release by hand, which bypasses every
    # check below.
    #
    # Here it was five legacy tags (v0.6.53/62/72/76, v0.7.2) that point at
    # different commits in this fork than in the upstream this clone also has a
    # remote for. They have nothing to do with the release being cut, so they
    # must not silently block it — but they must not be force-overwritten
    # either, because which lineage is authoritative is the operator's call.
    #
    # LC_ALL=C because the tag list below is scraped out of git's human-facing
    # error text. Under a localised git the awk match silently finds nothing,
    # and the operator gets the generic failure instead of the named tags and
    # the exact command that fixes them — which is the whole point of this
    # block.
    if ! FETCH_ERR="$(LC_ALL=C git fetch "$RELEASE_REMOTE" --tags 2>&1)"; then
        printf '%s\n' "$FETCH_ERR" >&2
        CLOBBER="$(printf '%s\n' "$FETCH_ERR" | awk '/would clobber existing tag/ {print $3}' | tr '\n' ' ')"
        if [[ -n "$CLOBBER" ]]; then
            fail "cannot fetch tags from $RELEASE_REMOTE: these local tags disagree with it — ${CLOBBER}
       They are unrelated to $TAG, but the fetch cannot complete while they differ.
       Inspect one with:   git rev-parse <tag>   vs   git ls-remote $RELEASE_REMOTE refs/tags/<tag>
       If $RELEASE_REMOTE is authoritative, drop the local copies and retry:
           git tag -d ${CLOBBER}&& git fetch $RELEASE_REMOTE --tags"
        fi
        fail "cannot fetch tags from $RELEASE_REMOTE (see the git output above) — refusing to publish against a stale tag list."
    fi
    [[ "$(git rev-parse --abbrev-ref HEAD)" == "main" ]] || fail "publish from main."

    # HEAD must be exactly the release remote's main — reject BOTH unpushed local commits
    # (which the tag would ship) AND being behind.
    LOCAL_SHA="$(git rev-parse HEAD)"
    REMOTE_SHA="$(git rev-parse "$RELEASE_REMOTE/main")"
    [[ "$LOCAL_SHA" == "$REMOTE_SHA" ]] \
        || fail "local main ($LOCAL_SHA) != $RELEASE_REMOTE/main ($REMOTE_SHA) — push or pull --ff-only so they match before publishing."
    [[ -z "$(git status --porcelain)" ]] || fail "working tree is dirty — commit before publishing."

    # Tag must be new AND strictly newer than the highest published tag —
    # a v0.9.0 after v0.10.3 would roll latest.json backwards.
    if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null \
       || git ls-remote --exit-code --tags "$RELEASE_REMOTE" "$TAG" >/dev/null 2>&1; then
        fail "tag $TAG already exists — pick the next version."
    fi
    HIGHEST=""
    while read -r t; do
        [[ "$t" =~ ^rapid-mac-v[0-9]+\.[0-9]+\.[0-9]+$ ]] || continue   # ignore prereleases / odd tags / engine tags
        if [[ -z "$HIGHEST" ]] || version_gt "$t" "$HIGHEST"; then HIGHEST="$t"; fi
    done < <(git ls-remote --tags --refs "$RELEASE_REMOTE" 'rapid-mac-v[0-9]*' 2>/dev/null | sed 's#.*/##')
    if [[ -n "$HIGHEST" ]]; then
        version_gt "$TAG" "$HIGHEST" \
            || fail "tag $TAG is not newer than the latest release $HIGHEST — refusing a backwards release."
        PREVIOUS_BUILD="$(git show "${HIGHEST}:apps/rapid-mac/Resources/Info.plist" \
            | plutil -extract CFBundleVersion raw -o - - 2>/dev/null || true)"
        [[ "$PREVIOUS_BUILD" =~ ^[1-9][0-9]*$ ]] \
            || fail "cannot read CFBundleVersion from $HIGHEST — Sparkle ordering cannot be verified."
        (( PLIST_BUILD > PREVIOUS_BUILD )) \
            || fail "CFBundleVersion $PLIST_BUILD must exceed $HIGHEST build $PREVIOUS_BUILD for Sparkle."
    fi

    # Public release RELIES on CI firing on the tag. Refuse if it can't.
    # MONOREPO: the workflow lives at the repository root, not under
    # apps/rapid-mac, and triggers on the ``rapid-mac-v*`` tag pattern.
    if ! awk '/^on:/{o=1} o&&/^[a-z]/&&!/^on:/{o=0} o&&/push:/{p=1} o&&p&&/tags:/{print "yes";exit}' \
            "$RELEASE_WORKFLOW" 2>/dev/null | grep -q yes; then
        warn "$RELEASE_WORKFLOW does NOT trigger on tag push — pushing $TAG would build/publish nothing."
        [[ "${FORCE_PUBLISH:-0}" == 1 ]] || fail "aborting (set FORCE_PUBLISH=1 to push the tag anyway)."
    fi

    note "pushing $TAG to $RELEASE_REMOTE → CI will build, notarise, and attach the DMG to the GitHub Release"
    git tag "$TAG"
    git push "$RELEASE_REMOTE" "$TAG"
    note 'watch it: gh run watch $(gh run list --workflow=rapid-mac-release.yml --limit=1 --json databaseId -q ".[0].databaseId")'
    printf '\033[32m✅ %s pushed. CI is cutting the public release. (Nothing was built locally.)\033[0m\n' "$TAG"
    exit 0
fi

# ── dogfood (default) : local build for you / testers ────────────────────
detect_identity
export CODESIGN_IDENTITY
if [[ -n "$CODESIGN_IDENTITY" ]]; then note "signing identity: $CODESIGN_IDENTITY"
else warn "no Developer ID — building AD-HOC (runs on your Mac via right-click→Open; not Gatekeeper-valid elsewhere)."; fi

# Notarise only when we have BOTH a Developer ID identity (ad-hoc can't be
# notarised) and a usable notary setup.
NOTARISE=0
if [[ -n "$CODESIGN_IDENTITY" ]] && notary_ready; then NOTARISE=1; fi

note "build.sh (SwiftUI .app + bundled sidecar)"
bash scripts/build.sh

# Staple the .app BEFORE packing the DMG so the shipped DMG contains a
# stapled app (matches CI + Gatekeeper offline first launch).
if [[ "$NOTARISE" == 1 ]]; then
    note "notarise + staple .app"
    ditto -c -k --keepParent "build/Rapid-MLX Desktop.app" "build/Rapid-MLX-Desktop.zip"
    bash scripts/notarize.sh "build/Rapid-MLX-Desktop.zip" "build/Rapid-MLX Desktop.app"
    DMG_FROM="stapled app"
else
    DMG_FROM="app (un-notarised)"
fi

note "dmg.sh + validate-dmg (DMG built from the $DMG_FROM)"
bash scripts/dmg.sh
bash scripts/validate-dmg.sh build/rapid-mlx-desktop.dmg

if [[ "$NOTARISE" == 1 ]]; then
    note "notarise + staple DMG"
    bash scripts/notarize.sh build/rapid-mlx-desktop.dmg build/rapid-mlx-desktop.dmg
    xcrun stapler validate build/rapid-mlx-desktop.dmg
    SHAREABLE="notarised — safe to hand to testers"
else
    SHAREABLE="NOT notarised — local use only (right-click → Open)"
fi

printf '\033[32m✅ dogfood DMG ready: build/rapid-mlx-desktop.dmg\n   %s\n   No tag pushed, no GitHub Release, no CI. Public release: %s --publish rapid-mac-vX.Y.Z\033[0m\n' \
    "$SHAREABLE" "$0"
