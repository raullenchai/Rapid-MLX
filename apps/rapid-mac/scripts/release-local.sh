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
# ── PUBLIC (--publish) — RETIRED ────────────────────────────────────────
#   The public release path is retired fail-closed (#2301). A local tag push
#   would bypass the signed Desktop candidate gate, the live release-blocker /
#   main-head checks, and the protected `rapid-mac-tag` reviewer approval that
#   the canonical flow enforces. ``--publish`` now exits non-zero immediately
#   (before sourcing RAPID_RELEASE_ENV or probing identity) and directs the
#   operator to the canonical flow: bump PR → auto-release.yml validates a signed
#   candidate at the exact commit → reviewer approves at the protected gate →
#   Desktop tag + DMG are published. For a local build, run without args or
#   ``--check`` — no tag, no release is ever created locally.
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
if [[ $# -eq 0 ]]; then
    MODE="dogfood"
elif [[ $# -eq 1 && "$1" == "--check" ]]; then
    MODE="check"
elif [[ $# -eq 2 && "$1" == "--publish" ]]; then
    MODE="publish"
else
    {
        echo "usage:"
        echo "  $0                              # dogfood build (no args)"
        echo "  $0 --check                      # verify signing/notary setup"
        echo "  $0 --publish ...                # RETIRED — see --publish refusal (use the bump PR flow)"
    } >&2
    exit 2
fi

note() { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mrelease-local: %s\033[0m\n' "$*" >&2; }

# ── --publish : RETIRED (fail-closed) ───────────────────────────────────────
# A local tag push can no longer cut a public Desktop release (#2301): it would
# bypass the signed Desktop candidate gate, the live release-blocker and
# main-head checks, and the protected `rapid-mac-tag` reviewer approval that the
# canonical flow enforces. We refuse HERE, immediately after strict arg parsing
# and BEFORE sourcing RAPID_RELEASE_ENV or probing the signing identity — a
# disabled public command must execute no operator-owned shell file and touch no
# secrets, not merely avoid pushing a tag. We deliberately do not build a second
# local approval system.
if [[ "$MODE" == "publish" ]]; then
    {
        printf '❌ --publish is retired and cannot be used to release the Desktop app.\n'
        printf '\n'
        printf 'A local tag push bypasses the release-safety gates this repo now requires\n'
        printf '(#2301): the signed/notarised Desktop candidate at the exact commit, the\n'
        printf 'live release-blocker and main-head checks, and the protected rapid-mac-tag\n'
        printf 'reviewer approval. Nothing was sourced, probed, fetched, or pushed.\n'
        printf '\n'
        printf 'Use the canonical release flow instead:\n'
        printf '  1. open a PR with subject  chore: bump version to X.Y.Z\n'
        printf '  2. merge it to main — auto-release.yml builds + validates a signed\n'
        printf '     Desktop candidate at the exact commit and gathers live evidence\n'
        printf '  3. at the protected rapid-mac-tag gate, approve the printed exact SHA\n'
        printf '     — the Desktop tag and DMG are then published automatically.\n'
        printf '\n'
        printf 'For a local build only (no tag, no release): run %s or %s --check.\n' "$0" "$0"
    } >&2
    exit 1
fi

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

printf '\033[32m✅ dogfood DMG ready: build/rapid-mlx-desktop.dmg\n   %s\n   No tag pushed, no GitHub Release, no CI.\n   To RELEASE to users use the canonical flow (bump PR → auto-release → rapid-mac-tag approval), not a local push.\033[0m\n' \
    "$SHAREABLE"
