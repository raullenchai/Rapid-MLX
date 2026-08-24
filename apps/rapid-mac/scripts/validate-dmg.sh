#!/usr/bin/env bash
# Post-build validation for build/rapid-mlx-desktop.dmg.
#
# Closes audit P1 `release.yml:106–110` — "DMG bg image / icon
# positions / Applications symlink not validated post-build." The
# Applications link, branded background and persisted Finder layout
# are all load-bearing first-install UX. A staging regression must
# fail the release instead of silently publishing a plain disk image.
#
# Steps:
#   1. Attach the DMG read-only at its normal /Volumes location.
#   2. Assert exactly one *.app exists at the root with a parseable
#      Info.plist + the expected ``com.rapidmlx.rapid`` bundle id.
#      Strict default (#164): only ``Rapid-MLX Desktop.app`` passes.
#      Pass ``--allow-legacy`` or set
#      ``RAPID_VALIDATE_DMG_ALLOW_LEGACY=1`` to also accept the
#      pre-v0.5.22 ``Rapid.app`` name (used when re-validating a
#      legacy build artifact locally).
#   3. Assert the Applications symlink exists and points at /Applications.
#   4. Assert the 720x460 branded background and Finder .DS_Store
#      are present.
#   5. Always detach on exit (trap), even on assertion failure.
#
# Usage:
#   scripts/validate-dmg.sh                                # uses build/rapid-mlx-desktop.dmg
#   scripts/validate-dmg.sh /path/to/some.dmg
#   scripts/validate-dmg.sh --allow-legacy                 # uses default DMG, accepts legacy
#   scripts/validate-dmg.sh /path/to/some.dmg --allow-legacy
#   RAPID_VALIDATE_DMG_ALLOW_LEGACY=1 scripts/validate-dmg.sh
#
# Exit code 0 ⇒ DMG looks shippable. Non-zero ⇒ release blocked.

set -euo pipefail

# Parse args: --allow-legacy can appear in any position.
ALLOW_LEGACY="${RAPID_VALIDATE_DMG_ALLOW_LEGACY:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DMG=""
for arg in "$@"; do
    case "$arg" in
        --allow-legacy) ALLOW_LEGACY=1 ;;
        --*) echo "validate-dmg: unknown option $arg" >&2; exit 1 ;;
        *)
            if [[ -z "$DMG" ]]; then
                DMG="$arg"
            else
                echo "validate-dmg: too many positional args" >&2
                exit 1
            fi
            ;;
    esac
done
DMG="${DMG:-$ROOT/build/rapid-mlx-desktop.dmg}"

if [[ ! -f "$DMG" ]]; then
    echo "validate-dmg: DMG not found at $DMG" >&2
    exit 1
fi

MOUNT=""
DEVICE=""
ATTACHED=0

cleanup() {
    if [[ "$ATTACHED" -eq 1 ]]; then
        local detach_target="${DEVICE:-$MOUNT}"
        if [[ -n "$detach_target" ]]; then
            hdiutil detach "$detach_target" -quiet \
                || hdiutil detach "$detach_target" -force -quiet \
                || true
        fi
    fi
}
trap cleanup EXIT

echo "==> attaching $DMG as a user-mounted volume"
# Let DiskImages choose /Volumes/<volume name>, matching a real double-click.
# An arbitrary mktemp mount point makes Finder identify the disk by the random
# directory name and can resolve .DS_Store aliases against a stale prior mount.
ATTACH_OUTPUT="$(hdiutil attach "$DMG" -nobrowse -readonly)"
ATTACHED=1
DEVICE="$(printf '%s\n' "$ATTACH_OUTPUT" | awk '$1 ~ /^\/dev\// { print $1; exit }')"
MOUNT="$(printf '%s\n' "$ATTACH_OUTPUT" | awk -F '\t' 'NF >= 3 && $3 != "" { print $3 }' | tail -1)"
[[ -n "$MOUNT" && -d "$MOUNT" ]] || {
    echo "$ATTACH_OUTPUT" >&2
    echo "validate-dmg: FAIL — could not determine mounted volume path" >&2
    exit 1
}
echo "==> mounted at $MOUNT"

fail() {
    echo "validate-dmg: FAIL — $*" >&2
    exit 1
}

# 1. Exactly one *.app at the root (rules out a typoed bundle name
# that would still pass a hard-coded check).
#
# Default (strict, what CI uses): only the v0.5.22 canonical name
# "Rapid-MLX Desktop.app" passes — a silent regression that
# resurrects the old "Rapid.app" name fails the workflow instead
# of slipping through. The ``--allow-legacy`` opt-in (parsed
# above) re-admits the legacy name for local re-validation of
# older artifacts.
APPS=()
LEGACY_ARTIFACT=0
while IFS= read -r entry; do
    [[ -n "$entry" ]] && APPS+=("$entry")
done < <(find "$MOUNT" -maxdepth 1 -type d -name "*.app" -not -name ".*" -print)

[[ "${#APPS[@]}" -ge 1 ]] || fail "no *.app at root of DMG (found: $(ls -1 "$MOUNT"))"
[[ "${#APPS[@]}" -eq 1 ]] || fail "expected exactly one *.app at root, found ${#APPS[@]}: ${APPS[*]}"
APP="${APPS[0]}"
APP_NAME="$(basename "$APP")"
case "$APP_NAME" in
    "Rapid-MLX Desktop.app")
        echo "==> bundle: $APP_NAME"
        ;;
    "Rapid.app")
        if [[ "$ALLOW_LEGACY" == "1" ]]; then
            LEGACY_ARTIFACT=1
            echo "==> bundle: $APP_NAME (legacy name accepted via RAPID_VALIDATE_DMG_ALLOW_LEGACY=1)"
        else
            fail "legacy bundle name 'Rapid.app' found (expected 'Rapid-MLX Desktop.app'). Re-run with RAPID_VALIDATE_DMG_ALLOW_LEGACY=1 to accept legacy artifacts."
        fi
        ;;
    *)
        fail "unexpected bundle name '$APP_NAME' (expected 'Rapid-MLX Desktop.app')"
        ;;
esac

# 2. Info.plist parses + has the bundle id we expect.
INFO="$APP/Contents/Info.plist"
[[ -f "$INFO" ]] || fail "Info.plist missing inside $APP_NAME"
BUNDLE_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$INFO" 2>/dev/null || true)"
[[ -n "$BUNDLE_ID" ]] || fail "could not read CFBundleIdentifier from Info.plist"
echo "==> bundle id: $BUNDLE_ID"
[[ "$BUNDLE_ID" == "com.rapidmlx.rapid" ]] || fail "unexpected bundle id '$BUNDLE_ID' (expected 'com.rapidmlx.rapid')"

# 3. Applications symlink at the root, pointing at /Applications.
APPS_LINK="$MOUNT/Applications"
[[ -L "$APPS_LINK" ]] || fail "Applications drop-target symlink missing at DMG root"
TARGET="$(readlink "$APPS_LINK")"
[[ "$TARGET" == "/Applications" ]] || fail "Applications symlink points at '$TARGET', expected '/Applications'"
echo "==> Applications -> $TARGET"

# 4. Branded Finder presentation. Legacy mode exists specifically to inspect
# pre-v0.5.22 artifacts, which predate this presentation contract.
if [[ "$LEGACY_ARTIFACT" == "1" ]]; then
    echo "==> Finder presentation: skipped for pre-v0.5.22 legacy artifact"
    echo "==> validate-dmg: OK"
    exit 0
fi

# Dot-prefixed support files remain hidden
# from the user's icon view but must survive both UDRW -> UDZO conversion and
# release notarisation.
BACKGROUND="$MOUNT/.background/background.png"
[[ -f "$BACKGROUND" ]] || fail "Finder background missing at .background/background.png"
BG_WIDTH="$(sips -g pixelWidth "$BACKGROUND" | awk '/pixelWidth:/ {print $2}')"
BG_HEIGHT="$(sips -g pixelHeight "$BACKGROUND" | awk '/pixelHeight:/ {print $2}')"
[[ "$BG_WIDTH" == "720" && "$BG_HEIGHT" == "460" ]] \
    || fail "Finder background is ${BG_WIDTH}x${BG_HEIGHT}, expected 720x460"
[[ -s "$MOUNT/.DS_Store" ]] || fail "Finder layout .DS_Store missing or empty"

# A present PNG is not enough: structurally parse the active icvp blob and
# verify its backgroundImageAlias points to the volume-relative image.
python3 "$ROOT/scripts/verify-dmg-background.py" "$MOUNT/.DS_Store" \
    || fail "Finder background alias is missing or invalid"

# Do not stop at file presence: a .DS_Store can carry an absolute alias to the
# build-time mount and look complete on disk while Finder silently falls back
# to a blank, auto-arranged window for users. Open the final read-only image
# and read back the presentation through Finder itself.
FINDER_LAYOUT="$(osascript - "$MOUNT" <<'APPLESCRIPT'
on pointText(p)
    return (item 1 of p as text) & "," & (item 2 of p as text)
end pointText

on rectText(r)
    return (item 1 of r as text) & "," & (item 2 of r as text) & "," & (item 3 of r as text) & "," & (item 4 of r as text)
end rectText

on run argv
    set volumeFolder to POSIX file (item 1 of argv) as alias
    tell application "Finder"
        open volumeFolder
        delay 0.5
        set dmgWindow to container window of volumeFolder
        set appPosition to position of item "Rapid-MLX Desktop.app" of volumeFolder
        set applicationsPosition to position of item "Applications" of volumeFolder
        set iconSizeValue to icon size of icon view options of dmgWindow
        set windowBounds to bounds of dmgWindow
        close dmgWindow
        return my pointText(appPosition) & "|" & my pointText(applicationsPosition) & "|" & (iconSizeValue as text) & "|" & my rectText(windowBounds)
    end tell
end run
APPLESCRIPT
)" || fail "Finder could not read the persisted DMG layout"

[[ "$FINDER_LAYOUT" == "180,228|540,228|96|180,120,900,580" ]] \
    || fail "unexpected Finder layout '$FINDER_LAYOUT' (expected app|Applications|icon|bounds = 180,228|540,228|96|180,120,900,580)"
echo "==> Finder presentation: ${BG_WIDTH}x${BG_HEIGHT}; $FINDER_LAYOUT"

echo "==> validate-dmg: OK"
