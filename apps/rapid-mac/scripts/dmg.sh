#!/usr/bin/env bash
# dmg.sh — wrap build/"Rapid-MLX Desktop.app" into a draggable
# rapid-mlx-desktop.dmg (legacy alias Rapid.dmg published alongside
# in CI — see release.yml R2 upload step).
#
# We deliberately use Apple's built-in `hdiutil`, Finder and `sips` (and don't
# pull in `create-dmg` from Homebrew) so a contributor can build a release on a
# clean machine without `brew install`. ``configure-dmg-layout.sh`` applies the
# custom background and left-to-right icon positions while the intermediate
# image is writable.
#
# Layout inside the volume:
#   Rapid-MLX Desktop.app  ← the SwiftUI executable bundle
#   Applications  ───────┐ ← symlink so the user drags the app onto it
#                        └→ /Applications
#
# The DMG is codesigned with the same identity as the .app
# (CODESIGN_IDENTITY env var; ad-hoc "-" by default). For a real
# release the caller signs with a Developer ID identity and then runs
# scripts/notarize.sh to notarise + staple the DMG — that is what makes
# it install with zero Gatekeeper warnings, even offline.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
APP="$BUILD/Rapid-MLX Desktop.app"
DMG="$BUILD/rapid-mlx-desktop.dmg"
STAGING="$BUILD/dmg-staging"
UDRW="$BUILD/rapid-mlx-desktop.udrw.dmg"
VOL_NAME="Rapid-MLX Desktop"
MOUNT=""

cleanup() {
    if [[ -n "$MOUNT" && -d "$MOUNT" ]] && mount | grep -Fq " on $MOUNT "; then
        hdiutil detach "$MOUNT" -quiet 2>/dev/null \
            || hdiutil detach "$MOUNT" -force -quiet 2>/dev/null \
            || true
    fi
    [[ -n "$MOUNT" ]] && rmdir "$MOUNT" 2>/dev/null || true
    rm -rf "$STAGING"
    rm -f "$UDRW"
}
trap cleanup EXIT

if [[ ! -d "$APP" ]]; then
    echo "==> Rapid-MLX Desktop.app missing — running build.sh first"
    bash "$ROOT/scripts/build.sh"
fi

echo "==> staging $STAGING"
rm -rf "$STAGING"
mkdir -p "$STAGING"
# `cp -R` follows the SwiftUI .app bundle structure correctly; -p
# preserves the executable bit so the embedded Mach-O launches.
cp -R "$APP" "$STAGING/Rapid-MLX Desktop.app"
ln -s /Applications "$STAGING/Applications"

echo "==> hdiutil create writable layout image"
rm -f "$UDRW" "$DMG"
# Finder can only persist icon positions and the background picture on a
# writable volume. Build UDRW first, apply the presentation, then convert to
# the same zlib-compressed UDZO shipping format used previously.
hdiutil create \
    -volname "$VOL_NAME" \
    -srcfolder "$STAGING" \
    -ov \
    -fs HFS+ \
    -format UDRW \
    "$UDRW" \
    >/dev/null

MOUNT="$(mktemp -d "${TMPDIR:-/tmp}/rapid-dmg-layout-XXXXXX")"
hdiutil attach "$UDRW" -nobrowse -mountpoint "$MOUNT" -quiet
bash "$ROOT/scripts/configure-dmg-layout.sh" "$MOUNT"
hdiutil detach "$MOUNT" -quiet || hdiutil detach "$MOUNT" -force -quiet
rmdir "$MOUNT" 2>/dev/null || true
MOUNT=""

echo "==> hdiutil convert UDRW -> UDZO ($DMG)"
hdiutil convert "$UDRW" -format UDZO -ov -o "$DMG" >/dev/null
rm -f "$UDRW"

SIGN_IDENTITY="${CODESIGN_IDENTITY:--}"
if [[ "$SIGN_IDENTITY" == "-" ]]; then
    echo "==> ad-hoc codesign $DMG"
else
    echo "==> Developer ID codesign $DMG ($SIGN_IDENTITY)"
fi
codesign --force --sign "$SIGN_IDENTITY" "$DMG"
codesign --verify "$DMG"

echo "==> hdiutil verify (CRC + structure)"
hdiutil verify "$DMG" >/dev/null

SIZE="$(du -h "$DMG" | cut -f1)"
echo
echo "rapid-mlx-desktop.dmg ready at: $DMG ($SIZE)"
echo "Test mount with:   open '$DMG'"
