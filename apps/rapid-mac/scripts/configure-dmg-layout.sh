#!/usr/bin/env bash
# Apply Rapid-MLX's Finder presentation to a mounted, writable DMG volume.
#
# Both the canonical full DMG and the public slim bootstrapper DMG call this
# helper. Keeping the presentation here prevents the two release paths from
# drifting into different first-install experiences.
#
# Usage: scripts/configure-dmg-layout.sh /Volumes/Rapid-MLX\ Desktop
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MOUNT="${1:-}"
BACKGROUND_SOURCE="$ROOT/Resources/dmg-background.svg"
BACKGROUND_DIR="$MOUNT/.background"
BACKGROUND="$BACKGROUND_DIR/background.png"

if [[ -z "$MOUNT" || ! -d "$MOUNT" ]]; then
    echo "configure-dmg-layout: writable mount point required (got '$MOUNT')" >&2
    exit 1
fi
if [[ ! -f "$MOUNT/Rapid-MLX Desktop.app/Contents/Info.plist" ]]; then
    echo "configure-dmg-layout: Rapid-MLX Desktop.app missing at volume root" >&2
    exit 1
fi
if [[ ! -L "$MOUNT/Applications" || "$(readlink "$MOUNT/Applications")" != "/Applications" ]]; then
    echo "configure-dmg-layout: Applications -> /Applications symlink missing" >&2
    exit 1
fi
if [[ ! -f "$BACKGROUND_SOURCE" ]]; then
    echo "configure-dmg-layout: background source missing: $BACKGROUND_SOURCE" >&2
    exit 1
fi

mkdir -p "$BACKGROUND_DIR"
# sips ships with macOS and can rasterise SVG through ImageIO. PNG is used in
# the volume because Finder's background-picture support is reliable for PNG
# across every macOS version the app supports.
sips -s format png "$BACKGROUND_SOURCE" --out "$BACKGROUND" >/dev/null

WIDTH="$(sips -g pixelWidth "$BACKGROUND" | awk '/pixelWidth:/ {print $2}')"
HEIGHT="$(sips -g pixelHeight "$BACKGROUND" | awk '/pixelHeight:/ {print $2}')"
if [[ "$WIDTH" != "720" || "$HEIGHT" != "460" ]]; then
    echo "configure-dmg-layout: background raster is ${WIDTH}x${HEIGHT}, expected 720x460" >&2
    exit 1
fi

# Finder is the supported writer for its private .DS_Store layout data. Refer
# to the mounted volume by POSIX alias rather than by display name so a second
# Rapid-MLX image mounted by a developer cannot receive this window layout.
osascript - "$MOUNT" <<'APPLESCRIPT'
on run argv
    set mountPath to item 1 of argv
    set volumeFolder to POSIX file mountPath as alias

    tell application "Finder"
        open volumeFolder
        delay 0.4

        set dmgWindow to container window of volumeFolder
        set current view of dmgWindow to icon view
        set toolbar visible of dmgWindow to false
        set statusbar visible of dmgWindow to false
        set pathbar visible of dmgWindow to false
        set sidebar width of dmgWindow to 0
        set bounds of dmgWindow to {180, 120, 900, 580}

        set iconOptions to icon view options of dmgWindow
        set arrangement of iconOptions to not arranged
        set icon size of iconOptions to 96
        set text size of iconOptions to 13
        set shows item info of iconOptions to false
        set shows icon preview of iconOptions to true
        -- Store a volume-relative HFS alias in .DS_Store. A POSIX alias to
        -- the temporary build mount works during packaging but breaks when
        -- the user later mounts the DMG at /Volumes/Rapid-MLX Desktop;
        -- Finder then discards the background and falls back to auto-arrange.
        set background picture of iconOptions to file ".background:background.png" of volumeFolder

        set position of item "Rapid-MLX Desktop.app" of volumeFolder to {180, 228}
        set position of item "Applications" of volumeFolder to {540, 228}

        update volumeFolder without registering applications
        delay 1
        close dmgWindow
    end tell
end run
APPLESCRIPT

# Finder sometimes flushes the .DS_Store shortly after the close command.
# Bound the wait so a broken Finder session fails the package instead of
# silently publishing an unstyled image.
for _ in 1 2 3 4 5; do
    [[ -s "$MOUNT/.DS_Store" ]] && break
    sleep 1
done
if [[ ! -s "$MOUNT/.DS_Store" ]]; then
    echo "configure-dmg-layout: Finder did not persist .DS_Store" >&2
    exit 1
fi

sync
echo "==> Finder layout: 720x460, app (180,228) -> Applications (540,228)"
