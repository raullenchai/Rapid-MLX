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
BACKGROUND_SOURCE="$ROOT/Resources/dmg-background.png"
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
rm -f "$MOUNT/.DS_Store"
# Ship the pre-rendered PNG instead of relying on ImageIO's SVG support on the
# release runner. The adjacent SVG remains the editable design source.
cp "$BACKGROUND_SOURCE" "$BACKGROUND"

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

# Reopen the volume and read the values back through Finder before detaching.
# File existence alone is insufficient: Finder can create .DS_Store before the
# final positions and window options have been flushed.
PERSISTED_LAYOUT="$(osascript - "$MOUNT" <<'APPLESCRIPT'
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
        delay 0.4
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
)"

EXPECTED_LAYOUT="180,228|540,228|96|180,120,900,580"
if [[ "$PERSISTED_LAYOUT" != "$EXPECTED_LAYOUT" ]]; then
    echo "configure-dmg-layout: Finder persisted unexpected layout '$PERSISTED_LAYOUT' (expected '$EXPECTED_LAYOUT')" >&2
    exit 1
fi

# Finder scripting cannot resolve a dot-prefixed background file back to a
# Finder item. Parse the icvp blob structurally instead, so unrelated or stale
# strings elsewhere in .DS_Store cannot satisfy the background contract.
python3 "$ROOT/scripts/verify-dmg-background.py" "$MOUNT/.DS_Store" >/dev/null

sync
echo "==> Finder layout: 720x460, app (180,228) -> Applications (540,228)"
