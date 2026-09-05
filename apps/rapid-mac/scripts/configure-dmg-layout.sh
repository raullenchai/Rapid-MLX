#!/usr/bin/env bash
# Apply Rapid-MLX's Finder presentation to a mounted, writable DMG volume.
#
# Both the canonical full DMG and the public slim bootstrapper DMG call this
# helper. Keeping the presentation here prevents the two release paths from
# drifting into different first-install experiences.
#
# The layout is NOT written by Finder. Finder object AppleEvents against a
# mounted volume intermittently hang or silently fail to persist a .DS_Store
# on macOS 26, so the release path cannot depend on them (#2240). Instead we
# install a committed, versioned, deterministic .DS_Store template
# (Resources/finder-layout.DS_Store) whose icon positions, window bounds, icon
# size and volume-relative background alias are fixed and verified
# structurally (scripts/verify-dmg-layout.py) — no Finder scripting.
#
# Usage: scripts/configure-dmg-layout.sh /Volumes/Rapid-MLX\ Desktop
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MOUNT="${1:-}"
BACKGROUND_SOURCE="$ROOT/Resources/dmg-background.png"
BACKGROUND_DIR="$MOUNT/.background"
BACKGROUND="$BACKGROUND_DIR/background.png"
LAYOUT_TEMPLATE="$ROOT/Resources/finder-layout.DS_Store"

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
if [[ ! -f "$LAYOUT_TEMPLATE" ]]; then
    echo "configure-dmg-layout: layout template missing: $LAYOUT_TEMPLATE" >&2
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

# Install the committed deterministic layout template. .DS_Store must be
# removed first so the template is the sole writer and the volume starts from
# a clean state; `sync` flushes it before the image is detached and repacked.
cp "$LAYOUT_TEMPLATE" "$MOUNT/.DS_Store"
sync

# The template must match the declared Rapid-MLX layout structurally (window
# bounds, icon view, icon positions, volume-relative background alias, and no
# build-host/mount strings). verify-dmg-layout.py subsumes the background
# alias check and replaces Finder readback for persistence verification.
python3 "$ROOT/scripts/verify-dmg-layout.py" "$MOUNT/.DS_Store"

sync
echo "==> Finder layout: 720x460, app (180,228) -> Applications (540,228)"
