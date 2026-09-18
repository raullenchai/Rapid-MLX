#!/usr/bin/env python3
"""Regenerate the deterministic Rapid-MLX DMG Finder layout template.

Writes ``Resources/finder-layout.DS_Store`` — the committed ``.DS_Store``
that ``configure-dmg-layout.sh`` copies onto the DMG volume root — from the
declared layout below, using the same ``ds_store`` module Finder-layout
tooling such as dmgbuild uses. The template is never edited by hand: change
a constant here, re-run, and let ``verify-dmg-layout.py`` gate the result.

Records written, all of which Finder reads when it opens the volume:

  ``.``/``bwsp``   browser-window settings — Finder's own schema: a
                   ``WindowBounds`` ``"{{x, y}, {w, h}}"`` string plus the
                   sidebar/toolbar/statusbar/pathbar/tab-view switches (all
                   off, so the install page is a bare icon window).
  ``.``/``icvp``   icon-view settings — icon/text size, free arrangement,
                   the ``backgroundColor*`` triple Finder reads
                   unconditionally, and ``backgroundType`` 2 with a
                   ``backgroundImageAlias`` pointing at
                   ``.background/background.png``.
  ``.``/``icvl``   view mode (``icnv`` = icon view).
  ``.``/``vSrn``   Finder view-settings version (always 1).
  ``<item>``/``Iloc``  icon positions for the three volume items.

Both plist records MUST be flat bplists whose root object is a dictionary.
Finder on macOS 14 and 26.6 aborts with
``-[__NSCFData count]: unrecognized selector`` when the root is a ``<data>``
object instead (a bplist wrapped in another bplist), which is what the first
template shipped in 0.14.0–0.14.2 contained (#3468). ``ds_store``'s
``PlistCodec`` produces a flat plist when it is handed a ``dict``; handing it
pre-serialised bytes is what created the nested form. This script therefore
assigns dicts only, and the verifier now rejects any nested encoding.

Everything else was measured against a live Finder (macOS 26.5) with
dmgbuild's output as the reference implementation, one difference at a time:

  - Finder silently ignores the whole icvp record (default icon size, no
    background) unless ``backgroundColorRed/Green/Blue`` are present, even
    in image-background mode.
  - The background alias resolves by path, not CNID, and only when it
    carries the volume mount point (tag 0x0013, ``/Volumes/Rapid-MLX
    Desktop``) alongside the volume-relative POSIX path. That mount point is
    a property of the volume name, not of the build host, so it is the one
    ``/Volumes/...`` string the template is allowed to carry. The record is
    built here with ``mac_alias`` from constants only (zero CNIDs, epoch
    dates), so the template is byte-for-byte reproducible.

Usage:
    uv run --no-project --with ds_store --with mac_alias \
        apps/rapid-mac/scripts/make-dmg-layout.py
    python3 apps/rapid-mac/scripts/verify-dmg-layout.py \\
        apps/rapid-mac/Resources/finder-layout.DS_Store
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from ds_store import DSStore
    from mac_alias import Alias, TargetInfo, VolumeInfo
    from mac_alias.alias import mac_epoch
except ImportError:  # pragma: no cover - documented in the usage line
    print(
        "make-dmg-layout: the `ds_store` and `mac_alias` modules are required; run via\n"
        "  uv run --no-project --with ds_store --with mac_alias "
        "apps/rapid-mac/scripts/make-dmg-layout.py",
        file=sys.stderr,
    )
    raise SystemExit(2)

TEMPLATE = (
    Path(__file__).resolve().parent.parent / "Resources" / "finder-layout.DS_Store"
)

# Install window: 720x460 at (180, 120), matching the 720x460 background
# raster. Finder's WindowBounds string is "{{x, y}, {width, height}}" in
# top-left screen coordinates.
WINDOW_ORIGIN = (180, 120)
WINDOW_SIZE = (720, 460)

ICON_SIZE = 96.0
TEXT_SIZE = 13.0
GRID_SPACING = 100.0

# Icon centres. ``.background`` is parked below the window fold so Finder
# never draws it on the install page even with hidden files shown — see
# configure-dmg-layout.sh and verify-dmg-layout.py for the rationale.
ICON_POSITIONS = {
    "Rapid-MLX Desktop.app": (180, 228),
    "Applications": (540, 228),
    ".background": (100, 560),
}

VOLUME_NAME = "Rapid-MLX Desktop"
# Finder mounts a disk image at /Volumes/<volume name>; the alias needs this
# exact string (tag 0x0013) to resolve the background by path.
VOLUME_MOUNT_POINT = f"/Volumes/{VOLUME_NAME}"
BACKGROUND_DIR = ".background"
BACKGROUND_FILE = "background.png"


def background_alias() -> bytes:
    """v2 Alias Manager record for ``.background/background.png``.

    Built from constants only — no CNIDs (0), epoch dates, blank creator and
    type codes — so the bytes never depend on the machine that ran this
    script. Finder resolves it through the mount point + POSIX path pair.

    The mount point is the fixed ``/Volumes/Rapid-MLX Desktop`` string, which
    is a property of the volume name, not the build host. It is safe even when
    macOS assigns a suffixed mount point (``/Volumes/Rapid-MLX Desktop 1``,
    when a same-named volume is already mounted): verified by opening two
    same-named DMGs concurrently on macOS 26.5, Finder resolves the background
    from the *current* volume via the volume-relative path, so the install
    page renders correctly on the suffixed mount. This mirrors dmgbuild, which
    writes the same fixed mount point.
    """
    volume = VolumeInfo(
        name=VOLUME_NAME,
        creation_date=mac_epoch,
        fs_type=b"H+",
        disk_type=0,
        attribute_flags=0,
        fs_id=b"\0\0",
        posix_path=VOLUME_MOUNT_POINT,
    )
    target = TargetInfo(
        kind=0,  # file
        filename=BACKGROUND_FILE,
        folder_cnid=0,
        cnid=0,
        creation_date=mac_epoch,
        creator_code=b"\0\0\0\0",
        type_code=b"\0\0\0\0",
        levels_from=-1,
        levels_to=-1,
        folder_name=BACKGROUND_DIR,
        cnid_path=None,
        carbon_path=f"{VOLUME_NAME}:{BACKGROUND_DIR}:{BACKGROUND_FILE}",
        posix_path=f"/{BACKGROUND_DIR}/{BACKGROUND_FILE}",
    )
    return Alias(
        appinfo=b"\0\0\0\0", version=2, volume=volume, target=target
    ).to_bytes()


def browser_window_settings() -> dict:
    (x, y), (w, h) = WINDOW_ORIGIN, WINDOW_SIZE
    return {
        "ContainerShowSidebar": False,
        "PreviewPaneVisibility": False,
        "ShowPathbar": False,
        "ShowSidebar": False,
        "ShowStatusBar": False,
        "ShowTabView": False,
        "ShowToolbar": False,
        "SidebarWidth": 0,
        "WindowBounds": f"{{{{{x}, {y}}}, {{{w}, {h}}}}}",
    }


def icon_view_settings() -> dict:
    return {
        "viewOptionsVersion": 1,
        "backgroundType": 2,
        "backgroundImageAlias": background_alias(),
        # Read unconditionally by Finder even in image mode; without the
        # triple the whole record is dropped.
        "backgroundColorRed": 1.0,
        "backgroundColorGreen": 1.0,
        "backgroundColorBlue": 1.0,
        "arrangeBy": "none",
        "gridOffsetX": 0.0,
        "gridOffsetY": 0.0,
        "gridSpacing": GRID_SPACING,
        "iconSize": ICON_SIZE,
        "textSize": TEXT_SIZE,
        "labelOnBottom": True,
        "showIconPreview": True,
        "showItemInfo": False,
        "scrollPositionX": 0.0,
        "scrollPositionY": 0.0,
    }


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build into a sibling temp file and atomically replace the destination
    # only once it is complete: an exception or interruption mid-write must
    # never leave the repository without its known-good template.
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        with DSStore.open(str(tmp), "w+") as store:
            store["."]["vSrn"] = ("long", 1)
            # View-mode record (icon view), as dmgbuild writes for its
            # icon-view default and as Finder itself persists.
            store["."]["icvl"] = ("type", "icnv")
            # Assign dicts, never bytes: PlistCodec serialises a dict as a flat
            # bplist, which is the only encoding Finder accepts (#3468).
            store["."]["bwsp"] = browser_window_settings()
            store["."]["icvp"] = icon_view_settings()
            for name, position in ICON_POSITIONS.items():
                store[name]["Iloc"] = position
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else TEMPLATE
    write_template(target)
    print(f"make-dmg-layout: wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
