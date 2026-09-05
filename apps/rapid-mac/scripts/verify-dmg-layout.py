#!/usr/bin/env python3
"""Structurally verify the deterministic Rapid-MLX DMG Finder layout.

The DMG install-page layout (icon positions, window bounds, icon size and
volume-relative background picture) is packaged from a committed, versioned
`.DS_Store` template rather than being written by Finder. On macOS 26 Finder
AppleEvents against a mounted volume intermittently hang or silently fail to
persist the `.DS_Store`, so the release path must not depend on them. This
validator parses the template (or any produced `.DS_Store`) directly and
asserts it matches the declared layout:

  - exactly one window-bounds (`bwsp`) record,
  - exactly one icon-view (`icvp`) record with volume-relative background
    alias, image backgroundType and the expected icon/text metrics,
  - icon-position (`Iloc`) records for exactly the two volume items
    (``Rapid-MLX Desktop.app`` and ``Applications``) at the expected spots,
  - no build-host / temp-mount / absolute-mount strings anywhere in the file,
    so the template stays deterministic and remountable on any machine.

The `ds_store` writer used to generate the template nests the `icvp` / `bwsp`
bplist payloads one level deep as a bplist ``<data>`` object, so both the
nested and the flat (Finder-produced) encodings are accepted here.

Usage: scripts/verify-dmg-layout.py /path/to/.DS_Store
"""

from __future__ import annotations

import plistlib
import struct
import sys
from pathlib import Path

BWSB_BLOB_MARKER = b"bwspblob"
ICVP_BLOB_MARKER = b"icvpblob"
ILOC_BLOB_MARKER = b"Ilocblob"

# The committed template (generated with the `ds_store`/`mac_alias` modules)
# stores the HFS path with plain ``:`` separators --
# "Rapid-MLX Desktop:.background:background.png" -- while some Finder-written
# files insert a ``\x00`` before the filename ("...:.background:\x00background.png").
# Both encode the same volume-relative target, so the HFS path is compared
# after normalising any ``\x00`` away. The POSIX path is compared exactly.
EXPECTED_HFS_PATH = b"Rapid-MLX Desktop:.background:background.png"
EXPECTED_POSIX_PATH = b"/.background/background.png"
ALIAS_FIXED_HEADER_SIZE = 150

# Expected positions: app | Applications | iconSize | window bounds. This is
# the same contract the old Finder readback compared against:
#   "180,228|540,228|96|180,120,900,580"
EXPECTED_APP_POSITION = (180, 228)
EXPECTED_APPLICATIONS_POSITION = (540, 228)
EXPECTED_ICON_SIZE = 96.0
EXPECTED_TEXT_SIZE = 13.0
EXPECTED_BOUNDS = {"left": 180, "top": 120, "right": 900, "bottom": 580}

# Substrings that would make the template non-deterministic across hosts or
# bind it to a specific build mount. A .DS_Store carrying any of these must be
# rejected: the layout must stay volume-relative and remountable anywhere.
FORBIDDEN_SUBSTRINGS = (
    b"rapid-dmg-layout-",
    b"/tmp/",
    b"/var/folders/",
    b"raullenstudio",
    b"/Volumes/Rapid-MLX Desktop",
)


def _load_bplist(payload: bytes) -> object:
    """Load a bplist payload, unwrapping the `ds_store`-style nested <data>.

    ``ds_store`` wraps the inner `icvp`/`bwsp` bplist inside an outer bplist
    whose single object is a ``<data>`` blob of the inner bplist. plistlib
    therefore yields ``bytes`` that are themselves a ``bplist00``; unwrap
    those so callers always receive the inner dict.
    """
    value = plistlib.loads(payload)
    while isinstance(value, (bytes, bytearray)) and value.startswith(b"bplist00"):
        value = plistlib.loads(value)
    if isinstance(value, (bytes, bytearray)):
        raise ValueError("bplist payload is not a record dict")
    return value


def _find_records(data: bytes, marker: bytes) -> list[bytes]:
    """Return the raw payload of every ``<marker><len><payload>`` record."""
    records: list[bytes] = []
    cursor = 0
    while True:
        marker_pos = data.find(marker, cursor)
        if marker_pos < 0:
            return records
        length_offset = marker_pos + len(marker)
        payload_offset = length_offset + 4
        if payload_offset > len(data):
            raise ValueError(f"{marker.decode()} length field is truncated")
        payload_length = struct.unpack(">I", data[length_offset:payload_offset])[0]
        payload = data[payload_offset : payload_offset + payload_length]
        if len(payload) != payload_length:
            raise ValueError(f"{marker.decode()} blob payload is truncated")
        records.append(payload)
        cursor = marker_pos + 1


def _pascal_string(data: bytes, offset: int, capacity: int, field: str) -> bytes:
    if offset + capacity > len(data):
        raise ValueError(f"alias {field} field is truncated")
    length = data[offset]
    if length >= capacity:
        raise ValueError(f"alias {field} length exceeds its fixed field")
    return data[offset + 1 : offset + 1 + length]


def parse_alias_target(alias: bytes) -> tuple[bytes, bytes]:
    """Decode the v2 Alias Manager record and return its HFS/POSIX paths."""
    if len(alias) < ALIAS_FIXED_HEADER_SIZE + 4:
        raise ValueError("backgroundImageAlias is truncated")
    record_size = struct.unpack(">H", alias[4:6])[0]
    if record_size != len(alias):
        raise ValueError(
            f"backgroundImageAlias size field is {record_size}, actual {len(alias)}"
        )
    if struct.unpack(">H", alias[6:8])[0] != 2:
        raise ValueError("backgroundImageAlias is not a v2 Alias Manager record")
    if struct.unpack(">H", alias[8:10])[0] != 0:
        raise ValueError("backgroundImageAlias does not target a file")

    volume_name = _pascal_string(alias, 10, 28, "volume name")
    file_name = _pascal_string(alias, 50, 64, "file name")
    if volume_name != b"Rapid-MLX Desktop" or file_name != b"background.png":
        raise ValueError("backgroundImageAlias header targets the wrong volume or file")

    cursor = ALIAS_FIXED_HEADER_SIZE
    tags: dict[int, bytes] = {}
    terminated = False
    while cursor + 4 <= len(alias):
        tag, length = struct.unpack(">HH", alias[cursor : cursor + 4])
        cursor += 4
        if tag == 0xFFFF:
            if length != 0 or cursor != len(alias):
                raise ValueError("backgroundImageAlias has an invalid terminator")
            terminated = True
            break
        value_end = cursor + length
        if value_end > len(alias):
            raise ValueError(f"backgroundImageAlias tag 0x{tag:04x} is truncated")
        if tag in tags:
            raise ValueError(f"backgroundImageAlias repeats tag 0x{tag:04x}")
        tags[tag] = alias[cursor:value_end]
        cursor = value_end + (length % 2)

    if not terminated:
        raise ValueError("backgroundImageAlias has no terminator")
    try:
        parent = tags[0x0000]
        hfs_path = tags[0x0002]
        posix_path = tags[0x0012]
    except KeyError as exc:
        raise ValueError(
            f"backgroundImageAlias is missing path tag 0x{exc.args[0]:04x}"
        ) from None
    if parent != b".background":
        raise ValueError("backgroundImageAlias targets the wrong parent directory")
    return hfs_path, posix_path


def _iloc_filename(data: bytes, iloc_marker_pos: int) -> str:
    """Attribute an ``Iloc`` record to its owning filename.

    A .DS_Store B-tree leaf record is laid out as::

        <nlen:>I><filename:utf16be><code:4s><type:4s><value>

    The ``Ilocblob`` marker spans ``code``(``Iloc``) + ``type``(``blob``), so
    the UTF-16BE filename's final character ends immediately before the
    marker. Walk backward to find the ``>I`` nlen such that
    ``nlen*2 == marker_pos - (p + 4)``; that pins the filename uniquely.
    """
    for p in range(iloc_marker_pos - 4, max(0, iloc_marker_pos - 400), -1):
        span = iloc_marker_pos - (p + 4)
        if span <= 0 or span % 2:
            continue
        name_len = struct.unpack(">I", data[p : p + 4])[0]
        if name_len * 2 == span and name_len > 0:
            try:
                return data[p + 4 : iloc_marker_pos].decode("utf-16-be")
            except UnicodeDecodeError as exc:
                raise ValueError("Iloc record has a malformed owning filename") from exc
    raise ValueError("Iloc record could not be attributed to a filename")


def _parse_iloc_payload(payload: bytes) -> tuple[int, int]:
    """Decode ``(x, y)`` from an Iloc payload and sanity-check the flags."""
    if len(payload) != 16:
        raise ValueError("Iloc blob payload is not 16 bytes")
    x, y, flag1, flag2 = struct.unpack(">IIII", payload)
    # Third and fourth words are reserved flags; a change to their meaning
    # indicates a different (or corrupted) record layout.
    if flag1 != 0xFFFFFFFF or flag2 != 0xFFFF0000:
        raise ValueError("Iloc blob has unexpected reserved flags")
    return x, y


def verify(path: Path) -> None:
    data = path.read_bytes()

    bwsp_records = _find_records(data, BWSB_BLOB_MARKER)
    if len(bwsp_records) != 1:
        raise ValueError(f"expected exactly one bwsp record, found {len(bwsp_records)}")
    bounds = _load_bplist(bwsp_records[0])
    if not isinstance(bounds, dict) or bounds != EXPECTED_BOUNDS:
        raise ValueError(
            f"unexpected window bounds {bounds!r}, expected {EXPECTED_BOUNDS}"
        )

    icvp_records = _find_records(data, ICVP_BLOB_MARKER)
    if len(icvp_records) != 1:
        raise ValueError(f"expected exactly one icvp record, found {len(icvp_records)}")
    icvp = _load_bplist(icvp_records[0])
    if not isinstance(icvp, dict):
        raise ValueError("icvp record is not a dict")
    if icvp.get("backgroundType") != 2:
        raise ValueError("icvp backgroundType is not image mode (2)")
    if icvp.get("iconSize") != EXPECTED_ICON_SIZE:
        raise ValueError(
            f"icvp iconSize is {icvp.get('iconSize')!r}, expected {EXPECTED_ICON_SIZE}"
        )
    if icvp.get("textSize") != EXPECTED_TEXT_SIZE:
        raise ValueError(
            f"icvp textSize is {icvp.get('textSize')!r}, expected {EXPECTED_TEXT_SIZE}"
        )
    if icvp.get("showIconPreview") is not True:
        raise ValueError("icvp showIconPreview is not true")
    if icvp.get("showItemInfo") is not False:
        raise ValueError("icvp showItemInfo is not false")
    if icvp.get("labelOnBottom") is not True:
        raise ValueError("icvp labelOnBottom is not true")
    if icvp.get("arrangeBy") != "none":
        raise ValueError(
            f"icvp arrangeBy is {icvp.get('arrangeBy')!r}, expected 'none'"
        )

    alias = icvp.get("backgroundImageAlias")
    if not isinstance(alias, bytes):
        raise ValueError(
            "icvp backgroundImageAlias is missing or not binary alias data"
        )
    hfs_path, posix_path = parse_alias_target(alias)
    # Accept both separator variants for the HFS path, then require the exact
    # volume-relative preferences (never an absolute build mount path).
    normalized_hfs = hfs_path.replace(b"\x00", b"")
    if normalized_hfs != EXPECTED_HFS_PATH:
        raise ValueError(
            f"backgroundImageAlias HFS path {hfs_path!r} does not target .background/background.png"
        )
    if posix_path != EXPECTED_POSIX_PATH:
        raise ValueError(
            f"backgroundImageAlias POSIX path {posix_path!r} is not volume-relative"
        )

    # Collect every Iloc record and attribute each to its owning filename.
    positions: dict[str, tuple[int, int]] = {}
    iloc_cursor = 0
    while True:
        marker_pos = data.find(ILOC_BLOB_MARKER, iloc_cursor)
        if marker_pos < 0:
            break
        payload = _find_records(data[marker_pos:], ILOC_BLOB_MARKER)[0]
        name = _iloc_filename(data, marker_pos)
        positions[name] = _parse_iloc_payload(payload)
        iloc_cursor = marker_pos + 1

    if positions != {
        "Rapid-MLX Desktop.app": EXPECTED_APP_POSITION,
        "Applications": EXPECTED_APPLICATIONS_POSITION,
    }:
        raise ValueError(f"unexpected icon positions {positions!r}")

    # Determinism / no-host-embedding gate. Any of these means the template was
    # produced against a specific build mount or host and is not shippable.
    for substring in FORBIDDEN_SUBSTRINGS:
        if substring in data:
            raise ValueError(
                f".DS_Store embeds forbidden host/mount string {substring!r}"
            )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} /path/to/.DS_Store", file=sys.stderr)
        return 2
    try:
        verify(Path(argv[1]))
    except (OSError, ValueError) as exc:
        print(f"verify-dmg-layout: FAIL — {exc}", file=sys.stderr)
        return 1
    print("verify-dmg-layout: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
