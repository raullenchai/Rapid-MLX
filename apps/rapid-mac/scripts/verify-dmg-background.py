#!/usr/bin/env python3
"""Verify the active Finder icon-view background record in a .DS_Store."""

from __future__ import annotations

import plistlib
import struct
import sys
from pathlib import Path

ICVP_BLOB_MARKER = b"icvpblob"
EXPECTED_ALIAS_PARTS = (
    b"Rapid-MLX Desktop:.background:\x00background.png",
    b"/.background/background.png",
)
ALIAS_FIXED_HEADER_SIZE = 150


def read_icvp_records(data: bytes) -> list[dict[str, object]]:
    """Extract structurally valid icvp blob records from a DS_Store payload."""
    records: list[dict[str, object]] = []
    cursor = 0
    while True:
        marker = data.find(ICVP_BLOB_MARKER, cursor)
        if marker < 0:
            return records
        length_offset = marker + len(ICVP_BLOB_MARKER)
        payload_offset = length_offset + 4
        if payload_offset > len(data):
            raise ValueError("icvp blob length field is truncated")
        payload_length = struct.unpack(">I", data[length_offset:payload_offset])[0]
        payload = data[payload_offset : payload_offset + payload_length]
        if len(payload) != payload_length:
            raise ValueError("icvp blob payload is truncated")
        if payload.startswith(b"bplist00"):
            try:
                value = plistlib.loads(payload)
            except plistlib.InvalidFileException:
                pass
            else:
                if isinstance(value, dict):
                    records.append(value)
        cursor = marker + 1


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


def verify(path: Path) -> None:
    records = read_icvp_records(path.read_bytes())
    if len(records) != 1:
        raise ValueError(
            f"expected exactly one structural icvp record, found {len(records)}"
        )

    record = records[0]
    if record.get("backgroundType") != 2:
        raise ValueError("icvp backgroundType is not image mode (2)")

    alias = record.get("backgroundImageAlias")
    if not isinstance(alias, bytes):
        raise ValueError(
            "icvp backgroundImageAlias is missing or not binary alias data"
        )
    alias_paths = parse_alias_target(alias)
    if alias_paths != EXPECTED_ALIAS_PARTS:
        raise ValueError("backgroundImageAlias path tags target the wrong file")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} /path/to/.DS_Store", file=sys.stderr)
        return 2
    try:
        verify(Path(argv[1]))
    except (OSError, ValueError) as exc:
        print(f"verify-dmg-background: FAIL — {exc}", file=sys.stderr)
        return 1
    print("verify-dmg-background: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
