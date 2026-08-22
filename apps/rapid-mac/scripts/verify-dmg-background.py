#!/usr/bin/env python3
"""Verify the active Finder icon-view background record in a .DS_Store."""

from __future__ import annotations

import plistlib
import struct
import sys
from pathlib import Path

ICVP_BLOB_MARKER = b"icvpblob"
EXPECTED_ALIAS_PARTS = (
    b"Rapid-MLX Desktop:.background:",
    b"/.background/background.png",
)


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
        if payload_offset <= len(data):
            payload_length = struct.unpack(">I", data[length_offset:payload_offset])[0]
            payload = data[payload_offset : payload_offset + payload_length]
            if len(payload) == payload_length and payload.startswith(b"bplist00"):
                try:
                    value = plistlib.loads(payload)
                except plistlib.InvalidFileException:
                    pass
                else:
                    if isinstance(value, dict):
                        records.append(value)
        cursor = marker + 1


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
    for expected in EXPECTED_ALIAS_PARTS:
        if expected not in alias:
            raise ValueError(f"backgroundImageAlias missing {expected.decode()!r}")


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
