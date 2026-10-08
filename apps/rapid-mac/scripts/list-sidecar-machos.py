#!/usr/bin/env python3
"""List every regular Mach-O file in a staged Desktop sidecar."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Thin 32/64-bit and universal Mach-O magic, in either byte order.
_MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}


def is_macho(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(4) in _MACHO_MAGICS


def list_machos(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"sidecar root is not a directory: {root}")
    found: list[Path] = []

    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, _subdirs, filenames in os.walk(
        root, followlinks=False, onerror=raise_walk_error
    ):
        parent = Path(directory)
        for filename in filenames:
            candidate = parent / filename
            if "\n" in os.fspath(candidate):
                raise ValueError(f"sidecar path contains a newline: {candidate!s}")
            if (
                candidate.is_file()
                and not candidate.is_symlink()
                and is_macho(candidate)
            ):
                found.append(candidate)
    return sorted(found, key=os.fspath)


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} SIDECAR_ROOT OUTPUT_FILE", file=sys.stderr)
        return 2
    root, output = map(Path, sys.argv[1:])
    try:
        machos = list_machos(root)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    output.write_text("".join(f"{path}\n" for path in machos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
