# SPDX-License-Identifier: Apache-2.0
"""Character chunks with stable source locators and natural boundaries."""

import re
from dataclasses import dataclass

from .extraction import SkippedError, Unit

MAX_CHUNKS_PER_FILE = 65_536

CHUNKING_VERSION = "3"
BOUNDARIES = (r"\n(?=#{1,6} )", r"\n\s*\n", r"\n", r"\s+|[。！？；，]")


def _boundary(text: str, low: int, high: int, *, first: bool = False) -> int | None:
    """Prefer document structure, then whitespace or CJK punctuation."""
    for pattern in BOUNDARIES:
        positions = [
            match.end() for match in re.compile(pattern).finditer(text, low, high)
        ]
        if positions:
            return positions[0] if first else positions[-1]
    return None


@dataclass(frozen=True)
class Chunk:
    text: str
    line: int | None
    page: int | None
    kind: str = "text"
    timestamp: float | None = None


def chunk_units(
    units: list[Unit], target: int = 1200, overlap: int = 180
) -> list[Chunk]:
    if target < 2 or not 0 <= overlap < target // 2:
        raise ValueError("overlap must be less than half the chunk target")
    result: list[Chunk] = []
    for unit in units:
        start = 0
        line_position, line_number = 0, 1
        text = unit.text
        while start < len(text):
            end = min(len(text), start + target)
            if end < len(text):
                low = start + int(target * 0.75)
                end = _boundary(text, low, end) or _boundary(text, start, end) or end
            raw = text[start:end]
            leading = len(raw) - len(raw.lstrip())
            position = start + leading
            line_number += text.count("\n", line_position, position)
            line_position = position
            if raw.strip():
                if len(result) >= MAX_CHUNKS_PER_FILE:
                    raise SkippedError("chunk size cap")
                result.append(
                    Chunk(
                        raw.strip(),
                        None if unit.page else line_number,
                        unit.page,
                    )
                )
            if end == len(text):
                break
            desired = max(start + 1, end - overlap)
            # Snap forward so overlap cannot trap us on the previous boundary.
            # No separator means this is an oversized token (or unspaced CJK).
            boundary = _boundary(text, desired, end, first=True)
            start = boundary or (end if text[end].isspace() else desired)
    return result
