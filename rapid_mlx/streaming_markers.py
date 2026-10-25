# SPDX-License-Identifier: Apache-2.0
"""Request-local recognition of control spellings split across decoded chunks."""

import re


def partial_marker_length(text: str, markers: tuple[str, ...]) -> int:
    """Longest suffix that could still become a marker on the next chunk."""
    return max(
        (
            size
            for marker in markers
            for size in range(1, min(len(text) + 1, len(marker)))
            if text.endswith(marker[:size])
        ),
        default=0,
    )


class StreamingMarkerBuffer:
    """Release complete spans, retaining only an ambiguous marker suffix.

    Recognition does not remove bytes: the owning parser decides their meaning.
    This keeps tool arguments and literal text out of global sanitization rules.
    """

    def __init__(self, markers: tuple[str, ...]):
        self.markers = markers
        self.pending = ""
        self._pattern = (
            re.compile(
                "|".join(
                    re.escape(marker)
                    for marker in sorted(markers, key=len, reverse=True)
                )
            )
            if markers
            else None
        )

    def process(self, delta: str) -> str:
        text = self.pending + delta
        resolved = 0
        if self._pattern is not None:
            for match in self._pattern.finditer(text):
                resolved = match.end()
        held = partial_marker_length(text[resolved:], self.markers)
        self.pending = text[-held:] if held else ""
        return text[:-held] if held else text

    def flush(self) -> str:
        text, self.pending = self.pending, ""
        return text
