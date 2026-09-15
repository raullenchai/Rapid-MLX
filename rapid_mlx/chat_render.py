# SPDX-License-Identifier: Apache-2.0
"""Markdown rendering for the built-in chat terminal."""

from __future__ import annotations

import os
import sys
import time
from typing import TextIO

from markdown_it import MarkdownIt
from rich.cells import get_character_cell_size
from rich.console import Console
from rich.markdown import Heading, Markdown
from rich.theme import Theme

_CHAT_THEME = Theme(
    {
        "markdown.h1": "bold cyan",
        "markdown.h1.border": "cyan",
        "markdown.h2": "bold magenta",
        "markdown.h3": "bold cyan",
        "markdown.h4": "bold",
        "markdown.h5": "italic",
        "markdown.h6": "dim",
        "markdown.strong": "bold bright_white",
        "markdown.code": "bold yellow",
        "markdown.block_quote": "italic magenta",
        "markdown.list": "cyan",
        "markdown.item.bullet": "bold cyan",
        "markdown.item.number": "bold cyan",
        "markdown.link": "underline bright_blue",
        "markdown.link_url": "underline bright_blue",
        "markdown.table.border": "cyan",
        "markdown.table.header": "bold cyan",
    }
)


class _LeftHeading(Heading):
    """Use agent-style left-aligned headings, including level one."""

    LEVEL_ALIGN = {f"h{level}": "left" for level in range(1, 7)}


class _ChatMarkdown(Markdown):
    elements = {**Markdown.elements, "heading_open": _LeftHeading}


def _cell_suffix(text: str, width: int) -> str:
    """Return the longest suffix that fits in *width* terminal cells."""

    cells = 0
    suffix: list[str] = []
    for char in reversed(text):
        char_cells = get_character_cell_size(char)
        if cells + char_cells > width:
            break
        cells += char_cells
        suffix.append(char)
    return "".join(reversed(suffix))


def supports_rich_output(stream: TextIO) -> bool:
    """Whether *stream* can safely use interactive terminal formatting."""

    if (
        not bool(getattr(stream, "isatty", lambda: False)())
        or "NO_COLOR" in os.environ
        or "CI" in os.environ
    ):
        return False
    return not _console(stream).is_dumb_terminal


def terminal_safe_text(text: str) -> str:
    """Remove terminal control characters from untrusted display text."""

    return "".join(char for char in text if char.isprintable())


def _console(stream: TextIO) -> Console:
    return Console(
        file=stream,
        force_terminal=True,
        color_system="auto",
        highlight=False,
        theme=_CHAT_THEME,
    )


def _markdown(text: str) -> _ChatMarkdown:
    markdown = _ChatMarkdown(
        text,
        code_theme="monokai",
        hyperlinks=True,
    )
    parser = MarkdownIt("commonmark", {"html": False})
    parser.enable("strikethrough").enable("table")
    markdown.parsed = parser.parse(text)
    return markdown


def render_markdown(text: str, *, stream: TextIO | None = None) -> None:
    """Render a completed answer, preserving raw output for pipes."""

    target = stream or sys.stdout
    if not supports_rich_output(target):
        target.write(text)
        target.flush()
        return
    _console(target).print(_markdown(text))


class StreamingMarkdownRenderer:
    """Preview a TTY answer, then replace it with one correct Markdown render."""

    def __init__(self, stream: TextIO | None = None):
        self._stream = stream or sys.stdout
        self._enabled = supports_rich_output(self._stream)
        self._preview_width = (
            max(0, _console(self._stream).size.width - 1) if self._enabled else 0
        )
        self._chunks: list[str] = []
        self._preview = ""
        self._tokens = 0
        self._start: float | None = None

    def __enter__(self) -> StreamingMarkdownRenderer:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.finish()

    def write(self, text: str) -> None:
        if not text:
            return
        if not self._enabled:
            self._stream.write(text)
            self._stream.flush()
            return

        self._chunks.append(text)
        # One SSE delta ≈ one token for the streaming backends chat talks to,
        # so a delta counter gives a live progress read during long answers
        # (the authoritative count still prints on the speed line at the end).
        self._tokens += 1
        if self._start is None:
            self._start = time.monotonic()
        preview_piece: list[str] = []
        previous_was_space = not self._preview or self._preview.endswith(" ")
        for char in text:
            if char.isspace():
                if not previous_was_space:
                    preview_piece.append(" ")
                previous_was_space = True
            elif char.isprintable():
                preview_piece.append(char)
                previous_was_space = False
        if preview_piece and self._preview_width:
            badge = f"{self._tokens} tok · {time.monotonic() - self._start:.0f}s  "
            preview_width = self._preview_width - len(badge)
            if preview_width < 8:
                # Terminal too narrow to fit the badge and a useful preview;
                # keep the preview and drop the badge rather than truncate both.
                badge = ""
                preview_width = self._preview_width
            self._preview = _cell_suffix(
                self._preview + "".join(preview_piece),
                preview_width,
            )
            self._stream.write(f"\r\x1b[2K\x1b[2m{badge}{self._preview}\x1b[0m")
            self._stream.flush()

    def finish(self) -> None:
        if self._enabled and self._chunks:
            self._stream.write("\r\x1b[2K")
            render_markdown("".join(self._chunks), stream=self._stream)
            self._chunks.clear()
            self._preview = ""
