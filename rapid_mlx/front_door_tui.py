"""Small terminal primitives for the interactive launcher."""

from __future__ import annotations

import os
import select
import sys
import termios
import tty


def read_key() -> str:
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        raw = os.read(fd, 1)
        if raw == b"\x1b":
            if select.select([fd], [], [], 0.05)[0]:
                sequence = os.read(fd, 2)
                return {b"[A": "up", b"[B": "down", b"[C": "right", b"[D": "left"}.get(
                    sequence, "unknown"
                )
            return "quit"
        if raw in (b"", b"\x03", b"\x04"):
            return "quit"
        if raw in (b"\r", b"\n"):
            return "enter"
        if raw in (b"\x7f", b"\x08"):
            return "backspace"
        return raw.decode("utf-8", errors="replace")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def fit(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def welcome(width: int = 80) -> str:
    return "\n".join(
        fit(line, width)
        for line in [
            "Welcome to Rapid-MLX!",
            "",
            "Fast local models",
            "  Run models on Apple Silicon.",
            "Plug into your coding tools",
            "  Use your favorite coding agents.",
            "Your data stays on your Mac",
            "  Keep your conversations local.",
            "",
            "Press Enter to continue",
        ]
    )


def _paint(text: str, code: str, color: bool) -> str:
    return (
        f"\x1b[{code}m{text}\x1b[0m"
        if color and not os.environ.get("NO_COLOR")
        else text
    )


def _list_screen(header, rows, cursor, footer, width, height, color):
    # Keep the selected row visible even on short terminals.
    slots = max(1, (height - len(header) - 1) // 3)
    start = max(0, min(cursor - slots + 1, len(rows) - slots))
    lines = [
        _paint(fit(line, width), "1" if i == 0 else "2", color)
        for i, line in enumerate(header)
    ]
    for i in range(start, min(len(rows), start + slots)):
        label, description = rows[i]
        lines.append(
            _paint(
                fit(f"{'▸' if i == cursor else ' '} {label}", width),
                "1;97" if i == cursor else "1",
                color,
            )
        )
        lines.append(_paint(fit(f"    {description}", width), "2", color))
        lines.append("")
    if not rows:
        lines.append("  No matching models")
    if not rows:
        lines.append("")
    lines.append(_paint(fit(footer, width), "2", color))
    return "\n".join(lines)


def launcher(
    title, hardware, items, cursor, server="", width=80, height=24, color=False
):
    header = [title, hardware] + ([server] if server else []) + [""]
    return _list_screen(
        header,
        [(label, desc) for _, label, desc in items],
        cursor,
        "↑/↓ navigate • enter launch • → configure • esc quit",
        width,
        height,
        color,
    )


def picker(tool, entries, cursor, query="", width=80, height=24, color=False):
    return _list_screen(
        [
            f"Select model for {tool}: {query or 'Type to filter...'}",
            "",
            "Recommended for this Mac",
        ],
        entries,
        cursor,
        "↑/↓ navigate • enter select • ← back",
        width,
        height,
        color,
    )


def confirm(model: str, yes: bool = True, width: int = 80) -> str:
    choices = "▸ Yes    No" if yes else "  Yes  ▸ No"
    return "\n".join(
        fit(line, width)
        for line in [
            f"Download {model}?",
            "",
            choices,
            "",
            "←/→ navigate • enter confirm • esc cancel",
        ]
    )


def progress(name: str, done: int, total: int, speed: float, width: int = 80) -> str:
    def size(value: float) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1000 or unit == "TB":
                return f"{value:.0f} {unit}"
            value /= 1000
        return "0 B"

    fraction = min(1.0, done / total) if total else 0
    bar_width = max(1, min(20, width - 58))
    filled = int(fraction * bar_width)
    return fit(
        f"pulling {name}: {fraction:3.0%} ▕{'█' * filled}{' ' * (bar_width - filled)}▏ {size(done)}/{size(total)}  {size(speed)}/s",
        width,
    )
