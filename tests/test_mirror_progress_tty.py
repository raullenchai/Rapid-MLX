# SPDX-License-Identifier: Apache-2.0
"""The mirror pull's byte-progress renderer must adapt to its output sink.

Two contracts share one ``_ProgressTracker``:

* **Piped stdout** (the desktop app's captured subprocess, a redirected log,
  pytest capture) keeps the machine-readable ``[bytes] D/T`` heartbeat the
  desktop's ``DownloadProgress`` parser reads. Breaking this format silently
  freezes the desktop progress bar — the regression this file guards.
* **A human TTY** (someone running bare ``rapid-mlx chat`` in a terminal) gets
  ONE in-place bar redrawn with ``\\r`` instead of the hundreds of raw
  ``[bytes]`` lines a multi-GB cold pull used to scroll (0.11 first-run
  dogfood papercut). Per-file completion lines route through ``write_line`` so
  they erase the bar cleanly rather than shredding it.

These are direct unit tests of the tracker: fast and hermetic, no network.
``_mirror._PROGRESS_HEARTBEAT_SECONDS`` is dropped to 0 so every ``add`` emits.
"""

from __future__ import annotations

import io
import os
import sys

import pytest

from rapid_mlx import _mirror
from rapid_mlx._mirror import _ProgressTracker


class _FakeTTY(io.StringIO):
    """A captured stdout that claims to be an interactive terminal."""

    def isatty(self) -> bool:
        return True


class _FakePipe(io.StringIO):
    """A captured stdout that is NOT a terminal (desktop pipe / log)."""

    def isatty(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _every_add_emits(monkeypatch):
    # No throttle — every add() renders, so a single add is observable.
    monkeypatch.setattr(_mirror, "_PROGRESS_HEARTBEAT_SECONDS", 0.0)
    # Clear NO_COLOR so the _FakeTTY tests exercise the ANSI bar. NO_COLOR
    # does NOT switch to the non-TTY [bytes] protocol — isatty() alone selects
    # the bar; NO_COLOR only drops ANSI styling (covered by
    # test_no_color_tty_keeps_bar_without_ansi).
    monkeypatch.delenv("NO_COLOR", raising=False)
    # Pin a wide terminal so the width-adaptive bar renders a full 24-cell bar
    # deterministically, independent of the CI runner's real COLUMNS.
    monkeypatch.setenv("COLUMNS", "100")


def test_non_tty_keeps_machine_bytes_heartbeat(monkeypatch):
    """Piped stdout emits ``[bytes] D/T`` and never the human bar."""
    fake = _FakePipe()
    monkeypatch.setattr(sys, "stdout", fake)
    t = _ProgressTracker(total=1000)
    t.add(400)
    t.add(600)  # done == total
    t.flush()
    out = fake.getvalue()
    assert "[bytes] 400/1000" in out
    assert "[bytes] 1000/1000" in out
    # No terminal control sequences or bar glyphs leak onto a pipe.
    assert "\r" not in out
    assert "↓" not in out
    assert "\x1b" not in out


def test_tty_draws_single_inplace_bar(monkeypatch):
    """A human terminal gets one carriage-return bar, no ``[bytes]`` lines."""
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    t = _ProgressTracker(total=1000)
    t.add(250)
    t.add(250)  # 50%
    mid = fake.getvalue()
    assert "[bytes]" not in mid  # machine format never shown to a human
    assert "50%" in mid and "↓" in mid
    # ONE line, redrawn in place: NO newline anywhere mid-stream (a scroll
    # would insert one), and every redraw is a carriage-return + erase-line.
    assert "\n" not in mid
    assert mid.count("\r\x1b[2K") == 2  # exactly the two add() redraws
    t.add(500)  # 100%
    t.flush()
    final = fake.getvalue()
    assert "100%" in final
    assert final.endswith("\n")  # flush finalizes the bar's row
    # The finalizing newline is the ONLY newline in the whole TTY session —
    # proof the bar never scrolled.
    assert final.count("\n") == 1


def test_tty_bar_clamps_display_at_100_percent(monkeypatch):
    """An oversized/corrupt stream can't render >100% (same clamp as add)."""
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    t = _ProgressTracker(total=1000)
    t.add(1500)  # Content-Length lied / proxy injected bytes
    out = fake.getvalue()
    assert "100%" in out
    assert "150%" not in out


def test_tty_write_line_erases_bar_then_prints(monkeypatch):
    """A completion line erases the live bar and lands on its own row."""
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    t = _ProgressTracker(total=1000)
    t.add(300)  # bar now live
    t.write_line("  [1/3] config.json")
    out = fake.getvalue()
    # erase-line sequence, the completion text, then a newline
    assert "\r\x1b[2K  [1/3] config.json\n" in out


def test_non_tty_write_line_is_plain_print(monkeypatch):
    """On a pipe, write_line == the old ``_print_dim`` (byte-identical)."""
    fake = _FakePipe()
    monkeypatch.setattr(sys, "stdout", fake)
    t = _ProgressTracker(total=1000)
    t.write_line("  [1/3] config.json")
    out = fake.getvalue()
    assert out == "  [1/3] config.json\n"
    assert "\r" not in out and "\x1b" not in out


def test_tty_stays_silent_when_total_unknown(monkeypatch):
    """total==0 (HF didn't expose sizes) → no bar, no divide-by-zero."""
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    t = _ProgressTracker(total=0)
    t.add(500)
    t.flush()
    assert fake.getvalue() == ""


def test_narrow_tty_never_wraps(monkeypatch):
    """A narrow terminal must not wrap a redraw onto a second row: the bar
    shrinks (or drops to a compact percentage + bytes readout) and no single
    redrawn row exceeds the column count (codex #1259).
    """
    monkeypatch.setenv("COLUMNS", "20")  # overrides the fixture's wide default
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    t = _ProgressTracker(total=500_000_000)
    t.add(250_000_000)  # 50%
    t.write_line("  [1/2] model.safetensors")
    t.add(250_000_000)  # 100%
    t.flush()
    out = fake.getvalue()
    assert "[bytes]" not in out  # still the human path, not the machine flood
    assert "↓" in out  # a bar (compact or full) is present
    # Every redrawn bar row (split on \r, escapes + trailing newline removed)
    # must fit within the 20-column terminal. write_line status lines are
    # one-shot (not redrawn), so exempt them from the width check.
    for seg in out.replace("\x1b[2K", "").split("\r"):
        row = seg.split("\n")[0]
        if "↓" in row:
            assert len(row) <= 20, f"bar row exceeds width: {row!r} ({len(row)})"


def _render_bar_under_pty(cols, total, adds, monkeypatch):
    """Run a real ``_ProgressTracker`` with stdout wired to a PTY sized to
    ``cols`` columns, and return the captured output. Exercises the true
    ``os.get_terminal_size(self._out.fileno())`` width path — no ``COLUMNS``,
    no ``_FakeTTY``."""
    import fcntl
    import pty
    import struct
    import termios
    import threading

    master, slave = pty.openpty()
    # Size the PTY to `cols` columns (rows irrelevant to the bar).
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, cols, 0, 0))

    # Drain the master concurrently: on macOS, closing the slave can discard
    # buffered data before a serial read gets to it, so read as it's written.
    captured = bytearray()

    def _reader():
        while True:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            captured.extend(chunk)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    slave_f = os.fdopen(slave, "w", buffering=1)
    monkeypatch.setattr(sys, "stdout", slave_f)
    try:
        t = _ProgressTracker(total=total)
        for a in adds:
            t.add(a)
        t.flush()
    finally:
        slave_f.flush()
        slave_f.close()  # slave EOF → the reader thread exits
        monkeypatch.setattr(sys, "stdout", sys.__stdout__)
    reader.join(timeout=2)
    os.close(master)
    return bytes(captured).decode("utf-8", "replace")


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY unavailable")
def test_pty_width_measures_output_stream_not_default(monkeypatch):
    """The bar sizes to ITS OWN output PTY, not the process's default
    terminal (codex #1259 BLOCKING). Two PTYs of differing widths, COLUMNS
    unset: every bar row must fit its own PTY's width. If width detection
    fell back to a constant (e.g. 80), the 34-col PTY's rows would overflow
    and this fails.
    """
    monkeypatch.delenv("COLUMNS", raising=False)  # force the fileno path
    for cols in (34, 72):
        out = _render_bar_under_pty(
            cols, 500_000_000, [250_000_000, 250_000_000], monkeypatch
        )
        assert "[bytes]" not in out
        assert "↓" in out
        for seg in out.replace("\x1b[2K", "").split("\r"):
            row = seg.split("\n")[0]
            if "↓" in row:
                assert len(row) <= cols, f"cols={cols}: row {row!r} len={len(row)}"


def test_no_color_tty_keeps_bar_without_ansi(monkeypatch):
    """NO_COLOR must NOT re-trigger the machine ``[bytes]`` flood on a
    terminal — that's the exact regression this feature removes, and the
    NO_COLOR user is precisely who wants clean output. NO_COLOR only drops
    ANSI: the bar still redraws in place via ``\\r`` + space-pad, emitting
    zero escape sequences.
    """
    monkeypatch.setenv("NO_COLOR", "1")  # re-set (autouse fixture cleared it)
    fake = _FakeTTY()
    monkeypatch.setattr(sys, "stdout", fake)
    t = _ProgressTracker(total=1000)
    t.add(250)
    t.add(250)  # 50%
    t.write_line("  [1/3] config.json")  # erases the bar without ANSI
    t.add(500)  # 100%
    t.flush()
    out = fake.getvalue()
    assert "[bytes]" not in out  # no machine flood for a NO_COLOR terminal
    assert "\x1b" not in out  # zero ANSI escapes
    assert "\r" in out  # still an in-place bar
    assert "↓" in out and "100%" in out
    assert "[1/3] config.json" in out
    assert out.endswith("\n")  # flush finalized the row
