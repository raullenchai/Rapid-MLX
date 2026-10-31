# SPDX-License-Identifier: Apache-2.0
"""Progress for redirected human output and the Desktop protocol."""

import io
import itertools
import re
import sys

from rapid_mlx import _mirror


def test_human_log_milestones(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_PROCESS_ROLE", raising=False)
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    clock = itertools.count()
    monkeypatch.setattr(_mirror.time, "monotonic", lambda: next(clock) * 0.6)
    tracker = _mirror._ProgressTracker(total=16_300_000_000)
    for _ in range(100):
        tracker.add(163_000_000)
    tracker.flush()
    lines = stream.getvalue().splitlines()
    assert len(lines) <= 11
    assert all("[bytes]" not in line for line in lines)
    matches = [
        re.fullmatch(r"  Downloaded \d+\.\d of 16\.3 GB \((\d+)%\)", line)
        for line in lines
    ]
    assert all(matches)
    percentages = [int(match.group(1)) for match in matches]
    assert all(a < b for a, b in zip(percentages, percentages[1:]))
    assert lines[-1] == "  Downloaded 16.3 of 16.3 GB (100%)"


def test_human_log_slow_progress_and_flush(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_PROCESS_ROLE", raising=False)
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    now = [0.0]
    monkeypatch.setattr(_mirror.time, "monotonic", lambda: now[0])
    tracker = _mirror._ProgressTracker(total=1_000_000_000)
    tracker.add(100_000_000)
    now[0] = 31
    tracker.add(10_000_000)
    tracker.flush()
    tracker.flush()
    assert stream.getvalue().splitlines() == [
        "  Downloaded 0.1 of 1.0 GB (10%)",
        "  Downloaded 0.1 of 1.0 GB (11%)",
    ]


def test_desktop_pipe_keeps_machine_protocol(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_PROCESS_ROLE", "desktop-sidecar")
    monkeypatch.setattr(_mirror, "_PROGRESS_HEARTBEAT_SECONDS", 0.0)
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    tracker = _mirror._ProgressTracker(total=100)
    tracker.add(40)
    tracker.add(60)
    tracker.flush()
    assert (
        stream.getvalue() == "  [bytes] 40/100\n  [bytes] 100/100\n  [bytes] 100/100\n"
    )
