# SPDX-License-Identifier: Apache-2.0
"""Progress for redirected human output and the Desktop protocol."""

import io
import itertools
import re
import sys

from rapid_mlx import _mirror


def test_human_emits_only_when_progress_is_due(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_PROCESS_ROLE", raising=False)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(_mirror.time, "monotonic", lambda: 100.0)
    tracker = _mirror._ProgressTracker(total=1000)
    calls = []
    monkeypatch.setattr(tracker, "_emit", lambda: calls.append(None))

    for _ in range(50):
        tracker.add(1)
    assert len(calls) == 0

    tracker.add(50)
    assert len(calls) == 1


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
    assert all("[bytes]" not in line for line in lines)
    matches = [
        re.fullmatch(r"  Downloaded \d+\.\d of 16\.3 GB \((\d+)%\)", line)
        for line in lines
    ]
    assert all(matches)
    percentages = [int(match.group(1)) for match in matches]
    assert percentages == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
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


def test_human_timed_progress_prints_when_rounded_line_is_unchanged(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_PROCESS_ROLE", raising=False)
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    now = [0.0]
    monkeypatch.setattr(_mirror.time, "monotonic", lambda: now[0])
    tracker = _mirror._ProgressTracker(total=100_000_000_000)

    tracker.add(10_000_000_000)
    now[0] = 31.0
    tracker.add(5)
    assert stream.getvalue().splitlines() == [
        "  Downloaded 10.0 of 100.0 GB (10%)",
        "  Downloaded 10.0 of 100.0 GB (10%)",
    ]


def test_human_timer_runs_from_milestone(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_PROCESS_ROLE", raising=False)
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    now = [0.0]
    monkeypatch.setattr(_mirror.time, "monotonic", lambda: now[0])
    tracker = _mirror._ProgressTracker(total=1_000_000_000)

    now[0] = 1.0
    tracker.add(100_000_000)
    now[0] = 30.0
    tracker.add(1_000_000)
    assert len(stream.getvalue().splitlines()) == 1
    now[0] = 31.5
    tracker.add(9_000_000)
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


def test_current_line_only_for_desktop(monkeypatch, tmp_path, capsys):
    import json
    from unittest.mock import patch

    from tests.test_mirror_pull import (
        _catalog_payload,
        _FakeResponse,
        _mk_model_info,
        _UrlRouter,
    )

    repo = "mlx-community/Qwen3-0.6B-4bit"
    revision = "feed" * 10
    catalog = _catalog_payload([("qwen3-0.6b-4bit", repo, "mirrored")])
    monkeypatch.setenv("RAPID_MLX_MODEL_MIRROR", "https://models.rapidmlx.com")
    for role in (None, "desktop-sidecar"):
        if role:
            monkeypatch.setenv("RAPID_MLX_PROCESS_ROLE", role)
        else:
            monkeypatch.delenv("RAPID_MLX_PROCESS_ROLE", raising=False)
        router = _UrlRouter()
        router.add(
            "https://models.rapidmlx.com/api/models",
            _FakeResponse(200, json.dumps(catalog).encode()),
        )
        router.add(
            f"https://models.rapidmlx.com/{repo}/model.safetensors",
            _FakeResponse(200, b"x" * 100),
        )
        with (
            patch("urllib.request.urlopen", side_effect=router),
            patch(
                "huggingface_hub.model_info",
                return_value=_mk_model_info(revision, [("model.safetensors", 100)]),
            ),
        ):
            assert _mirror.download_with_mirror_fallback(
                repo, cache_dir=tmp_path / str(role)
            )
        output = capsys.readouterr().out
        assert ("[current] model.safetensors" in output) is bool(role)
