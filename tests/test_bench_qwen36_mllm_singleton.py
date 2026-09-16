"""Qualification-gate contracts for the singleton benchmark harness."""

import subprocess
import sys

from scripts.bench_qwen36_mllm_singleton import (
    _checker_pass,
    _hash_streams_exact,
    _lifecycle_passes,
    _percent_change,
)


def _lifecycle() -> dict:
    return {
        "cancellation": {
            "request_id_published": True,
            "accepted": True,
            "tokens_before_abort": 12,
        },
        "recovery": {"exact": True, "completion_tokens": 8},
        "queued_concurrency": {"both_nonempty": True, "serialized": True},
        "abort_soak": {"pass": True, "failures": [], "iterations": 50},
    }


def test_lifecycle_gate_requires_every_contract() -> None:
    lifecycle = _lifecycle()
    assert _lifecycle_passes(lifecycle, 50)

    lifecycle["queued_concurrency"]["serialized"] = False
    assert not _lifecycle_passes(lifecycle, 50)

    lifecycle = _lifecycle()
    lifecycle["abort_soak"]["failures"] = ["iteration-7"]
    assert not _lifecycle_passes(lifecycle, 50)


def test_hash_gate_requires_within_phase_determinism() -> None:
    off = [
        {"send": 1, "sha256": "cold"},
        {"send": 1, "sha256": "cold"},
        {"send": 2, "sha256": "warm"},
    ]
    auto = [
        {"send": 1, "sha256": "cold"},
        {"send": 1, "sha256": "cold"},
        {"send": 2, "sha256": "warm"},
    ]
    assert _hash_streams_exact(off, auto)

    auto[1]["sha256"] = "different"
    assert not _hash_streams_exact(off, auto)


def test_hash_gate_requires_corresponding_send_streams() -> None:
    off = [{"send": 1, "sha256": "same"}]
    auto = [{"send": 2, "sha256": "same"}]
    assert not _hash_streams_exact(off, auto)


def test_regex_checker_requires_the_complete_shortcut() -> None:
    checker = {
        "type": "regex",
        "pattern": r"(?i)(?:⌘\s*n|(?:command|cmd)\s*\+?\s*n)",
    }
    assert _checker_pass(checker, "⌘N")
    assert _checker_pass(checker, "Command + N")
    assert not _checker_pass(checker, "new chat")
    assert not _checker_pass(checker, "n")


def test_percent_change_handles_zero_or_missing_baselines() -> None:
    assert _percent_change(15.0, 10.0) == 50.0
    assert _percent_change(0.0, 0.0) is None
    assert _percent_change(None, 1.0) is None


def test_zero_pairs_requires_explicit_lifecycle_mode() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.bench_qwen36_mllm_singleton",
            "--model",
            "unused",
            "--pairs",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "valid only with --lifecycle" in result.stderr
