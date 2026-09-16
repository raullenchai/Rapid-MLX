"""Qualification-gate contracts for the singleton benchmark harness."""

from scripts.bench_qwen36_mllm_singleton import (
    _hash_streams_exact,
    _lifecycle_passes,
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
