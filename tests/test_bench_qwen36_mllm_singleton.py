"""Qualification-gate contracts for the singleton benchmark harness."""

import subprocess
import sys

from scripts.bench_qwen36_mllm_singleton import (
    _checker_pass,
    _hash_streams_exact,
    _lifecycle_passes,
    _measured_fastpath_engaged,
    _percent_change,
    _warm_phase_qualified,
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
        {"pair": 1, "send": 1, "sha256": "cold"},
        {"pair": 2, "send": 1, "sha256": "cold"},
        {"pair": 1, "send": 2, "sha256": "warm"},
    ]
    auto = [
        {"pair": 1, "send": 1, "sha256": "cold"},
        {"pair": 2, "send": 1, "sha256": "cold"},
        {"pair": 1, "send": 2, "sha256": "warm"},
    ]
    assert _hash_streams_exact(off, auto)

    auto[1]["sha256"] = "different"
    assert not _hash_streams_exact(off, auto)

    off[1]["sha256"] = "different"
    assert not _hash_streams_exact(off, auto)


def test_hash_gate_requires_corresponding_send_streams() -> None:
    off = [{"pair": 1, "send": 1, "sha256": "same"}]
    auto = [{"pair": 1, "send": 2, "sha256": "same"}]
    assert not _hash_streams_exact(off, auto)

    auto = [
        {"pair": 1, "send": 1, "sha256": "same"},
        {"pair": 2, "send": 1, "sha256": "same"},
    ]
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


def test_json_checker_requires_types_and_semantic_values() -> None:
    checker = {
        "type": "json_shape",
        "keys": ["status", "buttons"],
        "types": {"status": "str", "buttons": "list"},
        "field_terms": {"status": ["ready"], "buttons": ["start"]},
    }
    assert _checker_pass(checker, '{"status":"Ready","buttons":["Start"]}')
    assert _checker_pass(
        checker, '```json\n{"status":"Ready","buttons":["Start"]}\n```'
    )
    assert not _checker_pass(checker, '{"status":null,"buttons":[]}')
    assert not _checker_pass(checker, '{"status":"Idle","buttons":["Stop"]}')
    assert not _checker_pass(
        checker, 'Here is the JSON:\n{"status":"Ready","buttons":["Start"]}'
    )


def test_terms_checker_enforces_minimum_word_count() -> None:
    checker = {"type": "terms", "required": ["ready"], "min_words": 3}
    assert _checker_pass(checker, "system is ready")
    assert not _checker_pass(checker, "ready now")


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


def test_nonpositive_max_tokens_is_rejected() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.bench_qwen36_mllm_singleton",
            "--model",
            "unused",
            "--max-tokens",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "must be positive" in result.stderr


def test_nonpositive_abort_iterations_is_rejected() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.bench_qwen36_mllm_singleton",
            "--model",
            "unused",
            "--abort-iterations",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--abort-iterations must be positive" in result.stderr


def test_unknown_case_id_is_rejected() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.bench_qwen36_mllm_singleton",
            "--model",
            "unused",
            "--cases",
            "typo-case",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unknown --cases id(s): typo-case" in result.stderr


def test_lifecycle_requires_two_selected_media_cases() -> None:
    for selected in (("text-warm-01",), ("ocr-01",)):
        command = [
            sys.executable,
            "-m",
            "scripts.bench_qwen36_mllm_singleton",
            "--model",
            "unused",
            "--lifecycle",
        ]
        for case_id in selected:
            command.extend(("--cases", case_id))
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 2
        assert (
            "requires at least two selected cases with image fixtures" in result.stderr
        )


def test_fastpath_engagement_ignores_unmeasured_warmup_counter() -> None:
    phase = {
        "singleton_batches": 1,
        "per_case": {"ocr-01": [{"singleton_batch_delta": 0}]},
    }
    assert not _measured_fastpath_engaged(phase)

    phase["per_case"]["ocr-01"].append({"singleton_batch_delta": 1})
    assert _measured_fastpath_engaged(phase)


def test_warm_gate_requires_measured_hit_and_candidate_singleton() -> None:
    baseline = {
        "per_case": {
            "text-warm-01": [
                {"send": 1, "prefix_cache_hit_delta": 0, "singleton_batch_delta": 0},
                {"send": 2, "prefix_cache_hit_delta": 1, "singleton_batch_delta": 0},
            ]
        }
    }
    candidate = {
        "per_case": {
            "text-warm-01": [
                {"send": 1, "prefix_cache_hit_delta": 0, "singleton_batch_delta": 1},
                {"send": 2, "prefix_cache_hit_delta": 1, "singleton_batch_delta": 1},
            ]
        }
    }
    assert _warm_phase_qualified(baseline, False)
    assert _warm_phase_qualified(candidate, True)

    candidate["per_case"]["text-warm-01"][1]["prefix_cache_hit_delta"] = 0
    assert not _warm_phase_qualified(candidate, True)
