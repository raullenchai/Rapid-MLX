"""Fail-closed qualification contracts for the checkpoint HTTP benchmark."""

import copy
import hashlib
import json
import time
from pathlib import Path

import pytest

from scripts.benchmark_hybrid_checkpoints import (
    ARMS,
    CASES,
    stream_receipt,
    summarize,
    validate_receipt,
)


def receipt(cached=0, ttft=1.0, text="warm"):
    return {
        "usage": {
            "prompt_tokens": 6500,
            "completion_tokens": 32,
            "total_tokens": 6532,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
        "cached_tokens": cached,
        "ttft_s": ttft,
        "elapsed_s": 3.0,
        "finish_reason": "length",
        "stream_done": True,
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def matrix():
    result = {
        "rounds": 2,
        "contract": "cold",
        "min_speedup": 1.1,
        "arms": [],
        "rows": [],
    }
    for arm in sorted(ARMS):
        prefill, checkpoint = int(arm[7]), int(arm[-1])
        result["arms"].append(
            {
                "name": arm,
                "prefill": prefill,
                "checkpoint_max": checkpoint,
                "server_exit": -15,
            }
        )
        for rd in range(2):
            for case, (_, cached) in CASES.items():
                on = bool(checkpoint)
                result["rows"].append(
                    {
                        "arm": arm,
                        "round": rd,
                        "case": case,
                        "seed": receipt(text="seed"),
                        "cold": receipt(),
                        "warm": receipt(
                            cached=cached if on else 0,
                            ttft=0.5 if on and cached else 1.0,
                        ),
                    }
                )
    return result


def test_complete_matrix_joins_by_identity_and_reports_each_group():
    result = matrix()
    result["rows"].reverse()
    summary = summarize(result)
    assert summary["passed"]
    assert summary["cold_exact_pairs"] == summary["rows"] == 24
    assert len(summary["incremental_pairs"]) == 12
    assert len(summary["groups"]) == 6


def test_incremental_contract_does_not_hide_cold_drift():
    result = matrix()
    for row in result["rows"]:
        row["cold"]["output_sha256"] = receipt(text="cold")["output_sha256"]
        row["output_exact"] = True  # Saved pass booleans are not evidence.
    assert not summarize(result)["passed"]
    result["contract"] = "incremental"
    summary = summarize(result)
    assert summary["passed"] and summary["incremental_passed"]
    assert not summary["cold_passed"] and summary["cold_exact_pairs"] == 0


@pytest.mark.parametrize("phase", ["seed", "warm", "cold"])
def test_checkpoint_on_off_output_or_usage_difference_fails(phase):
    result = matrix()
    result["contract"] = "incremental"
    result["rows"][0][phase]["output_sha256"] = "f" * 64
    assert not summarize(result)["passed"]
    result = matrix()
    result["contract"] = "incremental"
    result["rows"][0][phase]["usage"]["prompt_tokens"] += 1
    result["rows"][0][phase]["usage"]["total_tokens"] += 1
    assert not summarize(result)["passed"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "unknown",
        "missing_arm",
        "duplicate_arm",
        "error",
        "killed",
        "wrong_config",
        "no_cache",
        "cold_cache",
        "off_cache",
    ],
)
def test_incomplete_or_wrong_execution_cannot_qualify(mutation):
    result = matrix()
    if mutation == "missing":
        result["rows"].pop()
    elif mutation == "duplicate":
        result["rows"][-1] = copy.deepcopy(result["rows"][0])
    elif mutation == "unknown":
        result["rows"][0]["case"] = "other"
    elif mutation == "missing_arm":
        result["arms"].pop()
    elif mutation == "duplicate_arm":
        result["arms"][-1] = copy.deepcopy(result["arms"][0])
    elif mutation == "error":
        result["arms"][0]["error"] = "startup failed"
    elif mutation == "killed":
        result["arms"][0]["server_exit"] = -9
    elif mutation == "wrong_config":
        result["arms"][0]["checkpoint_max"] = 4
    elif mutation == "no_cache":
        row = next(
            r
            for r in result["rows"]
            if r["arm"].endswith("4") and r["case"] == "late_edit"
        )
        row["warm"] = receipt()
    else:
        row = result["rows"][0]
        row["cold" if mutation == "cold_cache" else "warm"] = receipt(cached=2048)
    with pytest.raises(ValueError):
        summarize(result)


def test_each_edit_group_must_meet_its_performance_gate():
    result = matrix()
    for row in result["rows"]:
        if row["arm"] == "prefill1-checkpoint4" and row["case"] == "mid_edit":
            row["warm"]["ttft_s"] = 1.5
    summary = summarize(result)
    assert summary["incremental_passed"] and not summary["performance_passed"]
    assert not summary["passed"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("stream_done", False),
        ("ttft_s", float("nan")),
        ("ttft_s", 0),
        ("elapsed_s", 0.5),
        ("output_sha256", ""),
        ("cached_tokens", True),
        ("cached_tokens", 2048),
        ("finish_reason", "stop"),
    ],
)
def test_invalid_receipts_are_rejected(key, value):
    item = receipt()
    item[key] = value
    with pytest.raises(ValueError):
        validate_receipt(item)


def stream():
    return [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "OK"}}]}),
        "data: "
        + json.dumps(
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
        ),
        "data: " + json.dumps({"choices": [], "usage": receipt()["usage"]}),
        "data: [DONE]",
    ]


def test_stream_requires_content_finish_usage_and_terminal_marker():
    result = stream_receipt(iter(stream()), time.perf_counter() - 0.1)
    assert result["stream_done"] and result["usage"]["completion_tokens"] == 32
    for missing in range(4):
        lines = stream()
        lines.pop(missing)
        with pytest.raises((ValueError, KeyError)):
            stream_receipt(iter(lines), time.perf_counter() - 0.1)


def test_stream_error_after_output_still_fails():
    lines = stream()
    lines.insert(1, 'data: {"error": {"message": "decode failed"}}')
    with pytest.raises(ValueError, match="server stream error"):
        stream_receipt(iter(lines), time.perf_counter() - 0.1)


def test_short_eos_warmup_does_not_relax_measured_response_contract():
    item = receipt()
    item["usage"]["completion_tokens"] = 1
    item["usage"]["total_tokens"] = 6501
    item["finish_reason"] = "stop"
    validate_receipt(item, require_full_budget=False)
    with pytest.raises(ValueError, match="32-token"):
        validate_receipt(item)


def test_recorded_m5_matrix_qualifies_incremental_but_rejects_cold_contract():
    fixture = (
        Path(__file__).resolve().parents[1]
        / "docs/engineering/performance/fixtures/m5-checkpoints-2026-10-09/checkpoint-matrix.json"
    )
    result = json.loads(fixture.read_text())
    summary = summarize(result)
    assert summary == result["summary"]
    assert summary["passed"] and len(summary["incremental_pairs"]) == 18
    assert summary["cold_exact_pairs"] == 26 and summary["cold_total_pairs"] == 36
    result["contract"] = "cold"
    assert not summarize(result)["passed"]
