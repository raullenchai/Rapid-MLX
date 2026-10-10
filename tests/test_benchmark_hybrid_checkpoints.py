"""Fail-closed qualification contracts for the checkpoint HTTP benchmark."""

import copy
import hashlib
import json
import time
from pathlib import Path

import pytest

from scripts.benchmark_hybrid_checkpoints import (
    CASES,
    child_environment,
    controlled_environment,
    server_command,
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
        "schema_version": 2,
        "model": "/cached/snapshot",
        "port": 8617,
        "rounds": 2,
        "contract": "cold",
        "min_speedup": 1.1,
        "arms": [],
        "rows": [],
    }
    schedule = [
        (0, "off-on", 0),
        (0, "off-on", 4),
        (1, "on-off", 4),
        (1, "on-off", 0),
        (1, "off-on", 0),
        (1, "off-on", 4),
        (0, "on-off", 4),
        (0, "on-off", 0),
    ]
    for prefill, order, checkpoint in schedule:
        arm = f"{order}-prefill{prefill}-checkpoint{checkpoint}"
        result["arms"].append(
            {
                "name": arm,
                "command": server_command(
                    "/env/bin/python", result["model"], result["port"]
                ),
                "order": order,
                "prefill": prefill,
                "checkpoint_max": checkpoint,
                "server_exit": -15,
                "controlled_env": controlled_environment(prefill, checkpoint),
                "prefill_evidence": [
                    "[gdn_prefill] blocked-seq GDN prefill kernel installed"
                    if prefill
                    else "[gdn_prefill] disabled via RAPID_MLX_GDN_PREFILL=0"
                ],
            }
        )
        for rd in range(2):
            for case, (_, cached) in CASES.items():
                on = bool(checkpoint)
                result["rows"].append(
                    {
                        "arm": arm,
                        "order": order,
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
    assert summary["cold_exact_pairs"] == summary["rows"] == 48
    assert len(summary["incremental_pairs"]) == 24
    assert len(summary["groups"]) == 12


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
        if row["arm"] == "off-on-prefill1-checkpoint4" and row["case"] == "mid_edit":
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


def test_stream_rejects_duplicate_choice_zero_in_one_event():
    lines = stream()
    event = json.loads(lines[0][5:])
    event["choices"].append(dict(event["choices"][0]))
    lines[0] = "data: " + json.dumps(event)
    with pytest.raises(ValueError, match="additional choice"):
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
    # Historical fixed-order receipts remain observations, never certification.
    assert not summary["passed"] and not summary["order_balanced"]
    assert summary["incremental_passed"] and len(summary["incremental_pairs"]) == 18
    assert summary["cold_exact_pairs"] == 26 and summary["cold_total_pairs"] == 36
    result["contract"] = "cold"
    assert not summarize(result)["passed"]


@pytest.mark.parametrize("mutation", ["environment", "missing_log", "wrong_log"])
def test_arm_requires_controlled_environment_and_effective_prefill(mutation):
    result = matrix()
    arm = result["arms"][0]
    if mutation == "environment":
        arm["controlled_env"]["RAPID_MLX_GDN_PREFILL"] = "1"
    elif mutation == "missing_log":
        arm.pop("prefill_evidence")
    else:
        arm["prefill_evidence"] = [
            "[gdn_prefill] blocked-seq GDN prefill kernel installed"
        ]
    with pytest.raises(ValueError):
        summarize(result)


def test_child_environment_excludes_ambient_inference_and_import_overrides(monkeypatch):
    for key, value in {
        "RAPID_MLX_GDN_PREFILL": "9",
        "RAPID_MLX_HYBRID_CHECKPOINT_MAX": "99",
        "RAPID_MLX_API_KEY": "test-only-credential",
        "RAPID_MLX_QSA_INDEXED_SPLITK": "0",
        "PYTHONPATH": "/unrelated/imports",
        "PYTHONHOME": "/unrelated/python",
        "DYLD_LIBRARY_PATH": "/unrelated/libraries",
        "HF_TOKEN": "test-only-credential",
        "HF_HOME": "/unrelated/cache",
        "MLX_METAL_FAST_SYNCH": "1",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("HOME", "/test/home")
    env = child_environment(0, 4)
    assert env["HOME"] == "/test/home"
    assert {k: env[k] for k in controlled_environment(0, 4)} == controlled_environment(
        0, 4
    )
    assert set(env) <= set(controlled_environment(0, 4)) | {
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
    }


def test_child_environment_runs_python_without_ambient_pythonhome(monkeypatch):
    import subprocess
    import sys

    monkeypatch.setenv("PYTHONHOME", "/unrelated/python")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/imports")
    proc = subprocess.run(
        [sys.executable, "-c", "import json; print(json.dumps({'ok': True}))"],
        env=child_environment(1, 0),
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert json.loads(proc.stdout) == {"ok": True}


@pytest.mark.parametrize(
    "mutation", ["missing_order", "row_order", "arm_order", "sequence"]
)
def test_order_evidence_cannot_be_missing_or_mixed(mutation):
    result = matrix()
    if mutation == "missing_order":
        result["arms"] = [a for a in result["arms"] if a["order"] == "off-on"]
        result["rows"] = [r for r in result["rows"] if r["order"] == "off-on"]
    elif mutation == "row_order":
        result["rows"][0]["order"] = "on-off"
    elif mutation == "arm_order":
        result["arms"][0]["order"] = "on-off"
    else:
        result["arms"][0], result["arms"][1] = result["arms"][1], result["arms"][0]
    with pytest.raises(ValueError):
        summarize(result)


def test_each_order_must_pass_even_when_pooled_gain_is_large():
    result = matrix()
    for row in result["rows"]:
        if row["arm"] == "on-off-prefill0-checkpoint4" and row["case"] == "late_edit":
            row["warm"]["ttft_s"] = 1.01
    summary = summarize(result)
    assert summary["incremental_passed"] and not summary["performance_passed"]
    assert not summary["passed"]


@pytest.mark.parametrize("mutation", ["chunk", "model", "speculation", "extra"])
def test_noncanonical_launch_cannot_qualify(mutation):
    result = matrix()
    cmd = result["arms"][0]["command"]
    if mutation == "chunk":
        cmd[cmd.index("--prefill-step-size") + 1] = "1024"
    elif mutation == "model":
        cmd[4] = "/other/model"
    elif mutation == "speculation":
        cmd.remove("--no-spec-decode")
    else:
        cmd.extend(["--temperature", "1"])
    with pytest.raises(ValueError, match="noncanonical"):
        summarize(result)
