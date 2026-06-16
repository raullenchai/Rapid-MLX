# SPDX-License-Identifier: Apache-2.0
"""``build_submission_payload(tier='all', smoke_result=..., harness_result=...)``
populates all three sections AND the result validates against the
schema.

This is the future-state path that PR #4 (or wherever
``bench --tier all --submit`` gets wired in) will call. We test the
builder here so that downstream wiring only needs to verify that it
*calls* the builder correctly — the payload shape is locked in.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "community-benchmarks" / "schema.json"


def _stub_inputs():
    from vllm_mlx.community_bench.hardware import Hardware, Software
    from vllm_mlx.community_bench.runner import (
        BenchResult,
        BucketResult,
        RoundResult,
    )

    hw = Hardware(chip="Apple M4 Pro", ram_gb=24, cpu_cores=12, gpu_cores=20)
    sw = Software(macos="26.5.1", rapid_mlx="0.7.6", mlx="0.31.2", python="3.12.13")
    rounds = [
        RoundResult(decode_tps=42.0, prefill_tps=500.0, ttft_ms=120.0)
        for _ in range(5)
    ]
    bench = BenchResult(
        short=BucketResult(rounds_raw=rounds),
        long=BucketResult(rounds_raw=rounds),
        peak_ram_mb=8192,
        prompt_hash="deadbeefcafebabe",
        sampling="greedy",
    )
    return hw, sw, bench


_SMOKE = {
    "boot_time_ms": 1234.5,
    "first_prompt_ok": True,
    "first_token_latency_ms": 87.0,
    "response_excerpt": "The capital of France is Paris.",
}
_HARNESS = {
    "codex":     {"passed": True,  "duration_s": 11.4, "error_excerpt": None},
    "opencode":  {"passed": True,  "duration_s":  9.8, "error_excerpt": None},
    "hermes":    {"passed": False, "duration_s":  3.1, "error_excerpt": "tool_call schema mismatch"},
    "aider":     {"passed": True,  "duration_s": 14.2, "error_excerpt": None},
    "langchain": {"passed": True,  "duration_s": 18.5, "error_excerpt": None},
}


def _build_full() -> dict:
    from vllm_mlx.community_bench.submission import build_submission_payload

    hw, sw, bench = _stub_inputs()
    return build_submission_payload(
        hardware=hw,
        software=sw,
        alias="qwen3.5-9b-4bit",
        hf_path="mlx-community/Qwen3.5-9B-4bit",
        bench=bench,
        notes="full-tier smoke",
        now=datetime(2026, 6, 15, 10, 30, 0, tzinfo=timezone.utc),
        tier="all",
        smoke_result=_SMOKE,
        harness_result=_HARNESS,
    )


def test_full_payload_has_all_three_sections() -> None:
    payload = _build_full()
    assert payload["schema_version"] == 2
    assert payload["tier"] == "all"
    assert payload["smoke_result"] == _SMOKE
    assert payload["harness_result"] == _HARNESS
    # The speed buckets MUST still be present alongside the new
    # sections — tier='all' is additive, not "this is a non-speed run".
    assert "buckets" in payload
    assert {"short", "long"} <= payload["buckets"].keys()


def test_full_payload_validates_against_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(instance=_build_full(), schema=schema)


def test_tier_smoke_alone_validates() -> None:
    """``tier='smoke'`` populates only smoke_result; harness_result is
    omitted entirely. Must validate."""
    from vllm_mlx.community_bench.submission import build_submission_payload

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    hw, sw, bench = _stub_inputs()
    payload = build_submission_payload(
        hardware=hw,
        software=sw,
        alias="qwen3.5-9b-4bit",
        hf_path="mlx-community/Qwen3.5-9B-4bit",
        bench=bench,
        notes=None,
        now=datetime(2026, 6, 15, tzinfo=timezone.utc),
        tier="smoke",
        smoke_result=_SMOKE,
    )
    assert payload["tier"] == "smoke"
    assert payload["smoke_result"] == _SMOKE
    assert "harness_result" not in payload
    jsonschema.validate(instance=payload, schema=schema)


def test_tier_harness_alone_validates() -> None:
    from vllm_mlx.community_bench.submission import build_submission_payload

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    hw, sw, bench = _stub_inputs()
    payload = build_submission_payload(
        hardware=hw,
        software=sw,
        alias="qwen3.5-9b-4bit",
        hf_path="mlx-community/Qwen3.5-9B-4bit",
        bench=bench,
        notes=None,
        now=datetime(2026, 6, 15, tzinfo=timezone.utc),
        tier="harness",
        harness_result=_HARNESS,
    )
    assert payload["tier"] == "harness"
    assert payload["harness_result"] == _HARNESS
    assert "smoke_result" not in payload
    jsonschema.validate(instance=payload, schema=schema)
