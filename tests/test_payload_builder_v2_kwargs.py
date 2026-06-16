# SPDX-License-Identifier: Apache-2.0
"""``build_submission_payload`` v2 kwarg contract.

The existing ``--submit`` flow calls ``build_submission_payload``
without the new v2 kwargs. After bumping ``SCHEMA_VERSION`` to 2 and
adding the kwargs, that flow MUST keep producing the same wire shape
it always did — modulo the version integer itself.

This pins three properties:

1. Default-kwargs ``build_submission_payload`` produces a payload
   identical to a v1 payload except for ``schema_version`` (now 2).
2. ``tier="speed"`` produces a payload with ``"tier": "speed"`` added
   and nothing else changed.
3. The payload still validates against the v2 schema.

If any of these break, a contributor who upgrades their CLI but
doesn't change their workflow has their submissions silently rejected
or — worse — silently mislabelled.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "community-benchmarks" / "schema.json"


def _stub_inputs():
    """Recreate the synthetic Hardware/Software/BenchResult bundle that
    the existing ``tests/test_community_bench.py`` uses for payload tests.
    Keeping the inputs in lockstep here avoids cross-test imports.
    """
    from vllm_mlx.community_bench.hardware import Hardware, Software
    from vllm_mlx.community_bench.runner import (
        BenchResult,
        BucketResult,
        RoundResult,
    )

    hw = Hardware(chip="Apple M4 Pro", ram_gb=24, cpu_cores=12, gpu_cores=20)
    sw = Software(macos="26.5.1", rapid_mlx="0.7.6", mlx="0.31.2", python="3.12.13")
    rounds = [
        RoundResult(decode_tps=42.0, prefill_tps=500.0, ttft_ms=120.0) for _ in range(5)
    ]
    bench = BenchResult(
        short=BucketResult(rounds_raw=rounds),
        long=BucketResult(rounds_raw=rounds),
        peak_ram_mb=8192,
        prompt_hash="deadbeefcafebabe",
        sampling="greedy",
    )
    return hw, sw, bench


def _build(tier=None, smoke_result=None, harness_result=None) -> dict:
    from vllm_mlx.community_bench.submission import build_submission_payload

    hw, sw, bench = _stub_inputs()
    return build_submission_payload(
        hardware=hw,
        software=sw,
        alias="qwen3.5-9b-4bit",
        hf_path="mlx-community/Qwen3.5-9B-4bit",
        bench=bench,
        notes="unit test",
        now=datetime(2026, 6, 15, 10, 30, 0, tzinfo=timezone.utc),
        tier=tier,
        smoke_result=smoke_result,
        harness_result=harness_result,
    )


def test_default_kwargs_emit_no_v2_only_fields() -> None:
    """No tier kwarg ⇒ no tier / smoke_result / harness_result in the
    wire shape. This is what makes a default-flow v2 submission a
    drop-in superset of v1.
    """
    payload = _build()
    assert "tier" not in payload
    assert "smoke_result" not in payload
    assert "harness_result" not in payload


def test_default_kwargs_match_v1_modulo_version() -> None:
    """The byte-level snapshot: pop the version integer and the payload
    is identical to what the same call produced under v1 (which we
    reconstruct here by mutating the version to 1).
    """
    payload = _build()
    assert payload["schema_version"] == 2
    # ``submission_id`` is uuid4-derived so it's not byte-stable across
    # runs; everything else IS deterministic given the frozen ``now``,
    # the hardcoded sampling, and the synthetic bench result.
    payload.pop("submission_id")
    payload["schema_version"] = 1  # the only legitimate v1↔v2 delta
    expected = {
        "schema_version": 1,
        "submitted_at": "2026-06-15T10:30:00+00:00",
        "hardware": {
            "chip": "Apple M4 Pro",
            "ram_gb": 24,
            "cpu_cores": 12,
            "gpu_cores": 20,
        },
        "software": {
            "macos": "26.5.1",
            "rapid_mlx": "0.7.6",
            "mlx": "0.31.2",
            "python": "3.12.13",
        },
        "model": {
            "alias": "qwen3.5-9b-4bit",
            "hf_path": "mlx-community/Qwen3.5-9B-4bit",
        },
        # config / buckets are tested for shape by
        # ``test_build_payload_matches_schema`` in the existing suite;
        # here we only need the v1-equivalence delta, not the full
        # nested literal.
        "config": payload["config"],
        "buckets": payload["buckets"],
        "notes": "unit test",
        "peak_ram_mb": 8192,
    }
    assert payload == expected


def test_tier_speed_emits_only_tier_field() -> None:
    """Explicit ``tier="speed"`` adds nothing other than the tier
    string itself. This is the path PR #2 will route the existing
    ``--submit`` flow through.
    """
    payload = _build(tier="speed")
    assert payload["tier"] == "speed"
    assert "smoke_result" not in payload
    assert "harness_result" not in payload


def test_tier_speed_payload_validates_against_v2_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(instance=_build(tier="speed"), schema=schema)


def test_default_kwargs_payload_validates_against_v2_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(instance=_build(), schema=schema)


def test_invalid_tier_value_raises() -> None:
    """Builder validates tier values at the boundary — the schema check
    in CI would also catch this, but failing in-process gives the
    future CLI dispatcher a clean ``ValueError`` to report.
    """
    with pytest.raises(ValueError, match="tier must be one of"):
        _build(tier="totally-made-up")


def test_tier_smoke_without_smoke_result_raises() -> None:
    """Boundary check: ``tier='smoke'`` without the matching result
    block is a misuse, not an empty bucket."""
    with pytest.raises(ValueError, match="smoke_result"):
        _build(tier="smoke")


def test_tier_harness_without_harness_result_raises() -> None:
    with pytest.raises(ValueError, match="harness_result"):
        _build(tier="harness")


def test_smoke_result_without_smoke_tier_raises() -> None:
    """The inverse: passing data with no matching tier is a mislabel."""
    smoke = {
        "boot_time_ms": 100.0,
        "first_prompt_ok": True,
        "first_token_latency_ms": 10.0,
        "response_excerpt": "ok",
    }
    with pytest.raises(ValueError, match="smoke_result was provided"):
        _build(tier="speed", smoke_result=smoke)


def test_harness_result_without_harness_tier_raises() -> None:
    harness = {
        adapter: {"passed": True, "duration_s": 1.0, "error_excerpt": None}
        for adapter in ("codex", "opencode", "hermes", "aider", "langchain")
    }
    with pytest.raises(ValueError, match="harness_result was provided"):
        _build(tier="speed", harness_result=harness)
