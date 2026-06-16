# SPDX-License-Identifier: Apache-2.0
"""Schema v2 tier ↔ result invariants: when ``tier`` declares a bucket,
the matching result block MUST be present.

Two invariants enforced by the top-level ``allOf`` in
``community-benchmarks/schema.json``:

1. ``tier in ("smoke", "all")``    → ``smoke_result`` required.
2. ``tier in ("harness", "all")`` → ``harness_result`` required.

Without these checks, an aggregator can't tell whether a missing block
means "this tier didn't run" or "the tier ran but the data was lost".
Both interpretations corrupt the dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "community-benchmarks" / "schema.json"
SUBMISSIONS_DIR = REPO_ROOT / "community-benchmarks" / "submissions"


def _v2_base_payload() -> dict:
    """Realistic v2-shaped starting payload: a real v1 row with the
    version bumped. Tests below populate the tier/result fields."""
    files = sorted(SUBMISSIONS_DIR.glob("*.json"))
    if not files:
        pytest.skip("no existing v1 submissions to base the test on")
    payload = json.loads(files[0].read_text())
    payload["schema_version"] = 2
    return payload


_VALID_SMOKE = {
    "boot_time_ms": 1234.0,
    "first_prompt_ok": True,
    "first_token_latency_ms": 87.5,
    "response_excerpt": "Hello!",
}
_VALID_HARNESS = {
    adapter: {"passed": True, "duration_s": 12.3, "error_excerpt": None}
    for adapter in ("codex", "opencode", "hermes", "aider", "langchain")
}


def test_tier_harness_without_harness_result_rejected() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())

    payload = _v2_base_payload()
    payload["tier"] = "harness"
    # No harness_result populated — must fail.
    with pytest.raises(jsonschema.ValidationError) as excinfo:
        jsonschema.validate(instance=payload, schema=schema)
    # Make sure the error fingerprint identifies the missing field —
    # otherwise a schema bug could trip "rejected" via the wrong path
    # (e.g. ``tier`` value rejected) and the test would pass for the
    # wrong reason.
    assert "harness_result" in str(excinfo.value)


def test_tier_smoke_without_smoke_result_rejected() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())

    payload = _v2_base_payload()
    payload["tier"] = "smoke"
    with pytest.raises(jsonschema.ValidationError) as excinfo:
        jsonschema.validate(instance=payload, schema=schema)
    assert "smoke_result" in str(excinfo.value)


def test_tier_all_without_either_result_rejected() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())

    payload = _v2_base_payload()
    payload["tier"] = "all"
    # Neither smoke_result nor harness_result. Must fail on the FIRST
    # missing field that jsonschema reports — both invariants apply.
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=schema)


def test_tier_all_with_only_smoke_rejected() -> None:
    """``tier=all`` requires BOTH blocks; only one is still partial."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())

    payload = _v2_base_payload()
    payload["tier"] = "all"
    payload["smoke_result"] = _VALID_SMOKE
    with pytest.raises(jsonschema.ValidationError) as excinfo:
        jsonschema.validate(instance=payload, schema=schema)
    assert "harness_result" in str(excinfo.value)


def test_harness_result_missing_adapter_rejected() -> None:
    """Every one of the five adapter slots is required when
    ``harness_result`` is populated. Omitting one would let a
    contributor publish a partial gauntlet result.
    """
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())

    payload = _v2_base_payload()
    payload["tier"] = "harness"
    partial = dict(_VALID_HARNESS)
    del partial["langchain"]
    payload["harness_result"] = partial
    with pytest.raises(jsonschema.ValidationError) as excinfo:
        jsonschema.validate(instance=payload, schema=schema)
    assert "langchain" in str(excinfo.value)


def test_harness_result_extra_adapter_rejected() -> None:
    """``additionalProperties: false`` on ``harness_result`` rejects
    unknown adapter slots — otherwise a contributor could ship an
    ad-hoc adapter the dashboard would silently ignore.
    """
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())

    payload = _v2_base_payload()
    payload["tier"] = "harness"
    extra = dict(_VALID_HARNESS)
    extra["my-cool-adapter"] = {
        "passed": True,
        "duration_s": 1.0,
        "error_excerpt": None,
    }
    payload["harness_result"] = extra
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=schema)


def test_smoke_result_excerpt_oversize_rejected() -> None:
    """``response_excerpt`` is capped at 200 chars to keep submissions
    compact. A 1000-char paste from a chatty model would otherwise leak
    arbitrary content into the corpus.
    """
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())

    payload = _v2_base_payload()
    payload["tier"] = "smoke"
    payload["smoke_result"] = dict(_VALID_SMOKE, response_excerpt="x" * 201)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=schema)
