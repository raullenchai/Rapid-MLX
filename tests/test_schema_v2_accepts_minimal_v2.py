# SPDX-License-Identifier: Apache-2.0
"""Schema v2 minimal-shape acceptance: a payload that only bumps
``schema_version`` to 2 (and otherwise carries the exact v1 fields)
must validate.

This is the contract the aggregator banks on: an aggregator that
understands v2 can treat a minimal v2 row identically to a v1 row,
because the only difference on the wire is the integer in the version
slot. Without this guarantee, contributors who upgrade their CLI but
don't run the new ``--tier`` flag would suddenly be unable to submit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "community-benchmarks" / "schema.json"
SUBMISSIONS_DIR = REPO_ROOT / "community-benchmarks" / "submissions"


def _first_v1_payload() -> dict:
    """Read any existing v1 submission as a realistic starting point.

    Building one by hand from scratch would force us to track every
    ``const`` / ``required`` field the schema enforces — and the schema
    is the contract under test. Reusing a real v1 payload keeps the
    test honest: anything beyond the version bump is unchanged.
    """
    files = sorted(SUBMISSIONS_DIR.glob("*.json"))
    if not files:
        pytest.skip("no existing v1 submissions to base the test on")
    return json.loads(files[0].read_text())


def test_v2_minimal_payload_validates() -> None:
    """A v2 payload with no new fields populated is just a v1 payload
    with ``schema_version: 2``. It MUST validate.
    """
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())

    payload = _first_v1_payload()
    payload["schema_version"] = 2
    # No tier / smoke_result / harness_result — this is the "minimal v2"
    # case the aggregator must treat identically to v1.

    jsonschema.validate(instance=payload, schema=schema)


def test_v2_payload_with_tier_speed_validates() -> None:
    """Explicit ``tier: 'speed'`` is the legacy bench tier label and
    must NOT require smoke_result/harness_result. This is the path the
    existing ``--submit`` flow will take once PR #2 routes
    ``--tier speed --submit`` through ``build_submission_payload``.
    """
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())

    payload = _first_v1_payload()
    payload["schema_version"] = 2
    payload["tier"] = "speed"

    jsonschema.validate(instance=payload, schema=schema)


def test_v2_payload_with_full_tier_all_validates() -> None:
    """``tier: 'all'`` + populated smoke + harness is the full v2 row."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())

    payload = _first_v1_payload()
    payload["schema_version"] = 2
    payload["tier"] = "all"
    payload["smoke_result"] = {
        "boot_time_ms": 1234.0,
        "first_prompt_ok": True,
        "first_token_latency_ms": 87.5,
        "response_excerpt": "Hello! I'd be happy to help.",
    }
    payload["harness_result"] = {
        adapter: {"passed": True, "duration_s": 12.3, "error_excerpt": None}
        for adapter in ("codex", "opencode", "hermes", "aider", "langchain")
    }

    jsonschema.validate(instance=payload, schema=schema)


def test_v1_payload_with_v2_field_is_rejected() -> None:
    """A v1 payload carrying a v2-only field is ambiguous (aggregator
    would have to guess which tier produced it). The ``allOf`` block in
    the schema enforces this rejection so the wire shape per version
    stays unambiguous.
    """
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())

    payload = _first_v1_payload()
    # schema_version stays 1 — but we sneak in a v2-only field.
    payload["tier"] = "speed"

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=schema)
