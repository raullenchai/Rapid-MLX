# SPDX-License-Identifier: Apache-2.0
"""Schema v2 backwards-compatibility: every existing v1 submission file
on disk must continue to validate against the v2 schema.

Schema evolution from v1 → v2 is *additive only* — three optional
top-level fields (``tier``, ``smoke_result``, ``harness_result``) plus a
widened ``schema_version`` enum (``[1, 2]``). The historical v1
submissions in ``community-benchmarks/submissions/`` are the regression
corpus. If any of them ever stop validating, the evolution stopped
being additive and the schema change is wrong.

We re-validate by importing ``jsonschema`` directly rather than
shelling out to ``scripts/validate.py`` because we only want the
*schema* gate exercised here — the alias / sanity / dedup gates are
covered by the existing ``test_community_bench.py`` corpus tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "community-benchmarks" / "schema.json"
SUBMISSIONS_DIR = REPO_ROOT / "community-benchmarks" / "submissions"


def _existing_submissions() -> list[Path]:
    """Every JSON file in ``community-benchmarks/submissions/``.

    Empty list short-circuits the parametrize loop — pytest will skip
    the test rather than report zero collected, so the harness still
    reports a clean run on a fresh checkout that hasn't accumulated
    submissions yet.
    """
    if not SUBMISSIONS_DIR.exists():
        return []
    return sorted(SUBMISSIONS_DIR.glob("*.json"))


@pytest.mark.parametrize(
    "submission_path",
    _existing_submissions(),
    ids=lambda p: p.name,
)
def test_existing_v1_submission_validates_against_v2_schema(
    submission_path: Path,
) -> None:
    """Each v1 file in the corpus must validate against the v2 schema.

    The schema's ``schema_version`` enum was widened from ``const: 1``
    to ``enum: [1, 2]``; existing files keep ``schema_version: 1`` and
    none of the new ``tier`` / ``smoke_result`` / ``harness_result``
    fields. The top-level ``allOf`` includes a guard that REJECTS v1
    payloads carrying any v2-only field, so a v1 file that accidentally
    started using the new fields would fail loudly. Neither of those
    fixture files does — this test is the regression for that.
    """
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    payload = json.loads(submission_path.read_text())
    # Sanity: each corpus file we're guarding really IS a v1 payload.
    # If this ever flips to 2, the test below is no longer testing
    # backwards compatibility — it's just self-checking v2.
    assert payload["schema_version"] == 1, (
        f"{submission_path.name} is not a v1 submission anymore; "
        f"move it out of the v1 regression set."
    )
    # Will raise jsonschema.ValidationError if the schema rejects it.
    jsonschema.validate(instance=payload, schema=schema)


def test_corpus_is_nonempty() -> None:
    """Belt-and-braces: if the v1 corpus is empty, the parametrize
    loop above silently does nothing. Fail explicitly so a future
    accidental ``rm`` of the submissions dir doesn't hide a schema
    regression behind a green CI.
    """
    files = _existing_submissions()
    assert len(files) >= 2, (
        f"Expected at least the two seed v1 submissions in "
        f"{SUBMISSIONS_DIR}; found {len(files)}. The v1 regression "
        f"corpus must stay populated."
    )
