# SPDX-License-Identifier: Apache-2.0
"""The submission must be acceptable to the service that receives it.

A model loaded from a warm Hugging Face cache records a ``resolved_revision``
and real quantization facts. The deployed ingestion validator allowlists
neither, and ``preview_run`` used to send the archived record verbatim — so the
ordinary case measured fine and then failed at upload with
``model.components[0].source.resolved_revision is not upload-allowlisted``.

These tests drive the real ``preview_run`` and validate its **exact** payload
against :mod:`tests.ingestion_contract`, a mirror of the worker's own rules.
The local ``BenchmarkRunValidator`` cannot catch this: it is the archive's
schema, and the archive is allowed to be richer than the wire.
"""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any

import pytest

from rapid_mlx.community_bench import atomic_upload
from rapid_mlx.community_bench.publication import (
    describe_withheld,
    project_run_for_publication,
)
from tests.ingestion_contract import (
    IngestionRejected,
    cross_check_against_worker,
    validate_model,
    validate_submission,
    worker_source,
)

#: A record produced by a REAL packaged benchmark run of
#: ``lfm2.5-1b-4bit`` on this machine, saved verbatim. Its model identity
#: carries a ``resolved_revision`` and 4-bit ``affine`` quantization facts
#: because the checkpoint was already in the Hugging Face cache — i.e. every
#: run after the first. Nothing here is hand-written to make a point.
_FIXTURE = Path(__file__).parent / "fixtures" / "community_bench_cached_run.json"


def _cached_run() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text())


def test_the_fixture_really_is_a_warm_cache_record() -> None:
    """Guards the premise: if the engine stopped recording provenance this
    whole suite would pass vacuously."""

    component = _cached_run()["model"]["components"][0]
    assert component["source"]["resolved_revision"]
    assert component["quantization"]["kind"] == "weights"


# ---------------------------------------------------------------------------
# The archived record is what the service rejects
# ---------------------------------------------------------------------------


def test_the_archived_record_is_rejected_by_the_service() -> None:
    """The premise, stated as a test: this is the bug being fixed."""

    with pytest.raises(IngestionRejected) as error:
        validate_submission(_cached_run())
    assert "resolved_revision is not upload-allowlisted" in str(error.value)


def test_quantization_alone_is_also_rejected() -> None:
    run = _cached_run()
    source = run["model"]["components"][0]["source"]
    del source["resolved_revision"]
    # The allowlist is checked before the value, so a real quantization block
    # is refused for carrying `group_size` at all — not merely for being
    # non-unknown. Either way it cannot be published.
    with pytest.raises(
        IngestionRejected,
        match=r"model\.components\[0\]\.quantization\.\w+ is not upload-allowlisted",
    ):
        validate_submission(run)


def test_a_two_key_non_unknown_quantization_is_also_rejected() -> None:
    """And one that fits the allowlist but states real facts is refused too."""

    run = _cached_run()
    component = run["model"]["components"][0]
    del component["source"]["resolved_revision"]
    component["quantization"] = {"kind": "weights", "base_dtype": "bfloat16"}
    with pytest.raises(IngestionRejected, match="model quantization"):
        validate_submission(run)


# ---------------------------------------------------------------------------
# The projected submission is accepted
# ---------------------------------------------------------------------------


def test_the_projection_is_accepted_by_the_service() -> None:
    public, _ = project_run_for_publication(_cached_run())
    validate_submission(public)  # raises on rejection


def test_the_exact_preview_payload_passes_the_ingestion_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bytes that would actually leave this Mac, not a reconstruction."""

    monkeypatch.setattr(atomic_upload, "peek_install_id", lambda: "0123456789ab")
    preview = atomic_upload.preview_run(
        _cached_run(), url="https://rapidmlx.com/api/benchmarks/atomic"
    )

    # The serialized body is the contract's subject, so parse it back rather
    # than trusting the parallel `payload` dict.
    on_the_wire = json.loads(preview["payload_json"])
    validate_submission(on_the_wire)
    assert on_the_wire == preview["payload"]

    source = on_the_wire["model"]["components"][0]["source"]
    assert source == {
        "kind": "huggingface",
        "repo_id": "mlx-community/LFM2.5-1.2B-Instruct-4bit",
    }
    assert on_the_wire["model"]["components"][0]["quantization"] == {
        "kind": "unknown",
        "base_dtype": "unknown",
    }


def test_a_bare_record_is_projected_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cold-cache run already satisfies the allowlist; nothing is touched."""

    bare = _cached_run()
    component = bare["model"]["components"][0]
    component["source"] = {
        "kind": "huggingface",
        "repo_id": "mlx-community/LFM2.5-1.2B-Instruct-4bit",
    }
    component["quantization"] = {"kind": "unknown", "base_dtype": "unknown"}

    public, withheld = project_run_for_publication(bare)
    assert public == bare
    assert withheld == []


@pytest.mark.parametrize("model", [None, "not-an-object", {"components": None}])
def test_projection_leaves_records_without_component_objects_unchanged(
    model: object,
) -> None:
    run = {"model": model, "measurements": []}
    public, withheld = project_run_for_publication(run)
    assert public == run
    assert withheld == []


def test_projection_skips_non_object_components() -> None:
    run = {"model": {"components": [None, "invalid"]}}
    public, withheld = project_run_for_publication(run)
    assert public == run
    assert withheld == []


def test_withheld_fact_value_semantics_are_type_safe() -> None:
    from rapid_mlx.community_bench.publication import WithheldFact

    fact = WithheldFact("model.source", "private", "not public")
    assert fact == WithheldFact("model.source", "private", "not public")
    assert fact != WithheldFact("model.source", "other", "not public")
    assert fact != object()
    assert "model.source" in repr(fact)


# ---------------------------------------------------------------------------
# Nothing is discarded silently
# ---------------------------------------------------------------------------


def test_the_local_record_is_not_mutated() -> None:
    run = _cached_run()
    before = copy.deepcopy(run)
    project_run_for_publication(run)
    assert run == before, "projecting the submission edited the archived record"


def test_every_withheld_fact_is_reported() -> None:
    _, withheld = project_run_for_publication(_cached_run())
    paths = {fact.path for fact in withheld}
    assert paths == {
        "model.components[0].source.resolved_revision",
        "model.components[0].quantization",
    }
    # The value is carried, not just the key, so the disclosure can show what
    # is being held back rather than only that something is.
    by_path = {fact.path: fact for fact in withheld}
    assert (
        by_path["model.components[0].source.resolved_revision"].value
        == "125e006d991147f3b432249d1bdf0821987f12b0"
    )
    assert by_path["model.components[0].quantization"].value["weight_bits_x2"] == 8
    assert all(fact.reason for fact in withheld)


def test_the_preview_carries_the_withheld_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atomic_upload, "peek_install_id", lambda: "0123456789ab")
    preview = atomic_upload.preview_run(
        _cached_run(), url="https://rapidmlx.com/api/benchmarks/atomic"
    )
    paths = {item["path"] for item in preview["withheld"]}
    assert "model.components[0].source.resolved_revision" in paths
    # Serializable, because a GUI reads this out of `--preview --json`.
    json.dumps(preview["withheld"])


def test_consent_prints_what_is_not_being_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atomic_upload, "peek_install_id", lambda: "0123456789ab")
    preview = atomic_upload.preview_run(
        _cached_run(), url="https://rapidmlx.com/api/benchmarks/atomic"
    )
    out = io.StringIO()
    atomic_upload._ask_consent(
        preview["payload"],
        target=preview["target"],
        stdin=io.StringIO("n\n"),
        stdout=out,
        withheld=preview["withheld"],
    )
    printed = out.getvalue()
    assert "stay on this Mac and are NOT in the payload above" in printed
    assert "resolved_revision" in printed
    assert "weight_bits_x2" in printed


def test_describe_withheld_is_empty_when_nothing_was_held_back() -> None:
    assert describe_withheld([]) == []


# ---------------------------------------------------------------------------
# The digests describe the document that is actually sent
# ---------------------------------------------------------------------------


def test_digests_cover_the_projected_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The approval flow compares digests across preview and upload. If they
    were taken before the projection, the user would approve one document and
    a different one would be sent."""

    monkeypatch.setattr(atomic_upload, "peek_install_id", lambda: "0123456789ab")
    preview = atomic_upload.preview_run(
        _cached_run(), url="https://rapidmlx.com/api/benchmarks/atomic"
    )
    recomputed = atomic_upload.atomic_run_digest(json.loads(preview["payload_json"]))
    assert recomputed == preview["payload_digest"]


# ---------------------------------------------------------------------------
# The mirror itself
# ---------------------------------------------------------------------------


def test_mirror_agrees_with_a_local_worker_checkout() -> None:
    """When a checkout is present, the mirror must match it.

    Skipped on machines without one — which is exactly why the mirror exists
    rather than the test reading the worker directly.
    """

    if worker_source() is None:
        pytest.skip("no rapidmlx.com worker checkout on this machine")
    assert cross_check_against_worker() == []


def test_mirror_rejects_a_submission_with_an_extra_model_key() -> None:
    run = _cached_run()
    public, _ = project_run_for_publication(run)
    public["model"]["notes"] = "hello"
    with pytest.raises(
        IngestionRejected, match="model.notes is not upload-allowlisted"
    ):
        validate_model(public["model"], "text_generation")
