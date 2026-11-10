# SPDX-License-Identifier: Apache-2.0
"""Publication preserves known identity facts accepted by the service.

These tests validate the real preview bytes with the server contract mirror.
"""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any

import pytest

from rapid_mlx.community_bench import atomic_upload
from rapid_mlx.community_bench.benchmark_contracts import BenchmarkRunValidator
from rapid_mlx.community_bench.hardware import Hardware, Software
from rapid_mlx.community_bench.publication import (
    describe_withheld,
    project_run_for_publication,
)
from rapid_mlx.community_bench.run_builder import (
    build_run,
    execution_config,
    execution_config_digest,
)
from tests.ingestion_contract import (
    IngestionRejected,
    cross_check_against_worker,
    validate_model,
    validate_submission,
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
# Cached identity facts are now accepted by the service
# ---------------------------------------------------------------------------


def test_the_archived_record_is_accepted_by_the_service() -> None:
    validate_submission(_cached_run())


def test_quantization_alone_is_also_accepted() -> None:
    run = _cached_run()
    del run["model"]["components"][0]["source"]["resolved_revision"]
    validate_submission(run)


def test_a_two_key_non_unknown_quantization_is_also_rejected() -> None:
    """A weights block must still carry method and bit width."""

    run = _cached_run()
    component = run["model"]["components"][0]
    del component["source"]["resolved_revision"]
    component["quantization"] = {"kind": "weights", "base_dtype": "bfloat16"}
    with pytest.raises(IngestionRejected, match="requires method and weight_bits_x2"):
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

    assert on_the_wire["model"] == _cached_run()["model"]
    assert preview["withheld"] == []


def test_known_facts_survive_the_real_publication_contract(monkeypatch) -> None:
    monkeypatch.setattr(atomic_upload, "peek_install_id", lambda: "0123456789ab")
    run = _cached_run()
    component = run["model"]["components"][0]
    component["source"].update(resolved_revision="a" * 40, subfolder="variants/4bit")
    component["quantization"] = {
        "kind": "weights_and_activations",
        "method": "affine",
        "weight_bits_x2": 8,
        "activation_bits_x2": 16,
        "group_size": 64,
        "base_dtype": "bfloat16",
    }
    run = build_run(
        repo_id=component["source"]["repo_id"],
        task_type="text_generation",
        hardware=Hardware("Apple M4 Pro", 24, 12, 16),
        software=Software("15.6", "0.13.4", "0.32.2", "3.12.13"),
        started_at=run["started_at"],
        model_identity=run["model"],
        execution=run["execution"],
        workload=run["workload"],
        measurements=run["measurements"],
    )
    before = copy.deepcopy(run)
    preview = atomic_upload.preview_run(run)
    payload = json.loads(preview["payload_json"])
    validate_submission(payload)
    assert payload["model"] == before["model"]
    assert preview["withheld"] == []
    assert run == before


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


def test_chunk_size_and_dtype_are_published_unchanged() -> None:
    run = _cached_run()
    component = run["model"]["components"][0]
    component["source"] = {
        "kind": "huggingface",
        "repo_id": component["source"]["repo_id"],
    }
    component["quantization"] = {"kind": "unknown", "base_dtype": "unknown"}
    language = copy.deepcopy(run["execution"]["task"]["language"])
    language["kv_cache"] = {"mode": "full_precision", "dtype": "bfloat16"}
    language["prefill_chunk_size"] = 2048
    language["pflash"] = {
        "mode": "auto",
        "threshold": 1024,
        "keep_ratio_millionths": 200000,
    }
    language["speculative_decoding"] = {
        "method": "mtp",
        "assistant_source": "external_model",
        "max_draft_tokens": 4,
        "draft_model_identity_digest": "sha256:" + "a" * 64,
    }
    run["execution"] = execution_config(
        "text_generation", language=language, compute_dtype="bfloat16"
    )
    before = copy.deepcopy(run)

    public, withheld = project_run_for_publication(run)

    assert public["execution"]["task"]["language"] == language
    assert public["execution"]["resources"]["compute_dtype"] == "bfloat16"
    assert withheld == []
    assert public["execution"]["config_digest"] == execution_config_digest(
        public["execution"]
    )
    assert public["execution"]["config_digest"] == run["execution"]["config_digest"]
    assert run == before
    BenchmarkRunValidator().validate(public)


@pytest.mark.parametrize("chunk_size", [512, 1024])
def test_other_chunk_sizes_are_published_unchanged(chunk_size: int) -> None:
    run = _cached_run()
    component = run["model"]["components"][0]
    component["source"] = {
        "kind": "huggingface",
        "repo_id": component["source"]["repo_id"],
    }
    component["quantization"] = {"kind": "unknown", "base_dtype": "unknown"}
    language = copy.deepcopy(run["execution"]["task"]["language"])
    language["prefill_chunk_size"] = chunk_size
    run["execution"] = execution_config("text_generation", language=language)

    public, withheld = project_run_for_publication(run)

    assert public == run
    assert withheld == []
    assert public["execution"]["config_digest"] == run["execution"]["config_digest"]


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
    run = _cached_run()
    source = run["model"]["components"][0]["source"]
    source["local_path"] = "/Users/private/checkpoint"
    source["future_fact"] = "not accepted"
    public, withheld = project_run_for_publication(run)
    assert {fact.path: fact.value for fact in withheld} == {
        "model.components[0].source.local_path": "/Users/private/checkpoint",
        "model.components[0].source.future_fact": "not accepted",
    }
    assert public["model"]["components"][0]["source"] == {
        key: value
        for key, value in source.items()
        if key not in {"local_path", "future_fact"}
    }
    assert all(fact.reason for fact in withheld)


@pytest.mark.parametrize("kind", ["local", "unknown"])
def test_non_huggingface_sources_never_publish_local_provenance(kind) -> None:
    run = {
        "model": {
            "components": [
                {
                    "source": {
                        "kind": kind,
                        "path": "/Users/private/checkpoint",
                        "resolved_revision": "/Users/private/revision",
                        "subfolder": "/Users/private/variant",
                    }
                }
            ]
        }
    }
    public, withheld = project_run_for_publication(run)
    assert public["model"]["components"][0]["source"] == {"kind": kind}
    assert {fact.path.rsplit(".", 1)[1] for fact in withheld} == {
        "path",
        "resolved_revision",
        "subfolder",
    }


def test_the_preview_carries_the_withheld_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atomic_upload, "peek_install_id", lambda: "0123456789ab")
    preview = atomic_upload.preview_run(
        _cached_run(), url="https://rapidmlx.com/api/benchmarks/atomic"
    )
    assert preview["withheld"] == []
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
    assert "stay on this Mac and are NOT in the payload above" not in printed
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
    """Always check the fixture; also require equality with any site checkout."""

    assert cross_check_against_worker() == []


def test_mirror_rejects_a_submission_with_an_extra_model_key() -> None:
    run = _cached_run()
    public, _ = project_run_for_publication(run)
    public["model"]["notes"] = "hello"
    with pytest.raises(
        IngestionRejected, match="model.notes is not upload-allowlisted"
    ):
        validate_model(public["model"], "text_generation")


@pytest.mark.parametrize(
    "quantization",
    [
        {"kind": "none", "base_dtype": "float16"},
        {"kind": "mixed", "method": "affine", "weight_bits_x2": 7},
        {"kind": "unknown"},
        {"kind": "weights", "method": "affine", "weight_bits_x2": 8, "group_size": 64},
    ],
)
def test_all_accepted_quantization_kinds_survive(quantization):
    run = _cached_run()
    run["model"]["components"][0]["quantization"] = quantization
    public, withheld = project_run_for_publication(run)
    validate_submission(public)
    assert public == run
    assert withheld == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("resolved_revision", "main"),
        ("resolved_revision", "A" * 40),
        ("subfolder", "/Users/private/model"),
        ("subfolder", "../model"),
        ("subfolder", "variant\\model"),
        ("subfolder", ""),
    ],
)
def test_mirror_rejects_invalid_source_provenance(field, value):
    run = _cached_run()
    run["model"]["components"][0]["source"][field] = value
    with pytest.raises(IngestionRejected, match=field):
        validate_submission(run)


@pytest.mark.parametrize(
    "quantization",
    [
        {"kind": "weights", "method": "affine"},
        {"kind": "weights_and_activations", "method": "affine", "weight_bits_x2": 8},
        {"kind": "none", "weight_bits_x2": 8},
        {"kind": "unknown", "weight_bits_x2": True},
        {"kind": "unknown", "weight_bits_x2": 1},
        {"kind": "unknown", "activation_bits_x2": 65},
        {"kind": "unknown", "group_size": 4097},
        {"kind": "unknown", "method": "/Users/private/method"},
        {"kind": "unknown", "base_dtype": "fp16"},
        {"kind": "other"},
        {"kind": "unknown", "local_path": "/Users/private/model"},
    ],
)
def test_mirror_rejects_invalid_quantization(quantization):
    run = _cached_run()
    run["model"]["components"][0]["quantization"] = quantization
    with pytest.raises(IngestionRejected, match="quantization"):
        validate_submission(run)


def test_contract_uses_fixture_without_a_site_checkout(monkeypatch):
    import tests.ingestion_contract as contract

    monkeypatch.delenv("RAPID_MLX_SITE_CHECKOUT", raising=False)
    monkeypatch.setattr(contract, "_WORKER_CANDIDATES", ())
    assert contract.cross_check_against_worker() == []


def test_configured_site_checkout_must_be_available(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPID_MLX_SITE_CHECKOUT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="configured site checkout"):
        cross_check_against_worker()


def test_consent_lists_only_the_remaining_withheld_fields():
    run = _cached_run()
    run["model"]["components"][0]["source"]["path"] = "/Users/private/model"
    public, withheld = project_run_for_publication(run)
    out = io.StringIO()
    atomic_upload._ask_consent(
        public,
        target="https://rapidmlx.com/api/benchmarks/atomic",
        stdin=io.StringIO("n\n"),
        stdout=out,
        withheld=[fact.as_dict() for fact in withheld],
    )
    disclosure = out.getvalue().split("These details stay on this Mac", 1)[1]
    assert "model.components[0].source.path" in disclosure
    assert "/Users/private/model" in disclosure
    assert "resolved_revision" not in disclosure
    assert "quantization" not in disclosure
    assert "/Users/private/model" not in json.dumps(public)
