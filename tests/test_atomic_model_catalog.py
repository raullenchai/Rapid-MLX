# SPDX-License-Identifier: Apache-2.0
"""Product-wide catalog, digest, registry, and shadow-migration contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from vllm_mlx.catalog import (
    AtomicRegistry,
    CatalogValidationError,
    ContractValidator,
    build_catalog_bundle,
    build_legacy_catalog_snapshot,
    build_legacy_recommendation_policy,
    canonical_json_bytes,
    rcj_digest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_packaged_schemas_are_exact_proto_copies() -> None:
    packaged = ROOT / "vllm_mlx" / "catalog" / "schemas"
    sources = list((ROOT / "proto" / "model-runtime" / "v1").glob("*.schema.json"))
    sources += list((ROOT / "proto" / "model-catalog" / "v1").glob("*.schema.json"))
    for source in sources:
        assert (packaged / source.name).read_bytes() == source.read_bytes()


def test_rcj_is_stable_and_rejects_nonportable_numbers() -> None:
    assert canonical_json_bytes({"z": "e\u0301", "a": 1}) == (
        '{"a":1,"z":"é"}'.encode()
    )
    assert rcj_digest({"a": [1, True, None]}) == rcj_digest({"a": [1, True, None]})
    with pytest.raises(TypeError, match="floating-point"):
        canonical_json_bytes({"score": 0.5})
    with pytest.raises(ValueError, match="safe range"):
        canonical_json_bytes({"integer": 9_007_199_254_740_992})
    with pytest.raises(TypeError, match="ASCII"):
        canonical_json_bytes({"é": "not an ASCII key"})


def test_legacy_projection_is_complete_deduplicated_and_schema_valid() -> None:
    snapshot = build_legacy_catalog_snapshot()
    ContractValidator().validate_catalog_snapshot(snapshot)
    aliases = {item["alias"]: item for item in snapshot["aliases"]}
    models = {item["registry_model_id"]: item for item in snapshot["models"]}
    assert len(aliases) == len(snapshot["aliases"])
    assert len(models) == len(snapshot["models"])
    assert len(models) < len(aliases), "aliases sharing artifacts must deduplicate"

    assert aliases["qwen3.8-27b-4bit"]["capabilities"]["task_types"] == [
        "text_generation"
    ]
    assert aliases["flux2-klein-4b"]["capabilities"]["operation_modes"] == [
        "text_to_image",
        "image_to_image",
    ]
    assert aliases["qwen3-aligner"]["capabilities"]["task_types"] == [
        "forced_alignment"
    ]
    assert aliases["qwen3-tts-clone"]["capabilities"]["operation_modes"] == [
        "voice_cloning"
    ]
    assert aliases["whisper-large-v3"]["capabilities"]["operation_modes"] == [
        "transcription",
        "translation",
    ]
    assert aliases["whisper-tiny"]["availability"]["desktop"] is False


def test_catalog_digest_covers_ordered_records_and_policy_reference() -> None:
    bundle = build_catalog_bundle()
    snapshot = bundle["snapshot"]
    assert snapshot["recommendation_policy_digests"] == [
        bundle["recommendation_policies"][0]["policy_digest"]
    ]
    changed = copy.deepcopy(snapshot)
    changed["aliases"][0]["availability"]["website"] = False
    projection = {
        key: changed[key]
        for key in (
            "schema_version",
            "models",
            "aliases",
            "recommendation_policy_digests",
        )
    }
    assert rcj_digest(projection) != snapshot["catalog_digest"]


def test_recommendation_adapter_scales_measurements_and_validates_tasks() -> None:
    snapshot = build_legacy_catalog_snapshot()
    policy = build_legacy_recommendation_policy(snapshot)
    first = policy["tiers"][0]["picks"][0]
    assert policy["machine_dimension"] == "physical_memory_mib"
    assert first["footprint_mib"] == 3 * 1024
    assert first["decode_tokens_per_second_x100"] == 9350
    assert first["evidence_status"] == "legacy_measured"

    aliases = {item["alias"]: item for item in snapshot["aliases"]}
    broken = copy.deepcopy(policy)
    broken["tiers"][0]["picks"][0]["alias"] = "kokoro"
    broken["policy_digest"] = rcj_digest(
        {key: value for key, value in broken.items() if key != "policy_digest"}
    )
    with pytest.raises(CatalogValidationError, match="policy task_type"):
        ContractValidator().validate_recommendation_policy(broken, aliases=aliases)


def test_shadow_bundle_preserves_legacy_alias_surface() -> None:
    report = build_catalog_bundle()["shadow_report"]
    assert report["mode"] == "shadow"
    assert report["equivalent"] is True
    assert report["failures"] == []
    assert report["legacy_alias_count"] == report["projected_alias_count"]
    assert report["task_counts"]["speech_synthesis"] > 0
    assert report["task_counts"]["speech_recognition"] > 0
    assert report["task_counts"]["video_generation"] > 0


def test_atomic_registry_is_idempotent_and_detects_tampering(tmp_path: Path) -> None:
    identity = json.loads(
        (
            ROOT / "proto/model-runtime/v1/examples/model-identity.llm.example.json"
        ).read_text()
    )
    registry = AtomicRegistry(tmp_path)
    digest = registry.put("model_identity", identity)
    assert registry.put("model_identity", identity) == digest
    assert registry.get("model_identity", digest) == identity

    path = tmp_path / "model_identity" / f"{digest.removeprefix('sha256:')}.json"
    stored = json.loads(path.read_text())
    stored["components"][0]["artifact"]["total_size_bytes"] += 1
    path.write_text(json.dumps(stored))
    with pytest.raises(CatalogValidationError, match="content-address"):
        registry.get("model_identity", digest)

    # Fields outside the digest projection are still contract-validated when
    # read; content addressing alone must not turn malformed metadata valid.
    path.write_bytes(canonical_json_bytes(identity) + b"\n")
    stored = json.loads(path.read_text())
    stored["identity_strength"] = "invented"
    path.write_text(json.dumps(stored))
    with pytest.raises(CatalogValidationError, match="identity_strength"):
        registry.get("model_identity", digest)


def test_catalog_snapshot_rejects_alias_target_drift() -> None:
    snapshot = build_legacy_catalog_snapshot()
    snapshot["aliases"][0]["target"]["registry_model_id"] = "legacy/hf/missing"
    snapshot["catalog_digest"] = rcj_digest(
        {
            key: snapshot[key]
            for key in (
                "schema_version",
                "models",
                "aliases",
                "recommendation_policy_digests",
            )
        }
    )
    with pytest.raises(CatalogValidationError, match="does not resolve"):
        ContractValidator().validate_catalog_snapshot(snapshot)
