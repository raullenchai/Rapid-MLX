# SPDX-License-Identifier: Apache-2.0
"""Product-wide catalog, digest, registry, and shadow-migration contracts."""

from __future__ import annotations

import copy
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import vllm_mlx.catalog.registry as registry_module
from vllm_mlx.catalog import (
    AtomicRegistry,
    CatalogValidationError,
    ContractValidator,
    build_catalog_bundle,
    build_legacy_catalog_snapshot,
    build_legacy_recommendation_policy,
    canonical_json_bytes,
    load_product_recommendation_policy,
    rcj_digest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_packaged_schemas_are_exact_proto_copies() -> None:
    packaged = ROOT / "vllm_mlx" / "catalog" / "schemas"
    copies = {
        ROOT
        / "proto/model-runtime/v1/model-identity.schema.json": "model-identity.schema.json",
        ROOT
        / "proto/model-runtime/v1/machine-observation.schema.json": "machine-observation.schema.json",
        ROOT
        / "proto/model-runtime/v1/execution-config.schema.json": "execution-config.schema.json",
        ROOT
        / "proto/model-catalog/v1/model-alias.schema.json": "model-alias-v1.schema.json",
        ROOT
        / "proto/model-catalog/v2/model-alias.schema.json": "model-alias.schema.json",
        ROOT
        / "proto/model-catalog/v1/model-registry-record.schema.json": "model-registry-record.schema.json",
        ROOT
        / "proto/model-catalog/v1/recommendation-policy.schema.json": "recommendation-policy.schema.json",
        ROOT
        / "proto/model-catalog/v1/catalog-snapshot.schema.json": "catalog-snapshot-v1.schema.json",
        ROOT
        / "proto/model-catalog/v2/catalog-snapshot.schema.json": "catalog-snapshot.schema.json",
    }
    for source, destination in copies.items():
        assert (packaged / destination).read_bytes() == source.read_bytes()


def test_registry_subfolder_contract_matches_consumers() -> None:
    record = copy.deepcopy(build_legacy_catalog_snapshot()["models"][0])
    record["source"]["subfolder"] = "quant/"
    with pytest.raises(CatalogValidationError, match="subfolder"):
        ContractValidator().validate("model_registry_record", record)


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
    with pytest.raises(TypeError, match="keys must be strings"):
        canonical_json_bytes({1: "not a string key"})
    with pytest.raises(TypeError, match="cannot encode bytes"):
        canonical_json_bytes(b"not JSON")


def test_legacy_projection_covers_image_edit_and_vision_profiles() -> None:
    from vllm_mlx.catalog.legacy import _main_capabilities

    image = SimpleNamespace(
        modality="image-gen",
        hf_path="publisher/qwen-image-edit",
        is_text_only=False,
        experimental=False,
    )
    vision = SimpleNamespace(
        modality="vision",
        hf_path="publisher/vlm",
        is_text_only=False,
        experimental=False,
    )
    assert _main_capabilities(image)["operation_modes"] == ["image_to_image"]
    assert _main_capabilities(vision)["runtime_adapter"] == "mlx_vlm"


def test_legacy_projection_is_complete_deduplicated_and_schema_valid() -> None:
    snapshot = build_legacy_catalog_snapshot()
    ContractValidator().validate_catalog_snapshot(snapshot)
    aliases = {item["alias"]: item for item in snapshot["aliases"]}
    models = {item["registry_model_id"]: item for item in snapshot["models"]}
    assert len(aliases) == len(snapshot["aliases"])
    assert len(models) == len(snapshot["models"])
    assert len(models) < len(aliases), "aliases sharing artifacts must deduplicate"

    assert aliases["qwen3.8-27b-4bit"]["capabilities"]["task_types"] == [
        "text_generation",
        "vision_language",
    ]
    assert aliases["qwen3.8-27b-4bit"]["capabilities"]["operation_modes"] == [
        "chat",
        "image_understanding",
    ]
    assert aliases["qwen3.8-27b-4bit"]["capabilities"]["is_text_only"] is False
    assert aliases["flux2-klein-4b"]["capabilities"]["operation_modes"] == [
        "text_to_image",
        "image_to_image",
    ]
    assert aliases["qwen3-aligner"]["capabilities"]["task_types"] == [
        "speech_recognition"
    ]
    assert aliases["qwen3-aligner"]["capabilities"]["operation_modes"] == [
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


def test_legacy_projection_includes_uncached_user_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm_mlx.model_aliases import list_builtin_aliases
    from vllm_mlx.user_aliases import set_user_alias

    monkeypatch.setenv(
        "RAPID_MLX_USER_ALIASES_FILE", str(tmp_path / "user-aliases.json")
    )
    set_user_alias("MyModel", "qwen3.8-27b-4bit", list_builtin_aliases())

    snapshot = build_legacy_catalog_snapshot()
    aliases = {item["alias"]: item for item in snapshot["aliases"]}
    assert aliases["MyModel"]["target"] == aliases["qwen3.8-27b-4bit"]["target"]
    assert aliases["MyModel"]["origin"] == "user"
    assert aliases["qwen3.8-27b-4bit"]["origin"] == "builtin"
    assert build_catalog_bundle()["shadow_report"]["equivalent"] is True


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


def test_product_recommendation_policy_is_atomic_ssot_and_validates_tasks() -> None:
    bundle = build_catalog_bundle()
    snapshot = bundle["snapshot"]
    policy = bundle["recommendation_policies"][0]
    assert policy == load_product_recommendation_policy(snapshot)
    assert policy == build_legacy_recommendation_policy(snapshot)
    assert all(
        "minimum_memory_mib" in tier and "floor_gb" not in tier
        for tier in policy["tiers"]
    )
    assert all(
        "footprint_gb" not in pick for tier in policy["tiers"] for pick in tier["picks"]
    )
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

    unresolved = copy.deepcopy(policy)
    unresolved["tiers"][0]["picks"][0]["alias"] = "missing-recommendation-alias"
    unresolved["policy_digest"] = rcj_digest(
        {key: value for key, value in unresolved.items() if key != "policy_digest"}
    )
    with pytest.raises(CatalogValidationError, match="does not resolve"):
        ContractValidator().validate_recommendation_policy(unresolved, aliases=aliases)


def test_shadow_bundle_preserves_legacy_alias_surface() -> None:
    report = build_catalog_bundle()["shadow_report"]
    assert report["mode"] == "shadow"
    assert report["equivalent"] is True
    assert report["failures"] == []
    assert report["legacy_alias_count"] == report["projected_alias_count"]
    assert report["task_counts"]["speech_synthesis"] > 0
    assert report["task_counts"]["speech_recognition"] > 0
    assert report["task_counts"]["video_generation"] > 0


def test_shadow_report_names_alias_and_recommendation_drift() -> None:
    from vllm_mlx.catalog.legacy import build_shadow_report

    bundle = build_catalog_bundle()
    snapshot = copy.deepcopy(bundle["snapshot"])
    policy = copy.deepcopy(bundle["recommendation_policies"][0])
    removed = snapshot["aliases"].pop()
    policy["tiers"][0]["picks"][0]["alias"] = removed["alias"]
    report = build_shadow_report(snapshot, policy)
    assert report["failures"] == [
        "alias_set_mismatch",
        "recommendation_alias_missing",
    ]


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


def test_atomic_registry_accepts_schema_valid_unresolved_identity(
    tmp_path: Path,
) -> None:
    identity = json.loads(
        (
            ROOT / "proto/model-runtime/v1/examples/model-identity.llm.example.json"
        ).read_text()
    )
    identity["identity_strength"] = "unresolved"
    identity.pop("identity_digest")
    registry = AtomicRegistry(tmp_path)

    digest = registry.put("model_identity", identity)
    assert digest == rcj_digest(
        {
            key: identity[key]
            for key in ("schema_version", "pipeline_kind", "components")
        }
    )
    assert registry.get("model_identity", digest) == identity


def test_atomic_registry_rechecks_declared_digest_on_read(tmp_path: Path) -> None:
    identity = json.loads(
        (
            ROOT / "proto/model-runtime/v1/examples/model-identity.llm.example.json"
        ).read_text()
    )
    registry = AtomicRegistry(tmp_path)
    digest = registry.put("model_identity", identity)
    path = tmp_path / "model_identity" / f"{digest.removeprefix('sha256:')}.json"
    tampered = copy.deepcopy(identity)
    tampered["identity_digest"] = "sha256:" + "f" * 64
    path.write_text(json.dumps(tampered))

    with pytest.raises(CatalogValidationError, match="declared digest"):
        registry.get("model_identity", digest)


def test_atomic_registry_rejects_noncanonical_model_components(tmp_path: Path) -> None:
    identity = json.loads(
        (
            ROOT / "proto/model-runtime/v1/examples/model-identity.llm.example.json"
        ).read_text()
    )
    duplicate = copy.deepcopy(identity["components"][0])
    duplicate["role"] = "adapter"
    identity["components"].append(duplicate)
    identity["identity_digest"] = rcj_digest(
        {
            key: identity[key]
            for key in ("schema_version", "pipeline_kind", "components")
        }
    )
    with pytest.raises(CatalogValidationError, match="component_id values"):
        AtomicRegistry(tmp_path).put("model_identity", identity)

    identity["components"][1]["component_id"] = "adapter"
    identity["identity_digest"] = rcj_digest(
        {
            key: identity[key]
            for key in ("schema_version", "pipeline_kind", "components")
        }
    )
    with pytest.raises(CatalogValidationError, match="must be sorted"):
        AtomicRegistry(tmp_path).put("model_identity", identity)


def test_atomic_registry_requires_catalog_context_for_recommendations(
    tmp_path: Path,
) -> None:
    bundle = build_catalog_bundle()
    snapshot = bundle["snapshot"]
    policy = bundle["recommendation_policies"][0]
    registry = AtomicRegistry(tmp_path)
    with pytest.raises(ValueError, match="requires catalog_snapshot"):
        registry.put("recommendation_policy", policy)

    invalid = copy.deepcopy(policy)
    invalid["tiers"][1]["minimum_memory_mib"] = 1
    invalid["policy_digest"] = rcj_digest(
        {key: value for key, value in invalid.items() if key != "policy_digest"}
    )
    with pytest.raises(CatalogValidationError, match="strictly increasing"):
        registry.put("recommendation_policy", invalid, catalog_snapshot=snapshot)

    missing_preset = copy.deepcopy(policy)
    missing_preset["tiers"][0]["picks"][0]["execution_preset_id"] = "missing"
    missing_preset["policy_digest"] = rcj_digest(
        {key: value for key, value in missing_preset.items() if key != "policy_digest"}
    )
    with pytest.raises(CatalogValidationError, match="does not resolve"):
        registry.put(
            "recommendation_policy",
            missing_preset,
            catalog_snapshot=snapshot,
        )

    digest = registry.put("recommendation_policy", policy, catalog_snapshot=snapshot)
    assert (
        registry.get("recommendation_policy", digest, catalog_snapshot=snapshot)
        == policy
    )


def test_atomic_registry_never_replaces_a_concurrent_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = json.loads(
        (
            ROOT / "proto/model-runtime/v1/examples/model-identity.llm.example.json"
        ).read_text()
    )
    winner = copy.deepcopy(first)
    winner["identity_strength"] = "repository_revision"
    winner_payload = canonical_json_bytes(winner) + b"\n"

    def publish_winner(_source: object, target: object) -> None:
        Path(target).write_bytes(winner_payload)
        raise FileExistsError

    monkeypatch.setattr(registry_module.os, "link", publish_winner)
    registry = AtomicRegistry(tmp_path)
    with pytest.raises(CatalogValidationError, match="content-address collision"):
        registry.put("model_identity", first)

    digest = first["identity_digest"]
    stored = tmp_path / "model_identity" / f"{digest.removeprefix('sha256:')}.json"
    assert stored.read_bytes() == winner_payload


def test_atomic_registry_syncs_directory_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = json.loads(
        (
            ROOT / "proto/model-runtime/v1/examples/model-identity.llm.example.json"
        ).read_text()
    )
    real_fsync = registry_module.os.fsync
    synced_directory = False

    def observing_fsync(descriptor: int) -> None:
        nonlocal synced_directory
        if stat.S_ISDIR(registry_module.os.fstat(descriptor).st_mode):
            synced_directory = True
        real_fsync(descriptor)

    monkeypatch.setattr(registry_module.os, "fsync", observing_fsync)
    AtomicRegistry(tmp_path).put("model_identity", identity)
    assert synced_directory is True


def test_atomic_registry_supports_each_atomic_projection(tmp_path: Path) -> None:
    examples = ROOT / "proto/model-runtime/v1/examples"
    execution = json.loads((examples / "execution.text.example.json").read_text())
    profile = {"chip": "Apple M4", "memory_gib": 32, "cpu_cores": 10}
    conditions = {
        "power_source": "ac",
        "low_power_mode": False,
        "thermal_state": "nominal",
        "memory_pressure": "normal",
    }
    machine = {
        "schema_version": 1,
        "profile_completeness": "partial",
        "profile": profile,
        "profile_digest": rcj_digest(profile),
        "os": {"name": "macOS", "version": "26.0", "architecture": "arm64"},
        "conditions_before": conditions,
        "conditions_after": conditions,
    }
    snapshot = build_legacy_catalog_snapshot()
    registry = AtomicRegistry(tmp_path)
    for kind, document in (
        ("machine_observation", machine),
        ("execution_config", execution),
        ("catalog_snapshot", snapshot),
    ):
        digest = registry.put(kind, document)
        assert registry.get(kind, digest) == document


def test_atomic_registry_rejects_unreferenced_policy_and_bad_addresses(
    tmp_path: Path,
) -> None:
    bundle = build_catalog_bundle()
    snapshot = copy.deepcopy(bundle["snapshot"])
    policy = bundle["recommendation_policies"][0]
    snapshot["recommendation_policy_digests"] = []
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
    registry = AtomicRegistry(tmp_path)
    with pytest.raises(CatalogValidationError, match="not referenced"):
        registry.put("recommendation_policy", policy, catalog_snapshot=snapshot)
    for digest in ("not-a-digest", "sha256:" + "G" * 64):
        with pytest.raises(ValueError, match="digest must be"):
            registry.get("model_identity", digest)


def test_contract_validator_rejects_unknown_contract() -> None:
    with pytest.raises(KeyError, match="unknown contract"):
        ContractValidator().validate("not_registered", {})


def _rehash_snapshot(snapshot: dict) -> None:
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["models"].append(copy.deepcopy(value["models"][0])),
            "duplicate registry_model_id",
        ),
        (
            lambda value: value["aliases"].append(copy.deepcopy(value["aliases"][0])),
            "duplicate alias",
        ),
        (
            lambda value: value["aliases"][0].update(
                default_execution_preset_id="missing",
                execution_presets=[
                    {
                        "preset_id": "balanced",
                        "execution_config_digest": "sha256:" + "a" * 64,
                        "evidence": {"status": "unverified_legacy"},
                    }
                ],
            ),
            "does not resolve to exactly one",
        ),
    ],
)
def test_catalog_snapshot_rejects_cross_record_duplicates_and_missing_defaults(
    mutation, message: str
) -> None:
    snapshot = build_legacy_catalog_snapshot()
    mutation(snapshot)
    _rehash_snapshot(snapshot)
    with pytest.raises(CatalogValidationError, match=message):
        ContractValidator().validate_catalog_snapshot(snapshot)


def test_catalog_snapshot_rejects_target_metadata_and_digest_drift() -> None:
    snapshot = build_legacy_catalog_snapshot()
    alias = snapshot["aliases"][0]
    alias["target"]["resolution_status"] = "resolved"
    alias["target"]["model_identity_digest"] = "sha256:" + "a" * 64
    _rehash_snapshot(snapshot)
    with pytest.raises(CatalogValidationError, match="does not match"):
        ContractValidator().validate_catalog_snapshot(snapshot)

    snapshot = build_legacy_catalog_snapshot()
    snapshot["catalog_digest"] = "sha256:" + "f" * 64
    with pytest.raises(CatalogValidationError, match="RCJ-1 projection"):
        ContractValidator().validate_catalog_snapshot(snapshot)


def test_recommendation_policy_rejects_duplicate_roles_missing_alias_and_digest() -> (
    None
):
    bundle = build_catalog_bundle()
    aliases = {item["alias"]: item for item in bundle["snapshot"]["aliases"]}
    validator = ContractValidator()

    duplicate = copy.deepcopy(bundle["recommendation_policies"][0])
    duplicate["tiers"][0]["picks"].append(
        copy.deepcopy(duplicate["tiers"][0]["picks"][0])
    )
    duplicate["policy_digest"] = rcj_digest(
        {key: value for key, value in duplicate.items() if key != "policy_digest"}
    )
    with pytest.raises(CatalogValidationError, match="roles must be unique"):
        validator.validate_recommendation_policy(duplicate, aliases=aliases)

    missing = copy.deepcopy(bundle["recommendation_policies"][0])
    missing["tiers"][0]["picks"][0]["alias"] = "missing-alias"
    missing["policy_digest"] = rcj_digest(
        {key: value for key, value in missing.items() if key != "policy_digest"}
    )
    with pytest.raises(CatalogValidationError, match="does not resolve in the catalog"):
        validator.validate_recommendation_policy(missing, aliases=aliases)

    drifted = copy.deepcopy(bundle["recommendation_policies"][0])
    drifted["policy_digest"] = "sha256:" + "f" * 64
    with pytest.raises(CatalogValidationError, match="RCJ-1 projection"):
        validator.validate_recommendation_policy(drifted, aliases=aliases)


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


def test_catalog_snapshot_rejects_task_without_compatible_operation() -> None:
    snapshot = build_legacy_catalog_snapshot()
    alias = snapshot["aliases"][0]
    alias["capabilities"]["task_types"] = ["text_generation"]
    alias["capabilities"]["operation_modes"] = ["text_to_image"]
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
    with pytest.raises(CatalogValidationError, match="no compatible operation"):
        ContractValidator().validate_catalog_snapshot(snapshot)


def test_catalog_snapshot_rejects_duplicate_execution_preset_ids() -> None:
    snapshot = build_legacy_catalog_snapshot()
    alias = snapshot["aliases"][0]
    preset = {
        "preset_id": "balanced",
        "execution_config_digest": "sha256:" + "a" * 64,
        "evidence": {"status": "unverified_legacy"},
    }
    alias["default_execution_preset_id"] = "balanced"
    alias["execution_presets"] = [
        preset,
        {**preset, "execution_config_digest": "sha256:" + "b" * 64},
    ]
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
    with pytest.raises(CatalogValidationError, match="duplicate preset_id"):
        ContractValidator().validate_catalog_snapshot(snapshot)
