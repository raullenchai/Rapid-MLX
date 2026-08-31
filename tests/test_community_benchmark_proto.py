# SPDX-License-Identifier: Apache-2.0
"""Contract tests for atomic model/runtime and community benchmark schemas."""

from __future__ import annotations

import copy
import hashlib
import json
import unicodedata
from pathlib import Path

import jsonschema
import pytest
import referencing

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTO_ROOT = REPO_ROOT / "proto"
RUNTIME_ROOT = PROTO_ROOT / "model-runtime" / "v1"
CATALOG_ROOT = PROTO_ROOT / "model-catalog" / "v1"
BENCH_ROOT = PROTO_ROOT / "community-benchmark" / "v1"

SCHEMA_PATHS = (
    RUNTIME_ROOT / "model-identity.schema.json",
    RUNTIME_ROOT / "machine-observation.schema.json",
    RUNTIME_ROOT / "execution-config.schema.json",
    CATALOG_ROOT / "model-alias.schema.json",
    CATALOG_ROOT / "model-registry-record.schema.json",
    CATALOG_ROOT / "recommendation-policy.schema.json",
    CATALOG_ROOT / "catalog-snapshot.schema.json",
    BENCH_ROOT / "benchmark-run.schema.json",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schemas() -> dict[str, dict]:
    return {path.name: _load(path) for path in SCHEMA_PATHS}


@pytest.fixture(scope="module")
def registry(schemas):
    resources = (
        (schema["$id"], referencing.Resource.from_contents(schema))
        for schema in schemas.values()
    )
    return referencing.Registry().with_resources(resources)


def _validator(schema: dict, registry):
    return jsonschema.Draft202012Validator(
        schema, registry=registry, format_checker=jsonschema.FormatChecker()
    )


def _reject_floats(value: object) -> None:
    if isinstance(value, float):
        raise TypeError("RCJ-1 forbids floating-point values")
    if isinstance(value, dict):
        for key, child in value.items():
            if not key.isascii():
                raise TypeError("RCJ-1 object keys must be ASCII")
            _reject_floats(child)
    elif isinstance(value, list):
        for child in value:
            _reject_floats(child)


def _normalize_nfc(value: object) -> object:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_normalize_nfc(child) for child in value]
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, child in value.items():
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise ValueError("RCJ-1 key collision after NFC normalization")
            normalized[normalized_key] = _normalize_nfc(child)
        return normalized
    return value


def _canonical(value: object) -> bytes:
    _reject_floats(value)
    return json.dumps(
        _normalize_nfc(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _digest(value: object) -> str:
    return f"sha256:{hashlib.sha256(_canonical(value)).hexdigest()}"


def test_all_schemas_are_valid_draft_2020_12(schemas) -> None:
    for schema in schemas.values():
        jsonschema.Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


@pytest.mark.parametrize("kind", ("llm", "vlm", "image", "video"))
def test_all_model_pipeline_examples_validate_and_match_digest(
    schemas, registry, kind
) -> None:
    example = _load(RUNTIME_ROOT / "examples" / f"model-identity.{kind}.example.json")
    _validator(schemas["model-identity.schema.json"], registry).validate(example)
    projection = {
        key: example[key] for key in ("schema_version", "pipeline_kind", "components")
    }
    assert _digest(projection) == example["identity_digest"]


@pytest.mark.parametrize("kind", ("text", "vlm", "image", "video"))
def test_all_execution_examples_validate_and_match_digest(
    schemas, registry, kind
) -> None:
    example = _load(RUNTIME_ROOT / "examples" / f"execution.{kind}.example.json")
    _validator(schemas["execution-config.schema.json"], registry).validate(example)
    projection = {key: example[key] for key in ("task_type", "resources", "task")}
    assert _digest(projection) == example["config_digest"]


def test_alias_is_a_reference_layer_not_embedded_identity(schemas, registry) -> None:
    example = _load(CATALOG_ROOT / "examples" / "model-alias.example.json")
    _validator(schemas["model-alias.schema.json"], registry).validate(example)
    assert "hf_path" not in json.dumps(example)
    assert "model_identity_digest" in example["target"]
    assert "execution_config_digest" in example["execution_presets"][0]


def test_promoted_alias_preset_requires_scoped_evidence(schemas, registry) -> None:
    example = _load(CATALOG_ROOT / "examples" / "model-alias.example.json")
    evidence = example["execution_presets"][0]["evidence"]
    evidence["status"] = "promoted"
    del evidence["machine_profile_digest"]
    errors = list(
        _validator(schemas["model-alias.schema.json"], registry).iter_errors(example)
    )
    assert any("machine_profile_digest" in error.message for error in errors)


def test_unresolved_alias_has_no_identity_digest_or_presets(schemas, registry) -> None:
    example = _load(CATALOG_ROOT / "examples" / "model-alias.example.json")
    example["target"]["resolution_status"] = "unresolved"
    del example["target"]["model_identity_digest"]
    example["default_execution_preset_id"] = None
    example["execution_presets"] = []
    _validator(schemas["model-alias.schema.json"], registry).validate(example)


@pytest.mark.parametrize(
    ("task_type", "pipeline_kind"),
    (
        ("speech_synthesis", "speech_synthesis"),
        ("speech_recognition", "speech_recognition"),
    ),
)
def test_audio_atomic_contracts_are_reachable(
    schemas, registry, task_type, pipeline_kind
) -> None:
    identity = _load(RUNTIME_ROOT / "examples" / "model-identity.llm.example.json")
    identity["pipeline_kind"] = pipeline_kind
    identity["identity_digest"] = _digest(
        {
            key: identity[key]
            for key in ("schema_version", "pipeline_kind", "components")
        }
    )
    _validator(schemas["model-identity.schema.json"], registry).validate(identity)

    execution = _load(RUNTIME_ROOT / "examples" / "execution.text.example.json")
    execution["task_type"] = task_type
    execution["task"] = {
        "kind": task_type,
        "audio": {"streaming": True, "batch_size": 1, "compute_backend": "gpu"},
    }
    execution["config_digest"] = _digest(
        {key: execution[key] for key in ("task_type", "resources", "task")}
    )
    _validator(schemas["execution-config.schema.json"], registry).validate(execution)


@pytest.mark.parametrize("kind", ("image", "video"))
def test_launch_modality_benchmark_examples_validate(schemas, registry, kind) -> None:
    example = _load(BENCH_ROOT / "examples" / f"benchmark-run.{kind}.example.json")
    _validator(schemas["benchmark-run.schema.json"], registry).validate(example)
    assert example["model"]["pipeline_kind"] == f"{kind}_generation"
    assert example["execution"]["task_type"] == f"{kind}_generation"
    assert example["workload"]["task_type"] == f"{kind}_generation"
    identity = _load(RUNTIME_ROOT / "examples" / f"model-identity.{kind}.example.json")
    assert example["model"]["components"] == identity["components"]
    assert example["model"]["identity_digest"] == identity["identity_digest"]


def test_benchmark_run_rejects_cross_object_task_mismatch(schemas, registry) -> None:
    example = _load(BENCH_ROOT / "examples" / "benchmark-run.image.example.json")
    example["execution"] = _load(
        RUNTIME_ROOT / "examples" / "execution.video.example.json"
    )
    errors = list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(example)
    )
    assert any(list(error.absolute_path)[:1] == ["execution"] for error in errors)


def test_vlm_workload_and_measurement_union_is_reachable(schemas, registry) -> None:
    base_id = schemas["benchmark-run.schema.json"]["$id"]
    workload = {
        "protocol_id": "custom-vlm",
        "protocol_version": 1,
        "protocol_strength": "custom",
        "protocol_digest": "sha256:" + "a" * 64,
        "task_type": "vision_language",
        "dataset": {"id": "public-vlm", "version": 1, "digest": "sha256:" + "b" * 64},
        "concurrency": 1,
        "cases": [
            {
                "case_id": "one-image",
                "warmup_rounds": 0,
                "measured_rounds": 1,
                "media_kind": "image",
                "media_count": 1,
                "width": 1024,
                "height": 1024,
                "target_output_tokens": 64,
            }
        ],
    }
    measurement = {
        "case_id": "one-image",
        "round_index": 1,
        "total_duration_ms": 1000.0,
        "peak_active_memory_mib": 4096,
        "completed": True,
        "prompt_tokens": 32,
        "output_tokens": 64,
        "ttft_ms": 400.0,
        "decode_duration_ms": 600.0,
        "media_encode_duration_ms": 250.0,
    }
    for name, value in (("workload", workload), ("measurement", measurement)):
        _validator({"$ref": f"{base_id}#/$defs/{name}"}, registry).validate(value)

    workload["cases"][0]["frames"] = 1
    assert list(
        _validator({"$ref": f"{base_id}#/$defs/workload"}, registry).iter_errors(
            workload
        )
    )
    del workload["cases"][0]["frames"]
    workload["cases"][0]["media_kind"] = "video"
    errors = list(
        _validator({"$ref": f"{base_id}#/$defs/workload"}, registry).iter_errors(
            workload
        )
    )
    assert errors
    workload["cases"][0]["frames"] = 24
    _validator({"$ref": f"{base_id}#/$defs/workload"}, registry).validate(workload)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("model", "local_path", "/Users/alice/model"),
        ("machine", "hostname", "alice-mac"),
        ("execution", "environment", {"TOKEN": "secret"}),
        ("workload", "prompt", "private prompt"),
    ),
)
def test_upload_is_a_strict_privacy_allowlist(
    schemas, registry, section, key, value
) -> None:
    example = _load(BENCH_ROOT / "examples" / "benchmark-run.image.example.json")
    example[section][key] = value
    errors = list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(example)
    )
    assert errors


def test_client_cannot_upload_server_verdicts(schemas, registry) -> None:
    example = _load(BENCH_ROOT / "examples" / "benchmark-run.image.example.json")
    example["validation"] = {"verified": True, "rank": 1}
    assert list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(example)
    )


def test_repository_identity_requires_all_component_revisions(
    schemas, registry
) -> None:
    example = _load(RUNTIME_ROOT / "examples" / "model-identity.vlm.example.json")
    del example["components"][1]["source"]["resolved_revision"]
    errors = list(
        _validator(schemas["model-identity.schema.json"], registry).iter_errors(example)
    )
    assert any("resolved_revision" in error.message for error in errors)


def test_unresolved_identity_has_no_digest_but_can_participate(
    schemas, registry
) -> None:
    example = _load(RUNTIME_ROOT / "examples" / "model-identity.llm.example.json")
    example["identity_strength"] = "unresolved"
    del example["identity_digest"]
    del example["components"][0]["source"]["resolved_revision"]
    _validator(schemas["model-identity.schema.json"], registry).validate(example)


def test_local_identity_rejects_repo_coordinates_and_requires_content_manifest(
    schemas, registry
) -> None:
    example = _load(RUNTIME_ROOT / "examples" / "model-identity.llm.example.json")
    example["identity_strength"] = "local_manifest"
    example["components"][0]["source"]["kind"] = "local"
    errors = list(
        _validator(schemas["model-identity.schema.json"], registry).iter_errors(example)
    )
    assert errors


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("display", {"name": "Alice's private finance model"}),
        ("family", {"id": "alice/private-model"}),
    ),
)
def test_atomic_identity_rejects_client_authored_labels(
    schemas, registry, field, value
) -> None:
    example = _load(RUNTIME_ROOT / "examples" / "model-identity.image.example.json")
    example[field] = value
    assert list(
        _validator(schemas["model-identity.schema.json"], registry).iter_errors(example)
    )


def test_component_change_changes_identity_digest() -> None:
    example = _load(RUNTIME_ROOT / "examples" / "model-identity.image.example.json")
    projection = {
        key: example[key] for key in ("schema_version", "pipeline_kind", "components")
    }
    before = _digest(projection)
    projection["components"][0]["source"]["resolved_revision"] = "f" * 40
    assert _digest(projection) != before


def test_unquantized_identity_rejects_quantization_only_fields(
    schemas, registry
) -> None:
    example = _load(RUNTIME_ROOT / "examples" / "model-identity.image.example.json")
    quantization = example["components"][1]["quantization"]
    assert quantization["kind"] == "none"
    quantization.update({"method": "bogus", "weight_bits_x2": 7, "group_size": 64})
    errors = list(
        _validator(schemas["model-identity.schema.json"], registry).iter_errors(example)
    )
    assert errors


def test_task_discriminator_rejects_cross_modality_execution(schemas, registry) -> None:
    example = _load(RUNTIME_ROOT / "examples" / "execution.image.example.json")
    example["task_type"] = "video_generation"
    assert list(
        _validator(schemas["execution-config.schema.json"], registry).iter_errors(
            example
        )
    )


def test_mtp_and_quantized_kv_require_reproducibility_fields(schemas, registry) -> None:
    example = _load(RUNTIME_ROOT / "examples" / "execution.text.example.json")
    del example["task"]["language"]["speculative_decoding"]["max_draft_tokens"]
    del example["task"]["language"]["kv_cache"]["bits_x2"]
    errors = list(
        _validator(schemas["execution-config.schema.json"], registry).iter_errors(
            example
        )
    )
    messages = " ".join(error.message for error in errors)
    assert "max_draft_tokens" in messages
    assert "bits_x2" in messages


def test_external_mtp_assistant_requires_identity_and_changes_config_digest(
    schemas, registry
) -> None:
    example = _load(RUNTIME_ROOT / "examples" / "execution.text.example.json")
    speculative = example["task"]["language"]["speculative_decoding"]
    speculative["assistant_source"] = "external_model"
    assert list(
        _validator(schemas["execution-config.schema.json"], registry).iter_errors(
            example
        )
    )

    speculative["draft_model_identity_digest"] = "sha256:" + "a" * 64
    _validator(schemas["execution-config.schema.json"], registry).validate(example)
    projection = {key: example[key] for key in ("task_type", "resources", "task")}
    before = _digest(projection)
    speculative["draft_model_identity_digest"] = "sha256:" + "b" * 64
    assert _digest(projection) != before


def test_embedded_mtp_rejects_external_assistant_identity(schemas, registry) -> None:
    example = _load(RUNTIME_ROOT / "examples" / "execution.text.example.json")
    example["task"]["language"]["speculative_decoding"][
        "draft_model_identity_digest"
    ] = "sha256:" + "a" * 64
    assert list(
        _validator(schemas["execution-config.schema.json"], registry).iter_errors(
            example
        )
    )


def test_kv_precision_fields_are_mutually_consistent(schemas, registry) -> None:
    example = _load(RUNTIME_ROOT / "examples" / "execution.text.example.json")
    cache = example["task"]["language"]["kv_cache"]
    cache["mode"] = "full_precision"
    cache["dtype"] = "float16"
    assert list(
        _validator(schemas["execution-config.schema.json"], registry).iter_errors(
            example
        )
    )

    cache.pop("bits_x2")
    cache.pop("group_size")
    _validator(schemas["execution-config.schema.json"], registry).validate(example)
    cache["mode"] = "quantized"
    assert list(
        _validator(schemas["execution-config.schema.json"], registry).iter_errors(
            example
        )
    )


def test_temporal_chunking_requires_size_only_when_enabled(schemas, registry) -> None:
    example = _load(RUNTIME_ROOT / "examples" / "execution.video.example.json")
    chunking = example["task"]["temporal_chunking"]
    del chunking["frames_per_chunk"]
    assert list(
        _validator(schemas["execution-config.schema.json"], registry).iter_errors(
            example
        )
    )
    chunking.update({"enabled": False, "frames_per_chunk": 16})
    assert list(
        _validator(schemas["execution-config.schema.json"], registry).iter_errors(
            example
        )
    )
    del chunking["frames_per_chunk"]
    _validator(schemas["execution-config.schema.json"], registry).validate(example)


def test_scaled_config_values_reject_floats(schemas, registry) -> None:
    run = _load(BENCH_ROOT / "examples" / "benchmark-run.image.example.json")
    run["workload"]["cases"][0]["guidance_millionths"] = 3.5
    assert list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(run)
    )


def test_digest_projected_values_reject_unsafe_integers(schemas, registry) -> None:
    unsafe = 9007199254740992
    execution = _load(RUNTIME_ROOT / "examples" / "execution.image.example.json")
    execution["resources"]["wired_memory_limit_mib"] = unsafe
    assert list(
        _validator(schemas["execution-config.schema.json"], registry).iter_errors(
            execution
        )
    )

    run = _load(BENCH_ROOT / "examples" / "benchmark-run.image.example.json")
    run["workload"]["protocol_version"] = unsafe
    assert list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(run)
    )


def test_failed_outcome_is_structured_without_measurements(schemas, registry) -> None:
    run = _load(BENCH_ROOT / "examples" / "benchmark-run.image.example.json")
    run["outcome"] = {"status": "failed", "failure_code": "model_load_oom"}
    del run["measurements"]
    _validator(schemas["benchmark-run.schema.json"], registry).validate(run)
    run["outcome"]["error_message"] = "/Users/alice/private-model failed"
    assert list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(run)
    )


def test_completed_outcome_rejects_incomplete_measurement(schemas, registry) -> None:
    run = _load(BENCH_ROOT / "examples" / "benchmark-run.image.example.json")
    run["measurements"][0]["completed"] = False
    errors = list(
        _validator(schemas["benchmark-run.schema.json"], registry).iter_errors(run)
    )
    assert any(list(error.absolute_path)[-1:] == ["completed"] for error in errors)


def test_registered_protocols_and_datasets_match_digests(schemas, registry) -> None:
    workload_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": schemas["benchmark-run.schema.json"]["$id"] + "#/$defs/workload",
    }
    for path in (BENCH_ROOT / "protocols").glob("*.json"):
        entry = _load(path)
        workload = entry["workload"]
        _validator(workload_schema, registry).validate(workload)
        projection = {k: v for k, v in workload.items() if k != "protocol_digest"}
        assert _digest(projection) == workload["protocol_digest"]
        assert entry["protocol_digest"] == workload["protocol_digest"]
    for path in (BENCH_ROOT / "datasets").glob("*.json"):
        entry = _load(path)
        projection = {k: v for k, v in entry.items() if k != "digest"}
        assert _digest(projection) == entry["digest"]


def test_public_media_dataset_contains_reproducible_inputs() -> None:
    dataset = _load(BENCH_ROOT / "datasets" / "rapid-public-prompts-v1.json")
    assert dataset["license"] == "CC0-1.0"
    assert {case["case_id"] for case in dataset["cases"]} == {
        "t2i-1024-square",
        "t2v-480p-81f",
    }
    assert all(case["prompt"].strip() for case in dataset["cases"])


def test_synthetic_token_dataset_has_cross_language_golden_vector() -> None:
    dataset = _load(BENCH_ROOT / "datasets" / "rapid-synthetic-token-dataset-v1.json")
    generator = dataset["generator"]
    assert generator["input_representation"] == "token_ids"
    state = generator["seed"]
    upper_exclusive = min(dataset["golden_vector"]["tokenizer_vocab_size"], 100000)
    actual = []
    for _ in dataset["golden_vector"]["first_token_ids"]:
        state = (state ^ ((state << 13) & 0xFFFFFFFF)) & 0xFFFFFFFF
        state = (state ^ (state >> 17)) & 0xFFFFFFFF
        state = (state ^ ((state << 5) & 0xFFFFFFFF)) & 0xFFFFFFFF
        actual.append(256 + state % (upper_exclusive - 256))
    assert actual == dataset["golden_vector"]["first_token_ids"]


def test_registered_launch_examples_exactly_match_protocol_registry() -> None:
    for kind in ("image", "video"):
        run = _load(BENCH_ROOT / "examples" / f"benchmark-run.{kind}.example.json")
        protocol = _load(BENCH_ROOT / "protocols" / f"rapid-{kind}-speed-v1.json")
        assert run["workload"] == protocol["workload"]


def test_machine_digest_is_profile_only() -> None:
    run = _load(BENCH_ROOT / "examples" / "benchmark-run.image.example.json")
    profile = copy.deepcopy(run["machine"]["profile"])
    run["machine"]["conditions_after"]["thermal_state"] = "serious"
    assert _digest(profile) == run["machine"]["profile_digest"]


def test_canonical_json_handles_unicode_and_rejects_float_exponents() -> None:
    value = {"path": "模型/权重.safetensors", "size_bytes": 123}
    assert (
        _canonical(value).decode()
        == '{"path":"模型/权重.safetensors","size_bytes":123}'
    )
    assert _canonical({"name": "é"}) == _canonical({"name": "e\u0301"})
    assert _digest({"name": "é"}) == _digest({"name": "e\u0301"})
    with pytest.raises(TypeError, match="forbids floating-point"):
        _canonical({"temperature": 0.7, "tiny": 1e-7})
