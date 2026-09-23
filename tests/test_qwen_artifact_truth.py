from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import rapid_mlx.runtime.qwen_artifact as qwen_artifact
from rapid_mlx.runtime.qwen_artifact import (
    ArtifactIdentityStatus,
    CandidateStorage,
    MTPFileShape,
    TargetWeightLayout,
    probe_qwen_artifact,
)
from rapid_mlx.spec_decode.mtp.qwen3_5_inject import _find_mtp_weights_file

FIXTURES = Path(__file__).parent / "fixtures" / "qwen_artifacts"

EXPECTED = {
    "qwen35_4b_4bit": {
        "outer": "qwen3_5",
        "text": "qwen3_5_text",
        "hidden": 2560,
        "layers": 32,
        "layer_counts": {"full_attention": 8, "linear_attention": 24},
        "target_layout": TargetWeightLayout.INDEXED_SAFETENSORS,
        "target_shards": 1,
        "quant_overrides": 0,
        "mtp_shape": MTPFileShape.ROOT_MODEL_AMBIGUOUS,
        "mtp_storage": CandidateStorage.REGULAR_FILE,
    },
    "qwen36_35b_4bit": {
        "outer": "qwen3_5_moe",
        "text": "qwen3_5_moe_text",
        "hidden": 2048,
        "layers": 40,
        "layer_counts": {"full_attention": 10, "linear_attention": 30},
        "target_layout": TargetWeightLayout.INDEXED_SAFETENSORS,
        "target_shards": 4,
        "quant_overrides": 80,
        "mtp_shape": MTPFileShape.NONE,
        "mtp_storage": CandidateStorage.NONE,
    },
    "qwen38_27b_4bit": {
        "outer": "qwen3_5",
        "text": "qwen3_5_text",
        "hidden": 5120,
        "layers": 64,
        "layer_counts": {"full_attention": 16, "linear_attention": 48},
        "target_layout": TargetWeightLayout.INDEXED_SAFETENSORS,
        "target_shards": 3,
        "quant_overrides": 0,
        "mtp_shape": MTPFileShape.NESTED_MODEL,
        "mtp_storage": CandidateStorage.SYMLINK,
    },
}


def _fixture(name: str) -> tuple[Path, dict]:
    root = FIXTURES / name
    return root, json.loads((root / "snapshot.json").read_text(encoding="utf-8"))


def _materialize_snapshot(
    tmp_path: Path, name: str, *, include_mtp: bool = True
) -> tuple[Path, dict]:
    fixture, metadata = _fixture(name)
    snapshot = tmp_path / name
    snapshot.mkdir()
    config_bytes = (fixture / "config.json").read_bytes()
    (snapshot / "config.json").write_bytes(config_bytes)

    shards = metadata["target_shards"]
    weight_map = {}
    for index, shard in enumerate(shards):
        (snapshot / shard).touch()
        weight_map[f"fixture.tensor.{index}"] = shard
    if metadata["target_layout"] == "indexed_safetensors":
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )

    candidate = metadata["mtp_candidate"]
    if include_mtp and candidate is not None:
        candidate_path = snapshot / candidate["path"]
        candidate_path.parent.mkdir(parents=True)
        if candidate["storage"] == "symlink":
            blob = tmp_path / f"{name}-mtp-blob"
            blob.touch()
            candidate_path.symlink_to(blob)
        else:  # pragma: no cover - fixtures currently exercise the HF symlink shape
            candidate_path.touch()
    return snapshot, metadata


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_exact_cached_qwen_config_and_snapshot_facts(tmp_path: Path, name: str):
    snapshot, metadata = _materialize_snapshot(tmp_path, name)
    expected = EXPECTED[name]

    # The fixtures preserve the complete parsed config.json objects from the
    # pinned cached revisions. A casual fixture simplification must not
    # silently redefine the canonical artifact truth preserved here.
    fixture_config = json.loads(
        (FIXTURES / name / "config.json").read_text(encoding="utf-8")
    )
    canonical_config = json.dumps(
        fixture_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert hashlib.sha256(canonical_config).hexdigest() == metadata["config_sha256"]

    truth = probe_qwen_artifact(
        snapshot,
        source_repo=metadata["source_repo"],
        revision=metadata["revision"],
        provenance_verified=True,
    )

    assert truth.identity_status is ArtifactIdentityStatus.RESOLVER_VERIFIED_IMMUTABLE
    assert truth.config_sha256 == metadata["config_sha256"]
    assert truth.outer_model_type == expected["outer"]
    assert truth.text_model_type == expected["text"]
    assert truth.mtp_num_hidden_layers == 1
    assert truth.geometry.hidden_size == expected["hidden"]
    assert truth.geometry.num_hidden_layers == expected["layers"]
    assert dict(truth.geometry.layer_type_counts) == expected["layer_counts"]
    assert truth.geometry.layer_types_sha256 is not None
    assert truth.quantization.bits == 4
    assert truth.quantization.group_size == 64
    assert truth.quantization.mode == "affine"
    assert truth.quantization.override_count == expected["quant_overrides"]
    assert truth.target_weights.layout is expected["target_layout"]
    assert truth.target_weights.shard_count == expected["target_shards"]
    assert truth.mtp_locator.shape is expected["mtp_shape"]
    assert truth.mtp_locator.storage is expected["mtp_storage"]

    status = truth.to_status_dict()
    assert status["identity_is_immutable"] is True
    assert str(tmp_path) not in json.dumps(status)


def test_qwen38_is_qwen35_text_not_qwen4_exp(tmp_path: Path):
    snapshot, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")

    truth = probe_qwen_artifact(
        snapshot,
        source_repo=metadata["source_repo"],
        revision=metadata["revision"],
        provenance_verified=True,
    )

    assert (truth.outer_model_type, truth.text_model_type) == (
        "qwen3_5",
        "qwen3_5_text",
    )
    assert "qwen4_exp" not in {
        truth.outer_model_type,
        truth.text_model_type,
    }


def test_qwen38_nested_sidecar_symlink_is_accepted_without_reading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    snapshot, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")
    original_read_bytes = Path.read_bytes

    def _reject_weight_reads(path: Path) -> bytes:
        if path.suffix == ".safetensors":
            raise AssertionError("artifact probe must not read weight content")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", _reject_weight_reads)
    truth = probe_qwen_artifact(
        snapshot,
        source_repo=metadata["source_repo"],
        revision=metadata["revision"],
        provenance_verified=True,
    )

    assert _find_mtp_weights_file(snapshot) == snapshot / "mtp/model.safetensors"
    assert truth.mtp_locator.accepted is True
    assert truth.mtp_locator.relative_path == "mtp/model.safetensors"
    assert truth.mtp_locator.storage is CandidateStorage.SYMLINK


def test_root_sharded_target_layout_alone_is_not_an_mtp_sidecar(tmp_path: Path):
    snapshot, metadata = _materialize_snapshot(
        tmp_path, "qwen38_27b_4bit", include_mtp=False
    )

    assert _find_mtp_weights_file(snapshot) is None
    truth = probe_qwen_artifact(
        snapshot,
        source_repo=metadata["source_repo"],
        revision=metadata["revision"],
        provenance_verified=True,
    )
    assert truth.target_weights.layout is TargetWeightLayout.INDEXED_SAFETENSORS
    assert truth.target_weights.shard_count == 3
    assert truth.mtp_locator.accepted is False
    assert truth.mtp_locator.shape is MTPFileShape.NONE


@pytest.mark.parametrize(
    ("source_repo", "revision", "verified", "expected"),
    [
        (None, None, False, ArtifactIdentityStatus.MISSING_SOURCE),
        (None, "a" * 40, True, ArtifactIdentityStatus.MISSING_SOURCE),
        ("org/model", None, True, ArtifactIdentityStatus.MISSING_REVISION),
        ("org/model", "main", True, ArtifactIdentityStatus.MUTABLE_REVISION),
        ("org/model", "a" * 39, True, ArtifactIdentityStatus.MUTABLE_REVISION),
        (
            "org/model",
            "a" * 40,
            False,
            ArtifactIdentityStatus.DECLARED_IMMUTABLE_UNVERIFIED,
        ),
        (
            "org/model",
            "a" * 40,
            True,
            ArtifactIdentityStatus.RESOLVER_VERIFIED_IMMUTABLE,
        ),
    ],
)
def test_artifact_identity_fails_closed_without_verified_revision(
    tmp_path: Path,
    source_repo: str | None,
    revision: str | None,
    verified: bool,
    expected: ArtifactIdentityStatus,
):
    snapshot, _metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")

    truth = probe_qwen_artifact(
        snapshot,
        source_repo=source_repo,
        revision=revision,
        provenance_verified=verified,
    )

    assert truth.identity_status is expected
    assert truth.to_status_dict()["identity_is_immutable"] is (
        expected is ArtifactIdentityStatus.RESOLVER_VERIFIED_IMMUTABLE
    )


def test_spoofed_revision_string_is_not_verified_by_syntax(tmp_path: Path):
    snapshot, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")

    truth = probe_qwen_artifact(
        snapshot,
        source_repo=metadata["source_repo"],
        revision="0" * 40,
    )

    assert truth.identity_status is ArtifactIdentityStatus.DECLARED_IMMUTABLE_UNVERIFIED
    assert truth.to_status_dict()["identity_is_immutable"] is False


@pytest.mark.parametrize(
    ("shard_name", "expected_layout"),
    [
        ("model-00002-of-00002.safetensors", TargetWeightLayout.INCOMPLETE_INDEX),
        ("../outside.safetensors", TargetWeightLayout.INVALID_INDEX),
        ("/tmp/outside.safetensors", TargetWeightLayout.INVALID_INDEX),
    ],
)
def test_target_index_fails_closed_on_missing_or_escaping_shard(
    tmp_path: Path, shard_name: str, expected_layout: TargetWeightLayout
):
    snapshot, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["fixture.missing"] = shard_name
    index_path.write_text(json.dumps(index), encoding="utf-8")

    truth = probe_qwen_artifact(
        snapshot,
        source_repo=metadata["source_repo"],
        revision=metadata["revision"],
        provenance_verified=True,
    )

    assert truth.target_weights.layout is expected_layout
    if expected_layout is TargetWeightLayout.INCOMPLETE_INDEX:
        assert truth.target_weights.missing_shard_count == 1


def test_future_unknown_mtp_locator_path_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    snapshot, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    unknown = snapshot / "future" / "head.safetensors"
    unknown.parent.mkdir()
    unknown.touch()
    monkeypatch.setattr(qwen_artifact, "_find_mtp_weights_file", lambda _path: unknown)

    truth = probe_qwen_artifact(
        snapshot,
        source_repo=metadata["source_repo"],
        revision=metadata["revision"],
        provenance_verified=True,
    )

    assert truth.mtp_locator.accepted is False
    assert truth.mtp_locator.shape is MTPFileShape.UNSUPPORTED
    assert truth.mtp_locator.relative_path is None


def test_mtp_locator_candidate_outside_snapshot_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    snapshot, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    outside = tmp_path / "outside.safetensors"
    outside.touch()
    monkeypatch.setattr(qwen_artifact, "_find_mtp_weights_file", lambda _path: outside)

    truth = probe_qwen_artifact(
        snapshot,
        source_repo=metadata["source_repo"],
        revision=metadata["revision"],
        provenance_verified=True,
    )

    assert truth.mtp_locator.accepted is False
    assert truth.mtp_locator.shape is MTPFileShape.UNSUPPORTED
