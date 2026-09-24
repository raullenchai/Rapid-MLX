from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import ModuleType

import pytest

import rapid_mlx.qwen_artifact_layout as qwen_layout
import rapid_mlx.qwen_runtime_plan as qwen_plan
import rapid_mlx.runtime.qwen_artifact as qwen_artifact
from rapid_mlx.qwen_artifact_layout import (
    MTPWeightPathState,
    MTPWeightStorage,
    find_mtp_weights_file,
    inspect_mtp_weights_layout,
)
from rapid_mlx.runtime.qwen_artifact import (
    ArtifactIdentityStatus,
    ArtifactProbeError,
    QwenArtifactTruth,
    TargetWeightLayout,
    VerifiedHubSnapshotBinding,
    probe_qwen_artifact,
    probe_resolved_qwen_artifact,
    to_runtime_drafter_identity,
    to_verified_runtime_target,
    verify_hub_snapshot_binding,
)
from rapid_mlx.spec_decode.mtp.qwen3_5_inject import _find_mtp_weights_file
from scripts.extract_qwen_artifact_receipt import main as extract_receipt

FIXTURES = Path(__file__).parent / "fixtures" / "qwen_artifacts"
CONFIGURED_HUB_CACHE_ROOT = qwen_artifact._configured_hub_cache_root

EXPECTED = {
    "qwen35_4b_4bit": {
        "outer": "qwen3_5",
        "text": "qwen3_5_text",
        "hidden": 2560,
        "layers": 32,
        "layer_counts": {"full_attention": 8, "linear_attention": 24},
        "quant_overrides": 0,
        "target_layout": TargetWeightLayout.INDEXED_SAFETENSORS,
        "mtp_state": MTPWeightPathState.ROOT_MODEL_AMBIGUOUS,
    },
    "qwen36_35b_4bit": {
        "outer": "qwen3_5_moe",
        "text": "qwen3_5_moe_text",
        "hidden": 2048,
        "layers": 40,
        "layer_counts": {"full_attention": 10, "linear_attention": 30},
        "quant_overrides": 80,
        "target_layout": TargetWeightLayout.INDEXED_SAFETENSORS,
        "mtp_state": MTPWeightPathState.NOT_FOUND,
    },
    "qwen38_27b_4bit": {
        "outer": "qwen3_5",
        "text": "qwen3_5_text",
        "hidden": 5120,
        "layers": 64,
        "layer_counts": {"full_attention": 16, "linear_attention": 48},
        "quant_overrides": 0,
        "target_layout": TargetWeightLayout.INDEXED_SAFETENSORS,
        "mtp_state": MTPWeightPathState.NESTED_MODEL,
    },
}


@pytest.fixture(autouse=True)
def _configured_test_hub_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = tmp_path / "hub"
    hub.mkdir(exist_ok=True)
    monkeypatch.setattr(
        qwen_artifact, "_configured_hub_cache_root", lambda: hub.resolve()
    )


def _fixture(name: str) -> tuple[Path, dict]:
    root = FIXTURES / name
    return root, json.loads((root / "snapshot.json").read_text(encoding="utf-8"))


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _materialize_snapshot(
    tmp_path: Path,
    name: str,
    *,
    include_mtp: bool = True,
    subfolder: str | None = None,
) -> tuple[Path, Path, dict]:
    fixture, metadata = _fixture(name)
    hub = tmp_path / "hub"
    repo_cache = hub / ("models--" + metadata["source_repo"].replace("/", "--"))
    snapshot = repo_cache / "snapshots" / metadata["revision"]
    if subfolder is not None:
        snapshot = snapshot / subfolder
    snapshot.mkdir(parents=True)
    blobs = repo_cache / "blobs"
    blobs.mkdir()

    def _materialize_metadata(name: str) -> None:
        content = (fixture / name).read_bytes()
        blob = blobs / hashlib.sha256(content).hexdigest()
        blob.write_bytes(content)
        leaf = snapshot / name
        leaf.symlink_to(os.path.relpath(blob, leaf.parent))

    _materialize_metadata("config.json")
    if metadata["target_layout"] == "indexed_safetensors":
        _materialize_metadata("model.safetensors.index.json")

    for shard in metadata["target_shards"]:
        target = snapshot / shard
        target.parent.mkdir(parents=True, exist_ok=True)
        blob = repo_cache / "blobs" / metadata["target_blob_ids"][shard]
        blob.touch()
        target.symlink_to(os.path.relpath(blob, target.parent))

    candidate = metadata["mtp_candidate"]
    if include_mtp and candidate is not None:
        candidate_path = snapshot / candidate["path"]
        candidate_path.parent.mkdir(parents=True)
        blob = repo_cache / "blobs" / candidate["blob_id"]
        blob.touch()
        os.truncate(blob, candidate["file_size_bytes"])
        relative_blob = os.path.relpath(blob, candidate_path.parent)
        candidate_path.symlink_to(relative_blob)
        candidate_path.with_name(candidate_path.name + ".sha256").write_text(
            f"{candidate['declared_sha256']}  {candidate['declared_name']}\n",
            encoding="utf-8",
        )
    return snapshot, hub, metadata


def _binding(snapshot: Path, metadata: dict, *, subfolder: str | None = None):
    binding = verify_hub_snapshot_binding(
        snapshot,
        repo_id=metadata["source_repo"],
        revision=metadata["revision"],
        subfolder=subfolder,
    )
    assert binding is not None
    return binding


def _replace_with_symlink(path: Path, target: Path) -> None:
    path.unlink()
    path.symlink_to(os.path.relpath(target, path.parent))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_exact_cached_qwen_config_index_and_snapshot_facts(tmp_path: Path, name: str):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, name)
    expected = EXPECTED[name]
    fixture = FIXTURES / name
    config = json.loads((fixture / "config.json").read_text(encoding="utf-8"))
    assert _canonical_digest(config) == metadata["config_sha256"]
    if metadata["target_layout"] == "indexed_safetensors":
        index = json.loads(
            (fixture / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        assert _canonical_digest(index) == metadata["index_sha256"]
        assert tuple(sorted(set(index["weight_map"].values()))) == tuple(
            metadata["target_shards"]
        )
    else:
        assert metadata["target_layout"] == "single_safetensors"
        assert "index_sha256" not in metadata
        assert not (fixture / "model.safetensors.index.json").exists()

    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))

    assert truth.identity_status is ArtifactIdentityStatus.VERIFIED_HUB_SNAPSHOT
    assert truth.source_repo == metadata["source_repo"]
    assert truth.revision == metadata["revision"]
    assert truth.verification_id.startswith("hf-snapshot-sha256:")
    assert truth.config_sha256 == metadata["config_sha256"]
    assert truth.outer_model_type == expected["outer"]
    assert truth.text_model_type == expected["text"]
    assert truth.mtp_num_hidden_layers == 1
    assert truth.geometry.hidden_size == expected["hidden"]
    assert truth.geometry.num_hidden_layers == expected["layers"]
    assert len(truth.geometry.layer_types) == expected["layers"]
    assert dict(truth.geometry.layer_type_counts) == expected["layer_counts"]
    assert truth.quantization.override_count == expected["quant_overrides"]
    assert truth.target_weights.layout is expected["target_layout"]
    assert truth.target_weights.shards == tuple(metadata["target_shards"])
    assert truth.target_weights.index_sha256 == metadata.get("index_sha256")
    assert dict(truth.target_weights.file_identities) == {
        shard: f"hf_blob:{blob_id}"
        for shard, blob_id in metadata["target_blob_ids"].items()
    }
    assert truth.mtp_locator.state is expected["mtp_state"]
    status = truth.to_status_dict()
    assert status["identity_is_immutable"] is True
    assert str(tmp_path) not in json.dumps(status)


@pytest.mark.parametrize("blob_length", [40, 64])
def test_verified_metadata_uses_canonical_direct_repo_blob_symlinks(
    tmp_path: Path, blob_length: int
):
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    repo_cache = snapshot.parents[1]
    blobs = repo_cache / "blobs"
    assert blobs.is_dir()
    assert not blobs.is_symlink()
    for blob_character, name in zip(
        ("a", "b"),
        ("config.json", "model.safetensors.index.json"),
        strict=True,
    ):
        leaf = snapshot / name
        replacement = blobs / (blob_character * blob_length)
        replacement.write_bytes(leaf.resolve(strict=True).read_bytes())
        _replace_with_symlink(leaf, replacement)
        resolved = leaf.resolve(strict=True)
        assert leaf.is_symlink()
        assert resolved.parent == blobs
        assert len(resolved.name) == blob_length
        assert resolved.is_file()
        assert not resolved.is_symlink()

    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    assert truth.identity_status is ArtifactIdentityStatus.VERIFIED_HUB_SNAPSHOT
    assert to_verified_runtime_target(truth) is not None


@pytest.mark.parametrize(
    "metadata_name", ["config.json", "model.safetensors.index.json"]
)
@pytest.mark.parametrize(
    "replacement",
    [
        "regular",
        "broken",
        "external",
        "repo_sibling",
        "sibling_revision",
        "other_repo",
        "nested_blob",
        "symlinked_blob_dir",
    ],
)
def test_verified_metadata_noncanonical_provenance_fails_closed(
    tmp_path: Path, metadata_name: str, replacement: str
) -> None:
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    repo_cache = snapshot.parents[1]
    leaf = snapshot / metadata_name
    original = leaf.resolve(strict=True).read_bytes()

    if replacement == "symlinked_blob_dir":
        blobs = repo_cache / "blobs"
        external_blobs = tmp_path / "external-blobs"
        external_blobs.mkdir()
        for blob in blobs.iterdir():
            blob.rename(external_blobs / blob.name)
        blobs.rmdir()
        blobs.symlink_to(external_blobs, target_is_directory=True)
    else:
        leaf.unlink()
        if replacement == "regular":
            leaf.write_bytes(original)
        else:
            blob_id = "e" * 64
            if replacement == "broken":
                target = tmp_path / "missing" / blob_id
            elif replacement == "external":
                target = tmp_path / "external" / blob_id
            elif replacement == "repo_sibling":
                target = repo_cache / "sibling" / blob_id
            elif replacement == "sibling_revision":
                target = repo_cache / "snapshots" / ("f" * 40) / blob_id
            elif replacement == "other_repo":
                target = hub / "models--other--repo" / "blobs" / blob_id
            else:
                target = repo_cache / "blobs" / "nested" / blob_id
            if replacement != "broken":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original)
            leaf.symlink_to(os.path.relpath(target, leaf.parent))

    with pytest.raises(ArtifactProbeError, match="canonical same-repo blob"):
        probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))


def test_qwen38_is_qwen35_text_not_qwen4_exp(tmp_path: Path):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")
    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    assert (truth.outer_model_type, truth.text_model_type) == (
        "qwen3_5",
        "qwen3_5_text",
    )
    assert "qwen4_exp" not in {truth.outer_model_type, truth.text_model_type}


def test_qwen35_pinned_manifest_requires_real_single_shard_index_receipt(
    tmp_path: Path,
):
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen35_4b_4bit")
    assert metadata["index_sha256"] == (
        "ec2c1084ed0e9f71599ff497f5f07489bddaf8774f6346ee2117e5ce71ff3ca8"
    )
    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    assert truth.target_weights.layout is TargetWeightLayout.INDEXED_SAFETENSORS
    assert truth.target_weights.shards == ("model.safetensors",)
    assert truth.target_weights.index_sha256 == metadata["index_sha256"]
    assert dict(truth.target_weights.file_identities) == {
        "model.safetensors": f"hf_blob:{metadata['target_blob_ids']['model.safetensors']}"
    }
    assert truth.mtp_locator.state is MTPWeightPathState.ROOT_MODEL_AMBIGUOUS


def test_qwen38_nested_sidecar_has_symlink_blob_receipt_without_weight_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")
    original_read_bytes = Path.read_bytes

    def _reject_weight_reads(path: Path) -> bytes:
        if path.suffix == ".safetensors":
            raise AssertionError("artifact probe must not read weight content")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", _reject_weight_reads)
    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    candidate = metadata["mtp_candidate"]
    assert truth.mtp_locator.state is MTPWeightPathState.NESTED_MODEL
    assert truth.mtp_locator.storage is MTPWeightStorage.SYMLINK
    assert truth.mtp_locator.relative_path == "mtp/model.safetensors"
    assert truth.mtp_locator.hf_blob_id == candidate["blob_id"]
    assert truth.mtp_locator.content_identity == f"hf_blob:{candidate['blob_id']}"
    assert truth.mtp_locator.declared_sha256 == candidate["declared_sha256"]
    assert truth.mtp_locator.declared_sha256_matches_blob is True
    assert truth.mtp_locator.file_size_bytes == candidate["file_size_bytes"]
    drafter = to_runtime_drafter_identity(truth)
    assert isinstance(drafter, qwen_plan.QwenDrafterIdentity)
    assert drafter.repo == metadata["source_repo"]
    assert drafter.revision == metadata["revision"]
    assert drafter.artifact_path == "mtp/model.safetensors"
    assert drafter.artifact_verification_id == truth.mtp_locator.content_identity


@pytest.mark.parametrize("escape_kind", ["outside", "sibling_revision"])
def test_sidecar_symlinked_parent_escape_has_no_trusted_content_identity(
    tmp_path: Path, escape_kind: str
):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")
    repo_cache = hub / ("models--" + metadata["source_repo"].replace("/", "--"))
    mtp_dir = snapshot / "mtp"
    (mtp_dir / "model.safetensors").unlink()
    (mtp_dir / "model.safetensors.sha256").unlink()
    mtp_dir.rmdir()
    if escape_kind == "outside":
        escaped_dir = tmp_path / "escaped-sidecar"
    else:
        escaped_dir = repo_cache / "snapshots" / ("f" * 40) / "mtp"
    escaped_dir.mkdir(parents=True)
    blob = repo_cache / "blobs" / metadata["mtp_candidate"]["blob_id"]
    escaped_candidate = escaped_dir / "model.safetensors"
    escaped_candidate.symlink_to(os.path.relpath(blob, escaped_dir))
    mtp_dir.symlink_to(os.path.relpath(escaped_dir, snapshot), target_is_directory=True)

    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    assert truth.mtp_locator.state is MTPWeightPathState.NESTED_MODEL
    assert truth.mtp_locator.hf_blob_id is None
    assert truth.mtp_locator.content_identity is None
    with pytest.raises(ArtifactProbeError, match="trusted sidecar content identity"):
        to_runtime_drafter_identity(truth)


def test_sidecar_symlinked_repo_blobs_dir_has_no_trusted_content_identity(
    tmp_path: Path,
):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")
    repo_cache = hub / ("models--" + metadata["source_repo"].replace("/", "--"))
    blobs_dir = repo_cache / "blobs"
    external_blobs = tmp_path / "external-sidecar-blobs"
    external_blobs.mkdir()
    for blob in blobs_dir.iterdir():
        blob.rename(external_blobs / blob.name)
    blobs_dir.rmdir()
    blobs_dir.symlink_to(external_blobs, target_is_directory=True)

    with pytest.raises(ArtifactProbeError, match="canonical same-repo blob"):
        probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))


def test_repointing_sidecar_to_another_direct_repo_blob_changes_identity(
    tmp_path: Path,
):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")
    binding = _binding(snapshot, metadata)
    before = probe_qwen_artifact(snapshot, binding=binding)
    repo_cache = hub / ("models--" + metadata["source_repo"].replace("/", "--"))
    replacement_blob = repo_cache / "blobs" / ("c" * 64)
    replacement_blob.touch()
    _replace_with_symlink(snapshot / "mtp" / "model.safetensors", replacement_blob)

    after = probe_qwen_artifact(snapshot, binding=binding)
    assert before.mtp_locator.content_identity != after.mtp_locator.content_identity
    drafter = to_runtime_drafter_identity(after)
    assert drafter.artifact_verification_id == f"hf_blob:{'c' * 64}"


def test_regular_sidecar_and_declared_sha_cannot_mint_drafter_identity(
    tmp_path: Path,
):
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    sidecar = snapshot / "mtp.safetensors"
    sidecar.write_bytes(b"untrusted regular sidecar")
    declared = "d" * 64
    sidecar.with_name("mtp.safetensors.sha256").write_text(
        f"{declared}  mtp.safetensors\n", encoding="utf-8"
    )

    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    assert truth.mtp_locator.state is MTPWeightPathState.ROOT_MTP
    assert truth.mtp_locator.declared_sha256 == declared
    assert truth.mtp_locator.hf_blob_id is None
    assert truth.mtp_locator.content_identity is None
    with pytest.raises(ArtifactProbeError, match="trusted sidecar content identity"):
        to_runtime_drafter_identity(truth)


def test_drafter_conversion_requires_verified_capability_and_repo_relative_path(
    tmp_path: Path,
):
    subfolder = "weights/4bit"
    snapshot, _hub, metadata = _materialize_snapshot(
        tmp_path, "qwen38_27b_4bit", subfolder=subfolder
    )
    with pytest.raises(
        ArtifactProbeError, match="resolver-verified artifact capability"
    ):
        to_runtime_drafter_identity(probe_qwen_artifact(snapshot))

    truth = probe_qwen_artifact(
        snapshot,
        binding=_binding(snapshot, metadata, subfolder=subfolder),
    )
    drafter = to_runtime_drafter_identity(truth)
    assert drafter.repo == metadata["source_repo"]
    assert drafter.revision == metadata["revision"]
    assert drafter.artifact_path == "weights/4bit/mtp/model.safetensors"
    assert drafter.artifact_verification_id == (
        f"hf_blob:{metadata['mtp_candidate']['blob_id']}"
    )


def test_root_model_is_categorically_ambiguous_not_head_receipt(tmp_path: Path):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen35_4b_4bit")
    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    assert _find_mtp_weights_file(snapshot) == snapshot / "model.safetensors"
    assert truth.mtp_locator.state is MTPWeightPathState.ROOT_MODEL_AMBIGUOUS
    assert truth.mtp_locator.hf_blob_id is None
    assert truth.mtp_locator.content_identity is None
    assert truth.mtp_locator.declared_sha256 is None
    assert "accepted" not in truth.mtp_locator.to_status_dict()
    assert "eligible" not in truth.mtp_locator.to_status_dict()


def test_root_sharded_target_layout_alone_is_not_a_locator_candidate(tmp_path: Path):
    snapshot, hub, metadata = _materialize_snapshot(
        tmp_path, "qwen38_27b_4bit", include_mtp=False
    )
    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    assert find_mtp_weights_file(snapshot) is None
    assert truth.mtp_locator.state is MTPWeightPathState.NOT_FOUND


@pytest.mark.parametrize(
    "repo_id",
    [
        "https://huggingface.co/org/model",
        "user@example.com/model",
        "/local/model",
        "file://local/model",
        "org/model?token=secret",
        "org/model#fragment",
        "org/repo/extra",
        "org--alias/model",
        "org..alias/model",
    ],
)
def test_noncanonical_sources_cannot_be_bound_or_echoed(tmp_path: Path, repo_id: str):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    assert (
        verify_hub_snapshot_binding(
            snapshot,
            repo_id=repo_id,
            revision=metadata["revision"],
        )
        is None
    )
    status = probe_qwen_artifact(snapshot).to_status_dict()
    assert status["source_repo"] is None
    assert status["revision"] is None
    assert repo_id not in json.dumps(status)


def test_arbitrary_tmp_dir_and_spoofed_revision_stay_unverified(tmp_path: Path):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    arbitrary = tmp_path / "arbitrary"
    arbitrary.mkdir()
    assert (
        verify_hub_snapshot_binding(
            arbitrary,
            repo_id=metadata["source_repo"],
            revision=metadata["revision"],
        )
        is None
    )
    assert (
        verify_hub_snapshot_binding(
            snapshot,
            repo_id=metadata["source_repo"],
            revision="0" * 40,
        )
        is None
    )
    assert (
        probe_qwen_artifact(snapshot).identity_status
        is ArtifactIdentityStatus.UNVERIFIED_SNAPSHOT
    )


def test_canonical_hub_subfolder_is_bound_and_reported(tmp_path: Path):
    subfolder = "weights/mlx"
    snapshot, _hub, metadata = _materialize_snapshot(
        tmp_path, "qwen36_35b_4bit", subfolder=subfolder
    )
    binding = _binding(snapshot, metadata, subfolder=subfolder)
    truth = probe_qwen_artifact(snapshot, binding=binding)
    assert truth.identity_status is ArtifactIdentityStatus.VERIFIED_HUB_SNAPSHOT
    assert truth.target_subfolder == subfolder
    assert truth.to_status_dict()["target_subfolder"] == subfolder


@pytest.mark.parametrize(
    "subfolder",
    [".", "../escape", "/absolute", "weights//mlx", "weights/./mlx", "weights\\mlx"],
)
def test_noncanonical_hub_subfolder_fails_closed(tmp_path: Path, subfolder: str):
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    assert (
        verify_hub_snapshot_binding(
            snapshot,
            repo_id=metadata["source_repo"],
            revision=metadata["revision"],
            subfolder=subfolder,
        )
        is None
    )


@pytest.mark.parametrize("escape_kind", ["blobs", "sibling_revision"])
def test_hub_subfolder_symlink_escape_fails_closed(tmp_path: Path, escape_kind: str):
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    repo_cache = snapshot.parents[1]
    if escape_kind == "blobs":
        target = repo_cache / "blobs"
    else:
        target = snapshot.parent / ("f" * 40)
        target.mkdir()
    escaped = snapshot / "escaped"
    escaped.symlink_to(os.path.relpath(target, snapshot), target_is_directory=True)
    assert (
        verify_hub_snapshot_binding(
            escaped,
            repo_id=metadata["source_repo"],
            revision=metadata["revision"],
            subfolder="escaped",
        )
        is None
    )


def test_revision_root_symlink_escape_fails_closed(tmp_path: Path):
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    escaped_root = tmp_path / "escaped-revision"
    snapshot.rename(escaped_root)
    snapshot.symlink_to(escaped_root, target_is_directory=True)
    assert (
        verify_hub_snapshot_binding(
            snapshot,
            repo_id=metadata["source_repo"],
            revision=metadata["revision"],
        )
        is None
    )


def test_revision_root_symlink_to_sibling_revision_fails_closed(tmp_path: Path):
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    sibling = snapshot.parent / ("f" * 40)
    snapshot.rename(sibling)
    snapshot.symlink_to(sibling.name, target_is_directory=True)
    assert (
        verify_hub_snapshot_binding(
            snapshot,
            repo_id=metadata["source_repo"],
            revision=metadata["revision"],
        )
        is None
    )


def test_symlinked_repo_cache_ancestor_is_allowed(tmp_path: Path):
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    repo_cache = snapshot.parents[1]
    resolved_repo_cache = tmp_path / "resolved-repo-cache"
    repo_cache.rename(resolved_repo_cache)
    repo_cache.symlink_to(resolved_repo_cache, target_is_directory=True)
    binding = _binding(snapshot, metadata)
    assert binding.matches(snapshot)
    truth = probe_qwen_artifact(snapshot, binding=binding)
    assert to_verified_runtime_target(truth) is not None


def test_verified_binding_constructor_is_not_a_public_trust_bit(tmp_path: Path):
    with pytest.raises(TypeError, match="verify_hub_snapshot_binding"):
        VerifiedHubSnapshotBinding(  # type: ignore[call-arg]
            repo_id="org/model",
            revision="a" * 40,
            subfolder=None,
            artifact_dir=tmp_path,
            repo_cache_dir=tmp_path,
        )


def test_binding_for_one_snapshot_cannot_label_another(tmp_path: Path):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    binding = _binding(snapshot, metadata)
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text('{"model_type":"qwen3_5"}')
    truth = probe_qwen_artifact(other, binding=binding)
    assert truth.identity_status is ArtifactIdentityStatus.BINDING_MISMATCH
    assert truth.source_repo is None
    assert truth.revision is None
    assert truth.verification_id is None


def test_artifact_truth_direct_construction_and_replace_cannot_forge_capability(
    tmp_path: Path,
):
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    verified = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    unverified = probe_qwen_artifact(snapshot)
    public_fields = {
        item.name: getattr(verified, item.name)
        for item in fields(QwenArtifactTruth)
        if not item.name.startswith("_")
    }
    with pytest.raises(TypeError, match="probe_qwen_artifact"):
        QwenArtifactTruth(**public_fields)
    with pytest.raises(TypeError, match="probe_qwen_artifact"):
        replace(verified, source_repo="attacker/forged")
    with pytest.raises(TypeError, match="probe_qwen_artifact"):
        replace(
            unverified,
            identity_status=ArtifactIdentityStatus.VERIFIED_HUB_SNAPSHOT,
            source_repo=metadata["source_repo"],
            revision=metadata["revision"],
            verification_id="forged",
        )


@pytest.mark.parametrize(
    ("shard_name", "expected_layout"),
    [
        ("model-99999-of-99999.safetensors", TargetWeightLayout.INCOMPLETE_INDEX),
        ("../outside.safetensors", TargetWeightLayout.INVALID_INDEX),
        ("/tmp/outside.safetensors", TargetWeightLayout.INVALID_INDEX),
    ],
)
def test_target_index_fails_closed_on_missing_or_escaping_shard(
    tmp_path: Path, shard_name: str, expected_layout: TargetWeightLayout
):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["fixture.missing"] = shard_name
    index_path.write_text(json.dumps(index), encoding="utf-8")
    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    assert truth.target_weights.layout is expected_layout
    assert truth.target_weights.index_sha256 == _canonical_digest(index)


@pytest.mark.parametrize(
    ("name", "expected_layout"),
    [
        ("qwen36_35b_4bit", TargetWeightLayout.INDEXED_SAFETENSORS),
        ("qwen35_4b_4bit", TargetWeightLayout.SINGLE_SAFETENSORS),
    ],
)
def test_verified_repo_blob_weight_symlinks_remain_valid(
    tmp_path: Path, name: str, expected_layout: TargetWeightLayout
):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, name)
    if expected_layout is TargetWeightLayout.SINGLE_SAFETENSORS:
        (snapshot / "model.safetensors.index.json").unlink()
    repo_cache = hub / ("models--" + metadata["source_repo"].replace("/", "--"))
    for index, shard in enumerate(metadata["target_shards"], start=1):
        blob = repo_cache / "blobs" / f"{index:064x}"
        blob.touch()
        _replace_with_symlink(snapshot / shard, blob)

    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    assert truth.target_weights.layout is expected_layout
    assert truth.target_weights.shards == tuple(metadata["target_shards"])


def test_repointing_same_indexed_shard_name_to_another_repo_blob_changes_receipt(
    tmp_path: Path,
):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    binding = _binding(snapshot, metadata)
    before = probe_qwen_artifact(snapshot, binding=binding)
    shard_name = metadata["target_shards"][0]
    shard = snapshot / shard_name
    repo_cache = hub / ("models--" + metadata["source_repo"].replace("/", "--"))
    replacement_blob = repo_cache / "blobs" / ("b" * 64)
    replacement_blob.touch()
    _replace_with_symlink(shard, replacement_blob)

    after = probe_qwen_artifact(snapshot, binding=binding)
    assert before.target_weights.layout is TargetWeightLayout.INDEXED_SAFETENSORS
    assert after.target_weights.layout is TargetWeightLayout.INDEXED_SAFETENSORS
    assert before.target_weights.shards == after.target_weights.shards
    assert before.target_weights.index_sha256 == after.target_weights.index_sha256
    assert (
        dict(before.target_weights.file_identities)[shard_name]
        != (dict(after.target_weights.file_identities)[shard_name])
    )
    assert before.verification_id != after.verification_id


def test_regular_weight_bytes_cannot_mint_even_when_name_and_index_match(
    tmp_path: Path,
):
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    binding = _binding(snapshot, metadata)
    shard = snapshot / metadata["target_shards"][0]
    shard.unlink()
    shard.write_bytes(b"first regular weight bytes")
    first = probe_qwen_artifact(snapshot, binding=binding)
    shard.write_bytes(b"changed regular weight bytes")
    changed = probe_qwen_artifact(snapshot, binding=binding)

    for truth in (first, changed):
        assert truth.target_weights.layout is TargetWeightLayout.INVALID_WEIGHTS
        assert truth.target_weights.file_identities == ()
    assert to_verified_runtime_target(first) is None
    with pytest.raises(ArtifactProbeError, match="target weights are incomplete"):
        to_verified_runtime_target(changed)


@pytest.mark.parametrize(
    "escape_kind",
    [
        "dangling",
        "external",
        "sibling_revision",
        "nested_blob",
        "symlinked_blob_dir",
    ],
)
def test_indexed_weight_symlink_escape_cannot_mint_runtime_target(
    tmp_path: Path, escape_kind: str
):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    repo_cache = hub / ("models--" + metadata["source_repo"].replace("/", "--"))
    shard = snapshot / metadata["target_shards"][0]
    if escape_kind == "dangling":
        target = tmp_path / "missing.safetensors"
    elif escape_kind == "external":
        target = tmp_path / "external.safetensors"
        target.touch()
    elif escape_kind == "sibling_revision":
        target = repo_cache / "snapshots" / ("f" * 40) / "model.safetensors"
        target.parent.mkdir()
        target.touch()
    elif escape_kind == "nested_blob":
        target = repo_cache / "blobs" / "nested" / ("a" * 64)
        target.parent.mkdir()
        target.touch()
    else:
        external_blobs = tmp_path / "external-blobs"
        external_blobs.mkdir()
        target = external_blobs / ("a" * 64)
        target.touch()
        blobs_dir = repo_cache / "blobs"
        for blob in blobs_dir.iterdir():
            blob.rename(external_blobs / blob.name)
        blobs_dir.rmdir()
        blobs_dir.symlink_to(external_blobs, target_is_directory=True)
        target = repo_cache / "blobs" / target.name
    _replace_with_symlink(shard, target)

    if escape_kind == "symlinked_blob_dir":
        with pytest.raises(ArtifactProbeError, match="canonical same-repo blob"):
            probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    else:
        truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
        assert truth.target_weights.layout is TargetWeightLayout.INVALID_WEIGHTS
        with pytest.raises(ArtifactProbeError, match="target weights are incomplete"):
            to_verified_runtime_target(truth)


def test_single_weight_external_symlink_cannot_mint_runtime_target(tmp_path: Path):
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen35_4b_4bit")
    (snapshot / "model.safetensors.index.json").unlink()
    external = tmp_path / "external-single.safetensors"
    external.touch()
    _replace_with_symlink(snapshot / "model.safetensors", external)

    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    assert truth.target_weights.layout is TargetWeightLayout.INVALID_WEIGHTS
    with pytest.raises(ArtifactProbeError, match="target weights are incomplete"):
        to_verified_runtime_target(truth)


def test_production_injector_uses_shared_locator_with_identical_precedence(
    tmp_path: Path,
):
    nested = tmp_path / "mtp" / "model.safetensors"
    nested.parent.mkdir()
    nested.touch()
    explicit = tmp_path / "model-mtp.safetensors"
    explicit.touch()
    assert _find_mtp_weights_file(tmp_path) == explicit
    assert find_mtp_weights_file(tmp_path) == explicit
    layout = inspect_mtp_weights_layout(tmp_path)
    assert layout.state is MTPWeightPathState.ROOT_MODEL_MTP
    assert layout.candidate == explicit


def test_future_unknown_locator_path_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    unknown = tmp_path / "future" / "head.safetensors"
    unknown.parent.mkdir()
    unknown.touch()
    monkeypatch.setattr(qwen_layout, "find_mtp_weights_file", lambda _path: unknown)
    layout = inspect_mtp_weights_layout(tmp_path)
    assert layout.state is MTPWeightPathState.UNSUPPORTED
    assert layout.relative_path is None
    assert layout.candidate is None


def test_receipt_extractor_reproduces_fixture_truth(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")
    assert (
        extract_receipt(
            [
                "--snapshot-dir",
                str(snapshot),
                "--repo-id",
                metadata["source_repo"],
                "--revision",
                metadata["revision"],
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["schema_version"] == 1
    assert receipt["target_weights"]["index_sha256"] == metadata["index_sha256"]
    assert receipt["target_weights"]["shards"] == metadata["target_shards"]
    assert receipt["target_weights"]["file_identities"] == {
        shard: f"hf_blob:{blob_id}"
        for shard, blob_id in metadata["target_blob_ids"].items()
    }
    assert receipt["mtp_locator"]["hf_blob_id"] == metadata["mtp_candidate"]["blob_id"]
    assert receipt["mtp_locator"]["content_identity"] == (
        f"hf_blob:{metadata['mtp_candidate']['blob_id']}"
    )
    assert len(receipt["geometry"]["layer_types"]) == 64
    assert str(tmp_path) not in json.dumps(receipt)


def test_resolved_snapshot_probe_derives_only_pinned_local_identity(tmp_path: Path):
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")

    truth = probe_resolved_qwen_artifact(
        snapshot,
        repo_id=metadata["source_repo"],
    )

    assert truth is not None
    assert truth.source_repo == metadata["source_repo"]
    assert truth.revision == metadata["revision"]
    assert str(tmp_path) not in json.dumps(truth.to_status_dict())

    local_path_truth = probe_resolved_qwen_artifact(
        snapshot,
        repo_id=str(snapshot),
    )
    assert local_path_truth is not None
    assert local_path_truth.source_repo == metadata["source_repo"]
    assert local_path_truth.revision == metadata["revision"]
    assert str(tmp_path) not in json.dumps(local_path_truth.to_status_dict())

    subfolder_snapshot, _hub, subfolder_metadata = _materialize_snapshot(
        tmp_path, "qwen36_35b_4bit", subfolder="4bit"
    )
    subfolder_truth = probe_resolved_qwen_artifact(
        subfolder_snapshot,
        repo_id=str(subfolder_snapshot),
    )
    assert subfolder_truth is not None
    assert subfolder_truth.source_repo == subfolder_metadata["source_repo"]
    assert subfolder_truth.target_subfolder == "4bit"


def test_resolved_snapshot_probe_omits_local_and_wrong_repo_paths(tmp_path: Path):
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")
    local_copy = tmp_path / "local-copy"
    local_copy.mkdir()
    (local_copy / "config.json").write_text("{}", encoding="utf-8")

    assert (
        probe_resolved_qwen_artifact(local_copy, repo_id=metadata["source_repo"])
        is None
    )
    assert probe_resolved_qwen_artifact(snapshot, repo_id="other/repo") is None

    spoof = (
        tmp_path
        / "outside"
        / ("models--" + metadata["source_repo"].replace("/", "--"))
        / "snapshots"
        / metadata["revision"]
    )
    spoof.mkdir(parents=True)
    assert probe_resolved_qwen_artifact(spoof, repo_id=str(spoof)) is None


def test_conversion_seam_owns_exact_target_only_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    fake = ModuleType("rapid_mlx.qwen_runtime_plan")

    class _Target:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Verified:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def _mint_verified(**kwargs):
        return _Verified(**kwargs)

    fake.QwenTargetIdentity = _Target
    fake._mint_verified_qwen_target = _mint_verified
    monkeypatch.setitem(sys.modules, "rapid_mlx.qwen_runtime_plan", fake)
    converted = to_verified_runtime_target(truth)
    assert converted.verification_id == truth.verification_id
    assert converted.verification_authority == "rapid_mlx.qwen_artifact:hub-snapshot-v1"
    assert converted.identity.target_repo == metadata["source_repo"]
    assert converted.identity.layer_layout == truth.geometry.layer_types
    assert converted.identity.cache_geometry == tuple(
        sorted(converted.identity.cache_geometry)
    )
    assert "drafter" not in converted.identity.__dict__
    assert (
        json.loads(converted.identity.weight_layout)["index_sha256"]
        == metadata["index_sha256"]
    )


def test_conversion_seam_composes_with_integrated_private_runtime_mint(
    tmp_path: Path,
):
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    converted = to_verified_runtime_target(truth)
    assert isinstance(converted, qwen_plan.VerifiedQwenTarget)
    assert converted.identity.target_repo == metadata["source_repo"]
    assert converted.identity.layer_layout == truth.geometry.layer_types
    assert converted.verification_id == truth.verification_id
    assert converted.verification_authority == (
        "rapid_mlx.qwen_artifact:hub-snapshot-v1"
    )


@pytest.mark.parametrize(
    "mutation",
    ["config", "index", "target_shard", "mtp_link"],
)
def test_runtime_conversion_rejects_stale_artifact_then_accepts_fresh_probe(
    tmp_path: Path, mutation: str
) -> None:
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")
    binding = _binding(snapshot, metadata)
    old_truth = probe_qwen_artifact(snapshot, binding=binding)
    repo_cache = snapshot.parents[1]

    if mutation in {"config", "index"}:
        name = "config.json" if mutation == "config" else "model.safetensors.index.json"
        leaf = snapshot / name
        content = leaf.resolve(strict=True).read_bytes() + b"\n "
        replacement = repo_cache / "blobs" / hashlib.sha256(content).hexdigest()
        replacement.write_bytes(content)
        _replace_with_symlink(leaf, replacement)
    elif mutation == "target_shard":
        leaf = snapshot / metadata["target_shards"][0]
        replacement = repo_cache / "blobs" / ("d" * 64)
        replacement.touch()
        _replace_with_symlink(leaf, replacement)
    else:
        leaf = snapshot / metadata["mtp_candidate"]["path"]
        target = leaf.resolve(strict=True)
        # Recreate the same canonical link: the portable receipt is unchanged,
        # but the private leaf-metadata seal must reject the old observation.
        _replace_with_symlink(leaf, target)

    assert to_verified_runtime_target(old_truth) is None
    fresh_truth = probe_qwen_artifact(snapshot, binding=binding)
    fresh_target = to_verified_runtime_target(fresh_truth)
    assert isinstance(fresh_target, qwen_plan.VerifiedQwenTarget)
    assert fresh_target.verification_id == fresh_truth.verification_id


def test_fresh_conversion_reads_only_resolved_metadata_and_never_network_or_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")
    binding = _binding(snapshot, metadata)
    truth = probe_qwen_artifact(snapshot, binding=binding)
    metadata_leaves = {
        snapshot / "config.json",
        snapshot / "model.safetensors.index.json",
    }
    metadata_blobs = {path.resolve(strict=True) for path in metadata_leaves}
    tensor_blobs = {
        (snapshot / shard).resolve(strict=True) for shard in metadata["target_shards"]
    }
    tensor_blobs.add(
        (snapshot / metadata["mtp_candidate"]["path"]).resolve(strict=True)
    )
    opened: set[Path] = set()
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        opened.add(path)
        if path in tensor_blobs:
            raise AssertionError("runtime conversion must not open tensor content")
        return original_open(path, *args, **kwargs)

    def reject_network(*_args, **_kwargs):
        raise AssertionError("runtime conversion must remain offline")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", reject_network)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", reject_network)

    assert to_verified_runtime_target(truth) is not None
    assert metadata_blobs <= opened
    assert metadata_leaves.isdisjoint(opened)
    assert tensor_blobs.isdisjoint(opened)


def test_private_freshness_state_is_redacted_from_status_and_repr(tmp_path: Path):
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")
    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    rendered = json.dumps(truth.to_status_dict(), sort_keys=True)
    receipt = json.dumps(truth.to_receipt_dict(), sort_keys=True)

    assert str(tmp_path) not in rendered
    assert str(tmp_path) not in receipt
    assert str(tmp_path) not in repr(truth)
    for private_name in ("binding", "observation_seal", "portable_receipt_sha256"):
        assert private_name not in rendered
        assert private_name not in receipt


def test_conversion_seam_rejects_unverified_and_incomplete_layers(tmp_path: Path):
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    unverified = probe_qwen_artifact(snapshot)
    verified = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    with pytest.raises(
        ArtifactProbeError, match="resolver-verified artifact capability"
    ):
        to_verified_runtime_target(unverified)
    config_path = snapshot / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["text_config"]["layer_types"] = []
    config_path.write_text(json.dumps(config), encoding="utf-8")
    incomplete = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    with pytest.raises(ArtifactProbeError, match="ordered layer layout"):
        to_verified_runtime_target(incomplete)


def test_layout_classifier_rejects_outside_and_non_file_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path.parent / "outside-mtp.safetensors"
    outside.touch(exist_ok=True)
    monkeypatch.setattr(qwen_layout, "find_mtp_weights_file", lambda _path: outside)
    assert inspect_mtp_weights_layout(tmp_path).state is MTPWeightPathState.UNSUPPORTED

    candidate_dir = tmp_path / "mtp.safetensors"
    candidate_dir.mkdir()
    monkeypatch.setattr(
        qwen_layout, "find_mtp_weights_file", lambda _path: candidate_dir
    )
    layout = inspect_mtp_weights_layout(tmp_path)
    assert layout.state is MTPWeightPathState.UNSUPPORTED
    assert layout.storage is MTPWeightStorage.OTHER


def test_binding_and_path_parsers_cover_fail_closed_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    binding = _binding(snapshot, metadata)
    assert binding.matches(tmp_path / "missing") is False
    assert qwen_artifact._canonical_repo_id(object()) is None
    monkeypatch.setattr("huggingface_hub.utils.validate_repo_id", lambda _repo: None)
    assert qwen_artifact._canonical_repo_id("a/b/c") is None
    assert qwen_artifact._canonical_subfolder(1) is None

    from huggingface_hub import constants as hub_constants

    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path))
    assert CONFIGURED_HUB_CACHE_ROOT() == tmp_path.resolve()
    original_import = __import__

    def reject_hub_constants(name, *args, **kwargs):
        if name == "huggingface_hub.constants":
            raise ImportError("synthetic missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_hub_constants)
    assert CONFIGURED_HUB_CACHE_ROOT() is None


def test_binding_returns_none_when_cache_root_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    monkeypatch.setattr(qwen_artifact, "_configured_hub_cache_root", lambda: None)
    assert (
        verify_hub_snapshot_binding(
            snapshot,
            repo_id=metadata["source_repo"],
            revision=metadata["revision"],
        )
        is None
    )


def test_truth_constructor_rejects_inconsistent_private_capabilities(
    tmp_path: Path,
) -> None:
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    values = {
        item.name: getattr(truth, item.name)
        for item in fields(QwenArtifactTruth)
        if not item.name.startswith("_")
    }
    with pytest.raises(TypeError, match="invalid Qwen artifact runtime capability"):
        QwenArtifactTruth(
            **values,
            _runtime_capability=object(),
            _mint_token=qwen_artifact._TRUTH_MINT_TOKEN,
        )
    with pytest.raises(ValueError, match="require resolver capability"):
        QwenArtifactTruth(
            **{**values, "source_repo": None},
            _runtime_capability=qwen_artifact._VERIFIED_RUNTIME_CAPABILITY,
            _mint_token=qwen_artifact._TRUTH_MINT_TOKEN,
        )
    with pytest.raises(ValueError, match="must be redacted"):
        QwenArtifactTruth(
            **{
                **values,
                "identity_status": ArtifactIdentityStatus.UNVERIFIED_SNAPSHOT,
                "verification_id": None,
            },
            _runtime_capability=None,
            _mint_token=qwen_artifact._TRUTH_MINT_TOKEN,
        )


@pytest.mark.parametrize(
    ("index_payload", "expected"),
    [
        ([], TargetWeightLayout.INVALID_INDEX),
        ({"weight_map": {}}, TargetWeightLayout.INVALID_INDEX),
        ({"weight_map": {"weight": 7}}, TargetWeightLayout.INVALID_INDEX),
    ],
)
def test_malformed_target_indexes_fail_closed(
    tmp_path: Path, index_payload: object, expected: TargetWeightLayout
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(index_payload), encoding="utf-8"
    )
    assert probe_qwen_artifact(snapshot).target_weights.layout is expected


def test_unreadable_index_receipt_and_config_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, _hub, _metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    index_path = snapshot / "model.safetensors.index.json"
    receipt_path = snapshot / "mtp.safetensors.sha256"
    candidate = snapshot / "mtp.safetensors"
    candidate.touch()
    receipt_path.write_text("a" * 64 + "  mtp.safetensors\n", encoding="utf-8")
    original_read_text = Path.read_text

    def reject_selected(path: Path, *args, **kwargs):
        if path in {index_path, receipt_path}:
            raise OSError("synthetic unreadable metadata")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_selected)
    assert (
        qwen_artifact._target_weights(snapshot, None).layout
        is TargetWeightLayout.INVALID_INDEX
    )
    assert qwen_artifact._declared_sidecar_sha256(candidate) is None

    config = tmp_path / "bad-config"
    config.mkdir()
    (config / "config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ArtifactProbeError, match="contain an object"):
        probe_qwen_artifact(config)
    with pytest.raises(ArtifactProbeError, match="readable config"):
        probe_qwen_artifact(tmp_path / "missing-config")


def test_orphan_shards_are_reported_without_an_index(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model-00001-of-00002.safetensors").touch()
    truth = probe_qwen_artifact(snapshot)
    assert truth.target_weights.layout is TargetWeightLayout.ORPHAN_SHARDS
    assert truth.target_weights.shards == ("model-00001-of-00002.safetensors",)


def test_weight_and_sidecar_metadata_errors_drop_content_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, hub, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")
    binding = _binding(snapshot, metadata)
    repo_cache = hub / ("models--" + metadata["source_repo"].replace("/", "--"))
    shard = snapshot / metadata["target_shards"][0]
    blob = (
        repo_cache / "blobs" / metadata["target_blob_ids"][metadata["target_shards"][0]]
    )

    directory = tmp_path / "not-a-file"
    directory.mkdir()
    assert (
        qwen_artifact._weight_file_identity(
            directory, snapshot_dir=tmp_path, binding=binding
        )
        is None
    )

    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external = external_dir / "weight.safetensors"
    external.symlink_to(blob)
    assert (
        qwen_artifact._weight_file_identity(
            external, snapshot_dir=snapshot, binding=binding
        )
        is None
    )

    original_resolve = Path.resolve

    def reject_blobs(path: Path, *args, **kwargs):
        if path == repo_cache / "blobs":
            raise OSError("synthetic blobs resolution failure")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", reject_blobs)
    assert (
        qwen_artifact._weight_file_identity(
            shard, snapshot_dir=snapshot, binding=binding
        )
        is None
    )


def test_sidecar_blob_validation_and_stat_failures_are_non_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")
    binding = _binding(snapshot, metadata)
    candidate = snapshot / "mtp" / "model.safetensors"

    dangling = tmp_path / "dangling.safetensors"
    dangling.symlink_to(tmp_path / "missing-blob")
    assert (
        qwen_artifact._hf_blob_id(dangling, snapshot_dir=tmp_path, binding=binding)
        is None
    )

    invalid_blob = binding._repo_cache_dir / "blobs" / "not-a-blob-id"
    invalid_blob.touch()
    _replace_with_symlink(candidate, invalid_blob)
    assert (
        qwen_artifact._hf_blob_id(candidate, snapshot_dir=snapshot, binding=binding)
        is None
    )
    valid_blob = (
        binding._repo_cache_dir / "blobs" / metadata["mtp_candidate"]["blob_id"]
    )
    _replace_with_symlink(candidate, valid_blob)

    wrong_binding = _binding(snapshot, metadata)
    object.__setattr__(wrong_binding, "_artifact_dir", tmp_path.resolve())
    assert (
        qwen_artifact._hf_blob_id(
            candidate, snapshot_dir=snapshot, binding=wrong_binding
        )
        is None
    )

    original_resolve = Path.resolve

    def reject_blobs(path: Path, *args, **kwargs):
        if path == binding._repo_cache_dir / "blobs":
            raise OSError("synthetic blobs resolution failure")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", reject_blobs)
    assert (
        qwen_artifact._hf_blob_id(candidate, snapshot_dir=snapshot, binding=binding)
        is None
    )
    monkeypatch.setattr(Path, "resolve", original_resolve)

    original_stat = Path.stat

    def reject_candidate(path: Path, *args, **kwargs):
        if path == candidate:
            raise OSError("synthetic stat failure")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", reject_candidate)
    layout = qwen_layout.MTPWeightLayout(
        state=MTPWeightPathState.NESTED_MODEL,
        relative_path="mtp/model.safetensors",
        storage=MTPWeightStorage.SYMLINK,
        candidate=candidate,
    )
    monkeypatch.setattr(
        qwen_artifact, "inspect_mtp_weights_layout", lambda _path: layout
    )
    monkeypatch.setattr(qwen_artifact, "_hf_blob_id", lambda *_args, **_kwargs: None)
    locator = qwen_artifact._mtp_locator(snapshot, binding)
    assert locator.file_size_bytes is None


def test_resolved_snapshot_derivation_rejects_malformed_cache_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "plain" / "snapshot"
    artifact.mkdir(parents=True)
    assert (
        qwen_artifact._derive_selected_snapshot_repo_id(artifact, repo_id=str(tmp_path))
        is None
    )
    assert (
        qwen_artifact._derive_selected_snapshot_repo_id(artifact, repo_id=str(artifact))
        is None
    )

    cache = tmp_path / "hub"
    cache.mkdir(exist_ok=True)
    monkeypatch.setattr(qwen_artifact, "_configured_hub_cache_root", lambda: None)
    shaped = cache / "models--org--model" / "snapshots" / ("a" * 40)
    shaped.mkdir(parents=True)
    assert (
        qwen_artifact._derive_selected_snapshot_repo_id(shaped, repo_id=str(shaped))
        is None
    )
    monkeypatch.setattr(qwen_artifact, "_configured_hub_cache_root", lambda: cache)

    valid = cache / "models--org--model" / "snapshots" / ("c" * 40)
    valid.mkdir(parents=True)
    original_resolve = Path.resolve

    def reject_cache_parent(path: Path, *args, **kwargs):
        if path == cache:
            raise OSError("synthetic cache resolution failure")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", reject_cache_parent)
    assert (
        qwen_artifact._derive_selected_snapshot_repo_id(valid, repo_id=str(valid))
        is None
    )
    monkeypatch.setattr(Path, "resolve", original_resolve)

    for entry in ("repo--org--model", "models--a--b--c", "models--.bad"):
        candidate = cache / entry / "snapshots" / ("b" * 40)
        candidate.mkdir(parents=True)
        assert (
            qwen_artifact._derive_selected_snapshot_repo_id(
                candidate, repo_id=str(candidate)
            )
            is None
        )


def test_resolved_probe_rejects_invalid_path_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RaisingPath:
        def __init__(self, _value):
            raise TypeError("synthetic invalid path")

    monkeypatch.setattr(qwen_artifact, "Path", RaisingPath)
    assert probe_resolved_qwen_artifact(object(), repo_id="org/model") is None
    assert (
        qwen_artifact._derive_selected_snapshot_repo_id(object(), repo_id=object())
        is None
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda truth: object.__setattr__(
                truth, "identity_status", ArtifactIdentityStatus.UNVERIFIED_SNAPSHOT
            ),
            "verified Hub identity",
        ),
        (
            lambda truth: object.__setattr__(truth, "outer_model_type", None),
            "model types",
        ),
        (
            lambda truth: object.__setattr__(
                truth, "geometry", replace(truth.geometry, hidden_size=None)
            ),
            "target identity rejected",
        ),
        (
            lambda truth: object.__setattr__(
                truth, "quantization", replace(truth.quantization, bits=None)
            ),
            "quantization identity",
        ),
        (
            lambda truth: object.__setattr__(
                truth,
                "target_weights",
                replace(truth.target_weights, shards=()),
            ),
            "no shards",
        ),
        (
            lambda truth: object.__setattr__(
                truth,
                "target_weights",
                replace(truth.target_weights, file_identities=()),
            ),
            "identities are incomplete",
        ),
        (
            lambda truth: object.__setattr__(
                truth,
                "target_weights",
                replace(truth.target_weights, index_sha256=None),
            ),
            "no canonical index digest",
        ),
    ],
)
def test_runtime_target_conversion_rejects_corrupted_verified_truth(
    tmp_path: Path, mutation, message: str
) -> None:
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen36_35b_4bit")
    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    mutation(truth)
    if message == "verified Hub identity":
        with pytest.raises(ArtifactProbeError, match=message):
            to_verified_runtime_target(truth)
    else:
        assert to_verified_runtime_target(truth) is None


def test_runtime_conversion_rejects_wrong_types_and_unavailable_core_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ArtifactProbeError, match="requires QwenArtifactTruth"):
        to_verified_runtime_target(object())
    with pytest.raises(ArtifactProbeError, match="requires QwenArtifactTruth"):
        to_runtime_drafter_identity(object())

    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")
    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    fake = ModuleType("rapid_mlx.qwen_runtime_plan")
    monkeypatch.setitem(sys.modules, "rapid_mlx.qwen_runtime_plan", fake)
    with pytest.raises(ArtifactProbeError, match="identity API is unavailable"):
        to_verified_runtime_target(truth)


def test_runtime_conversion_wraps_core_identity_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, _hub, metadata = _materialize_snapshot(tmp_path, "qwen38_27b_4bit")
    truth = probe_qwen_artifact(snapshot, binding=_binding(snapshot, metadata))
    fake = ModuleType("rapid_mlx.qwen_runtime_plan")

    class RejectTarget:
        def __init__(self, **_kwargs):
            raise ValueError("synthetic target rejection")

    class RejectDrafter:
        def __init__(self, **_kwargs):
            raise ValueError("synthetic drafter rejection")

    fake.QwenTargetIdentity = RejectTarget
    fake._mint_verified_qwen_target = lambda **_kwargs: None
    fake.QwenDrafterIdentity = RejectDrafter
    monkeypatch.setitem(sys.modules, "rapid_mlx.qwen_runtime_plan", fake)
    with pytest.raises(ArtifactProbeError, match="target identity rejected"):
        to_verified_runtime_target(truth)
    with pytest.raises(ArtifactProbeError, match="drafter identity rejected"):
        to_runtime_drafter_identity(truth)
