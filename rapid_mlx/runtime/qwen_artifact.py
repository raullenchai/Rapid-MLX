# SPDX-License-Identifier: Apache-2.0
"""Offline, content-free artifact facts for Qwen runtime planning.

The probe in this module reads ``config.json``, the safetensors index, optional
sidecar ``.sha256`` receipts, and filesystem metadata from an already-resolved
local snapshot. It never imports or loads a model, opens weight tensor content,
contacts the Hub, or reports an absolute cache path.

This is deliberately an artifact *truth* surface, not an eligibility gate.
For example, the current MTP locator accepts a root ``model.safetensors``
file even though that filename alone cannot prove the file is an MTP head.
The closed ``root_model_ambiguous`` result preserves that distinction so a
future runtime-plan resolver can fail closed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from rapid_mlx.qwen_artifact_layout import (
    MTPWeightPathState,
    MTPWeightStorage,
    inspect_mtp_weights_layout,
)


class ArtifactProbeError(ValueError):
    """Raised when a local snapshot cannot produce trustworthy config facts."""


class TargetWeightLayout(str, Enum):
    """Closed target-weight layouts visible without opening tensor files."""

    INDEXED_SAFETENSORS = "indexed_safetensors"
    SINGLE_SAFETENSORS = "single_safetensors"
    ORPHAN_SHARDS = "orphan_shards"
    NONE = "none"
    INVALID_INDEX = "invalid_index"
    INCOMPLETE_INDEX = "incomplete_index"


class ArtifactIdentityStatus(str, Enum):
    """Whether resolver-supplied provenance is safe for qualification."""

    VERIFIED_HUB_SNAPSHOT = "verified_hub_snapshot"
    UNVERIFIED_SNAPSHOT = "unverified_snapshot"
    BINDING_MISMATCH = "binding_mismatch"


_BINDING_TOKEN = object()
_IMMUTABLE_HF_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True, init=False)
class VerifiedHubSnapshotBinding:
    """Resolver-validated binding between a canonical Hub id and local snapshot.

    Instances can only be created by :func:`verify_hub_snapshot_binding`, which
    verifies the standard Hub-cache directory layout and the exact resolved
    directory.  The private paths are intentionally absent from status output.
    """

    repo_id: str
    revision: str
    subfolder: str | None
    _artifact_dir: Path = field(repr=False)
    _repo_cache_dir: Path = field(repr=False)

    def __init__(
        self,
        *,
        repo_id: str,
        revision: str,
        subfolder: str | None,
        artifact_dir: Path,
        repo_cache_dir: Path,
        _token: object | None = None,
    ) -> None:
        if _token is not _BINDING_TOKEN:
            raise TypeError("use verify_hub_snapshot_binding()")
        object.__setattr__(self, "repo_id", repo_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "subfolder", subfolder)
        object.__setattr__(self, "_artifact_dir", artifact_dir)
        object.__setattr__(self, "_repo_cache_dir", repo_cache_dir)

    def matches(self, artifact_dir: Path) -> bool:
        """Return whether ``artifact_dir`` is the exact verified directory."""

        try:
            return artifact_dir.resolve(strict=True) == self._artifact_dir
        except OSError:
            return False


def _canonical_repo_id(repo_id: object) -> tuple[str, ...] | None:
    if not isinstance(repo_id, str) or repo_id != repo_id.strip():
        return None
    try:
        from huggingface_hub.utils import validate_repo_id

        validate_repo_id(repo_id)
    except (ImportError, ValueError):
        return None
    parts = tuple(repo_id.split("/"))
    if len(parts) not in {1, 2}:
        return None
    return parts


def _canonical_subfolder(subfolder: object) -> str | None:
    if subfolder is None:
        return None
    if not isinstance(subfolder, str) or not subfolder:
        return None
    path = PurePosixPath(subfolder)
    if (
        "\\" in subfolder
        or path.is_absolute()
        or path.as_posix() != subfolder
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        return None
    return subfolder


def _configured_hub_cache_root() -> Path | None:
    """Return Hugging Face's configured canonical model-cache root."""

    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        root = Path(HF_HUB_CACHE).expanduser().resolve(strict=True)
    except (ImportError, OSError):  # core dependency, but identity must fail closed
        return None
    return root if root.is_dir() else None


def verify_hub_snapshot_binding(
    snapshot_dir: str | Path,
    *,
    repo_id: str,
    revision: str,
    subfolder: str | None = None,
) -> VerifiedHubSnapshotBinding | None:
    """Validate and bind a canonical local HF snapshot, otherwise return None.

    This is metadata-only and never downloads.  Mutable revisions, URLs,
    userinfo, local paths, non-canonical subfolders, and directories outside
    Hugging Face's configured
    ``<HF_HUB_CACHE>/models--<repo>/snapshots/<full-commit>`` fail closed.
    """

    repo_parts = _canonical_repo_id(repo_id)
    if repo_parts is None or not _IMMUTABLE_HF_REVISION.fullmatch(revision):
        return None
    canonical_subfolder = _canonical_subfolder(subfolder)
    if subfolder is not None and canonical_subfolder is None:
        return None

    cache_root = _configured_hub_cache_root()
    if cache_root is None:
        return None
    repo_cache = cache_root / ("models--" + "--".join(repo_parts))
    snapshots_dir = repo_cache / "snapshots"
    snapshot_root = snapshots_dir / revision
    # The requested revision itself must be the canonical directory entry.
    # Repo-cache ancestors may be symlinked, but following this final component
    # could silently relabel a sibling revision as the requested commit.
    if snapshot_root.is_symlink():
        return None
    expected = (
        snapshot_root.joinpath(*PurePosixPath(canonical_subfolder).parts)
        if canonical_subfolder is not None
        else snapshot_root
    )
    try:
        resolved_snapshots_dir = snapshots_dir.resolve(strict=True)
        resolved_snapshot_root = snapshot_root.resolve(strict=True)
        resolved_expected = expected.resolve(strict=True)
        resolved_actual = Path(snapshot_dir).resolve(strict=True)
        resolved_repo_cache = repo_cache.resolve(strict=True)
    except OSError:
        return None
    if (
        resolved_snapshots_dir.parent != resolved_repo_cache
        or resolved_snapshot_root.parent != resolved_snapshots_dir
        or not resolved_snapshot_root.is_dir()
        or resolved_actual != resolved_expected
        or not resolved_actual.is_dir()
    ):
        return None
    try:
        resolved_expected.relative_to(resolved_snapshot_root)
    except ValueError:
        return None
    return VerifiedHubSnapshotBinding(
        repo_id="/".join(repo_parts),
        revision=revision,
        subfolder=canonical_subfolder,
        artifact_dir=resolved_actual,
        repo_cache_dir=resolved_repo_cache,
        _token=_BINDING_TOKEN,
    )


@dataclass(frozen=True)
class QwenGeometry:
    """Closed model geometry used by exact qualification rows."""

    hidden_size: int | None
    num_hidden_layers: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    full_attention_interval: int | None
    linear_num_key_heads: int | None
    linear_num_value_heads: int | None
    linear_key_head_dim: int | None
    linear_value_head_dim: int | None
    num_experts: int | None
    num_experts_per_tok: int | None
    layer_types: tuple[str, ...]
    layer_type_counts: tuple[tuple[str, int], ...]
    layer_types_sha256: str | None

    def to_status_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping without model content or local paths."""

        return {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "full_attention_interval": self.full_attention_interval,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "layer_type_counts": dict(self.layer_type_counts),
            "layer_types_sha256": self.layer_types_sha256,
        }


@dataclass(frozen=True)
class QwenQuantization:
    """Content-free summary of base and per-module quantization metadata."""

    bits: int | None
    group_size: int | None
    mode: str | None
    override_count: int
    override_bits: tuple[int, ...]
    override_group_sizes: tuple[int, ...]

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
            "override_count": self.override_count,
            "override_bits": list(self.override_bits),
            "override_group_sizes": list(self.override_group_sizes),
        }


@dataclass(frozen=True)
class TargetWeights:
    """Target checkpoint file shape, derived without reading tensor content."""

    layout: TargetWeightLayout
    shard_count: int
    missing_shard_count: int
    shards: tuple[str, ...]
    index_sha256: str | None

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout.value,
            "shard_count": self.shard_count,
            "missing_shard_count": self.missing_shard_count,
            "shards": list(self.shards),
            "index_sha256": self.index_sha256,
        }


@dataclass(frozen=True)
class MTPLocatorTruth:
    """What today's production locator observes, without eligibility claims."""

    state: MTPWeightPathState
    relative_path: str | None
    storage: MTPWeightStorage
    file_size_bytes: int | None
    hf_blob_id: str | None
    declared_sha256: str | None
    declared_sha256_matches_blob: bool | None

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "relative_path": self.relative_path,
            "storage": self.storage.value,
            "file_size_bytes": self.file_size_bytes,
            "hf_blob_id": self.hf_blob_id,
            "declared_sha256": self.declared_sha256,
            "declared_sha256_matches_blob": self.declared_sha256_matches_blob,
        }


_TRUTH_MINT_TOKEN = object()
_VERIFIED_RUNTIME_CAPABILITY = object()


@dataclass(frozen=True, slots=True, init=False)
class QwenArtifactTruth:
    """Privately minted facts about one already-resolved Qwen artifact.

    Public fields remain JSON-ready, but callers cannot construct or replace
    this object into a verified state. Only a probe with a matching resolver
    binding retains the private capability accepted by runtime conversion.
    """

    source_repo: str | None
    revision: str | None
    target_subfolder: str | None
    identity_status: ArtifactIdentityStatus
    verification_id: str | None
    config_sha256: str
    outer_model_type: str | None
    text_model_type: str | None
    mtp_num_hidden_layers: int | None
    geometry: QwenGeometry
    quantization: QwenQuantization
    target_weights: TargetWeights
    mtp_locator: MTPLocatorTruth
    _runtime_capability: object | None = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        *,
        source_repo: str | None,
        revision: str | None,
        target_subfolder: str | None,
        identity_status: ArtifactIdentityStatus,
        verification_id: str | None,
        config_sha256: str,
        outer_model_type: str | None,
        text_model_type: str | None,
        mtp_num_hidden_layers: int | None,
        geometry: QwenGeometry,
        quantization: QwenQuantization,
        target_weights: TargetWeights,
        mtp_locator: MTPLocatorTruth,
        _runtime_capability: object | None = None,
        _mint_token: object | None = None,
    ) -> None:
        if _mint_token is not _TRUTH_MINT_TOKEN:
            raise TypeError("QwenArtifactTruth must be minted by probe_qwen_artifact()")
        if (
            _runtime_capability is not None
            and _runtime_capability is not _VERIFIED_RUNTIME_CAPABILITY
        ):
            raise TypeError("invalid Qwen artifact runtime capability")
        verified = _runtime_capability is _VERIFIED_RUNTIME_CAPABILITY
        if verified != (
            identity_status is ArtifactIdentityStatus.VERIFIED_HUB_SNAPSHOT
            and source_repo is not None
            and revision is not None
            and verification_id is not None
        ):
            raise ValueError("verified artifact fields require resolver capability")
        if not verified and any(
            value is not None
            for value in (source_repo, revision, target_subfolder, verification_id)
        ):
            raise ValueError("unverified artifact identity fields must be redacted")
        object.__setattr__(self, "source_repo", source_repo)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "target_subfolder", target_subfolder)
        object.__setattr__(self, "identity_status", identity_status)
        object.__setattr__(self, "verification_id", verification_id)
        object.__setattr__(self, "config_sha256", config_sha256)
        object.__setattr__(self, "outer_model_type", outer_model_type)
        object.__setattr__(self, "text_model_type", text_model_type)
        object.__setattr__(self, "mtp_num_hidden_layers", mtp_num_hidden_layers)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "quantization", quantization)
        object.__setattr__(self, "target_weights", target_weights)
        object.__setattr__(self, "mtp_locator", mtp_locator)
        object.__setattr__(self, "_runtime_capability", _runtime_capability)

    def to_status_dict(self) -> dict[str, Any]:
        """Return the neutral local-status payload for later plan integration."""

        return {
            "source_repo": self.source_repo,
            "revision": self.revision,
            "target_subfolder": self.target_subfolder,
            "identity_status": self.identity_status.value,
            "identity_is_immutable": (
                self._runtime_capability is _VERIFIED_RUNTIME_CAPABILITY
            ),
            "verification_id": self.verification_id,
            "config_sha256": self.config_sha256,
            "outer_model_type": self.outer_model_type,
            "text_model_type": self.text_model_type,
            "mtp_num_hidden_layers": self.mtp_num_hidden_layers,
            "geometry": self.geometry.to_status_dict(),
            "quantization": self.quantization.to_status_dict(),
            "target_weights": self.target_weights.to_status_dict(),
            "mtp_locator": self.mtp_locator.to_status_dict(),
        }

    def to_receipt_dict(self) -> dict[str, Any]:
        """Return reproducible artifact facts, including ordered layer layout."""

        receipt = self.to_status_dict()
        receipt["schema_version"] = 1
        receipt["geometry"] = {
            **receipt["geometry"],
            "layer_types": list(self.geometry.layer_types),
        }
        return receipt


_SHARD_NAME = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")


def _optional_int(value: Any) -> int | None:
    # bool is an int subclass but is never valid geometry/quantization truth.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _geometry(text_config: dict[str, Any]) -> QwenGeometry:
    raw_layer_types = text_config.get("layer_types")
    layer_types: tuple[str, ...] = ()
    if isinstance(raw_layer_types, list) and all(
        isinstance(item, str) for item in raw_layer_types
    ):
        layer_types = tuple(raw_layer_types)

    counts: tuple[tuple[str, int], ...] = ()
    digest = None
    if layer_types:
        counts = tuple(sorted(Counter(layer_types).items()))
        encoded = json.dumps(layer_types, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()

    return QwenGeometry(
        hidden_size=_optional_int(text_config.get("hidden_size")),
        num_hidden_layers=_optional_int(text_config.get("num_hidden_layers")),
        num_attention_heads=_optional_int(text_config.get("num_attention_heads")),
        num_key_value_heads=_optional_int(text_config.get("num_key_value_heads")),
        full_attention_interval=_optional_int(
            text_config.get("full_attention_interval")
        ),
        linear_num_key_heads=_optional_int(text_config.get("linear_num_key_heads")),
        linear_num_value_heads=_optional_int(text_config.get("linear_num_value_heads")),
        linear_key_head_dim=_optional_int(text_config.get("linear_key_head_dim")),
        linear_value_head_dim=_optional_int(text_config.get("linear_value_head_dim")),
        num_experts=_optional_int(text_config.get("num_experts")),
        num_experts_per_tok=_optional_int(text_config.get("num_experts_per_tok")),
        layer_types=layer_types,
        layer_type_counts=counts,
        layer_types_sha256=digest,
    )


def _quantization(config: dict[str, Any]) -> QwenQuantization:
    raw = config.get("quantization")
    quant = raw if isinstance(raw, dict) else {}
    overrides = [value for value in quant.values() if isinstance(value, dict)]
    override_bits = sorted(
        {
            value
            for override in overrides
            if (value := _optional_int(override.get("bits"))) is not None
        }
    )
    override_group_sizes = sorted(
        {
            value
            for override in overrides
            if (value := _optional_int(override.get("group_size"))) is not None
        }
    )
    return QwenQuantization(
        bits=_optional_int(quant.get("bits")),
        group_size=_optional_int(quant.get("group_size")),
        mode=_optional_str(quant.get("mode")),
        override_count=len(overrides),
        override_bits=tuple(override_bits),
        override_group_sizes=tuple(override_group_sizes),
    )


def _target_weights(snapshot_dir: Path) -> TargetWeights:
    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return TargetWeights(TargetWeightLayout.INVALID_INDEX, 0, 0, (), None)
        index_sha256 = _canonical_json_sha256(payload)
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            return TargetWeights(
                TargetWeightLayout.INVALID_INDEX, 0, 0, (), index_sha256
            )
        shard_values = list(weight_map.values())
        if not all(isinstance(value, str) and value for value in shard_values):
            return TargetWeights(
                TargetWeightLayout.INVALID_INDEX, 0, 0, (), index_sha256
            )
        shards = tuple(sorted(set(shard_values)))
        if not shards:
            return TargetWeights(
                TargetWeightLayout.INVALID_INDEX, 0, 0, (), index_sha256
            )
        shard_paths: list[Path] = []
        for shard in shards:
            pure = PurePosixPath(shard)
            if (
                pure.is_absolute()
                or not pure.parts
                or "\\" in shard
                or pure.as_posix() != shard
                or any(part in {"", ".", ".."} for part in pure.parts)
                or pure.suffix != ".safetensors"
            ):
                return TargetWeights(
                    TargetWeightLayout.INVALID_INDEX,
                    len(shards),
                    0,
                    shards,
                    index_sha256,
                )
            shard_paths.append(snapshot_dir.joinpath(*pure.parts))
        missing = sum(not path.is_file() for path in shard_paths)
        if missing:
            return TargetWeights(
                TargetWeightLayout.INCOMPLETE_INDEX,
                len(shards),
                missing,
                shards,
                index_sha256,
            )
        return TargetWeights(
            TargetWeightLayout.INDEXED_SAFETENSORS,
            len(shards),
            0,
            shards,
            index_sha256,
        )

    if (snapshot_dir / "model.safetensors").is_file():
        return TargetWeights(
            TargetWeightLayout.SINGLE_SAFETENSORS,
            1,
            0,
            ("model.safetensors",),
            None,
        )

    orphan_shards = [
        path
        for path in snapshot_dir.iterdir()
        if path.is_file() and _SHARD_NAME.fullmatch(path.name)
    ]
    if orphan_shards:
        names = tuple(sorted(path.name for path in orphan_shards))
        return TargetWeights(
            TargetWeightLayout.ORPHAN_SHARDS,
            len(names),
            0,
            names,
            None,
        )
    return TargetWeights(TargetWeightLayout.NONE, 0, 0, (), None)


_HEX_BLOB_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SHA256_RECEIPT = re.compile(r"^([0-9a-f]{64})[ \t]+\*?[^\r\n]+$")


def _declared_sidecar_sha256(candidate: Path) -> str | None:
    receipt = candidate.with_name(candidate.name + ".sha256")
    if not receipt.is_file():
        return None
    try:
        line = receipt.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    match = _SHA256_RECEIPT.fullmatch(line)
    return match.group(1) if match is not None else None


def _hf_blob_id(
    candidate: Path, binding: VerifiedHubSnapshotBinding | None
) -> str | None:
    if binding is None or not candidate.is_symlink():
        return None
    try:
        resolved = candidate.resolve(strict=True)
        blobs_dir = (binding._repo_cache_dir / "blobs").resolve(strict=True)
    except OSError:
        return None
    if resolved.parent != blobs_dir or not _HEX_BLOB_ID.fullmatch(resolved.name):
        return None
    return resolved.name


def _mtp_locator(
    snapshot_dir: Path, binding: VerifiedHubSnapshotBinding | None
) -> MTPLocatorTruth:
    layout = inspect_mtp_weights_layout(snapshot_dir)
    candidate = layout.candidate
    # The root model filename is inherently ambiguous: it may be the entire
    # target trunk (as in Qwen3.5-4B). Do not attach head-like receipts to it.
    receipt_states = {
        MTPWeightPathState.ROOT_MTP,
        MTPWeightPathState.ROOT_MODEL_MTP,
        MTPWeightPathState.NESTED_MODEL,
    }
    if candidate is None or layout.state not in receipt_states:
        return MTPLocatorTruth(
            state=layout.state,
            relative_path=layout.relative_path,
            storage=layout.storage,
            file_size_bytes=None,
            hf_blob_id=None,
            declared_sha256=None,
            declared_sha256_matches_blob=None,
        )

    try:
        size = candidate.stat().st_size
    except OSError:
        size = None
    blob_id = _hf_blob_id(candidate, binding)
    declared_sha256 = _declared_sidecar_sha256(candidate)
    matches = (
        declared_sha256 == blob_id
        if declared_sha256 is not None and blob_id is not None and len(blob_id) == 64
        else None
    )
    return MTPLocatorTruth(
        state=layout.state,
        relative_path=layout.relative_path,
        storage=layout.storage,
        file_size_bytes=size,
        hf_blob_id=blob_id,
        declared_sha256=declared_sha256,
        declared_sha256_matches_blob=matches,
    )


def probe_qwen_artifact(
    snapshot_dir: str | Path,
    *,
    binding: VerifiedHubSnapshotBinding | None = None,
) -> QwenArtifactTruth:
    """Probe one already-resolved snapshot without loading or reading weights.

    Provenance is accepted only as a :class:`VerifiedHubSnapshotBinding`
    produced by the canonical-layout validator. An arbitrary repo string,
    revision, URL, or path cannot enter status through this API. The returned
    status mapping never includes ``snapshot_dir``.
    """

    snapshot = Path(snapshot_dir)
    config_path = snapshot / "config.json"
    try:
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactProbeError("snapshot has no readable config.json") from exc
    if not isinstance(config, dict):
        raise ArtifactProbeError("snapshot config.json must contain an object")

    nested_text = config.get("text_config")
    text_config = nested_text if isinstance(nested_text, dict) else config
    if binding is None:
        identity_status = ArtifactIdentityStatus.UNVERIFIED_SNAPSHOT
        verified_binding = None
    elif binding.matches(snapshot):
        identity_status = ArtifactIdentityStatus.VERIFIED_HUB_SNAPSHOT
        verified_binding = binding
    else:
        identity_status = ArtifactIdentityStatus.BINDING_MISMATCH
        verified_binding = None

    geometry = _geometry(text_config)
    quantization = _quantization(config)
    target_weights = _target_weights(snapshot)
    mtp_locator = _mtp_locator(snapshot, verified_binding)
    config_sha256 = _canonical_json_sha256(config)
    verification_id = None
    if verified_binding is not None:
        verification_id = "hf-snapshot-sha256:" + _canonical_json_sha256(
            {
                "repo_id": verified_binding.repo_id,
                "revision": verified_binding.revision,
                "subfolder": verified_binding.subfolder,
                "config_sha256": config_sha256,
                "index_sha256": target_weights.index_sha256,
                "shards": target_weights.shards,
            }
        )

    return QwenArtifactTruth(
        source_repo=(verified_binding.repo_id if verified_binding else None),
        revision=(verified_binding.revision if verified_binding else None),
        target_subfolder=(verified_binding.subfolder if verified_binding else None),
        identity_status=identity_status,
        verification_id=verification_id,
        # Canonical JSON keeps the artifact identity stable across an
        # irrelevant trailing newline or indentation-only repack.
        config_sha256=config_sha256,
        outer_model_type=_optional_str(config.get("model_type")),
        text_model_type=_optional_str(text_config.get("model_type")),
        mtp_num_hidden_layers=_optional_int(text_config.get("mtp_num_hidden_layers")),
        geometry=geometry,
        quantization=quantization,
        target_weights=target_weights,
        mtp_locator=mtp_locator,
        _runtime_capability=(
            _VERIFIED_RUNTIME_CAPABILITY if verified_binding is not None else None
        ),
        _mint_token=_TRUTH_MINT_TOKEN,
    )


def probe_resolved_qwen_artifact(
    snapshot_dir: str | Path,
    *,
    repo_id: str,
) -> QwenArtifactTruth | None:
    """Probe an already-selected canonical Hub snapshot, otherwise omit it.

    Revision and subfolder come only from the concrete local snapshot path;
    this helper never reads a mutable Hub ref or performs a network lookup.
    The normal binding validator remains authoritative for repo/path identity.
    """

    try:
        artifact_dir = Path(snapshot_dir).expanduser().absolute()
    except (OSError, TypeError, ValueError):
        return None

    snapshot_parent = next(
        (parent for parent in artifact_dir.parents if parent.name == "snapshots"),
        None,
    )
    if snapshot_parent is None:
        return None
    try:
        relative = artifact_dir.relative_to(snapshot_parent)
    except ValueError:
        return None
    if not relative.parts:
        return None
    revision = relative.parts[0]
    subfolder = "/".join(relative.parts[1:]) or None
    binding = verify_hub_snapshot_binding(
        artifact_dir,
        repo_id=repo_id,
        revision=revision,
        subfolder=subfolder,
    )
    if binding is None:
        return None
    return probe_qwen_artifact(artifact_dir, binding=binding)


def _runtime_cache_geometry(truth: QwenArtifactTruth) -> tuple[tuple[str, str], ...]:
    geometry = truth.geometry
    required = {
        "full_attention_interval": geometry.full_attention_interval,
        "hidden_size": geometry.hidden_size,
        "linear_key_head_dim": geometry.linear_key_head_dim,
        "linear_num_key_heads": geometry.linear_num_key_heads,
        "linear_num_value_heads": geometry.linear_num_value_heads,
        "linear_value_head_dim": geometry.linear_value_head_dim,
        "num_attention_heads": geometry.num_attention_heads,
        "num_hidden_layers": geometry.num_hidden_layers,
        "num_key_value_heads": geometry.num_key_value_heads,
    }
    if any(value is None for value in required.values()):
        raise ArtifactProbeError("artifact has incomplete Qwen cache geometry")
    optional = {
        "num_experts": geometry.num_experts,
        "num_experts_per_tok": geometry.num_experts_per_tok,
    }
    values = {**required, **{k: v for k, v in optional.items() if v is not None}}
    return tuple(sorted((key, str(value)) for key, value in values.items()))


def to_verified_runtime_target(truth: QwenArtifactTruth):
    """Convert verified truth to the core target-only identity, fail closed.

    Import is delayed so this truth slice remains independently cherry-pickable
    while the core planner lands.  The conversion lives here—not in a caller—so
    ordered layers, quantization, weight receipts, and cache geometry have one
    canonical mapping.
    """

    if not isinstance(truth, QwenArtifactTruth):
        raise ArtifactProbeError("runtime conversion requires QwenArtifactTruth")
    if truth._runtime_capability is not _VERIFIED_RUNTIME_CAPABILITY:
        raise ArtifactProbeError(
            "runtime conversion requires resolver-verified artifact capability"
        )
    if (
        truth.identity_status is not ArtifactIdentityStatus.VERIFIED_HUB_SNAPSHOT
        or truth.source_repo is None
        or truth.revision is None
        or truth.verification_id is None
    ):
        raise ArtifactProbeError("runtime conversion requires verified Hub identity")
    if not truth.outer_model_type or not truth.text_model_type:
        raise ArtifactProbeError("artifact has incomplete Qwen model types")
    if not truth.geometry.layer_types or truth.geometry.num_hidden_layers != len(
        truth.geometry.layer_types
    ):
        raise ArtifactProbeError("artifact has incomplete ordered layer layout")
    if (
        truth.quantization.bits is None
        or truth.quantization.group_size is None
        or truth.quantization.mode is None
    ):
        raise ArtifactProbeError("artifact has incomplete quantization identity")
    if (
        truth.target_weights.layout
        not in {
            TargetWeightLayout.INDEXED_SAFETENSORS,
            TargetWeightLayout.SINGLE_SAFETENSORS,
        }
        or truth.target_weights.missing_shard_count
    ):
        raise ArtifactProbeError("artifact target weights are incomplete")
    if not truth.target_weights.shards:
        raise ArtifactProbeError("artifact target weight receipt has no shards")
    if (
        truth.target_weights.layout is TargetWeightLayout.INDEXED_SAFETENSORS
        and truth.target_weights.index_sha256 is None
    ):
        raise ArtifactProbeError("indexed artifact has no canonical index digest")

    try:
        from rapid_mlx.qwen_runtime_plan import (
            QwenTargetIdentity,
            _mint_verified_qwen_target,
        )
    except (ImportError, AttributeError) as exc:
        raise ArtifactProbeError(
            "corrected Qwen target-only runtime identity API is unavailable"
        ) from exc

    quantization = json.dumps(
        truth.quantization.to_status_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    weight_layout = json.dumps(
        truth.target_weights.to_status_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        identity = QwenTargetIdentity(
            target_repo=truth.source_repo,
            target_revision=truth.revision,
            target_subfolder=truth.target_subfolder,
            outer_model_type=truth.outer_model_type,
            language_model_type=truth.text_model_type,
            quantization=quantization,
            weight_layout=weight_layout,
            layer_layout=truth.geometry.layer_types,
            cache_geometry=_runtime_cache_geometry(truth),
        )
        return _mint_verified_qwen_target(
            identity=identity,
            verification_id=truth.verification_id,
            verification_authority="rapid_mlx.qwen_artifact:hub-snapshot-v1",
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactProbeError(
            "runtime target identity rejected artifact truth"
        ) from exc


__all__ = [
    "ArtifactProbeError",
    "ArtifactIdentityStatus",
    "MTPLocatorTruth",
    "QwenArtifactTruth",
    "QwenGeometry",
    "QwenQuantization",
    "TargetWeightLayout",
    "TargetWeights",
    "probe_qwen_artifact",
    "probe_resolved_qwen_artifact",
    "to_verified_runtime_target",
    "VerifiedHubSnapshotBinding",
    "verify_hub_snapshot_binding",
]
