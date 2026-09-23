# SPDX-License-Identifier: Apache-2.0
"""Offline, content-free artifact facts for Qwen runtime planning.

The probe in this module only reads ``config.json`` and the safetensors index
shape of an already-resolved local snapshot.  It never imports a model,
opens weight tensors, contacts the Hub, or reports an absolute cache path.

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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from rapid_mlx.spec_decode.mtp.qwen3_5_inject import _find_mtp_weights_file


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


class MTPFileShape(str, Enum):
    """Candidate shapes accepted by the current Qwen3.5 MTP locator."""

    ROOT_MTP = "root_mtp"
    ROOT_MODEL_MTP = "root_model_mtp"
    NESTED_MODEL = "nested_model"
    ROOT_MODEL_AMBIGUOUS = "root_model_ambiguous"
    UNSUPPORTED = "unsupported"
    NONE = "none"


class CandidateStorage(str, Enum):
    """Filesystem representation of a locator candidate."""

    SYMLINK = "symlink"
    REGULAR_FILE = "regular_file"
    NONE = "none"


class ArtifactIdentityStatus(str, Enum):
    """Whether resolver-supplied provenance is safe for qualification."""

    RESOLVER_VERIFIED_IMMUTABLE = "resolver_verified_immutable"
    DECLARED_IMMUTABLE_UNVERIFIED = "declared_immutable_unverified"
    MISSING_SOURCE = "missing_source"
    MISSING_REVISION = "missing_revision"
    MUTABLE_REVISION = "mutable_revision"


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

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout.value,
            "shard_count": self.shard_count,
            "missing_shard_count": self.missing_shard_count,
        }


@dataclass(frozen=True)
class MTPLocatorTruth:
    """What today's ``_find_mtp_weights_file`` accepts in this snapshot."""

    accepted: bool
    shape: MTPFileShape
    relative_path: str | None
    storage: CandidateStorage

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "shape": self.shape.value,
            "relative_path": self.relative_path,
            "storage": self.storage.value,
        }


@dataclass(frozen=True)
class QwenArtifactTruth:
    """Frozen, JSON-ready facts about one already-resolved Qwen artifact."""

    source_repo: str | None
    revision: str | None
    identity_status: ArtifactIdentityStatus
    config_sha256: str
    outer_model_type: str | None
    text_model_type: str | None
    mtp_num_hidden_layers: int | None
    geometry: QwenGeometry
    quantization: QwenQuantization
    target_weights: TargetWeights
    mtp_locator: MTPLocatorTruth

    def to_status_dict(self) -> dict[str, Any]:
        """Return the neutral local-status payload for later plan integration."""

        return {
            "source_repo": self.source_repo,
            "revision": self.revision,
            "identity_status": self.identity_status.value,
            "identity_is_immutable": (
                self.identity_status
                is ArtifactIdentityStatus.RESOLVER_VERIFIED_IMMUTABLE
            ),
            "config_sha256": self.config_sha256,
            "outer_model_type": self.outer_model_type,
            "text_model_type": self.text_model_type,
            "mtp_num_hidden_layers": self.mtp_num_hidden_layers,
            "geometry": self.geometry.to_status_dict(),
            "quantization": self.quantization.to_status_dict(),
            "target_weights": self.target_weights.to_status_dict(),
            "mtp_locator": self.mtp_locator.to_status_dict(),
        }


_IMMUTABLE_HF_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHARD_NAME = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")


def _optional_int(value: Any) -> int | None:
    # bool is an int subclass but is never valid geometry/quantization truth.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _geometry(text_config: dict[str, Any]) -> QwenGeometry:
    raw_layer_types = text_config.get("layer_types")
    layer_types: list[str] | None = None
    if isinstance(raw_layer_types, list) and all(
        isinstance(item, str) for item in raw_layer_types
    ):
        layer_types = raw_layer_types

    counts: tuple[tuple[str, int], ...] = ()
    digest = None
    if layer_types is not None:
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
            return TargetWeights(TargetWeightLayout.INVALID_INDEX, 0, 0)
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            return TargetWeights(TargetWeightLayout.INVALID_INDEX, 0, 0)
        shard_values = list(weight_map.values())
        if not all(isinstance(value, str) and value for value in shard_values):
            return TargetWeights(TargetWeightLayout.INVALID_INDEX, 0, 0)
        shards = set(shard_values)
        if not shards:
            return TargetWeights(TargetWeightLayout.INVALID_INDEX, 0, 0)
        shard_paths: list[Path] = []
        for shard in shards:
            pure = PurePosixPath(shard)
            if (
                pure.is_absolute()
                or not pure.parts
                or "\\" in shard
                or any(part in {"", ".", ".."} for part in pure.parts)
                or pure.suffix != ".safetensors"
            ):
                return TargetWeights(TargetWeightLayout.INVALID_INDEX, len(shards), 0)
            shard_paths.append(snapshot_dir.joinpath(*pure.parts))
        missing = sum(not path.is_file() for path in shard_paths)
        if missing:
            return TargetWeights(
                TargetWeightLayout.INCOMPLETE_INDEX, len(shards), missing
            )
        return TargetWeights(TargetWeightLayout.INDEXED_SAFETENSORS, len(shards), 0)

    if (snapshot_dir / "model.safetensors").is_file():
        return TargetWeights(TargetWeightLayout.SINGLE_SAFETENSORS, 1, 0)

    orphan_shards = [
        path
        for path in snapshot_dir.iterdir()
        if path.is_file() and _SHARD_NAME.fullmatch(path.name)
    ]
    if orphan_shards:
        return TargetWeights(TargetWeightLayout.ORPHAN_SHARDS, len(orphan_shards), 0)
    return TargetWeights(TargetWeightLayout.NONE, 0, 0)


def _mtp_locator(snapshot_dir: Path) -> MTPLocatorTruth:
    candidate = _find_mtp_weights_file(snapshot_dir)
    if candidate is None:
        return MTPLocatorTruth(
            accepted=False,
            shape=MTPFileShape.NONE,
            relative_path=None,
            storage=CandidateStorage.NONE,
        )

    try:
        relative = candidate.relative_to(snapshot_dir).as_posix()
    except ValueError:
        return MTPLocatorTruth(
            accepted=False,
            shape=MTPFileShape.UNSUPPORTED,
            relative_path=None,
            storage=CandidateStorage.NONE,
        )
    if not candidate.is_file():
        return MTPLocatorTruth(
            accepted=False,
            shape=MTPFileShape.UNSUPPORTED,
            relative_path=None,
            storage=CandidateStorage.NONE,
        )
    shapes = {
        "mtp.safetensors": MTPFileShape.ROOT_MTP,
        "model-mtp.safetensors": MTPFileShape.ROOT_MODEL_MTP,
        "mtp/model.safetensors": MTPFileShape.NESTED_MODEL,
        "model.safetensors": MTPFileShape.ROOT_MODEL_AMBIGUOUS,
    }
    storage = (
        CandidateStorage.SYMLINK
        if candidate.is_symlink()
        else CandidateStorage.REGULAR_FILE
    )
    shape = shapes.get(relative)
    if shape is None:
        return MTPLocatorTruth(
            accepted=False,
            shape=MTPFileShape.UNSUPPORTED,
            relative_path=None,
            storage=storage,
        )
    return MTPLocatorTruth(
        accepted=True,
        shape=shape,
        relative_path=relative,
        storage=storage,
    )


def probe_qwen_artifact(
    snapshot_dir: str | Path,
    *,
    source_repo: str | None = None,
    revision: str | None = None,
    provenance_verified: bool = False,
) -> QwenArtifactTruth:
    """Probe one already-resolved snapshot without loading or reading weights.

    ``source_repo`` and ``revision`` are provenance supplied by the resolver;
    this function never guesses them from a machine-specific cache path.
    ``provenance_verified`` is a trust-boundary signal: only the resolver that
    bound this exact directory to the supplied source and revision may set it.
    A caller-provided 40-hex string without that signal remains explicitly
    unverified and cannot claim immutable identity. The returned status mapping
    never includes ``snapshot_dir``.
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
    source = source_repo.strip() if isinstance(source_repo, str) else None
    source = source or None
    resolved_revision = revision.strip() if isinstance(revision, str) else None
    resolved_revision = resolved_revision or None
    if source is None:
        identity_status = ArtifactIdentityStatus.MISSING_SOURCE
    elif resolved_revision is None:
        identity_status = ArtifactIdentityStatus.MISSING_REVISION
    elif not _IMMUTABLE_HF_REVISION.fullmatch(resolved_revision):
        identity_status = ArtifactIdentityStatus.MUTABLE_REVISION
    elif provenance_verified is True:
        identity_status = ArtifactIdentityStatus.RESOLVER_VERIFIED_IMMUTABLE
    else:
        identity_status = ArtifactIdentityStatus.DECLARED_IMMUTABLE_UNVERIFIED

    return QwenArtifactTruth(
        source_repo=source,
        revision=resolved_revision,
        identity_status=identity_status,
        # Canonical JSON keeps the artifact identity stable across an
        # irrelevant trailing newline or indentation-only repack.
        config_sha256=hashlib.sha256(
            json.dumps(
                config,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        outer_model_type=_optional_str(config.get("model_type")),
        text_model_type=_optional_str(text_config.get("model_type")),
        mtp_num_hidden_layers=_optional_int(text_config.get("mtp_num_hidden_layers")),
        geometry=_geometry(text_config),
        quantization=_quantization(config),
        target_weights=_target_weights(snapshot),
        mtp_locator=_mtp_locator(snapshot),
    )


__all__ = [
    "ArtifactProbeError",
    "ArtifactIdentityStatus",
    "CandidateStorage",
    "MTPFileShape",
    "MTPLocatorTruth",
    "QwenArtifactTruth",
    "QwenGeometry",
    "QwenQuantization",
    "TargetWeightLayout",
    "TargetWeights",
    "probe_qwen_artifact",
]
