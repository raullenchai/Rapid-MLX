# SPDX-License-Identifier: Apache-2.0
"""Startup-only mlx-lm adapter validation and content identity."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

ADAPTER_METADATA = "rapid_mlx_adapter.json"


@dataclass(frozen=True)
class Adapter:
    path: str
    digest: str

    def model_identity(self, base: str) -> str:
        """Default public model ID; operators may override the served name."""
        return f"{base}:lora:{self.digest[:12]}"

    def cache_identity(self, base: str) -> str:
        """Internal namespace retains the complete content digest."""
        return f"{base}:lora:{self.digest}"


def _base_identity(name: str) -> str:
    """Compare Hub bases by repo, and other local bases by absolute path."""
    from huggingface_hub import constants

    from .model_aliases import resolve_model

    # Identity must not depend on the cached revision or external model roots.
    resolved = resolve_model(name, include_external_roots=False)
    path = Path(resolved).expanduser()
    if not path.is_dir():
        return resolved.casefold()
    path = Path(os.path.abspath(path))
    cache = Path(os.path.abspath(Path(constants.HF_HUB_CACHE).expanduser()))
    # Inspect the cache spelling before resolving symlinks: a cached repository
    # may itself be a symlink to storage outside the cache root.
    roots = (cache, cache.resolve())
    for candidate, root in (
        (candidate, root) for candidate in (path, path.resolve()) for root in roots
    ):
        try:
            parts = candidate.relative_to(root).parts
        except ValueError:
            continue
        if len(parts) >= 3 and parts[1] == "snapshots":
            repo = parts[0].split("--")
            if len(repo) == 3 and repo[0] == "models" and all(repo[1:]):
                return "/".join(repo[1:]).casefold()
    return str(path.resolve())


def validate_adapter(path: str, model: str) -> Adapter:
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"--adapter must be an existing directory: {directory}")
    digest = hashlib.sha256()
    for filename in ("adapter_config.json", "adapters.safetensors"):
        source = directory / filename
        if not source.is_file():
            raise ValueError(f"--adapter missing required file: {source}")
        digest.update(filename.encode())
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    try:
        config = json.loads((directory / "adapter_config.json").read_text())
    except (ValueError, OSError) as exc:
        raise ValueError(f"--adapter invalid adapter_config.json: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(config.get("model"), str):
        raise ValueError("--adapter adapter_config.json must declare its base 'model'")
    bases = [config["model"]]
    if (metadata := directory / ADAPTER_METADATA).exists():
        try:
            base = json.loads(metadata.read_text())["base_model"]
            if not isinstance(base, str) or not base:
                raise ValueError("base_model must be a nonempty string")
        except (ValueError, TypeError, KeyError, OSError) as exc:
            raise ValueError(f"--adapter invalid {ADAPTER_METADATA}: {exc}") from exc
        bases.insert(0, base)
    selected_base = _base_identity(model)
    for base in bases:
        if _base_identity(base) != selected_base:
            raise ValueError(
                f"--adapter base mismatch: trained on {base!r}, serving {model!r}"
            )
    if config.get("fine_tune_type", "lora") != "lora":
        raise ValueError("--adapter currently supports only mlx-lm LoRA adapters")
    return Adapter(str(directory), digest.hexdigest())


def reject_adapter_options(
    *, mllm=False, disk_stream=False, speculative=False, lazy=False
):
    for enabled, name in (
        (mllm, "vision/MLLM models"),
        (disk_stream, "disk streaming"),
        (speculative, "MTP/speculative draft heads"),
        (lazy, "lazy loading/idle unloading or resident model replacement"),
    ):
        if enabled:
            raise ValueError(f"--adapter is unsupported with {name}")


class _UninspectableQwen35WeightsError(ValueError):
    pass


def _has_qwen35_draft_tensors(snapshot: Path) -> bool:
    from . import checkpoint_integrity as integrity
    from ._download_gate import _numbered_shards_are_complete
    from .model_metadata import MAX_WEIGHT_INDEX_BYTES, _read_json

    index = (
        _read_json(
            snapshot / integrity.INDEX_FILENAME, max_bytes=MAX_WEIGHT_INDEX_BYTES
        )
        or {}
    )
    weight_map = index.get("weight_map")
    try:
        if isinstance(weight_map, dict) and weight_map:
            if any(
                integrity._safe_shard_name(shard) is None
                for shard in weight_map.values()
            ) or integrity.incomplete_checkpoint_reason(snapshot):
                raise integrity._UnreadableHeaderError
            # Shard headers can contain tensors omitted from the index.
            shards = set(snapshot.glob("model*.safetensors"))
            shards.update(snapshot / shard for shard in weight_map.values())
        else:
            if _numbered_shards_are_complete(str(snapshot)) is False:
                raise integrity._UnreadableHeaderError
            shards = set(snapshot.glob("*.safetensors"))
        names = [name for shard in shards for name in integrity._read_header(shard)]
        if not names:
            raise integrity._UnreadableHeaderError
    except (OSError, integrity._UnreadableHeaderError) as exc:
        raise _UninspectableQwen35WeightsError(
            "--adapter cannot inspect Qwen3.5 checkpoint weights"
        ) from exc
    return any(part.startswith("mtp") for name in names for part in name.split("."))


def validate_adapter_checkpoint(
    model: str,
    *,
    strict: bool = False,
    require_weights: bool = False,
    config_only: bool = False,
) -> bool:
    """Check adapter compatibility and return whether this is dense Qwen3.5.

    CLI routing checks config only and can defer when it is unavailable. Strict
    routing needs config. The final loader requires a concrete directory with
    inspectable Qwen3.5 weights.
    """
    from .model_aliases import resolve_model, resolve_revision
    from .model_metadata import read_model_metadata

    resolved = model
    metadata = None
    inspection_error: Exception | None = None
    try:
        resolved = resolve_model(model)
        reference = model if resolve_revision(model) else resolved
        metadata = read_model_metadata(
            resolved if Path(resolved).is_dir() else reference
        )
    except Exception as exc:
        inspection_error = exc
    config = metadata.config if metadata else None
    snapshot = metadata.snapshot_dir if metadata else None
    if not config:
        try:
            config = _read_checkpoint_config(
                str(snapshot) if snapshot else resolved, model
            )
        except Exception as exc:
            inspection_error = exc
    if not config:
        if strict or require_weights:
            raise ValueError(
                f"--adapter cannot inspect config.json for {model!r}. "
                "Restore the checkpoint's config.json or download the base model again."
            ) from inspection_error
        return False
    if config_only and not require_weights:
        return _validate_checkpoint_config(config, None)
    local_path = Path(resolved)
    is_local = local_path.is_dir()
    snapshot = snapshot or (local_path if is_local else None)
    if require_weights and not Path(model).is_dir():
        # The loader must consume the same directory we inspect.
        snapshot = None
    try:
        qwen35 = _validate_checkpoint_config(config, snapshot)
    except _UninspectableQwen35WeightsError:
        if require_weights or is_local:
            raise
        # A Hub snapshot may contain only prefetched metadata.
        return strict and _validate_checkpoint_config(config, None)
    if qwen35 and snapshot is None:
        if require_weights:
            raise _UninspectableQwen35WeightsError(
                "--adapter cannot inspect Qwen3.5 checkpoint weights"
            )
        return strict
    return qwen35


def _read_checkpoint_config(model: str, reference: str) -> dict | None:
    from . import model_downloads
    from .model_aliases import resolve_revision, resolve_subfolder
    from .model_metadata import _read_json

    path = Path(model)
    if path.is_dir():
        return _read_json(path / "config.json")
    subfolder = resolve_subfolder(reference)
    filename = f"{subfolder}/config.json" if subfolder else "config.json"
    repo = reference if resolve_revision(reference) else model
    try:
        config = model_downloads.hf_hub_download(repo, filename, local_files_only=True)
    except Exception:
        config = model_downloads.hf_hub_download(repo, filename)
    return _read_json(Path(config))


def _validate_checkpoint_config(config: dict, snapshot: Path | None) -> bool:
    from .spec_decode.mtp.detect import _mtp_num_hidden_layers

    qwen35 = config.get("model_type") == "qwen3_5"
    if (
        "audio_config" in config
        or ("vision_config" in config and not qwen35)
        or config.get("model_type") == "lfm3_vl"
    ):
        raise ValueError("--adapter is unsupported with vision/MLLM or audio models")
    # Converted dense Qwen3.5 packs can retain MTP config without draft weights.
    if (
        _read_num_mtp_layers(config)
        or (not qwen35 and _mtp_num_hidden_layers(config))
        or (qwen35 and snapshot is not None and _has_qwen35_draft_tensors(snapshot))
    ):
        raise ValueError("--adapter is unsupported with MTP checkpoint heads")
    quant = config.get("quantization_config") or {}
    if "fp8" in str(quant).lower():
        raise ValueError("--adapter is unsupported with fp8 repack checkpoints")
    return qwen35


def _read_num_mtp_layers(config: dict[str, object]) -> int:
    """Read num_nextn_predict_layers from the root or nested text config."""
    count = config.get("num_nextn_predict_layers")
    if type(count) is int and count:
        return count
    if isinstance(text_config := config.get("text_config"), dict):
        count = text_config.get("num_nextn_predict_layers")
    return count if type(count) is int else 0


def validate_adapter_runtime(model: str, *, cua_only: bool = False) -> None:
    """Reject alternate model runtimes that cannot serve mlx-lm adapters."""
    from .audio.probe import is_audio_model_alias
    from .audio.registry import resolve_local_audio_folder
    from .model_aliases import resolve_profile
    from .models.deepseek_v41_native.artifacts import is_product_target

    if is_product_target(model) or cua_only:
        raise ValueError(
            "--adapter is unsupported with dedicated native/media runtimes"
        )
    profile = resolve_profile(model)
    if (
        (profile and profile.modality != "text")
        or is_audio_model_alias(model)
        or resolve_local_audio_folder(model)
    ):
        raise ValueError(
            "--adapter supports text models only (no image/video/audio/diffusion)"
        )


def prepare_adapter_cli(args) -> None:
    """Run before CLI dispatch can branch to an alternate weight loader."""
    if getattr(args, "adapter", None) is None:
        return
    model = getattr(args, "model", None)
    if not model:
        raise ValueError("--adapter requires a text base model")
    validate_adapter(args.adapter, model)
    validate_adapter_runtime(model, cua_only=getattr(args, "cua_only", False))
    payload = getattr(args, "speculative_config", None)
    reject_adapter_options(
        mllm=getattr(args, "mllm", False),
        disk_stream=getattr(args, "disk_stream", False),
        speculative=bool(payload)
        or any(
            getattr(args, key, False)
            for key in (
                "enable_mtp",
                "enable_dflash",
                "enable_ddtree",
                "mtp_sidecar",
                "dflash_drafter_path",
                "force_spec_decode",
            )
        )
        or getattr(args, "spec_decode", "none") not in (None, "none"),
        lazy=any(
            getattr(args, key, False)
            for key in (
                "lazy_load",
                "idle_unload_seconds",
                "resident_model",
                "resident_models",
            )
        ),
    )
    if validate_adapter_checkpoint(model, config_only=True) and not getattr(
        args, "no_mllm", False
    ):
        args.no_mllm = True
        print("Serving text-only because --adapter is set; image inputs are disabled")
    # Automatic acceleration profiles must not select an unadapted draft head.
    args.no_spec_decode = True
