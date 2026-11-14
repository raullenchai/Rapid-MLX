# SPDX-License-Identifier: Apache-2.0
"""Startup-only mlx-lm adapter validation and content identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


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
    from .model_aliases import resolve_model
    from .utils.tokenizer import _local_snapshot_if_cached

    resolved = _local_snapshot_if_cached(resolve_model(name))
    path = Path(resolved).expanduser()
    return str(path.resolve()) if path.is_dir() else resolved


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
    if _base_identity(config["model"]) != _base_identity(model):
        raise ValueError(
            f"--adapter base mismatch: trained on {config['model']!r}, serving {model!r}"
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


def validate_adapter_checkpoint(model: str) -> None:
    from .spec_decode.mtp.detect import _mtp_num_hidden_layers
    from .utils.tokenizer import _read_model_config_json, _read_num_mtp_layers

    config = _read_model_config_json(model) or {}
    if (
        any(key in config for key in ("vision_config", "audio_config"))
        or config.get("model_type") == "lfm3_vl"
    ):
        raise ValueError("--adapter is unsupported with vision/MLLM or audio models")
    if _read_num_mtp_layers(config) or _mtp_num_hidden_layers(config):
        raise ValueError("--adapter is unsupported with MTP checkpoint heads")
    quant = config.get("quantization_config") or {}
    if "fp8" in str(quant).lower():
        raise ValueError("--adapter is unsupported with fp8 repack checkpoints")


def prepare_adapter_cli(args) -> None:
    """Run before CLI dispatch can branch to an alternate weight loader."""
    if getattr(args, "adapter", None) is None:
        return
    model = getattr(args, "model", None)
    if not model:
        raise ValueError("--adapter requires a text base model")
    validate_adapter(args.adapter, model)
    from .audio.probe import is_audio_model_alias
    from .audio.registry import resolve_local_audio_folder
    from .model_aliases import resolve_profile
    from .models.deepseek_v41_native.artifacts import is_product_target

    if is_product_target(model) or getattr(args, "cua_only", False):
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
    validate_adapter_checkpoint(model)
    # Automatic acceleration profiles must not select an unadapted draft head.
    args.no_spec_decode = True
