# SPDX-License-Identifier: Apache-2.0
"""SAM-Audio adapters that never enter dependency-owned pickle fallbacks."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path


def _snapshot(name: str) -> Path:
    path = Path(name).expanduser()
    if path.is_dir():
        return path
    from rapid_mlx.model_downloads import snapshot_download

    # Resolve once; config, weights and tokenizer subsequently use this snapshot.
    # Legacy tensor archives use only the patched restricted loader.
    return Path(
        snapshot_download(
            name,
            allow_patterns=[
                "*.safetensors",
                "checkpoint.pt",
                "pytorch_model.bin",
                "*.json",
                "*.model",
                "*.txt",
            ],
        )
    )


def _config(path: Path) -> dict:
    with (path / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("SAM-Audio configuration must contain a JSON object")
    return config


def _require_weights(path: Path, *, text_encoder: bool = False) -> list[Path]:
    weights = (
        [path / "model.safetensors"]
        if text_encoder
        else sorted(path.glob("*.safetensors"))
    )
    if weights and all(weight.is_file() for weight in weights):
        return weights
    legacy = path / ("pytorch_model.bin" if text_encoder else "checkpoint.pt")
    if legacy.is_file():
        return [legacy]
    raise ValueError(f"SAM-Audio at {path} contains no supported tensor checkpoint")


def _tensor_weights(path: Path):
    import mlx.core as mx

    if path.suffix == ".safetensors":
        return mx.load(str(path))
    import torch

    from rapid_mlx.utils.torch_checkpoint import load_tensor_checkpoint

    weights = load_tensor_checkpoint(path)
    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]
    if not isinstance(weights, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in weights.items()
    ):
        raise ValueError("SAM-Audio legacy checkpoint must contain only named tensors")
    return {
        key: mx.array(
            value.detach().cpu().float().numpy()
            if value.dtype == torch.bfloat16
            else value.detach().cpu().numpy()
        )
        for key, value in weights.items()
    }


def _load_t5(name: str):
    """Load native T5 architecture from tensor data, never upstream torch.load."""
    import mlx.core as mx
    from mlx_audio.sts.models.sam_audio.text_encoder import T5Config, T5Encoder
    from transformers import AutoTokenizer

    path = _snapshot(name)
    weights_path = _require_weights(path, text_encoder=True)[0]
    config_data = _config(path)
    # Match upstream from_pretrained's HF defaults (these two differ from
    # T5Config's native defaults) while keeping its architecture implementation.
    defaults = T5Config(is_gated_act=False, dense_act_fn="relu")
    config = T5Config(
        **{
            field.name: config_data.get(field.name, getattr(defaults, field.name))
            for field in fields(T5Config)
        }
    )
    weights = _tensor_weights(weights_path)
    model = T5Encoder(config)
    model.load_weights(list(T5Encoder.sanitize(weights).items()))
    mx.eval(model.parameters())
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=False, local_files_only=True
    )
    return model, tokenizer


def load_sam_audio(name: str):
    """Construct SAM with owned tensor loaders and an instance-local safe T5 path."""
    from mlx_audio.sts import SAMAudio
    from mlx_audio.sts.models.sam_audio.config import SAMAudioConfig
    from mlx_audio.sts.models.sam_audio.model import _load_weights
    from mlx_audio.sts.models.sam_audio.text_encoder import T5TextEncoder

    path = _snapshot(name)
    files = _require_weights(path)
    weights = {}
    for file in files:
        weights.update(_tensor_weights(file))
    config = SAMAudioConfig.from_dict(_config(path))

    class TensorOnlyTextEncoder(T5TextEncoder):
        def _lazy_load(self):
            if self.model is None:
                model, tokenizer = _load_t5(self.config.name)
                self.model, self.tokenizer = model, tokenizer

    model = SAMAudio(config)
    model.text_encoder = TensorOnlyTextEncoder(config.text_encoder)
    model = model.post_load_hook(path)
    model = _load_weights(model, weights)
    return model, model.processor
