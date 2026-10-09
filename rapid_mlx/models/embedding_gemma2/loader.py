# SPDX-License-Identifier: MIT
# Copyright © 2025 Prince Canuma
# Adapted from pinned MLX-VLM encoder_loader.py; see NOTICE.
"""Strict native text encoder loading without changing the vision runtime."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from .config import ModelConfig
from .embedding_gemma2 import Model


def load(model_name: str):
    from mlx_vlm.utils import _quantization_for_module_path, get_model_path

    path = get_model_path(model_name)
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "embedding_gemma2":
        raise ValueError("Expected an embedding_gemma2 embedding checkpoint")
    if config.get("dtype", config.get("torch_dtype")) == "float16":
        raise ValueError("EmbeddingGemma 2 requires BF16 or FP32, never FP16")
    quantization = config.get("quantization")
    quantization_specs = (
        [quantization]
        + [value for value in quantization.values() if isinstance(value, dict)]
        if quantization is not None
        else []
    )
    if any(
        spec.get("mode", "affine") != "affine"
        or spec.get("bits") != 4
        or spec.get("group_size") != 64
        for spec in quantization_specs
    ):
        raise ValueError(
            "EmbeddingGemma 2 supports BF16 or standard affine 4-bit weights"
        )
    model = Model(ModelConfig.from_dict(config))
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        if any(
            Path(name).is_absolute() or ".." in Path(name).parts
            for name in weight_map.values()
        ):
            raise ValueError("Weight shards must belong to the model directory")
        files = [path / name for name in sorted(set(weight_map.values()))]
    else:
        files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors found in {path}")
    weights = {}
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
        weights.update(mx.load(str(file)))
    weights = model.sanitize(weights)
    if any(weight.dtype == mx.float16 for weight in weights.values()):
        raise ValueError("EmbeddingGemma 2 requires BF16 or FP32, never FP16")
    if quantization is not None:

        def predicate(name, module):
            if not hasattr(module, "to_quantized"):
                return False
            override = _quantization_for_module_path(quantization, name, model)
            if override is not None:
                return override
            return f"{name}.scales" in weights

        nn.quantize(
            model, group_size=64, bits=4, mode="affine", class_predicate=predicate
        )
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )
    return model, tokenizer
