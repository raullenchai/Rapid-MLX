# SPDX-License-Identifier: Apache-2.0
"""Shard-at-a-time loading with exact parameter and shape validation."""

from __future__ import annotations

import json
import re
from pathlib import Path


def load_component(
    model, root: Path, *, dtype, key_map=lambda k: k, cancel_check=lambda: None
):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    expected = dict(tree_flatten(model.parameters()))
    seen = set()
    indexes = list(root.glob("*.safetensors.index.json"))
    if indexes:
        weight_map = json.loads(indexes[0].read_text())["weight_map"]
        files = sorted(
            {file for key, file in weight_map.items() if key_map(key) is not None}
        )
    else:
        files = sorted(p.name for p in root.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors weights found for {root.name}")
    for filename in files:
        cancel_check()
        shard = mx.load(str(root / filename))
        converted = []
        for key, value in shard.items():
            name = key_map(key)
            if name is None:
                continue
            if name not in expected or name in seen:
                raise ValueError(f"Unexpected or duplicate {root.name} tensor: {name}")
            if value.ndim == 5 and name.endswith("weight"):
                value = value.transpose(0, 2, 3, 4, 1)
            if value.shape != expected[name].shape:
                raise ValueError(
                    f"Shape mismatch for {root.name}.{name}: {value.shape} != {expected[name].shape}"
                )
            value = value.astype(dtype)
            # Never retain fp32 source shards while loading the next shard.
            mx.eval(value)
            converted.append((name, value))
            seen.add(name)
        model.load_weights(converted, strict=False)
        del shard, converted
        mx.clear_cache()
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"Missing {root.name} tensors: {sorted(missing)}")


def t5_key(key):
    if key in ("shared.weight", "encoder.embed_tokens.weight"):
        # Safetensors removes tied aliases; the released checkpoint uses shared.
        return "wte.weight" if key == "shared.weight" else None
    if key == "encoder.final_layer_norm.weight":
        return "encoder.ln.weight"
    if key == "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight":
        return "encoder.relative_attention_bias.embeddings.weight"
    key = key.replace("encoder.block.", "encoder.layers.")
    key = key.replace(".layer.0.layer_norm.", ".ln1.").replace(
        ".layer.1.layer_norm.", ".ln2."
    )
    key = key.replace(".layer.0.SelfAttention.", ".attention.")
    key = key.replace(".layer.1.DenseReluDense.", ".dense.")
    for old, new in (
        ("q", "query_proj"),
        ("k", "key_proj"),
        ("v", "value_proj"),
        ("o", "out_proj"),
    ):
        key = key.replace(f".attention.{old}.", f".attention.{new}.")
    return key


def qwen_key(key, layers):
    key = key.removeprefix("model.")
    match = re.match(r"layers\.(\d+)\.", key)
    if key == "norm.weight" or (match and int(match[1]) >= layers):
        return None
    return key
