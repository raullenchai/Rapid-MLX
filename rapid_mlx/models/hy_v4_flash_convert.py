# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 conversion through the standard import worker and MLX saver."""

import json
import logging
from functools import partial
from pathlib import Path

import numpy as np

from .hy_v4_flash_config import checkpoint_key, dequantize_fp8, quantization_policy


def convert_fp8(source, destination, bits, group_size):
    """Return False for ordinary checkpoints; consume raw FP8 before mx.load."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_tokenizer, quantize_model, save

    from ..fp8_repack import _open_shards, _raw_to_mx
    from .hy_v4_flash import Model, ModelArgs

    source, destination = Path(source), Path(destination)
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "hy_v4_flash":
        return False
    weight_map, shards = _open_shards(source)
    if not any(
        entry["dtype"] == "F8_E4M3"
        for shard in shards.values()
        for entry in shard.header.values()
    ):
        return False
    if destination.exists():
        raise ValueError("Conversion destination already exists")
    model = Model(ModelArgs.from_dict(config))
    block = tuple(
        config.get("quantization_config", {}).get("weight_block_size", [128, 128])
    )
    config.pop("quantization_config", None)
    config.get("text_config", {}).pop("quantization_config", None)
    model, config = quantize_model(
        model,
        config,
        32,
        bits,
        quant_predicate=partial(quantization_policy, group_size=group_size, bits=bits),
    )
    expected = dict(tree_flatten(model.parameters()))
    result = {}

    def read(name):
        raw, dtype, shape = shards[weight_map[name]].read(name)
        if dtype in ("I64", "I32"):
            return mx.array(
                raw.view(np.int64 if dtype == "I64" else np.int32).reshape(shape)
            )
        return _raw_to_mx(raw, dtype, shape)

    if any(checkpoint_key(name) is None for name in weight_map):
        logging.getLogger(__name__).info(
            "Hy4 Flash text-only: skipping vision tower and dfly_layers weights"
        )
    for name in weight_map:
        target = checkpoint_key(name)
        if target is None or name.endswith("_scale_inv"):
            continue
        raw, dtype, shape = shards[weight_map[name]].read(name)
        if dtype == "F8_E4M3":
            scale = np.array(read(name + "_scale_inv").astype(mx.float32))
            value = mx.array(dequantize_fp8(raw.reshape(shape), scale, block))
        else:
            value = read(name)
        # Sanitize before quantization, including the packed expert gate/up split.
        mapped = model.sanitize({name: value})
        for key, value in mapped.items():
            path = key.removesuffix(".weight")
            if key.endswith(".weight") and path + ".scales" in expected:
                params = config["quantization"].get(path, config["quantization"])
                packed, scales, biases = mx.quantize(
                    value, group_size=params["group_size"], bits=params["bits"]
                )
                result[key], result[path + ".scales"], result[path + ".biases"] = (
                    packed,
                    scales,
                    biases,
                )
                mx.eval(packed, scales, biases)
            else:
                result[key] = value
                mx.eval(value)
    model.load_weights(list(result.items()), strict=True)
    model.eval()
    tokenizer = load_tokenizer(source)
    save(destination, source, model, tokenizer, config)
    return True
