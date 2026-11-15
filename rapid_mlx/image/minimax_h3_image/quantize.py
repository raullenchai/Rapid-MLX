"""Strict, incremental local-weight loading and selective affine quantization.

Only DiT block/refiner Linear modules and Qwen language decoder Linear modules
are eligible. Each Linear is quantized as soon as its last tensor is loaded,
including across shard boundaries. Unloaded random parameters stay lazy; only
one safetensors shard mapping is open at a time. No full BF16 model is evaluated.
The text encoder uses this same loader, after mlx-vlm's layer-50 selection and
key sanitization; embeddings and the vision tower retain checkpoint precision.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def validate_bits(bits: int | None) -> None:
    if bits not in (None, 4, 8):
        raise ValueError("quantize must be 4, 8, or None")


def quantization_predicate(path: str, module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) and path.startswith(
        (
            "transformer_blocks.",
            "token_refiner.refiner_blocks.",
            "language_model.model.layers.",
        )
    )


def stream_load(
    model: nn.Module,
    directory: Path,
    *,
    bits: int | None = None,
    sanitize: Callable[[dict[str, mx.array]], dict[str, mx.array]] | None = None,
    select: Callable[[str], bool] | None = None,
    cast: Callable[[str, mx.array], mx.array] | None = None,
) -> None:
    """Consume every selected key exactly once; require every parameter and shape.

    ``select`` is only for the explicitly trimmed text encoder. Other components
    accept no ignored checkpoint tensors. Quantization is delayed until both
    weight and bias (when present) have arrived, then evaluated before proceeding.
    """
    validate_bits(bits)
    expected = dict(tree_flatten(model.parameters()))
    eligible = (
        {
            path: {f"{path}.{key}" for key in module.parameters()}
            for path, module in model.named_modules()
            if quantization_predicate(path, module)
        }
        if bits
        else {}
    )
    remaining = {path: set(keys) for path, keys in eligible.items()}
    owner = {key: path for path, keys in eligible.items() for key in keys}
    seen: set[str] = set()
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {directory}")
    index_files = sorted(directory.glob("*.safetensors.index.json"))
    if len(index_files) > 1:
        raise ValueError(f"multiple checkpoint indexes in {directory}")
    weight_map = (
        json.loads(index_files[0].read_text())["weight_map"] if index_files else None
    )
    source_seen: set[str] = set()
    for file in files:
        shard = mx.load(str(file))
        for source_key in list(shard):
            if source_key in source_seen:
                raise ValueError(f"duplicate checkpoint tensor: {source_key}")
            source_seen.add(source_key)
            if weight_map is not None and weight_map.get(source_key) != file.name:
                raise ValueError(f"checkpoint index disagrees for {source_key}")
            value = shard.pop(source_key)
            if select is not None and not select(source_key):
                del value
                continue
            tensors = sanitize({source_key: value}) if sanitize else {source_key: value}
            for key, tensor in tensors.items():
                if key not in expected or key in seen:
                    raise ValueError(f"unexpected or duplicate parameter: {key}")
                if tensor.shape != expected[key].shape:
                    raise ValueError(
                        f"shape mismatch for {key}: {tensor.shape} != {expected[key].shape}"
                    )
                # Drop the initial lazy tensor so it cannot retain its random graph.
                del expected[key]
                tensor = cast(key, tensor) if cast else tensor
                model.load_weights([(key, tensor)], strict=False)
                seen.add(key)
                if key in owner:
                    path = owner[key]
                    remaining[path].remove(key)
                    if not remaining[path]:
                        nn.quantize(
                            model,
                            group_size=64,
                            bits=bits,
                            class_predicate=lambda p, m, target=path: (
                                p == target and isinstance(m, nn.Linear)
                            ),
                        )
                        module = dict(model.named_modules())[path]
                        mx.eval(module.parameters())
                else:
                    mx.eval(tensor)
            del tensors, value, tensor
        del shard
        mx.clear_cache()
    if weight_map is not None and source_seen != set(weight_map):
        raise ValueError("checkpoint index contains missing tensors")
    if expected:
        raise ValueError(f"missing checkpoint parameters: {sorted(expected)}")
    model.eval()
