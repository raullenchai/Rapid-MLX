"""Strict local Hy4 loading with bounded materialization and Engram SSD rows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from safetensors import SafetensorError, safe_open

from ..model_load_errors import typed_mlx_load_boundaries
from ..runtime.ubc_evict import ubc_evict
from .engram_offload import DiskQuantizedEngramEmbedding, resolve_indexed_shard
from .hy_v4_flash import Model, ModelArgs


def _engram_specs(config):
    args = ModelArgs.from_dict(config)
    q = config.get("quantization")
    if not isinstance(q, dict):
        raise ValueError("Engram SSD offload requires affine-quantized Engram tables")
    specs = {}
    for layer_id in args.engram_layer_ids:
        prefix = (
            f"model.layers.{layer_id - 1}.engram_layer.multi_head_embedding.embedding"
        )
        entry = q.get(prefix, q)
        if (
            not isinstance(entry, dict)
            or entry.get("mode", q.get("mode", "affine")) != "affine"
        ):
            raise ValueError(
                "Engram SSD offload requires affine-quantized Engram tables"
            )
        bits, group = (
            entry.get("bits", q.get("bits")),
            entry.get("group_size", q.get("group_size")),
        )
        dim = args.engram_n_embed_per_ngram // args.engram_n_head_per_ngram
        if (
            bits not in (2, 3, 4, 6, 8)
            or not isinstance(group, int)
            or group <= 0
            or dim % group
            or dim * bits % 32
        ):
            raise ValueError("Invalid affine Engram dimensions or quantization")
        specs[prefix] = (layer_id - 1, dim, group, bits)
    if not specs:
        raise ValueError("Engram SSD offload requires Engram layers")
    return specs


def _index(path):
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        raise ValueError("Engram SSD offload requires a safetensors index")
    data = json.loads(index.read_text())
    mapping = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Engram SSD offload requires a valid weight map")
    for key, filename in mapping.items():
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Invalid indexed shard for {key}")
        resolve_indexed_shard(str(path), filename)
    return mapping


def supports_engram_ssd_offload(path: str | Path) -> bool:
    """Check metadata and shard containment without loading model parameters."""
    path = Path(path)
    try:
        specs = _engram_specs(json.loads((path / "config.json").read_text()))
        mapping = _index(path)
        for prefix in specs:
            for leaf, dtype in (
                ("weight", "U32"),
                ("scales", "BF16"),
                ("biases", "BF16"),
            ):
                key = f"{prefix}.{leaf}"
                shard = resolve_indexed_shard(str(path), mapping[key])
                with safe_open(str(shard), framework="np") as file:
                    if file.get_slice(key).get_dtype() != dtype:
                        return False
        return True
    except (OSError, TypeError, ValueError, KeyError, SafetensorError):
        return False


def evaluate_weights(items, *, max_bytes=256 << 20, max_tensors=16):
    """Bound each evaluation by bytes and count; an indivisible tensor runs alone."""
    batch, size = [], 0
    for _, value in items:
        if batch and (size + value.nbytes > max_bytes or len(batch) >= max_tensors):
            mx.eval(batch)
            batch, size = [], 0
        batch.append(value)
        size += value.nbytes
    if batch:
        mx.eval(batch)


def load(path: str | Path, *, engram_ssd_offload: bool | None = None):
    """Load local weights strictly; None enables supported offload unless HY4_ENGRAM_SSD_OFFLOAD=0."""
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    if engram_ssd_offload is None:
        engram_ssd_offload = os.environ.get(
            "HY4_ENGRAM_SSD_OFFLOAD", "1"
        ) != "0" and supports_engram_ssd_offload(path)
    mapping = _index(path) if engram_ssd_offload else None
    specs = _engram_specs(config) if engram_ssd_offload else {}
    disk_keys = {
        f"{prefix}.{leaf}"
        for prefix in specs
        for leaf in ("weight", "scales", "biases")
    }
    if mapping is not None and disk_keys - mapping.keys():
        raise ValueError(
            f"Missing indexed Engram tensors: {sorted(disk_keys - mapping.keys())}"
        )
    model = Model(ModelArgs.from_dict(config))
    readers = []
    try:
        for prefix, (layer, dim, group, bits) in specs.items():
            assert mapping is not None
            embedding = model.layers[layer].engram_layer.multi_head_embedding
            reader = DiskQuantizedEngramEmbedding(
                {
                    f"{prefix}.{leaf}": resolve_indexed_shard(
                        str(path), mapping[f"{prefix}.{leaf}"]
                    )
                    for leaf in ("weight", "scales", "biases")
                },
                weight_key=f"{prefix}.weight",
                scales_key=f"{prefix}.scales",
                biases_key=f"{prefix}.biases",
                num_embeddings=embedding.embedding.weight.shape[0],
                dim=dim,
                group_size=group,
                bits=bits,
            )
            readers.append(reader)
            embedding.embedding = reader
        weights = {}
        shards = (
            sorted(path.glob("*.safetensors"))
            if mapping is None
            else sorted({path / name for name in mapping.values()})
        )
        if not shards:
            raise ValueError("No Hy4 safetensors found")
        for shard in shards:
            if mapping is not None:
                indexed_keys = {
                    key for key, name in mapping.items() if path / name == shard
                }
                with safe_open(str(shard), framework="np") as file:
                    if set(file.keys()) != indexed_keys:
                        raise ValueError(
                            "Checkpoint tensors do not match safetensors index"
                        )
            if mapping is not None and all(
                key in disk_keys
                for key, name in mapping.items()
                if path / name == shard
            ):
                continue
            loaded = cast(dict[str, mx.array], mx.load(str(shard), stream=mx.cpu))
            for key, value in loaded.items():
                if mapping is not None and mapping.get(key) != str(
                    shard.relative_to(path)
                ):
                    raise ValueError(f"Tensor does not match safetensors index: {key}")
                if key in disk_keys:
                    continue
                if key in weights:
                    raise ValueError(f"Duplicate Hy4 checkpoint tensor: {key}")
                weights[key] = value
            del loaded
        if mapping is not None and set(mapping) - disk_keys != set(weights):
            raise ValueError("Checkpoint tensors do not match safetensors index")
        weights = model.sanitize(weights)
        q = config.get("quantization")
        if q:
            quantized_paths = {
                key.removesuffix(".scales")
                for key in weights
                if key.endswith(".scales")
            }

            def predicate(prefix, module):
                if prefix in specs or not hasattr(module, "to_quantized"):
                    return False
                return q.get(prefix, prefix in quantized_paths)

            with typed_mlx_load_boundaries():
                nn.quantize(
                    model,
                    group_size=q["group_size"],
                    bits=q["bits"],
                    mode=q.get("mode", "affine"),
                    class_predicate=predicate,
                )
        with typed_mlx_load_boundaries():
            model.load_weights(list(weights.items()), strict=True)
            evaluate_weights(tree_flatten(model.parameters()))
        del weights
        for shard in shards:
            ubc_evict(str(shard))
        model.eval()
        return model, config
    except Exception:
        for reader in readers:
            reader.close()
        raise
