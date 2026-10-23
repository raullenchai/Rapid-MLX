"""Safetensors loading utilities.

Ported from ltx-core/src/ltx_core/loader/sft_loader.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open

from ltx_core_mlx.loader.primitives import StateDict
from ltx_core_mlx.loader.sd_ops import SDOps

_KNOWN_MLX_EXTENSIONS = {".safetensors", ".npz", ".npy", ".gguf"}


def _load_weights(path: str) -> dict[str, mx.array]:
    """Load weights, handling extensionless HF cache blobs via explicit format."""
    if Path(path).suffix in _KNOWN_MLX_EXTENSIONS:
        return mx.load(path)
    return mx.load(path, format="safetensors")


class SafetensorsStateDictLoader:
    """Loads weights from safetensors files with optional key remapping."""

    def metadata(self, path: str) -> dict:
        """Extract metadata from a safetensors file."""
        with safe_open(path, framework="numpy") as f:
            meta = f.metadata()
            return meta if meta else {}

    def load(
        self,
        path: str | list[str],
        sd_ops: SDOps | None = None,
    ) -> StateDict:
        """Load state dict from safetensors file(s) with optional key remapping.

        Args:
            path: Path or list of paths to safetensors files.
            sd_ops: Optional key renaming/filtering operations.

        Returns:
            StateDict with loaded weights.
        """
        sd: dict[str, mx.array] = {}
        size = 0
        dtypes: set[mx.Dtype] = set()

        model_paths = path if isinstance(path, list) else [path]
        for shard_path in model_paths:
            weights = _load_weights(shard_path)
            for name, value in weights.items():
                expected_name = name if sd_ops is None else sd_ops.apply_to_key(name)
                if expected_name is None:
                    continue

                if sd_ops is not None:
                    key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                else:
                    from ltx_core_mlx.loader.sd_ops import KeyValueOperationResult

                    key_value_pairs = [KeyValueOperationResult(expected_name, value)]

                for key, val in key_value_pairs:
                    size += val.nbytes
                    dtypes.add(val.dtype)
                    sd[key] = val

        return StateDict(sd=sd, size=size, dtype=dtypes)


class SafetensorsModelStateDictLoader:
    """Loads weights and configuration metadata from safetensors model files."""

    def __init__(self, weight_loader: SafetensorsStateDictLoader | None = None):
        self.weight_loader = weight_loader if weight_loader is not None else SafetensorsStateDictLoader()

    def metadata(self, path: str) -> dict:
        """Extract model configuration from safetensors metadata."""
        with safe_open(path, framework="numpy") as f:
            meta = f.metadata()
            if meta is None or "config" not in meta:
                return {}
            return json.loads(meta["config"])

    def load(
        self,
        path: str | list[str],
        sd_ops: SDOps | None = None,
    ) -> StateDict:
        """Load state dict, delegating to the weight loader."""
        return self.weight_loader.load(path, sd_ops)
