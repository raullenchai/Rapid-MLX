"""Utility classes for CogVideoX-Fun MLX port."""

import json
import os
from pathlib import Path
from typing import Optional

import mlx.core as mx


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution parameterized by mean and logvar.

    Used by the VAE to represent the latent distribution.
    """

    def __init__(self, parameters: mx.array):
        self.mean, self.logvar = mx.split(parameters, 2, axis=-1)
        self.logvar = mx.clip(self.logvar, -30.0, 20.0)
        self.std = mx.exp(0.5 * self.logvar)

    def sample(self) -> mx.array:
        return self.mean + self.std * mx.random.normal(self.mean.shape)

    def mode(self) -> mx.array:
        return self.mean


def load_config(path: str) -> dict:
    """Load model config from a JSON file."""
    config_file = os.path.join(path, "config.json")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"{config_file} does not exist")
    with open(config_file, "r") as f:
        return json.load(f)


def load_weights(path: str) -> dict:
    """Load safetensors weights from a directory, handling sharded files."""

    p = Path(path)
    safetensors_files = sorted(p.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {path}")

    weights = {}
    for f in safetensors_files:
        w = mx.load(str(f))
        weights.update(w)
    return weights


def transpose_conv_weight(w: mx.array) -> mx.array:
    """Transpose a conv weight from PyTorch to MLX format.

    PyTorch Conv3d: (O, I, kD, kH, kW) -> MLX: (O, kD, kH, kW, I)
    PyTorch Conv2d: (O, I, kH, kW) -> MLX: (O, kH, kW, I)
    PyTorch Conv1d: (O, I, K) -> MLX: (O, K, I)
    """
    if w.ndim == 5:
        return w.transpose(0, 2, 3, 4, 1)
    elif w.ndim == 4:
        return w.transpose(0, 2, 3, 1)
    elif w.ndim == 3:
        return w.transpose(0, 2, 1)
    return w


def _is_conv_weight(key: str, weight: mx.array) -> bool:
    """Heuristic: a weight is a conv weight if it ends with .weight and has 3-5 dims
    and is not a linear/embedding (which are 2D)."""
    if not key.endswith(".weight"):
        return False
    return weight.ndim >= 3


def convert_pytorch_weights(weights: dict) -> dict:
    """Convert PyTorch weights to MLX format (transpose conv weights)."""
    converted = {}
    for key, w in weights.items():
        if _is_conv_weight(key, w):
            converted[key] = transpose_conv_weight(w)
        else:
            converted[key] = w
    return converted


def load_mlx_weights(path: str, component: str) -> dict:
    """Load weights from an mlx-forge converted model directory.

    Handles:
    - Single-file format: {component}.safetensors (with "{component}." prefix in keys)
    - Subdirectory format: {component}/diffusion_pytorch_model.safetensors (no prefix)

    Returns weights with the component prefix stripped.
    """
    p = Path(path)

    # Try mlx-forge flat format first: transformer.safetensors, vae.safetensors
    flat_file = p / f"{component}.safetensors"
    if flat_file.exists():
        weights = mx.load(str(flat_file))
        # Strip component prefix
        prefix = f"{component}."
        return {k.removeprefix(prefix): v for k, v in weights.items()}

    # Try subdirectory format (PyTorch/HuggingFace)
    sub_dir = p / component
    if sub_dir.is_dir():
        shard_files = sorted(sub_dir.glob("*.safetensors"))
        if shard_files:
            weights = {}
            for f in shard_files:
                weights.update(mx.load(str(f)))
            # These are PyTorch weights — need transposition
            return convert_pytorch_weights(weights)

    raise FileNotFoundError(f"No weights for '{component}' found in {path}")


def get_quantize_config(path: str) -> Optional[dict]:
    """Read quantize_config.json if present. Returns None if not quantized."""
    config_file = os.path.join(path, "quantize_config.json")
    if not os.path.exists(config_file):
        return None
    with open(config_file) as f:
        return json.load(f)


def quantize_model_from_weights(model, weights: dict, path: str, component: str):
    """Replace Linear layers with QuantizedLinear where weights have .scales keys.

    Must be called BEFORE load_weights so that QuantizedLinear layers
    are in place to receive the quantized weight format.
    """
    import mlx.nn as nn

    qconfig = get_quantize_config(path)
    if qconfig is not None:
        q = qconfig.get("quantization", {})
        if component in q.get("skip_components", []):
            return
        bits = q.get("bits", 4)
        group_size = q.get("group_size", 64)
    else:
        # No quantize_config.json but the weights carry .scales keys (the
        # caller checked): bits/group_size are fully determined by the
        # shapes — for a Linear of in_dim I, scales.shape[-1] = I/group_size
        # and the packed weight.shape[-1] = I*bits/32. Silently returning
        # here used to leave the Linears unconverted and crash 3 layers
        # deep in an addmm shape error.
        bits = group_size = None

    # Find which layers are quantized by looking for .scales keys
    quantized_paths = set()
    for key in weights:
        if key.endswith(".scales"):
            quantized_paths.add(key.removesuffix(".scales"))

    # Replace Linear with QuantizedLinear for each quantized path
    def _set_nested(obj, path_parts, value):
        for part in path_parts[:-1]:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        last = path_parts[-1]
        if last.isdigit():
            obj[int(last)] = value
        else:
            setattr(obj, last, value)

    def _get_nested(obj, path_parts):
        for part in path_parts:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    for qpath in quantized_paths:
        parts = qpath.split(".")
        try:
            linear = _get_nested(model, parts)
        except (AttributeError, IndexError, TypeError):
            continue
        if not isinstance(linear, nn.Linear):
            continue

        in_dim = linear.weight.shape[1]
        out_dim = linear.weight.shape[0]
        if bits is None:
            layer_group = in_dim // weights[qpath + ".scales"].shape[-1]
            layer_bits = weights[qpath + ".weight"].shape[-1] * 32 // in_dim
        else:
            layer_group, layer_bits = group_size, bits
        has_bias = hasattr(linear, "bias") and linear.bias is not None
        ql = nn.QuantizedLinear(in_dim, out_dim, bias=has_bias, group_size=layer_group, bits=layer_bits)
        _set_nested(model, parts, ql)
