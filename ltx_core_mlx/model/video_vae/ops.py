"""Per-channel statistics and weight remapping for Video VAE.

Ported from ltx-core/src/ltx_core/model/video_vae/ops.py
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class PerChannelStatistics(nn.Module):
    """Stores per-channel mean and std for latent (de)normalization.

    Produces keys: ``mean``, ``std``
    """

    def __init__(self, channels: int):
        super().__init__()
        self.mean = mx.zeros((channels,))
        self.std = mx.ones((channels,))


class EncoderPerChannelStatistics(nn.Module):
    """Stores per-channel normalization stats for the encoder.

    Weight file keys use underscore prefix (``_mean_of_means``, ``_std_of_means``)
    but MLX nn.Module ignores underscore-prefixed attributes. We store them
    without the underscore and remap during weight loading.

    Produces keys: ``mean_of_means``, ``std_of_means``
    """

    def __init__(self, channels: int):
        super().__init__()
        self.mean_of_means = mx.zeros((channels,))
        self.std_of_means = mx.ones((channels,))


# Weight key remapping: safetensors key -> module key
_ENCODER_KEY_REMAP: dict[str, str] = {
    "per_channel_statistics._mean_of_means": "per_channel_statistics.mean_of_means",
    "per_channel_statistics._std_of_means": "per_channel_statistics.std_of_means",
}


def remap_encoder_weight_keys(
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    """Remap weight keys from safetensors format to module format.

    Handles the underscore-prefixed ``_mean_of_means`` / ``_std_of_means`` keys
    in the encoder's per-channel statistics.

    Args:
        weights: Weight dict with keys from the safetensors file.

    Returns:
        New dict with remapped keys.
    """
    remapped: dict[str, mx.array] = {}
    for k, v in weights.items():
        new_key = _ENCODER_KEY_REMAP.get(k, k)
        remapped[new_key] = v
    return remapped
