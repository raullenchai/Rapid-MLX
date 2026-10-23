"""Normalization functions for Video VAE.

Ported from ltx-core/src/ltx_core/model/video_vae/normalization.py
"""

from __future__ import annotations

import mlx.core as mx


def pixel_norm(x: mx.array, eps: float = 1e-8) -> mx.array:
    """PixelNorm: x / sqrt(mean(x^2, dim=channels) + eps).

    No learnable parameters -- matches the reference VAE's PixelNorm.
    Applied per-pixel across the channel dimension (last dim in BFHWC).
    """
    return mx.fast.rms_norm(x, weight=None, eps=eps)
