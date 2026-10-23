"""MLP (feed-forward) blocks.

Ported from ltx-core/src/ltx_core/model/transformer/feed_forward.py

Weight keys (relative to parent ``ff`` or ``audio_ff``):
    ``proj_in.{weight,bias}``  -- input projection with GELU (tanh approx)
    ``proj_out.{weight,bias}`` -- output projection
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class FeedForward(nn.Module):
    """Feed-forward network with GELU (tanh approx) activation.

    Architecture: Linear -> GELU_approx -> Linear

    Weight keys: ``proj_in.{weight,bias}``, ``proj_out.{weight,bias}``
    """

    def __init__(self, dim: int, dim_out: int | None = None, mult: float = 4.0):
        super().__init__()
        dim_out = dim_out or dim
        inner_dim = int(dim * mult)
        self.proj_in = nn.Linear(dim, inner_dim)
        self.proj_out = nn.Linear(inner_dim, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj_out(nn.gelu_approx(self.proj_in(x)))
