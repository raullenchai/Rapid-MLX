"""Adaptive Layer Norm.

Ported from ltx-core/src/ltx_core/model/transformer/adaln.py
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.transformer.timestep_embedding import TimestepEmbedder


class AdaLayerNormSingle(nn.Module):
    """Adaptive Layer Norm that produces modulation parameters from timestep.

    Weight keys:
        ``emb.timestep_embedder.linear1.{weight,bias}``
        ``emb.timestep_embedder.linear2.{weight,bias}``
        ``linear.{weight,bias}``

    Args:
        dim: Model dimension (output of timestep MLP and input to linear).
        num_params: Number of modulation parameters to produce.
        timestep_dim: Dimension of the input timestep embedding. Defaults to dim.
    """

    def __init__(self, dim: int, num_params: int = 6, timestep_dim: int | None = None):
        super().__init__()
        self.num_params = num_params
        t_dim = timestep_dim or dim
        self.emb = TimestepEmbedder(t_dim, dim)
        self.linear = nn.Linear(dim, num_params * dim)

    def __call__(self, timestep_emb: mx.array) -> tuple[mx.array, mx.array]:
        """Compute adaptive norm parameters.

        Args:
            timestep_emb: Timestep embedding of shape (B, timestep_dim).

        Returns:
            Tuple of (modulation_params, embedded_timestep) where:
            - modulation_params: shape (B, num_params * dim)
            - embedded_timestep: shape (B, dim) -- intermediate embedding
              used by the output block for adaptive scale/shift.
        """
        embedded = self.emb(timestep_emb)
        params = self.linear(nn.silu(embedded))
        return params, embedded
