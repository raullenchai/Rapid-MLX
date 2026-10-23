"""Timestep embedding re-exports + LTX-specific wrapper."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_arsenal.diffusion import TimestepEmbedding, get_timestep_embedding

__all__ = ["TimestepEmbedder", "TimestepEmbedding", "get_timestep_embedding"]


class TimestepEmbedder(nn.Module):
    """Container matching weight key ``emb.timestep_embedder.*``.

    Weight keys: ``timestep_embedder.linear1.*``, ``timestep_embedder.linear2.*``.
    """

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(in_channels, time_embed_dim)

    def __call__(self, sample: mx.array) -> mx.array:
        return self.timestep_embedder(sample)
