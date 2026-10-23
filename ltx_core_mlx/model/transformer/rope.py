"""Rotary Position Embeddings (RoPE) — SPLIT type.

Ported from ltx-core/src/ltx_core/model/transformer/rope.py

The reference uses log-spaced frequency indices with fractional
positions, NOT standard RoPE with 1/theta^k.
"""

from __future__ import annotations

import math
from enum import Enum

import mlx.core as mx


class RoPEType(Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"


def generate_freq_grid(
    theta: float,
    num_pos_dims: int,
    inner_dim: int,
) -> mx.array:
    """Generate log-spaced frequency indices for RoPE.

    Reference: ltx-core rope.py generate_freq_grid_pytorch

    Args:
        theta: Base frequency (default 10000.0).
        num_pos_dims: Number of position dimensions (3 for video, 1 for audio).
        inner_dim: Total attention dimension (heads * head_dim).

    Returns:
        Frequency indices of shape (inner_dim // (2 * num_pos_dims),).
    """
    n_elem = 2 * num_pos_dims
    num_freqs = inner_dim // n_elem

    indices = theta ** mx.linspace(
        math.log(1.0) / math.log(theta),
        math.log(theta) / math.log(theta),
        num_freqs,
    ).astype(mx.float32)

    return indices * (math.pi / 2.0)


def compute_freqs(
    freq_indices: mx.array,
    positions: mx.array,
    max_pos: list[int],
) -> mx.array:
    """Compute RoPE frequency angles from positions and freq grid.

    Reference: ltx-core rope.py generate_freqs + get_fractional_positions

    Args:
        freq_indices: (num_freqs,) from generate_freq_grid.
        positions: (B, N, num_pos_dims) integer position indices.
        max_pos: Maximum position per axis for normalization.

    Returns:
        Frequency angles of shape (B, N, num_freqs * num_pos_dims).
    """
    num_pos_dims = positions.shape[-1]

    # Fractional positions: pos / max_pos -> [0, 1]
    frac_positions = mx.stack(
        [positions[:, :, i].astype(mx.float32) / max_pos[i] for i in range(num_pos_dims)],
        axis=-1,
    )  # (B, N, num_pos_dims)

    # Scale to [-1, 1] and multiply with freq indices
    # (B, N, num_pos_dims, 1) * (num_freqs,) -> (B, N, num_pos_dims, num_freqs)
    scaled = freq_indices * (frac_positions[..., None] * 2.0 - 1.0)

    # Transpose last two dims and flatten: (B, N, num_freqs, num_pos_dims) -> (B, N, num_freqs * num_pos_dims)
    freqs = scaled.transpose(0, 1, 3, 2).reshape(positions.shape[0], positions.shape[1], -1)
    return freqs


def precompute_rope_freqs(
    positions: mx.array,
    inner_dim: int,
    num_heads: int,
    theta: float = 10000.0,
    max_pos: list[int] | None = None,
    rope_type: str = "split",
) -> tuple[mx.array, mx.array, str]:
    """Precompute per-head RoPE cos/sin frequencies.

    Reference: ltx-core rope.py precompute_freqs_cis

    Args:
        positions: (B, N, num_pos_dims) position indices.
        inner_dim: Total inner dimension (heads * head_dim).
        num_heads: Number of attention heads.
        theta: Base frequency.
        max_pos: Maximum positions per axis. Defaults to [20, 2048, 2048].
        rope_type: ``"split"`` (default, LTX-2.3 main transformer + connector)
            or ``"interleaved"`` (legacy mode, kept for completeness).
            Upstream switched the default from ``"interleaved"`` to ``"split"``
            in PR #212 since all production checkpoints use SPLIT.

    Returns:
        Tuple of (cos_freqs, sin_freqs, rope_type).
    """
    num_pos_dims = positions.shape[-1]
    if max_pos is None:
        max_pos = [20, 2048, 2048][:num_pos_dims]

    freq_indices = generate_freq_grid(theta, num_pos_dims, inner_dim)
    freqs = compute_freqs(freq_indices, positions, max_pos)
    B, N, num_freqs = freqs.shape

    if rope_type == "interleaved":
        # INTERLEAVED: repeat_interleave cos/sin, then pad to inner_dim
        cos_f = mx.cos(freqs)
        sin_f = mx.sin(freqs)
        # Repeat each freq for the pair: (B, N, num_freqs) -> (B, N, 2*num_freqs)
        cos_f = mx.repeat(cos_f, 2, axis=-1)
        sin_f = mx.repeat(sin_f, 2, axis=-1)

        pad_size = inner_dim - cos_f.shape[-1]
        if pad_size > 0:
            cos_pad = mx.ones((*cos_f.shape[:-1], pad_size))
            sin_pad = mx.zeros((*sin_f.shape[:-1], pad_size))
            cos_f = mx.concatenate([cos_pad, cos_f], axis=-1)
            sin_f = mx.concatenate([sin_pad, sin_f], axis=-1)

        # Reshape to per-head: (B, N, inner_dim) -> (B, N, H, head_dim) -> (B, H, N, head_dim)
        head_dim = inner_dim // num_heads
        cos_f = cos_f.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
        sin_f = sin_f.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
        return cos_f, sin_f, rope_type

    else:  # split
        # SPLIT: pad raw freqs to inner_dim//2, compute cos/sin
        expected = inner_dim // 2
        pad_size = expected - num_freqs
        if pad_size > 0:
            padding = mx.zeros((*freqs.shape[:-1], pad_size))
            freqs = mx.concatenate([padding, freqs], axis=-1)

        cos_f = mx.cos(freqs)
        sin_f = mx.sin(freqs)
        head_dim_half = inner_dim // (2 * num_heads)
        cos_f = cos_f.reshape(B, N, num_heads, head_dim_half).transpose(0, 2, 1, 3)
        sin_f = sin_f.reshape(B, N, num_heads, head_dim_half).transpose(0, 2, 1, 3)
        return cos_f, sin_f, rope_type


@mx.compile
def apply_rope_interleaved(
    x: mx.array,
    cos_freqs: mx.array,
    sin_freqs: mx.array,
) -> mx.array:
    """Apply RoPE with INTERLEAVED layout: adjacent pairs rotated together.

    Reference: ltx-core rope.py apply_interleaved_rotary_emb

    For each pair (x[2i], x[2i+1]), applies:
        out[2i]   = x[2i]   * cos - x[2i+1] * sin
        out[2i+1] = x[2i+1] * cos + x[2i]   * sin

    Args:
        x: (B, H, N, dim).
        cos_freqs: (B, H, N, dim) pre-computed cos values.
        sin_freqs: (B, H, N, dim) pre-computed sin values.
    """
    cos_f = cos_freqs.astype(x.dtype)
    sin_f = sin_freqs.astype(x.dtype)
    # Build rotated version: swap pairs with sign change
    # [..., (x0, x1, x2, x3, ...)] -> [..., (-x1, x0, -x3, x2, ...)]
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)  # (..., dim//2, 2)
    x1 = x_pairs[..., 0]  # even indices
    x2 = x_pairs[..., 1]  # odd indices
    x_rot = mx.stack([-x2, x1], axis=-1).reshape(x.shape)
    return x * cos_f + x_rot * sin_f


@mx.compile
def apply_rope_split(
    x: mx.array,
    cos_freqs: mx.array,
    sin_freqs: mx.array,
) -> mx.array:
    """Apply RoPE with SPLIT layout: first half cos, second half sin.

    Reference: ltx-core rope.py apply_split_rotary_emb

    Args:
        x: (B, H, N, dim).
        cos_freqs: (B, H, N, dim//2) pre-computed cos values.
        sin_freqs: (B, H, N, dim//2) pre-computed sin values.
    """
    cos_f = cos_freqs.astype(x.dtype)
    sin_f = sin_freqs.astype(x.dtype)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], axis=-1)


def get_frequencies(
    positions: mx.array,
    dim: int,
    theta: float = 10000.0,
) -> mx.array:
    """Standard RoPE frequencies (1/theta^k style).

    Used by the text connector (NOT by the DiT transformer which uses
    log-spaced frequencies via precompute_rope_freqs).

    Args:
        positions: Position indices of shape (...,).
        dim: Embedding dimension (must be even).
        theta: Base frequency.

    Returns:
        Frequencies of shape (..., dim // 2).
    """
    half_dim = dim // 2
    freqs = 1.0 / (theta ** (mx.arange(0, half_dim).astype(mx.float32) / half_dim))
    angles = positions[..., None].astype(mx.float32) * freqs
    return angles


def get_positional_embedding(
    positions: mx.array,
    dim: int,
    theta: float = 10000.0,
) -> mx.array:
    """Compute full positional embeddings (cos, sin concatenated).

    Used by the text connector, not by the DiT transformer.

    Args:
        positions: Shape (..., num_axes). Each axis gets its own sub-embedding.
        dim: Total embedding dimension.

    Returns:
        Positional embedding of shape (..., dim).
    """
    num_axes = positions.shape[-1]
    dim_per_axis = dim // num_axes
    half_dim = dim_per_axis // 2

    embeddings = []
    for i in range(num_axes):
        freqs = 1.0 / (theta ** (mx.arange(0, half_dim).astype(mx.float32) / half_dim))
        angles = positions[..., i : i + 1].astype(mx.float32) * freqs
        embeddings.append(mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1))

    return mx.concatenate(embeddings, axis=-1)
