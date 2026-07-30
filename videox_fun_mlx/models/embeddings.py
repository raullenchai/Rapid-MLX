"""Positional embeddings for CogVideoX-Fun MLX port.

Provides:
- get_3d_sincos_pos_embed: 3D sinusoidal positional embeddings (temporal + spatial)
- get_3d_rotary_pos_embed: 3D RoPE computation returning (cos, sin)
- apply_rotary_emb: applies rotary embeddings to a tensor
"""

from typing import Optional, Tuple, Union

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# 1D sinusoidal helpers (numpy, computed once)
# ---------------------------------------------------------------------------


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """1D sinusoidal positional embedding from a position grid.

    Args:
        embed_dim: must be even.
        pos: 1-D array of positions, shape (M,).

    Returns:
        Array of shape (M, embed_dim) with [sin, cos] concatenated.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.outer(pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb.astype(np.float32)


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """2D sinusoidal positional embedding from a 2D grid.

    Args:
        embed_dim: must be even.
        grid: shape (2, 1, H, W) or (2, H*W).

    Returns:
        Array of shape (H*W, embed_dim).
    """
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0].reshape(-1))
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1].reshape(-1))
    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


# ---------------------------------------------------------------------------
# 3D sinusoidal positional embedding
# ---------------------------------------------------------------------------


def get_3d_sincos_pos_embed(
    embed_dim: int,
    spatial_size: Union[int, Tuple[int, int]],
    temporal_size: int,
    spatial_interpolation_scale: float = 1.0,
    temporal_interpolation_scale: float = 1.0,
) -> mx.array:
    """Create 3D sinusoidal positional embeddings.

    Splits embed_dim into 3/4 spatial + 1/4 temporal, builds per-axis sincos
    embeddings with numpy, broadcasts to full 3D grid, and returns as mx.array.

    Args:
        embed_dim: embedding dimension, must be divisible by 4.
        spatial_size: (H, W) or a single int applied to both.
        temporal_size: number of frames T.
        spatial_interpolation_scale: divides spatial positions.
        temporal_interpolation_scale: divides temporal positions.

    Returns:
        mx.array of shape (T, H*W, embed_dim).
    """
    if embed_dim % 4 != 0:
        raise ValueError("`embed_dim` must be divisible by 4")
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    embed_dim_spatial = 3 * embed_dim // 4
    embed_dim_temporal = embed_dim // 4

    # --- Spatial ---
    grid_h = np.arange(spatial_size[0], dtype=np.float32) / spatial_interpolation_scale
    grid_w = np.arange(spatial_size[1], dtype=np.float32) / spatial_interpolation_scale
    # meshgrid with indexing="xy": first output varies along columns (w), second along rows (h)
    grid = np.meshgrid(grid_w, grid_h, indexing="xy")
    grid = np.stack(grid, axis=0)  # (2, H, W)
    grid = grid.reshape(2, 1, spatial_size[0], spatial_size[1])
    pos_embed_spatial = _get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)  # (H*W, D_s)

    # --- Temporal ---
    grid_t = np.arange(temporal_size, dtype=np.float32) / temporal_interpolation_scale
    pos_embed_temporal = _get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)  # (T, D_t)

    # --- Broadcast and concat ---
    # spatial: (1, H*W, D_s) -> (T, H*W, D_s)
    pos_embed_spatial = np.tile(pos_embed_spatial[None, :, :], (temporal_size, 1, 1))
    # temporal: (T, 1, D_t) -> (T, H*W, D_t)
    hw = spatial_size[0] * spatial_size[1]
    pos_embed_temporal = np.tile(pos_embed_temporal[:, None, :], (1, hw, 1))

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)  # (T, H*W, D)
    return mx.array(pos_embed)


# ---------------------------------------------------------------------------
# 1D rotary positional embedding helper
# ---------------------------------------------------------------------------


def _get_1d_rotary_pos_embed(
    dim: int,
    pos: np.ndarray,
    theta: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1D rotary positional embedding frequencies.

    Uses the CogVideoX convention: cos/sin are repeat-interleaved so that each
    pair of adjacent elements shares the same frequency (matches diffusers'
    ``repeat_interleave_real=True`` path).

    Args:
        dim: half the head dimension for this axis (actual output has size 2*dim
             because of repeat-interleave).
        pos: 1-D position array, shape (S,).
        theta: RoPE base frequency.

    Returns:
        (cos, sin) each of shape (S, 2*dim) -- i.e. repeat-interleaved.
    """
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))  # (dim/2,)
    angles = np.outer(pos.astype(np.float64), freqs)  # (S, dim/2)
    cos = np.cos(angles).astype(np.float32)  # (S, dim/2)
    sin = np.sin(angles).astype(np.float32)  # (S, dim/2)
    # repeat_interleave along last dim: each freq value appears twice
    cos = np.repeat(cos, 2, axis=1)  # (S, dim)
    sin = np.repeat(sin, 2, axis=1)  # (S, dim)
    return cos, sin


# ---------------------------------------------------------------------------
# 3D rotary positional embedding (RoPE for video tokens)
# ---------------------------------------------------------------------------


def get_3d_rotary_pos_embed(
    embed_dim: int,
    crops_coords: Tuple[Tuple[int, int], Tuple[int, int]],
    grid_size: Tuple[int, int],
    temporal_size: int,
    theta: int = 10000,
    grid_type: str = "linspace",
    max_size: Optional[Tuple[int, int]] = None,
) -> Tuple[mx.array, mx.array]:
    """RoPE for video tokens with 3D structure.

    Ported from ``videox_fun/pipeline/pipeline_cogvideox_fun_inpaint.py:48-139``.

    Args:
        embed_dim: head dimension (hidden_size // num_heads).
        crops_coords: ((top, left), (bottom, right)) crop coordinates.
        grid_size: (grid_h, grid_w) spatial grid.
        temporal_size: number of temporal positions.
        theta: RoPE base frequency.
        grid_type: "linspace" or "slice".
        max_size: required when grid_type == "slice".

    Returns:
        (cos, sin) each of shape (T * H * W, embed_dim).
    """
    grid_size_h, grid_size_w = grid_size

    if grid_type == "linspace":
        start, stop = crops_coords
        grid_h = np.linspace(start[0], stop[0], grid_size_h, endpoint=False, dtype=np.float32)
        grid_w = np.linspace(start[1], stop[1], grid_size_w, endpoint=False, dtype=np.float32)
        grid_t = np.linspace(0, temporal_size, temporal_size, endpoint=False, dtype=np.float32)
    elif grid_type == "slice":
        if max_size is None:
            raise ValueError("`max_size` is required when grid_type == 'slice'")
        max_h, max_w = max_size
        grid_h = np.arange(max_h, dtype=np.float32)
        grid_w = np.arange(max_w, dtype=np.float32)
        grid_t = np.arange(temporal_size, dtype=np.float32)
    else:
        raise ValueError(f"Invalid grid_type: {grid_type}")

    # Dimension splits (must match the original exactly)
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    # Per-axis 1D RoPE
    t_cos, t_sin = _get_1d_rotary_pos_embed(dim_t, grid_t, theta=theta)
    h_cos, h_sin = _get_1d_rotary_pos_embed(dim_h, grid_h, theta=theta)
    w_cos, w_sin = _get_1d_rotary_pos_embed(dim_w, grid_w, theta=theta)

    if grid_type == "slice":
        t_cos, t_sin = t_cos[:temporal_size], t_sin[:temporal_size]
        h_cos, h_sin = h_cos[:grid_size_h], h_sin[:grid_size_h]
        w_cos, w_sin = w_cos[:grid_size_w], w_sin[:grid_size_w]

    def combine_time_height_width(freqs_t: np.ndarray, freqs_h: np.ndarray, freqs_w: np.ndarray) -> np.ndarray:
        # Broadcast to (T, H, W, dim_axis) then concatenate along last dim
        ft = np.tile(freqs_t[:, None, None, :], (1, grid_size_h, grid_size_w, 1))
        fh = np.tile(freqs_h[None, :, None, :], (temporal_size, 1, grid_size_w, 1))
        fw = np.tile(freqs_w[None, None, :, :], (temporal_size, grid_size_h, 1, 1))
        combined = np.concatenate([ft, fh, fw], axis=-1)
        return combined.reshape(temporal_size * grid_size_h * grid_size_w, -1)

    cos = combine_time_height_width(t_cos, h_cos, w_cos)
    sin = combine_time_height_width(t_sin, h_sin, w_sin)
    return mx.array(cos), mx.array(sin)


# ---------------------------------------------------------------------------
# Apply rotary embedding
# ---------------------------------------------------------------------------


def apply_rotary_emb(
    x: mx.array,
    freqs: Tuple[mx.array, mx.array],
    use_real_unbind_dim: int = -1,
) -> mx.array:
    """Apply rotary positional embedding to tensor *x*.

    Matches the diffusers CogVideoX convention (use_real=True,
    use_real_unbind_dim=-1, repeat_interleave_real=True).

    The cos/sin tensors have shape (S, D) where D equals x's last dim (because
    frequencies are repeat-interleaved). They are broadcast over batch and head
    dims automatically.

    Algorithm (for use_real_unbind_dim=-1, the CogVideoX default):
      1. Reshape x to (..., D//2, 2) to get adjacent pairs.
      2. Separate real/imag parts: x_real = pairs[..., 0], x_imag = pairs[..., 1].
      3. Build rotated version: stack([-x_imag, x_real], dim=-1).flatten(-2).
      4. out = x * cos + x_rotated * sin.

    Args:
        x: input tensor of shape (B, H, S, D) or any shape with last dim D.
        freqs: (cos, sin) each of shape (S, D), broadcast over leading dims.
        use_real_unbind_dim: -1 for CogVideoX/flux style (pair along last dim),
            -2 for StableAudio/CogView4 style (split in half).

    Returns:
        Rotated tensor, same shape as x.
    """
    cos, sin = freqs

    if use_real_unbind_dim == -1:
        # CogVideoX / flux / hunyuan-dit convention
        # Reshape to pairs: (..., D) -> (..., D//2, 2)
        orig_shape = x.shape
        paired = x.reshape(*orig_shape[:-1], -1, 2)
        x_real = paired[..., 0]  # (..., D//2)
        x_imag = paired[..., 1]  # (..., D//2)
        # Build rotated: [-imag, real] interleaved back to (..., D)
        x_rotated = mx.stack([-x_imag, x_real], axis=-1).reshape(orig_shape)
    elif use_real_unbind_dim == -2:
        # StableAudio / CogView4 convention: split in half
        d = x.shape[-1]
        x_real = x[..., : d // 2]
        x_imag = x[..., d // 2 :]
        x_rotated = mx.concatenate([-x_imag, x_real], axis=-1)
    else:
        raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

    # Mirror diffusers: compute the rotation in float32, cast the result
    # back to x.dtype (embeddings.py:1231 — the deliberate fp32 island).
    # Without the cast, fp32 cos/sin tables promote bf16 q/k to fp32 and
    # contaminate the whole attention (MLX strict promotion).
    return (x.astype(mx.float32) * cos + x_rotated.astype(mx.float32) * sin).astype(x.dtype)
