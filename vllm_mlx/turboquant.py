# SPDX-License-Identifier: Apache-2.0
"""
TurboQuant KV cache compression for prefix cache.

V-only asymmetric compression: K stays FP16, V is quantized to 3-4 bits
using random orthogonal rotation + Lloyd-Max codebook quantization.

Based on the TurboQuant paper (arXiv 2504.19874, ICLR 2026).

Usage::

    config = TurboQuantConfig(bits=3)
    tq_cache = TurboQuantKVCache.from_kv_cache(kv_cache, config)
    restored = tq_cache.to_kv_cache()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurboQuantConfig:
    """TurboQuant compression settings."""

    bits: int = 3  # 3 or 4
    group_size: int = 32
    rotation_seed: int = 42

    def __post_init__(self):
        if self.bits not in (3, 4):
            raise ValueError(f"bits must be 3 or 4, got {self.bits}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")


def auto_select_bits(head_dim: int) -> int:
    """Select bit width based on head dimension.

    3-bit is safe for head_dim >= 96 (cosine > 0.95).
    4-bit is required for head_dim = 64 (3-bit degrades below 0.85).
    """
    return 3 if head_dim >= 96 else 4


# ---------------------------------------------------------------------------
# Lloyd-Max codebooks (precomputed for unit Gaussian)
# ---------------------------------------------------------------------------

# Optimal Lloyd-Max quantizer for N(0,1) data.
# Centroids = conditional expectations E[X | X in bin_i].
# Boundaries = decision thresholds between adjacent centroids.
# Reference: Lloyd (1982), Max (1960). Values from scipy Lloyd-Max solver.
# fmt: off

# 3-bit: 8 centroids, 7 boundaries
_LLOYD_MAX_3BIT = mx.array([
    -2.1519, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1519
], dtype=mx.float16)

_LLOYD_MAX_3BIT_BOUNDS = mx.array([
    -1.7479, -1.0500, -0.5005, 0.0000, 0.5005, 1.0500, 1.7479
], dtype=mx.float16)

# 4-bit: 16 centroids, 15 boundaries
_LLOYD_MAX_4BIT = mx.array([
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
     0.1284,  0.3881,  0.6568,  0.9423,  1.2562,  1.6180,  2.0690,  2.7326
], dtype=mx.float16)

_LLOYD_MAX_4BIT_BOUNDS = mx.array([
    -2.4008, -1.8435, -1.4371, -1.0993, -0.7996, -0.5224, -0.2582, 0.0000,
     0.2582,  0.5224,  0.7996,  1.0993,  1.4371,  1.8435,  2.4008
], dtype=mx.float16)
# fmt: on

LLOYD_MAX_CODEBOOKS = {3: _LLOYD_MAX_3BIT, 4: _LLOYD_MAX_4BIT}
LLOYD_MAX_BOUNDARIES = {3: _LLOYD_MAX_3BIT_BOUNDS, 4: _LLOYD_MAX_4BIT_BOUNDS}


# ---------------------------------------------------------------------------
# Bit-packing: 2 indices per uint8 (nibble packing)
# ---------------------------------------------------------------------------


def _pack_nibbles(indices: mx.array) -> mx.array:
    """Pack pairs of 4-bit indices into uint8 (2 per byte).

    Input shape: (..., N) where N is even. Values in [0, 15].
    Output shape: (..., N//2) dtype uint8.
    """
    # Pad to even length if needed
    *batch, n = indices.shape
    if n % 2 != 0:
        indices = mx.pad(indices, [(0, 0)] * len(batch) + [(0, 1)])
        n += 1

    reshaped = indices.reshape(*batch, n // 2, 2)
    high = reshaped[..., 0].astype(mx.uint8) << 4
    low = reshaped[..., 1].astype(mx.uint8) & 0x0F
    return (high | low).astype(mx.uint8)


def _unpack_nibbles(packed: mx.array, original_len: int) -> mx.array:
    """Unpack uint8 nibble-packed array back to individual indices.

    Input shape: (..., N//2) dtype uint8.
    Output shape: (..., original_len) dtype uint8.
    """
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    *batch, n_packed = packed.shape
    # Interleave high and low nibbles
    unpacked = mx.concatenate(
        [mx.expand_dims(high, -1), mx.expand_dims(low, -1)], axis=-1
    ).reshape(*batch, n_packed * 2)
    return unpacked[..., :original_len]


# ---------------------------------------------------------------------------
# Rotation matrix (cached per head_dim)
# ---------------------------------------------------------------------------

_rotation_cache: dict[tuple[int, int], mx.array] = {}


def generate_rotation_matrix(dim: int, seed: int = 42) -> mx.array:
    """Generate a fixed random orthogonal matrix Q via QR decomposition.

    Result is cached per (dim, seed) — called once per unique head_dim.
    """
    key = (dim, seed)
    if key in _rotation_cache:
        return _rotation_cache[key]

    # Use numpy for deterministic QR (mlx doesn't have linalg.qr)
    rng = np.random.RandomState(seed)
    random_matrix = rng.randn(dim, dim).astype(np.float32)
    q, _ = np.linalg.qr(random_matrix)
    # Keep float32 for rotation to preserve orthogonality during matmul.
    # The V data is upcast to float32 for rotation, then back to float16.
    rotation = mx.array(q, dtype=mx.float32)

    _rotation_cache[key] = rotation
    return rotation


# ---------------------------------------------------------------------------
# Encode / Decode
# ---------------------------------------------------------------------------


def turboquant_encode(
    values: mx.array,
    bits: int,
    group_size: int,
    rotation: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compress V tensor using TurboQuant.

    Args:
        values: V tensor, shape (..., seq_len, head_dim). FP16.
        bits: 3 or 4.
        group_size: Elements per quantization group.
        rotation: Orthogonal matrix, shape (head_dim, head_dim).

    Returns:
        (packed_indices, scales, zeros) where:
        - packed_indices: uint8, shape (..., seq_len, ceil(head_dim/2)) — nibble-packed
        - scales: float16, shape (..., seq_len, n_groups) — per-group scale
        - zeros: float16, shape (..., seq_len, n_groups) — per-group mean
    """
    # 1. Rotate along head_dim: V @ Q^T (in float32 for precision)
    rotated = values.astype(mx.float32) @ rotation.T

    # 2. Per-group normalize to unit Gaussian
    orig_shape = rotated.shape
    head_dim = orig_shape[-1]
    n_groups = (head_dim + group_size - 1) // group_size

    # Pad if head_dim not divisible by group_size
    if head_dim % group_size != 0:
        pad_size = group_size * n_groups - head_dim
        rotated = mx.pad(rotated, [(0, 0)] * (len(orig_shape) - 1) + [(0, pad_size)])

    # Reshape to (..., seq_len, n_groups, group_size)
    grouped = rotated.reshape(*orig_shape[:-1], n_groups, group_size)

    # Compute per-group statistics
    group_mean = mx.mean(grouped, axis=-1, keepdims=True)  # (..., n_groups, 1)
    group_std = mx.maximum(
        mx.sqrt(mx.mean((grouped - group_mean) ** 2, axis=-1, keepdims=True)),
        mx.array(1e-6, dtype=mx.float16),
    )

    # Normalize to ~N(0,1)
    normalized = (grouped - group_mean) / group_std

    # 3. Quantize using Lloyd-Max codebook via broadcasting comparison
    # For each value, count how many boundaries it exceeds → gives the bin index.
    # boundaries shape: (n_levels - 1,), normalized shape: (..., group_size)
    boundaries = LLOYD_MAX_BOUNDARIES[bits]
    # Expand for broadcasting: normalized[..., None] > boundaries[None, ...]
    # Sum across boundary dim gives index
    expanded = mx.expand_dims(normalized, axis=-1)  # (..., group_size, 1)
    # boundaries reshaped to (1, ..., 1, n_bounds) for broadcast
    bounds = boundaries.reshape((1,) * len(normalized.shape) + (-1,))
    indices = mx.sum(expanded > bounds, axis=-1).astype(mx.uint8)  # (..., group_size)

    # Reshape indices back to (..., seq_len, padded_head_dim)
    indices = indices.reshape(*orig_shape[:-1], n_groups * group_size)
    # Trim padding
    if head_dim % group_size != 0:
        indices = indices[..., :head_dim]

    # Scales and zeros: squeeze keepdim
    scales = group_std.squeeze(-1)  # (..., seq_len, n_groups)
    zeros = group_mean.squeeze(-1)  # (..., seq_len, n_groups)

    # 4. Bit-pack indices: 2 per uint8 (halves index memory)
    packed_indices = _pack_nibbles(indices)

    return packed_indices, scales, zeros


def turboquant_decode(
    packed_indices: mx.array,
    scales: mx.array,
    zeros: mx.array,
    bits: int,
    group_size: int,
    rotation: mx.array,
    head_dim: int,
) -> mx.array:
    """Decompress V tensor from TurboQuant format.

    Args:
        packed_indices: nibble-packed uint8 indices, shape (..., seq_len, head_dim//2)
        scales: float16 per-group scale, shape (..., seq_len, n_groups)
        zeros: float16 per-group mean, shape (..., seq_len, n_groups)
        bits: 3 or 4
        group_size: Elements per quantization group
        rotation: Orthogonal matrix, shape (head_dim, head_dim)
        head_dim: Original head dimension (before any padding)

    Returns:
        Reconstructed V tensor, shape (..., seq_len, head_dim). FP16.
    """
    codebook = LLOYD_MAX_CODEBOOKS[bits]
    n_groups = scales.shape[-1]

    # 1. Unpack nibble-packed indices and look up codebook values
    indices = _unpack_nibbles(packed_indices, head_dim)
    dequantized = codebook[indices]  # (..., seq_len, head_dim)

    # 2. Pad if needed, reshape to groups
    padded_dim = n_groups * group_size
    if head_dim < padded_dim:
        pad_size = padded_dim - head_dim
        dequantized = mx.pad(
            dequantized, [(0, 0)] * (len(dequantized.shape) - 1) + [(0, pad_size)]
        )

    orig_batch_shape = dequantized.shape[:-1]
    grouped = dequantized.reshape(*orig_batch_shape, n_groups, group_size)

    # 3. Denormalize: x = x * scale + mean
    scales_expanded = mx.expand_dims(scales, axis=-1)  # (..., n_groups, 1)
    zeros_expanded = mx.expand_dims(zeros, axis=-1)
    grouped = grouped * scales_expanded + zeros_expanded

    # 4. Reshape back and trim padding
    rotated = grouped.reshape(*orig_batch_shape, padded_dim)
    if head_dim < padded_dim:
        rotated = rotated[..., :head_dim]

    # 5. Inverse rotation: V_reconstructed = rotated @ Q (float32 for precision)
    values = rotated.astype(mx.float32) @ rotation

    return values.astype(mx.float16)


# ---------------------------------------------------------------------------
# TurboQuantKVCache — prefix cache storage wrapper
# ---------------------------------------------------------------------------


class TurboQuantKVCache:
    """KV cache with TurboQuant V compression for prefix cache storage.

    K stays FP16. V is compressed to 3-4 bits using rotation + Lloyd-Max.
    This class is used in the prefix cache (store/fetch), not during
    model forward passes.
    """

    def __init__(
        self,
        keys: mx.array,
        values_compressed: tuple[mx.array, mx.array, mx.array],
        offset: int,
        config: TurboQuantConfig,
        head_dim: int,
    ):
        self.keys = keys
        self.values_compressed = values_compressed  # (indices, scales, zeros)
        self.offset = offset
        self.config = config
        self.head_dim = head_dim

    @classmethod
    def from_kv_cache(cls, kv_cache, config: TurboQuantConfig) -> TurboQuantKVCache:
        """Compress a standard KVCache into TurboQuant format."""
        keys = kv_cache.keys
        values = kv_cache.values
        offset = kv_cache.offset

        if keys is None or values is None:
            return cls(
                keys=None,
                values_compressed=(None, None, None),
                offset=0,
                config=config,
                head_dim=0,
            )

        # Get actual data up to offset
        if offset < keys.shape[-2]:
            keys = keys[..., :offset, :]
            values = values[..., :offset, :]

        head_dim = values.shape[-1]
        rotation = generate_rotation_matrix(head_dim, config.rotation_seed)

        indices, scales, zeros = turboquant_encode(
            values, config.bits, config.group_size, rotation
        )

        return cls(
            keys=keys,
            values_compressed=(indices, scales, zeros),
            offset=offset,
            config=config,
            head_dim=head_dim,
        )

    def to_kv_cache(self):
        """Decompress back to a standard KVCache."""
        from mlx_lm.models.cache import KVCache

        kv = KVCache()

        if self.keys is None:
            return kv

        rotation = generate_rotation_matrix(self.head_dim, self.config.rotation_seed)
        indices, scales, zeros = self.values_compressed

        values = turboquant_decode(
            indices,
            scales,
            zeros,
            self.config.bits,
            self.config.group_size,
            rotation,
            self.head_dim,
        )

        kv.keys = self.keys
        kv.values = values
        kv.offset = self.offset
        return kv

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> None:
        """Trim n tokens from the end."""
        if self.keys is not None and n > 0:
            new_offset = max(0, self.offset - n)
            self.keys = self.keys[..., :new_offset, :]
            indices, scales, zeros = self.values_compressed
            self.values_compressed = (
                indices[..., :new_offset, :] if indices is not None else None,
                scales[..., :new_offset, :] if scales is not None else None,
                zeros[..., :new_offset, :] if zeros is not None else None,
            )
            self.offset = new_offset

    @property
    def memory_bytes(self) -> int:
        """Estimate memory usage in bytes."""
        total = 0
        if self.keys is not None:
            total += self.keys.nbytes
        indices, scales, zeros = self.values_compressed
        if indices is not None:
            total += indices.nbytes
        if scales is not None:
            total += scales.nbytes
        if zeros is not None:
            total += zeros.nbytes
        return total
