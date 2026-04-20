# SPDX-License-Identifier: Apache-2.0
"""
TurboQuant KV cache — wedge for mlx-lm PR #1059.

Subclasses mlx-lm's _BaseCache to provide PolarQuant-compressed KV cache.
Stores K and V as bit-packed indices + norms. Dequantizes on every
update_and_fetch() call (full materialization).

When mlx-lm merges TurboQuant natively, this file can be deleted and
replaced with: from mlx_lm.models.turboquant import TurboQuantKVCache

Based on: https://github.com/ml-explore/mlx-lm/pull/1059
Algorithm: PolarQuant (arXiv 2504.19874, ICLR 2026)
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx_lm.models.cache import _BaseCache, create_attention_mask

# ---------------------------------------------------------------------------
# Lloyd-Max optimal centroids and boundaries for N(0,1)
# Scaled by 1/sqrt(head_dim) at runtime
# ---------------------------------------------------------------------------

_CENTROIDS = {
    2: [-1.5104, -0.4528, 0.4528, 1.5104],
    3: [-2.1519, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1519],
    4: [
        -2.7331,
        -2.0698,
        -1.6189,
        -1.2570,
        -0.9431,
        -0.6573,
        -0.3884,
        -0.1285,
        0.1285,
        0.3884,
        0.6573,
        0.9431,
        1.2570,
        1.6189,
        2.0698,
        2.7331,
    ],
}

_BOUNDARIES = {
    2: [-5.0, -0.9816, 0.0, 0.9816, 5.0],
    3: [-5.0, -1.7479, -1.0499, -0.5005, 0.0, 0.5005, 1.0499, 1.7479, 5.0],
    4: [
        -5.0,
        -2.4015,
        -1.8443,
        -1.4380,
        -1.1001,
        -0.8002,
        -0.5229,
        -0.2585,
        0.0,
        0.2585,
        0.5229,
        0.8002,
        1.1001,
        1.4380,
        1.8443,
        2.4015,
        5.0,
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotation_matrix(dim: int, seed: int = 42) -> mx.array:
    """Haar-distributed random orthogonal matrix via QR of Gaussian."""
    key = mx.random.key(seed)
    g = mx.random.normal(shape=(dim, dim), key=key)
    q, r = mx.linalg.qr(g, stream=mx.cpu)
    sign = mx.sign(mx.diag(r))
    sign = mx.where(sign == 0, 1, sign)
    return q * sign


def _load_codebook(bits: int, dim: int):
    s = 1.0 / math.sqrt(dim)
    c = mx.array(_CENTROIDS[bits], dtype=mx.float32) * s
    b = mx.array(_BOUNDARIES[bits], dtype=mx.float32) * s
    return c, b


def _quantize(vectors: mx.array, rotation_t: mx.array, boundaries: mx.array):
    """Normalize → rotate → digitize."""
    norms = mx.linalg.norm(vectors, axis=-1, keepdims=True)
    rotated = (vectors / mx.maximum(norms, 1e-8)) @ rotation_t
    inner = boundaries[1:-1]
    indices = mx.zeros(rotated.shape, dtype=mx.uint8)
    for b in range(inner.shape[0]):
        indices = indices + (rotated > inner[b]).astype(mx.uint8)
    return indices, norms


def _dequantize(
    indices: mx.array, norms: mx.array, rotation: mx.array, centroids: mx.array
) -> mx.array:
    """Lookup centroids → inverse rotate → rescale."""
    return centroids[indices] @ rotation * norms


def _pack(indices: mx.array, bits: int) -> mx.array:
    """Pack b-bit indices into uint32."""
    shape = indices.shape
    dim = shape[-1]
    vpi = 32 // bits  # values per int
    n_packed = (dim + vpi - 1) // vpi
    pad_size = n_packed * vpi - dim
    if pad_size > 0:
        indices = mx.concatenate(
            [indices, mx.zeros((*shape[:-1], pad_size), dtype=indices.dtype)],
            axis=-1,
        )
    reshaped = indices.reshape(*shape[:-1], n_packed, vpi).astype(mx.uint32)
    shifts = mx.arange(vpi, dtype=mx.uint32) * bits
    shifted = reshaped << shifts
    packed = shifted[..., 0]
    for i in range(1, vpi):
        packed = packed | shifted[..., i]
    return packed


def _unpack(packed: mx.array, bits: int, dim: int) -> mx.array:
    """Unpack uint32 back to b-bit indices."""
    shape = packed.shape
    vpi = 32 // bits
    mask = (1 << bits) - 1
    shifts = mx.arange(vpi, dtype=mx.uint32) * bits
    extracted = (packed[..., None] >> shifts) & mask
    return extracted.reshape(*shape[:-1], shape[-1] * vpi)[..., :dim].astype(mx.uint8)


# ---------------------------------------------------------------------------
# TurboQuantKVCache — drop-in _BaseCache replacement
# ---------------------------------------------------------------------------


class TurboQuantKVCache(_BaseCache):
    """KV cache with PolarQuant compression.

    Drop-in replacement for KVCache. Stores K and V as bit-packed indices
    plus per-vector norms. Dequantizes on every update_and_fetch().

    Args:
        bits: Quantization bits (2, 3, or 4). Default 4.
    """

    step = 256

    def __init__(self, bits: int = 4):
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
        self.turbo_bits = bits
        self.offset = 0
        self._head_dim: int | None = None
        self._k_indices: mx.array | None = None
        self._k_norms: mx.array | None = None
        self._v_indices: mx.array | None = None
        self._v_norms: mx.array | None = None
        self._centroids: mx.array | None = None
        self._boundaries: mx.array | None = None
        self._rotation: mx.array | None = None
        self._rotation_t: mx.array | None = None

    def _init_codebook(self, head_dim: int) -> None:
        self._head_dim = head_dim
        self._centroids, self._boundaries = _load_codebook(self.turbo_bits, head_dim)
        self._rotation = _rotation_matrix(head_dim)
        self._rotation_t = self._rotation.T

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, head_dim = keys.shape
        prev = self.offset
        if self._centroids is None:
            self._init_codebook(head_dim)

        # Quantize new tokens
        k_idx, k_norms = _quantize(keys, self._rotation_t, self._boundaries)
        v_idx, v_norms = _quantize(values, self._rotation_t, self._boundaries)
        pk = _pack(k_idx, self.turbo_bits)
        pv = _pack(v_idx, self.turbo_bits)

        # Expand storage if needed
        if self._k_indices is None or (prev + num_steps) > self._k_indices.shape[2]:
            self._expand(B, n_kv_heads, num_steps, keys.dtype, pk.shape[-1])

        # Store packed indices + norms
        self._k_indices[..., prev : prev + num_steps, :] = pk
        self._k_norms[..., prev : prev + num_steps, :] = k_norms
        self._v_indices[..., prev : prev + num_steps, :] = pv
        self._v_norms[..., prev : prev + num_steps, :] = v_norms
        self.offset += num_steps

        # Dequantize full history for attention
        all_k = _dequantize(
            _unpack(self._k_indices[..., : self.offset, :], self.turbo_bits, head_dim),
            self._k_norms[..., : self.offset, :],
            self._rotation,
            self._centroids,
        )
        all_v = _dequantize(
            _unpack(self._v_indices[..., : self.offset, :], self.turbo_bits, head_dim),
            self._v_norms[..., : self.offset, :],
            self._rotation,
            self._centroids,
        )
        return all_k, all_v

    def _expand(self, batch_size, n_kv_heads, new_steps, dtype, packed_dim):
        alloc = ((self.step + new_steps - 1) // self.step) * self.step
        shape = (batch_size, n_kv_heads, alloc)

        new_ki = mx.zeros((*shape, packed_dim), dtype=mx.uint32)
        new_kn = mx.zeros((*shape, 1), dtype=dtype)
        new_vi = mx.zeros((*shape, packed_dim), dtype=mx.uint32)
        new_vn = mx.zeros((*shape, 1), dtype=dtype)

        if self._k_indices is not None and self.offset > 0:
            old = (
                self._k_indices[..., : self.offset, :],
                self._k_norms[..., : self.offset, :],
                self._v_indices[..., : self.offset, :],
                self._v_norms[..., : self.offset, :],
            )
            self._k_indices, self._k_norms, self._v_indices, self._v_norms = (
                mx.concatenate([o, n], axis=2)
                for o, n in zip(old, (new_ki, new_kn, new_vi, new_vn))
            )
        else:
            self._k_indices = new_ki
            self._k_norms = new_kn
            self._v_indices = new_vi
            self._v_norms = new_vn

    # -- _BaseCache interface --

    def size(self):
        return self.offset

    @property
    def state(self):
        if self._k_indices is None:
            return []
        return [
            self._k_indices[..., : self.offset, :],
            self._k_norms[..., : self.offset, :],
            self._v_indices[..., : self.offset, :],
            self._v_norms[..., : self.offset, :],
        ]

    @state.setter
    def state(self, v):
        if v is not None and v:
            self._k_indices, self._k_norms, self._v_indices, self._v_norms = v
            self.offset = self._k_indices.shape[2]

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.turbo_bits, self._head_dim or 0)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.turbo_bits = int(v[0]), int(v[1])
        head_dim = int(v[2])
        if head_dim > 0:
            self._init_codebook(head_dim)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self._k_indices is None

    @property
    def nbytes(self):
        if self._k_indices is None:
            return 0
        return sum(
            a[..., : self.offset, :].nbytes
            for a in (self._k_indices, self._k_norms, self._v_indices, self._v_norms)
        )
