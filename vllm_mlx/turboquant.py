# SPDX-License-Identifier: Apache-2.0
"""
TurboQuant KV cache compression for prefix cache.

Two compression modes:

* ``v4`` (V-only, original PR #157): K stays FP16, V is quantized to
  3-4 bits using random orthogonal rotation + Lloyd-Max codebook
  quantization.
* ``k8v4`` (R15 Phase 4, this PR): K is 8-bit per-coord symmetric
  uniform after a Walsh-Hadamard rotation, V is 4-bit Lloyd-Max as in
  ``v4``. Reaches ~4.6× total KV compression on dense models per
  arozanov/turboquant-mlx, but is mutually exclusive with the legacy
  ``--kv-cache-quantization`` flag — the schemes overlap on the same
  cache slabs.

Both modes are opt-in and gated behind ``--kv-cache-turboquant``. The
Walsh-Hadamard fast path runs when the model's head_dim is a power of
two (Qwen-class 64/128); other head dimensions fall back to the
original QR-decomposed orthogonal matrix so callers don't lose access
to the V4 path on, e.g., Llama-3 head_dim=128.

Based on the TurboQuant paper (arXiv 2504.19874, ICLR 2026) and the
arozanov/turboquant-mlx implementation (Apache-2.0; see
``vllm_mlx/kernels/turboquant_fused.metal`` for the vendored fused
Metal kernel).

Usage::

    # V-only (regression-preserved from PR #157)
    config = TurboQuantConfig(bits=3)
    tq_cache = TurboQuantKVCache.from_kv_cache(kv_cache, config)
    restored = tq_cache.to_kv_cache()

    # K8V4 mix (R15 Phase 4)
    config = TurboQuantConfig(mode="k8v4", bits=4)
    tq_cache = TurboQuantKVCache.from_kv_cache(kv_cache, config)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# Supported TurboQuant compression modes. Order matters for the
# Prometheus ``rapid_mlx_turboquant_mode`` gauge — the string value is
# the label.
TURBOQUANT_MODES: tuple[str, ...] = ("v4", "k8v4")
DEFAULT_TURBOQUANT_MODE = "v4"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurboQuantConfig:
    """TurboQuant compression settings.

    Attributes:
        bits: V-side quantization bits. 3 or 4. The K-side bit width is
            fixed at 8 when ``mode="k8v4"`` and ignored otherwise.
        group_size: Per-group bucket size for the V-side Lloyd-Max
            normalization. ``32`` is the published default; 64 is safe
            on head_dim>=128.
        rotation_seed: Seed for the rotation diagonal / QR matrix —
            same value must be used at encode and decode time.
        mode: ``"v4"`` (V-only, legacy default) or ``"k8v4"`` (R15
            Phase 4 mixed K-8bit + V-4bit). When set to ``"k8v4"`` the
            V-side ``bits`` field is forced to 4 because the K8 path is
            only validated against V4 in the paper / bench.
    """

    bits: int = 3  # 3 or 4 — V side
    group_size: int = 32
    rotation_seed: int = 42
    mode: str = DEFAULT_TURBOQUANT_MODE

    def __post_init__(self):
        if self.mode not in TURBOQUANT_MODES:
            raise ValueError(
                f"mode must be one of {TURBOQUANT_MODES}, got {self.mode!r}"
            )
        # K8V4 is only validated with V=4-bit. Reject anything else
        # explicitly so callers don't silently degrade to V=3-bit + K8
        # (which is outside the arozanov bench envelope and would slip
        # past the default-on flip gate without a quality signal).
        if self.mode == "k8v4" and self.bits != 4:
            raise ValueError(
                f"mode='k8v4' requires bits=4 (got {self.bits}); the K8 "
                "path is only validated against V4 per the arozanov bench"
            )
        if self.bits not in (3, 4):
            raise ValueError(f"bits must be 3 or 4, got {self.bits}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")

    @property
    def k_bits(self) -> int | None:
        """Effective K-side bit width (``None`` when K is FP16)."""
        return 8 if self.mode == "k8v4" else None


def auto_select_bits(head_dim: int) -> int:
    """Select bit width based on head dimension.

    3-bit is safe for head_dim >= 96 (cosine > 0.95).
    4-bit is required for head_dim = 64 (3-bit degrades below 0.85).
    """
    return 3 if head_dim >= 96 else 4


# ---------------------------------------------------------------------------
# Skip-list registry (R15 Phase 4)
# ---------------------------------------------------------------------------
# Architectures where TurboQuant must NOT engage — either the K/V cache
# layout is non-contiguous (sliding window) or the K projection is
# already low-rank (MLA). Detection mirrors
# :mod:`vllm_mlx.kv_cache_dtype` so the two safelists stay in lock-step.

SKIP_REASON_SLIDING = "sliding-window"
SKIP_REASON_MLA = "mla"
SKIP_REASON_OTHER = "other"

# Family-pattern → reason. Patterns are case-insensitive substring
# matches against the alias key + resolved HF path. Patterns are
# ordered most-specific-first so a future Gemma 4 / DeepSeek V5 release
# does not silently slip through.
MODELS_INCOMPATIBLE_WITH_TURBOQUANT: dict[str, str] = {
    # Sliding-window attention rotates a fixed window per layer; the
    # TurboQuant V-side codebook assumes contiguous tokens.
    r"gemma[-_]?3": SKIP_REASON_SLIDING,
    r"gpt[-_]?oss": SKIP_REASON_SLIDING,
    # Multi-head Latent Attention — K projection is already compressed
    # via q_lora_rank / kv_lora_rank, stacking quant on top compounds
    # error on reasoning workloads.
    r"deepseek[-_]?v3": SKIP_REASON_MLA,
    r"deepseek[-_]?v4": SKIP_REASON_MLA,
    r"kimi[-_]?k2\.?5": SKIP_REASON_MLA,
    r"kimi[-_]?k2\.?6": SKIP_REASON_MLA,
}

_COMPILED_SKIP_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), reason)
    for pat, reason in MODELS_INCOMPATIBLE_WITH_TURBOQUANT.items()
)


def is_incompatible_with_turboquant(
    *,
    model_name: str | None = None,
    hf_path: str | None = None,
    hf_config: dict[str, Any] | None = None,
    alias_metadata: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Return ``(skip?, reason)`` for a candidate model.

    Detection order mirrors :mod:`vllm_mlx.kv_cache_dtype._is_sliding_window`
    /  ``_is_mla``:

    1. Alias metadata (cheapest, contributor-curated).
    2. HF ``config.json`` (canonical when available).
    3. Substring patterns on ``model_name`` + ``hf_path`` (fallback for
       servers that haven't loaded the HF config yet).

    The reason string is the Prometheus label value emitted on the
    ``rapid_mlx_turboquant_skipped_total`` counter.
    """
    # 1. Alias hints.
    if alias_metadata:
        if alias_metadata.get("sliding_window"):
            return True, SKIP_REASON_SLIDING
        if alias_metadata.get("is_mla"):
            return True, SKIP_REASON_MLA

    # 2. HF config hints.
    if hf_config:
        sw = hf_config.get("sliding_window")
        if isinstance(sw, int) and sw > 0:
            return True, SKIP_REASON_SLIDING
        model_type = hf_config.get("model_type")
        if isinstance(model_type, str):
            mt = model_type.lower()
            if mt in {"gemma3", "gemma3_text", "gpt_oss"}:
                return True, SKIP_REASON_SLIDING
            if mt in {"deepseek_v3", "deepseek_v4"}:
                return True, SKIP_REASON_MLA
        # MLA rank-pair check (DeepSeek-class): rank pair only counts
        # when paired with a known MLA family name, mirroring
        # kv_cache_dtype._is_mla.
        needle = f"{model_name or ''} {hf_path or ''}".lower()
        if any(
            pat in needle
            for pat in ("deepseek-v3", "deepseek_v3", "deepseek-v4", "deepseek_v4")
        ):
            q_rank = hf_config.get("q_lora_rank")
            kv_rank = hf_config.get("kv_lora_rank")
            if (
                isinstance(q_rank, int)
                and q_rank > 0
                and isinstance(kv_rank, int)
                and kv_rank > 0
            ):
                return True, SKIP_REASON_MLA

    # 3. Substring patterns over model name + HF path.
    needle = f"{model_name or ''} {hf_path or ''}"
    for pattern, reason in _COMPILED_SKIP_PATTERNS:
        if pattern.search(needle):
            return True, reason

    return False, None


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
_hadamard_signs_cache: dict[tuple[int, int], mx.array] = {}


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def walsh_hadamard_transform(x: mx.array) -> mx.array:
    """Fast Walsh-Hadamard Transform along the last axis.

    O(d log d) butterfly. Input dimension must be a power of two; the
    output is scaled by ``1/sqrt(d)`` so the transform is orthogonal
    (and therefore norm-preserving) — same convention as
    arozanov/turboquant-mlx.

    The K8V4 path uses this as the rotation preprocessing for the K
    side; the V side keeps the QR-decomposed orthogonal matrix from
    the original V-only PR so the existing decode quality envelope is
    unchanged.
    """
    d = x.shape[-1]
    if not _is_power_of_two(d):
        raise ValueError(f"walsh_hadamard_transform: dim {d} is not a power of 2")
    h = 1
    while h < d:
        x_reshaped = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        even = x_reshaped[..., 0, :]
        odd = x_reshaped[..., 1, :]
        new_even = even + odd
        new_odd = even - odd
        x = mx.stack([new_even, new_odd], axis=-2).reshape(*x.shape[:-1], d)
        h *= 2
    return x * (1.0 / math.sqrt(d))


def random_hadamard_signs(dim: int, seed: int = 42) -> mx.array:
    """Cached ``(dim,)`` ±1 diagonal for the randomized Hadamard rotation.

    Same seed → same signs across encode and decode. Cached on
    ``(dim, seed)`` to amortize the numpy → mx.array hop on the hot
    path. Float32 matches the QR rotation dtype so the two paths share
    the same upcast policy.
    """
    key = (dim, seed)
    cached = _hadamard_signs_cache.get(key)
    if cached is not None:
        return cached
    rng = np.random.RandomState(seed)
    signs = (rng.randint(0, 2, size=dim) * 2 - 1).astype(np.float32)
    arr = mx.array(signs, dtype=mx.float32)
    _hadamard_signs_cache[key] = arr
    return arr


def randomized_hadamard_rotate(values: mx.array, signs: mx.array) -> mx.array:
    """``WHT(values * signs)`` — randomized Hadamard rotation."""
    return walsh_hadamard_transform(values.astype(mx.float32) * signs)


def randomized_hadamard_inverse(values: mx.array, signs: mx.array) -> mx.array:
    """Inverse of :func:`randomized_hadamard_rotate`.

    Because WHT (with the ``1/sqrt(d)`` normalization) is its own
    inverse and ``diag(signs)`` is its own inverse, the inverse is
    ``WHT(values) * signs``.
    """
    return walsh_hadamard_transform(values.astype(mx.float32)) * signs


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
# K-side 8-bit encode / decode (R15 Phase 4)
# ---------------------------------------------------------------------------
# K8 stores per-coordinate symmetric uint8 indices (offset by 128 so
# the storage type is unsigned, matching the arozanov Metal kernel
# output) plus one float16 scale per vector. The vector is first L2-
# normalized to the unit sphere and randomized-Hadamard rotated, so a
# per-vector absmax + symmetric 8-bit grid (-127..127) suffices —
# 0.3 dB inside the Lloyd-Max scheme at d>=64 per the arozanov bench.


def turboquant_k8_encode(
    keys: mx.array,
    signs: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compress K tensor using the K8 path of the K8V4 mix.

    Args:
        keys: K tensor, shape ``(..., seq_len, head_dim)``. FP16/FP32.
            ``head_dim`` must be a power of 2.
        signs: Random Hadamard diagonal from
            :func:`random_hadamard_signs`, shape ``(head_dim,)``.

    Returns:
        ``(packed_uint8, norms_fp32, k_scales_fp32)`` where:

        * ``packed_uint8`` — shape ``(..., seq_len, head_dim)`` dtype
          ``uint8``. Storage is symmetric int8 offset by +128 so the
          dtype is unsigned (matches the fused Metal kernel output).
        * ``norms_fp32`` — shape ``(..., seq_len)``. Per-vector L2
          norm. Stored in fp32 because fp16 would underflow on small-
          magnitude key vectors near the end of a long context.
        * ``k_scales_fp32`` — shape ``(..., seq_len)``. Per-vector
          int8 scale: ``absmax_after_rotation / 127``.
    """
    head_dim = keys.shape[-1]
    if not _is_power_of_two(head_dim):
        raise ValueError(
            f"turboquant_k8_encode: head_dim {head_dim} is not a power of 2; "
            "K8V4 requires Walsh-Hadamard which needs power-of-2 dims"
        )

    keys_f32 = keys.astype(mx.float32)
    norms = mx.sqrt(mx.sum(keys_f32 * keys_f32, axis=-1, keepdims=True))
    safe_norms = mx.maximum(norms, mx.array(1e-8, dtype=mx.float32))
    unit = keys_f32 / safe_norms

    rotated = walsh_hadamard_transform(unit * signs)

    # Per-vector absmax → symmetric 8-bit scale.
    amax = mx.maximum(
        mx.max(mx.abs(rotated), axis=-1, keepdims=True),
        mx.array(1e-8, dtype=mx.float32),
    )
    k_scales = amax / 127.0
    quantized = mx.round(rotated / k_scales)
    quantized = mx.clip(quantized, -127.0, 127.0)
    # Offset by +128 so storage is uint8 (matches Metal output).
    packed = (quantized + 128.0).astype(mx.uint8)
    return packed, norms.squeeze(-1), k_scales.squeeze(-1)


def turboquant_k8_decode(
    packed: mx.array,
    norms: mx.array,
    k_scales: mx.array,
    signs: mx.array,
    head_dim: int,
) -> mx.array:
    """Decompress K tensor from K8 storage back to FP16."""
    # Recover signed int8 values, then dequantize.
    signed = packed.astype(mx.float32) - 128.0
    scales_expanded = mx.expand_dims(k_scales.astype(mx.float32), axis=-1)
    rotated = signed * scales_expanded

    # Inverse randomized Hadamard: WHT(rotated) * signs.
    unit = walsh_hadamard_transform(rotated) * signs
    norms_expanded = mx.expand_dims(norms.astype(mx.float32), axis=-1)
    keys = unit * norms_expanded
    return keys.astype(mx.float16)


def turboquant_k8_encode_fused(
    keys: mx.array,
    signs: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fused-Metal-kernel K8 encode, falling back to the unfused path.

    Tries :func:`vllm_mlx.kernels.turboquant_fused.fused_quantize_k8`
    first; if Metal is unavailable / compilation fails / the dispatch
    raises, falls back transparently to :func:`turboquant_k8_encode`.

    The two paths share the contract — same shapes, same dtypes,
    same packing offset (+128) — so callers can swap the fused and
    unfused functions without conditional unpacking. The test suite
    enforces a 1e-4 RMSE bound between the two outputs.
    """
    head_dim = keys.shape[-1]
    if not _is_power_of_two(head_dim):
        # Same guard as the unfused path; head_dim must be power-of-2
        # for the WHT butterfly. Caller should have skipped K8V4 mode
        # before reaching here.
        raise ValueError(
            f"turboquant_k8_encode_fused: head_dim {head_dim} is not a power of 2"
        )

    try:
        from .kernels.turboquant_fused import fused_quantize_k8
    except Exception:
        return turboquant_k8_encode(keys, signs)

    # The fused kernel only accepts (n_vecs, dim) layouts — collapse
    # the leading batch axes, dispatch, then restore the original
    # shape. ``mx.fast.metal_kernel`` requires float32 inputs which the
    # binding handles internally.
    leading = keys.shape[:-1]
    n_vecs = 1
    for s in leading:
        n_vecs *= s
    flat = keys.reshape(n_vecs, head_dim)
    fused = fused_quantize_k8(flat, signs, head_dim)
    if fused is None:
        return turboquant_k8_encode(keys, signs)

    packed_flat, norms_flat, k_scales_flat = fused
    packed = packed_flat.reshape(*leading, head_dim)
    norms = norms_flat.reshape(*leading)
    k_scales = k_scales_flat.reshape(*leading)
    return packed, norms, k_scales


def fused_kernel_status() -> str:
    """Return ``"available"`` or ``"fallback"`` for the metrics gauge.

    Called once at serve boot (so the
    ``rapid_mlx_turboquant_fused_kernel{status}`` gauge is stable for
    the lifetime of the process).
    """
    try:
        from .kernels.turboquant_fused import is_metal_available
    except Exception:
        return "fallback"
    return "available" if is_metal_available() else "fallback"


# ---------------------------------------------------------------------------
# TurboQuantKVCache — prefix cache storage wrapper
# ---------------------------------------------------------------------------


class TurboQuantKVCache:
    """KV cache with TurboQuant compression for prefix cache storage.

    Two modes are supported, gated by ``config.mode``:

    * ``v4`` — K stays FP16, V is 3-4-bit Lloyd-Max (PR #157 legacy).
      Exposed through the ``keys`` and ``values_compressed`` fields.
    * ``k8v4`` — K is 8-bit per-coord symmetric uniform after a
      randomized Hadamard rotation, V is 4-bit Lloyd-Max. The K side
      is exposed through ``keys_compressed`` (a 3-tuple of packed
      indices, norms, scales) and the ``keys`` attribute is ``None``.

    Used in the prefix cache (store/fetch), not during model forward
    passes — both paths return a fully-materialized FP16 KV cache via
    :meth:`to_kv_cache`.
    """

    def __init__(
        self,
        keys: mx.array | None,
        values_compressed: tuple[mx.array | None, mx.array | None, mx.array | None],
        offset: int,
        config: TurboQuantConfig,
        head_dim: int,
        *,
        keys_compressed: (
            tuple[mx.array | None, mx.array | None, mx.array | None] | None
        ) = None,
    ):
        self.keys = keys
        self.values_compressed = values_compressed  # (indices, scales, zeros)
        # ``keys_compressed`` is populated only in K8V4 mode; in V4 mode
        # it stays ``None`` and ``keys`` carries the FP16 K tensor.
        self.keys_compressed = keys_compressed
        self.offset = offset
        self.config = config
        self.head_dim = head_dim

    @property
    def mode(self) -> str:
        """Compression mode (``"v4"`` or ``"k8v4"``)."""
        return self.config.mode

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
                keys_compressed=None,
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

        # K8V4: also encode the K side. K8 needs head_dim power-of-2
        # for the Walsh-Hadamard fast path; if that's not the case the
        # caller already decided to use V4 (the wiring layer guards
        # against it before reaching here), so we trust ``config.mode``
        # and raise loudly otherwise — a silent V4 downgrade would
        # break the "default-on at 0.95× decode" exit criterion.
        keys_compressed: (
            tuple[mx.array | None, mx.array | None, mx.array | None] | None
        ) = None
        out_keys: mx.array | None = keys
        if config.mode == "k8v4":
            signs = random_hadamard_signs(head_dim, config.rotation_seed)
            k_packed, k_norms, k_scales = turboquant_k8_encode(keys, signs)
            keys_compressed = (k_packed, k_norms, k_scales)
            out_keys = None

        return cls(
            keys=out_keys,
            values_compressed=(indices, scales, zeros),
            offset=offset,
            config=config,
            head_dim=head_dim,
            keys_compressed=keys_compressed,
        )

    def to_kv_cache(self):
        """Decompress back to a standard KVCache."""
        from mlx_lm.models.cache import KVCache

        kv = KVCache()

        if self.keys is None and self.keys_compressed is None:
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

        # Reconstruct K — either from FP16 storage (V4) or from the K8
        # packed indices + norms + scales (K8V4).
        if self.keys_compressed is not None:
            signs = random_hadamard_signs(self.head_dim, self.config.rotation_seed)
            k_packed, k_norms, k_scales = self.keys_compressed
            keys = turboquant_k8_decode(
                k_packed, k_norms, k_scales, signs, self.head_dim
            )
        else:
            keys = self.keys

        kv.keys = keys
        kv.values = values
        kv.offset = self.offset
        return kv

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> None:
        """Trim n tokens from the end."""
        if n <= 0:
            return
        new_offset = max(0, self.offset - n)
        if self.keys is not None:
            self.keys = self.keys[..., :new_offset, :]
        indices, scales, zeros = self.values_compressed
        self.values_compressed = (
            indices[..., :new_offset, :] if indices is not None else None,
            scales[..., :new_offset, :] if scales is not None else None,
            zeros[..., :new_offset, :] if zeros is not None else None,
        )
        # Trim the K8 side too: packed indices have a head_dim trailing
        # axis (NOT n_groups), norms / scales have a seq trailing axis.
        if self.keys_compressed is not None:
            k_packed, k_norms, k_scales = self.keys_compressed
            self.keys_compressed = (
                k_packed[..., :new_offset, :] if k_packed is not None else None,
                k_norms[..., :new_offset] if k_norms is not None else None,
                k_scales[..., :new_offset] if k_scales is not None else None,
            )
        self.offset = new_offset

    @property
    def memory_bytes(self) -> int:
        """Estimate memory usage in bytes.

        Stable across modes — the radix-index uses this for the K-side
        bytes-per-token estimate (R15 #303 stacking) and would mis-size
        a node if K8V4 reported only the V slab.
        """
        total = 0
        if self.keys is not None:
            total += self.keys.nbytes
        if self.keys_compressed is not None:
            for arr in self.keys_compressed:
                if arr is not None:
                    total += arr.nbytes
        for arr in self.values_compressed:
            if arr is not None:
                total += arr.nbytes
        return total
