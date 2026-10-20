"""Core layers: RMSNorm, YaRN rope with the inverse path, clamped SwiGLU.

Transcribed from the reference ``model.py``. Details that matter:

* ``rms_norm_eps`` is ``1e-20`` in the release — effectively zero, kept exact;
* rope pairs *adjacent* elements as (real, imag) — not split halves — and is
  applied to the **last** ``rope_head_dim`` channels only;
* attention applies the rope **inverse** (conjugate) to its output before the
  grouped output projection, which is why the conjugate path exists.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from . import fusion
from .fusion import fuse


class RMSNorm(nn.Module):
    """RMSNorm computed in fp32, returning the input dtype (matches reference)."""

    def __init__(self, dim: int, eps: float = 1e-20):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        # Keep rsqrt eager: compiled CPU reciprocal approximations change fp32.
        dtype = x.dtype
        xf = x.astype(mx.float32)
        var = mx.mean(mx.square(xf), axis=-1, keepdims=True)
        xf = xf * mx.rsqrt(var + self.eps)
        return (self.weight * xf).astype(dtype)


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
):
    """YaRN-scaled rotary frequencies as a (cos, sin) pair of shape [seqlen, dim//2].

    ``original_seq_len == 0`` disables YaRN entirely — the reference uses that for
    the pure sliding-window layers (which also drop back to the base theta).
    """

    def corrected_dim(rotations):
        return (
            dim
            * math.log(original_seq_len / (rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    freqs = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    if original_seq_len > 0:
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = mx.clip(
            (mx.arange(dim // 2, dtype=mx.float32) - low) / max(high - low, 1e-3), 0, 1
        )
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = mx.arange(seqlen, dtype=mx.float32)
    ang = t[:, None] * freqs[None, :]
    return mx.cos(ang), mx.sin(ang)


@fuse
def apply_rotary_emb(
    x: mx.array, cos: mx.array, sin: mx.array, inverse: bool = False
) -> mx.array:
    """Rotate ``x`` [..., s, (h,) d] by adjacent-pair complex multiplication.

    ``cos``/``sin``: [s, d//2]. ``inverse=True`` conjugates — attention uses this
    on its output to remove the query rotation again.
    """
    dtype = x.dtype
    xf = x.astype(mx.float32)
    shape = xf.shape
    xf = xf.reshape(*shape[:-1], shape[-1] // 2, 2)
    xr, xi = xf[..., 0], xf[..., 1]

    if xr.ndim == 4:  # [b, s, h, d//2]
        c, s = cos[None, :, None, :], sin[None, :, None, :]
    else:  # [b, s, d//2]
        c, s = cos[None, :, :], sin[None, :, :]
    if inverse:
        s = -s

    out = mx.stack([xr * c - xi * s, xr * s + xi * c], axis=-1)
    return out.reshape(shape).astype(dtype)


def rope_tail(
    x: mx.array, rd: int, cos: mx.array, sin: mx.array, inverse: bool = False
) -> mx.array:
    """Apply rope to the last ``rd`` channels, leave the rest untouched."""
    if _rope_kernel_supports(x, rd, cos, sin):
        return _rope_tail_kernel(x, rd, cos, sin, inverse)
    return mx.concatenate(
        [x[..., :-rd], apply_rotary_emb(x[..., -rd:], cos, sin, inverse)], axis=-1
    )


# One thread per adjacent (real, imag) pair: untouched pairs are copied, tail
# pairs rotate with the reference's FP32 products and a single rounding to the
# input dtype — the same values as slice + rotate + concatenate in one launch.
_ROPE_SOURCE = r"""
    const uint gid = thread_position_in_grid.x;
    constexpr uint HALF = D / 2;
    if (gid >= ROWS * HALF) {
        return;
    }
    const uint row = gid / HALF;
    const uint pair = gid % HALF;
    const uint base = row * D + 2 * pair;
    constexpr uint FIRST_ROTATED = (D - RD) / 2;
    if (pair < FIRST_ROTATED) {
        out[base] = x[base];
        out[base + 1] = x[base + 1];
        return;
    }
    const uint freq = (row / HEADS) % SEQ * (RD / 2) + (pair - FIRST_ROTATED);
    const float c = cos_table[freq];
    const float s = INVERSE ? -sin_table[freq] : sin_table[freq];
    const float real = static_cast<float>(x[base]);
    const float imag = static_cast<float>(x[base + 1]);
    const float real_c = real * c;
    const float imag_s = imag * s;
    const float real_s = real * s;
    const float imag_c = imag * c;
    out[base] = static_cast<T>(real_c - imag_s);
    out[base + 1] = static_cast<T>(real_s + imag_c);
"""
_ROPE_THREADS = 256


@lru_cache(maxsize=1)
def _rope_kernel():
    return mx.fast.metal_kernel(
        name="rapid_v41_rope_tail",
        input_names=["x", "cos_table", "sin_table"],
        output_names=["out"],
        source=_ROPE_SOURCE,
        ensure_row_contiguous=True,
    )


def _rope_kernel_supports(x: mx.array, rd: int, cos: mx.array, sin: mx.array) -> bool:
    return (
        fusion.ENABLED
        and mx.default_device() == mx.gpu
        and x.ndim in (3, 4)
        and x.dtype in (mx.float32, mx.bfloat16, mx.float16)
        and 0 < rd <= x.shape[-1]
        and rd % 2 == 0
        and x.shape[-1] % 2 == 0
        and x.size > 0
        and cos.dtype == sin.dtype == mx.float32
        and cos.shape == sin.shape == (x.shape[1], rd // 2)
    )


def _rope_tail_kernel(
    x: mx.array, rd: int, cos: mx.array, sin: mx.array, inverse: bool
) -> mx.array:
    width = int(x.shape[-1])
    rows = x.size // width
    pairs = rows * width // 2
    (out,) = _rope_kernel()(
        inputs=[x, cos, sin],
        template=[
            ("T", x.dtype),
            ("D", width),
            ("RD", rd),
            ("HEADS", int(x.shape[2]) if x.ndim == 4 else 1),
            ("SEQ", int(x.shape[1])),
            ("ROWS", rows),
            ("INVERSE", bool(inverse)),
        ],
        grid=(pairs, 1, 1),
        threadgroup=(min(_ROPE_THREADS, pairs), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out
