# Copyright © 2025 Apple Inc.
# SPDX-License-Identifier: MIT
# Adapted from the installed mlx_lm/quant/gptq.py.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Column-ordered affine GPTQ, with native packing including three-bit words."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class QuantConfig:
    group_size: int = 64
    block_size: int = 128
    damping: float = 0.1
    min_tokens: int = 2048
    low_token_damping: float = 1.0

    def __post_init__(self):
        if type(self.group_size) is not int or self.group_size not in (32, 64, 128):
            raise ValueError("unsupported affine group size")
        if (
            type(self.block_size) is not int
            or self.block_size < self.group_size
            or self.block_size % self.group_size
        ):
            raise ValueError("block size must be a multiple of group size")
        if (
            type(self.min_tokens) is not int
            or type(self.damping) not in (int, float)
            or type(self.low_token_damping) not in (int, float)
            or not np.isfinite(self.damping)
            or self.damping <= 0
            or not np.isfinite(self.low_token_damping)
            or self.low_token_damping < self.damping
            or self.min_tokens < 0
        ):
            raise ValueError("invalid damping or minimum routed-token count")


def pack_codes(codes, bits):
    """MLX affine layout is a little-endian contiguous bit stream per row.

    In particular 32 three-bit codes occupy three U32s, without padding every
    ten codes. Using the power-of-two pack factor here silently corrupts QMV.
    """
    q = np.asarray(codes, dtype=np.uint32)
    if bits not in (2, 3, 4, 6, 8) or q.shape[-1] * bits % 32:
        raise ValueError("nonintegral native affine packing")
    if np.any(q >= 2**bits):
        raise ValueError("code outside quantizer range")
    # Eight codes at a time fit into uint64 for every supported width.
    v = q.reshape(*q.shape[:-1], -1, 8).astype(np.uint64)
    v = np.bitwise_or.reduce(v << (np.arange(8, dtype=np.uint64) * bits), axis=-1)
    raw = (v[..., None] >> (np.arange(bits, dtype=np.uint64) * 8)) & 255
    raw = np.ascontiguousarray(raw.astype(np.uint8)).reshape(*q.shape[:-1], -1)
    return mx.array(raw.view("<u4"))


def gram(chunks, dim, *, check=lambda: None):
    """Unnormalized weighted Gram; chunks yield (rows, nonnegative weights)."""
    h = mx.zeros((dim, dim), mx.float32)
    count = 0
    for x, weights in chunks:
        x = mx.array(x).astype(mx.float32).reshape(-1, dim)
        count += len(x)
        if weights is not None:
            weights = mx.array(weights).astype(mx.float32).reshape(-1)
            if not bool(mx.all(mx.isfinite(weights) & (weights >= 0)).item()):
                raise ValueError("invalid routing weight")
            x = x * mx.sqrt(weights[:, None])
        h = h + x.T @ x
        mx.eval(h)
        check()
    return h, count


def inverse_factor(h, config, count):
    """Upper Cholesky factor of the damped inverse, factored in float64 on the host.

    NumPy rather than an MLX CPU stream: freeing Metal buffers from the MLX CPU
    scheduler thread segfaults intermittently on Apple M5 (MLX 0.32.4).
    """
    h = np.array(h.astype(mx.float32), dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("Hessian must be square")
    diag = np.diag(h).copy()
    if not np.all(np.isfinite(h)) or np.any(diag < 0):
        raise ValueError("nonfinite or negative Hessian")
    dead = diag <= 0
    coefficient = (
        config.low_token_damping if count < config.min_tokens else config.damping
    )
    damp = max(coefficient * float(np.mean(diag)), 1e-6)
    h[np.diag_indices_from(h)] += damp
    lower_inv = np.linalg.inv(np.linalg.cholesky(h))
    upper = np.linalg.cholesky(lower_inv.T @ lower_inv).T
    return (
        mx.array(upper.astype(np.float32)),
        mx.array(dead),
        dict(
            tokens=count,
            low_token=count < config.min_tokens,
            damping_coefficient=coefficient,
            damping_absolute=damp,
            dead_columns=int(np.sum(dead)),
        ),
    )


def gptq(
    weight,
    h,
    bits,
    config=QuantConfig(),
    *,
    count=2048,
    check=lambda: None,
    factor=None,
):
    """Lazy block GPTQ. Refit each group's grid on the error-updated weights.

    Keep integer codes explicitly: requantizing reconstructed weights refits the
    grids and discards the error compensation. Only packing runs on the host.
    Scales/biases use the final BF16 storage precision during rounding itself.
    """
    if bits not in (2, 3, 4, 8):
        raise ValueError("GPTQ supports 2/3/4/8 bits")
    rows, cols = weight.shape
    if cols % config.group_size or h.shape != (cols, cols):
        raise ValueError("weight/Hessian shape or group mismatch")
    upper, dead, stats = inverse_factor(h, config, count) if factor is None else factor
    stats = dict(stats)
    w = weight.astype(mx.float32)
    # Unobserved coordinates have no calibration loss; zero them as in GPTQ.
    w = mx.where(dead[None], 0, w)
    codes = np.empty((rows, cols), np.uint32)
    scales, biases = [], []
    for start in range(0, cols, config.block_size):
        end = min(cols, start + config.block_size)
        block = mx.array(w[:, start:end])
        errors = mx.zeros_like(block)
        for local in range(end - start):
            col = start + local
            if col % config.group_size == 0:
                _, s, b = mx.quantize(
                    block[:, local : local + config.group_size].astype(mx.bfloat16),
                    group_size=config.group_size,
                    bits=bits,
                )
                scales.append(s)
                biases.append(b)
                s, b = s.astype(mx.float32), b.astype(mx.float32)
            value = block[:, local : local + 1]
            code = mx.clip(mx.round((value - b) / s), 0, 2**bits - 1)
            rounded = s * code + b
            error = (value - rounded) / upper[col, col]
            block[:, local:] -= error @ upper[col : col + 1, col:end]
            errors[:, local : local + 1] = error
            mx.eval(block, errors, code)
            codes[:, col] = np.asarray(code).ravel()
        w[:, end:] -= errors @ upper[start:end, end:]
        mx.eval(w)
        check()
    packed = (
        pack_codes(codes, bits),
        mx.concatenate(scales, axis=-1),
        mx.concatenate(biases, axis=-1),
    )
    mx.eval(*packed)
    return packed, stats


def gptq_batched(
    weights,
    factors,
    bits,
    config=QuantConfig(),
    *,
    dead=None,
    check=lambda: None,
):
    """Independent GPTQ fits with a leading expert dimension.

    ``factors`` contains upper inverse factors from :func:`inverse_factor`;
    factorization deliberately remains NumPy float64, never MLX CPU Cholesky.
    Pass its dead masks as ``dead``. The returned native packed triple has a
    leading expert axis. Statistics remain owned by each factor's caller.
    Only the block error propagation is a batched matrix multiply. Group grids
    and column rounding follow ``gptq`` including BF16 scale/bias precision.
    """
    if bits not in (2, 3, 4, 8):
        raise ValueError("GPTQ supports 2/3/4/8 bits")
    if weights.ndim != 3 or weights.shape[0] < 1:
        raise ValueError("weights must have shape [experts, rows, cols]")
    experts, rows, cols = weights.shape
    if cols % config.group_size or factors.shape != (experts, cols, cols):
        raise ValueError("weight/factor shape or group mismatch")
    if dead is None:
        dead = mx.zeros((experts, cols), mx.bool_)
    if dead.shape != (experts, cols):
        raise ValueError("dead mask must have shape [experts, cols]")
    w = mx.where(dead[:, None, :], 0, weights.astype(mx.float32))
    codes = np.empty((experts, rows, cols), np.uint32)
    scales, biases = [], []
    for start in range(0, cols, config.block_size):
        end = min(cols, start + config.block_size)
        block = mx.array(w[:, :, start:end])
        errors = mx.zeros_like(block)
        block_codes = mx.zeros(block.shape, mx.uint32)
        for local in range(end - start):
            col = start + local
            if col % config.group_size == 0:
                _, s, b = mx.quantize(
                    block[:, :, local : local + config.group_size].astype(mx.bfloat16),
                    group_size=config.group_size,
                    bits=bits,
                )
                scales.append(s)
                biases.append(b)
                s, b = s.astype(mx.float32), b.astype(mx.float32)
            value = block[:, :, local : local + 1]
            code = mx.clip(mx.round((value - b) / s), 0, 2**bits - 1)
            rounded = s * code + b
            error = (value - rounded) / factors[:, col, col][:, None, None]
            block[:, :, local:] -= error @ factors[:, col : col + 1, col:end]
            errors[:, :, local : local + 1] = error
            block_codes[:, :, local : local + 1] = code.astype(mx.uint32)
            mx.eval(block, errors, block_codes)
        codes[:, :, start:end] = np.asarray(block_codes)
        w[:, :, end:] -= errors @ factors[:, start:end, end:]
        mx.eval(w)
        check()
    packed = (
        pack_codes(codes, bits),
        mx.concatenate(scales, axis=-1),
        mx.concatenate(biases, axis=-1),
    )
    mx.eval(*packed)
    return packed


def relative_error(weight, packed, h, bits, group_size=64):
    """Squared projection-output error / source energy for the supplied Gram."""
    q = mx.dequantize(*packed, group_size=group_size, bits=bits, dtype=mx.float32)
    w = weight.astype(mx.float32)
    delta = w - q
    numerator = mx.sum((delta @ h) * delta)
    denominator = mx.sum((w @ h) * w)
    mx.eval(numerator, denominator)
    return float(numerator.item() / max(denominator.item(), 1e-30))
