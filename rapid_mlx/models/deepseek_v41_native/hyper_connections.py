# SPDX-License-Identifier: MIT
# Modified by Rapid-MLX contributors for the native V4.1 runtime.
"""Hyper-Connections — the residual stream is ``hc_mult`` (=4) parallel copies.

V4.1 keeps V4's Sinkhorn machinery but **staggers** the coefficients: the mixes a
sub-layer computes are consumed one sub-layer *later*. From the reference
``Block.forward``:

* ``hc_mixes`` on the attention input yields ``(attn_pre, attn_post, attn_comb)``;
  attention's own ``hc_pre`` uses the ``pre`` produced by the *previous* layer's
  FFN (identity one-hot on copy 0 at the very first layer), while ``attn_post`` /
  ``attn_comb`` are used by attention's ``hc_post`` immediately;
* ``attn_pre`` then collapses the FFN input, whose own mixes hand ``ffn_pre`` to
  the *next* layer — and after the last layer, to the LM head's final collapse.

``comb`` is pushed toward doubly-stochastic by 20 Sinkhorn sweeps.
``hc_post`` computes ``out[k] = post[k]*x + sum_j comb[j,k]*residual[j]`` —
the residual is indexed by the **summed** axis j (comb's first index), a
known bug magnet; broadcasting residual onto k instead stays finite and
plausible but wrong.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx


def _metal_fast_path_available() -> bool:
    """Return whether custom Metal kernels can execute on the active device."""
    return (
        mx.default_device() == mx.gpu
        and hasattr(mx, "metal")
        and mx.metal.is_available()
    )


_SINKHORN_SOURCE = r"""
    const uint row = thread_position_in_grid.x;
    if (row >= ROWS) return;
    float values[16];
    for (int i = 0; i < 16; ++i) values[i] = x[row * 16 + i];
    for (int iteration = 0; iteration < ITERS; ++iteration) {
        if (iteration > 0) {
            for (int r = 0; r < 4; ++r) {
                float total = 0.0f;
                for (int c = 0; c < 4; ++c) total += values[r * 4 + c];
                total += eps[0];
                for (int c = 0; c < 4; ++c) values[r * 4 + c] /= total;
            }
        }
        for (int c = 0; c < 4; ++c) {
            float total = 0.0f;
            for (int r = 0; r < 4; ++r) total += values[r * 4 + c];
            total += eps[0];
            for (int r = 0; r < 4; ++r) values[r * 4 + c] /= total;
        }
    }
    for (int i = 0; i < 16; ++i) y[row * 16 + i] = values[i];
"""


@cache
def _sinkhorn_kernel():
    return mx.fast.metal_kernel(
        name="rapid_deepseek_v41_sinkhorn",
        input_names=["x", "eps"],
        output_names=["y"],
        source=_SINKHORN_SOURCE,
        header="#pragma clang fp reassociate(off)\n#pragma clang fp contract(off)\n",
    )


def _sinkhorn_reference(comb: mx.array, eps: float, iters: int) -> mx.array:
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return comb


def _sinkhorn(comb: mx.array, eps: float, iters: int) -> mx.array:
    if (
        comb.shape[-2:] != (4, 4)
        or comb.dtype != mx.float32
        or not comb.size
        or not _metal_fast_path_available()
    ):
        return _sinkhorn_reference(comb, eps, iters)
    rows = comb.size // 16
    return _sinkhorn_kernel()(
        inputs=[comb, mx.array([eps], mx.float32)],
        template=[("ROWS", rows), ("ITERS", max(1, iters))],
        grid=(rows, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[comb.shape],
        output_dtypes=[mx.float32],
    )[0]


@mx.compile
def split_sinkhorn(
    mixes: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Split one projection into (pre, post, comb) — transcribed from
    ``hc_split_sinkhorn_kernel``. Layout: first hc entries -> pre, next hc ->
    post, remaining hc*hc -> comb row-major."""
    hc = hc_mult
    m = mixes.astype(mx.float32)
    scale = hc_scale.astype(mx.float32)
    base = hc_base.astype(mx.float32)

    pre = mx.sigmoid(m[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * mx.sigmoid(m[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])

    comb = m[..., 2 * hc :] * scale[2] + base[2 * hc :]
    comb = comb.reshape(*comb.shape[:-1], hc, hc)
    # row softmax + eps, one column normalization, then (iters-1) full sweeps;
    # the eps sits inside the divisions, as in the kernel
    comb = mx.softmax(comb, axis=-1) + eps
    comb = _sinkhorn(comb, eps, sinkhorn_iters)
    return pre, post, comb


@mx.compile
def hc_mixes(
    x: mx.array,
    hc_fn: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
):
    """The coefficient projection: RMS-normalize the flattened [b,s,hc*d] stream
    (rsqrt applied AFTER the linear, matching the reference's operation order),
    project, split. Returns (pre, post, comb)."""
    xf = x.reshape(*x.shape[:2], -1).astype(mx.float32)
    rsqrt = mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + norm_eps)
    mixes = (xf @ hc_fn.astype(mx.float32).T) * rsqrt
    return split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)


@mx.compile
def hc_pre(x: mx.array, pre_mix: mx.array) -> mx.array:
    """Collapse the hc copies into one: [b,s,hc,d] x [b,s,hc] -> [b,s,d], fp32 sum."""
    y = mx.sum(pre_mix[..., None].astype(mx.float32) * x.astype(mx.float32), axis=2)
    return y.astype(x.dtype)


@mx.compile
def _hc_post_reference(
    x: mx.array, residual: mx.array, post: mx.array, comb: mx.array
) -> mx.array:
    """out[k] = post[k]*x + sum_j comb[j,k]*residual[j].

    x [b,s,d], residual [b,s,hc,d], post [b,s,hc], comb [b,s,hc,hc] -> [b,s,hc,d].
    The residual must sit on the j (summed) axis of the product.
    """
    prod = comb[..., None] * residual[..., :, None, :]  # [b, s, j, k, d]
    out = post[..., None] * x[..., None, :] + mx.sum(prod, axis=2)
    return out.astype(x.dtype)


_HC_POST_SOURCE = r"""
    const uint z = thread_position_in_grid.x;
    if (z >= ROWS * D) return;
    const uint row = z / D, d = z % D;
    float residual_values[4];
    for (uint i = 0; i < 4; ++i)
        residual_values[i] = float(residual[(row * 4 + i) * D + d]);
    const float value = float(x[z]);
    for (uint j = 0; j < 4; ++j) {
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i)
            sum = fma(comb[row * 16 + i * 4 + j], residual_values[i], sum);
        y[(row * 4 + j) * D + d] = T(post[row * 4 + j] * value + sum);
    }
"""


@cache
def _hc_post_kernel():
    return mx.fast.metal_kernel(
        name="rapid_deepseek_v41_hc_post",
        input_names=["x", "residual", "post", "comb"],
        output_names=["y"],
        source=_HC_POST_SOURCE,
        header="#pragma clang fp contract(off)\n",
    )


def _fused_hc_post(
    x: mx.array, residual: mx.array, post: mx.array, comb: mx.array
) -> mx.array:
    return _hc_post_kernel()(
        inputs=[x, residual, post, comb],
        template=[("T", x.dtype), ("ROWS", x.size // x.shape[-1]), ("D", x.shape[-1])],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[residual.shape],
        output_dtypes=[x.dtype],
    )[0]


def hc_post(
    x: mx.array, residual: mx.array, post: mx.array, comb: mx.array
) -> mx.array:
    """Expand into four residual streams, using Metal for the release layout."""
    if (
        x.ndim == 3
        and x.size
        and x.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and residual.dtype == x.dtype
        and residual.shape == (*x.shape[:-1], 4, x.shape[-1])
        and post.shape == (*x.shape[:-1], 4)
        and comb.shape == (*x.shape[:-1], 4, 4)
        and post.dtype == mx.float32
        and comb.dtype == mx.float32
        and _metal_fast_path_available()
    ):
        return _fused_hc_post(x, residual, post, comb)
    return _hc_post_reference(x, residual, post, comb)


_HC_PRE_NORM_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    threadgroup float sums[8];
    float values[(D + 255) / 256];
    float total = 0.0f;
    for (uint t = 0; t < (D + 255) / 256; ++t) {
        const uint d = tid + 256 * t;
        float value = 0.0f;
        if (d < D) {
            for (uint i = 0; i < 4; ++i)
                value = fma(float(x[(row * 4 + i) * D + d]), pre[row * 4 + i], value);
            value = float(T(value));
        }
        values[t] = value;
        total += value * value;
    }
    total = simd_sum(total);
    if (lane == 0) sums[simd] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float sum = lane < 8 ? sums[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0) sums[0] = rsqrt(sum / float(D) + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < (D + 255) / 256; ++t) {
        const uint d = tid + 256 * t;
        if (d < D) y[row * D + d] = T((values[t] * sums[0]) * float(weight[d]));
    }
"""


@cache
def _hc_pre_norm_kernel():
    return mx.fast.metal_kernel(
        name="rapid_deepseek_v41_hc_pre_norm",
        input_names=["x", "pre", "weight", "eps"],
        output_names=["y"],
        source=_HC_PRE_NORM_SOURCE,
        header="#pragma clang fp contract(off)\n",
    )


def _fused_hc_pre_norm(
    x: mx.array, pre: mx.array, weight: mx.array, eps: float
) -> mx.array:
    width = x.shape[-1]
    rows = x.size // (4 * width)
    return _hc_pre_norm_kernel()(
        inputs=[x, pre, weight, mx.array([eps], mx.float32)],
        template=[("T", x.dtype), ("D", width)],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*x.shape[:-2], width)],
        output_dtypes=[x.dtype],
    )[0]


@mx.compile
def _hc_pre_norm_reference(
    x: mx.array, pre: mx.array, weight: mx.array, eps: float
) -> mx.array:
    h = hc_pre(x, pre)
    hf = h.astype(mx.float32)
    variance = mx.mean(mx.square(hf), axis=-1, keepdims=True)
    return (weight * hf * mx.rsqrt(variance + eps)).astype(h.dtype)


def hc_pre_norm(x: mx.array, pre: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Collapse the four streams and RMS-normalize in one Metal dispatch."""
    if (
        x.ndim == 4
        and x.size
        and x.shape[-2] == 4
        and x.shape[-1] <= 8192
        and x.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and pre.dtype == mx.float32
        and pre.shape == x.shape[:-1]
        and weight.shape == (x.shape[-1],)
        and weight.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and _metal_fast_path_available()
    ):
        return _fused_hc_pre_norm(x, pre, weight, eps)
    return _hc_pre_norm_reference(x, pre, weight, eps)


def make_identity_pre_mix(b: int, s: int, hc_mult: int) -> mx.array:
    """The initial one-hot mix: copy 0 only."""
    m = mx.zeros((b, s, hc_mult), dtype=mx.float32)
    return mx.concatenate([mx.ones((b, s, 1), dtype=mx.float32), m[..., 1:]], axis=-1)
