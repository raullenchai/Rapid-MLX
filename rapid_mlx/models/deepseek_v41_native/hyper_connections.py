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

import logging
import threading
from functools import cache

import mlx.core as mx

logger = logging.getLogger(__name__)


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
def _hc_mixes_reference(
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


# Decode-time coefficient projection in two dispatches instead of ~16. The
# stock path materializes FP32 copies of the stream and of the checkpoint's
# BF16 projection, the square, the row sum, the GEMV, the scaled mixes, three
# slices, three split epilogues, the softmax and the Sinkhorn kernel; these
# kernels convert in registers instead (the conversions are exact). They
# keep the stock operation order of each step (the installed MLX's FP32 GEMV lane split for K >= 16 * N: eight
# simdgroups across K, four contiguous values per lane, shuffle-down tree,
# then the simdgroup partials summed in order; the 1024-thread row reduction
# for the sum of squares; the stock sigmoid, softmax and Sinkhorn arithmetic),
# so each row equals its own single-row stock result. A one-time self-check
# per configuration proves it on the running MLX/GPU; a difference, or a
# compile/launch failure during that check, keeps the stock path. After a
# configuration qualifies, its kernels run like any other MLX kernel: a later
# GPU execution failure surfaces at evaluation, exactly as a stock kernel
# failure would, and is not retried on the stock graph. Inputs are passed with
# the default row-contiguous copy, so strided streams, scales and bases reach
# the kernels densely packed.
HC_MIXES_ENABLED = True
HC_MIXES_MAX_ROWS = 8
_GEMV_SPLIT = 8  # simdgroups across K per output (stock lane split)
_SSQ_SIMDGROUPS = 32  # stock row reduction: 1024 threads, four values each
_FLOAT_TYPES = (mx.bfloat16, mx.float16, mx.float32)

_HC_MIXES_PARTIAL_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint group = thread_position_in_grid.y;
    const uint row = thread_position_in_grid.z;
    const device T* input = x + size_t(row) * K;
    if (group < N * SPLIT) {
        const uint out = group / SPLIT, split = group % SPLIT;
        const device W* weight = w + size_t(out) * K;
        float result = 0.0f;
        int offset = int(split * 32 + lane) * 4;
        for (int block = 0; block < K / (SPLIT * 128); ++block) {
            float values[4];
            float weights[4];
            for (int i = 0; i < 4; ++i) values[i] = float(input[offset + i]);
            for (int i = 0; i < 4; ++i) weights[i] = float(weight[offset + i]);
            for (int i = 0; i < 4; ++i) result += weights[i] * values[i];
            offset += SPLIT * 128;
        }
        for (ushort delta = 16; delta >= 1; delta >>= 1)
            result += simd_shuffle_down(result, delta);
        if (lane == 0) partial[(size_t(row) * SPLIT + split) * N + out] = result;
    } else {
        const uint simd = group - N * SPLIT;
        const device T* values = input + (simd * 32 + lane) * 4;
        float total = 0.0f;
        for (int block = 0; block < K / (SSQ * 128); ++block) {
            for (int i = 0; i < 4; ++i) {
                total = square_add(float(values[i]), total);
            }
            values += SSQ * 128;
        }
        total = simd_sum(total);
        if (lane == 0) squares[size_t(row) * SSQ + simd] = total;
    }
"""

_HC_MIXES_PARTIAL_HEADER = r"""
// The stock row reduction adds stored squares; never fuse them into an FMA.
static inline float square_add(float value, float total) {
    #pragma clang fp contract(off)
    const float square = value * value;
    return square + total;
}
"""

_HC_MIXES_FINISH_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.y;
    const float total = simd_sum(squares[size_t(row) * SSQ + lane]);
    if (lane != 0) return;
    const float r = rsqrt_ref(scale_add(total, inv_k[0], eps[1]));
    float m[N];
    for (int o = 0; o < N; ++o) {
        float acc = partial[size_t(row) * SPLIT * N + o];
        for (int s = 1; s < SPLIT; ++s) acc += partial[(size_t(row) * SPLIT + s) * N + o];
        m[o] = acc * r;
    }
    float scale[3];
    float base[N];
    for (int i = 0; i < 3; ++i) scale[i] = float(hc_scale[i]);
    for (int i = 0; i < N; ++i) base[i] = float(hc_base[i]);
    for (int i = 0; i < 4; ++i) {
        pre[row * 4 + i] = sigmoid_ref(scale_add(m[i], scale[0], base[i])) + eps[0];
        post[row * 4 + i] =
            2.0f * sigmoid_ref(scale_add(m[4 + i], scale[1], base[4 + i]));
    }
    float values[16];
    for (int r4 = 0; r4 < 4; ++r4) {
        float ld[4];
        for (int c = 0; c < 4; ++c)
            ld[c] = scale_add(m[8 + r4 * 4 + c], scale[2], base[8 + r4 * 4 + c]);
        float maxval = metal::numeric_limits<float>::lowest();
        for (int c = 0; c < 4; ++c) maxval = (maxval < ld[c]) ? ld[c] : maxval;
        float normalizer = 0.0f;
        for (int c = 0; c < 4; ++c) {
            ld[c] = metal::fast::exp(ld[c] - maxval);
            normalizer += ld[c];
        }
        normalizer = 1 / normalizer;
        for (int c = 0; c < 4; ++c) values[r4 * 4 + c] = ld[c] * normalizer + eps[0];
    }
    sinkhorn_ref(values, eps[0], ITERS);
    for (int i = 0; i < 16; ++i) comb[row * 16 + i] = values[i];
"""

_HC_MIXES_FINISH_HEADER = r"""
// Every intermediate below is a stored FP32 tensor on the stock path.
#pragma clang fp contract(off)
static inline float scale_add(float value, float scale, float bias) {
    return value * scale + bias;
}
static inline float rsqrt_ref(float x) { return metal::precise::rsqrt(x); }
static inline float sigmoid_ref(float x) {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
#pragma clang fp reassociate(off)
static inline void sinkhorn_ref(thread float* values, float eps, int iters) {
    for (int iteration = 0; iteration < iters; ++iteration) {
        if (iteration > 0) {
            for (int r = 0; r < 4; ++r) {
                float total = 0.0f;
                for (int c = 0; c < 4; ++c) total += values[r * 4 + c];
                total += eps;
                for (int c = 0; c < 4; ++c) values[r * 4 + c] /= total;
            }
        }
        for (int c = 0; c < 4; ++c) {
            float total = 0.0f;
            for (int r = 0; r < 4; ++r) total += values[r * 4 + c];
            total += eps;
            for (int r = 0; r < 4; ++r) values[r * 4 + c] /= total;
        }
    }
}
"""


@cache
def _hc_mixes_partial_kernel():
    return mx.fast.metal_kernel(
        name="rapid_deepseek_v41_hc_mixes_partial",
        input_names=["x", "w"],
        output_names=["partial", "squares"],
        source=_HC_MIXES_PARTIAL_SOURCE,
        header=_HC_MIXES_PARTIAL_HEADER,
    )


@cache
def _hc_mixes_finish_kernel():
    return mx.fast.metal_kernel(
        name="rapid_deepseek_v41_hc_mixes_finish",
        input_names=["partial", "squares", "hc_scale", "hc_base", "inv_k", "eps"],
        output_names=["pre", "post", "comb"],
        source=_HC_MIXES_FINISH_SOURCE,
        header=_HC_MIXES_FINISH_HEADER,
    )


def _fused_hc_mixes(
    x: mx.array,
    hc_fn: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
):
    lead = x.shape[:2]
    width = x.shape[-2] * x.shape[-1]
    rows = x.size // width
    outputs = hc_fn.shape[0]
    partial, squares = _hc_mixes_partial_kernel()(
        inputs=[x.reshape(rows, width), hc_fn],
        template=[
            ("T", x.dtype),
            ("W", hc_fn.dtype),
            ("K", width),
            ("N", outputs),
            ("SPLIT", _GEMV_SPLIT),
            ("SSQ", _SSQ_SIMDGROUPS),
        ],
        grid=(32, outputs * _GEMV_SPLIT + _SSQ_SIMDGROUPS, rows),
        threadgroup=(32, 8, 1),
        output_shapes=[(rows, _GEMV_SPLIT, outputs), (rows, _SSQ_SIMDGROUPS)],
        output_dtypes=[mx.float32, mx.float32],
    )
    pre, post, comb = _hc_mixes_finish_kernel()(
        inputs=[
            partial,
            squares,
            hc_scale,
            hc_base,
            mx.array([1.0 / width], dtype=mx.float32),
            mx.array([hc_eps, norm_eps], dtype=mx.float32),
        ],
        template=[
            ("N", outputs),
            ("SPLIT", _GEMV_SPLIT),
            ("SSQ", _SSQ_SIMDGROUPS),
            ("ITERS", max(1, sinkhorn_iters)),
        ],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, 4), (rows, 4), (rows, 16)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return (
        pre.reshape(*lead, 4),
        post.reshape(*lead, 4),
        comb.reshape(*lead, 4, 4),
    )


def _row_contiguous(a: mx.array) -> bool:
    """Whether ``a`` is stored row-major (the checkpoint layout).

    The stock GEMV picks its reduction order from the projection's layout and
    the fused kernel reproduces only the row-major one. Arrays whose storage
    cannot be inspected (placeholders inside an ``mx.compile`` trace) report
    False and take the stock graph.

    The projection is a loaded, already-evaluated weight, so ``mx.eval`` and
    the buffer export return without waiting on in-flight GPU work (measured
    under a 200 ms pending GPU graph on M3 Ultra: ~0.85 us per call, 80 calls
    per token ~70 us total). Only a still-lazy weight would be computed here.
    """
    try:
        mx.eval(a)  # a placeholder raises here, as a Python exception
        return memoryview(a).c_contiguous
    except Exception:
        return False


def _hc_mixes_supported(x, hc_fn, hc_scale, hc_base, hc_mult) -> bool:
    if not HC_MIXES_ENABLED or not _metal_fast_path_available():
        return False
    if x.ndim != 4 or hc_mult != 4 or x.shape[-2] != 4 or not x.size:
        return False
    width = 4 * x.shape[-1]
    rows = x.size // width
    return (
        1 <= rows <= HC_MIXES_MAX_ROWS
        and hc_fn.shape == (24, width)
        and hc_scale.shape == (3,)
        and hc_base.shape == (24,)
        and all(p.dtype in _FLOAT_TYPES for p in (x, hc_fn, hc_scale, hc_base))
        # the stock lane splits this order reproduces: GEMV for K >= 16 * N
        # with whole 1024-wide blocks, and whole 4096-wide reduction blocks
        and width % (_SSQ_SIMDGROUPS * 128) == 0
        and _row_contiguous(hc_fn)
    )


# (input dtypes, width, iters, norm_eps, hc_eps) -> the fused kernels match
# stock. The input dtypes and the width fix the kernel specializations.
_HC_MIXES_VERDICTS: dict = {}
_HC_MIXES_LOCK = threading.Lock()


def _hc_mixes_probe(x, hc_fn, hc_scale, hc_base):
    """Concrete inputs with the caller's layout and dtypes.

    The probe never reads the caller's arrays, so qualifying from inside an
    ``mx.compile`` trace evaluates no placeholder. Rows of growing magnitude
    reach both sigmoid branches and a wide spread of softmax inputs; the
    weights span the release scale of the coefficient projection.
    """
    keys = mx.random.split(mx.random.key(0x4C), 4)
    width = 4 * x.shape[-1]
    spread = mx.power(2.0, mx.arange(HC_MIXES_MAX_ROWS) - 3.0)
    probe = (
        mx.random.normal((HC_MIXES_MAX_ROWS, 1, 4, x.shape[-1]), key=keys[0])
        * spread[:, None, None, None]
    ).astype(x.dtype)
    fn = (mx.random.normal((24, width), key=keys[1]) * 0.05).astype(hc_fn.dtype)
    scale = (mx.random.uniform(0.25, 4.0, (3,), key=keys[2])).astype(hc_scale.dtype)
    base = (mx.random.normal((24,), key=keys[3])).astype(hc_base.dtype)
    return probe, fn, scale, base


def _hc_mixes_qualified(x, hc_fn, hc_scale, hc_base, iters, norm_eps, hc_eps):
    key = (
        x.dtype,
        hc_fn.dtype,
        hc_scale.dtype,
        hc_base.dtype,
        x.shape[-1],
        iters,
        norm_eps,
        hc_eps,
    )
    verdict = _HC_MIXES_VERDICTS.get(key)
    if verdict is not None:
        return verdict
    with _HC_MIXES_LOCK:
        if key in _HC_MIXES_VERDICTS:
            return _HC_MIXES_VERDICTS[key]
        verdict = False
        try:
            probe, fn, scale, base = _hc_mixes_probe(x, hc_fn, hc_scale, hc_base)
            expected = [
                _hc_mixes_reference(
                    probe[r : r + 1], fn, scale, base, 4, iters, norm_eps, hc_eps
                )
                for r in range(HC_MIXES_MAX_ROWS)
            ]
            actual = _fused_hc_mixes(probe, fn, scale, base, iters, norm_eps, hc_eps)
            verdict = all(
                mx.array_equal(actual[i][r : r + 1], expected[r][i]).item()
                for r in range(HC_MIXES_MAX_ROWS)
                for i in range(3)
            )
        except Exception:  # compile/launch failure on this runtime: stay stock
            logger.warning("fused V4.1 HC mixes unavailable", exc_info=True)
        if not verdict:
            logger.warning("fused V4.1 HC mixes disabled: stock path")
        _HC_MIXES_VERDICTS[key] = verdict
    return verdict


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
    """(pre, post, comb) for the stream; two Metal dispatches at decode shapes.

    Each decode row equals its own single-row stock result (see the kernels
    above); prefill and unsupported layouts take the stock graph."""
    if _hc_mixes_supported(
        x, hc_fn, hc_scale, hc_base, hc_mult
    ) and _hc_mixes_qualified(
        x, hc_fn, hc_scale, hc_base, sinkhorn_iters, norm_eps, hc_eps
    ):
        return _fused_hc_mixes(
            x, hc_fn, hc_scale, hc_base, sinkhorn_iters, norm_eps, hc_eps
        )
    return _hc_mixes_reference(
        x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, norm_eps, hc_eps
    )


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
