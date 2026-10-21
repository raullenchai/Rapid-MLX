# SPDX-License-Identifier: MIT
# Modified by Rapid-MLX contributors for the native V4.1 runtime.
#
# The decode attention kernel adapts the simdgroup-matrix design of MIT-licensed
# code (see NOTICE in this directory), distributed under the following license:
#
# MIT License
#
# Copyright (c) 2023 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Fused single-query sparse attention for V4.1 decode.

One decode step of :func:`sparse_attention.sparse_attn` plus the attention
output's inverse rope tail, as one Metal dispatch instead of ~25 graph nodes
(window gather, index build, concatenations, gather, two einsums, the masked
softmax, the rope tail and the dtype casts). Keys are read in place: the window
from the cache ring (after this step's write), compressed rows from the kv
source's cache through the selected indices.

A speculative verify chunk (up to ``MAX_QUERY_ROWS`` tokens, within the window
ring) runs the same kernel with one threadgroup row per query row. Each query
row reads the keys its own decode step would see, so its result is
bit-identical to the one-row call at that position; keys that a later row of
the chunk has already overwritten in the ring are read from a snapshot of
those ring rows taken before the write (:func:`overwritten_rows`).

One threadgroup per (8-head group, batch row, query row) walks the keys in
64-key blocks.
Scores are 8x8 simdgroup matrix products (8 heads x 8 keys, over four quarters
of the head dimension), the softmax is online (running max seeded with the
sink, per-block correction), and the value sum is again 8x8 matrix products in
which each simdgroup owns a slice of the output dimensions. Each key row is
read once per 8 heads instead of once per head.

Every rounding point of the graph path is kept (cached keys rounded to the
query dtype, fp32 logits and weights, the query-dtype attention output, separately rounded rope products, query-dtype
rotated output). Only the fp32 accumulation order of the dot products and sums
differs, plus the online softmax's per-block rescaling (an extra rounded
``exp(m_prev - m)`` factor per block), so each sum is within a few
``K * 2^-24 * sum|terms|`` of the graph value.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from .kernels import custom_kernels_enabled


# Module switch: tests and A/B runs turn the fused path off to compare.
ENABLED = True

NEG_INF = -1e30
_HEADS_PER_GROUP = 8
_DIMS_PER_GROUP = 128

_KEY_ROW = r"""
// Cache row of key j for the query at position start (WE window keys, oldest
// first, then KC compressed keys). Window keys come from the ring after this
// call's writes; a key whose ring slot a later query row of the same call has
// already overwritten (position < LIVE0) comes from ``old``, the slots' prior
// contents, row ``position - (LIVE0 - NQ)``. Compressed keys are the rows
// selected by cidx (-1 when masked, or when the index falls outside the NC
// compressed rows, so no load can leave the buffer). ``src`` reports the
// buffer: 0 ring, 1 compressed, 2 old. Small inputs arrive in the constant
// address space, so the index pointer type is a template.
template <typename P>
inline int key_row(int j, int start, int WE, int OFF, int NC, int W, int LIVE0, int NQ, P ci, thread int& src) {
    if (j < WE) {
        const int position = start - WE + 1 + j;
        if (position < LIVE0) {
            src = 2;
            return position - (LIVE0 - NQ);
        }
        src = 0;
        return position % W;
    }
    src = 1;
    const int r = ci[j - WE] - OFF;
    return r >= 0 && r < NC ? r : -1;
}

template <typename T>
inline const device T* key_base(int src, const device T* ring, const device T* comp, const device T* old) {
    return src == 1 ? comp : (src == 2 ? old : ring);
}
"""

_ATTEND = r"""
    #pragma clang fp contract(off)
    constexpr int TPG = SL * 32;
    constexpr int CHAINS = 32;  // 8 key tiles x 4 quarters of D
    constexpr int NV = D / (8 * SL);
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint head0 = threadgroup_position_in_grid.x * 8;
    // one threadgroup per (head group, batch row, query row): query row qi of
    // a call that covers positions first .. first + NQ - 1
    const int NQ = meta[5];
    const uint b = threadgroup_position_in_grid.y / uint(NQ);
    const int qi = int(threadgroup_position_in_grid.y % uint(NQ));
    const uint qrow = b * uint(NQ) + uint(qi);
    const int first = meta[0], KC = meta[2], OFF = meta[3], NC = meta[4];
    const int start = first + qi, WE = min(W, start + 1);
    const int LIVE0 = first + NQ - W;  // oldest position still in the ring
    const int KT = WE + KC, NB = (KT + 63) / 64;
    // simdgroup 8x8 fragment: this lane holds row fm, columns fn and fn + 1
    const uint fm = ((lane >> 2) & 4) + ((lane >> 1) & 3);
    const uint fn = (((lane >> 2) & 2) << 1) + ((lane & 1) << 1);
    threadgroup float qs[8 * D];
    threadgroup float dots[CHAINS * 64];
    threadgroup float prob[8 * 64];
    threadgroup int rows[64];
    threadgroup int srcs[64];
    threadgroup float run_max[8], blk_max[8], corr[8], den_blk[8], den_run[8];
    // the query, its rope tail applied: products rounded separately, result
    // rounded to the query dtype
    for (uint i = tid; i < uint(8 * D / 2); i += TPG) {
        const uint hh = i / (D / 2), dim = (i % (D / 2)) * 2;
        const device auto* qh = q + (size_t(qrow) * H + head0 + hh) * D;
        float x0 = float(qh[dim]), x1 = float(qh[dim + 1]);
        if (dim >= uint(D - RD)) {
            const uint p = (dim - uint(D - RD)) / 2;
            const float c = cosv[qi * (RD / 2) + p], sn = sinv[qi * (RD / 2) + p];
            const float a = x0 * c, bb = x1 * sn;
            const float e = x0 * sn, f = x1 * c;
            x0 = float(OT(a - bb));
            x1 = float(OT(e + f));
        }
        qs[hh * D + dim] = x0;
        qs[hh * D + dim + 1] = x1;
    }
    // the sink joins the max from the start, as the graph's row max does
    if (tid < 8) { run_max[tid] = max(float(sink[head0 + tid]), -1e30f); den_run[tid] = 0.0f; }
    const uint slice = sg;  // PV: this simdgroup owns NV 8-wide dimension tiles
    simdgroup_matrix<float, 8, 8> acc[NV], pv[NV], a, bm;
    for (int v = 0; v < NV; ++v) {
        acc[v].thread_elements()[0] = 0.0f; acc[v].thread_elements()[1] = 0.0f;
    }
    const device auto* wk = win_kv + size_t(b) * W * D;
    const device auto* ck = comp_kv + size_t(b) * NC * D;
    const device auto* ok = old_kv + size_t(b) * NQ * D;
    for (int blk = 0; blk < NB; ++blk) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 64) {
            const int j = blk * 64 + int(tid);
            int src = 0;
            rows[tid] = j < KT ? key_row(j, start, WE, OFF, NC, W, LIVE0, NQ, cidx + qrow * KC, src) : -1;
            srcs[tid] = src;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // scores: chain = (key tile, quarter), 8 heads x 8 keys over D / 4 dims
        for (uint chain = sg; chain < uint(CHAINS); chain += SL) {
            const uint tile = chain / 4, quarter = chain % 4;
            const int r0 = rows[tile * 8 + fn], r1 = rows[tile * 8 + fn + 1];
            const device auto* k0 = key_base(srcs[tile * 8 + fn], wk, ck, ok) + size_t(max(r0, 0)) * D;
            const device auto* k1 = key_base(srcs[tile * 8 + fn + 1], wk, ck, ok) + size_t(max(r1, 0)) * D;
            simdgroup_matrix<float, 8, 8> score, qa, kb;
            score.thread_elements()[0] = 0.0f; score.thread_elements()[1] = 0.0f;
            for (int k = int(quarter) * (D / 4); k < int(quarter + 1) * (D / 4); k += 8) {
                qa.thread_elements()[0] = qs[fm * D + k + fn];
                qa.thread_elements()[1] = qs[fm * D + k + fn + 1];
                kb.thread_elements()[0] = r0 >= 0 ? float(OT(k0[k + fm])) : 0.0f;
                kb.thread_elements()[1] = r1 >= 0 ? float(OT(k1[k + fm])) : 0.0f;
                simdgroup_multiply_accumulate(score, qa, kb, score);
            }
            dots[chain * 64 + fm * 8 + fn] = score.thread_elements()[0];
            dots[chain * 64 + fm * 8 + fn + 1] = score.thread_elements()[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8 * 64; i += TPG) {
            const uint hh = i / 64, key = i % 64, base = (key / 8) * 4 * 64 + hh * 8 + key % 8;
            const float sum = ((dots[base] + dots[base + 64]) + dots[base + 128]) + dots[base + 192];
            prob[i] = rows[key] >= 0 ? sum * scale[0] : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 8) {
            float m = run_max[tid];
            for (int s = 0; s < 64; ++s) m = max(m, prob[tid * 64 + s]);
            blk_max[tid] = m;
            corr[tid] = metal::precise::exp(run_max[tid] - m);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8 * 64; i += TPG) {
            const float s = prob[i];
            prob[i] = s == -INFINITY ? 0.0f : metal::precise::exp(s - blk_max[i / 64]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 8) {
            float total = 0.0f;
            for (int s = 0; s < 64; ++s) total += prob[tid * 64 + s];
            den_blk[tid] = total;
        }
        // values: 8 heads x NV dimension tiles over the block's 8 key tiles
        for (int v = 0; v < NV; ++v) {
            pv[v].thread_elements()[0] = 0.0f; pv[v].thread_elements()[1] = 0.0f;
        }
        for (uint tile = 0; tile < 8; ++tile) {
            a.thread_elements()[0] = prob[fm * 64 + tile * 8 + fn];
            a.thread_elements()[1] = prob[fm * 64 + tile * 8 + fn + 1];
            const int r = rows[tile * 8 + fm];
            const device auto* kr = key_base(srcs[tile * 8 + fm], wk, ck, ok) + size_t(max(r, 0)) * D;
            for (int v = 0; v < NV; ++v) {
                const uint d = slice * (D / SL) + v * 8 + fn;
                bm.thread_elements()[0] = r >= 0 ? float(OT(kr[d])) : 0.0f;
                bm.thread_elements()[1] = r >= 0 ? float(OT(kr[d + 1])) : 0.0f;
                simdgroup_multiply_accumulate(pv[v], a, bm, pv[v]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float cf = corr[fm];
        for (int v = 0; v < NV; ++v) {
            acc[v].thread_elements()[0] = acc[v].thread_elements()[0] * cf + pv[v].thread_elements()[0];
            acc[v].thread_elements()[1] = acc[v].thread_elements()[1] * cf + pv[v].thread_elements()[1];
        }
        if (tid < 8) {
            den_run[tid] = den_run[tid] * corr[tid] + den_blk[tid];
            run_max[tid] = blk_max[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint h = head0 + fm;
    const float denom = den_run[fm] + metal::precise::exp(float(sink[h]) - run_max[fm]);
    for (int v = 0; v < NV; ++v) {
        const uint d = slice * (D / SL) + v * 8 + fn;
        float re = float(OT(acc[v].thread_elements()[0] / denom));
        float im = float(OT(acc[v].thread_elements()[1] / denom));
        // inverse rope on the tail pairs: (re, im) * conj(cos + i sin)
        if (d >= uint(D - RD)) {
            const uint p = (d - uint(D - RD)) / 2;
            const float c = cosv[qi * (RD / 2) + p], sn = -sinv[qi * (RD / 2) + p];
            const float a0 = re * c, b0 = im * sn;
            const float a1 = re * sn, b1 = im * c;
            const float r0 = a0 - b0, r1 = a1 + b1;
            re = float(OT(r0));
            im = float(OT(r1));
        }
        out[(size_t(qrow) * H + h) * D + d] = re;
        out[(size_t(qrow) * H + h) * D + d + 1] = im;
    }
"""


@lru_cache(maxsize=None)
def _kernels():
    attend_k = mx.fast.metal_kernel(
        name="rapid_v41_decode_attn",
        input_names=[
            "q",
            "win_kv",
            "comp_kv",
            "old_kv",
            "cidx",
            "meta",
            "scale",
            "sink",
            "cosv",
            "sinv",
        ],
        output_names=["out"],
        source=_ATTEND,
        header=_KEY_ROW + "#include <metal_simdgroup_matrix>\n",
    )
    return attend_k


@lru_cache(maxsize=None)
def _scalar(value: float) -> mx.array:
    a = mx.array([value], dtype=mx.float32)
    mx.eval(a)
    return a


# Query rows one call covers: decode (1) and speculative verify chunks.
MAX_QUERY_ROWS = 8


def supports(x: mx.array, win_kv: mx.array, heads: int, head_dim: int, rd: int) -> bool:
    """Whether the fused kernels cover this attention call: one decode token,
    or a speculative verify chunk of up to ``MAX_QUERY_ROWS`` tokens that fits
    in the window ring."""
    if not (
        ENABLED
        and custom_kernels_enabled()
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    ):
        return False
    return (
        x.ndim == 3
        and 1 <= x.shape[1] <= min(MAX_QUERY_ROWS, win_kv.shape[1])
        and x.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and win_kv.dtype in (mx.float32, mx.bfloat16, mx.float16)
        and win_kv.shape[0] == x.shape[0]
        and win_kv.shape[-1] == head_dim
        and heads % _HEADS_PER_GROUP == 0
        and head_dim % _DIMS_PER_GROUP == 0
        and head_dim <= 512
        and 0 < rd <= head_dim
        and rd % 2 == 0
    )


def overwritten_rows(win_kv: mx.array, start_pos: int, n: int) -> mx.array | None:
    """The ring rows an ``n``-token write at ``start_pos`` replaces, read before
    the write: earlier query rows of the same call still attend to them."""
    if n == 1:
        return None
    return win_kv[:, (start_pos + mx.arange(n)) % win_kv.shape[1]]


def attend(
    q: mx.array,
    win_kv: mx.array,
    start_pos: int,
    comp_kv: mx.array | None,
    cidx: mx.array | None,
    offset: int,
    sink: mx.array,
    cos: mx.array,
    sin: mx.array,
    scale: float,
    rd: int,
    old_kv: mx.array | None = None,
) -> mx.array:
    """Attention output of each query row, inverse-roped.

    q [b, n, h, d] (rope tail not yet applied) for positions start_pos ..
    start_pos + n - 1; win_kv [>=b, window, d], the ring after this call's
    writes; old_kv [b, n, d], :func:`overwritten_rows` before those writes
    (None when n == 1); comp_kv [>=b, n_comp, d] with cidx [b, n, k]
    concat-position indices (``offset`` = first compressed position, -1 =
    masked; an index outside [offset, offset + n_comp) is treated as masked),
    or None. cos/sin: these positions' rope rows, [n, rd // 2]. Returns
    [b, n, h, d] float32 holding q-dtype values (the graph path's rounding,
    cast for wo_a). Each query row is computed exactly as a one-row call at
    its position would compute it.
    """
    bsz, n, heads, dim = q.shape
    window = win_kv.shape[1]
    if not 1 <= n <= min(MAX_QUERY_ROWS, window):
        raise ValueError(
            f"query rows must be between 1 and min(MAX_QUERY_ROWS={MAX_QUERY_ROWS}, "
            f"window={window}); got {n}"
        )
    we = min(window, start_pos + 1)
    if n > 1 and (old_kv is None or old_kv.shape[1] != n):
        raise ValueError("a multi-row call needs the n ring rows it overwrites")
    if comp_kv is None or cidx is None or cidx.shape[-1] == 0:
        kc, n_comp = 0, 0
        comp_kv = win_kv  # never read: every key index is below WE
        cidx = mx.zeros((1,), dtype=mx.int32)
    else:
        kc, n_comp = cidx.shape[-1], comp_kv.shape[1]
        cidx = cidx.reshape(bsz * n, kc).astype(mx.int32)
    if old_kv is None:
        old_kv = win_kv  # never read: one row overwrites nothing it attends to
    meta = mx.array([start_pos, we, kc, offset, n_comp, n], dtype=mx.int32)
    # dimension slices: one simdgroup per slice, each owning whole 8-wide tiles
    slices = 32
    while (dim // 8) % slices:
        slices //= 2
    attend_k = _kernels()
    out = attend_k(
        inputs=[
            q,
            win_kv,
            comp_kv,
            old_kv,
            cidx,
            meta,
            _scalar(scale),
            sink,
            cos,
            sin,
        ],
        template=[
            ("OT", q.dtype),
            ("H", heads),
            ("D", dim),
            ("RD", rd),
            ("W", window),
            ("SL", slices),
        ],
        grid=(heads // _HEADS_PER_GROUP * slices * 32, bsz * n, 1),
        threadgroup=(slices * 32, 1, 1),
        output_shapes=[(bsz * n * heads * dim,)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(bsz, n, heads, dim)
