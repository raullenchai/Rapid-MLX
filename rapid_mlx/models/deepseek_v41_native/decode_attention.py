# SPDX-License-Identifier: MIT
# Modified by Rapid-MLX contributors for the native V4.1 runtime.
"""Fused single-query sparse attention for V4.1 decode.

One decode step of :func:`sparse_attention.sparse_attn` plus the attention
output's inverse rope tail, as two Metal dispatches instead of ~25 graph nodes
(window gather, index build, concatenations, gather, two einsums, the masked
softmax, the rope tail and the dtype casts). Keys are read in place: the window
from the cache ring (after this step's write), compressed rows from the kv
source's cache through the selected indices.

* ``scores``: one threadgroup per (32-key block, 8-head group, batch row); each
  simdgroup owns one head and reduces the q.k dots of the block, writing
  ``scale * q.k`` (or ``-inf`` for masked keys).
* ``combine``: one threadgroup per (128 output dims, head, batch row); it takes
  the row max (with the sink and the ``-1e30`` floor), the exponentials, the
  denominator with ``exp(sink - max)``, the weighted value sum, the division,
  the bf16 rounding, and the inverse rope of the tail pairs.

Every rounding point of the graph path is kept (cached keys rounded to the
query dtype, fp32 logits and weights, the query-dtype attention output, separately rounded rope products, query-dtype
rotated output). Only the fp32 accumulation order of the dot products and sums
differs, so each sum is within ``K * 2^-24 * sum|terms|`` of the graph value.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

# Module switch: tests and A/B runs turn the fused path off to compare.
ENABLED = True

NEG_INF = -1e30
_HEADS_PER_GROUP = 8
_KEYS_PER_BLOCK = 32
_DIMS_PER_GROUP = 128

_KEY_ROW = r"""
// Cache row of key j: the window ring for j < WE (oldest first), else the
// compressed row selected by cidx (or -1 when masked). Small inputs arrive in
// the constant address space, so the index pointer type is a template.
template <typename P>
inline int key_row(int j, int start, int WE, int OFF, int W, P ci, thread bool& compressed) {
    compressed = j >= WE;
    if (!compressed) return (start - WE + 1 + j) % W;
    const int c = ci[j - WE];
    return c >= 0 ? c - OFF : -1;
}
"""

_SCORES = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint blk = threadgroup_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.z;
    const uint h = threadgroup_position_in_grid.y * 8 + sg;
    const int start = meta[0], WE = meta[1], KC = meta[2], OFF = meta[3], NC = meta[4];
    const int KT = WE + KC;
    threadgroup int rows[32];
    threadgroup bool comp[32];
    if (sg == 0) {
        bool c;
        const int j = int(blk) * 32 + int(lane);
        rows[lane] = j < KT ? key_row(j, start, WE, OFF, W, cidx + b * KC, c) : -1;
        comp[lane] = j < KT && c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qv[DPL];
    const device auto* qh = q + (size_t(b) * H + h) * D + lane * DPL;
    for (int i = 0; i < DPL; ++i) qv[i] = float(qh[i]);
    for (int k = 0; k < 32; ++k) {
        const int j = int(blk) * 32 + k;
        if (j >= KT) break;
        const int row = rows[k];
        float dot = 0.0f;
        if (row >= 0) {
            const device auto* kr = comp[k]
                ? comp_kv + (size_t(b) * NC + row) * D + lane * DPL
                : win_kv + (size_t(b) * W + row) * D + lane * DPL;
            for (int i = 0; i < DPL; ++i) dot += qv[i] * float(OT(kr[i]));
        }
        dot = simd_sum(dot);
        if (lane == 0)
            scores[(size_t(b) * H + h) * KP + j] = row >= 0 ? dot * scale[0] : -INFINITY;
    }
"""

_COMBINE = r"""
    #pragma clang fp contract(off)
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint h = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const uint d = threadgroup_position_in_grid.x * 128 + tid;
    const int start = meta[0], WE = meta[1], KC = meta[2], OFF = meta[3], NC = meta[4];
    const int KT = WE + KC;
    threadgroup float w[KMAX];
    threadgroup int rows[KMAX];
    threadgroup bool comp[KMAX];
    threadgroup float red[4];
    const device float* s = scores + (size_t(b) * H + h) * KP;
    float m = -INFINITY;
    for (int j = int(tid); j < KT; j += 128) {
        bool c;
        rows[j] = key_row(j, start, WE, OFF, W, cidx + b * KC, c);
        comp[j] = c;
        m = max(m, s[j]);
    }
    m = simd_max(m);
    if (lane == 0) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = max(max(red[0], red[1]), max(red[2], red[3]));
    const float sink_h = sink[h];
    m = max(max(m, sink_h), -1e30f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (int j = int(tid); j < KT; j += 128) {
        const float p = s[j] == -INFINITY ? 0.0f : metal::precise::exp(s[j] - m);
        w[j] = p;
        total += p;
    }
    total = simd_sum(total);
    if (lane == 0) red[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denom = ((red[0] + red[1]) + (red[2] + red[3])) + metal::precise::exp(sink_h - m);
    float acc = 0.0f;
    for (int j = 0; j < KT; ++j) {
        const int row = rows[j];
        if (row < 0) continue;
        const device auto* kr = comp[j] ? comp_kv + (size_t(b) * NC + row) * D
                                        : win_kv + (size_t(b) * W + row) * D;
        acc += w[j] * float(OT(kr[d]));
    }
    float v = float(OT(acc / denom));
    // inverse rope on the tail pairs: (re, im) * conj(cos + i sin)
    const float partner = simd_shuffle_xor(v, 1u);
    if (d >= uint(D - RD)) {
        const uint p = (d - uint(D - RD)) / 2;
        const float c = cosv[p], sn = -sinv[p];
        const bool odd = (d & 1u) != 0u;
        const float re = odd ? partner : v, im = odd ? v : partner;
        float r;
        if (odd) { const float a = re * sn; const float bb = im * c; r = a + bb; }
        else { const float a = re * c; const float bb = im * sn; r = a - bb; }
        v = float(OT(r));
    }
    out[(size_t(b) * H + h) * D + d] = v;
"""


@lru_cache(maxsize=None)
def _kernels():
    scores = mx.fast.metal_kernel(
        name="rapid_v41_decode_attn_scores",
        input_names=["q", "win_kv", "comp_kv", "cidx", "meta", "scale"],
        output_names=["scores"],
        source=_SCORES,
        header=_KEY_ROW,
    )
    combine = mx.fast.metal_kernel(
        name="rapid_v41_decode_attn_combine",
        input_names=["scores", "win_kv", "comp_kv", "cidx", "meta", "sink", "cosv", "sinv"],
        output_names=["out"],
        source=_COMBINE,
        header=_KEY_ROW,
    )
    return scores, combine


@lru_cache(maxsize=None)
def _scalar(value: float) -> mx.array:
    a = mx.array([value], dtype=mx.float32)
    mx.eval(a)
    return a


def supports(q: mx.array, win_kv: mx.array, rd: int, heads: int) -> bool:
    """Whether the fused kernels cover this decode step (else: the graph path)."""
    if not (ENABLED and mx.default_device() == mx.gpu):
        return False
    if q.ndim != 4 or q.shape[1] != 1 or q.shape[2] != heads:
        return False
    d = q.shape[-1]
    return (
        q.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and win_kv.dtype in (mx.float32, mx.bfloat16, mx.float16)
        and win_kv.shape[0] >= q.shape[0]
        and win_kv.shape[-1] == d
        and heads % _HEADS_PER_GROUP == 0
        and d % _DIMS_PER_GROUP == 0
        and 0 < rd <= d
        and rd % 2 == 0
    )


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
) -> mx.array:
    """Attention output of one decode query per batch row, inverse-roped.

    q [b, 1, h, d] (rope applied); win_kv [>=b, window, d], the ring after this
    step's write; comp_kv [>=b, n_comp, d] with cidx [b, 1, k] concat-position
    indices (``offset`` = first compressed position, -1 = masked), or None.
    cos/sin: this position's rope row, [1, rd // 2]. Returns [b, 1, h, d] in
    q's dtype.
    """
    bsz, _, heads, dim = q.shape
    window = win_kv.shape[1]
    we = min(window, start_pos + 1)
    if comp_kv is None or cidx is None or cidx.shape[-1] == 0:
        kc, n_comp = 0, 0
        comp_kv = win_kv  # never read: every key index is below WE
        cidx = mx.zeros((1,), dtype=mx.int32)
    else:
        kc, n_comp = cidx.shape[-1], comp_kv.shape[1]
        cidx = cidx.reshape(bsz, kc).astype(mx.int32)
    kt = we + kc
    kp = -(-kt // _KEYS_PER_BLOCK) * _KEYS_PER_BLOCK
    meta = mx.array([start_pos, we, kc, offset, n_comp], dtype=mx.int32)
    scores_k, combine_k = _kernels()
    scores = scores_k(
        inputs=[q, win_kv, comp_kv, cidx, meta, _scalar(scale)],
        template=[
            ("OT", q.dtype),
            ("H", heads),
            ("D", dim),
            ("DPL", dim // 32),
            ("W", window),
            ("KP", kp),
        ],
        grid=(kp // _KEYS_PER_BLOCK * 256, heads // _HEADS_PER_GROUP, bsz),
        threadgroup=(256, 1, 1),
        output_shapes=[(bsz * heads * kp,)],
        output_dtypes=[mx.float32],
    )[0]
    kmax = -(-kt // 128) * 128
    out = combine_k(
        inputs=[scores, win_kv, comp_kv, cidx, meta, sink, cos, sin],
        template=[
            ("OT", q.dtype),
            ("H", heads),
            ("D", dim),
            ("RD", rd),
            ("W", window),
            ("KP", kp),
            ("KMAX", kmax),
        ],
        grid=(dim, heads, bsz),
        threadgroup=(_DIMS_PER_GROUP, 1, 1),
        output_shapes=[(bsz * heads * dim,)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(bsz, 1, heads, dim)
