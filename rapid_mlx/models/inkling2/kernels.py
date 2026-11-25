# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: MIT

import math
from collections.abc import Callable
from functools import cache
from typing import cast

import mlx.core as mx

from rapid_mlx.models.inkling2.compat import row_threads, scalar

__all__ = [
    "depth_attention",
    "relative_attention",
]


# One thread owns a channel, including its FP32 history. Do not contract the
# dynamic multiply/reduce: the reference materializes the products before sum.
# Static Conv1d, in contrast, accumulates with FMA then rounds to weight dtype.
_SHORT_CONVOLUTION_SOURCE = """
    #pragma clang fp contract(off)
    uint i = thread_position_in_grid.x;
    const int C = x_shape[2];
    const int b = i / C;
    const int c = i % C;
    if (b >= x_shape[0]) return;
    float history[K];
    for (int j = 0; j < K - 1; ++j) {
        history[j] = HAS_STATE
            ? float(state[b * state_strides[0] + j * state_strides[1]
                          + c * state_strides[2]]) : 0.0f;
    }
    const float residual = float(x[b * x_strides[0] + c * x_strides[2]]);
    history[K - 1] = HAS_MASK && !mask[b * mask_strides[0]] ? 0.0f : residual;

    float acc = 0.0f;
    if (DYNAMIC) {
        const int h = c / (C / H);
        for (int lag = 0; lag < K; ++lag) {
            float w = float(weight[lag * weight_strides[0] + c * weight_strides[1]])
                + float(dynamic[b * dynamic_strides[0]
                                + (h * K + lag) * dynamic_strides[2]]);
            float product = history[K - 1 - lag] * w;
            acc = acc + product;
        }
    } else {
        for (int j = 0; j < K; ++j) {
            float sample = float(W(history[j]));
            acc = metal::fma(sample, float(weight[c * weight_strides[0]
                                                   + j * weight_strides[1]]), acc);
        }
        acc = float(W(acc));
    }
    out[i] = T((acc + residual) / divisor);

    // A length <= 0 retains the old state, as _conv_state's clipped gather
    // does. The mask only zeros the new conv input; the residual is unmasked.
    const int start = HAS_LENGTHS && lengths[b * lengths_strides[0]] <= 0 ? 0 : 1;
    for (int j = 0; j < K - 1; ++j) {
        next_state[(b * (K - 1) + j) * C + c] = history[start + j];
    }
"""

_short_convolution_kernel = cast(
    Callable[..., list[mx.array]],
    mx.fast.metal_kernel(
        name="inkling2_short_convolution_decode",
        input_names=["x", "state", "weight", "dynamic", "mask", "lengths", "divisor"],
        output_names=["out", "next_state"],
        source=_SHORT_CONVOLUTION_SOURCE,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
    ),
)


def short_convolution_decode(
    x: mx.array,
    state: mx.array | None,
    weight: mx.array,
    mask: mx.array | None,
    kernel_size: int,
    *,
    dynamic: mx.array | None = None,
    num_heads: int = 1,
    lengths: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Single-token convolution and retained FP32 state, without input copies."""
    B, _, C = x.shape
    K = kernel_size
    # Shaped, cached placeholders let either specialization use strided loads
    # without allocating zero history/masks on every layer visit.
    empty = scalar(0.0).reshape(1, 1, 1)
    out, state = _short_convolution_kernel(
        inputs=[
            x,
            state if state is not None else empty,
            weight,
            dynamic if dynamic is not None else empty,
            mask if mask is not None else empty,
            lengths if lengths is not None else empty,
            scalar(math.sqrt((2 * K if dynamic is not None else K) + 1)),
        ],
        template=[
            ("T", x.dtype),
            ("W", weight.dtype),
            ("K", K),
            ("H", num_heads),
            ("DYNAMIC", dynamic is not None),
            ("HAS_STATE", state is not None),
            ("HAS_MASK", mask is not None),
            ("HAS_LENGTHS", lengths is not None),
        ],
        grid=(B * C, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, (B, K - 1, C)],
        output_dtypes=[x.dtype, mx.float32],
    )
    return out, state


# One SIMD group computes both low-rank expansion dot products for a channel.
# The projection results and mixing stay FP32, matching the controller's eager
# up projections. No normalized input or learned weight is captured or cached.
_MESSAGE_MIX_SOURCE = """
    #pragma clang fp contract(off)
    const uint lane = thread_index_in_simdgroup;
    const uint i = thread_position_in_grid.x / 32;
    const uint b = i / D;
    const uint d = i % D;
    if (b >= x_shape[0]) return;
    float f = 0.0f, g = 0.0f;
    for (uint r = lane; r < RANK; r += 32) {
        const float s = float(low[b * low_strides[0] + r * low_strides[1]]);
        f = metal::fma(s, float(forget[d * forget_strides[0] + r * forget_strides[1]]), f);
        g = metal::fma(s, float(gain[d * gain_strides[0] + r * gain_strides[1]]), g);
    }
    f = simd_sum(f);
    g = simd_sum(g);
    if (lane == 0) {
        const float a = (1.0f - metal::tanh(f)) * float(x[b * x_strides[0] + d * x_strides[1]]);
        const float m = metal::tanh(g) * float(message[b * message_strides[0] + d * message_strides[1]]);
        out[i] = a + m;
    }
"""

_message_mix_kernel = cast(
    Callable[..., list[mx.array]],
    mx.fast.metal_kernel(
        name="inkling2_message_mix_decode",
        input_names=["x", "message", "low", "forget", "gain"],
        output_names=["out"],
        source=_MESSAGE_MIX_SOURCE,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
    ),
)


def message_mix_decode(x, message, low, forget, gain):
    D = x.shape[-1]
    return _message_mix_kernel(
        inputs=[
            x.reshape(-1, D),
            message.reshape(-1, D),
            low.reshape(x.shape[0], -1),
            forget,
            gain,
        ],
        template=[("D", D), ("RANK", low.shape[-1])],
        grid=(x.size * 32, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.float32],
    )[0]


_DECODE_MASK_SOURCE = """
    const int B = sizes[0];
    const int S = sizes[1];
    const uint i = thread_position_in_grid.x;
    if (i >= B * H * S) return;
    const int k = i % S;
    const int h = (i / S) % H;
    const int b = i / (S * H);
    const int end = ends[(ends_shape[0] == 1 ? 0 : b) * ends_strides[0]];
    const int shifted = ROTATING ? ((k - int(index) % S + S) % S) : k;
    const int pos = end - S + shifted;
    const int distance = S - 1 - shifted;
    float v = -INFINITY;
    if (pos >= 0 && (WINDOW == 0 || distance < WINDOW)) {
        v = distance < R ? float(bias[b * bias_strides[0] + h * bias_strides[2]
                                     + distance * bias_strides[3]]) : 0.0f;
    }
    out[i] = T(v);
"""

_decode_mask_kernel = cast(
    Callable[..., list[mx.array]],
    mx.fast.metal_kernel(
        name="inkling2_attention_mask_decode",
        input_names=["bias", "ends", "index", "sizes"],
        output_names=["out"],
        source=_DECODE_MASK_SOURCE,
        ensure_row_contiguous=False,
    ),
)


def attention_mask_decode(bias, end, index, size, window, dtype, *, rotating):
    B, _, H, R = bias.shape
    return _decode_mask_kernel(
        inputs=[
            bias,
            (end if isinstance(end, mx.array) else scalar(end, mx.int32)).reshape(-1),
            scalar(index, mx.int32),
            mx.array([B, size], dtype=mx.int32),
        ],
        template=[
            ("T", dtype),
            ("H", H),
            ("R", R),
            ("WINDOW", window),
            ("ROTATING", rotating),
        ],
        grid=(B * H * size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, H, 1, size)],
        output_dtypes=[dtype],
    )[0]


_ROUTE_WEIGHTS_SOURCE = """
    #pragma clang fp contract(off)
    const uint row = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    float total = 0.0f;
    for (uint j = lane; j < K; j += 32) {
        total += scores[row * scores_strides[0]
                        + indices[row * indices_strides[0] + j * indices_strides[1]] * scores_strides[1]];
    }
    total = simd_sum(total) + 1e-20f;
    for (uint j = lane; j < K; j += 32) {
        const float s = scores[row * scores_strides[0]
                        + indices[row * indices_strides[0] + j * indices_strides[1]] * scores_strides[1]];
        out[row * K + j] = (s / total) * route_scale;
    }
"""

_route_weights_kernel = cast(
    Callable[..., list[mx.array]],
    mx.fast.metal_kernel(
        name="inkling2_route_weights_decode",
        input_names=["scores", "indices", "route_scale"],
        output_names=["out"],
        source=_ROUTE_WEIGHTS_SOURCE,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
    ),
)


def route_weights_decode(scores, indices, route_scale):
    K = indices.shape[-1]
    return _route_weights_kernel(
        inputs=[
            scores.reshape(-1, scores.shape[-1]),
            indices.reshape(-1, K),
            scalar(route_scale),
        ],
        template=[("K", K)],
        grid=(scores.shape[0] * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[indices.shape],
        output_dtypes=[scores.dtype],
    )[0]


@cache
def _route_topk(n, k):
    try:
        kernel = mx.fast.metal_kernel(
            name="inkling2_route_topk_decode",
            input_names=["logits", "bias"],
            output_names=["indices", "out_scores"],
            source="""
                #pragma clang fp contract(off)
                const uint tid = thread_position_in_threadgroup.x;
                threadgroup float scores[N], order[N];
                if (tid < N) {
                    float x = logits[tid * logits_strides[2]];
                    float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                    scores[tid] = x < 0.0f ? y : 1.0f - y;
                    out_scores[tid] = scores[tid];
                    order[tid] = -(scores[tid] + bias[tid * bias_strides[0]]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < N) {
                    uint rank = 0;
                    const float a = order[tid];
                    for (uint j = 0; j < N; ++j) {
                        float b = order[j];
                        bool less = metal::isnan(a) ? !metal::isnan(b)
                            : !metal::isnan(b) && b < a;
                        bool equal = a == b || (metal::isnan(a) && metal::isnan(b));
                        rank += less || (equal && j < tid);
                    }
                    if (rank < K) {
                        indices[rank] = tid;
                    }
                }
            """,
            ensure_row_contiguous=False,
            compile_options={"math_mode": "safe"},
        )
        # Metal builds lazily: evaluate synthetic outputs before caching success.
        mx.eval(
            *kernel(
                inputs=[mx.zeros((1, 1, n)), mx.zeros((n,))],
                template=[("N", n), ("K", k)],
                grid=(128, 1, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(1, 1, k), (1, 1, n)],
                output_dtypes=[mx.uint32, mx.float32],
            )
        )
        return kernel
    except Exception:
        # Cache unavailability as well, so decode permanently uses the stock path.
        return None


def route_topk_decode(logits, bias, k, scale):
    """Single-row FP32 scoring/selection, with existing decode normalization."""
    kernel = _route_topk(logits.shape[-1], k)
    if kernel is None:
        return None
    indices, scores = kernel(
        inputs=[logits, bias],
        template=[("N", logits.shape[-1]), ("K", k)],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, 1, k), logits.shape],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return indices, route_weights_decode(scores, indices, scale)


# Attention of learned queries over the depth sources of each token, in one
# kernel.
_DEPTH_ATTENTION_SOURCE = """
    constexpr int N = K + 1;
    uint row = threadgroup_position_in_grid.x;
    uint rows = threadgroups_per_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    float acc[N * Q];
    for (int i = 0; i < N * Q; i++) {
        acc[i] = 0.0f;
    }
    for (int d = tid; d < D; d += TG) {
        float x[N];
        for (int i = 0; i < K; i++) {
            x[i] = blocks[(size_t(i) * rows + row) * D + d];
        }
        x[K] = partial[size_t(row) * D + d];
        for (int j = 0; j < Q; j++) {
            float qv = queries[j * D + d];
            for (int i = 0; i < N; i++) {
                acc[i * Q + j] += x[i] * qv;
            }
        }
    }

    threadgroup float buf[N * Q][TG / 32];
    for (int i = 0; i < N * Q; i++) {
        float v = simd_sum(acc[i]);
        if (lane == 0) {
            buf[i][sg] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float w[N * Q];
    for (int i = 0; i < N * Q; i++) {
        float t = 0.0f;
        for (int s = 0; s < TG / 32; s++) {
            t += buf[i][s];
        }
        w[i] = t * scale;
    }
    for (int j = 0; j < Q; j++) {
        float m = -INFINITY;
        for (int i = 0; i < N; i++) {
            m = max(m, w[i * Q + j]);
        }
        float z = 0.0f;
        for (int i = 0; i < N; i++) {
            w[i * Q + j] = metal::exp(w[i * Q + j] - m);
            z += w[i * Q + j];
        }
        for (int i = 0; i < N; i++) {
            w[i * Q + j] /= z;
        }
    }

    for (int d = tid; d < D; d += TG) {
        float x[N];
        for (int i = 0; i < K; i++) {
            x[i] = blocks[(size_t(i) * rows + row) * D + d];
        }
        x[K] = partial[size_t(row) * D + d];
        for (int j = 0; j < Q; j++) {
            float o = 0.0f;
            for (int i = 0; i < N; i++) {
                o += w[i * Q + j] * x[i];
            }
            out[(size_t(row) * Q + j) * D + d] = o;
        }
    }
"""

_depth_attention_kernel = cast(
    Callable[..., list[mx.array]],
    mx.fast.metal_kernel(
        name="depth_attention",
        input_names=["blocks", "partial", "queries", "scale"],
        output_names=["out"],
        source=_DEPTH_ATTENTION_SOURCE,
    ),
)


def depth_attention(
    blocks: mx.array, partial: mx.array, queries: mx.array, *, scale: float
) -> mx.array:
    """blocks: (K, ..., D) and partial: (..., D), float32, also the values.
    queries: float32 (Q, D). Returns (..., Q, D)."""
    D = partial.shape[-1]
    Q = queries.shape[0]
    TG = row_threads(D)
    return _depth_attention_kernel(
        inputs=[blocks, partial, queries, scalar(scale)],
        template=[("K", blocks.shape[0]), ("D", D), ("Q", Q), ("TG", TG)],
        grid=(partial.size // D * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[partial.shape[:-1] + (Q, D)],
        output_dtypes=[mx.float32],
    )[0]


# Causal flash attention with a relative position bias and a sink logit per
# query.
# Queries per threadgroup, 8 for each of its 4 simdgroups.
_BQ = 32

_TG = 128

# The online softmax of each row starts from its sink logit, so the zero-value
# sink needs no key.
_RELATIVE_ATTENTION_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int BK = 16;
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    const int L = sizes[0];
    const int S = sizes[1];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int batch = threadgroup_position_in_grid.z;
    const int head = threadgroup_position_in_grid.y;
    const int kv_head = head / (HQ / HKV);
    const int q0 = threadgroup_position_in_grid.x * BQ;
    const int start = key_start[batch];
    const float qscale = scale * LOG2E;

    // The lane holds row fm and columns fn, fn + 1 of each 8x8 fragment.
    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int diag = S - L + row;

    // q and k are clipped and v has unit RMS, so half holds them.
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 q_frag[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        half2 qv = half2(0.0h);
        if (row_ok) {
            const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
                + row * q_strides[2] + db * 8 + fn;
            qv = half2(float(qp[0]), float(qp[1]));
        }
        q_frag[db].thread_elements()[0] = qv.x;
        q_frag[db].thread_elements()[1] = qv.y;
    }
    simdgroup_float8x8 acc[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float row_max = row_ok ? float(sinks[(batch * L + row) * HQ + head]) * LOG2E : 0.0f;
    float row_sum = 1.0f;
    const int bias_row = batch * L + (row_ok ? row : 0);
    const device T* bias = rel + (size_t(bias_row) * HQ + head) * R;

    const int first_diag = S - L + q0;
    const int last_diag = S - L + min(q0 + BQ, L) - 1;
    int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
    first_key = max(first_key, start);
    const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
    const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

    for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = tid; c < BK * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k0 + r;
            vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            if (key < S) {
                auto kp = (const device vec<T, 4>*)(k_head + key * k_strides[2] + col);
                auto vp = (const device vec<T, 4>*)(v_head + key * v_strides[2] + col);
                kv[0] = kp[0];
                kv[1] = kp[1];
                vv[0] = vp[0];
                vv[1] = vp[1];
            }
            for (int j = 0; j < 8; ++j) {
                Ks[(col + j) * LDK + r] = half(float(kv[j / 4][j % 4]));
                Vs[r * LDV + col + j] = half(float(vv[j / 4][j % 4]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 scores[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
            }
        }

        float tile_max = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = diag - key;
                const bool visible = row_ok && d >= 0 && key >= start
                    && (WINDOW == 0 || d < WINDOW);
                float s = -INFINITY;
                if (visible) {
                    s = scores[kb].thread_elements()[e] * qscale;
                    if (d < R) {
                        s += float(bias[d]) * LOG2E;
                    }
                }
                scores[kb].thread_elements()[e] = s;
                tile_max = max(tile_max, s);
            }
        }
        // The 4 lanes of a row are lane ^ 1 and lane ^ 8 apart.
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float new_max = max(row_max, tile_max);
        const float factor = fast::exp2(row_max - new_max);
        float tile_sum = 0.0f;
        simdgroup_half8x8 probs[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                probs[kb].thread_elements()[e] = half(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum = row_sum * factor + tile_sum;
        row_max = new_max;
        for (int db = 0; db < D / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 v_frag;
                simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
            }
        }
    }

    if (row_ok) {
        device T* op = out + (size_t(batch * L + row) * HQ + head) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
            op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
        }
    }
"""

_relative_attention_kernel = cast(
    Callable[..., list[mx.array]],
    mx.fast.metal_kernel(
        name="relative_attention",
        input_names=["q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale"],
        output_names=["out"],
        source=_RELATIVE_ATTENTION_SOURCE,
        ensure_row_contiguous=False,
    ),
)


def relative_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    key_start: mx.array,
) -> mx.array:
    """q: (B, Hq, L, D), k and v: (B, Hkv, S, D), all bfloat16, with the queries
    at the last L keys. rel: (B, L, Hq, R), the bias at distances below R.
    sinks: (B, L, Hq). key_start: (B,), the first key of each row. Returns
    (B, L, Hq, D)."""
    B, Hq, L, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    return _relative_attention_kernel(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(rel),
            mx.contiguous(sinks.astype(mx.float32)),
            mx.contiguous(key_start.astype(mx.int32)),
            mx.array([L, S], dtype=mx.int32),
            scalar(scale),
        ],
        template=[
            ("T", q.dtype),
            ("D", D),
            ("HQ", Hq),
            ("HKV", Hkv),
            ("R", rel.shape[-1]),
            ("WINDOW", window),
            ("BQ", _BQ),
            ("TG", _TG),
        ],
        grid=((L + _BQ - 1) // _BQ * _TG, Hq, B),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(B, L, Hq, D)],
        output_dtypes=[q.dtype],
    )[0]
