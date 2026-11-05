# SPDX-License-Identifier: Apache-2.0
"""Experimental M5 QSA tensor-unit prefill. See NOTICE for source attribution.

Changed from the attributed attention source: consume Rapid's physical block
starts/counts/tails, handle empty rows, normalize strided inputs, and use scalar
loads so misaligned views do not invoke undefined vector-load behavior.
"""

from __future__ import annotations

import platform
from functools import lru_cache

import mlx.core as mx

from .._env import env_truthy
from .qsa_m5_gate import admit, architecture_supported
from .qsa_stage1 import _select_scores

ENABLE_ENV = "RAPID_MLX_QSA_M5_NAX"


@lru_cache(maxsize=1)
def hardware_supported() -> bool:
    if not mx.metal.is_available():
        return False
    return architecture_supported(
        str(mx.device_info().get("architecture", "")), platform.mac_ver()[0]
    )


def eligible(query_length: int, *, batch_size: int, training: bool = False) -> bool:
    return admit(
        query_length,
        batch_size=batch_size,
        training=training,
        enabled=env_truthy(ENABLE_ENV),
        hardware=hardware_supported,
        on_gpu=lambda: mx.default_device() == mx.gpu,
    )


def attention_supported(q, k, v, *, block_size: int, block_topk: int) -> bool:
    return bool(
        q.ndim == k.ndim == v.ndim == 4
        and q.shape[0:2] == (1, 24)
        and q.shape[-1] == 256
        and k.shape[:2] == (1, 2)
        and k.shape[-1] == 256
        and k.shape[2] > 0
        and k.shape == v.shape
        and q.dtype == k.dtype == v.dtype == mx.bfloat16
        and block_size == 4
        and block_topk == 512
    )


_HEADER = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
#define UNROLL _Pragma("clang loop unroll(full)")

inline vec<bfloat, 8> qsa_load8(const device bfloat* p) {
    vec<bfloat, 8> x;
    UNROLL for (short i = 0; i < 8; ++i) { x[i] = p[i]; }
    return x;
}
"""

_ATTN_SOURCE = r"""
    // Grid: (query, KV head); 2 simdgroups: sg = which 128-dim half of D.
    // Tensor-unit fragments (MLX NAX layout): each lane holds rows (fm, fm + 8)
    // x columns (fn .. fn + 3) of a 16x16 fragment. Rows are the 12 heads of
    // the query (rows 12..15 idle); columns of S are keys.
    constexpr int D = 256;
    constexpr int TDH = 8;      // 16-wide head-dim fragments per half
    constexpr int SB = 8;       // blocks (4 tokens each) per step: 32 keys
    constexpr int PV_K = PV_TERMS == 2 ? 32 : 16;
    threadgroup float xchg[2][8 * 32];

    const int tq = int(threadgroup_position_in_grid.x);
    const int kvh = int(threadgroup_position_in_grid.y);
    const ushort dh = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;

    const int kL = params[0];
    const float scale2 = scale[0] * 1.44269504089f;
    const int nsel = clamp(counts[tq], 0, TOPK);
    const int ntail = clamp(tail_counts[tq], 0, 4);
    const int U = nsel + (ntail > 0 ? 1 : 0);
    const int nsteps = (U + SB - 1) / SB;
    const bool r1_ok = fm + 8 < GQA;             // row fm + 8 is a real head
    const int h0 = kvh * GQA + fm;
    const int h1 = kvh * GQA + (r1_ok ? fm + 8 : fm);

    // Strides (elements): q [1, H, Lq, D], k/v [1, KVH, kL, D]; the last dim is
    // contiguous and rows are 16-byte aligned. Head dims are permuted per lane:
    // fragment pair (2j, 2j + 1) covers the 8 contiguous dims 32 j + 2 fn .. + 7
    // (the first four in fragment 2j) of Q/K and of V/O, so one 16-byte load
    // feeds both fragments. Q and K share the permutation (only the Q K^T
    // summation order changes); O undoes V's at the store.
    const int64_t sqh = q_strides[1], sql = q_strides[2];
    const uint skl = uint(k_strides[2]), svl = uint(v_strides[2]);
    const short fcol = 2 * fn;
    const device bfloat* kb = (const device bfloat*)k + kvh * k_strides[1] + dh * 128 + fcol;
    const device bfloat* vb = (const device bfloat*)v + kvh * v_strides[1] + dh * 128 + fcol;
    const device bfloat* qp0 = (const device bfloat*)q + h0 * sqh + tq * sql + dh * 128 + fcol;
    const device bfloat* qp1 = (const device bfloat*)q + h1 * sqh + tq * sql + dh * 128 + fcol;
    const device int* blk = sel + size_t(tq) * TOPK;

    constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
    constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, PV_K, false, false, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
    // Cooperative tensors: element 8 c + i is element i of the c-th 16x16
    // fragment along K (left) / along (K, N) (right) / along N (destination).
    auto qa_ct = qk_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto kb_ct = qk_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    using qa_t = metal::remove_addrspace_t<decltype(qa_ct)>;
    using kb_t = metal::remove_addrspace_t<decltype(kb_ct)>;
    auto pa_ct = pv_op.template get_left_input_cooperative_tensor<PT, bfloat, float>();
    auto vb_ct = pv_op.template get_right_input_cooperative_tensor<PT, bfloat, float>();
    using pa_t = metal::remove_addrspace_t<decltype(pa_ct)>;
    using vb_t = metal::remove_addrspace_t<decltype(vb_ct)>;
    // O over this half of D: of<j> holds dim fragments (2 j, 2 j + 1).
    auto of0 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    auto of1 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    auto of2 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    auto of3 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    UNROLL for (short i = 0; i < 16; ++i) {
      of0[i] = 0.0f;
      of1[i] = 0.0f;
      of2[i] = 0.0f;
      of3[i] = 0.0f;
    }
    float max_s[2] = {-FLT_MAX, -FLT_MAX};
    float sum_s[2] = {0.0f, 0.0f};

    // Step metadata, prefetched one step ahead: the block of each key row this
    // lane loads (keys 16 (r / 2) + fm + 8 (r % 2) -> slot key / 4) and the
    // number of visible tokens of this lane's S column block (slot 4 f + fn / 4).
    auto blk_at = [&](int u) -> int { return u < nsel ? blk[u] : 0; };
    int nrow_b[4];
    int ncol_n[2];
    auto fetch = [&](int u0) {
      UNROLL for (short r = 0; r < 4; ++r) {
        const int u = u0 + ((16 * (r >> 1) + fm + 8 * (r & 1)) >> 2);
        const int within = fm & 3;
        nrow_b[r] = u < nsel ? blk_at(u) + within
            : (u < U && within < ntail ? tails[tq * 4 + within] : 0);
      }
      UNROLL for (short f = 0; f < 2; ++f) {
        const int u = u0 + 4 * f + (fn >> 2);
        ncol_n[f] = u < nsel ? 4 : (u < U ? ntail : 0);
      }
    };
    fetch(0);

    for (int step = 0; step < nsteps; ++step) {
      // The tail block can extend up to three rows past kL: those keys are
      // masked, but P = 0 must never meet unwritten V (0 * NaN = NaN), so
      // they re-read the last valid row instead.
      uint krow[4];
      UNROLL for (short r = 0; r < 4; ++r) {
        krow[r] = uint(clamp(nrow_b[r], 0, kL - 1));
      }
      int nvis[2];
      UNROLL for (short f = 0; f < 2; ++f) {
        nvis[f] = ncol_n[f];
      }
      if (step + 1 < nsteps) {
        fetch((step + 1) * SB);
      }

      // S = Q K^T over this half of D: one 16x32x32 op per 32 head dims.
      auto s_ct = qk_op.template get_destination_cooperative_tensor<qa_t, kb_t, float>();
      UNROLL for (short i = 0; i < 16; ++i) {
        s_ct[i] = 0.0f;
      }
      UNROLL for (short jj = 0; jj < TDH / 2; ++jj) {
        const vec<bfloat, 8> qa = qsa_load8(qp0 + 32 * jj);
        const vec<bfloat, 8> qb = r1_ok ? qsa_load8(qp1 + 32 * jj) : vec<bfloat, 8>(0);
        const vec<bfloat, 8> a0 = qsa_load8(kb + (krow[0] * skl + 32 * jj));
        const vec<bfloat, 8> a1 = qsa_load8(kb + (krow[1] * skl + 32 * jj));
        const vec<bfloat, 8> a2 = qsa_load8(kb + (krow[2] * skl + 32 * jj));
        const vec<bfloat, 8> a3 = qsa_load8(kb + (krow[3] * skl + 32 * jj));
        UNROLL for (short j = 0; j < 4; ++j) {
          qa_ct[j] = qa[j];
          qa_ct[4 + j] = qb[j];
          qa_ct[8 + j] = qa[4 + j];
          qa_ct[12 + j] = qb[4 + j];
          kb_ct[j] = a0[j];
          kb_ct[4 + j] = a1[j];
          kb_ct[8 + j] = a2[j];
          kb_ct[12 + j] = a3[j];
          kb_ct[16 + j] = a0[4 + j];
          kb_ct[20 + j] = a1[4 + j];
          kb_ct[24 + j] = a2[4 + j];
          kb_ct[28 + j] = a3[4 + j];
        }
        qk_op.run(qa_ct, kb_ct, s_ct);
      }
      vec<float, 8> s[2];
      UNROLL for (short i = 0; i < 8; ++i) {
        s[0][i] = s_ct[i];
        s[1][i] = s_ct[8 + i];
      }

      // Add the other half's partial scores, one 16-key fragment at a time.
      UNROLL for (short f = 0; f < 2; ++f) {
        threadgroup float* mine = xchg[dh];
        const threadgroup float* peer = xchg[1 - dh];
        UNROLL for (short i = 0; i < 8; ++i) {
          mine[lane * 8 + i] = s[f][i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        UNROLL for (short i = 0; i < 8; ++i) {
          s[f][i] += peer[lane * 8 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      // Scale and mask: this lane's columns fn..fn+3 of fragment f are the four
      // tokens of one block; the first nvis of them are visible.
      UNROLL for (short f = 0; f < 2; ++f) {
        UNROLL for (short j = 0; j < 4; ++j) {
          const bool ok = j < nvis[f];
          s[f][j] = ok ? s[f][j] * scale2 : -INFINITY;
          s[f][4 + j] = ok ? s[f][4 + j] * scale2 : -INFINITY;
        }
      }
      // Online softmax per row (rows fm and fm + 8 of the fragments).
      float factor[2];
      UNROLL for (short i = 0; i < 2; ++i) {
        float m = -INFINITY;
        UNROLL for (short f = 0; f < 2; ++f) {
          m = max(m, max(max(s[f][4 * i], s[f][4 * i + 1]), max(s[f][4 * i + 2], s[f][4 * i + 3])));
        }
        m = max(m, simd_shuffle_xor(m, ushort(1)));
        m = max(m, simd_shuffle_xor(m, ushort(8)));
        const float new_max = max(max_s[i], m);
        float rs = 0.0f;
        UNROLL for (short f = 0; f < 2; ++f) {
          UNROLL for (short j = 0; j < 4; ++j) {
            s[f][4 * i + j] = fast::exp2(s[f][4 * i + j] - new_max);
            rs += s[f][4 * i + j];
          }
        }
        rs += simd_shuffle_xor(rs, ushort(1));
        rs += simd_shuffle_xor(rs, ushort(8));
        factor[i] = fast::exp2(max_s[i] - new_max);
        max_s[i] = new_max;
        sum_s[i] = sum_s[i] * factor[i] + rs;
      }
      // O *= factor (exact no-op when every factor of the simdgroup is 1).
      if (simd_any(factor[0] != 1.0f || factor[1] != 1.0f)) {
        UNROLL for (short h = 0; h < 2; ++h) {
          UNROLL for (short j = 0; j < 4; ++j) {
            of0[8 * h + j] *= factor[0];
            of0[8 * h + 4 + j] *= factor[1];
            of1[8 * h + j] *= factor[0];
            of1[8 * h + 4 + j] *= factor[1];
            of2[8 * h + j] *= factor[0];
            of2[8 * h + 4 + j] *= factor[1];
            of3[8 * h + j] *= factor[0];
            of3[8 * h + 4 + j] *= factor[1];
          }
        }
      }
      // O += P V over this half of D (V fragments are [16 keys x 16 dims]).
      // The tensor unit takes P as PV_TERMS pieces of type PT whose sum is P;
      // two pieces are packed along K into one 16x32x32 op.
      UNROLL for (short f = 0; f < 2; ++f) {
        vec<PT, 8> pc[PV_TERMS];
        {
          vec<float, 8> rest = s[f];
          UNROLL for (short tt = 0; tt < PV_TERMS; ++tt) {
            UNROLL for (short i = 0; i < 8; ++i) {
              const PT piece = PT(rest[i]);
              pc[tt][i] = piece;
              rest[i] -= float(piece);
            }
          }
        }
        UNROLL for (short id = 0; id < TDH; id += 2) {
          const vec<bfloat, 8> ra = qsa_load8(vb + (krow[2 * f] * svl + 16 * id));
          const vec<bfloat, 8> rb = qsa_load8(vb + (krow[2 * f + 1] * svl + 16 * id));
          UNROLL for (short j = 0; j < 4; ++j) {
            vb_ct[j] = ra[j];
            vb_ct[4 + j] = rb[j];
            vb_ct[8 + j] = ra[4 + j];
            vb_ct[12 + j] = rb[4 + j];
            if (PV_K == 32) {
              vb_ct[16 + j] = ra[j];
              vb_ct[20 + j] = rb[j];
              vb_ct[24 + j] = ra[4 + j];
              vb_ct[28 + j] = rb[4 + j];
            }
          }
          UNROLL for (short tt = 0; tt < PV_TERMS; tt += PV_K / 16) {
            UNROLL for (short i = 0; i < 8; ++i) {
              pa_ct[i] = pc[tt][i];
              if (PV_K == 32) {
                pa_ct[8 + i] = pc[tt + 1][i];
              }
            }
            if (id == 0) {
              pv_op.run(pa_ct, vb_ct, of0);
            } else if (id == 2) {
              pv_op.run(pa_ct, vb_ct, of1);
            } else if (id == 4) {
              pv_op.run(pa_ct, vb_ct, of2);
            } else {
              pv_op.run(pa_ct, vb_ct, of3);
            }
          }
        }
      }
    }

    UNROLL for (short i = 0; i < 2; ++i) {
      if (i == 1 && !r1_ok) {
        continue;
      }
      const float rr = sum_s[i] > 0.0f ? 1.0f / sum_s[i] : 0.0f;
      device bfloat* o = (device bfloat*)out + (size_t(tq) * (2 * GQA) + (i == 0 ? h0 : h1)) * D + dh * 128 + fcol;
      UNROLL for (short jj = 0; jj < TDH / 2; ++jj) {
        vec<bfloat, 8> w;
        UNROLL for (short j = 0; j < 4; ++j) {
          float e0, e1;
          if (jj == 0) {
            e0 = of0[4 * i + j];
            e1 = of0[8 + 4 * i + j];
          } else if (jj == 1) {
            e0 = of1[4 * i + j];
            e1 = of1[8 + 4 * i + j];
          } else if (jj == 2) {
            e0 = of2[4 * i + j];
            e1 = of2[8 + 4 * i + j];
          } else {
            e0 = of3[4 * i + j];
            e1 = of3[8 + 4 * i + j];
          }
          w[j] = bfloat(e0 * rr);
          w[4 + j] = bfloat(e1 * rr);
        }
        *(device vec<bfloat, 8>*)(o + 32 * jj) = w;
      }
    }
"""

_SCORES_SOURCE = r"""
    // One simdgroup owns 16 query rows x 32 compressed keys.
    const int L = params[0], N = params[1];
    const int q0 = int(threadgroup_position_in_grid.y) * 16;
    const int k0 = int(threadgroup_position_in_grid.x) * 32;
    const ushort lane = thread_index_in_simdgroup;
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    const short fcol = 2 * fn;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto qa = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto kb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    using qa_t = metal::remove_addrspace_t<decltype(qa)>;
    using kb_t = metal::remove_addrspace_t<decltype(kb)>;
    float total[16];
    UNROLL for (short e = 0; e < 16; ++e) { total[e] = 0.0f; }
    UNROLL for (short h = 0; h < 4; ++h) {
        auto s = op.template get_destination_cooperative_tensor<qa_t, kb_t, float>();
        UNROLL for (short e = 0; e < 16; ++e) { s[e] = 0.0f; }
        UNROLL for (short jj = 0; jj < 4; ++jj) {
            const int qr0 = min(q0 + fm, L - 1);
            const int qr1 = min(q0 + fm + 8, L - 1);
            const auto a = qsa_load8(q + (qr0 * 4 + h) * 128 + 32 * jj + fcol);
            const auto b = qsa_load8(q + (qr1 * 4 + h) * 128 + 32 * jj + fcol);
            vec<bfloat, 8> kr[4];
            UNROLL for (short r = 0; r < 4; ++r) {
                const int row = min(k0 + 16 * (r >> 1) + fm + 8 * (r & 1), N - 1);
                kr[r] = qsa_load8(k + row * 128 + 32 * jj + fcol);
            }
            UNROLL for (short j = 0; j < 4; ++j) {
                qa[j] = a[j]; qa[4+j] = b[j];
                qa[8+j] = a[4+j]; qa[12+j] = b[4+j];
                kb[j] = kr[0][j]; kb[4+j] = kr[1][j];
                kb[8+j] = kr[2][j]; kb[12+j] = kr[3][j];
                kb[16+j] = kr[0][4+j]; kb[20+j] = kr[1][4+j];
                kb[24+j] = kr[2][4+j]; kb[28+j] = kr[3][4+j];
            }
            op.run(qa, kb, s);
        }
        // Match the existing bf16 matmul output rounding before FP32 ReLU.
        UNROLL for (short e = 0; e < 16; ++e) {
            total[e] += max(float(bfloat(s[e])), 0.0f);
        }
    }
    UNROLL for (short f = 0; f < 2; ++f) {
        UNROLL for (short i = 0; i < 2; ++i) {
            const int row = q0 + fm + 8 * i;
            UNROLL for (short j = 0; j < 4; ++j) {
                const int col = k0 + 16 * f + fn + j;
                if (row < L && col < N) {
                    out[size_t(row) * N + col] = total[8*f + 4*i + j]
                        * 0.08838834764831844f;
                }
            }
        }
    }
"""


@lru_cache(maxsize=1)
def _scores_kernel():
    return mx.fast.metal_kernel(
        name="rapid_qsa_m5_scores",
        input_names=["q", "k", "params"],
        output_names=["out"],
        source=_SCORES_SOURCE,
        header=_HEADER,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _attention_kernel():
    return mx.fast.metal_kernel(
        name="rapid_qsa_m5_attention",
        input_names=[
            "q",
            "k",
            "v",
            "sel",
            "counts",
            "tails",
            "tail_counts",
            "params",
            "scale",
        ],
        output_names=["out"],
        source=_ATTN_SOURCE,
        header=_HEADER,
        ensure_row_contiguous=True,
    )


def scores(q: mx.array, pooled: mx.array) -> mx.array:
    """Unmasked [1,L,N] FP32 scores; direct entry requires M5 and the opt-in."""
    if q.ndim != 4 or not eligible(q.shape[1], batch_size=q.shape[0]):
        raise ValueError("M5 QSA route is unavailable or disabled")
    if not (
        q.ndim == 4
        and q.shape[2:] == (4, 128)
        and pooled.ndim == 3
        and pooled.shape[0] == 1
        and pooled.shape[2] == 128
        and pooled.shape[1] > 512
        and q.dtype == pooled.dtype == mx.bfloat16
    ):
        raise ValueError("unsupported M5 QSA scoring geometry")
    length, blocks = q.shape[1], pooled.shape[1]
    return _scores_kernel()(
        inputs=[q, pooled, mx.array([length, blocks], dtype=mx.int32)],
        grid=(((blocks + 31) // 32) * 32, (length + 15) // 16, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, length, blocks)],
        output_dtypes=[mx.float32],
    )[0]


def select(q, pooled, positions, *, block_topk=512, compress_ratio=4, training=False):
    """Return selected IDs, or None to preserve the existing selector graph."""
    if q.ndim != 4 or not eligible(
        q.shape[1], batch_size=q.shape[0], training=training
    ):
        return None
    if not (
        q.ndim == 4
        and q.shape[2:] == (4, 128)
        and pooled.ndim == 3
        and pooled.shape[0] == 1
        and pooled.shape[2] == 128
        and pooled.shape[1] > 512
        and q.dtype == pooled.dtype == mx.bfloat16
        and positions.shape == q.shape[:2]
        and positions.dtype in (mx.int32, mx.int64)
        and block_topk == 512
        and compress_ratio == 4
    ):
        return None
    s = scores(q, pooled)
    return _select_scores(
        s[0],
        positions[0],
        topk=512,
        compress_ratio=4,
    )[None].astype(mx.int32)


def attention(q, k, v, starts, counts, tails, tail_counts, *, scale):
    """Consume validated physical selection metadata; output [1,H,L,D]."""
    if q.ndim != 4 or not eligible(q.shape[2], batch_size=q.shape[0]):
        raise ValueError("M5 QSA route is unavailable or disabled")
    if not attention_supported(q, k, v, block_size=4, block_topk=512):
        raise ValueError("unsupported M5 QSA attention geometry")
    length = q.shape[2]
    arrays = (starts, counts, tails, tail_counts)
    if tuple(a.shape for a in arrays) != (
        (1, length, 512),
        (1, length),
        (1, length, 4),
        (1, length),
    ) or any(a.dtype != mx.int32 for a in arrays):
        raise ValueError("invalid M5 QSA compact metadata")
    out = _attention_kernel()(
        inputs=[
            q,
            k,
            v,
            starts,
            counts,
            tails,
            tail_counts,
            mx.array([k.shape[2]], dtype=mx.int32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[("GQA", 12), ("TOPK", 512), ("PT", mx.float16), ("PV_TERMS", 2)],
        grid=(length * 64, 2, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(1, length, 24, 256)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return out.transpose(0, 2, 1, 3)
