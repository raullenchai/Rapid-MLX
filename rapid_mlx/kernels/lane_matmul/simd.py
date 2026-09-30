# SPDX-License-Identifier: MIT
#
# Row-invariant lane matmul: the design and the original Metal kernels are
# TensorFold's (github.com/ashhart/TensorFold, MIT, (c) 2026 TensorFold
# contributors); this implementation is adapted from mlx2
# (github.com/pierre427/mlx2, src/mlx2/runtime/lane/). See LICENSE-LANE-MATMUL.

"""Row-exact small-M matmul for GPUs without tensor units (M1 to M4).

The M5 lane kernels (``matmul``) need Metal 4 tensor units.  This backend
gives the same contract on every other Apple GPU: every row of a call is
bitwise equal to that row computed alone, and the weights are read once for
all rows of a window.  It runs on the ``simdgroup_matrix`` units that every
Apple GPU has, and it is a different numerical law from both stock MLX and
the M5 lane kernels.

Three kernel families, chosen per projection at preparation and never by the
row count:

* 4-bit, groups of 32 or 64, bf16 scales: ``simd4``.  A scalar kernel for
  1-2 (1-3) rows and a matrix kernel from there; the per-group chain is the
  same fp32 FMA chain in both, so their bits agree.  ``check()`` proves that
  per weight shape on this GPU at install, and a shape that differs sends its
  1-4-row calls to the matrix kernel instead.  2-8-row calls in groups of 64
  read pre-scaled input fragments (same bits, less input work).
* 5/6/8-bit, groups of 64, bf16 scales: ``simd_bits``.  The same chain with
  integer codes; a scalar twin for one row, checked the same way (a shape
  that differs takes ``affine_rows``).
* anything else affine (2-8-bit, groups of 32/64/128, bf16/fp16/fp32 scales):
  ``affine_rows``.  Each lane dequantizes a 32-code block once and folds it
  into every row's own FMA chain.

Activations must be bf16 (the kernels read packed bf16 pairs); other dtypes
and unquantized weights raise ``LaneUnsupportedError`` and stay on stock kernels.

Mined from TensorFold ``9cd52ab`` (MIT License, Copyright (c) 2026 TensorFold
contributors): ``kernels/qwen/dense/v1/simd_qmm.py``, ``simd_qmm_bits.py``,
``affine_rows.py``, ``row_matmul.py`` (the backend dispatch) and
``kernels/threads.py``.  The Metal sources are unchanged; the fused input
prologues and dependency inputs are dropped.  Notice in LICENSE-LANE-MATMUL.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import mlx.core as mx

MAX_ROWS = 1 << 16
RT_MAX = 2  # 8-row tiles a threadgroup: more rows than 8 RT_MAX spread over the grid's y axis
MMA_SGS = 16  # physical simdgroups at most (fewer where a pipeline takes fewer threads): same chunks
GROUP = 64
SGS = 2  # simdgroups a threadgroup in the scalar kernel
NR = 2  # outputs a lane in the scalar kernel
XB = 32  # scalar kernel: groups of inputs staged at a time (one row)
SCALAR_ROWS = 4  # rows the scalar kernel can take
FRAGMENT_ROWS = 8  # 2..FRAGMENT_ROWS rows read prepared input fragments (groups of 64)
BITS_MMA = (5, 6, 8)
AFFINE_BITS = (2, 3, 4, 5, 6, 8)
AFFINE_GROUPS = (32, 64, 128)
AFFINE_SG = (
    8  # simdgroups a threadgroup: 256 threads, within every M1/M2 pipeline's limit
)

# Per-GPU outcomes of the install-time twin checks, keyed by weight shape.
mma_one_row: set[tuple[int, int, int]] = (
    set()
)  # (n, k, group): 4-bit 1-4-row calls use the MMA kernel
bits_fallback: set[tuple[int, int, int]] = (
    set()
)  # (n, k, bits): 5/6/8-bit shapes that take affine_rows


class SimdUnsupportedError(ValueError):
    """This projection or input cannot use the simd backend."""


# ---------------------------------------------------------------- threads ---
# Keep each launch within its pipeline's thread limit, which on M1 and M2
# falls as the kernel's register use rises (M3 on give every pipeline 1024).
_SAFE = 256
_LIMIT = re.compile(r"maximum allowed threads per threadgroup \((\d+)\)")
_TRACING = "function transformations"
_fitted: dict[Any, int] = {}


def _architecture() -> int | None:
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    found = re.match(r"applegpu_g(\d+)", str(info.get("architecture", "")))
    return int(found.group(1)) if found else None


_PROBING: list[bool] = []


def _probing() -> bool:
    if not _PROBING:
        try:
            arch = _architecture()
        except Exception:  # noqa: BLE001 - unknown device: probe to be safe
            arch = None
        _PROBING.append(arch is None or arch < 15)
    return _PROBING[0]


def _fit(key: tuple, sizes, launch, inputs=()):
    """launch(size) at the largest threadgroup size key's pipeline takes here."""
    size = _fitted.get(key)
    if size is not None:
        return launch(size)
    options = sorted({int(s) for s in sizes}, reverse=True)
    if not _probing():
        _fitted[key] = options[0]
        return launch(options[0])
    cap = None
    for size in options:
        if cap is not None and size > cap:
            continue
        try:
            if size > _SAFE:
                mx.eval(*[a for a in inputs if isinstance(a, mx.array)])
            out = launch(size)
            if size > _SAFE:
                mx.eval(out)
        except ValueError as err:
            if _TRACING in str(err):  # inside mx.compile: no launch can run alone
                return launch(next((s for s in options if s <= _SAFE), options[-1]))
            found = _LIMIT.search(str(err))
            if found is None:
                raise
            cap = int(found.group(1))
            continue
        _fitted[key] = size
        return out
    raise SimdUnsupportedError(
        f"this GPU allows {cap} threads a threadgroup for {key[0]}, "
        f"which needs {options[-1]}"
    )


# ------------------------------------------------------------ 4-bit simd ---
_HEADER = r"""
#define PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
// the bf16 at index e (0..7) of 8 packed bf16 as fp32
inline float bf8(uint4 v, int e) {
  const uint w = v[e / 2];
  return as_type<float>((e % 2) ? (w & 0xFFFF0000u) : (w << 16));
}
// a row's 8 inputs summed left to right
inline float sum8(uint4 v, float one) {
  float t = bf8(v, 0);
  for (int e = 1; e < 8; e++) t = fma(bf8(v, e), one, t);
  return t;
}
// 2^-4s
inline float pre(int s) { return as_type<float>(uint(127 - 4 * s) << 23); }
"""

_LOAD8 = "(((const device uint4*)X)[size_t(r) * (K / 8) + (j)])"

_SCALAR = r"""
  // RS rows (1 to 4). Lane (chunk c = lane % S, slot j = lane / S) runs chunk c of NR outputs n0 + j + (32 / S) u,
  // a whole group (WPG words) of each in registers, once a row; the threadgroup stages XB groups of each row's
  // inputs pre-scaled in chain order. A row's chain is the same at any RS.
  constexpr int WPG = GS / 8, NS = 8 / WPG;     // words a group; nibble stride of an MMA step
  constexpr int XP = GS == 64 ? 76 : 44;        // floats a staged group: GS inputs, WPG sums, pad (bank spread)
  threadgroup float xs[RS * XB * XP];
  const uint lane = thread_index_in_simdgroup;
  const int tid = int(simdgroup_index_in_threadgroup) * 32 + int(lane);
  const int c = int(lane) % S;
  constexpr int SLOTS = 32 / S;
  const int n0 = (int(threadgroup_position_in_grid.x) * SGS + int(simdgroup_index_in_threadgroup)) * (SLOTS * NR)
                 + int(lane) / S;
  constexpr int G = K / GS;
  const float one = ONE[0];
  const device uint4* wr[NR];
  const device bfloat* sr[NR];
  const device bfloat* br[NR];
  float acc[NR][RS];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) {
    const int nn = min(n0 + SLOTS * u, N - 1);
    wr[u] = (const device uint4*)(W + size_t(nn) * (K / 8));
    sr[u] = SC + size_t(nn) * G;
    br[u] = BI + size_t(nn) * G;
    PRAGMA_UNROLL
    for (int r = 0; r < RS; r++) acc[u][r] = 0.0f;
  }
  uint4 nw[NR][WPG / 4];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) for (int h = 0; h < WPG / 4; h++) nw[u][h] = c < G ? wr[u][(WPG / 4) * c + h] : uint4(0);
  for (int b0 = 0; b0 < G; b0 += XB) {
    const int nbk = min(XB, G - b0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < RS * nbk * WPG; idx += SGS * 32) {
      const int r = RS == 1 ? 0 : idx / (nbk * WPG);
      const int gl = (RS == 1 ? idx : idx - r * (nbk * WPG)) / WPG, j = idx % WPG;
      const uint4 v = LOAD8(r, WPG * (b0 + gl) + j);
      threadgroup float* xr = xs + r * (XB * XP) + gl * XP;
      PRAGMA_UNROLL
      for (int e = 0; e < 8; e++) xr[8 * (e / NS) + NS * j + e % NS] = bf8(v, e) * pre(e);
      xr[GS + j] = sum8(v, one);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = b0 + c; g < b0 + nbk; g += S) {
      uint4 wv[NR][WPG / 4];
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) for (int h = 0; h < WPG / 4; h++) wv[u][h] = nw[u][h];
      if (g + S < G) {
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++) for (int h = 0; h < WPG / 4; h++) nw[u][h] = wr[u][(WPG / 4) * (g + S) + h];
      }
      float xsum[RS];
      float P[NR][RS];
      PRAGMA_UNROLL
      for (int r = 0; r < RS; r++) {
        const threadgroup float* xg = xs + r * (XB * XP) + (g - b0) * XP;
        const float4 p0 = *(const threadgroup float4*)(xg + GS);
        xsum[r] = fma(fma(p0.w, one, p0.z), one, fma(p0.y, one, p0.x));
        if (GS == 64) {
          const float4 p1 = *(const threadgroup float4*)(xg + GS + 4);
          xsum[r] = fma(fma(fma(p1.w, one, p1.z), one, fma(p1.y, one, p1.x)), one, xsum[r]);
        }
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++) P[u][r] = 0.0f;
      }
      PRAGMA_UNROLL
      for (int s = 0; s < WPG; s++) {
        float xq[RS][8];
        PRAGMA_UNROLL
        for (int r = 0; r < RS; r++) {
          const threadgroup float* xg = xs + r * (XB * XP) + (g - b0) * XP + 8 * s;
          const float4 lo = *(const threadgroup float4*)(xg), hi = *(const threadgroup float4*)(xg + 4);
          xq[r][0] = lo.x; xq[r][1] = lo.y; xq[r][2] = lo.z; xq[r][3] = lo.w;
          xq[r][4] = hi.x; xq[r][5] = hi.y; xq[r][6] = hi.z; xq[r][7] = hi.w;
        }
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++)
          PRAGMA_UNROLL
          for (int i = 0; i < 8; i++) {
            const float q = float(wv[u][i / NS / 4][(i / NS) % 4] & (0xFu << (4 * (NS * s + i % NS))));
            PRAGMA_UNROLL
            for (int r = 0; r < RS; r++) P[u][r] = fma(xq[r][i], q, P[u][r]);
          }
      }
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) {
        const float sc = float(sr[u][g]), bi = float(br[u][g]);
        PRAGMA_UNROLL
        for (int r = 0; r < RS; r++) {
          acc[u][r] = fma(sc, P[u][r], acc[u][r]);
          acc[u][r] = fma(bi, xsum[r], acc[u][r]);
        }
      }
    }
  }
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++)
    PRAGMA_UNROLL
    for (int r = 0; r < RS; r++) {
      float v = acc[u][r];
      PRAGMA_UNROLL
      for (int m = 1; m < S; m <<= 1) v = fma(simd_shuffle_xor(v, ushort(m)), one, v);
      const int n = n0 + SLOTS * u;
      if (n < N && c == 0) OUT[size_t(r) * N + n] = bfloat(v);
    }
"""

_MMA = r"""
  // R rows: threadgroup (x, y) takes rows 8 RT y .. 8 RT y + 8 RT - 1 in RT tiles of 8 (rows >= R read row R - 1;
  // their results are dropped). SGS physical simdgroups compute all S arithmetic chunks in order.
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int R = X_shape[0];
  constexpr int G = K / GS, WPG = GS / 8, NS = 8 / WPG;   // groups; words a group; nibble stride
  const float one = ONE[0];
  const int nb = int(threadgroup_position_in_grid.x) * (8 * NT);
  const int rb = int(threadgroup_position_in_grid.y) * (8 * RT);
  threadgroup float red[S > 1 ? S * RT * NT * 64 : 1];
  const device uint2* W2 = (const device uint2*)W;
  const device uint* W1 = (const device uint*)W;
  int wrow[NT];
  for (int t = 0; t < NT; t++) wrow[t] = min(nb + 8 * t + fm, N - 1);
  int xr0[RT], xr1[RT];
  for (int rt = 0; rt < RT; rt++) { xr0[rt] = min(rb + 8 * rt + fn, R - 1); xr1[rt] = min(rb + 8 * rt + fn + 1, R - 1); }
  for (int c = sg; c < S; c += SGS) {
    float acc[RT][NT][2];
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++) { acc[rt][t][0] = 0.0f; acc[rt][t][1] = 0.0f; }
    for (int g = c; g < G; g += S) {
      uint2 wv[NT];
      PRAGMA_UNROLL
      for (int t = 0; t < NT; t++)
        if (GS == 64) wv[t] = W2[size_t(wrow[t]) * (K / 16) + 4 * g + fn / 2];
        else wv[t] = uint2(W1[size_t(wrow[t]) * (K / 8) + 4 * g + fn / 2]);
      uint4 xa[RT], xb[RT];
      float xs0[RT], xs1[RT];
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++) {
        xa[rt] = LOAD8(xr0[rt], WPG * g + fm / NS);
        xb[rt] = LOAD8(xr1[rt], WPG * g + fm / NS);
        float v = sum8(xa[rt], one), u = sum8(xb[rt], one);
        if (NS == 1) { v = fma(simd_shuffle_xor(v, ushort(2)), one, v); u = fma(simd_shuffle_xor(u, ushort(2)), one, u); }
        v = fma(simd_shuffle_xor(v, ushort(4)), one, v); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
        v = fma(simd_shuffle_xor(v, ushort(16)), one, v); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
        xs0[rt] = v; xs1[rt] = u;
      }
      simdgroup_matrix<float, 8, 8> P[RT][NT];
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++)
        for (int t = 0; t < NT; t++) P[rt][t] = simdgroup_matrix<float, 8, 8>(0.0f);
      PRAGMA_UNROLL
      for (int s = 0; s < WPG; s++) {
        const int e = NS * s + fm % NS;          // this lane's input of the step (its MMA-k is fm)
        const float ps = pre(e);
        const uint mask = 0xFu << (4 * NS * s), mask1 = 0xFu << (4 * (NS * s + NS - 1));
        simdgroup_matrix<float, 8, 8> bm[RT];
        PRAGMA_UNROLL
        for (int rt = 0; rt < RT; rt++) {
          bm[rt].thread_elements()[0] = bf8(xa[rt], e) * ps;
          bm[rt].thread_elements()[1] = bf8(xb[rt], e) * ps;
        }
        PRAGMA_UNROLL
        for (int t = 0; t < NT; t++) {
          simdgroup_matrix<float, 8, 8> am;
          am.thread_elements()[0] = float(wv[t].x & mask);
          am.thread_elements()[1] = float(wv[t].y & mask1);
          PRAGMA_UNROLL
          for (int rt = 0; rt < RT; rt++) simdgroup_multiply_accumulate(P[rt][t], am, bm[rt], P[rt][t]);
        }
      }
      PRAGMA_UNROLL
      for (int t = 0; t < NT; t++) {
        const float sc = float(SC[size_t(wrow[t]) * G + g]);
        const float bi = float(BI[size_t(wrow[t]) * G + g]);
        PRAGMA_UNROLL
        for (int rt = 0; rt < RT; rt++) {
          acc[rt][t][0] = fma(bi, xs0[rt], fma(sc, P[rt][t].thread_elements()[0], acc[rt][t][0]));
          acc[rt][t][1] = fma(bi, xs1[rt], fma(sc, P[rt][t].thread_elements()[1], acc[rt][t][1]));
        }
      }
    }
    if (S == 1) {
      for (int rt = 0; rt < RT; rt++)
        for (int t = 0; t < NT; t++)
          for (int e = 0; e < 2; e++) {
            const int row = rb + 8 * rt + fn + e, n = nb + 8 * t + fm;
            if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(acc[rt][t][e]);
          }
      return;
    }
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++)
        for (int e = 0; e < 2; e++) red[((c * RT + rt) * NT + t) * 64 + int(lane) * 2 + e] = acc[rt][t][e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int idx = sg * 32 + int(lane); idx < RT * NT * 64; idx += SGS * 32) {
    float v[S];
    for (int k = 0; k < S; k++) v[k] = red[k * (RT * NT * 64) + idx];
    for (int w = 1; w < S; w *= 2)
      for (int k = 0; k + w < S; k += 2 * w) v[k] = fma(v[k + w], one, v[k]);
    const int rt = idx / (NT * 64), t = (idx / 64) % NT, l = (idx % 64) / 2, e = idx % 2;
    const int lq = l / 4;
    const int row = rb + 8 * rt + (lq & 2) * 2 + (l % 2) * 2 + e, n = nb + 8 * t + (lq & 4) + ((l / 2) % 4);
    if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(v[0]);
  }
"""

_PREP = r"""
  // one simdgroup a (row tile, group): writes the MMA kernel's input fragments, pre-scaled
  // (XF[tile][g][s][lane] = x[8 tile + fn + 0 / 1][64 g + 8 fm + s] * 2^-4s, rows past R copy row R - 1), and each
  // row's x-sum of the group by the kernels' tree (XS[r][g])
  const uint lane = thread_index_in_simdgroup;
  const int unit = int(threadgroup_position_in_grid.x) * 4 + int(simdgroup_index_in_threadgroup);
  const int R = X_shape[0];
  constexpr int G = K / 64;
  const int T8 = (R + 7) / 8;
  if (unit >= T8 * G) return;
  const int tile = unit / G, g = unit % G;
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const float one = ONE[0];
  const int r0 = min(8 * tile + fn, R - 1), r1 = min(8 * tile + fn + 1, R - 1);
  const uint4 xa = LOAD8(r0, 8 * g + fm), xb = LOAD8(r1, 8 * g + fm);
  device float2* xf = (device float2*)XF + (size_t(tile) * G + g) * 256 + lane;
  PRAGMA_UNROLL
  for (int s = 0; s < 8; s++) xf[32 * s] = float2(bf8(xa, s) * pre(s), bf8(xb, s) * pre(s));
  float v = sum8(xa, one), u = sum8(xb, one);
  v = fma(simd_shuffle_xor(v, ushort(2)), one, v); u = fma(simd_shuffle_xor(u, ushort(2)), one, u);
  v = fma(simd_shuffle_xor(v, ushort(4)), one, v); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
  v = fma(simd_shuffle_xor(v, ushort(16)), one, v); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
  if (fm == 0) {
    if (8 * tile + fn < R) XS[size_t(8 * tile + fn) * G + g] = v;
    if (8 * tile + fn + 1 < R) XS[size_t(8 * tile + fn + 1) * G + g] = u;
  }
"""


def _fragment_source(mma: str) -> str:
    """The MMA kernel reading pre-scaled fragments and input sums (the same values and bits)."""
    x_block = mma[
        mma.index("      uint4 xa[RT], xb[RT];") : mma.index(
            "      simdgroup_matrix<float, 8, 8> P[RT][NT];"
        )
    ]
    out = mma.replace(
        "  const int R = X_shape[0];",
        "  const int R = XS_shape[0];\n  const int T8 = (R + 7) / 8;\n"
        "  const device float2* XF2 = (const device float2*)XF;",
    )
    out = out.replace(
        x_block,
        """      float xs0[RT], xs1[RT];
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++) { xs0[rt] = XS[size_t(xr0[rt]) * G + g]; xs1[rt] = XS[size_t(xr1[rt]) * G + g]; }
""",
    )
    old_bm = """          bm[rt].thread_elements()[0] = bf8(xa[rt], e) * ps;
          bm[rt].thread_elements()[1] = bf8(xb[rt], e) * ps;"""
    if old_bm not in out:
        raise AssertionError("simd fragment kernel anchor changed")
    out = out.replace(
        old_bm,
        """          const float2 f = XF2[(size_t(min(rb / 8 + rt, T8 - 1)) * G + g) * 256 + 32 * s + lane];
          bm[rt].thread_elements()[0] = f.x;
          bm[rt].thread_elements()[1] = f.y;""",
    )
    return out


# ------------------------------------------------------- 5/6/8-bit simd ---
_BITS_HEADER = (
    _HEADER
    + r"""
// code j of B-bit codes packed from bit 0 of v (j a compile-time constant after unrolling)
template <int B>
inline uint code_at(const thread uint* v, const int j) {
  const int bit = j * B, word = bit >> 5, shift = bit & 31;
  uint c = v[word] >> shift;
  if (shift + B > 32) c |= v[word + 1] << (32 - shift);
  return c & ((1u << B) - 1u);
}
// float(c) for c < 2^23 without a convert: the same value
inline float cf(uint c) { return as_type<float>(0x4B000000u | c) - 8388608.0f; }
"""
)

_BITS_MMA = r"""
  // R rows: threadgroup (x, y) takes rows 8 RT y .. in RT tiles of 8 (rows >= R read row R - 1, dropped); SGS
  // simdgroups compute the S chunks in order. A lane's A elements are codes 8 fn .. 8 fn + 15 of its group.
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int R = X_shape[0];
  constexpr int G = K / 64, WPR = K * B / 32, LW = B == 8 ? 4 : 3;
  const float one = ONE[0];
  const int nb = int(threadgroup_position_in_grid.x) * (8 * NT);
  const int rb = int(threadgroup_position_in_grid.y) * (8 * RT);
  threadgroup float red[S > 1 ? S * RT * NT * 64 : 1];
  int wrow[NT];
  for (int t = 0; t < NT; t++) wrow[t] = min(nb + 8 * t + fm, N - 1);
  int xr0[RT], xr1[RT];
  for (int rt = 0; rt < RT; rt++) { xr0[rt] = min(rb + 8 * rt + fn, R - 1); xr1[rt] = min(rb + 8 * rt + fn + 1, R - 1); }
  for (int c = sg; c < S; c += SGS) {
    float acc[RT][NT][2];
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++) { acc[rt][t][0] = 0.0f; acc[rt][t][1] = 0.0f; }
    for (int g = c; g < G; g += S) {
      const int bit0 = 64 * B * g + 8 * B * fn;
      const bool half_word = (bit0 & 31) != 0;               // 5-bit lanes fn = 2, 6 start 16 bits into a word
      uint v[NT][4];
      PRAGMA_UNROLL
      for (int t = 0; t < NT; t++) {
        const device uint* wp = W + size_t(wrow[t]) * WPR + (bit0 >> 5);
        uint w[4];
        PRAGMA_UNROLL
        for (int i = 0; i < 4; i++) w[i] = i < LW ? wp[i] : 0u;
        v[t][0] = half_word ? (w[0] >> 16) | (w[1] << 16) : w[0];
        v[t][1] = half_word ? (w[1] >> 16) | (w[2] << 16) : w[1];
        v[t][2] = half_word ? (w[2] >> 16) : w[2];
        v[t][3] = w[3];
      }
      uint4 xa[RT], xb[RT];
      float xs0[RT], xs1[RT];
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++) {
        xa[rt] = LOAD8(xr0[rt], 8 * g + fm);
        xb[rt] = LOAD8(xr1[rt], 8 * g + fm);
        float a = sum8(xa[rt], one), u = sum8(xb[rt], one);
        a = fma(simd_shuffle_xor(a, ushort(2)), one, a); u = fma(simd_shuffle_xor(u, ushort(2)), one, u);
        a = fma(simd_shuffle_xor(a, ushort(4)), one, a); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
        a = fma(simd_shuffle_xor(a, ushort(16)), one, a); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
        xs0[rt] = a; xs1[rt] = u;
      }
      simdgroup_matrix<float, 8, 8> P[RT][NT];
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++)
        for (int t = 0; t < NT; t++) P[rt][t] = simdgroup_matrix<float, 8, 8>(0.0f);
      PRAGMA_UNROLL
      for (int s = 0; s < 8; s++) {
        simdgroup_matrix<float, 8, 8> bm[RT];
        PRAGMA_UNROLL
        for (int rt = 0; rt < RT; rt++) {
          bm[rt].thread_elements()[0] = bf8(xa[rt], s);
          bm[rt].thread_elements()[1] = bf8(xb[rt], s);
        }
        PRAGMA_UNROLL
        for (int t = 0; t < NT; t++) {
          simdgroup_matrix<float, 8, 8> am;
          am.thread_elements()[0] = cf(code_at<B>(v[t], s));
          am.thread_elements()[1] = cf(code_at<B>(v[t], 8 + s));
          PRAGMA_UNROLL
          for (int rt = 0; rt < RT; rt++) simdgroup_multiply_accumulate(P[rt][t], am, bm[rt], P[rt][t]);
        }
      }
      PRAGMA_UNROLL
      for (int t = 0; t < NT; t++) {
        const float sc = float(SC[size_t(wrow[t]) * G + g]);
        const float bi = float(BI[size_t(wrow[t]) * G + g]);
        PRAGMA_UNROLL
        for (int rt = 0; rt < RT; rt++) {
          acc[rt][t][0] = fma(bi, xs0[rt], fma(sc, P[rt][t].thread_elements()[0], acc[rt][t][0]));
          acc[rt][t][1] = fma(bi, xs1[rt], fma(sc, P[rt][t].thread_elements()[1], acc[rt][t][1]));
        }
      }
    }
    if (S == 1) {
      for (int rt = 0; rt < RT; rt++)
        for (int t = 0; t < NT; t++)
          for (int e = 0; e < 2; e++) {
            const int row = rb + 8 * rt + fn + e, n = nb + 8 * t + fm;
            if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(acc[rt][t][e]);
          }
      return;
    }
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++)
        for (int e = 0; e < 2; e++) red[((c * RT + rt) * NT + t) * 64 + int(lane) * 2 + e] = acc[rt][t][e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int idx = sg * 32 + int(lane); idx < RT * NT * 64; idx += SGS * 32) {
    float v[S];
    for (int k = 0; k < S; k++) v[k] = red[k * (RT * NT * 64) + idx];
    for (int w = 1; w < S; w *= 2)
      for (int k = 0; k + w < S; k += 2 * w) v[k] = fma(v[k + w], one, v[k]);
    const int rt = idx / (NT * 64), t = (idx / 64) % NT, l = (idx % 64) / 2, e = idx % 2;
    const int lq = l / 4;
    const int row = rb + 8 * rt + (lq & 2) * 2 + (l % 2) * 2 + e, n = nb + 8 * t + (lq & 4) + ((l / 2) % 4);
    if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(v[0]);
  }
"""

_BITS_SCALAR = r"""
  // RS rows (1 to 4). Lane (chunk c = lane % S, slot j = lane / S) runs chunk c of NR outputs n0 + j + (32 / S) u,
  // a whole group (2 B words) of each in registers; the threadgroup stages XB groups of each row's inputs in chain
  // order (step s, k at 8 s + k). A row's chain is the same at any RS and the matrix kernel's.
  constexpr int GW = 2 * B, XP = 76, G = K / 64, WPR = K * B / 32;
  threadgroup float xs[RS * XB * XP];
  const uint lane = thread_index_in_simdgroup;
  const int tid = int(simdgroup_index_in_threadgroup) * 32 + int(lane);
  const int c = int(lane) % S;
  constexpr int SLOTS = 32 / S;
  const int n0 = (int(threadgroup_position_in_grid.x) * SGS + int(simdgroup_index_in_threadgroup)) * (SLOTS * NR)
                 + int(lane) / S;
  const float one = ONE[0];
  const device uint* wr[NR];
  const device bfloat* sr[NR];
  const device bfloat* br[NR];
  float acc[NR][RS];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) {
    const int nn = min(n0 + SLOTS * u, N - 1);
    wr[u] = W + size_t(nn) * WPR;
    sr[u] = SC + size_t(nn) * G;
    br[u] = BI + size_t(nn) * G;
    PRAGMA_UNROLL
    for (int r = 0; r < RS; r++) acc[u][r] = 0.0f;
  }
  uint nw[NR][GW];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) for (int h = 0; h < GW; h++) nw[u][h] = c < G ? wr[u][GW * c + h] : 0u;
  for (int b0 = 0; b0 < G; b0 += XB) {
    const int nbk = min(XB, G - b0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < RS * nbk * 8; idx += SGS * 32) {
      const int r = RS == 1 ? 0 : idx / (nbk * 8);
      const int gl = (RS == 1 ? idx : idx - r * (nbk * 8)) / 8, j = idx % 8;
      const uint4 v = LOAD8(r, 8 * (b0 + gl) + j);
      threadgroup float* xr = xs + r * (XB * XP) + gl * XP;
      PRAGMA_UNROLL
      for (int e = 0; e < 8; e++) xr[8 * e + j] = bf8(v, e);
      xr[64 + j] = sum8(v, one);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = b0 + c; g < b0 + nbk; g += S) {
      uint wv[NR][GW];
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) for (int h = 0; h < GW; h++) wv[u][h] = nw[u][h];
      if (g + S < G) {
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++) for (int h = 0; h < GW; h++) nw[u][h] = wr[u][GW * (g + S) + h];
      }
      float xsum[RS];
      float P[NR][RS];
      PRAGMA_UNROLL
      for (int r = 0; r < RS; r++) {
        const threadgroup float* xg = xs + r * (XB * XP) + (g - b0) * XP;
        const float4 p0 = *(const threadgroup float4*)(xg + 64), p1 = *(const threadgroup float4*)(xg + 68);
        xsum[r] = fma(fma(p0.w, one, p0.z), one, fma(p0.y, one, p0.x));
        xsum[r] = fma(fma(fma(p1.w, one, p1.z), one, fma(p1.y, one, p1.x)), one, xsum[r]);
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++) P[u][r] = 0.0f;
      }
      PRAGMA_UNROLL
      for (int s = 0; s < 8; s++) {
        float xq[RS][8];
        PRAGMA_UNROLL
        for (int r = 0; r < RS; r++) {
          const threadgroup float* xg = xs + r * (XB * XP) + (g - b0) * XP + 8 * s;
          const float4 lo = *(const threadgroup float4*)(xg), hi = *(const threadgroup float4*)(xg + 4);
          xq[r][0] = lo.x; xq[r][1] = lo.y; xq[r][2] = lo.z; xq[r][3] = lo.w;
          xq[r][4] = hi.x; xq[r][5] = hi.y; xq[r][6] = hi.z; xq[r][7] = hi.w;
        }
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++)
          PRAGMA_UNROLL
          for (int k = 0; k < 8; k++) {
            const float q = cf(code_at<B>(wv[u], 8 * k + s));
            PRAGMA_UNROLL
            for (int r = 0; r < RS; r++) P[u][r] = fma(xq[r][k], q, P[u][r]);
          }
      }
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) {
        const float sc = float(sr[u][g]), bi = float(br[u][g]);
        PRAGMA_UNROLL
        for (int r = 0; r < RS; r++) {
          acc[u][r] = fma(sc, P[u][r], acc[u][r]);
          acc[u][r] = fma(bi, xsum[r], acc[u][r]);
        }
      }
    }
  }
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++)
    PRAGMA_UNROLL
    for (int r = 0; r < RS; r++) {
      float v = acc[u][r];
      PRAGMA_UNROLL
      for (int m = 1; m < S; m <<= 1) v = fma(simd_shuffle_xor(v, ushort(m)), one, v);
      const int n = n0 + SLOTS * u;
      if (n < N && c == 0) OUT[size_t(r) * N + n] = bfloat(v);
    }
"""

# ----------------------------------------------------------- affine rows ---
_AFFINE_HEADER = r"""
#define PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
// code i of a 32-code block held in BITS words (i is a compile-time constant after unrolling)
template <int BITS>
inline uint code_at(const thread uint* w, const int i) {
  const int bit = i * BITS, word = bit >> 5, shift = bit & 31;
  uint v = w[word] >> shift;
  if (shift + BITS > 32) v |= w[word + 1] << (32 - shift);
  return v & ((1u << BITS) - 1u);
}
// the bf16 in the low (h = 0) or high (h = 1) half of v, as fp32
inline float bf_half(uint v, int h) { return as_type<float>(h ? (v & 0xFFFF0000u) : (v << 16)); }
"""

_AFFINE = r"""
  // A simdgroup: OPS outputs of up to RT rows. Lane l takes 32-code blocks l, l + 32, ... of each output: it reads the
  // block's BITS words once, dequantizes each code once (fma(scale, q, bias)) and folds it into every row's own fma
  // chain, then one simd_sum. A row's bits never depend on OPS, RT or the rows beside it.
  constexpr int BLOCKS = K / 32, GROUPS = K / GS, BPG = GS / 32, WPR = K * BITS / 32, STEPS = (BLOCKS + 31) / 32;
  const int lane = int(thread_index_in_simdgroup);
  const int n0 = (int(threadgroup_position_in_grid.x) * SG + int(simdgroup_index_in_threadgroup)) * OPS;
  const int r0 = int(threadgroup_position_in_grid.y) * RT;
  const int rows = min(RT, int(X_shape[0]) - r0);
  const device uint* xw = (const device uint*)X;
  int nn[OPS];
  float acc[OPS][RT];
  PRAGMA_UNROLL
  for (int u = 0; u < OPS; u++) {
    nn[u] = min(n0 + u, N - 1);
    PRAGMA_UNROLL
    for (int r = 0; r < RT; r++) acc[u][r] = 0.0f;
  }
  for (int s = 0; s < STEPS; s++) {
    const int b = s * 32 + lane;
    if (b >= BLOCKS) break;
    uint wd[OPS][BITS];
    float sc[OPS], bi[OPS];
    PRAGMA_UNROLL
    for (int u = 0; u < OPS; u++) {
      const device uint* p = W + size_t(nn[u]) * WPR + size_t(b) * BITS;
      PRAGMA_UNROLL
      for (int j = 0; j < BITS; j++) wd[u][j] = p[j];
      const size_t g = size_t(nn[u]) * GROUPS + b / BPG;
      sc[u] = float(SC[g]);
      bi[u] = float(BI[g]);
    }
    PRAGMA_UNROLL
    for (int c = 0; c < 4; c++) {
      float wv[OPS][8];
      PRAGMA_UNROLL
      for (int u = 0; u < OPS; u++)
        PRAGMA_UNROLL
        for (int i = 0; i < 8; i++) wv[u][i] = fma(sc[u], float(code_at<BITS>(wd[u], 8 * c + i)), bi[u]);
      PRAGMA_UNROLL
      for (int r = 0; r < RT; r++) {
        if (r < rows) {
          const size_t at = (size_t(r0 + r) * K + size_t(b) * 32 + 8 * c) / 2;
          float xv[8];
          PRAGMA_UNROLL
          for (int h = 0; h < 4; h++) {
            const uint v = xw[at + h];
            xv[2 * h] = bf_half(v, 0);
            xv[2 * h + 1] = bf_half(v, 1);
          }
          PRAGMA_UNROLL
          for (int u = 0; u < OPS; u++)
            PRAGMA_UNROLL
            for (int i = 0; i < 8; i++) acc[u][r] = fma(xv[i], wv[u][i], acc[u][r]);
        }
      }
    }
  }
  PRAGMA_UNROLL
  for (int r = 0; r < RT; r++) {
    if (r < rows) {
      PRAGMA_UNROLL
      for (int u = 0; u < OPS; u++) {
        const float total = simd_sum(acc[u][r]);
        if (lane == 0 && n0 + u < N) OUT[size_t(r0 + r) * N + n0 + u] = bfloat(total);
      }
    }
  }
"""

# ------------------------------------------------------------- launching ---
_KERNELS: dict[tuple, Any] = {}
_PLANS: dict[tuple, Any] = {}
_ONE: list = []


def _one():
    if not _ONE:
        _ONE.append(mx.array([1.0], dtype=mx.float32))
    return _ONE[0]


def _compiled(kind: str, consts: tuple[tuple[str, int], ...]) -> Any:
    """One kernel per kind and constants, constants in the source (MLX parses template args per call)."""
    key = (kind, consts)
    kernel = _KERNELS.get(key)
    if kernel is None:
        body, header, inputs, outputs = {
            "scalar": (_SCALAR, _HEADER, ["X", "W", "SC", "BI", "ONE"], ["OUT"]),
            "mma": (_MMA, _HEADER, ["X", "W", "SC", "BI", "ONE"], ["OUT"]),
            "prep": (_PREP, _HEADER, ["X", "ONE"], ["XF", "XS"]),
            "mmaf": (
                _fragment_source(_MMA),
                _HEADER,
                ["XF", "XS", "W", "SC", "BI", "ONE"],
                ["OUT"],
            ),
            "bits_scalar": (
                _BITS_SCALAR,
                _BITS_HEADER,
                ["X", "W", "SC", "BI", "ONE"],
                ["OUT"],
            ),
            "bits_mma": (
                _BITS_MMA,
                _BITS_HEADER,
                ["X", "W", "SC", "BI", "ONE"],
                ["OUT"],
            ),
            "affine": (_AFFINE, _AFFINE_HEADER, ["X", "W", "SC", "BI"], ["OUT"]),
        }[kind]
        source = (
            "".join(f"  constexpr int {k} = {v};\n" for k, v in consts)
            + f"  #define LOAD8(r, j) ({_LOAD8})\n"
            + body
            + "  #undef LOAD8\n"
        )
        name = f"rapid_lane_simd_{kind}_{hashlib.sha256((header + source).encode()).hexdigest()[:16]}"
        kernel = _KERNELS[key] = mx.fast.metal_kernel(
            name=name,
            input_names=inputs,
            output_names=outputs,
            source=source,
            header=header,
        )
    return kernel


def splits(n: int, k: int) -> int:
    """K chunks by weight shape alone (stacking projections changes the tree, so the bits)."""
    return 32 if n <= 64 else (16 if n <= 6144 else 8)


def _tiles(n: int, rows: int, s: int) -> int:
    nt = 4 if n % 32 == 0 else (2 if n % 16 == 0 else 1)
    while nt > 1 and s * ((rows + 7) // 8) * nt * 64 * 4 > 16384:
        nt //= 2
    return nt


def _scalar_block(rows: int, s: int, group: int = GROUP) -> int:
    if not 1 <= rows <= SCALAR_ROWS:
        return 0
    xb = XB if rows == 1 else max(s, XB // 2)
    return (
        xb
        if xb % s == 0 and rows * xb * (76 if group == 64 else 44) * 4 <= 20480
        else 0
    )


def _scalar_kind(rows: int, n: int, k: int, group: int) -> bool:
    limit = 3 if group == 32 and n > 6144 else 2
    return (
        rows <= limit
        and bool(_scalar_block(rows, splits(n, k), group))
        and (n, k, group) not in mma_one_row
    )


def _launch(
    kind: str, rows: int, n: int, k: int, group: int, bits: int, most: int = MMA_SGS
) -> tuple:
    """(constants, grid, threadgroup, output shapes); an MMA launch uses up to ``most`` simdgroups."""
    s = splits(n, k)
    if kind in ("scalar", "bits_scalar"):
        xb = _scalar_block(rows, s, group)
        if not xb:
            raise SimdUnsupportedError("scalar kernel cannot take these rows")
        if kind == "scalar":
            nr = NR if n > 2048 else 1
            extra = (("GS", group),)
        else:
            nr = 1 if bits == 8 or n <= 2048 else NR
            extra = (("B", bits),)
        sgs = max(1, 16 // ((32 // s) * nr)) if n > 2048 else 8
        per = sgs * (32 // s) * nr
        consts: tuple[tuple[str, int], ...] = (
            ("K", k),
            ("N", n),
            ("S", s),
            ("SGS", sgs),
            ("NR", nr),
            ("XB", xb),
            ("RS", rows),
            *extra,
        )
        return consts, (-(-n // per) * sgs * 32, 1, 1), (sgs * 32, 1, 1), [(rows, n)]
    rt = min(RT_MAX, (rows + 7) // 8)
    nt = _tiles(n, rt * 8, s)
    sgs = min(s, most)
    extra = (("B", bits),) if kind == "bits_mma" else (("GS", group),)
    consts = (
        ("K", k),
        ("N", n),
        ("S", s),
        ("SGS", sgs),
        ("NT", nt),
        ("RT", rt),
        *extra,
    )
    return (
        consts,
        (-(-n // (8 * nt)) * sgs * 32, -(-rows // (8 * rt)), 1),
        (sgs * 32, 1, 1),
        [(rows, n)],
    )


def _go(kind: str, plan: tuple, inputs: list) -> mx.array:
    consts, grid, tg, oshape = plan
    return _compiled(kind, consts)(
        inputs=inputs,
        grid=grid,
        threadgroup=tg,
        output_shapes=oshape,
        output_dtypes=[mx.bfloat16],
    )[0]


def _run(
    kind: str, rows: int, n: int, k: int, group: int, bits: int, inputs: list
) -> mx.array:
    """Launch through the call's cached plan; a new MMA plan takes the simdgroups its pipeline allows here."""
    key = (kind, rows, n, k, group, bits)
    plan = _PLANS.get(key)
    if plan is not None:
        return _go(kind, plan, inputs)
    if kind in ("scalar", "bits_scalar"):
        plan = _PLANS[key] = _launch(kind, rows, n, k, group, bits)
        return _go(kind, plan, inputs)
    consts = _launch(kind, rows, n, k, group, bits)[0]
    made: list[tuple] = []

    def launch(size: int) -> mx.array:
        made.append(_launch(kind, rows, n, k, group, bits, size // 32))
        return _go(kind, made[-1], inputs)

    pipeline = (f"lane-simd {kind}", tuple(c for c in consts if c[0] != "SGS"))
    out = _fit(
        pipeline,
        [32 * g for g in (16, 8, 4, 2, 1) if g <= dict(consts)["SGS"]],
        launch,
        inputs,
    )
    if pipeline in _fitted:
        _PLANS[key] = made[-1]
    return out


def _fragments(x2: mx.array) -> tuple[mx.array, mx.array]:
    rows, k = int(x2.shape[0]), int(x2.shape[1])
    units = -(-rows // 8) * (k // GROUP)
    xf, xs = _compiled("prep", (("K", k),))(
        inputs=[x2, _one()],
        grid=(-(-units // 4) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(units * 512,), (rows, k // GROUP)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return xf, xs


def _affine_launch(rows: int) -> tuple[int, int]:
    """(outputs a simdgroup, rows a threadgroup); every launch has the same per-row arithmetic."""
    return 2, (1 if rows == 1 else 2 if rows == 2 else 4 if rows <= 4 else 8)


def _padded(a: mx.array) -> mx.array:
    # Inputs under 8 elements are padded so the argument buffer's address
    # space stays stable across calls (TensorFold kernels/inputs.padded).
    if a.size >= 8:
        return a
    return mx.concatenate([a.reshape(-1), mx.zeros((8 - a.size,), dtype=a.dtype)])


# ------------------------------------------------------------- the path ---
def path(
    bits: int, group_size: int, n: int, k: int, scales_dtype, biases_dtype=None
) -> str:
    """Kernel family for one weight: "simd4", "simd_bits" or "affine" (fixed per weight).

    The matrix kernels read scales and biases as bf16; anything else takes
    ``affine_rows``, which reads their own dtype.
    """
    bf16 = scales_dtype == mx.bfloat16 and (biases_dtype or scales_dtype) == mx.bfloat16
    if (
        bits == 4
        and group_size in (32, 64)
        and bf16
        and n % 8 == 0
        and k % group_size == 0
    ):
        return "simd4"
    if (
        bits in BITS_MMA
        and group_size == GROUP
        and bf16
        and n % 8 == 0
        and k % GROUP == 0
        and (n, k, bits) not in bits_fallback
    ):
        return "simd_bits"
    return "affine"


def check_geometry(
    *,
    bits: int,
    group_size: int,
    mode: str,
    n: int,
    k: int,
    weight_dtype,
    scales_dtype,
    biases_dtype=None,
    weight_ndim: int = 2,
) -> None:
    """Raise SimdUnsupportedError unless one of the kernel families covers this weight."""
    if weight_ndim != 2:
        raise SimdUnsupportedError("packed weights must be rank 2")
    if biases_dtype is not None and biases_dtype != scales_dtype:
        raise SimdUnsupportedError("scales and biases must share one dtype")
    if mode != "affine":
        raise SimdUnsupportedError(f"quantization mode {mode!r} is not affine")
    if bits not in AFFINE_BITS or group_size not in AFFINE_GROUPS:
        raise SimdUnsupportedError(
            f"{bits}-bit weights in groups of {group_size} are not supported"
        )
    if weight_dtype != mx.uint32:
        raise SimdUnsupportedError("packed weights must be uint32")
    if scales_dtype not in (mx.bfloat16, mx.float16, mx.float32):
        raise SimdUnsupportedError("scales and biases must be bf16, fp16 or fp32")
    if k % group_size or k % 32:
        raise SimdUnsupportedError("K must be a multiple of the group size and of 32")
    if n < 1:
        raise SimdUnsupportedError("N must be positive")


def qmm(
    x2: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    *,
    kind: str | None = None,
) -> mx.array:
    """Row-exact bf16 ``x2 @ W.T`` for rows x K input; every row's bits are its one-row call's."""
    if x2.dtype != mx.bfloat16:
        raise SimdUnsupportedError("activations must be bf16")
    rows, k = int(x2.shape[0]), int(x2.shape[1])
    n = int(weight.shape[0])
    if not 1 <= rows <= MAX_ROWS:
        raise SimdUnsupportedError(f"simd matmul takes 1-{MAX_ROWS} rows, got {rows}")
    family = path(bits, group_size, n, k, scales.dtype, biases.dtype)
    x2 = mx.contiguous(x2)
    if family == "simd4":
        if kind is None and 2 <= rows <= FRAGMENT_ROWS and group_size == GROUP:
            # Same bits as the matrix kernel (and, checked, the scalar twin).
            xf, xs = _fragments(x2)
            return _run(
                "mmaf",
                rows,
                n,
                k,
                group_size,
                bits,
                [xf, xs, weight, scales, biases, _one()],
            )
        if kind is None:
            kind = "scalar" if _scalar_kind(rows, n, k, group_size) else "mma"
        return _run(
            kind, rows, n, k, group_size, bits, [x2, weight, scales, biases, _one()]
        )
    if family == "simd_bits":
        if kind is None:
            kind = (
                "scalar"
                if rows == 1 and _scalar_block(rows, splits(n, k), GROUP)
                else "mma"
            )
        return _run(
            f"bits_{kind}",
            rows,
            n,
            k,
            group_size,
            bits,
            [x2, weight, scales, biases, _one()],
        )
    ops, rt = _affine_launch(rows)
    consts = (
        ("K", k),
        ("N", n),
        ("BITS", bits),
        ("GS", group_size),
        ("OPS", ops),
        ("RT", rt),
        ("SG", AFFINE_SG),
    )
    return _compiled("affine", consts)(
        inputs=[
            x2,
            _padded(mx.contiguous(weight)),
            _padded(mx.contiguous(scales)),
            _padded(mx.contiguous(biases)),
        ],
        grid=(-(-n // (AFFINE_SG * ops)) * 32 * AFFINE_SG, -(-rows // rt), 1),
        threadgroup=(32 * AFFINE_SG, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[mx.bfloat16],
    )[0]


def check(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    *,
    seed: int = 0,
) -> bool:
    """The scalar twin's 1-4-row calls against the matrix kernel's rows, bit for bit, on this GPU.

    Records the outcome: a 4-bit shape that differs sends its 1-4-row calls to
    the matrix kernel; a 5/6/8-bit shape that differs takes ``affine_rows``.
    """
    n = int(weight.shape[0])
    k = int(weight.shape[1]) * 32 // bits
    family = path(bits, group_size, n, k, scales.dtype, biases.dtype)
    if family == "affine":
        return (n, k, bits) not in bits_fallback
    x = (mx.random.normal((8, k), key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)
    full = qmm(x, weight, scales, biases, group_size, bits, kind="mma")
    s = splits(n, k)
    calls = [(r, 1) for r in range(8)]
    calls += [
        (r, m)
        for m in range(2, SCALAR_ROWS + 1)
        if _scalar_block(m, s, group_size)
        for r in (0, 8 - m)
    ]
    same = all(
        bool(
            mx.array_equal(
                qmm(
                    x[r : r + m],
                    weight,
                    scales,
                    biases,
                    group_size,
                    bits,
                    kind="scalar",
                ),
                full[r : r + m],
            ).item()
        )
        for r, m in calls
    )
    if not same:
        (
            mma_one_row.add((n, k, group_size))
            if family == "simd4"
            else bits_fallback.add((n, k, bits))
        )
    return same


def rerouted(n: int, k: int, group_size: int, bits: int) -> str | None:
    """How a failed twin check changed this weight's route on this GPU, if it did.

    "affine": a 5/6/8-bit shape moved to ``affine_rows``, a different
    arithmetic, so it belongs in the law.  "mma": 4-bit 1-4-row calls take the
    matrix kernel, whose bits every other row count already uses.
    """
    if (n, k, bits) in bits_fallback:
        return "affine"
    if bits == 4 and (n, k, group_size) in mma_one_row:
        return "mma"
    return None


def available() -> bool:
    """True on an Apple GPU (the simdgroup matrix units every Metal GPU has)."""
    try:
        return bool(mx.default_device() == mx.gpu and mx.metal.is_available())
    except Exception:  # noqa: BLE001 - absent Metal means unavailable
        return False


__all__ = [
    "MAX_ROWS",
    "SimdUnsupportedError",
    "available",
    "check",
    "check_geometry",
    "path",
    "qmm",
    "splits",
]
