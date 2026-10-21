# The affine QMV helpers in ``_HEADER`` (``load_x``, ``qdot_w`` and the row
# loops, including mixed-width packing) and ``sigmoid_ref`` adapt MLX's
# Metal quantized and unary kernels,
# which carry this notice:
#
# MIT License
#
# Copyright © 2023 Apple Inc.
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

"""Fused decode MoE kernels for V4.1: three dispatches instead of ~40.

A decode step routes a handful of rows. The graph path spends one GPU dispatch
per elementwise op (score function, bias, top-k, normalization, clamps, SiLU,
casts, the weighted reduction), so a layer's MoE costs about forty serialized
launches although its weight traffic is a few megabytes. Here it is three:

* ``route`` — score function, selection bias, top-k, normalization and scale
  for every row in one threadgroup per row;
* ``up`` — routed (2/3/4-bit) and shared (2/4/8-bit) gate/up projections for every
  route of every row, with the clamped SwiGLU in the epilogue;
* ``down`` — routed and shared down projections plus the weighted combine.

Numerics replicate the stock path operation by operation: the affine QMV
kernels keep the stock lane geometry, values per thread, block order and
``scale * dot + sum * bias`` epilogue; projection outputs are rounded to the
stock output dtype before activation; top-k ties keep the stable-sort order;
the routed reduction runs in route order. :func:`moe_forward` qualifies each
expert layout once against the stock path and falls back if it differs.
"""

from __future__ import annotations

import logging
import weakref
from functools import lru_cache
from typing import NamedTuple

import mlx.core as mx

from . import head_gemv
from .kernels import custom_kernels_enabled

logger = logging.getLogger(__name__)

# Module switch for A/B runs and the negative-control tests.
ENABLED = True
MAX_ROWS = 8
_CHECK_ATTR = "_rapid_fused_moe_check"

_HEADER = r"""
template <typename T, int BITS, int VPT>
inline float load_x(const device T* x, thread float* xt) {
  float sum = 0;
  if (BITS == 2) {
    for (int i = 0; i < VPT; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      xt[i] = x[i];
      xt[i + 1] = x[i + 1] / 4.0f;
      xt[i + 2] = x[i + 2] / 16.0f;
      xt[i + 3] = x[i + 3] / 64.0f;
    }
  } else if (BITS == 3) {
    for (int i = 0; i < VPT; i += 8) {
      sum += float(x[i]) + float(x[i+1]) + float(x[i+2]) + float(x[i+3])
          + float(x[i+4]) + float(x[i+5]) + float(x[i+6]) + float(x[i+7]);
      xt[i] = x[i];
      xt[i+1] = x[i+1] / 8.0f;
      xt[i+2] = x[i+2] / 64.0f;
      xt[i+3] = x[i+3] / 2.0f;
      xt[i+4] = x[i+4] / 16.0f;
      xt[i+5] = x[i+5] / 128.0f;
      xt[i+6] = x[i+6] / 4.0f;
      xt[i+7] = x[i+7] / 32.0f;
    }
  } else if (BITS == 4) {
    for (int i = 0; i < VPT; i += 4) {
      sum += float(x[i]) + float(x[i+1]) + float(x[i+2]) + float(x[i+3]);
      xt[i] = x[i];
      xt[i+1] = x[i+1] / 16.0f;
      xt[i+2] = x[i+2] / 256.0f;
      xt[i+3] = x[i+3] / 4096.0f;
    }
  } else {
    for (int i = 0; i < VPT; i++) {
      sum += x[i];
      xt[i] = x[i];
    }
  }
  return sum;
}

template <int BITS, int VPT>
inline float qdot_w(
    const device uint8_t* w, const thread float* xt, float s, float b, float sum) {
  float accum = 0;
  if (BITS == 2) {
    for (int i = 0; i < VPT / 4; i++) {
      accum +=
          (xt[4 * i] * (w[i] & 0x03) + xt[4 * i + 1] * (w[i] & 0x0c) +
           xt[4 * i + 2] * (w[i] & 0x30) + xt[4 * i + 3] * (w[i] & 0xc0));
    }
  } else if (BITS == 3) {
    for (int i = 0; i < VPT / 8; i++) {
      const thread float* x = xt + 8 * i;
      const device uint8_t* p = w + 3 * i;
      accum += (p[0] & 0x07) * x[0];
      accum += (p[0] & 0x38) * x[1];
      accum += (p[0] & 0xc0) * x[2];
      accum += (p[1] & 0x01) * (x[2] * 256.0f);
      accum += (p[1] & 0x0e) * x[3];
      accum += (p[1] & 0x70) * x[4];
      accum += (p[1] & 0x80) * x[5];
      accum += (p[2] & 0x03) * (x[5] * 256.0f);
      accum += (p[2] & 0x1c) * x[6];
      accum += (p[2] & 0xe0) * x[7];
    }
  } else if (BITS == 4) {
    const device uint16_t* ws = reinterpret_cast<const device uint16_t*>(w);
    for (int i = 0; i < VPT / 4; i++) {
      accum += (xt[4*i] * (ws[i] & 0x000f) + xt[4*i+1] * (ws[i] & 0x00f0)
          + xt[4*i+2] * (ws[i] & 0x0f00) + xt[4*i+3] * (ws[i] & 0xf000));
    }
  } else {
    for (int i = 0; i < VPT; i++) {
      accum += xt[i] * w[i];
    }
  }
  return s * accum + sum * b;
}

// Lane partials of ROWS consecutive output rows of one affine matrix (and of
// a second matrix sharing the input when DUAL), in stock block order.
template <typename T, typename Q, int BITS, int VPT, int K, int ROWS, bool DUAL>
inline void rows_partial(
    const device T* x,
    const device uint32_t* w0, const device Q* s0, const device Q* b0,
    const device uint32_t* w1, const device Q* s1, const device Q* b1,
    size_t row, uint lane, thread float* acc0, thread float* acc1) {
  constexpr int BLOCK = VPT * 32;
  constexpr int ROW_BYTES = K * BITS / 8;
  constexpr int GROUPS = K / 64;
  const device uint8_t* wb0 = reinterpret_cast<const device uint8_t*>(w0);
  const device uint8_t* wb1 = reinterpret_cast<const device uint8_t*>(w1);
  float xt[VPT];
  for (int k = 0; k < K; k += BLOCK) {
    const int pos = k + int(lane) * VPT;
    if (pos >= K) {
      break;
    }
    const float sum = load_x<T, BITS, VPT>(x + pos, xt);
    const int byte = pos * BITS / 8;
    const int group = pos / 64;
    for (int r = 0; r < ROWS; r++) {
      const size_t rr = row + r;
      acc0[r] += qdot_w<BITS, VPT>(
          wb0 + rr * ROW_BYTES + byte, xt, float(s0[rr * GROUPS + group]),
          float(b0[rr * GROUPS + group]), sum);
      if (DUAL) {
        acc1[r] += qdot_w<BITS, VPT>(
            wb1 + rr * ROW_BYTES + byte, xt, float(s1[rr * GROUPS + group]),
            float(b1[rr * GROUPS + group]), sum);
      }
    }
  }
}

// Shared-expert variant: every row's partials over the same weight rows,
// block by block, so each weight byte is fetched once for all rows.
template <typename T, typename Q, int BITS, int VPT, int K, int NROWS, bool DUAL>
inline void rows_partial_multi(
    const device T* x,
    const device uint32_t* w0, const device Q* s0, const device Q* b0,
    const device uint32_t* w1, const device Q* s1, const device Q* b1,
    size_t row, uint lane, thread float (*acc0)[4], thread float (*acc1)[4]) {
  constexpr int BLOCK = VPT * 32;
  constexpr int ROW_BYTES = K * BITS / 8;
  constexpr int GROUPS = K / 64;
  const device uint8_t* wb0 = reinterpret_cast<const device uint8_t*>(w0);
  const device uint8_t* wb1 = reinterpret_cast<const device uint8_t*>(w1);
  float xt[VPT];
  for (int k = 0; k < K; k += BLOCK) {
    const int pos = k + int(lane) * VPT;
    if (pos >= K) {
      break;
    }
    const int byte = pos * BITS / 8;
    const int group = pos / 64;
    for (int t = 0; t < NROWS; t++) {
      const float sum = load_x<T, BITS, VPT>(x + t * K + pos, xt);
      for (int r = 0; r < 4; r++) {
        const size_t rr = row + r;
        acc0[t][r] += qdot_w<BITS, VPT>(
            wb0 + rr * ROW_BYTES + byte, xt, float(s0[rr * GROUPS + group]),
            float(b0[rr * GROUPS + group]), sum);
        if (DUAL) {
          acc1[t][r] += qdot_w<BITS, VPT>(
              wb1 + rr * ROW_BYTES + byte, xt, float(s1[rr * GROUPS + group]),
              float(b1[rr * GROUPS + group]), sum);
        }
      }
    }
  }
}

// Stock floating minimum/maximum: a NaN first operand propagates.
inline float minimum_ref(float x, float y) {
  return metal::isnan(x) ? x : (x < y ? x : y);
}

inline float maximum_ref(float x, float y) {
  return metal::isnan(x) ? x : (x > y ? x : y);
}

inline float sigmoid_ref(float x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""

_ROUTE = r"""
    // One threadgroup per row, one thread per expert.
    const uint e = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.y;
    threadgroup float biased[E];
    threadgroup float score[E];
    float s = scores[row * E + e] / params[0];
    float sp;
    if (metal::isnan(s)) {
      sp = metal::numeric_limits<float>::quiet_NaN();
    } else {
      const float hi = metal::max(s, 0.0f);
      const float lo = metal::min(s, 0.0f);
      sp = (lo == -INFINITY || hi == INFINITY) ? hi
          : (hi + log1p(metal::exp(lo - hi)));
    }
    const float sc = metal::precise::sqrt(sp);
    score[e] = sc;
    biased[e] = sc + bias[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Stable ascending order of -biased, NaN last (the stock sort order):
    // a strict total order, so the ranks are a permutation and every
    // picked slot is written even when the scores carry NaN.
    const float mine = -biased[e];
    const bool mine_nan = metal::isnan(mine);
    int rank = 0;
    for (uint j = 0; j < E; j++) {
      const float other = -biased[j];
      const bool other_nan = metal::isnan(other);
      const bool before = (mine_nan || other_nan)
          ? (!other_nan || (mine_nan && j < e))
          : (other < mine || (other == mine && j < e));
      rank += before ? 1 : 0;
    }
    threadgroup uint picked[TOPK];
    if (rank < TOPK) {
      picked[rank] = e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) {
      float w[TOPK];
      float total = 0.0f;
      for (int k = 0; k < TOPK; k++) {
        w[k] = score[picked[k]];
        total += w[k];
      }
      const float denom = total + 1e-20f;
      for (int k = 0; k < TOPK; k++) {
        float v = w[k];
        if (NORMALIZE) {
          v = v / denom;
        }
        weights[row * TOPK + k] = v * params[1];
        indices[row * TOPK + k] = picked[k];
      }
    }
"""

_UP = r"""
    // z: routed slots row-major (row, k), then one shared slot holding every
    // row so the shared weights are fetched once per threadgroup.
    const uint lane = thread_index_in_simdgroup;
    const int slot = int(threadgroup_position_in_grid.z);
    const int col = int(threadgroup_position_in_grid.y) * 8
        + int(simdgroup_index_in_threadgroup) * 4;
    const bool shared_slot = slot >= ROWS * TOPK;
    float g[ROWS][4] = {};
    float u[ROWS][4] = {};
    if (shared_slot) {
      rows_partial_multi<T, Q, SBITS, SVPT, K, ROWS, true>(
          x, sgw, sgs, sgb, suw, sus, sub, size_t(col), lane, g, u);
    } else {
      const size_t base = size_t(indices[slot]) * N + col;
      if (GBITS == UBITS && GVPT == UVPT) {
        rows_partial<T, Q, GBITS, GVPT, K, 4, true>(
            x + (slot / TOPK) * K, gw, gs, gb, uw, us, ub, base, lane, g[0], u[0]);
      } else {
        rows_partial<T, Q, GBITS, GVPT, K, 4, false>(
            x + (slot / TOPK) * K, gw, gs, gb, gw, gs, gb, base, lane, g[0], g[0]);
        rows_partial<T, Q, UBITS, UVPT, K, 4, false>(
            x + (slot / TOPK) * K, uw, us, ub, uw, us, ub, base, lane, u[0], u[0]);
      }
    }
    const int tokens = shared_slot ? ROWS : 1;
    for (int t = 0; t < tokens; t++) {
      const int out_slot = shared_slot ? ROWS * TOPK + t : slot;
      for (int r = 0; r < 4; r++) {
        const float gate_sum = simd_sum(g[t][r]);
        const float up_sum = simd_sum(u[t][r]);
        if (lane == 0) {
          float gate = float(static_cast<T>(gate_sum));
          float up = float(static_cast<T>(up_sum));
          if (LIMIT > 0) {
            up = minimum_ref(maximum_ref(up, -LIMIT), LIMIT);
            gate = minimum_ref(gate, LIMIT);
          }
          float h = (gate * sigmoid_ref(gate)) * up;
          if (shared_slot) {
            h = float(static_cast<T>(h));
          }
          act[out_slot * N + col + r] = h;
        }
      }
    }
"""

_DOWN = r"""
    // Four output columns of every row per threadgroup: simdgroup k < TOPK
    // runs route k of each row, simdgroup TOPK the shared expert for all rows
    // at once; simdgroup 0 then combines each row in route order.
    const uint lane = thread_index_in_simdgroup;
    const int sg = int(simdgroup_index_in_threadgroup);
    const int col = int(threadgroup_position_in_grid.y) * 4;
    threadgroup float part[TOPK + 1][ROWS][4];
    float acc[ROWS][4] = {};
    if (sg < TOPK) {
      for (int t = 0; t < ROWS; t++) {
        const int slot = t * TOPK + sg;
        const size_t base = size_t(indices[slot]) * D + col;
        rows_partial<float, Q, DBITS, DVPT, N, 4, false>(
            act + slot * N, dw, ds, db, dw, ds, db, base, lane, acc[t], acc[t]);
      }
    } else {
      rows_partial_multi<float, Q, SBITS, SDVPT, N, ROWS, false>(
          act + ROWS * TOPK * N, sdw, sds, sdb, sdw, sds, sdb,
          size_t(col), lane, acc, acc);
    }
    for (int t = 0; t < ROWS; t++) {
      // Routed outputs stay FP32 and are weighted here, rounded before the
      // sum like the stock product; the shared output rounds to the row
      // dtype. The barrier keeps product and sum from contracting.
      const float scale = sg < TOPK ? weights[t * TOPK + sg] : 1.0f;
      for (int r = 0; r < 4; r++) {
        const float v = simd_sum(acc[t][r]);
        if (lane == 0) {
          part[sg][t][r] = sg < TOPK ? v * scale : float(static_cast<T>(v));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && int(lane) < 4 * ROWS) {
      const int t = int(lane) / 4;
      const int r = int(lane) % 4;
      float total = 0.0f;
      for (int k = 0; k < TOPK; k++) {
        total += part[k][t][r];
      }
      total = total + part[TOPK][t][r];
      out[t * D + col + r] = static_cast<T>(total);
    }
"""


def _float_literal(value: float) -> str:
    return f"({float(value)!r}f)"


@lru_cache(maxsize=None)
def _route_kernel(normalize: bool):
    source = _ROUTE.replace("NORMALIZE", "true" if normalize else "false")
    return mx.fast.metal_kernel(
        name=f"rapid_v41_moe_route_{int(normalize)}",
        input_names=["scores", "bias", "params"],
        output_names=["weights", "indices"],
        source=source,
        header=_HEADER,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _up_kernel(limit: float):
    source = _UP.replace("LIMIT", _float_literal(limit))
    return mx.fast.metal_kernel(
        name="rapid_v41_moe_up_"
        + repr(float(limit)).replace(".", "p").replace("-", "m"),
        input_names=[
            "x",
            "indices",
            "gw",
            "gs",
            "gb",
            "uw",
            "us",
            "ub",
            "sgw",
            "sgs",
            "sgb",
            "suw",
            "sus",
            "sub",
        ],
        output_names=["act"],
        source=source,
        header=_HEADER,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _down_kernel():
    return mx.fast.metal_kernel(
        name="rapid_v41_moe_down",
        input_names=[
            "act",
            "indices",
            "weights",
            "dw",
            "ds",
            "db",
            "sdw",
            "sds",
            "sdb",
        ],
        output_names=["out"],
        source=_DOWN,
        header=_HEADER,
        ensure_row_contiguous=True,
    )


def _qmv_vpt(bits: int, k: int, n: int) -> int:
    """Values per lane of the stock single-row affine QMV for this shape."""
    # The fast kernel covers whole blocks and 8-row tiles and loads two packs
    # per lane above two bits; the general kernel loads one.
    pack = 8 if bits == 3 else 32 // bits
    fast = pack * (2 if bits > 2 else 1)
    return fast if k % (32 * fast) == 0 and n % 8 == 0 else pack


def _affine(module, bits: int, ndim: int) -> bool:
    scales = getattr(module, "scales", None)
    biases = getattr(module, "biases", None)
    return (
        getattr(module, "bits", None) == bits
        and getattr(module, "group_size", None) == 64
        and getattr(module, "mode", "affine") == "affine"
        and scales is not None
        and biases is not None
        and scales.dtype in (mx.bfloat16, mx.float16)
        and biases.dtype == scales.dtype
        and module.weight.ndim == scales.ndim == biases.ndim == ndim
        and tuple(scales.shape) == tuple(biases.shape)
    )


class Layout(NamedTuple):
    dim: int
    inter: int
    sbits: int
    gbits: int
    ubits: int
    dbits: int


def _packed_width(k: int, bits: int) -> int:
    """Number of uint32 columns; 3-bit stores eight values in three bytes."""
    return k * bits // 32


def _projection_shape(module, n: tuple[int, ...], k: int) -> bool:
    return tuple(module.weight.shape) == (*n, _packed_width(k, module.bits))


def _layout(moe) -> Layout | None:
    """Dimensions and per-projection bits when ``moe`` matches the fused kernels."""
    experts, shared, gate = moe.experts, moe.shared_experts, moe.gate
    gp, up, dp = experts.gate_proj, experts.up_proj, experts.down_proj
    w1, w2, w3 = shared.w1, shared.w2, shared.w3
    if (
        "bias" in gp
        or "bias" in up
        or "bias" in dp
        or hasattr(experts, "gate_up_proj")
        or not all(
            getattr(m, "bits", None) in (2, 3, 4) and _affine(m, m.bits, 3)
            for m in (gp, up, dp)
        )
    ):
        return None
    sbits = getattr(w1, "bits", None)
    if sbits not in (2, 4, 8) or not all(_affine(m, sbits, 2) for m in (w1, w2, w3)):
        return None
    if any("bias" in m for m in (w1, w2, w3)):
        return None
    experts_n, inter = map(int, gp.scales.shape[:2])
    dim = int(gp.scales.shape[2]) * 64
    q = gp.scales.dtype
    if not (
        all(m.scales.dtype == q for m in (up, dp, w1, w2, w3))
        and tuple(up.scales.shape) == tuple(gp.scales.shape)
        and tuple(dp.scales.shape) == (experts_n, dim, inter // 64)
        and tuple(w1.scales.shape) == tuple(w3.scales.shape) == (inter, dim // 64)
        and tuple(w2.scales.shape) == (dim, inter // 64)
        and _projection_shape(gp, (experts_n, inter), dim)
        and _projection_shape(up, (experts_n, inter), dim)
        and _projection_shape(dp, (experts_n, dim), inter)
        and _projection_shape(w1, (inter,), dim)
        and _projection_shape(w3, (inter,), dim)
        and _projection_shape(w2, (dim,), inter)
        and dim % 512 == 0
        # a 64- or 128-wide reduction takes the stock quad QMV kernel
        and inter % 64 == 0
        and inter > 128
        and gate.score_func == "sqrtsoftplus"
        and tuple(gate.bias.shape) == (experts_n,)
        and gate.bias.dtype == mx.float32
        # stock combines more than eight routes in strided partial sums
        and 1 <= gate.topk <= min(8, experts_n)
        and float(getattr(experts.activation, "limit", 0.0))
        == float(getattr(shared, "limit", 0.0))
        and experts_n <= 1024
        # multi-row router logits must come from the exact per-row GEMV
        and getattr(gate.weight, "dtype", None) == mx.bfloat16
        and tuple(gate.weight.shape) == (experts_n, dim)
        and dim < 16 * experts_n
    ):
        return None
    return Layout(dim, inter, sbits, gp.bits, up.bits, dp.bits)


def supports(x: mx.array) -> bool:
    """True when an MoE call on ``x`` is a decode-sized fused candidate."""
    if (
        not ENABLED
        or not custom_kernels_enabled()
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return False
    rows = 1
    for size in x.shape[:-1]:
        rows *= int(size)
    return 1 <= rows <= MAX_ROWS and x.dtype in (mx.bfloat16, mx.float16, mx.float32)


def _route(moe, scores: mx.array) -> tuple[mx.array, mx.array]:
    gate = moe.gate
    rows, experts_n = map(int, scores.shape)
    params = mx.array([gate.gate_temp, gate.route_scale], dtype=mx.float32)
    weights, indices = _route_kernel(bool(gate.norm_topk_prob and gate.topk > 1))(
        inputs=[scores, gate.bias, params],
        template=[("E", experts_n), ("TOPK", int(gate.topk))],
        grid=(experts_n, rows, 1),
        threadgroup=(experts_n, 1, 1),
        output_shapes=[(rows, gate.topk), (rows, gate.topk)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    return weights, indices


def _experts(
    moe, layout, xf: mx.array, weights: mx.array, indices: mx.array
) -> mx.array:
    dim, inter, sbits, gbits, ubits, dbits = layout
    rows = int(xf.shape[0])
    topk = int(moe.gate.topk)
    experts, shared = moe.experts, moe.shared_experts
    gp, up, dp = experts.gate_proj, experts.up_proj, experts.down_proj
    w1, w2, w3 = shared.w1, shared.w2, shared.w3
    flat = indices.reshape(-1)
    common = [("T", xf.dtype), ("Q", gp.scales.dtype), ("ROWS", rows), ("TOPK", topk)]
    (act,) = _up_kernel(float(getattr(experts.activation, "limit", 0.0)))(
        inputs=[
            xf,
            flat,
            gp.weight,
            gp.scales,
            gp.biases,
            up.weight,
            up.scales,
            up.biases,
            w1.weight,
            w1.scales,
            w1.biases,
            w3.weight,
            w3.scales,
            w3.biases,
        ],
        template=common
        + [
            ("K", dim),
            ("N", inter),
            ("SBITS", sbits),
            ("SVPT", _qmv_vpt(sbits, dim, inter)),
            ("GBITS", gbits),
            ("UBITS", ubits),
            ("GVPT", _qmv_vpt(gbits, dim, inter)),
            ("UVPT", _qmv_vpt(ubits, dim, inter)),
        ],
        grid=(32, inter // 4, rows * topk + 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows * (topk + 1), inter)],
        output_dtypes=[mx.float32],
    )
    (out,) = _down_kernel()(
        inputs=[
            act,
            flat,
            weights.reshape(-1),
            dp.weight,
            dp.scales,
            dp.biases,
            w2.weight,
            w2.scales,
            w2.biases,
        ],
        template=common
        + [
            ("N", inter),
            ("D", dim),
            ("SBITS", sbits),
            ("SDVPT", _qmv_vpt(sbits, inter, dim)),
            ("DBITS", dbits),
            ("DVPT", _qmv_vpt(dbits, inter, dim)),
        ],
        grid=(32, (topk + 1) * dim // 4, 1),
        threadgroup=(32, topk + 1, 1),
        output_shapes=[(rows, dim)],
        output_dtypes=[xf.dtype],
    )
    result: mx.array = out
    return result


def _fused(moe, layout, x: mx.array) -> mx.array:
    xf = x.reshape(-1, x.shape[-1])
    weights, indices = _route(moe, moe.gate.scores(xf))
    return _experts(moe, layout, xf, weights, indices).reshape(x.shape)


def _signature(moe) -> tuple[list, tuple]:
    """Arrays (by identity) and settings a qualification verdict depends on."""
    experts, shared, gate = moe.experts, moe.shared_experts, moe.gate
    modules = (
        experts.gate_proj,
        experts.up_proj,
        experts.down_proj,
        shared.w1,
        shared.w2,
        shared.w3,
    )
    # every kernel input array: a replaced one re-runs eligibility and the
    # self-check (None where an unquantized module has no such array)
    names = ("weight", "scales", "biases")
    arrays = [getattr(m, name, None) for m in modules for name in names]
    arrays += [getattr(gate, "weight", None), getattr(gate, "bias", None)]
    settings = (
        tuple("bias" in m for m in modules),
        tuple(
            (
                getattr(m, "bits", None),
                getattr(m, "group_size", None),
                getattr(m, "mode", "affine"),
            )
            for m in modules
        ),
        gate.score_func,
        gate.topk,
        bool(gate.norm_topk_prob),
        float(getattr(experts.activation, "limit", 0.0)),
        float(getattr(shared, "limit", 0.0)),
    )
    return arrays, settings


def _qualified(moe, stock, dtype) -> Layout | None:
    """This layer's layout once each fused row equals its single-row stock."""
    arrays, settings = _signature(moe)
    cached: tuple | None = getattr(moe, _CHECK_ATTR, None)
    verdicts: dict = {}
    if (
        cached is not None
        and cached[1] == settings
        and all(
            (ref is None and array is None) or (ref is not None and ref() is array)
            for ref, array in zip(cached[0], arrays, strict=True)
        )
    ):
        verdicts = cached[2]
    if dtype not in verdicts:
        layout = _layout(moe)
        ok = layout is not None
        if layout is not None:
            try:
                key = mx.random.key(41)
                probe = mx.random.normal((MAX_ROWS, layout[0]), key=key).astype(dtype)
                # Stock picks a different multi-row QMV, so the contract is
                # per row: batched decode matches single-stream decode. Every
                # row count is its own kernel specialization, so probe each.
                expected = [stock(probe[r : r + 1]) for r in range(MAX_ROWS)]
                for rows in range(1, MAX_ROWS + 1):
                    actual = _fused(moe, layout, probe[:rows])
                    for r in range(rows):
                        same = mx.array_equal(actual[r : r + 1], expected[r])
                        ok = ok and bool(same.item())
            except Exception:  # compile/launch failure on this runtime
                logger.warning("fused V4.1 MoE unavailable", exc_info=True)
                ok = False
            if not ok:
                logger.warning(
                    "fused V4.1 MoE disabled: stock path for %s", _layer_label(moe)
                )
        verdicts[dtype] = layout if ok else None
        # Plain attribute (not a parameter); the weak references re-qualify
        # replaced weights without keeping the old arrays alive.
        refs = tuple(None if a is None else weakref.ref(a) for a in arrays)
        object.__setattr__(moe, _CHECK_ATTR, (refs, settings, verdicts))
    verdict: Layout | None = verdicts[dtype]
    return verdict


def moe_forward(moe, x: mx.array, stock) -> mx.array | None:
    """Fused decode MoE output, or None when the stock path must run."""
    if not supports(x):
        return None
    layout = _qualified(moe, stock, x.dtype)
    if layout is None or int(x.shape[-1]) != layout[0]:
        return None
    if x.size > x.shape[-1] and not _router_rows_exact(moe.gate, x):
        return None
    try:
        return _fused(moe, layout, x)
    except Exception:  # graph-build failure after qualification: stay stock
        # (a device fault during lazy evaluation surfaces like any stock
        # kernel's; the self-check already ran these kernels on this layer)
        logger.warning(
            "fused V4.1 MoE failed: stock path for %s", _layer_label(moe), exc_info=True
        )
        getattr(moe, _CHECK_ATTR)[2][x.dtype] = None
        return None


def _router_rows_exact(gate, x: mx.array) -> bool:
    """True while several rows get single-row router logits (exact GEMV)."""
    probe = x.reshape(-1, x.shape[-1]).astype(mx.float32)
    return head_gemv._supports_weight(gate.weight, probe) and head_gemv._matches_stock(
        gate
    )


def _layer_label(moe, layer_id=None) -> str:
    """Identify a layer and its gate, up, down and shared quantization bits."""
    bits = [
        getattr(m, "bits", "dense")
        for m in (
            moe.experts.gate_proj,
            moe.experts.up_proj,
            moe.experts.down_proj,
            moe.shared_experts.w1,
        )
    ]
    index = getattr(moe, "layer_id", layer_id)
    return f"layer {index} ({bits[0]},{bits[1]},{bits[2]}|{bits[3]})"


def fused_layers(model, dtype) -> tuple[int, int]:
    """Count cached fused verdicts for dtype; pending layers count only in total.

    This snapshot does not qualify layers or run kernels. Read it after warmup
    to measure coverage, including fallbacks after the once-per-model log.
    """
    ffns = [
        ffn
        for layer in model.layers
        if (ffn := getattr(layer, "ffn", None)) is not None
    ]
    return (
        sum(
            getattr(ffn, _CHECK_ATTR, (None, None, {}))[2].get(dtype) is not None
            for ffn in ffns
        ),
        len(ffns),
    )


def log_coverage(model, dtype) -> None:
    """Report actual lazy qualification results once every layer has a verdict."""
    if getattr(model, "_rapid_moe_summary_logged", False):
        return
    verdicts: list[dict] = []
    fallback = []
    for i, layer in enumerate(model.layers):
        ffn = getattr(layer, "ffn", None)
        if ffn is None:
            continue
        verdict: dict = getattr(ffn, _CHECK_ATTR, (None, None, {}))[2]
        if dtype not in verdict:
            return
        verdicts.append(verdict)
        if verdict[dtype] is None:
            fallback.append(_layer_label(ffn, getattr(layer, "layer_id", i)))
    if not verdicts:
        return
    logger.info(
        "fused V4.1 MoE: %d/%d layers; fallback: %s",
        len(verdicts) - len(fallback),
        len(verdicts),
        ", ".join(fallback) or "none",
    )
    object.__setattr__(model, "_rapid_moe_summary_logged", True)
