# Metal reduction and precision structure adapted from mlx-vlm #2105.
# Copyright © 2025 Prince Canuma. Used under the MIT License.

"""Default-off fused Qwen4-Exp GDN operator for single-token decode."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cache
from threading import Lock
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

NUM_KEY_HEADS = 16
NUM_VALUE_HEADS = 48
KEY_HEAD_DIM = 128
VALUE_HEAD_DIM = 128
CONV_KERNEL = 4
KEY_DIM = NUM_KEY_HEADS * KEY_HEAD_DIM
VALUE_DIM = NUM_VALUE_HEADS * VALUE_HEAD_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
_THREADGROUP_Y_CANDIDATES = (32, 16, 8, 4)


@dataclass(frozen=True)
class FusedGdnAdmission:
    accepted: bool
    reason: str


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def _dtype(value: Any) -> Any:
    return getattr(value, "dtype", None)


def admit_qwen4_fused_gdn_decode(
    *,
    qkv: Any,
    z: Any,
    beta: Any,
    alpha: Any,
    conv_state: Any,
    recurrent_state: Any,
    conv_weight: Any,
    a_log: Any,
    dt_bias: Any,
    norm_weight: Any,
    mask: Any,
    cache_lengths: Any,
    record_rollback: bool,
    training: bool,
    sharded: bool,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    conv_kernel: int,
    gate_activation: str,
) -> FusedGdnAdmission:
    """Check the exact production geometry without evaluating MLX arrays."""
    if training:
        return FusedGdnAdmission(False, "training")
    if sharded:
        return FusedGdnAdmission(False, "distributed sharding")
    if record_rollback:
        return FusedGdnAdmission(False, "speculative rollback")
    if mask is not None:
        return FusedGdnAdmission(False, "masked decode")
    if cache_lengths is not None:
        return FusedGdnAdmission(False, "ragged cache lengths")
    if gate_activation != "sigmoid":
        return FusedGdnAdmission(False, f"output gate {gate_activation!r}")

    geometry = (
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        conv_kernel,
    )
    expected_geometry = (
        NUM_KEY_HEADS,
        NUM_VALUE_HEADS,
        KEY_HEAD_DIM,
        VALUE_HEAD_DIM,
        CONV_KERNEL,
    )
    if geometry != expected_geometry:
        return FusedGdnAdmission(False, f"unsupported geometry {geometry}")

    expected = {
        "qkv": (1, 1, CONV_DIM),
        "z": (1, 1, VALUE_DIM),
        "alpha": (1, 1, NUM_VALUE_HEADS),
        "beta": (1, 1, NUM_VALUE_HEADS),
        "conv_state": (1, CONV_KERNEL - 1, CONV_DIM),
        "recurrent_state": (
            1,
            NUM_VALUE_HEADS,
            VALUE_HEAD_DIM,
            KEY_HEAD_DIM,
        ),
        "conv_weight": (CONV_DIM, CONV_KERNEL, 1),
        "A_log": (NUM_VALUE_HEADS,),
        "dt_bias": (NUM_VALUE_HEADS,),
        "norm_weight": (VALUE_HEAD_DIM,),
    }
    values = {
        "qkv": qkv,
        "z": z,
        "alpha": alpha,
        "beta": beta,
        "conv_state": conv_state,
        "recurrent_state": recurrent_state,
        "conv_weight": conv_weight,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "norm_weight": norm_weight,
    }
    for name, expected_shape in expected.items():
        if _shape(values[name]) != expected_shape:
            return FusedGdnAdmission(
                False,
                f"{name} shape {_shape(values[name])}, expected {expected_shape}",
            )

    value_dtype = _dtype(qkv)
    if value_dtype != mx.bfloat16:
        return FusedGdnAdmission(False, f"unsupported activation dtype {value_dtype}")
    for name in (
        "z",
        "alpha",
        "beta",
        "conv_state",
        "conv_weight",
        "dt_bias",
        "norm_weight",
    ):
        if _dtype(values[name]) != value_dtype:
            return FusedGdnAdmission(False, f"{name} dtype {_dtype(values[name])}")
    if _dtype(recurrent_state) != mx.float32:
        return FusedGdnAdmission(False, "recurrent_state must be float32")
    if _dtype(a_log) not in (value_dtype, mx.float32):
        return FusedGdnAdmission(False, f"A_log dtype {_dtype(a_log)}")
    return FusedGdnAdmission(True, "eligible")


_HEADER = r"""
#include <metal_atomic>
template <typename U>
inline U mlx_sigmoid_precise(U x) {
  U e = static_cast<U>(metal::precise::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}

template <typename U>
inline U mlx_sigmoid_fast(U x) {
  U e = static_cast<U>(metal::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}

template <typename U>
inline U mlx_log1p_fast(U x) {
  float xf = float(x);
  float xp1 = 1.0f + xf;
  float out = xp1 == 1.0f ? xf : xf * (metal::log(xp1) / (xp1 - 1.0f));
  return static_cast<U>(out);
}

template <typename U>
inline U mlx_softplus_fast(U x) {
  if (metal::isnan(x))
    return metal::numeric_limits<U>::quiet_NaN();
  constexpr U inf = metal::numeric_limits<U>::infinity();
  U zero = static_cast<U>(0);
  U hi = metal::max(x, zero);
  U lo = metal::min(x, zero);
  return (lo == -inf || hi == inf)
      ? hi
      : (hi + mlx_log1p_fast(static_cast<U>(metal::exp(lo - hi))));
}

"""


_SOURCE = r"""
  const uint hv = threadgroup_position_in_grid.z;
  const uint hk = hv / RATIO;
  const uint lane = thread_position_in_threadgroup.x;
  const uint ty = thread_position_in_threadgroup.y;
  const uint tid = thread_index_in_threadgroup;

  constexpr int NT = 32 * TY;
  constexpr int NDK = DK / 32;
  constexpr int NDV = DV / TY;
  constexpr uint KD = (uint)(HK * DK);
  constexpr uint VD = (uint)(HV * DV);
  constexpr uint CD = 2u * KD + VD;

  threadgroup float sq[DK];
  threadgroup float sk[DK];
  threadgroup T sq_squared[DK];
  threadgroup T sk_squared[DK];
  threadgroup float sv[DV];
  threadgroup float sy[DV];
  threadgroup float shr[4];

  device const float* si = recurrent_state + (size_t)hv * DV * DK;
  device float* so = recurrent_state_out + (size_t)hv * DV * DK;
  float st[NDV][NDK];
  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i)
      st[j][i] = si[(size_t)dv * DK + NDK * lane + i];
  }

  for (uint idx = tid; idx < (uint)(2 * DK + DV); idx += NT) {
    uint part = idx / (uint)DK;
    uint d = idx - part * (uint)DK;
    uint c = part == 0u ? hk * DK + d
           : (part == 1u ? KD + hk * DK + d : 2u * KD + hv * DV + d);
    device const T* wc = conv_weight + (size_t)c * K;
    float acc = 0.0f;
    for (uint tap = 0; tap + 1 < (uint)K; ++tap)
      acc += float(conv_state[(size_t)tap * CD + c]) * float(wc[tap]);
    acc += float(qkv[c]) * float(wc[K - 1]);
    T xb = static_cast<T>(acc);
    T sig = mlx_sigmoid_fast(xb);
    T sl = xb * sig;
    if (part == 0u) sq[d] = float(sl);
    else if (part == 1u) sk[d] = float(sl);
    else sv[d] = float(sl);
    if (part == 2u || (hv % RATIO) == 0u) {
      for (uint tap = 0; tap + 2 < (uint)K; ++tap)
        conv_state_out[(size_t)tap * CD + c] =
            conv_state[(size_t)(tap + 1) * CD + c];
      conv_state_out[(size_t)(K - 2) * CD + c] = qkv[c];
    }
  }

  if (tid == 0u) {
    T av = alpha[hv] + dt_bias[hv];
    T sp = mlx_softplus_fast(av);
    shr[2] = metal::precise::exp(
        -metal::precise::exp(float(A_log[hv])) * float(sp));
    // Exhaustive bf16 sweep: mlx_sigmoid_precise<T> equals mx.sigmoid on
    // every finite bf16 input; the fast form differs on one (x ~ -6.85).
    shr[3] = float(mlx_sigmoid_precise(beta[hv]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint d = tid; d < (uint)DK; d += NT) {
    T qv = static_cast<T>(sq[d]);
    T kv = static_cast<T>(sk[d]);
    sq_squared[d] = static_cast<T>(qv * qv);
    sk_squared[d] = static_cast<T>(kv * kv);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup_index_in_threadgroup == 0u) {
    T pq = static_cast<T>(0), pk = static_cast<T>(0);
    uint base = 4u * lane;
    for (int i = 0; i < 4; ++i) {
      pq = static_cast<T>(sq_squared[base + i] + pq);
      pk = static_cast<T>(sk_squared[base + i] + pk);
    }
    pq = static_cast<T>(simd_sum(float(pq)));
    pk = static_cast<T>(simd_sum(float(pk)));
    if (lane == 0u) {
      T eps = static_cast<T>(1.0e-6f);
      T qdenom = pq + eps;
      T kdenom = pk + eps;
      shr[0] = float(static_cast<T>(metal::precise::rsqrt(qdenom)));
      shr[1] = float(static_cast<T>(metal::precise::rsqrt(kdenom)));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  T qscale = static_cast<T>(0.08838834764831845f);
  for (uint d = tid; d < (uint)DK; d += NT) {
    T q_normalized = static_cast<T>(static_cast<T>(sq[d]) * static_cast<T>(shr[0]));
    T k_normalized = static_cast<T>(static_cast<T>(sk[d]) * static_cast<T>(shr[1]));
    sq[d] = float(static_cast<T>(q_normalized * qscale));
    sk[d] = float(k_normalized);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    float kv = 0.0f;
    for (int i = 0; i < NDK; ++i) {
      uint s = NDK * lane + i;
      st[j][i] = st[j][i] * shr[2];
      kv += st[j][i] * sk[s];
    }
    kv = simd_sum(kv);
    float delta = (sv[dv] - kv) * shr[3];
    float out = 0.0f;
    for (int i = 0; i < NDK; ++i) {
      uint s = NDK * lane + i;
      st[j][i] = st[j][i] + sk[s] * delta;
      out += st[j][i] * sq[s];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0u)
      sy[dv] = float(static_cast<T>(out));
    for (int i = 0; i < NDK; ++i)
      so[(size_t)dv * DK + NDK * lane + i] = st[j][i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup_index_in_threadgroup == 0u) {
    float po = 0.0f;
    uint base = 4u * lane;
    for (int i = 0; i < 4; ++i) po += sy[base + i] * sy[base + i];
    po = simd_sum(po);
    if (lane == 0u)
      shr[0] = metal::precise::rsqrt(po / (float)DV + norm_eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = tid; d < (uint)DV; d += NT) {
    T normalized = static_cast<T>(sy[d] * shr[0]);
    normalized = norm_weight[d] * normalized;
    // Exhaustive bf16 sweep: the precise float32 sigmoid matches mx.sigmoid
    // on every finite bf16-valued gate; the fast form differs on ~1%.
    float x = float(normalized) * mlx_sigmoid_precise<float>(float(z[hv * DV + d]));
    output[hv * DV + d] = static_cast<T>(x);
  }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="rapid_qwen4_fused_gdn_decode",
        input_names=[
            "qkv",
            "z",
            "beta",
            "alpha",
            "conv_state",
            "conv_weight",
            "A_log",
            "dt_bias",
            "recurrent_state",
            "norm_weight",
            "norm_eps",
        ],
        output_names=["output", "conv_state_out", "recurrent_state_out"],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def qwen4_fused_gdn_decode(
    qkv,
    z,
    beta,
    alpha,
    conv_state,
    conv_weight,
    a_log,
    dt_bias,
    recurrent_state,
    norm_weight,
    norm_eps: float,
    *,
    threadgroup_y: int,
):
    """Run the fused graph after structural admission succeeds."""
    if threadgroup_y not in _THREADGROUP_Y_CANDIDATES:
        raise ValueError(
            f"unsupported threadgroup_y {threadgroup_y}; "
            f"expected one of {_THREADGROUP_Y_CANDIDATES}"
        )
    outputs = _kernel()(
        inputs=[
            qkv,
            z,
            beta,
            alpha,
            conv_state,
            conv_weight,
            a_log,
            dt_bias,
            recurrent_state,
            norm_weight,
            float(norm_eps),
        ],
        template=[
            ("T", qkv.dtype),
            ("HK", NUM_KEY_HEADS),
            ("HV", NUM_VALUE_HEADS),
            ("DK", KEY_HEAD_DIM),
            ("DV", VALUE_HEAD_DIM),
            ("K", CONV_KERNEL),
            ("TY", threadgroup_y),
            ("RATIO", NUM_VALUE_HEADS // NUM_KEY_HEADS),
        ],
        grid=(32, threadgroup_y, NUM_VALUE_HEADS),
        threadgroup=(32, threadgroup_y, 1),
        output_shapes=[
            (1, 1, VALUE_DIM),
            (1, CONV_KERNEL - 1, CONV_DIM),
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
        ],
        output_dtypes=[qkv.dtype, qkv.dtype, mx.float32],
    )
    return tuple(outputs)


def fused_gdn_runtime_supported() -> bool:
    """Report capability without constructing or launching a kernel."""
    return bool(
        hasattr(mx, "fast")
        and hasattr(mx.fast, "metal_kernel")
        and hasattr(mx, "metal")
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
    )


_PROBED_THREADGROUP_Y: int | None = None
_PROBE_COMPLETE = False
_PROBE_LOCK = Lock()


def probe_qwen4_fused_gdn_decode(dtype) -> int | None:
    """Compile candidates once and publish the supported geometry atomically."""
    global _PROBE_COMPLETE, _PROBED_THREADGROUP_Y
    if _PROBE_COMPLETE:
        return _PROBED_THREADGROUP_Y
    with _PROBE_LOCK:
        if _PROBE_COMPLETE:
            return _PROBED_THREADGROUP_Y
        if not fused_gdn_runtime_supported():
            _PROBE_COMPLETE = True
            return None

        qkv = mx.zeros((1, 1, CONV_DIM), dtype=dtype)
        z = mx.zeros((1, 1, VALUE_DIM), dtype=dtype)
        gates = mx.zeros((1, 1, NUM_VALUE_HEADS), dtype=dtype)
        conv_state = mx.zeros((1, CONV_KERNEL - 1, CONV_DIM), dtype=dtype)
        conv_weight = mx.zeros((CONV_DIM, CONV_KERNEL, 1), dtype=dtype)
        recurrent_state = mx.zeros(
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM), dtype=mx.float32
        )
        vector = mx.zeros((NUM_VALUE_HEADS,), dtype=dtype)
        A_log = mx.zeros((NUM_VALUE_HEADS,), dtype=mx.float32)
        norm_weight = mx.ones((VALUE_HEAD_DIM,), dtype=dtype)
        for threadgroup_y in _THREADGROUP_Y_CANDIDATES:
            try:
                outputs = qwen4_fused_gdn_decode(
                    qkv,
                    z,
                    gates,
                    gates,
                    conv_state,
                    conv_weight,
                    A_log,
                    vector,
                    recurrent_state,
                    norm_weight,
                    1.0e-6,
                    threadgroup_y=threadgroup_y,
                )
                mx.eval(*outputs)
                _PROBED_THREADGROUP_Y = threadgroup_y
                break
            except ValueError as exc:
                if "threads per threadgroup" in str(exc):
                    continue
                logger.info("Qwen4 fused GDN probe failed: %s", exc)
                break
            except RuntimeError as exc:
                logger.info(
                    "Qwen4 fused GDN threadgroup_y=%d is unavailable: %s",
                    threadgroup_y,
                    exc,
                )
                continue
        _PROBE_COMPLETE = True
        return _PROBED_THREADGROUP_Y


__all__ = [
    "FusedGdnAdmission",
    "admit_qwen4_fused_gdn_decode",
    "fused_gdn_runtime_supported",
    "probe_qwen4_fused_gdn_decode",
    "qwen4_fused_gdn_decode",
]
