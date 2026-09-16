# SPDX-License-Identifier: Apache-2.0
"""Blocked-sequential Metal kernel for Gated DeltaNet (GDN) prompt prefill.

Qwen3.5/3.6/3.8 hybrid checkpoints spend a large share of long-prompt
prefill inside the GDN linear-attention scan. mlx-lm's stock kernel splits
each v-head into ``Dv/4``-slice threadgroups, so every threadgroup re-reads
the same k/q rows from device memory — roughly 32x redundant traffic
(~13 GB per 16k-token layer call). This kernel computes the *exact same
sequential recurrence* (identical FLOPs — no chunked/WY reformulation)
restructured for Apple GPUs:

- k/q/v are staged into threadgroup memory in ``TB``-token blocks with
  coalesced cooperative loads; each row is read from device exactly once
  per threadgroup.
- The recurrent state stays in registers: each thread owns a
  ``(dv, 16-wide d-segment)`` fragment, and the ``k·state`` / ``q·state``
  contractions reduce across the 8 segment-threads of a dv row via
  ``simd_shuffle_down`` — no threadgroup barriers in the hot loop.

Numerics: all accumulation is fp32 and the state stays fp32, matching the
stock kernel's accuracy contract (state rel-err ~5e-8 on random inputs).

Kernel design adapted from the oMLX project
(https://github.com/jundot/omlx, Apache-2.0) ``qwen35_prefill`` package,
which reports ~2x over the stock kernel at 16k tokens (14.9ms vs 29.7ms
per layer call on M3 Ultra).

``install()`` wraps ``mlx_lm.models.gated_delta.gated_delta_kernel`` — the
module-level dispatch every GDN model path funnels through on the GPU
route — so callers that imported ``gated_delta_update`` directly are still
covered. The wrapper only takes the fast path for shapes the kernel is
built for (see ``_eligible``); everything else falls through to the stock
kernel unchanged. Set ``RAPID_MLX_GDN_PREFILL=0`` to disable.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

# The register layout hardwires the key dimension: 8 threads per dv row,
# each owning a 16-wide d-segment (4x float4) => exactly 128. Qwen3.5/3.6/
# 3.8 GDN all use Dk=128. Anything else falls back to the stock kernel.
_REQUIRED_DK = 128
# Each threadgroup owns a 32-row dv block.
_DV_BLOCK = 32
# Below this many prompt tokens the launch overhead beats the win.
_MIN_TOKENS = 64

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_KERNEL_SRC = """
    constexpr int TB = 32;                             // time block
    constexpr int DB = 32;                             // dv rows per threadgroup
    const int tid = thread_position_in_threadgroup.x;  // 0..255
    const int blk = threadgroup_position_in_grid.x;    // Dv/DB block
    const int hv  = threadgroup_position_in_grid.y;
    const int b   = threadgroup_position_in_grid.z;
    const int hk  = hv / (Hv / Hk);
    const int dv0 = blk * DB;

    // thread -> (dv row, 16-wide d segment); 8 threads per dv row, all in
    // the same simdgroup (lane = (dv%4)*8 + seg).
    const int dv  = tid / 8;            // 0..31
    const int seg = tid % 8;            // 0..7
    const int d0  = seg * 16;

    threadgroup InT k_s[TB][Dk + 8];
    threadgroup InT q_s[TB][Dk + 8];
    threadgroup InT v_s[TB][DB + 8];
    threadgroup float g_s[TB];
    threadgroup float b_s[TB];

    const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* q_base = q + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;
    const size_t krow = (size_t)Hk * Dk;

    // state fragment in registers: [dv0+dv][d0..d0+16]
    float4 st[4];
    {
        const device float4* S_in = (const device float4*)(
            state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0);
        for (int i = 0; i < 4; ++i) st[i] = S_in[i];
    }

    device InT* y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;

    for (int t0 = 0; t0 < T; t0 += TB) {
        const int tt = min(TB, T - t0);
        // cooperative staging (coalesced): k/q rows, v slice, g/beta
        for (int p = tid; p < tt * Dk; p += 256) {
            const int r = p / Dk, d = p % Dk;
            k_s[r][d] = k_base[(size_t)(t0 + r) * krow + d];
            q_s[r][d] = q_base[(size_t)(t0 + r) * krow + d];
        }
        for (int p = tid; p < tt * DB; p += 256) {
            const int r = p / DB, d = p % DB;
            v_s[r][d] = v_base[(size_t)(t0 + r) * Hv * Dv + d];
        }
        for (int p = tid; p < tt; p += 256) {
            g_s[p] = g[((size_t)b * T + t0 + p) * Hv + hv];
            b_s[p] = beta[((size_t)b * T + t0 + p) * Hv + hv];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = 0; t < tt; ++t) {
            const float gt = g_s[t];
            const float bt = b_s[t];
            const threadgroup vec<InT,4>* k4 =
                (const threadgroup vec<InT,4>*)&k_s[t][d0];
            const threadgroup vec<InT,4>* q4 =
                (const threadgroup vec<InT,4>*)&q_s[t][d0];
            float4 kf[4];
            for (int i = 0; i < 4; ++i) kf[i] = float4(k4[i]);
            // kv_mem = (g*state) . k ; decay applied to state first
            float4 p4 = 0.0f;
            for (int i = 0; i < 4; ++i) {
                st[i] *= gt;
                p4 += st[i] * kf[i];
            }
            float part = p4.x + p4.y + p4.z + p4.w;
            // reduce across the 8 segment-threads of this dv row
            part += simd_shuffle_down(part, 4);
            part += simd_shuffle_down(part, 2);
            part += simd_shuffle_down(part, 1);
            const float kv_mem = simd_shuffle(part, (tid % 32) / 8 * 8);
            const float delta = ((float)v_s[t][dv] - kv_mem) * bt;

            float4 o4 = 0.0f;
            for (int i = 0; i < 4; ++i) {
                st[i] += kf[i] * delta;
                o4 += st[i] * float4(q4[i]);
            }
            float out = o4.x + o4.y + o4.z + o4.w;
            out += simd_shuffle_down(out, 4);
            out += simd_shuffle_down(out, 2);
            out += simd_shuffle_down(out, 1);
            if (seg == 0) {
                y_base[(size_t)(t0 + t) * Hv * Dv + dv] = (InT)out;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        device float4* S_out = (device float4*)(
            state_out + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0);
        for (int i = 0; i < 4; ++i) S_out[i] = st[i];
    }
"""

# fp32 inputs (Qwen3.6's mamba_ssm_dtype) blow the 32 KiB threadgroup-memory
# limit at TB=32 (40,192 B for the 128/128 layout); TB=16 fits (20,096 B).
_SUPPORTED_BLOCK_T = (16, 32)
_kernels: dict[int, object] = {}


def _block_t_for(dtype: mx.Dtype) -> int:
    return 16 if dtype == mx.float32 else 32


def _get_kernel(block_t: int):
    kernel = _kernels.get(block_t)
    if kernel is None:
        source = _KERNEL_SRC.replace(
            "constexpr int TB = 32;", f"constexpr int TB = {block_t};"
        )
        kernel = mx.fast.metal_kernel(
            name=f"rapid_gdn_blocked_seq_tb{block_t}",
            input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "state_out"],
            source=source,
            header=_HEADER,
        )
        _kernels[block_t] = kernel
    return kernel


def gated_delta_blocked_seq(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Blocked-sequential GDN prefill (exact recurrence).

    q,k: [B,T,Hk,Dk]; v: [B,T,Hv,Dv]; g,beta: [B,T,Hv] (cast to fp32);
    state: [B,Hv,Dv,Dk] fp32. Returns y [B,T,Hv,Dv] (q.dtype), fp32 state.
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[2:]
    # This function is the single validation authority for the kernel's
    # hardwired layout contract: every invariant the Metal source assumes
    # is checked here and violations raise instead of silently computing
    # garbage or touching out-of-bounds GPU memory. ``_eligible`` mirrors
    # these checks only to route ineligible shapes to the stock kernel.
    if Dk != _REQUIRED_DK:
        raise ValueError(
            f"gated_delta_blocked_seq requires Dk == {_REQUIRED_DK}, got {Dk}"
        )
    if Dv % _DV_BLOCK != 0:
        raise ValueError(
            f"gated_delta_blocked_seq requires Dv % {_DV_BLOCK} == 0, got {Dv}"
        )
    if k.shape != q.shape or k.dtype != q.dtype:
        raise ValueError(
            "gated_delta_blocked_seq requires k to match q's shape and "
            f"dtype, got k {k.dtype} {tuple(k.shape)} vs q {q.dtype} "
            f"{tuple(q.shape)}"
        )
    if v.dtype != q.dtype or tuple(v.shape[:2]) != (B, T):
        raise ValueError(
            "gated_delta_blocked_seq requires v of shape [B, T, Hv, Dv] "
            f"with q's dtype, got {v.dtype} {tuple(v.shape)}"
        )
    if Hk <= 0 or Hv < Hk or Hv % Hk != 0:
        raise ValueError(
            "gated_delta_blocked_seq requires Hv to be a positive multiple "
            f"of Hk, got Hk={Hk} Hv={Hv}"
        )
    if g.shape != (B, T, Hv) or beta.shape != (B, T, Hv):
        raise ValueError(
            "gated_delta_blocked_seq requires g and beta of shape "
            f"{(B, T, Hv)}, got g {tuple(g.shape)} beta {tuple(beta.shape)}"
        )
    in_dtype = q.dtype
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    elif state.dtype != mx.float32 or state.shape != (B, Hv, Dv, Dk):
        # The kernel's pointer arithmetic assumes exactly this fp32 layout;
        # anything else would read/write out of bounds on the GPU.
        raise ValueError(
            "gated_delta_blocked_seq requires an fp32 state of shape "
            f"{(B, Hv, Dv, Dk)}, got {state.dtype} {tuple(state.shape)}"
        )
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    kernel = _get_kernel(_block_t_for(in_dtype))
    y, state_out = kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", in_dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(256 * (Dv // _DV_BLOCK), Hv, B),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[in_dtype, mx.float32],
    )
    return y, state_out


def _eligible(q, k, v, g, beta, state, mask) -> bool:
    """Fast-path gate. Anything outside falls back to the stock kernel.

    Pins every layout invariant the Metal kernel hardwires — k strides/dtype
    mirror q's, beta is a scalar-per-head [B, T, Hv] tensor like g — so
    mismatched-but-stock-tolerated inputs can never be read with wrong
    strides or element types. ``state`` may be ``None`` (first prefill
    chunk): the fast path initializes a zero fp32 state exactly like the
    stock update path; a caller-provided state must already be fp32.
    """
    return (
        mask is None
        and g.ndim == 3  # scalar per-head gating (vectorized g is 4-D)
        and q.shape[1] >= _MIN_TOKENS
        and q.shape[-1] == _REQUIRED_DK
        and k.shape == q.shape  # kernel walks k with q's strides
        and k.dtype == q.dtype  # single InT template for q/k/v
        and v.dtype == q.dtype
        and v.shape[:2] == q.shape[:2]  # same [B, T]
        and v.shape[-1] % _DV_BLOCK == 0
        # g and beta must be exactly [B, T, Hv]: the kernel indexes both as
        # ((b*T + t)*Hv + hv), so any other layout reads out of bounds.
        and g.shape == (q.shape[0], q.shape[1], v.shape[2])
        and beta.shape == g.shape
        # GQA head mapping: the kernel computes hk = hv / (Hv / Hk), which
        # requires Hv to be a positive multiple of Hk.
        and q.shape[2] > 0
        and v.shape[2] >= q.shape[2]
        and v.shape[2] % q.shape[2] == 0
        # A provided state must be the exact [B, Hv, Dv, Dk] fp32 layout the
        # kernel's pointer arithmetic assumes — anything else risks
        # out-of-bounds GPU access, so it falls back to the stock kernel.
        and (
            state is None
            or (
                state.dtype == mx.float32
                and state.shape == (q.shape[0], v.shape[2], v.shape[3], q.shape[3])
            )
        )
    )


_installed = False
# The unwrapped mlx-lm kernel, captured at install() time. Kept for
# callers (tests) that need the stock implementation after the module
# attribute has been rebound to the wrapper.
_original_kernel = None


def install() -> bool:
    """Idempotently wrap mlx-lm's GDN GPU dispatch with the fast path.

    Returns True when the wrapper is (already) in place, False when
    disabled or unavailable. Safe to call from any lane: models that never
    route through ``gated_delta_kernel`` are unaffected.
    """
    global _installed, _original_kernel
    if _installed:
        return True
    if os.environ.get("RAPID_MLX_GDN_PREFILL", "1") == "0":
        logger.info("[gdn_prefill] disabled via RAPID_MLX_GDN_PREFILL=0")
        return False
    if not mx.metal.is_available():
        return False
    try:
        from mlx_lm.models import gated_delta as _gd
    except ImportError:
        return False

    # getattr, not attribute access: an older mlx-lm without this symbol
    # must degrade to "no fast path", never break scheduler import.
    stock_kernel = getattr(_gd, "gated_delta_kernel", None)
    if stock_kernel is None:
        logger.info(
            "[gdn_prefill] mlx-lm has no gated_delta_kernel — fast path unavailable"
        )
        return False
    if getattr(stock_kernel, "_rapid_gdn_wrapper", False):
        # Already wrapped (e.g. this module was reloaded while the wrapper
        # stayed bound). Never wrap the wrapper — that would make the
        # "stock" fallback recurse into the fast path.
        _original_kernel = stock_kernel._stock
        _installed = True
        return True

    # Eagerly compile-and-run both kernel variants BEFORE rebinding
    # anything: mlx is lazy, so a Metal rejection would otherwise surface at
    # mx.eval() inside a production request, outside any wrapper try/except.
    # The probe uses the Qwen3.5/3.8 flagship geometry (Hk=16, Hv=48,
    # Dv=128) for both dtypes; other Hk/Hv/Dv specializations of the same
    # (TB, InT) variant differ only in integer address constants — the
    # threadgroup-memory and register footprint depend solely on Dk (fixed
    # 128), DB (fixed 32), TB, and InT, all exercised here — so a
    # specialization cannot fail compilation when its probed variant
    # succeeded. If this GPU can't run the kernel, stay on stock entirely.
    try:
        for probe_dtype in (mx.bfloat16, mx.float32):
            pq = mx.zeros((1, _MIN_TOKENS, 16, _REQUIRED_DK), dtype=probe_dtype)
            pv = mx.zeros((1, _MIN_TOKENS, 48, 128), dtype=probe_dtype)
            pg = mx.zeros((1, _MIN_TOKENS, 48), dtype=mx.float32)
            py, pst = gated_delta_blocked_seq(pq, pq, pv, pg, pg, None)
            mx.eval(py, pst)
    except Exception:  # noqa: BLE001 — any probe failure ⇒ stay on stock
        logger.exception(
            "[gdn_prefill] kernel validation probe failed — staying on the stock kernel"
        )
        return False

    # Kernel JIT happens lazily on the first eligible call. If Metal ever
    # rejects the source (exotic GPU, future toolchain change), the failure
    # must degrade to the stock kernel — permanently, so a broken fast path
    # can never crash more than zero requests.
    fast_path_dead = False

    def gated_delta_kernel_fast(q, k, v, g, beta, state, mask=None):
        nonlocal fast_path_dead
        if not fast_path_dead and _eligible(q, k, v, g, beta, state, mask):
            try:
                return gated_delta_blocked_seq(q, k, v, g, beta, state)
            except Exception:  # noqa: BLE001 — any fast-path failure ⇒ stock
                fast_path_dead = True
                logger.exception(
                    "[gdn_prefill] fast path failed — permanently falling "
                    "back to the stock kernel"
                )
        return stock_kernel(q, k, v, g, beta, state, mask)

    # Tag the wrapper and hang the unwrapped kernel off it so any caller —
    # including a reloaded copy of this module or a test comparing numerics
    # — can always recover the true stock implementation.
    gated_delta_kernel_fast._rapid_gdn_wrapper = True
    gated_delta_kernel_fast._stock = stock_kernel

    _original_kernel = stock_kernel
    _gd.gated_delta_kernel = gated_delta_kernel_fast
    _installed = True
    logger.info(
        "[gdn_prefill] blocked-seq GDN prefill kernel installed "
        "(prompt>=%d tokens, Dk=%d)",
        _MIN_TOKENS,
        _REQUIRED_DK,
    )
    return True
