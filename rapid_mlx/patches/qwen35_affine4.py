# SPDX-License-Identifier: MIT
"""Fused affine-4 selected experts and parallel routing for short Qwen decode blocks.

Adapted from MIT-licensed code, Copyright (c) 2026 David Dalcu.
See QWEN35_AFFINE4_LICENSE for the complete copyright and permission notice.
Expert accumulation differs from the composed path; see qualification evidence.
Enabled by default; set RAPID_MLX_QWEN35_AFFINE4=0 to disable immediately.
"""

import functools
import logging
import os
import sys
import threading
from pathlib import Path

import mlx.core as mx

from .._env import env_falsey
from ..lattice import QWEN35_AFFINE4, ModelProfile, enabled, resolve, runtime_context
from ..lattice.fusion import install_enabled

logger = logging.getLogger(__name__)
_GATE = None
_DOWN = None
_SIG = None
_ROUTER256 = None
_GATE_ROWS = 1
_DOWN_ROWS = 1
_LOCK = threading.RLock()


def _singleton_bank_views_enabled():
    """Request measured policy only within the already-qualified expert path."""
    return _bank_views_policy(os.environ.get("RAPID_MLX_QWEN35_AFFINE4_BANK_VIEWS"))


@functools.lru_cache(maxsize=4)
def _bank_views_policy(_override):
    # Called per MoE layer per token: resolve once per override value so the
    # hot path does not re-read device info or rescan the measured table.
    return enabled(
        "qwen35_affine4_bank_views",
        mx.device_info(),
        family="qwen35_moe",
        geometry=QWEN35_AFFINE4,
        # This binding is identical in plain decode and MTP singleton calls.
        mode="plain",
        width=1,
    )


def _bank_bindings(projection):
    """Bind a zero-offset prefix while retaining the complete allocation.

    Metal addresses experts using explicit kernel geometry. MLX charges a
    command buffer for each input's logical data span; binding every expert
    unnecessarily forces commits. A prefix view shares the complete backing
    buffer and its lifetime, including when a pending graph outlives the model.
    Make the entire bank contiguous BEFORE slicing: contiguating a sliced
    noncontiguous bank could allocate storage for only the first expert.
    """
    full = (projection.weight, projection.scales, projection.biases)
    cached = getattr(projection, "_rapid_affine4_bank_bindings", None)
    if cached is not None and all(a is b for a, b in zip(full, cached[0])):
        return cached[1]
    if not all(isinstance(a, mx.array) and a.ndim == 3 for a in full):
        return full
    views = tuple(mx.contiguous(a)[:1] for a in full)
    # Keep runtime-only bindings out of the parameter tree. A weight rebind
    # invalidates this entry; pending graphs retain their own old allocations.
    object.__setattr__(projection, "_rapid_affine4_bank_bindings", (full, views))
    return views


_ROUTER256_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint token = thread_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    threadgroup float probs[256];
    threadgroup float partial[8];
    threadgroup float gate_partial[8];
    threadgroup float normalizer[1];
    threadgroup float chosen[8];
    threadgroup uint chosen_i[8];
    float high = simd_max(float(logits[token * 256 + tid]));
    if (lane == 0) partial[sg] = high;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        high = simd_max(lane < 8 ? partial[lane] : -INFINITY);
        if (lane == 0) normalizer[0] = high;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float p = metal::precise::exp(float(logits[token * 256 + tid]) - normalizer[0]);
    float total = simd_sum(p);
    if (lane == 0) partial[sg] = total;
    float dot = 0.0f;
    for (uint j = 0; j < 8; ++j) {
        uint k = tid * 8 + j;
        dot += float(x[token * 2048 + k]) * shared_w[k];
    }
    dot = simd_sum(dot);
    if (lane == 0) gate_partial[sg] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        total = simd_sum(lane < 8 ? partial[lane] : 0.0f);
        dot = simd_sum(lane < 8 ? gate_partial[lane] : 0.0f);
        if (lane == 0) { normalizer[0] = total; glog[token] = T(dot); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    probs[tid] = float(T(p / normalizer[0]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (uint route = 0; route < 8; ++route) {
            float best = -INFINITY;
            // Keep the fallback index in bounds even for a non-finite row.
            uint best_i = 255;
            for (uint j = 0; j < 8; ++j) {
                uint k = lane + j * 32;
                float value = probs[k];
                if (value >= best) { best = value; best_i = k; }
            }
            float group_best = simd_max(best);
            uint winner = simd_max(best == group_best ? best_i : 0u);
            if (lane == 0) {
                chosen[route] = group_best; chosen_i[route] = winner;
                probs[winner] = -INFINITY;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            T norm = T(0);
            for (int j = 7; j >= 0; --j) norm += T(chosen[j]);
            for (uint j = 0; j < 8; ++j) {
                inds[token * 8 + 7-j] = chosen_i[j]; scores[token * 8 + 7-j] = T(chosen[j]) / norm;
            }
        }
    }
"""


def fused_route(block, x, logits):
    global _ROUTER256
    with _LOCK:
        if _ROUTER256 is None:
            _ROUTER256 = mx.fast.metal_kernel(
                name="rapid_affine4_route_shared",
                input_names=["x", "logits", "shared_w"],
                output_names=["inds", "scores", "glog"],
                source=_ROUTER256_SOURCE,
            )
    return _ROUTER256(
        inputs=[x, logits, block._rapid_affine4_shared_weight],
        template=[("T", x.dtype)],
        grid=(256, x.shape[1], 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, x.shape[1], 8), (1, x.shape[1], 8), (1, x.shape[1], 1)],
        output_dtypes=[mx.uint32, x.dtype, x.dtype],
    )


def kernels():
    if _SIG is None:
        with _LOCK:
            _initialize_kernels()
    return _GATE, _DOWN, _SIG


def _initialize_kernels():
    global _GATE, _DOWN, _SIG, _GATE_ROWS, _DOWN_ROWS
    if _SIG is None:
        root = Path(__file__).parent
        header = (root / "qwen35_affine4_dot.metal").read_text()
        gate = mx.fast.metal_kernel(
            name="rapid_affine4_gateup",
            input_names=[
                "x",
                "wg_q",
                "g_scales",
                "g_biases",
                "wu_q",
                "u_scales",
                "u_biases",
                "inds",
                "sigtab",
            ],
            output_names=["y"],
            header=header,
            source=(root / "qwen35_affine4_gateup.metal").read_text(),
        )
        down = mx.fast.metal_kernel(
            name="rapid_affine4_down",
            input_names=[
                "act",
                "wd_q",
                "d_scales",
                "d_biases",
                "inds",
                "scores",
                "sdown",
                "glog",
            ],
            output_names=["y"],
            header=header,
            source=(root / "qwen35_affine4_down.metal").read_text(),
        )
        sig = mx.sigmoid(mx.arange(65536, dtype=mx.uint16).view(mx.bfloat16))
        mx.eval(sig)
        # Each projection keeps its tile across singleton and verification.
        profile = ModelProfile(
            family="qwen35_moe",
            hidden_size=2048,
            intermediate_size=512,
            num_experts=256,
            top_k=8,
            bits=4,
            group_size=64,
        )
        _GATE_ROWS, _DOWN_ROWS = resolve(
            "qwen35_affine4_rows", runtime_context(mx.device_info(), model=profile)
        ).value
        _GATE, _DOWN, _SIG = gate, down, sig


def forward(block, x):
    gate, down, sigtab = kernels()
    sm = block.switch_mlp
    gu = sm.gate_up_proj
    dp = sm.down_proj
    tokens = x.shape[1]
    gate_rows, down_rows = _GATE_ROWS, _DOWN_ROWS
    if tokens == 1 and _singleton_bank_views_enabled():
        gw, gs, gb = _bank_bindings(gu)
        dw, ds, db = _bank_bindings(dp)
    else:
        gw, gs, gb = gu.weight, gu.scales, gu.biases
        dw, ds, db = dp.weight, dp.scales, dp.biases
    logits = block.gate(x)
    inds, scores, glog = fused_route(block, x, logits)
    shared = block.shared_expert(x)
    act = gate(
        inputs=[
            x,
            gw,
            gs,
            gb,
            gw,
            gs,
            gb,
            inds,
            sigtab,
        ],
        template=[
            ("T", x.dtype),
            ("ROWS", gate_rows),
            ("K", 2048),
            ("N", 512),
            ("GS", 64),
            ("TOPK", 8),
        ],
        grid=(32, 512 // gate_rows, 8 * tokens),
        threadgroup=(32, 2, 1),
        output_shapes=[(tokens, 8, 512)],
        output_dtypes=[x.dtype],
    )[0]
    out = down(
        inputs=[
            act,
            dw,
            ds,
            db,
            inds,
            scores,
            shared,
            glog,
        ],
        template=[
            ("T", x.dtype),
            ("ROWS", down_rows),
            ("I", 512),
            ("H", 2048),
            ("GS", 64),
            ("TOPK", 8),
            ("SHARED", 1),
        ],
        grid=(32, 2048 // down_rows, tokens),
        threadgroup=(32, 2, 1),
        output_shapes=[(1, tokens, 2048)],
        output_dtypes=[x.dtype],
    )[0]
    return out


def _block_classes():
    families = (
        ("mlx_lm.models.qwen3_next", "Qwen3NextSparseMoeBlock"),
        ("mlx_vlm.models.qwen3_5_moe.language", "Qwen3_5MoeSparseMoeBlock"),
    )
    return tuple(
        cls
        for module, name in families
        if (cls := getattr(sys.modules.get(module), name, None)) is not None
    )


def _compatible(block):
    """Check every layout/semantic assumption before exposing raw bank pointers."""
    if type(block) not in _block_classes():
        return False
    # The VLM class always renormalizes; the LM class exposes an explicit flag.
    if hasattr(block, "norm_topk_prob"):
        if block.norm_topk_prob is not True:
            return False
    elif type(block) is not getattr(
        sys.modules.get("mlx_vlm.models.qwen3_5_moe.language"),
        "Qwen3_5MoeSparseMoeBlock",
        None,
    ):
        return False
    if (
        getattr(block, "num_experts", 0) != 256
        or getattr(block, "top_k", 0) != 8
        or getattr(block, "sharding_group", None) is not None
        or not getattr(block, "_rapid_qwen35_fused_router", False)
    ):
        return False
    sm = getattr(block, "switch_mlp", None)
    gu = getattr(sm, "gate_up_proj", None)
    dp = getattr(sm, "down_proj", None)
    shared = getattr(block, "shared_expert_gate", None)
    shared_bits = getattr(shared, "bits", 0)
    if shared_bits not in (4, 8):
        return False
    for mod, shape, bits in (
        (gu, (256, 1024, 256), 4),
        (dp, (256, 2048, 64), 4),
        (shared, (1, 2048 * shared_bits // 32), shared_bits),
    ):
        if (
            mod is None
            or getattr(mod, "bits", None) != bits
            or getattr(mod, "group_size", None) != 64
            or getattr(mod, "mode", None) != "affine"
            or "bias" in mod
        ):
            return False
        weight = getattr(mod, "weight", None)
        scales = getattr(mod, "scales", None)
        biases = getattr(mod, "biases", None)
        if any(value is None for value in (weight, scales, biases)):
            return False
        if (
            weight.shape != shape
            or weight.dtype != mx.uint32
            or scales.dtype != mx.bfloat16
            or biases.dtype != mx.bfloat16
            or scales.shape != (*shape[:-1], shape[-1] * 32 // bits // 64)
            or biases.shape != scales.shape
        ):
            return False
    return True


def _patch_class(cls):
    if cls.__dict__.get("_rapid_affine4_patched", False):
        return
    original = cls.__call__

    def call(self, x):
        if (
            getattr(self, "_rapid_affine4", False)
            and not env_falsey("RAPID_MLX_QWEN35_AFFINE4")
            and x.ndim == 3
            and x.shape[0] == 1
            and 1 <= x.shape[1] <= 9
            and x.shape[2] == 2048
            and x.dtype == mx.bfloat16
        ):
            return forward(self, x)
        return original(self, x)

    cls.__call__ = call
    cls._rapid_affine4_patched = True
    cls._rapid_affine4_original = original


def install_qwen35_affine4(model):
    """Enroll compatible short decode blocks; larger batches/widths fall back.

    Qualified singleton bindings may use allocation-backed prefix views.
    No pointers escape their MLX arrays, and no class closure retains models,
    weights or request state. Wider verification retains full-bank bindings.
    """
    if env_falsey("RAPID_MLX_QWEN35_AFFINE4") or not mx.metal.is_available():
        return 0
    if not install_enabled("qwen35_affine4", mx, route="qwen35_affine4_install"):
        return 0
    named = getattr(model, "named_modules", None)
    if not callable(named):
        return 0
    blocks = [b for _, b in named() if _compatible(b)]
    enrolled = 0
    with _LOCK:
        for block in blocks:
            if getattr(block, "_rapid_affine4", False):
                continue
            try:
                projection = block.shared_expert_gate
                weight = mx.dequantize(
                    projection.weight,
                    projection.scales.astype(mx.float32),
                    projection.biases.astype(mx.float32),
                    group_size=64,
                    bits=projection.bits,
                )
                mx.eval(weight)
                object.__setattr__(block, "_rapid_affine4_shared_weight", weight)
                cls = type(block)
                original = getattr(cls, "_rapid_affine4_original", cls.__call__)
                # Deterministic probe: loading must not advance sampling RNG.
                x = mx.sin(mx.arange(2048, dtype=mx.float32) * 0.731)
                x = x.reshape(1, 1, 2048).astype(mx.bfloat16)
                expected = original(block, x).astype(mx.float32)
                actual = forward(block, x).astype(mx.float32)
                error = mx.sqrt(
                    mx.mean(mx.square(expected - actual))
                    / mx.maximum(mx.mean(mx.square(expected)), 1e-12)
                )
                mx.eval(error)
                if not 0 <= error.item() <= 0.02:
                    raise ValueError("relative RMS error exceeds 2%")
                _patch_class(cls)
                object.__setattr__(block, "_rapid_affine4", True)
                enrolled += 1
            except Exception:
                if hasattr(block, "_rapid_affine4_shared_weight"):
                    object.__delattr__(block, "_rapid_affine4_shared_weight")
                logger.warning(
                    "Affine4 probe declined block; using stock path", exc_info=True
                )
    if enrolled:
        logger.info("[qwen35_affine4] enrolled %d MoE blocks", enrolled)
    return enrolled
