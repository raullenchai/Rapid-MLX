# SPDX-License-Identifier: MIT
"""Selected affine-4 GeGLU experts for short Gemma 4 decode blocks.

Uses the MIT-licensed expert kernels; see QWEN35_AFFINE4_LICENSE.
The router and parallel dense MLP retain their composed calls. Set
RAPID_MLX_GEMMA4_AFFINE4=0 to disable enrollment and dispatch immediately.
"""

import functools
import logging
import sys
import threading
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .._env import env_falsey
from ..lattice.fusion import install_enabled

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()


def _families():
    """Inspect only resident model modules, preserving optional vision imports."""
    result = []
    for model, switch in (
        ("rapid_mlx.models.gemma4_vendored.language", "mlx_lm.models.switch_layers"),
        ("mlx_lm.models.gemma4_text", "mlx_lm.models.switch_layers"),
        ("mlx_vlm.models.gemma4.language", "mlx_vlm.models.switch_layers"),
    ):
        mod, sm = sys.modules.get(model), sys.modules.get(switch)
        names = (
            getattr(mod, "Experts", None),
            getattr(sm, "SwitchGLU", None),
            getattr(mod, "GeGLU", None),
            getattr(sm, "QuantizedSwitchLinear", None),
        )
        if all(isinstance(cls, type) for cls in names):
            result.append(names)
    return tuple(result)


def _compatible(block):
    family = next((f for f in _families() if type(block) is f[0]), None)
    if family is None:
        return False
    sm = getattr(block, "switch_glu", None)
    if (
        type(sm) is not family[1]
        or type(getattr(sm, "activation", None)) is not family[2]
    ):
        return False
    if any(getattr(m, "sharding_group", None) is not None for m in (block, sm)):
        return False
    combined = getattr(sm, "gate_up_proj", None)
    projections = [(getattr(sm, "down_proj", None), (128, 2816, 88))]
    if combined is not None:
        projections.append((combined, (128, 1408, 352)))
    else:
        projections.extend(
            (getattr(sm, name, None), (128, 704, 352))
            for name in ("gate_proj", "up_proj")
        )
    for mod, shape in projections:
        if (
            type(mod) is not family[3]
            or getattr(mod, "bits", None) != 4
            or getattr(mod, "group_size", None) != 64
            or getattr(mod, "mode", None) != "affine"
            or getattr(mod, "sharding_group", None) is not None
            or "bias" in mod
        ):
            return False
        weight, scales, biases = (
            getattr(mod, n, None) for n in ("weight", "scales", "biases")
        )
        if any(v is None for v in (weight, scales, biases)):
            return False
        if (
            weight.shape != shape
            or weight.dtype != mx.uint32
            or scales.dtype != mx.bfloat16
            or biases.dtype != mx.bfloat16
            or scales.shape != (*shape[:-1], shape[-1] // 8)
            or biases.shape != scales.shape
        ):
            return False
    return True


@functools.lru_cache(maxsize=1)
def kernels():
    root = Path(__file__).parent
    header = (root / "qwen35_affine4_dot.metal").read_text()
    gate = mx.fast.metal_kernel(
        name="rapid_gemma4_affine4_gateup",
        input_names=[
            "x",
            "wg_q",
            "g_scales",
            "g_biases",
            "wu_q",
            "u_scales",
            "u_biases",
            "inds",
            "gelutab",
        ],
        output_names=["y"],
        header=header,
        source=(root / "gemma4_affine4_gateup.metal").read_text(),
    )
    down = mx.fast.metal_kernel(
        name="rapid_gemma4_affine4_down",
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
    # Lookup preserves the activation dtype's rounding and compiled tanh semantics.
    table = nn.gelu_approx(mx.arange(65536, dtype=mx.uint16).view(mx.bfloat16))
    mx.eval(table)
    return gate, down, table


def forward(block, x, inds, scores):
    gate, down, table = kernels()
    sm = block.switch_glu
    combined = getattr(sm, "gate_up_proj", None)
    gp = combined if combined is not None else sm.gate_proj
    up = combined if combined is not None else sm.up_proj
    dp = sm.down_proj
    tokens = x.shape[1]
    act = gate(
        inputs=[
            x,
            gp.weight,
            gp.scales,
            gp.biases,
            up.weight,
            up.scales,
            up.biases,
            inds,
            table,
        ],
        template=[
            ("T", x.dtype),
            ("ROWS", 1),
            ("K", 2816),
            ("N", 704),
            ("GS", 64),
            ("TOPK", 8),
            ("COMBINED", int(combined is not None)),
        ],
        grid=(32, 704, 8 * tokens),
        threadgroup=(32, 2, 1),
        output_shapes=[(tokens, 8, 704)],
        output_dtypes=[x.dtype],
    )[0]
    # SHARED=0 never reads these dummy inputs. No dense-MLP arithmetic joins here.
    return down(
        inputs=[act, dp.weight, dp.scales, dp.biases, inds, scores, x, x],
        template=[
            ("T", x.dtype),
            ("ROWS", 1),
            ("I", 704),
            ("H", 2816),
            ("GS", 64),
            ("TOPK", 8),
            ("SHARED", 0),
        ],
        grid=(32, 2816, tokens),
        threadgroup=(32, 2, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def _eligible(block, x, inds, scores):
    return (
        getattr(block, "_rapid_gemma4_affine4", False)
        and not env_falsey("RAPID_MLX_GEMMA4_AFFINE4")
        and x.ndim == 3
        and x.shape[0] == 1
        and 1 <= x.shape[1] <= 9
        and x.shape[2] == 2816
        and x.dtype == mx.bfloat16
        and inds.shape == scores.shape == (1, x.shape[1], 8)
        and inds.dtype in (mx.uint32, mx.int32)
        and scores.dtype == mx.bfloat16
        and _compatible(block)
    )


def _patch_class(cls):
    if cls.__dict__.get("_rapid_gemma4_affine4_patched", False):
        return
    original = cls.__call__

    def call(self, x, top_k_indices, top_k_weights):
        if _eligible(self, x, top_k_indices, top_k_weights):
            return forward(self, x, top_k_indices, top_k_weights)
        return original(self, x, top_k_indices, top_k_weights)

    cls.__call__ = call
    cls._rapid_gemma4_affine4_original = original
    cls._rapid_gemma4_affine4_patched = True


def _probe(block):
    # No sampling RNG state is consumed during loading. Include the last expert
    # and non-unit per-expert scales; routing itself is deliberately unchanged.
    x = (
        mx.sin(mx.arange(2816, dtype=mx.float32) * 0.731)
        .reshape(1, 1, 2816)
        .astype(mx.bfloat16)
    )
    inds = mx.array([[[0, 17, 39, 58, 77, 92, 111, 127]]], mx.uint32)
    scores = mx.array([[[0.07, 0.11, 0.23, 0.04, 0.13, 0.31, 0.19, 0.09]]], mx.bfloat16)
    original = getattr(
        type(block), "_rapid_gemma4_affine4_original", type(block).__call__
    )
    expected = original(block, x, inds, scores).astype(mx.float32)
    actual = forward(block, x, inds, scores).astype(mx.float32)
    error = mx.sqrt(
        mx.mean(mx.square(expected - actual))
        / mx.maximum(mx.mean(mx.square(expected)), 1e-12)
    )
    mx.eval(error)
    return 0 <= error.item() <= 0.02


def install_gemma4_affine4(model):
    """Enroll only exact layouts that pass a deterministic 2% relative RMS probe."""
    if env_falsey("RAPID_MLX_GEMMA4_AFFINE4") or not mx.metal.is_available():
        return 0
    if not install_enabled("gemma4_affine4", mx, route="gemma4_affine4_install"):
        return 0
    named = getattr(model, "named_modules", None)
    if not callable(named):
        return 0
    enrolled = 0
    with _LOCK:
        for _, block in named():
            if getattr(block, "_rapid_gemma4_affine4", False) or not _compatible(block):
                continue
            try:
                if not _probe(block):
                    raise ValueError("relative RMS error exceeds 2%")
                _patch_class(type(block))
                object.__setattr__(block, "_rapid_gemma4_affine4", True)
                enrolled += 1
            except Exception:
                logger.warning(
                    "Gemma affine4 probe declined block; using stock path",
                    exc_info=True,
                )
    if enrolled:
        logger.info("[gemma4_affine4] enrolled %d expert blocks", enrolled)
    return enrolled
