# SPDX-License-Identifier: MIT
"""Selected affine-4 SwiGLU experts for LFM2-MoE singleton decode.

Uses the MIT-licensed expert kernels; see QWEN35_AFFINE4_LICENSE.
Enrollment requires a deterministic 2% relative-RMS probe. Disable enrollment
and dispatch with RAPID_MLX_LFM2MOE_AFFINE4=0.
"""

import logging
from functools import cache
from pathlib import Path

from .._env import env_falsey

logger = logging.getLogger(__name__)


@cache
def _runtime():
    import mlx.core as mx
    from mlx_lm.models.lfm2_moe import Lfm2MoeSparseMoeBlock
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwiGLU, SwitchGLU

    return mx, Lfm2MoeSparseMoeBlock, SwitchGLU, SwiGLU, QuantizedSwitchLinear


def compatible(block, *, installed=False):
    """Validate all bank strides against this block's configured dimensions.

    The upstream block retains E/top-k, but not its ModelArgs. Hidden and
    intermediate dimensions therefore come from its router and down projection;
    every expert bank must agree. Both separate and concatenated gate/up qualify.
    """
    if getattr(block, "switch_mlp", None) is None:
        return False
    mx, base, switch, activation, quantized = _runtime()
    cls = type(block)
    if installed:
        # Only the exact model-local subclass created by the router installer.
        from .moe_decode_fusion import _decode_type

        if cls is not _decode_type(base, base.__call__):
            return False
    elif cls is not base:
        return False
    sm = getattr(block, "switch_mlp", None)
    if (
        type(sm) is not switch
        or type(getattr(sm, "activation", None)) is not activation
    ):
        return False
    if any(getattr(m, "sharding_group", None) is not None for m in (block, sm)):
        return False
    router = getattr(block, "gate", None)
    if getattr(router, "sharding_group", None) is not None:
        return False
    rw = getattr(router, "weight", None)
    dp = getattr(sm, "down_proj", None)
    if rw is None or rw.ndim != 2 or type(dp) is not quantized:
        return False
    scales = getattr(dp, "scales", None)
    if scales is None or scales.ndim != 3:
        return False
    e, h = rw.shape
    i = scales.shape[-1] * 64
    k = block.top_k
    if (
        e != block.num_experts
        or not 32 <= e <= 256
        or e % 32
        or not 1 <= k <= min(8, e)
        or h <= 0
        or h % 128
        or i <= 0
        or i % 64
    ):
        return False
    combined = getattr(sm, "gate_up_proj", None)
    projections = [(dp, h, i)]
    if combined is not None:
        projections.append((combined, 2 * i, h))
    else:
        projections.extend(
            (getattr(sm, n, None), i, h) for n in ("gate_proj", "up_proj")
        )
    for mod, out_dim, in_dim in projections:
        if (
            type(mod) is not quantized
            or getattr(mod, "bits", None) != 4
            or getattr(mod, "group_size", None) != 64
            or getattr(mod, "mode", None) != "affine"
            or getattr(mod, "sharding_group", None) is not None
            or "bias" in mod
        ):
            return False
        w, s, b = (getattr(mod, name, None) for name in ("weight", "scales", "biases"))
        if any(v is None for v in (w, s, b)):
            return False
        if (
            w.shape != (e, out_dim, in_dim // 8)
            or w.dtype != mx.uint32
            or s.shape != (e, out_dim, in_dim // 64)
            or s.dtype != mx.bfloat16
            or b.shape != s.shape
            or b.dtype != mx.bfloat16
        ):
            return False
    return True


@cache
def kernels():
    mx = _runtime()[0]
    root = Path(__file__).parent
    header = (root / "qwen35_affine4_dot.metal").read_text()
    gate = mx.fast.metal_kernel(
        name="rapid_lfm2moe_affine4_gateup",
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
        source=(root / "lfm2moe_affine4_gateup.metal").read_text(),
    )
    down = mx.fast.metal_kernel(
        name="rapid_lfm2moe_affine4_down",
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
    table = mx.sigmoid(mx.arange(65536, dtype=mx.uint16).view(mx.bfloat16))
    mx.eval(table)
    return gate, down, table


def forward(block, x, inds, scores):
    gate, down, table = kernels()
    sm = block.switch_mlp
    combined = getattr(sm, "gate_up_proj", None)
    gp = combined if combined is not None else sm.gate_proj
    up = combined if combined is not None else sm.up_proj
    dp = sm.down_proj
    h, i, k = x.shape[-1], dp.scales.shape[-1] * 64, block.top_k
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
            ("K", h),
            ("N", i),
            ("GS", 64),
            ("TOPK", k),
            ("COMBINED", int(combined is not None)),
        ],
        grid=(32, i, k),
        threadgroup=(32, 2, 1),
        output_shapes=[(1, k, i)],
        output_dtypes=[x.dtype],
    )[0]
    # SHARED=0 never reads the final two inputs.
    return down(
        inputs=[act, dp.weight, dp.scales, dp.biases, inds, scores, x, x],
        template=[
            ("T", x.dtype),
            ("ROWS", 1),
            ("I", i),
            ("H", h),
            ("GS", 64),
            ("TOPK", k),
            ("SHARED", 0),
        ],
        grid=(32, h, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def eligible(block, x):
    # Called only after the router's Metal/device/singleton checks succeed.
    return (
        getattr(block, "_rapid_lfm2moe_affine4", False)
        and not env_falsey("RAPID_MLX_LFM2MOE_AFFINE4")
        and x.ndim == 3
        and x.shape[:2] == (1, 1)
        and x.dtype == _runtime()[0].bfloat16
        and x.shape[-1] == block.gate.weight.shape[-1]
        and compatible(block, installed=True)
    )


def _probe(block):
    mx = _runtime()[0]
    h, k, e = block.gate.weight.shape[-1], block.top_k, block.num_experts
    x = (
        mx.sin(mx.arange(h, dtype=mx.float32) * 0.731)
        .reshape(1, 1, h)
        .astype(mx.bfloat16)
    )
    # Include both ends of the bank and non-unit route weights, without RNG use.
    inds = mx.array(
        [[[e - 1 if k == 1 else j * (e - 1) // (k - 1) for j in range(k)]]], mx.uint32
    )
    scores = mx.arange(1, k + 1, dtype=mx.float32)
    scores = (scores / scores.sum()).reshape(1, 1, k).astype(mx.bfloat16)
    expected = (
        (block.switch_mlp(x, inds) * scores[..., None]).sum(-2).astype(mx.float32)
    )
    actual = forward(block, x, inds, scores).astype(mx.float32)
    error = mx.sqrt(
        mx.mean(mx.square(expected - actual))
        / mx.maximum(mx.mean(mx.square(expected)), 1e-12)
    )
    mx.eval(error)
    return 0 <= error.item() <= 0.02


def enroll(block):
    if env_falsey("RAPID_MLX_LFM2MOE_AFFINE4") or not compatible(block):
        return False
    try:
        if not _probe(block):
            raise ValueError("relative RMS error exceeds 2%")
    except Exception:
        logger.warning(
            "LFM2-MoE affine4 probe declined block; keeping composed experts",
            exc_info=True,
        )
        return False
    object.__setattr__(block, "_rapid_lfm2moe_affine4", True)
    return True
