# SPDX-License-Identifier: Apache-2.0
"""Numerical/ISA requirements; environment overrides cannot bypass them."""

from .profiles import TuningContext

QSA_ARCHITECTURES = {
    "qsa_stage1": frozenset({"applegpu_g14s", "applegpu_g15d"}),
    "qsa_indexed_splitk": frozenset({"applegpu_g15d"}),
}
QSA_MLX_VERSIONS = frozenset({"0.32.2"})


def requirement_reason(name: str, ctx: TuningContext) -> str | None:
    hw = ctx.hardware
    if name in QSA_ARCHITECTURES:
        if ctx.mlx not in QSA_MLX_VERSIONS:
            return f"unqualified MLX version {ctx.mlx}"
        if hw.architecture not in QSA_ARCHITECTURES[name]:
            return f"unqualified Metal architecture {hw.architecture}"
    elif name == "affine8_attention":
        if not (
            hw.gpu_generation == 15
            and hw.tier == "Ultra"
            and ctx.mlx == "0.32.3"
            and ctx.macos == "26.5.2"
        ):
            return "unqualified affine8 runtime"
    elif name == "lane_tensor_units":
        if hw.gpu_generation is None or hw.gpu_generation < 17:
            return "tensor units unavailable"
    return None
