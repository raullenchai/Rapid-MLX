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
    if (
        name
        in {
            "mtp_max_depth",
            "mtp_forward_costs_ms",
            "mtp_prefill_block_size",
            "mtp_gdn_fusion",
        }
        and not ctx.has_mtp_head
    ):
        return "MTP head unavailable"
    if name in QSA_ARCHITECTURES:
        if ctx.mlx not in QSA_MLX_VERSIONS:
            return f"unqualified MLX version {ctx.mlx}"
        if hw.architecture not in QSA_ARCHITECTURES[name]:
            return f"unqualified Metal architecture {hw.architecture}"
    elif name == "lane_tensor_units":
        if hw.gpu_generation is None or hw.gpu_generation < 17:
            return "tensor units unavailable"
    return None
