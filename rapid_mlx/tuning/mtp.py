# SPDX-License-Identifier: Apache-2.0
"""Depth policy at the installed single-request MTP capability boundary."""

from dataclasses import replace
from typing import cast

from . import model_profile, resolve, runtime_context


def _expert_quantization(model):
    """Uniform installed expert format; model args omit the quantization block."""
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return None
    try:
        formats = {
            (
                getattr(module, "bits", None),
                getattr(module, "group_size", None),
                getattr(module, "mode", None),
            )
            for name, module in named_modules()
            # Expert activation modules have no weight format to qualify.
            if ".switch_mlp." in name and hasattr(module, "weight")
        }
    except Exception:  # noqa: BLE001 -- unavailable facts cannot qualify a table row
        return None
    if len(formats) == 1:
        bits, group_size, mode = formats.pop()
        if type(bits) is int and type(group_size) is int and mode == "affine":
            return bits, group_size, mode
    return None


def resolve_max_depth(model, requested: int, *, explicit: bool) -> int:
    """Keep configured depths unless a qualified policy or override selects one.

    Explicit JSON/legacy depths win. A default decision preserves catalog and
    force-opt-in depths as well as the ordinary K=1. Checkpoint limits remain
    enforced by the caller after this policy, before calibration/generation.
    """
    if explicit:
        return requested
    has_head = (
        all(
            callable(getattr(model, name, None))
            for name in ("mtp_forward", "make_mtp_cache")
        )
        and getattr(model, "mtp", None) is not None
    )
    if not has_head:
        return requested

    import mlx.core as mx

    args = getattr(model, "args", None)
    config = vars(args) if hasattr(args, "__dict__") else {}
    profile = model_profile(config)
    if profile.num_experts:
        # Configuration declarations alone must not qualify mixed/unknown weights.
        profile = replace(profile, bits=None, group_size=None, quant_mode="unknown")
        quant = _expert_quantization(model)
        if quant is not None:
            profile = replace(
                profile, bits=quant[0], group_size=quant[1], quant_mode=quant[2]
            )
    ctx = runtime_context(
        mx.device_info(), model=profile, mode="mtp", has_mtp_head=has_head
    )
    decision = resolve("mtp_max_depth", ctx)
    return (
        int(cast(int, decision.value))
        if decision.source not in {"default", "capability"}
        else requested
    )
