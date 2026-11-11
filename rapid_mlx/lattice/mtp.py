# SPDX-License-Identifier: Apache-2.0
"""Depth policy at the installed single-request MTP capability boundary."""

import logging
from dataclasses import replace
from typing import cast

from . import LatticeContext, model_profile, resolve, runtime_context

logger = logging.getLogger(__name__)


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
    except Exception:  # noqa: BLE001 -- unavailable facts cannot qualify a lattice cell
        return None
    if len(formats) == 1:
        bits, group_size, mode = formats.pop()
        if type(bits) is int and type(group_size) is int and mode == "affine":
            return bits, group_size, mode
    return None


def _model_context(model) -> LatticeContext | None:
    """Snapshot installed-head and expert facts for MTP lattice cells."""
    has_head = (
        all(
            callable(getattr(model, name, None))
            for name in ("mtp_forward", "make_mtp_cache")
        )
        and getattr(model, "mtp", None) is not None
    )
    if not has_head:
        return None

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
    return runtime_context(
        mx.device_info(), model=profile, mode="mtp", has_mtp_head=has_head
    )


def resolve_max_depth(model, requested: int, *, explicit: bool) -> int:
    """Keep configured depths unless a qualified policy or override selects one.

    Explicit JSON/legacy depths win. A default decision preserves catalog and
    force-opt-in depths as well as the ordinary K=1. Checkpoint limits remain
    enforced by the caller after this policy, before calibration/generation.
    """
    if explicit:
        return requested
    ctx = _model_context(model)
    if ctx is None:
        return requested
    decision = resolve("mtp_max_depth", ctx)
    return (
        int(cast(int, decision.value))
        if decision.source not in {"default", "capability"}
        else requested
    )


def forward_costs(model, max_k: int) -> tuple[float, ...]:
    """Measured step, verify widths 2..K+1, and draft milliseconds.

    A partial curve cannot price an unmeasured width. Defer to live calibration
    when the selected cell lacks enough measurements for the configured cap.
    """
    if max_k <= 0:
        return ()
    ctx = _model_context(model)
    if ctx is None:
        return ()
    decision = resolve("mtp_forward_costs_ms", ctx)
    values = cast(tuple[float, ...], decision.value)
    if len(values) < max_k + 2:
        return ()
    logger.info(
        "[MTP] forward costs from %s, profile %s: %s",
        decision.source,
        decision.profile_version,
        values,
    )
    return (values[0], *values[1 : max_k + 1], values[-1])


def prefill_block_size(model) -> int:
    """Qualified prefill segmentation shared by cold cached and uncached requests."""
    ctx = _model_context(model)
    if ctx is None:
        return 0
    return int(cast(int, resolve("mtp_prefill_block_size", ctx).value))


def gdn_fusion_enabled(model) -> bool:
    """Measured cells opt into the rollback-compatible GDN execution paths."""
    ctx = _model_context(getattr(model, "language_model", model))
    if ctx is None:
        return False
    decision = resolve("mtp_gdn_fusion", ctx)
    logger.info(
        "[MTP] GDN fusion %s from %s, profile %s",
        decision.value,
        decision.source,
        decision.profile_version,
    )
    return bool(decision.value)
