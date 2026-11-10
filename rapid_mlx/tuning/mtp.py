# SPDX-License-Identifier: Apache-2.0
"""Depth policy at the installed single-request MTP capability boundary."""

from . import resolve, runtime_context


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
    ctx = runtime_context(
        mx.device_info(), config=config, mode="mtp", has_mtp_head=has_head
    )
    decision = resolve("mtp_max_depth", ctx)
    return (
        int(decision.value)
        if decision.source not in {"default", "blocked"}
        else requested
    )
