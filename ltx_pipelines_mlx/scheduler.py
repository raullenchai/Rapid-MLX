"""LTX-2 sigma schedules.

`ltx2_schedule` is a thin wrapper over `mlx_arsenal.diffusion.dynamic_shift_schedule`
that preserves LTX's original keyword name (``steps``) and default ``num_tokens``.
The predefined LTX-specific tables (DISTILLED_SIGMAS, STAGE_2_SIGMAS) and the
LTX-only helpers (get_sigma_schedule, sigma_to_timestep) stay local.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_arsenal.diffusion import dynamic_shift_schedule

_MAX_SHIFT_ANCHOR = 4096


def ltx2_schedule(
    steps: int,
    num_tokens: int = _MAX_SHIFT_ANCHOR,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> list[float]:
    """LTX-2 token-count-adaptive flow-matching sigma schedule."""
    return dynamic_shift_schedule(
        steps,
        num_tokens=num_tokens,
        base_shift=base_shift,
        max_shift=max_shift,
        stretch=stretch,
        terminal=terminal,
    )


__all__ = [
    "DISTILLED_SIGMAS",
    "STAGE_2_SIGMAS",
    "get_sigma_schedule",
    "ltx2_schedule",
    "sigma_to_timestep",
]

# Predefined sigma schedule for 8-step distilled model.
# 9 values = 8 steps (iterate consecutive pairs: sigmas[i], sigmas[i+1]).
DISTILLED_SIGMAS: list[float] = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]

# Sigma schedule for stage 2 refinement (two-stage pipeline).
# 4 values = 3 steps.
STAGE_2_SIGMAS: list[float] = [
    0.909375,
    0.725,
    0.421875,
    0.0,
]


def get_sigma_schedule(
    schedule_name: str = "distilled",
    num_steps: int | None = None,
) -> list[float]:
    """Get a sigma schedule by name.

    Args:
        schedule_name: "distilled" or "stage_2".
        num_steps: Optional number of steps (truncates schedule).

    Returns:
        List of sigma values.
    """
    if schedule_name == "distilled":
        sigmas = DISTILLED_SIGMAS
    elif schedule_name == "stage_2":
        sigmas = STAGE_2_SIGMAS
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")

    if num_steps is not None:
        sigmas = sigmas[:num_steps]
    return sigmas


def sigma_to_timestep(sigma: float) -> mx.array:
    """Convert sigma to timestep array.

    Args:
        sigma: Noise level.

    Returns:
        Timestep as (1,) array.
    """
    return mx.array([sigma], dtype=mx.bfloat16)
