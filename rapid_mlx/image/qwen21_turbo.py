# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-2.1-Turbo sampling schedule.

Turbo is a distilled checkpoint of Qwen-Image-2.1 with the same architecture.
At inference time it differs from the base model in two ways: it samples on
eight fixed sigma nodes that the checkpoint publishes as ``sample_sigmas`` in
its ``model_index.json`` (its scheduler config turns dynamic shifting off, so
there is no resolution shift), and it runs without classifier-free guidance.

The pinned mflux runtime resolves a dotted ``scheduler`` path to a
``BaseScheduler`` subclass, so the engine passes :data:`SCHEDULER_PATH` to
``generate_image`` and leaves mflux itself untouched. This module stays
import-light (no mlx / mflux) because catalog code reads the step count; the
sampler itself lives in :mod:`.qwen21_turbo_scheduler`.
"""

from __future__ import annotations

#: ``sample_sigmas`` from ``Qwen/Qwen-Image-2.1-Turbo`` ``model_index.json``.
TURBO_SIGMAS: tuple[float, ...] = (
    1.0,
    0.978453,
    0.95418,
    0.926626,
    0.89508,
    0.845148,
    0.704534,
    0.414568,
)
TURBO_STEPS = len(TURBO_SIGMAS)
#: The distilled checkpoint runs without classifier-free guidance.
TURBO_GUIDANCE = 1.0

SCHEDULER_PATH = "rapid_mlx.image.qwen21_turbo_scheduler.Qwen21TurboScheduler"


def turbo_sigmas(steps: int) -> list[float]:
    """Sigma ladder (including the terminal 0) for the published schedule.

    The checkpoint documents only its eight saved nodes; other schedules are
    unevaluated upstream, so any other step count is refused rather than
    synthesized.
    """

    if steps != TURBO_STEPS:
        raise ValueError(
            f"Qwen-Image 2.1 Turbo samples its published {TURBO_STEPS}-step "
            f"schedule only; got {steps} steps."
        )
    return [*TURBO_SIGMAS, 0.0]
