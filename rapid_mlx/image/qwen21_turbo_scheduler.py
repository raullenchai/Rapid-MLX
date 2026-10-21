# SPDX-License-Identifier: Apache-2.0
"""Sampler for Qwen-Image-2.1-Turbo's fixed sigma nodes (see :mod:`.qwen21_turbo`)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mflux.models.common.schedulers.base_scheduler import BaseScheduler

from .qwen21_turbo import turbo_sigmas


class Qwen21TurboScheduler(BaseScheduler):
    """First-order Euler sampler on the checkpoint's published sigma nodes.

    The Qwen-Image 2.1 transformer reads ``sigmas[t]`` as its timestep, so
    replacing the sigma ladder is the main difference from mflux's default
    linear scheduler. ``step`` is the same Euler update, accumulated in
    float32 like the reference pipeline so eight large steps do not compound
    bf16 rounding.
    """

    def __init__(self, config: Any):
        steps = int(config.num_inference_steps)
        self.config = config
        self._sigmas = mx.array(turbo_sigmas(steps), dtype=mx.float32)
        self._timesteps = mx.arange(steps, dtype=mx.float32)

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def step(
        self, noise: mx.array, timestep: int, latents: mx.array, **kwargs: Any
    ) -> mx.array:
        dt = self._sigmas[timestep + 1] - self._sigmas[timestep]
        return (latents.astype(mx.float32) + noise.astype(mx.float32) * dt).astype(
            latents.dtype
        )
