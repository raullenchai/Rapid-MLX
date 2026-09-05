"""Euler discrete scheduler (k-diffusion / Stable Diffusion style).

Trains with the inherited VP math (discrete timesteps) and samples with a
deterministic first-order ODE solver in sigma space, where
``sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod)``.
"""

from __future__ import annotations

import dataclasses
from typing import cast

import mlx.core as mx
import numpy as np

from .base import expand_to
from .ddpm import DDPMConfig, DDPMScheduler


@dataclasses.dataclass
class EulerConfig(DDPMConfig):
    timestep_spacing: str = "linspace"  # "leading" for Stable Diffusion / SDXL
    steps_offset: int = 0


class EulerDiscreteScheduler(DDPMScheduler):
    config_class = EulerConfig
    config: EulerConfig

    def __init__(self, config: EulerConfig | None = None):
        super().__init__(config or EulerConfig())
        # Full-resolution sigma table over the training schedule.
        self._train_sigmas = mx.sqrt((1.0 - self.alphas_cumprod) / self.alphas_cumprod)
        self.sigmas: mx.array | None = None

    def set_timesteps(self, num_inference_steps: int) -> None:
        T = self.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        train_sigmas = np.array(self._train_sigmas)  # ascending in index
        if self.config.timestep_spacing == "leading":
            step_ratio = T // num_inference_steps
            ts = (
                (np.arange(num_inference_steps) * step_ratio)
                .round()[::-1]
                .astype(np.float64)
            )
            ts = ts + self.config.steps_offset
        else:  # "linspace": fractional timesteps high->low
            ts = np.linspace(0, T - 1, num_inference_steps)[::-1].copy()
        sigmas = np.interp(ts, np.arange(T), train_sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = mx.array(sigmas)
        self.timesteps = mx.array(ts.astype(np.float32))
        self._step_index = 0

    @property
    def init_noise_sigma(self) -> float:
        """Std-dev for the initial latent noise (matches diffusers EulerDiscrete)."""
        assert self.sigmas is not None, "call set_timesteps() before sampling"
        sigma_max = float(self.sigmas[0])
        if self.config.timestep_spacing in ("linspace", "trailing"):
            return sigma_max
        return float((sigma_max**2 + 1.0) ** 0.5)

    def scale_model_input(self, sample: mx.array, t: mx.array) -> mx.array:
        assert self.sigmas is not None, "call set_timesteps() before sampling"
        sigma = self.sigmas[self._step_index]
        return sample / mx.sqrt(sigma**2 + 1.0)

    def _pred_original(
        self, model_output: mx.array, sigma: mx.array, sample: mx.array
    ) -> mx.array:
        pt = self.config.prediction_type
        if pt == "epsilon":
            return sample - sigma * model_output
        if pt == "sample":
            return model_output
        if pt == "v_prediction":
            return model_output * (-sigma / mx.sqrt(sigma**2 + 1.0)) + sample / (
                sigma**2 + 1.0
            )
        raise ValueError(
            f"EulerDiscreteScheduler does not support prediction_type={pt!r}."
        )

    def step(
        self,
        model_output: mx.array,
        t: mx.array,
        sample: mx.array,
        key: mx.array | None = None,
    ) -> mx.array:
        assert self.sigmas is not None, "call set_timesteps() before sampling"
        i = self._step_index
        sigma = self.sigmas[i]
        sigma_next = self.sigmas[i + 1]
        pred_original = self._pred_original(model_output, sigma, sample)
        derivative = (sample - pred_original) / sigma
        dt = sigma_next - sigma
        self._step_index += 1
        return sample + derivative * dt

    def add_noise_sigma(
        self, x0: mx.array, noise: mx.array, sigma: mx.array
    ) -> mx.array:
        """VE-style corruption used by img2img-style sampling starts."""
        return cast(mx.array, x0 + expand_to(sigma, x0.ndim) * noise)
