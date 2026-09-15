"""DDPM scheduler (variance-preserving diffusion).

Implements the shared VP math (alphas_cumprod, add_noise, prediction targets,
predicted-x0 recovery) plus the ancestral DDPM reverse step. DDIM reuses all of
this and only swaps the reverse step.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx

from .base import BetaSchedule, Scheduler, SchedulerConfig, expand_to, make_betas


@dataclasses.dataclass
class DDPMConfig(SchedulerConfig):
    beta_schedule: BetaSchedule = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    clip_sample: bool = False
    clip_sample_range: float = 1.0


class DDPMScheduler(Scheduler):
    config_class = DDPMConfig
    config: DDPMConfig

    def __init__(self, config: DDPMConfig | None = None):
        super().__init__(config or DDPMConfig())
        betas = make_betas(
            self.config.beta_schedule,
            self.config.num_train_timesteps,
            self.config.beta_start,
            self.config.beta_end,
        )
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = mx.cumprod(self.alphas)
        self._stride = 1

    # --- coefficient helpers ---------------------------------------------
    def _acp(self, t: mx.array) -> mx.array:
        return self.alphas_cumprod[t]

    def _sqrt_terms(self, t: mx.array, ndim: int) -> tuple[mx.array, mx.array]:
        acp = self._acp(t)
        return expand_to(mx.sqrt(acp), ndim), expand_to(mx.sqrt(1.0 - acp), ndim)

    # --- training ---------------------------------------------------------
    def sample_timesteps(self, batch_size: int, key: mx.array) -> mx.array:
        return mx.random.randint(
            0, self.config.num_train_timesteps, (batch_size,), key=key
        )

    def add_noise(self, x0: mx.array, noise: mx.array, t: mx.array) -> mx.array:
        sqrt_acp, sqrt_omacp = self._sqrt_terms(t, x0.ndim)
        return sqrt_acp * x0 + sqrt_omacp * noise

    def get_target(self, x0: mx.array, noise: mx.array, t: mx.array) -> mx.array:
        pt = self.config.prediction_type
        if pt == "epsilon":
            return noise
        if pt == "sample":
            return x0
        if pt == "v_prediction":
            sqrt_acp, sqrt_omacp = self._sqrt_terms(t, x0.ndim)
            return sqrt_acp * noise - sqrt_omacp * x0
        raise ValueError(
            f"{type(self).__name__} does not support prediction_type={pt!r}."
        )

    def predict_x0(
        self, model_output: mx.array, t: mx.array, sample: mx.array
    ) -> mx.array:
        """Recover predicted clean sample x0 from the network output."""
        sqrt_acp, sqrt_omacp = self._sqrt_terms(t, sample.ndim)
        pt = self.config.prediction_type
        if pt == "epsilon":
            x0 = (sample - sqrt_omacp * model_output) / sqrt_acp
        elif pt == "sample":
            x0 = model_output
        elif pt == "v_prediction":
            x0 = sqrt_acp * sample - sqrt_omacp * model_output
        else:  # pragma: no cover - guarded in get_target
            raise ValueError(pt)
        if self.config.clip_sample:
            r = self.config.clip_sample_range
            x0 = mx.clip(x0, -r, r)
        return x0

    # --- sampling ---------------------------------------------------------
    def set_timesteps(self, num_inference_steps: int) -> None:
        T = self.config.num_train_timesteps
        if num_inference_steps > T:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) > num_train_timesteps ({T})."
            )
        self.num_inference_steps = num_inference_steps
        self._stride = T // num_inference_steps
        self.timesteps = (mx.arange(num_inference_steps) * self._stride)[::-1]
        self._step_index = 0

    def step(
        self,
        model_output: mx.array,
        t: mx.array,
        sample: mx.array,
        key: mx.array | None = None,
    ) -> mx.array:
        ti = int(t.item()) if isinstance(t, mx.array) else int(t)
        prev = ti - self._stride
        acp_t = self.alphas_cumprod[ti]
        acp_prev = self.alphas_cumprod[prev] if prev >= 0 else mx.array(1.0)

        x0 = self.predict_x0(model_output, mx.array(ti), sample)

        beta_t = 1.0 - acp_t
        beta_prev = 1.0 - acp_prev
        current_alpha = acp_t / acp_prev
        current_beta = 1.0 - current_alpha

        x0_coef = mx.sqrt(acp_prev) * current_beta / beta_t
        sample_coef = mx.sqrt(current_alpha) * beta_prev / beta_t
        mean = x0_coef * x0 + sample_coef * sample

        if prev < 0:
            return mean
        var = current_beta * beta_prev / beta_t
        key = key if key is not None else mx.random.key(ti)
        noise = mx.random.normal(sample.shape, key=key)
        return mean + mx.sqrt(mx.maximum(var, 1e-20)) * noise
