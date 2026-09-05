"""Scheduler base class + shared diffusion math.

A *scheduler* owns the corruption process and the reverse step. The same object
is used for training (``add_noise`` + ``get_target`` + ``sample_timesteps``) and
for sampling (``set_timesteps`` + ``step``). This keeps all SDE/ODE math in one
place and lets networks stay agnostic to the noise schedule.

Two timestep conventions coexist, each scheduler picks one and is internally
consistent:

* **Discrete diffusion** (DDPM, DDIM): ``t`` is an integer index into the
  precomputed ``alphas_cumprod`` table.
* **Continuous flow / sigma** (Euler, FlowMatch): ``t`` is a float sigma / time.

The Trainer always obtains timesteps from ``sample_timesteps`` and passes them
straight back to ``add_noise``/``get_target``, so callers never need to know which
convention a scheduler uses.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import mlx.core as mx

from ..configuration import Config

PredictionType = Literal["epsilon", "v_prediction", "sample", "velocity"]
BetaSchedule = Literal["linear", "scaled_linear", "cosine"]


@dataclasses.dataclass
class SchedulerConfig(Config):
    num_train_timesteps: int = 1000
    prediction_type: PredictionType = "epsilon"


def make_betas(
    schedule: BetaSchedule,
    num_train_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> mx.array:
    """Return the ``(num_train_timesteps,)`` beta schedule."""
    if schedule == "linear":
        return mx.linspace(beta_start, beta_end, num_train_timesteps)
    if schedule == "scaled_linear":  # Stable Diffusion convention
        return mx.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
    if schedule == "cosine":  # Nichol & Dhariwal (2021)
        steps = num_train_timesteps + 1
        s = 0.008
        x = mx.linspace(0, num_train_timesteps, steps)
        acp = mx.cos(((x / num_train_timesteps) + s) / (1 + s) * mx.pi * 0.5) ** 2
        acp = acp / acp[0]
        betas = 1 - (acp[1:] / acp[:-1])
        return mx.clip(betas, 0, 0.999)
    raise ValueError(f"Unknown beta schedule {schedule!r}.")


def expand_to(coef: mx.array, ndim: int) -> mx.array:
    """Reshape a coefficient to broadcast against a rank-``ndim`` tensor.

    A ``(B,)`` per-sample coefficient becomes ``(B, 1, ..., 1)``; a 0-d scalar is
    returned unchanged (it already broadcasts).
    """
    if coef.ndim == 0:
        return coef
    return coef.reshape(coef.shape[0], *([1] * (ndim - 1)))


class Scheduler:
    """Abstract base. Subclasses implement the four core methods below."""

    config_class: type[SchedulerConfig] = SchedulerConfig

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.timesteps: mx.array | None = None
        self.num_inference_steps: int | None = None
        self._step_index = 0

    # --- training ---------------------------------------------------------
    def sample_timesteps(self, batch_size: int, key: mx.array) -> mx.array:
        """Draw a batch of training timesteps in this scheduler's convention."""
        raise NotImplementedError

    def add_noise(self, x0: mx.array, noise: mx.array, t: mx.array) -> mx.array:
        """Forward process: corrupt ``x0`` with ``noise`` at timestep ``t``."""
        raise NotImplementedError

    def get_target(self, x0: mx.array, noise: mx.array, t: mx.array) -> mx.array:
        """The quantity the network is trained to predict (per ``prediction_type``)."""
        raise NotImplementedError

    # --- sampling ---------------------------------------------------------
    def set_timesteps(self, num_inference_steps: int) -> None:
        """Configure the inference timestep grid (descending) and reset state."""
        raise NotImplementedError

    def step(
        self,
        model_output: mx.array,
        t: mx.array,
        sample: mx.array,
        key: mx.array | None = None,
    ) -> mx.array:
        """Take one reverse step, returning the previous (less-noisy) sample."""
        raise NotImplementedError

    def scale_model_input(self, sample: mx.array, t: mx.array) -> mx.array:
        """Optional pre-network input scaling (identity unless overridden)."""
        return sample

    def set_begin_index(self, begin_index: int) -> None:
        """Start sampling partway through the configured schedule.

        Image/video-to-X pipelines use this after adding noise to encoded input
        latents. Keeping the step cursor on the scheduler avoids pipeline code
        reaching into scheduler internals.
        """
        if self.timesteps is None:
            raise RuntimeError("call set_timesteps() before set_begin_index().")
        if not 0 <= begin_index < len(self.timesteps):
            raise ValueError(
                f"begin_index must be in [0, {len(self.timesteps) - 1}], got {begin_index}."
            )
        self._step_index = begin_index

    # --- persistence ------------------------------------------------------
    def save_pretrained(self, save_directory) -> None:
        self.config.save(save_directory)

    @classmethod
    def from_config(cls, config: SchedulerConfig) -> Scheduler:
        return cls(config)
