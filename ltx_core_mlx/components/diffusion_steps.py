"""Diffusion step primitives — ``EulerDiffusionStep``, ``Res2sDiffusionStep``, ``EulerCfgPpDiffusionStep``.

Mirrors upstream ``ltx_core.components.diffusion_steps`` 1:1 (PR #212).
Each ``step`` method advances ``(sample, denoised, sigmas, step_index)`` to the
next sample under its specific sampler dynamics.

Currently provided as available primitives — our existing ``utils/samplers.py``
still inlines this math for the live denoise loops (``denoise_loop``,
``guided_denoise_loop``, ``res2s_denoise_loop``). A future refactor can
delegate the per-step math here for tighter upstream parity.
"""

from __future__ import annotations

from typing import Protocol

import mlx.core as mx

from ltx_core_mlx.utils.diffusion import to_velocity


class DiffusionStepProtocol(Protocol):
    """Protocol mirroring upstream ``ltx_core.components.protocols.DiffusionStepProtocol``."""

    def step(
        self,
        sample: mx.array,
        denoised_sample: mx.array,
        sigmas: mx.array,
        step_index: int,
        **kwargs,
    ) -> mx.array: ...


def _get_ancestral_step(
    sigma_from: mx.array,
    sigma_to: mx.array,
    eta: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Compute ``(sigma_down, sigma_up)`` for one DDIM ancestral step.

    Both inputs are in the rescaled parameterization ``sigma / alpha``.
    Returns ``sigma_down`` (deterministic component) and ``sigma_up``
    (stochastic component) in the same rescaled space.
    """
    if not eta:
        return sigma_to, mx.zeros_like(sigma_to)
    variance = sigma_to**2 * mx.maximum(sigma_from**2 - sigma_to**2, 0.0) / sigma_from**2
    sigma_up = mx.minimum(eta * variance**0.5, sigma_to)
    sigma_down = mx.maximum(sigma_to**2 - sigma_up**2, 0.0) ** 0.5
    return sigma_down, sigma_up


class EulerDiffusionStep:
    """First-order Euler step. Mirrors upstream ``EulerDiffusionStep``.

    Single step from ``sigma`` to ``sigma_next`` via
    ``sample + velocity * dt`` where ``velocity = (sample - denoised) / sigma``.
    """

    def step(
        self,
        sample: mx.array,
        denoised_sample: mx.array,
        sigmas: mx.array,
        step_index: int,
        **_kwargs,
    ) -> mx.array:
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma
        velocity = to_velocity(sample, sigma, denoised_sample)
        out_dtype = sample.dtype
        return (sample.astype(mx.float32) + velocity.astype(mx.float32) * dt).astype(out_dtype)


class Res2sDiffusionStep:
    """Second-order res_2s step with SDE noise injection.

    Mirrors upstream ``Res2sDiffusionStep``. Used by ``res2s_denoise_loop``.
    Advances ``sample`` from ``sigma`` to ``sigma_next`` by mixing the
    deterministic update (from the denoised prediction) with injected noise
    via :meth:`get_sde_coeff`, producing variance-preserving transitions.
    """

    @staticmethod
    def get_sde_coeff(
        sigma_next: mx.array,
        sigma_up: mx.array | None = None,
        sigma_down: mx.array | None = None,
        sigma_max: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Compute ``(alpha_ratio, sigma_down, sigma_up)`` for the step.

        Given either ``sigma_down`` or ``sigma_up``, returns the mixing
        coefficients used for variance-preserving noise injection.
        """
        if sigma_down is not None:
            alpha_ratio = (1.0 - sigma_next) / (1.0 - sigma_down)
            sigma_up = mx.maximum(sigma_next**2 - sigma_down**2 * alpha_ratio**2, 0.0) ** 0.5
        elif sigma_up is not None:
            # Avoid sqrt(neg_num)
            sigma_up = mx.minimum(sigma_up, sigma_next * 0.9999)
            sigmax = sigma_max if sigma_max is not None else mx.ones_like(sigma_next)
            sigma_signal = sigmax - sigma_next
            sigma_residual = mx.maximum(sigma_next**2 - sigma_up**2, 0.0) ** 0.5
            alpha_ratio = sigma_signal + sigma_residual
            sigma_down = sigma_residual / alpha_ratio
        else:
            alpha_ratio = mx.ones_like(sigma_next)
            sigma_down = sigma_next
            sigma_up = mx.zeros_like(sigma_next)

        # Replace NaNs (sigma_up -> 0, sigma_down -> sigma_next, alpha_ratio -> 1)
        sigma_up = mx.where(mx.isnan(sigma_up), mx.zeros_like(sigma_up), sigma_up)
        sigma_down = mx.where(mx.isnan(sigma_down), sigma_next.astype(sigma_down.dtype), sigma_down)
        alpha_ratio = mx.where(mx.isnan(alpha_ratio), mx.ones_like(alpha_ratio), alpha_ratio)

        return alpha_ratio, sigma_down, sigma_up

    def step(
        self,
        sample: mx.array,
        denoised_sample: mx.array,
        sigmas: mx.array,
        step_index: int,
        noise: mx.array,
        eta: float = 0.5,
    ) -> mx.array:
        """Advance one step with SDE noise injection."""
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        alpha_ratio, sigma_down, sigma_up = self.get_sde_coeff(sigma_next, sigma_up=sigma_next * eta)
        out_dtype = denoised_sample.dtype

        if bool(mx.any(sigma_up == 0).item()) or bool(mx.any(sigma_next == 0).item()):
            return denoised_sample

        # Epsilon prediction
        eps_next = (sample - denoised_sample) / (sigma - sigma_next)
        denoised_next = sample - sigma * eps_next

        # Mix deterministic and stochastic components
        x_noised = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise
        return x_noised.astype(out_dtype)


class EulerCfgPpDiffusionStep:
    """CFG++ Euler step (https://arxiv.org/abs/2406.08070).

    Mirrors upstream ``EulerCfgPpDiffusionStep``. Uses the unconditioned
    prediction to compute the ODE derivative direction, keeping the
    conditioned prediction as the target denoised state. Ancestral (DDIM)
    noise injection in the rescaled sigma parameterization ``sigma / alpha``.
    """

    def __init__(self, eta: float = 1.0, s_noise: float = 1.0) -> None:
        self.eta = eta
        self.s_noise = s_noise

    def step(
        self,
        sample: mx.array,
        denoised_sample: mx.array,
        sigmas: mx.array,
        step_index: int,
        uncond_denoised: mx.array,
        noise: mx.array | None = None,
        **_kwargs,
    ) -> mx.array:
        """Advance one CFG++ Euler step.

        Args:
            sample: Current noisy latent x_t.
            denoised_sample: Conditioned denoised prediction x0^cond.
            sigmas: Full sigma schedule.
            step_index: Current step index.
            uncond_denoised: Unconditioned denoised prediction x0^uncond.
            noise: Noise for stochastic injection; ignored when eta=0 or s_noise=0.
        """
        sigma_s = sigmas[step_index].astype(mx.float32)
        sigma_t = sigmas[step_index + 1].astype(mx.float32)
        # Same epsilon clamp as upstream (avoid div-by-zero when sigma == 1.0).
        _eps = 1e-7
        alpha_s = mx.maximum(1.0 - sigma_s, _eps)
        alpha_t = mx.maximum(1.0 - sigma_t, _eps)

        x = sample.astype(mx.float32)
        denoised = denoised_sample.astype(mx.float32)
        uncond = uncond_denoised.astype(mx.float32)

        # ODE derivative: direction toward noise using uncond prediction (CFG++ correction)
        d = (x - alpha_s * uncond) / sigma_s

        # Ancestral step in rescaled sigma space (sigma / alpha)
        sigma_down, sigma_up = _get_ancestral_step(sigma_s / alpha_s, sigma_t / alpha_t, eta=self.eta)
        sigma_down = alpha_t * sigma_down

        x_next = alpha_t * denoised + sigma_down * d
        if noise is not None and self.eta > 0 and self.s_noise > 0:
            x_next = x_next + alpha_t * noise.astype(mx.float32) * self.s_noise * sigma_up
        return x_next.astype(sample.dtype)


__all__ = [
    "DiffusionStepProtocol",
    "EulerCfgPpDiffusionStep",
    "EulerDiffusionStep",
    "Res2sDiffusionStep",
]
