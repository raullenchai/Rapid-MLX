"""Diffusion sampling helpers — ``to_velocity`` / ``to_denoised``.

Mirrors upstream ``ltx_core.utils.to_velocity`` / ``to_denoised``. These are
the standard rectified-flow conversions between (sample, denoised) and
velocity at a given sigma.
"""

from __future__ import annotations

import mlx.core as mx


def to_velocity(sample: mx.array, sigma: float | mx.array, denoised_sample: mx.array) -> mx.array:
    """Convert ``(sample, denoised)`` to velocity at ``sigma``.

    Returns ``(sample - denoised) / sigma``, computed in fp32 then cast back
    to ``sample.dtype``. Raises if ``sigma`` is zero.
    """
    if isinstance(sigma, mx.array):
        sigma_f = float(sigma.item())
    else:
        sigma_f = float(sigma)
    if sigma_f == 0:
        raise ValueError("Sigma can't be 0.0")
    out_dtype = sample.dtype
    return ((sample.astype(mx.float32) - denoised_sample.astype(mx.float32)) / sigma_f).astype(out_dtype)


def to_denoised(sample: mx.array, sigma: float | mx.array, velocity: mx.array) -> mx.array:
    """Convert ``(sample, velocity)`` back to the denoised prediction at ``sigma``.

    Returns ``sample - sigma * velocity``, computed in fp32 then cast back.
    """
    if isinstance(sigma, mx.array):
        sigma_f = float(sigma.item())
    else:
        sigma_f = float(sigma)
    out_dtype = sample.dtype
    return (sample.astype(mx.float32) - sigma_f * velocity.astype(mx.float32)).astype(out_dtype)


__all__ = ["to_denoised", "to_velocity"]
