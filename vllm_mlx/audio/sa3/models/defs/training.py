"""Training utilities for the standalone Stable Audio 3 MLX models."""

from __future__ import annotations

import math
import typing as tp
from statistics import NormalDist

import mlx.core as mx
import numpy as np


_TIMESTEP_SAMPLERS = {
    "uniform",
    "logit_normal",
    "trunc_logit_normal",
    "log_snr",
    "log_snr_uniform",
}
_DISTRIBUTION_SHIFTS = {"none", "full", "flux", "logsnr"}
_STANDARD_NORMAL = NormalDist()


def sample_training_timesteps(
    sampler: str,
    batch_size: int,
    *,
    rng: np.random.Generator,
    options: dict[str, float] | None = None,
) -> np.ndarray:
    """Sample SA3 training timesteps with the PyTorch implementation's defaults."""

    sampler = str(sampler).strip().lower()
    if sampler not in _TIMESTEP_SAMPLERS:
        raise ValueError(f"Unsupported timestep sampler: {sampler!r}.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    options = options or {}
    if sampler == "uniform":
        values = rng.random(batch_size)
    elif sampler == "logit_normal":
        values = _sigmoid(rng.standard_normal(batch_size))
    elif sampler == "trunc_logit_normal":
        values = 1.0 - _truncated_logistic_normal_rescaled(
            batch_size,
            rng=rng,
        )
    elif sampler == "log_snr":
        mean = float(options.get("mean_logsnr", -1.2))
        std = float(options.get("std_logsnr", 2.0))
        logsnr = rng.standard_normal(batch_size) * std + mean
        values = np.clip(_sigmoid(-logsnr), 1e-4, 1.0 - 1e-4)
    else:
        minimum = float(options.get("min_logsnr", -6.0))
        maximum = float(options.get("max_logsnr", 5.0))
        if maximum <= minimum:
            raise ValueError("max_logsnr must be greater than min_logsnr.")
        logsnr = rng.uniform(minimum, maximum, batch_size)
        values = np.clip(_sigmoid(-logsnr), 1e-4, 1.0 - 1e-4)
    return np.asarray(values, dtype=np.float32)


def shift_training_timesteps(
    timesteps: tp.Sequence[float] | np.ndarray,
    sequence_length: int | tp.Sequence[int] | np.ndarray,
    *,
    shift_type: str = "full",
    options: dict[str, float | int | bool] | None = None,
) -> np.ndarray:
    """Apply an SA3 training distribution shift to a timestep batch."""

    shift_type = str(shift_type).strip().lower()
    if shift_type == "identity":
        shift_type = "none"
    if shift_type not in _DISTRIBUTION_SHIFTS:
        raise ValueError(f"Unsupported distribution shift: {shift_type!r}.")

    values = np.asarray(timesteps, dtype=np.float64).reshape(-1)
    lengths = np.asarray(sequence_length, dtype=np.float64).reshape(-1)
    if lengths.size == 1:
        lengths = np.repeat(lengths, values.size)
    if lengths.size != values.size:
        raise ValueError(
            "sequence_length must contain one value or match the timestep batch."
        )
    if shift_type == "none":
        return values.astype(np.float32)

    options = options or {}
    shifted = [
        _shift_timestep(
            float(timestep),
            float(length),
            shift_type=shift_type,
            options=options,
        )
        for timestep, length in zip(values, lengths, strict=True)
    ]
    return np.asarray(shifted, dtype=np.float32)


def rectified_flow_loss(
    model,
    clean,
    timesteps,
    *,
    noise=None,
    loss_mask=None,
    model_kwargs: dict[str, tp.Any] | None = None,
):
    """Compute the SA3 rectified-flow velocity loss for pre-encoded latents."""

    if noise is None:
        noise = mx.random.normal(clean.shape, dtype=clean.dtype)
    timesteps = timesteps.astype(mx.float32)
    alpha = (1.0 - timesteps)[:, None, None].astype(clean.dtype)
    sigma = timesteps[:, None, None].astype(clean.dtype)
    noised = clean * alpha + noise * sigma
    target = noise - clean
    prediction = model(noised, timesteps, **(model_kwargs or {}))
    mse = (prediction.astype(mx.float32) - target.astype(mx.float32)) ** 2

    if loss_mask is None:
        return mx.mean(mse)
    mask = loss_mask[:, None, :].astype(mx.float32)
    per_sample_denominator = mx.maximum(
        mx.sum(mask, axis=(1, 2)) * mse.shape[1],
        1.0,
    )
    per_sample_loss = mx.sum(mse * mask, axis=(1, 2)) / per_sample_denominator
    return mx.mean(per_sample_loss)


def _shift_timestep(
    timestep: float,
    sequence_length: float,
    *,
    shift_type: str,
    options: dict[str, float | int | bool],
) -> float:
    if timestep <= 0:
        return 0.0
    if timestep >= 1:
        return 1.0

    if shift_type == "full":
        minimum = float(options.get("min_length", 256))
        maximum = float(options.get("max_length", 4096))
        length = _clamp(sequence_length, minimum, maximum)
        base_shift = float(options.get("base_shift", 0.5))
        max_shift = float(options.get("max_shift", 1.15))
        mu = -(
            base_shift
            + (max_shift - base_shift) * (length - minimum) / (maximum - minimum)
        )
        exp_mu = math.exp(mu)
        shifted = 1.0 - exp_mu / (exp_mu + (1.0 / (1.0 - timestep) - 1.0))
        if bool(options.get("use_sine", False)):
            shifted = math.sin(shifted * math.pi / 2.0)
        return shifted

    if shift_type == "flux":
        minimum = float(options.get("min_length", 256))
        maximum = float(options.get("max_length", 4096))
        length = _clamp(sequence_length, minimum, maximum)
        alpha_min = max(float(options.get("alpha_min", 1.0)), 1e-8)
        alpha_max = max(float(options.get("alpha_max", 1.0)), 1e-8)
        denominator = math.log(maximum) - math.log(minimum)
        fraction = (math.log(length) - math.log(minimum)) / max(
            denominator,
            1e-8,
        )
        log_alpha = math.log(alpha_min) + fraction * (
            math.log(alpha_max) - math.log(alpha_min)
        )
        alpha = math.exp(log_alpha)
        return alpha * timestep / (1.0 + (alpha - 1.0) * timestep)

    anchor_length = float(options.get("anchor_length", 2000))
    anchor_logsnr = float(options.get("anchor_logsnr", -6.2))
    rate = float(options.get("rate", 1.0))
    logsnr_end = float(options.get("logsnr_end", 2.0))
    logsnr_start = anchor_logsnr - rate * math.log2(
        max(sequence_length, 1.0) / anchor_length
    )
    logsnr = logsnr_end - timestep * (logsnr_end - logsnr_start)
    return 1.0 / (1.0 + math.exp(logsnr))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


def _truncated_logistic_normal_rescaled(
    size: int,
    *,
    rng: np.random.Generator,
    left_trunc: float = 0.075,
    right_trunc: float = 1.0,
) -> np.ndarray:
    if not 0.0 < left_trunc < right_trunc <= 1.0:
        raise ValueError("Expected 0 < left_trunc < right_trunc <= 1.")

    lower_logit = math.log(left_trunc / (1.0 - left_trunc))
    lower_cdf = _STANDARD_NORMAL.cdf(lower_logit)
    upper_cdf = (
        1.0
        if right_trunc == 1.0
        else _STANDARD_NORMAL.cdf(math.log(right_trunc / (1.0 - right_trunc)))
    )
    uniforms = lower_cdf + (upper_cdf - lower_cdf) * rng.random(size)
    epsilon = np.finfo(np.float64).eps
    logits = np.asarray(
        [
            _STANDARD_NORMAL.inv_cdf(float(np.clip(value, epsilon, 1.0 - epsilon)))
            for value in uniforms
        ],
        dtype=np.float64,
    )
    samples = _sigmoid(logits)
    return (samples - left_trunc) / (right_trunc - left_trunc)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        raise ValueError("max_length must be greater than min_length.")
    return min(max(value, minimum), maximum)
