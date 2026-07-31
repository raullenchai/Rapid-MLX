# SPDX-License-Identifier: Apache-2.0
"""Shared output sample-rate and channel conversion for audio generators."""

from __future__ import annotations

import math

import numpy as np


def convert_audio_output(
    audio: np.ndarray,
    source_rate: int,
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """Return sample-first float32 audio in the requested output format."""
    if source_rate <= 0:
        raise ValueError("source sample rate must be positive")

    # MLX arrays do not expose NumPy's buffer protocol consistently across
    # releases, but their public ``tolist`` bridge is stable.
    source = audio.tolist() if hasattr(audio, "tolist") else audio
    value = np.asarray(source, dtype=np.float32)
    if value.ndim == 1:
        value = value[:, None]
    elif value.ndim == 2:
        # TTS backends are not uniform: most return sample-first arrays,
        # while a few return the channel-first shape used by ML models.
        if value.shape[0] in (1, 2) and value.shape[1] > 2:
            value = value.T
    else:
        raise ValueError(f"audio must be one- or two-dimensional, got {value.ndim}D")
    if value.shape[1] not in (1, 2):
        raise ValueError(f"audio must be mono or stereo, got {value.shape[1]} channels")

    target_channels = value.shape[1] if channels is None else channels
    if target_channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")
    if target_channels == 1 and value.shape[1] == 2:
        value = value.mean(axis=1, keepdims=True, dtype=np.float32)
    elif target_channels == 2 and value.shape[1] == 1:
        value = np.repeat(value, 2, axis=1)

    target_rate = source_rate if sample_rate is None else sample_rate
    if target_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if target_rate != source_rate:
        from scipy.signal import resample_poly

        divisor = math.gcd(source_rate, target_rate)
        value = resample_poly(
            value,
            target_rate // divisor,
            source_rate // divisor,
            axis=0,
        ).astype(np.float32, copy=False)

    value = np.clip(value, -1.0, 1.0)
    if target_channels == 1:
        value = value[:, 0]
    return value, target_rate, target_channels
