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

    raw_shape = getattr(audio, "shape", None)
    if raw_shape is None:
        raw_shape = np.shape(audio)
    shape = tuple(int(part) for part in raw_shape)
    if len(shape) == 1:
        source_channels = 1
    elif len(shape) == 2:
        # The public contract is sample-first, matching scipy.wavfile,
        # soundfile, and TTSEngine.to_bytes: (samples, channels).
        source_channels = shape[1]
    else:
        raise ValueError(f"audio must be one- or two-dimensional, got {len(shape)}D")
    if source_channels not in (1, 2):
        raise ValueError(
            f"audio must be mono or stereo, got {source_channels} channels"
        )
    if sample_rate is None and channels is None:
        # A missing output preference is a strict compatibility no-op: keep
        # dtype, object identity, out-of-range floats, and backend layout.
        return audio, source_rate, source_channels

    # NumPy is the hot path for music and most TTS engines; never turn a
    # multi-million-sample array into an equally large Python object graph.
    # Some MLX releases lack a working NumPy buffer bridge, so only those
    # array types fall back to their stable public ``tolist`` method.
    original = np.asarray(audio) if isinstance(audio, np.ndarray) else None
    if original is None:
        try:
            original = np.asarray(audio)
        except (TypeError, ValueError):
            source = audio.tolist() if hasattr(audio, "tolist") else audio
            original = np.asarray(source)
    if np.issubdtype(original.dtype, np.signedinteger):
        scale = float(
            max(abs(np.iinfo(original.dtype).min), np.iinfo(original.dtype).max)
        )
        value = original.astype(np.float32) / scale
    elif np.issubdtype(original.dtype, np.unsignedinteger):
        midpoint = float(np.iinfo(original.dtype).max + 1) / 2
        value = (original.astype(np.float32) - midpoint) / midpoint
    else:
        value = original.astype(np.float32, copy=False)
    if value.ndim == 1:
        value = value[:, None]
    elif value.ndim != 2:  # pragma: no cover - shape validation owns this branch
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
