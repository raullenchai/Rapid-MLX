"""Audio loading utilities using ffmpeg."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from ltx_core_mlx.utils.ffmpeg import find_ffmpeg


@dataclass
class AudioData:
    """Container for audio data."""

    waveform: mx.array  # (1, channels, samples)
    sample_rate: int


def load_audio(
    path: str | Path,
    target_sample_rate: int = 16000,
    start_time: float = 0.0,
    max_duration: float | None = None,
    mono: bool = False,
) -> AudioData | None:
    """Load audio from a file using ffmpeg.

    Args:
        path: Path to audio/video file.
        target_sample_rate: Target sample rate in Hz.
        start_time: Start time in seconds.
        max_duration: Maximum duration in seconds. None = read to end.
        mono: If True, downmix to mono.

    Returns:
        AudioData with waveform (1, channels, samples), or None if no audio.
    """
    ffmpeg = find_ffmpeg()
    path = str(path)

    cmd = [ffmpeg]
    if start_time > 0:
        cmd.extend(["-ss", str(start_time)])
    cmd.extend(["-i", path])
    if max_duration is not None:
        cmd.extend(["-t", str(max_duration)])

    channels = 1 if mono else 2
    cmd.extend(
        [
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            str(channels),
            "-ar",
            str(target_sample_rate),
            "-vn",
            "-",
        ]
    )

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        return None

    raw = result.stdout
    if not raw:
        return None

    samples = np.frombuffer(raw, dtype=np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).T  # (channels, samples)
    else:
        samples = samples.reshape(1, -1)  # (1, samples)

    waveform = mx.array(samples)[None, :, :]  # (1, channels, samples)
    return AudioData(waveform=waveform, sample_rate=target_sample_rate)
