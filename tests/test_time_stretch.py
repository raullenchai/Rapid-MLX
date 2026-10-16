# SPDX-License-Identifier: Apache-2.0
"""CPU-only WSOLA duration, pitch, and boundary contracts."""

import numpy as np
import pytest

from rapid_mlx.audio.time_stretch import time_stretch


def tone(sample_rate=24000, duration=1.0, frequency=220):
    return np.sin(
        2 * np.pi * frequency * np.arange(round(sample_rate * duration)) / sample_rate
    ).astype(np.float32)


@pytest.mark.parametrize("speed", [0.5, 0.75, 1.5, 2.0])
@pytest.mark.parametrize("sample_rate", [16000, 24000, 44100])
def test_duration_and_pitch(speed, sample_rate):
    audio = tone(sample_rate)
    output = time_stretch(audio, sample_rate, speed)
    assert len(output) == round(len(audio) / speed)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    frequencies = np.fft.rfftfreq(len(output), 1 / sample_rate)
    peak = frequencies[np.argmax(np.abs(np.fft.rfft(output)))]
    assert peak == pytest.approx(220, abs=3)
    assert np.sqrt(np.mean(output**2)) == pytest.approx(np.sqrt(0.5), abs=0.06)


@pytest.mark.parametrize("length,speed", [(24000, 1.0), (1, 1.0), (0, 0.5), (0, 2.0)])
def test_identity(length, speed):
    audio = np.zeros(length, dtype=np.float32)
    assert time_stretch(audio, 24000, speed) is audio


@pytest.mark.parametrize("length", [1, 2, 767, 768, 1535])
@pytest.mark.parametrize("speed", [0.5, 2.0])
def test_short_clips_resample(length, speed):
    audio = np.linspace(0.25, 0.75, length, dtype=np.float32)
    output = time_stretch(audio, 24000, speed)
    expected_length = max(1, round(length / speed))
    expected = np.linspace(audio[0], audio[-1], expected_length)
    assert len(output) == expected_length
    assert output.dtype == np.float32
    np.testing.assert_allclose(output, expected, atol=1e-7)


@pytest.mark.parametrize("speed", [0.5, 2.0])
@pytest.mark.parametrize("length", [768, 769, 1000, 24000])
def test_tail_keeps_tone_amplitude(length, speed):
    audio = tone(duration=length / 24000)
    output = time_stretch(audio, 24000, speed)
    overall_rms = np.sqrt(np.mean(output**2))
    assert overall_rms > 0.5
    assert np.sqrt(np.mean(output[-768:] ** 2)) == pytest.approx(overall_rms, rel=0.2)


@pytest.mark.parametrize("length", [768, 769, 1000])
@pytest.mark.parametrize("speed", [0.5, 2.0])
def test_frame_boundary_and_silence(length, speed):
    audio = np.zeros(length, dtype=np.float32)
    output = time_stretch(audio, 24000, speed)
    assert len(output) == round(length / speed)
    assert np.array_equal(output, np.zeros_like(output))


def test_reject_multichannel():
    with pytest.raises(ValueError, match="mono"):
        time_stretch(np.zeros((2400, 2), dtype=np.float32), 24000, 2.0)


@pytest.mark.parametrize("speed", [0, -1, np.nan, np.inf])
def test_invalid_speed(speed):
    with pytest.raises(ValueError, match="speed"):
        time_stretch(tone(), 24000, speed)


def test_invalid_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        time_stretch(tone(), 0, 2.0)
