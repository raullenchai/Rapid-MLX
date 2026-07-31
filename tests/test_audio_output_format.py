from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from pydantic import ValidationError

from vllm_mlx.api.models import AudioMusicRequest, AudioSpeechRequest
from vllm_mlx.audio.output_format import convert_audio_output
from vllm_mlx.routes.audio import _convert_music_wav


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as output:
        output.setnchannels(1 if audio.ndim == 1 else audio.shape[1])
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(audio.astype("<i2").tobytes())
    return payload.getvalue()


def test_tts_mono_can_be_resampled_and_expanded_to_stereo() -> None:
    source = np.sin(2 * np.pi * 440 * np.arange(24_000) / 24_000).astype(np.float32)

    converted, sample_rate, channels = convert_audio_output(
        source,
        24_000,
        sample_rate=44_100,
        channels=2,
    )

    assert sample_rate == 44_100
    assert channels == 2
    assert converted.shape == (44_100, 2)
    np.testing.assert_array_equal(converted[:, 0], converted[:, 1])
    assert np.max(np.abs(converted)) > 0.9
    spectrum = np.abs(np.fft.rfft(converted[:, 0]))
    peak_hz = np.fft.rfftfreq(len(converted), d=1 / sample_rate)[spectrum.argmax()]
    assert peak_hz == pytest.approx(440, abs=1)


def test_music_stereo_can_be_resampled_and_downmixed_to_mono() -> None:
    left = np.full(44_100, 8_000, dtype=np.int16)
    right = np.full(44_100, -4_000, dtype=np.int16)
    source = np.column_stack((left, right))

    payload, sample_rate, channels = _convert_music_wav(
        _wav_bytes(source, 44_100),
        24_000,
        1,
    )

    with wave.open(io.BytesIO(payload), "rb") as result:
        assert result.getframerate() == 24_000
        assert result.getnchannels() == 1
        assert result.getnframes() == 24_000
        samples = np.frombuffer(result.readframes(result.getnframes()), dtype="<i2")
    assert samples.mean() == pytest.approx(2_000, abs=20)
    assert samples.std() < 20
    assert sample_rate == 24_000
    assert channels == 1


def test_omitted_output_format_preserves_native_audio() -> None:
    source = np.full((100, 2), 32_000, dtype=np.int16)

    converted, sample_rate, channels = convert_audio_output(source, 44_100)

    assert converted is source
    assert converted.dtype == np.int16
    np.testing.assert_array_equal(converted, source)
    assert sample_rate == 44_100
    assert channels == 2


def test_music_omitted_output_format_preserves_wav_bytes() -> None:
    source = _wav_bytes(np.zeros((100, 2), dtype=np.int16), 44_100)

    converted, sample_rate, channels = _convert_music_wav(source, None, None)

    assert converted is source
    assert sample_rate == 44_100
    assert channels == 2


def test_requested_conversion_normalizes_int16_pcm() -> None:
    source = np.array([0, 16_384, -16_384, 32_767, -32_768], dtype=np.int16)

    converted, sample_rate, channels = convert_audio_output(source, 24_000, channels=2)

    assert sample_rate == 24_000
    assert channels == 2
    np.testing.assert_allclose(
        converted[:, 0],
        [0.0, 0.5, -0.5, 32_767 / 32_768, -1.0],
        atol=1e-6,
    )
    np.testing.assert_array_equal(converted[:, 0], converted[:, 1])


@pytest.mark.parametrize(
    ("source", "channels"),
    [
        (np.zeros((1, 1), dtype=np.float32), 1),
        (np.zeros((1, 2), dtype=np.float32), 2),
        (np.zeros((2, 1), dtype=np.float32), 1),
        (np.zeros((2, 2), dtype=np.float32), 2),
    ],
)
def test_short_sample_first_audio_has_unambiguous_channels(source, channels) -> None:
    converted, _, detected_channels = convert_audio_output(source, 24_000)

    assert converted is source
    assert detected_channels == channels


@pytest.mark.parametrize(
    ("request_type", "payload"),
    [
        (AudioSpeechRequest, {"input": "hello", "sample_rate": 7_999}),
        (AudioSpeechRequest, {"input": "hello", "sample_rate": True}),
        (AudioSpeechRequest, {"input": "hello", "channels": 3}),
        (AudioMusicRequest, {"input": "music", "sample_rate": 96_001}),
        (AudioMusicRequest, {"input": "music", "sample_rate": "44100"}),
        (AudioMusicRequest, {"input": "music", "channels": 0}),
        (
            AudioSpeechRequest,
            {"input": "hello", "response_format": "mp3", "sample_rate": 96_000},
        ),
        (
            AudioSpeechRequest,
            {"input": "hello", "response_format": "opus", "sample_rate": 44_100},
        ),
    ],
)
def test_audio_output_format_rejects_invalid_values(request_type, payload) -> None:
    with pytest.raises(ValidationError):
        request_type(**payload)
