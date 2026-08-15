"""Regression coverage for Whisper decoding past a short input's duration."""

import wave
from types import SimpleNamespace


def test_discards_only_segments_beyond_real_audio_duration():
    from vllm_mlx.audio.stt import _discard_impossible_whisper_tail

    segments = [
        {"start": 0.0, "end": 3.82, "text": " Correct phrase."},
        {"start": 3.6, "end": 20.62, "text": " invented"},
        {"start": 29.8, "end": 29.7, "text": " tail"},
    ]

    text, kept = _discard_impossible_whisper_tail(
        "Correct phrase. invented tail", segments, 3.85
    )

    assert text == "Correct phrase."
    assert kept == segments[:1]


def test_legitimate_repetition_inside_recording_is_preserved():
    from vllm_mlx.audio.stt import _discard_impossible_whisper_tail

    segments = [
        SimpleNamespace(start=0.0, end=1.0, text=" go"),
        SimpleNamespace(start=1.0, end=2.0, text=" go"),
        SimpleNamespace(start=2.0, end=3.0, text=" go"),
    ]

    text, kept = _discard_impossible_whisper_tail("go go go", segments, 3.0)

    assert text == "go go go"
    assert kept is segments


def test_unknown_timing_shape_does_not_rewrite_backend_result():
    from vllm_mlx.audio.stt import _discard_impossible_whisper_tail

    segments = [{"text": " backend-specific segment"}]
    text, kept = _discard_impossible_whisper_tail("backend result", segments, 1.0)

    assert text == "backend result"
    assert kept is segments


def test_all_impossible_segments_do_not_erase_transcript():
    from vllm_mlx.audio.stt import _discard_impossible_whisper_tail

    segments = [{"start": 20.0, "end": 21.0, "text": " backend result"}]
    text, kept = _discard_impossible_whisper_tail("backend result", segments, 1.0)

    assert text == "backend result"
    assert kept is segments


def test_wav_duration_does_not_require_optional_soundfile(tmp_path):
    from vllm_mlx.audio.stt import _audio_duration_seconds

    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\0\0" * 8_000)

    assert _audio_duration_seconds(str(path)) == 0.5
