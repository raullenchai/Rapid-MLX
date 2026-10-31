"""Regression coverage for Whisper decoding past a short input's duration."""

import os
import wave
from types import SimpleNamespace

import pytest


def test_discards_only_segments_beyond_real_audio_duration():
    from rapid_mlx.audio.stt import _discard_impossible_whisper_tail

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
    from rapid_mlx.audio.stt import _discard_impossible_whisper_tail

    segments = [
        SimpleNamespace(start=0.0, end=1.0, text=" go"),
        SimpleNamespace(start=1.0, end=2.0, text=" go"),
        SimpleNamespace(start=2.0, end=3.0, text=" go"),
    ]

    text, kept = _discard_impossible_whisper_tail("go go go", segments, 3.0)

    assert text == "go go go"
    assert kept is segments


def test_unknown_timing_shape_does_not_rewrite_backend_result():
    from rapid_mlx.audio.stt import _discard_impossible_whisper_tail

    segments = [{"text": " backend-specific segment"}]
    text, kept = _discard_impossible_whisper_tail("backend result", segments, 1.0)

    assert text == "backend result"
    assert kept is segments


def test_all_impossible_segments_do_not_erase_transcript():
    from rapid_mlx.audio.stt import _discard_impossible_whisper_tail

    segments = [{"start": 20.0, "end": 21.0, "text": " backend result"}]
    text, kept = _discard_impossible_whisper_tail("backend result", segments, 1.0)

    assert text == "backend result"
    assert kept is segments


def test_wav_duration_does_not_require_optional_soundfile(tmp_path):
    from rapid_mlx.audio.stt import _audio_duration_seconds

    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\0\0" * 8_000)

    assert _audio_duration_seconds(str(path)) == 0.5


def test_m4a_duration_uses_ffprobe_before_audio_decode(monkeypatch, tmp_path):
    import soundfile
    from mlx_audio.stt import utils

    from rapid_mlx.audio import stt

    path = tmp_path / "audio.m4a"
    path.write_bytes(b"not a wav")
    monkeypatch.setattr(
        soundfile, "info", lambda _: (_ for _ in ()).throw(RuntimeError())
    )
    monkeypatch.setattr(stt.shutil, "which", lambda name: "/usr/bin/ffprobe")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="2.75\n")

    monkeypatch.setattr(stt.subprocess, "run", run)
    monkeypatch.setattr(utils, "load_audio", lambda _: pytest.fail("unexpected decode"))
    assert stt._audio_duration_seconds(str(path)) == 2.75
    assert calls[0][0][-1] == "file:" + os.path.abspath(path)
    assert calls[0][1]["timeout"] == 5
