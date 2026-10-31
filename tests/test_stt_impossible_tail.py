"""Regression coverage for Whisper decoding past a short input's duration."""

import importlib
import os
import sys
import wave
from types import ModuleType, SimpleNamespace

import numpy as np
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
    from rapid_mlx.audio import stt

    path = tmp_path / "audio.m4a"
    path.write_bytes(b"not a wav")
    soundfile = ModuleType("soundfile")
    soundfile.info = lambda _: (_ for _ in ()).throw(RuntimeError())
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    monkeypatch.setattr(stt.shutil, "which", lambda name: "/usr/bin/ffprobe")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="2.75\n")

    monkeypatch.setattr(stt.subprocess, "run", run)
    try:
        from mlx_audio.stt import utils
    except ImportError:  # no-MLX hosts cannot decode at all
        utils = None
    if utils is not None:
        monkeypatch.setattr(
            utils, "load_audio", lambda _: pytest.fail("unexpected decode")
        )
    assert stt._audio_duration_seconds(str(path)) == 2.75
    assert calls[0][0] == [
        "/usr/bin/ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        "-protocol_whitelist",
        "file",
        "file:" + os.path.abspath(path),
    ]
    assert calls[0][1]["timeout"] == 5


def test_ffprobe_unavailable_duration_returns_none(monkeypatch, tmp_path):
    from rapid_mlx.audio import stt

    path = tmp_path / "audio.m4a"
    path.write_bytes(b"not a wav")
    soundfile = ModuleType("soundfile")
    soundfile.info = lambda _: (_ for _ in ()).throw(RuntimeError())
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    monkeypatch.setattr(stt.shutil, "which", lambda _: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        stt.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="N/A")
    )

    assert stt._audio_duration_seconds(str(path), allow_decode=False) is None


@pytest.fixture
def fake_audio_loader(monkeypatch):
    """Provide a decoder without requiring the optional MLX audio stack."""
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    for name in ("mlx_audio", "mlx_audio.stt"):
        try:
            importlib.import_module(name)
        except ImportError:
            package = ModuleType(name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, name, package)
    module = ModuleType("mlx_audio.stt.utils")
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", module)
    monkeypatch.delenv("RAPID_MLX_STT_VAD_PRETRIM", raising=False)
    return module


@pytest.mark.parametrize(
    ("length", "amplitude", "skipped"),
    [(0, 0.0, False), (8000, 0.0, False), (16000, 0.1, True)],
)
def test_vad_unavailable_uses_energy_fallback(
    monkeypatch, fake_audio_loader, length, amplitude, skipped
):
    from rapid_mlx.audio import stt

    waveform = np.full(length, amplitude, dtype=np.float32)
    fake_audio_loader.load_audio = lambda _: waveform
    monkeypatch.setattr(stt, "_get_vad_model", lambda: None)

    result = stt._maybe_vad_trim("audio.wav")

    assert result.skipped is skipped
    if not skipped:
        assert result.has_speech is False
    assert result.source_duration == len(waveform) / stt._VAD_SAMPLE_RATE


@pytest.mark.parametrize("case", ["empty", "exception", "degenerate"])
def test_vad_edge_cases_preserve_source_duration(monkeypatch, fake_audio_loader, case):
    from rapid_mlx.audio import stt

    waveform = np.zeros(0 if case == "empty" else 16000, dtype=np.float32)
    fake_audio_loader.load_audio = lambda _: waveform

    def timestamps(*args, **kwargs):
        if case == "exception":
            raise RuntimeError("VAD failure")
        return [{"start": 2.0, "end": 3.0}]

    monkeypatch.setattr(
        stt, "_get_vad_model", lambda: SimpleNamespace(get_speech_timestamps=timestamps)
    )
    result = stt._maybe_vad_trim("audio.wav")

    assert result.skipped is (case == "exception")
    if case != "exception":
        assert result.has_speech is False
    assert result.source_duration == len(waveform) / stt._VAD_SAMPLE_RATE


def test_duration_decode_fallback_can_be_disabled(monkeypatch, tmp_path):
    from rapid_mlx.audio import stt

    path = str(tmp_path / "missing.m4a")
    soundfile = ModuleType("soundfile")
    soundfile.info = lambda _: (_ for _ in ()).throw(RuntimeError("probe failed"))
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    monkeypatch.setattr(stt.shutil, "which", lambda _: None)
    decoded = []
    monkeypatch.setattr(
        stt,
        "_decoded_audio_duration",
        lambda audio_path: decoded.append(audio_path) or 1.5,
    )

    assert stt._audio_duration_seconds(path, allow_decode=False) is None
    assert decoded == []
    assert stt._audio_duration_seconds(path, allow_decode=True) == 1.5
    assert decoded == [path]


def test_cleanup_temp_audio_logs_permission_error(monkeypatch, caplog):
    from rapid_mlx.routes import audio

    calls = []

    def unlink(path):
        calls.append(path)
        raise PermissionError("access denied")

    monkeypatch.setattr(audio.os, "unlink", unlink)
    with caplog.at_level("WARNING", logger=audio.logger.name):
        audio._cleanup_temp_audio(None, "audio.wav", None)

    assert calls == ["audio.wav"]
    assert "Failed to unlink temp audio file audio.wav: access denied" in caplog.text


def test_cleanup_temp_audio_ignores_missing_file(monkeypatch, caplog):
    from rapid_mlx.routes import audio

    def unlink(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(audio.os, "unlink", unlink)
    with caplog.at_level("WARNING", logger=audio.logger.name):
        audio._cleanup_temp_audio("gone.wav")

    assert "Failed to unlink" not in caplog.text
