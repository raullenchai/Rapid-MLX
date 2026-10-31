# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for input duration, verbose segments and AIFF uploads."""

import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.requires_mlx


@pytest.fixture
def _stub_engine(monkeypatch):
    from rapid_mlx.audio import probe, stt
    from rapid_mlx.config import get_config
    from rapid_mlx.routes import audio

    class FakeEngine:
        def __init__(self, model_name):
            self.model_name = model_name

        def load(self):
            pass

    monkeypatch.setattr(stt, "STTEngine", FakeEngine)
    monkeypatch.setattr(probe, "require_mlx_audio_stt", lambda: None)
    monkeypatch.setattr(audio, "_stt_engine", None)
    monkeypatch.setattr(get_config(), "api_key", None)


@pytest.fixture
def client(_stub_engine):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.routes.audio import router

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as test_client:
        yield test_client


def _audio_bytes(fmt):
    samples = 0.2 * np.sin(2 * np.pi * 440 * np.arange(24000) / 16000)
    stream = io.BytesIO()
    sf.write(stream, samples, 16000, format=fmt, subtype="PCM_16")
    return stream.getvalue()


@pytest.mark.parametrize("segments", [None, []])
def test_duration_without_segments(tmp_path, segments):
    from rapid_mlx.audio.stt import STTEngine
    from rapid_mlx.routes.audio import _build_verbose_json_body

    path = tmp_path / "tone.wav"
    path.write_bytes(_audio_bytes("WAV"))
    engine = STTEngine("mlx-community/parakeet-tdt-0.6b-v2")
    engine._loaded = True
    engine.model = SimpleNamespace(
        generate=lambda *a, **kw: SimpleNamespace(
            text="hello", language="en", segments=segments
        )
    )
    result = engine.transcribe(path)
    assert result.duration == pytest.approx(1.5)
    body = _build_verbose_json_body(result)
    assert body["duration"] == pytest.approx(1.5)
    assert body["segments"][0]["start"] == 0.0
    assert body["segments"][0]["end"] == pytest.approx(1.5)


def test_verbose_preserves_finite_segment_metadata():
    from rapid_mlx.routes.audio import _build_verbose_json_body

    result = SimpleNamespace(
        text="hello",
        language="en",
        duration=1,
        segments=[
            dict(
                start=0,
                end=1,
                text="hello",
                seek=12,
                tokens=[1, 2],
                temperature=0.0,
                avg_logprob=float("nan"),
                compression_ratio=1.2,
                no_speech_prob=0.01,
            )
        ],
    )
    body = _build_verbose_json_body(result)
    seg = body["segments"][0]
    assert (seg["seek"], seg["tokens"], seg["temperature"]) == (12, [1, 2], 0.0)
    assert seg["compression_ratio"] == 1.2
    assert seg["no_speech_prob"] == 0.01
    assert "avg_logprob" not in seg
    assert isinstance(body["duration"], float)


@pytest.mark.parametrize("endpoint", ["transcriptions", "translations"])
def test_aiff_route(client, monkeypatch, endpoint):
    from rapid_mlx.audio import stt

    seen = []

    def transcribe(self, audio_path, language=None, task="transcribe"):
        assert audio_path.endswith(".wav")
        info = sf.info(audio_path)
        assert info.format == "WAV" and info.subtype == "PCM_16"
        seen.append((audio_path, task))
        return SimpleNamespace(
            text="hello", language="en", duration=info.duration, segments=None
        )

    monkeypatch.setattr(stt.STTEngine, "transcribe", transcribe, raising=False)
    response = client.post(
        f"/v1/audio/{endpoint}",
        data={"model": "whisper-large-v3", "response_format": "verbose_json"},
        files={"file": ("tone.aiff", _audio_bytes("AIFF"), "audio/aiff")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body["duration"], float)
    assert body["segments"][0]["end"] == pytest.approx(body["duration"])
    assert seen[0][1] == ("translate" if endpoint == "translations" else "transcribe")
    from pathlib import Path

    assert not Path(seen[0][0]).exists()


def test_garbage_upload_reports_supported_formats(client, monkeypatch):
    from rapid_mlx.audio import stt

    def transcribe(self, audio_path, language=None, task="transcribe"):
        sf.info(audio_path)

    monkeypatch.setattr(stt.STTEngine, "transcribe", transcribe, raising=False)
    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-large-v3"},
        files={"file": ("bad.wav", b"garbage", "audio/wav")},
    )
    assert response.status_code == 400
    message = response.json()["detail"]["error"]["message"]
    assert "Supported formats:" in message and "aiff" in message


def test_non_whisper_duration_falls_back_to_segments_when_decode_fails(
    tmp_path, monkeypatch
):
    from mlx_audio.stt import utils

    from rapid_mlx.audio import stt

    path = tmp_path / "unknown.audio"
    path.write_bytes(b"not a wave file")
    monkeypatch.setattr(sf, "info", lambda _: (_ for _ in ()).throw(ValueError()))
    monkeypatch.setattr(stt.shutil, "which", lambda _: None)

    def failed_decode(_):
        raise ValueError("decode unavailable")

    monkeypatch.setattr(utils, "load_audio", failed_decode)
    engine = stt.STTEngine("mlx-community/parakeet-tdt-0.6b-v2")
    engine._loaded = True
    engine.model = SimpleNamespace(
        generate=lambda *a, **kw: SimpleNamespace(
            text="hello", language="en", segments=[{"end": 1.25}]
        )
    )
    assert engine.transcribe(path).duration == 1.25


@pytest.mark.parametrize("no_speech", [False, True])
def test_decoded_duration_precedes_segments_and_vad_return(
    tmp_path, monkeypatch, no_speech
):
    from rapid_mlx.audio import stt

    path = tmp_path / "unknown.audio"
    path.write_bytes(b"audio")
    calls = []

    def duration_probe(path, *, allow_decode):
        calls.append(allow_decode)
        return 2.5 if allow_decode else None

    monkeypatch.setattr(stt, "_audio_duration_seconds", duration_probe)
    engine = stt.STTEngine("whisper-small")
    engine._loaded = True
    engine._enable_vad_pretrim = no_speech
    engine.model = SimpleNamespace(
        generate=lambda *a, **kw: SimpleNamespace(
            text="hello", language="en", segments=[{"start": 0, "end": 1.0}]
        )
    )
    if no_speech:
        monkeypatch.setattr(
            stt,
            "_maybe_vad_trim",
            lambda _: stt._VADTrimResult(skipped=False, has_speech=False),
        )
        engine.model.generate = lambda *a, **kw: pytest.fail("backend must not run")
    result = engine.transcribe(path)
    assert result.duration == 2.5
    assert calls == [False, True]
    if no_speech:
        assert result.segments == []


def test_duration_ffprobe_uses_local_file_protocol(monkeypatch):
    import wave

    from rapid_mlx.audio import stt

    def fail_probe(*args, **kwargs):
        raise ValueError("metadata unavailable")

    captured = []

    def run(argv, **kwargs):
        captured.extend(argv)
        return SimpleNamespace(stdout="1.5")

    monkeypatch.setattr(wave, "open", fail_probe)
    monkeypatch.setattr(sf, "info", fail_probe)
    monkeypatch.setattr(stt.shutil, "which", lambda _: "/usr/bin/ffprobe")
    monkeypatch.setattr(stt.subprocess, "run", run)

    assert stt._audio_duration_seconds("-upload") == 1.5
    index = captured.index("-protocol_whitelist")
    assert captured[index + 1] == "file"
    assert captured[-1].startswith("file:/")


def test_non_whisper_duration_decodes_when_no_timed_segments(tmp_path, monkeypatch):
    import wave

    from mlx_audio.stt import utils

    from rapid_mlx.audio import stt
    from rapid_mlx.routes.audio import _build_verbose_json_body

    path = tmp_path / "unknown.audio"
    path.write_bytes(b"audio")

    def fail_probe(*args, **kwargs):
        raise ValueError("metadata unavailable")

    monkeypatch.setattr(wave, "open", fail_probe)
    monkeypatch.setattr(sf, "info", fail_probe)
    monkeypatch.setattr(stt.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        utils, "load_audio", lambda _: np.zeros(int(2.5 * stt._VAD_SAMPLE_RATE))
    )
    engine = stt.STTEngine("mlx-community/parakeet-tdt-0.6b-v2")
    engine._loaded = True
    engine.model = SimpleNamespace(
        generate=lambda *a, **kw: SimpleNamespace(text="hello", language="en")
    )
    result = engine.transcribe(path)
    assert result.duration == 2.5
    body = _build_verbose_json_body(result)
    assert body["duration"] == 2.5
    assert body["segments"][0]["end"] == 2.5


def test_aiff_expansion_limit_cleans_up(client, monkeypatch):
    from pathlib import Path

    from rapid_mlx.routes import audio

    monkeypatch.setattr(audio, "MAX_AUDIO_UPLOAD_SIZE", 128)
    destinations = []
    transcode = audio._transcode_aiff_to_wav

    def capture(source, destination):
        destinations.append(destination)
        transcode(source, destination)

    class OversizedAudio:
        frames = 235
        channels = 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(audio, "_transcode_aiff_to_wav", capture)
    monkeypatch.setattr(sf, "SoundFile", lambda *args, **kwargs: OversizedAudio())
    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-large-v3"},
        files={"file": ("compressed.aifc", b"FORM\x00\x00\x00\x04AIFC", "audio/aiff")},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "Audio upload too large: exceeds 128 bytes"
    assert len(destinations) == 1
    assert not Path(destinations[0]).exists()
