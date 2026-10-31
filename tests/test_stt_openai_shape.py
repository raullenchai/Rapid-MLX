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
    assert seen[0][1] == (
        "translate" if endpoint == "translations" else "transcribe"
    )
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
