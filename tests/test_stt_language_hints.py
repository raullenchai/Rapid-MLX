import io
import sys
import types
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.audio.stt import STTEngine


def wav(seconds=1, noise=False):
    out = io.BytesIO()
    with wave.open(out, "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        samples = (
            np.random.default_rng(2).integers(-20, 20, seconds * 16000, dtype=np.int16)
            if noise
            else np.zeros(seconds * 16000, dtype=np.int16)
        )
        f.writeframes(samples.tobytes())
    return out.getvalue()


class Backend:
    def __init__(self, model_type="", prompts=None, sentences=None):
        self.config = SimpleNamespace(
            model_type=model_type, support_languages=["Chinese", "English", "Japanese"]
        )
        if prompts is not None:
            self.prompt_dictionary = prompts
        self.calls = []
        self.sentences = sentences

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if args and isinstance(args[0], np.ndarray):
            assert len(args[0]) <= 60 * 16000
        return SimpleNamespace(
            text="hello",
            language="en",
            sentences=self.sentences,
            items=[SimpleNamespace(text="hello", start_time=0.0, end_time=1.0)],
        )


@pytest.fixture
def route(monkeypatch):
    from rapid_mlx.audio import probe
    from rapid_mlx.config import get_config
    from rapid_mlx.routes import audio

    monkeypatch.setattr(probe, "require_mlx_audio_stt", lambda: None)
    server = types.ModuleType("rapid_mlx.server")
    server._emit_audio_model_served_once = lambda *args: None
    monkeypatch.setitem(sys.modules, "rapid_mlx.server", server)
    package = types.ModuleType("mlx_audio")
    package.__path__ = []
    stt = types.ModuleType("mlx_audio.stt")
    stt.__path__ = []
    utils = types.ModuleType("mlx_audio.stt.utils")
    utils.load_audio = lambda path, sr: np.zeros(0, dtype=np.float32)
    package.stt = stt
    stt.utils = utils
    monkeypatch.setitem(sys.modules, "mlx_audio", package)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils)
    from rapid_mlx.runtime import audio_worker

    async def run(_lane, _model, _operation, func, *args, **kwargs):
        return func(*args, **kwargs)

    def run_sync(_lane, _model, _operation, func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(audio_worker, "run_audio_mlx", run)
    monkeypatch.setattr(audio_worker, "run_audio_mlx_sync", run_sync)
    backend = Backend()

    def load(engine):
        engine.model = backend
        engine._loaded = True
        engine._enable_vad_pretrim = False

    monkeypatch.setattr(STTEngine, "load", load)
    monkeypatch.setattr(
        "rapid_mlx.utils.model_file_guard.checked_stt_loader", lambda loader: loader
    )
    cfg = get_config()
    monkeypatch.setattr(cfg, "api_key", None)
    audio._stt_engine = audio._aligner_engine = None
    app = FastAPI()
    app.include_router(audio.router)
    with TestClient(app) as client:
        yield client, backend
    audio._stt_engine = audio._aligner_engine = None


def post(client, model, content=None, **data):
    return client.post(
        "/v1/audio/transcriptions",
        data={"model": model, **data},
        files={"file": ("clip.wav", content or wav(), "audio/wav")},
    )


@pytest.mark.parametrize(
    "model,language,expected",
    [
        ("nemotron-asr", "zh", "zh-CN"),
        ("nemotron-asr", "ja", "ja-JP"),
        ("nemotron-asr", "zh-TW", "zh-TW"),
        ("nemotron-asr", "en", "en"),
        ("qwen3-asr", "zh", "Chinese"),
        ("qwen3-asr", "ja", "Japanese"),
        ("qwen3-asr", "Chinese", "Chinese"),
        ("whisper-base", "zh", "zh"),
        ("qwen3-aligner", "zh", "Chinese"),
    ],
)
def test_language(route, model, language, expected):
    client, backend = route
    if model == "nemotron-asr":
        backend.prompt_dictionary = {
            "en-US": 0,
            "en": 1,
            "zh-CN": 2,
            "zh-TW": 3,
            "ja-JP": 4,
        }
    data = {"language": language}
    if model == "qwen3-aligner":
        data["text"] = "hello"
    response = post(client, model, **data)
    assert response.status_code == 200, response.text
    assert backend.calls[0][1]["language"] == expected


@pytest.mark.parametrize("model", ["parakeet", "nemotron-asr"])
@pytest.mark.parametrize("fmt", ["srt", "verbose_json"])
def test_sentence_timing(route, model, fmt):
    client, backend = route
    backend.sentences = [
        SimpleNamespace(text="One", start=0.1, end=0.4),
        SimpleNamespace(text="Two", start=0.5, end=0.9),
    ]
    response = post(client, model, response_format=fmt)
    assert response.status_code == 200, response.text
    if fmt == "srt":
        assert "00:00:00,100 --> 00:00:00,400" in response.text
        assert "00:00:00,500 --> 00:00:00,900" in response.text
    else:
        assert [(x["start"], x["end"]) for x in response.json()["segments"]] == [
            (0.1, 0.4),
            (0.5, 0.9),
        ]


@pytest.mark.parametrize("language", ["fr", None])
def test_granite_language(route, language):
    client, backend = route
    backend.config.model_type = "granite_speech5_ctc"
    data = {"response_format": "verbose_json"}
    if language:
        data["language"] = language
    response = post(client, "granite-speech", **data)
    assert response.status_code == 200, response.text
    assert response.json().get("language") == language


def test_long_firered(route, monkeypatch):
    client, backend = route
    backend.config.model_type = "fireredasr2"
    monkeypatch.setattr(
        "mlx_audio.stt.utils.load_audio",
        lambda path, sr: np.ones(240 * sr, dtype=np.float32) * 0.001,
    )
    response = post(
        client, "firered-asr2", wav(240, noise=True), response_format="verbose_json"
    )
    assert response.status_code == 200, response.text
    segments = response.json()["segments"]
    assert len(segments) >= 4
    assert segments[0]["start"] == 0 and segments[-1]["end"] == 240
    assert all(a["end"] == b["start"] for a, b in zip(segments, segments[1:]))
    assert all(len(call[0][0]) <= 60 * 16000 for call in backend.calls)


def test_firered_cuts_at_quiet_frame_and_joins_cjk():
    from rapid_mlx.audio.stt import _transcribe_firered_chunks

    audio = np.ones(65 * 16000, dtype=np.float32)
    audio[57 * 16000 : 57 * 16000 + 320] = 0

    class Model:
        def generate(self, samples, **kwargs):
            return SimpleNamespace(text="你好")

    text, segments = _transcribe_firered_chunks(Model(), audio, {})
    assert segments[0]["end"] == pytest.approx(57.02)
    assert segments[1]["start"] == segments[0]["end"]
    assert text == "你好你好"
