# SPDX-License-Identifier: Apache-2.0
"""STT-word-timestamps — word-level timestamps on ``/v1/audio/transcriptions``.

OpenAI's Whisper API accepts ``timestamp_granularities[]`` (``word`` /
``segment``) on the transcription request and, for ``response_format=
verbose_json``, returns a top-level ``words`` array of
``{word, start, end}`` when word granularity is requested. Pre-feature the
route parsed ``model`` / ``language`` / ``response_format`` from the
multipart form but silently dropped ``timestamp_granularities[]``, so
word-level captions were impossible.

These tests pin the new contract without downloading weights:

* the request Pydantic model accepts the field;
* ``_normalise_timestamp_granularities`` validates + de-dups values;
* ``_iter_words_for_verbose`` / ``_build_verbose_json_body`` emit the
  OpenAI word shape only when requested and omit it otherwise;
* ``STTEngine.transcribe`` forwards ``word_timestamps=True`` to a Whisper
  backend only when ``word`` is requested, and never forwards the flag to
  a non-Whisper (Parakeet) backend (which would raise);
* the full route path (TestClient + a stubbed engine) returns a non-empty,
  monotonic ``words`` array for ``verbose_json`` + ``word``.
"""

from __future__ import annotations

import importlib.machinery
import io
import math
import struct
import sys
import types
import wave

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_tone_wav(duration_s: float = 0.25, freq_hz: float = 440.0) -> bytes:
    sample_rate = 16000
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for i in range(n_samples):
            sample = int(8000 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
            w.writeframes(struct.pack("<h", sample))
    return buf.getvalue()


class _FakeWordResult:
    """Whisper-shaped result: segments carry a ``words`` list of the
    mlx-audio ``{word, start, end, probability}`` dict shape."""

    text = "hello world"
    language = "en"
    duration = 1.4
    segments = [
        {
            "start": 0.0,
            "end": 0.7,
            "text": "hello",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5, "probability": 0.99},
            ],
        },
        {
            "start": 0.7,
            "end": 1.4,
            "text": "world",
            "words": [
                {"word": "world", "start": 0.7, "end": 1.2, "probability": 0.98},
            ],
        },
    ]


class _FakeNoWordResult:
    """Non-Whisper-shaped result: segments have NO ``words`` list."""

    text = "hello world"
    language = "en"
    duration = 1.4
    segments = [
        {"start": 0.0, "end": 1.4, "text": "hello world"},
    ]


class _FakeEngine:
    """Mirrors the ``STTEngine`` surface the route depends on.

    Accepts the new ``timestamp_granularities`` kwarg and returns a
    word-laden result only when ``word`` is requested, so both the
    words-present and words-absent route branches are exercised
    deterministically.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def load(self):
        pass

    def transcribe(
        self, audio_path, language=None, task="transcribe", timestamp_granularities=None
    ):
        if timestamp_granularities and "word" in timestamp_granularities:
            return _FakeWordResult()
        return _FakeNoWordResult()


@pytest.fixture
def _stub_engine(monkeypatch):
    """Stub the STTEngine + mlx_audio probe so the route runs without weights."""
    from vllm_mlx.audio import probe
    from vllm_mlx.routes import audio as audio_route

    fake_mlx_audio = types.ModuleType("mlx_audio")
    fake_mlx_audio.__path__ = []
    fake_mlx_audio.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio", loader=None, is_package=True
    )
    fake_stt = types.ModuleType("mlx_audio.stt")
    fake_stt.__path__ = []
    fake_stt.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio.stt", loader=None, is_package=True
    )
    fake_stt_utils = types.ModuleType("mlx_audio.stt.utils")
    fake_stt_utils.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio.stt.utils", loader=None
    )
    fake_stt_utils.load_model = lambda *_a, **_kw: None
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", fake_stt)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", fake_stt_utils)

    probe._reset_probe_cache()

    monkeypatch.setattr("vllm_mlx.audio.stt.STTEngine", _FakeEngine, raising=False)
    audio_stt_mod = sys.modules.get("vllm_mlx.audio.stt")
    if audio_stt_mod is not None:
        monkeypatch.setattr(audio_stt_mod, "STTEngine", _FakeEngine)

    audio_route._stt_engine = None
    yield
    audio_route._stt_engine = None
    probe._reset_probe_cache()


def _mount_audio_app() -> tuple[TestClient, callable]:
    from vllm_mlx.config import get_config
    from vllm_mlx.routes import audio as audio_route

    app = FastAPI()
    app.include_router(audio_route.router)
    cfg = get_config()
    saved = cfg.api_key
    cfg.api_key = None

    def _restore():
        cfg.api_key = saved

    return TestClient(app), _restore


def _post(client, data):
    return client.post(
        "/v1/audio/transcriptions",
        data=data,
        files={"file": ("tone.wav", _make_tone_wav(), "audio/wav")},
    )


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class TestRequestModel:
    def test_accepts_timestamp_granularities(self):
        from vllm_mlx.api.models import AudioTranscriptionRequest

        req = AudioTranscriptionRequest(
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
        assert req.timestamp_granularities == ["word", "segment"]

    def test_default_is_none(self):
        from vllm_mlx.api.models import AudioTranscriptionRequest

        assert AudioTranscriptionRequest().timestamp_granularities is None


# ---------------------------------------------------------------------------
# Granularity normalisation
# ---------------------------------------------------------------------------


class TestNormaliseGranularities:
    def test_none_and_empty_resolve_to_none(self):
        from vllm_mlx.routes.audio import _normalise_timestamp_granularities

        assert _normalise_timestamp_granularities(None) is None
        assert _normalise_timestamp_granularities([]) is None

    def test_lowercases_and_dedups(self):
        from vllm_mlx.routes.audio import _normalise_timestamp_granularities

        assert _normalise_timestamp_granularities(["Word", "word", "SEGMENT"]) == [
            "word",
            "segment",
        ]

    def test_invalid_value_raises_400(self):
        from fastapi import HTTPException

        from vllm_mlx.routes.audio import _normalise_timestamp_granularities

        with pytest.raises(HTTPException) as exc:
            _normalise_timestamp_granularities(["words"])
        assert exc.value.status_code == 400
        assert exc.value.detail["error"]["param"] == "timestamp_granularities"


# ---------------------------------------------------------------------------
# verbose_json builder / word flattener
# ---------------------------------------------------------------------------


class TestVerboseJsonBuilder:
    def test_words_omitted_by_default(self):
        from vllm_mlx.routes.audio import _build_verbose_json_body

        body = _build_verbose_json_body(_FakeWordResult(), timestamp_granularities=None)
        # Default (no granularities) is unchanged: segments present, no words.
        assert "segments" in body
        assert "words" not in body

    def test_segment_only_has_no_words(self):
        from vllm_mlx.routes.audio import _build_verbose_json_body

        body = _build_verbose_json_body(
            _FakeWordResult(), timestamp_granularities=["segment"]
        )
        assert "segments" in body
        assert "words" not in body

    def test_word_only_emits_words_and_drops_segments(self):
        from vllm_mlx.routes.audio import _build_verbose_json_body

        body = _build_verbose_json_body(
            _FakeWordResult(), timestamp_granularities=["word"]
        )
        assert "segments" not in body
        assert body["words"] == [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.7, "end": 1.2},
        ]
        # Exactly the three OpenAI keys — no leaked ``probability``.
        for w in body["words"]:
            assert set(w.keys()) == {"word", "start", "end"}
            assert isinstance(w["start"], float)
            assert isinstance(w["end"], float)

    def test_both_granularities_emit_both(self):
        from vllm_mlx.routes.audio import _build_verbose_json_body

        body = _build_verbose_json_body(
            _FakeWordResult(), timestamp_granularities=["word", "segment"]
        )
        assert len(body["segments"]) == 2
        assert len(body["words"]) == 2

    def test_non_whisper_result_yields_empty_words_not_crash(self):
        from vllm_mlx.routes.audio import _build_verbose_json_body

        # A backend that produced no per-word data must degrade to an
        # empty words array, never raise.
        body = _build_verbose_json_body(
            _FakeNoWordResult(), timestamp_granularities=["word"]
        )
        assert body["words"] == []


# ---------------------------------------------------------------------------
# STTEngine forwarding
# ---------------------------------------------------------------------------


class _RecordingModel:
    """Fake mlx-audio model whose ``generate`` records the kwargs it saw."""

    def __init__(self):
        self.seen_kwargs = None

    def generate(self, audio_input, **kwargs):
        self.seen_kwargs = kwargs
        return _FakeWordResult()


class TestEngineForwarding:
    def _make_engine(self, model_name):
        from vllm_mlx.audio.stt import STTEngine

        eng = STTEngine(model_name, enable_vad_pretrim=False)
        eng._loaded = True
        eng.model = _RecordingModel()
        return eng

    def test_whisper_forwards_word_timestamps_when_word_requested(self, tmp_path):
        eng = self._make_engine("mlx-community/whisper-large-v3-turbo")
        wav = tmp_path / "a.wav"
        wav.write_bytes(_make_tone_wav())
        eng.transcribe(str(wav), timestamp_granularities=["word"])
        assert eng.model.seen_kwargs.get("word_timestamps") is True

    def test_whisper_omits_flag_without_word_granularity(self, tmp_path):
        eng = self._make_engine("mlx-community/whisper-large-v3-turbo")
        wav = tmp_path / "a.wav"
        wav.write_bytes(_make_tone_wav())
        eng.transcribe(str(wav), timestamp_granularities=["segment"])
        assert "word_timestamps" not in eng.model.seen_kwargs

    def test_whisper_uses_deterministic_greedy_decode(self, tmp_path):
        eng = self._make_engine("mlx-community/whisper-large-v3-turbo")
        wav = tmp_path / "a.wav"
        wav.write_bytes(_make_tone_wav())
        eng.transcribe(str(wav))
        assert eng.model.seen_kwargs["temperature"] == 0.0

    def test_parakeet_never_gets_word_timestamps(self, tmp_path):
        # Parakeet's generate() has no word_timestamps kwarg; forwarding it
        # would raise. The engine must omit it even when word is requested.
        eng = self._make_engine("mlx-community/parakeet-tdt-0.6b-v2")
        wav = tmp_path / "a.wav"
        wav.write_bytes(_make_tone_wav())
        result = eng.transcribe(str(wav), timestamp_granularities=["word"])
        assert "word_timestamps" not in eng.model.seen_kwargs
        assert result.text  # did not crash


# ---------------------------------------------------------------------------
# Full route (TestClient) — hermetic
# ---------------------------------------------------------------------------


class TestRouteWordTimestamps:
    def test_default_verbose_json_has_no_words(self, _stub_engine):
        client, restore = _mount_audio_app()
        try:
            r = _post(
                client, {"model": "whisper-large-v3", "response_format": "verbose_json"}
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        body = r.json()
        assert "segments" in body
        assert "words" not in body

    def test_word_granularity_emits_monotonic_words(self, _stub_engine):
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={
                    "model": "whisper-large-v3",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "word",
                },
                files={"file": ("tone.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        body = r.json()
        words = body.get("words")
        assert isinstance(words, list) and len(words) >= 1, body
        prev_end = -1.0
        for w in words:
            assert set(w.keys()) == {"word", "start", "end"}
            assert w["start"] <= w["end"]
            assert w["start"] >= prev_end - 1e-6
            prev_end = w["end"]

    def test_invalid_granularity_returns_400(self, _stub_engine):
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={
                    "model": "whisper-large-v3",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "words",
                },
                files={"file": ("tone.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        assert r.status_code == 400, r.text

    def test_granularity_without_verbose_json_is_400(self, _stub_engine):
        # OpenAI contract: timestamp_granularities[] requires verbose_json.
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={
                    "model": "whisper-large-v3",
                    "response_format": "json",
                    "timestamp_granularities[]": "word",
                },
                files={"file": ("tone.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        assert r.status_code == 400, r.text
        assert r.json()["detail"]["error"]["param"] == "timestamp_granularities"

    def test_word_granularity_on_non_whisper_is_400(self, _stub_engine):
        # Word timings are Whisper-only; a non-Whisper model must 400 rather
        # than return an empty words[] that falsely claims fulfillment.
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={
                    "model": "parakeet",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "word",
                },
                files={"file": ("tone.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        assert r.status_code == 400, r.text
        assert (
            r.json()["detail"]["error"]["code"] == "invalid_model_for_word_timestamps"
        )

    def test_segment_granularity_on_non_whisper_is_ok(self, _stub_engine):
        # segment-level timestamps work on every engine — no rejection.
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={
                    "model": "parakeet",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                },
                files={"file": ("tone.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        body = r.json()
        assert "segments" in body
        assert "words" not in body


# ---------------------------------------------------------------------------
# Request-model validation + malformed-word robustness
# ---------------------------------------------------------------------------


class TestRequestModelValidation:
    def test_rejects_unknown_value(self):
        from pydantic import ValidationError

        from vllm_mlx.api.models import AudioTranscriptionRequest

        with pytest.raises(ValidationError):
            AudioTranscriptionRequest(timestamp_granularities=["frame"])

    def test_normalises_and_dedups(self):
        from vllm_mlx.api.models import AudioTranscriptionRequest

        req = AudioTranscriptionRequest(
            timestamp_granularities=["Word", "word", "SEGMENT"]
        )
        assert req.timestamp_granularities == ["word", "segment"]


class TestMalformedWordTimings:
    def test_non_finite_and_non_numeric_words_dropped(self):
        from vllm_mlx.routes.audio import _iter_words_for_verbose

        class _Result:
            segments = [
                {
                    "words": [
                        {"word": "ok", "start": 0.0, "end": 0.5},
                        {"word": "nan", "start": float("nan"), "end": 1.0},
                        {"word": "inf", "start": 1.0, "end": float("inf")},
                        {"word": "bad", "start": "x", "end": 2.0},
                        {"word": "missing", "start": None, "end": None},
                    ]
                }
            ]

        words = _iter_words_for_verbose(_Result())
        # Only the clean word survives. NaN / inf / non-numeric / missing
        # timings are all dropped — never fabricated into a 0.0 timestamp.
        rendered = [w["word"] for w in words]
        assert rendered == ["ok"], rendered
        for w in words:
            assert math.isfinite(w["start"]) and math.isfinite(w["end"])
