# SPDX-License-Identifier: Apache-2.0
"""Forced-alignment STT (Qwen3-ForcedAligner) — engine + HTTP route.

Forced alignment is the inverse of ASR: given audio AND the KNOWN
transcript, the model returns per-unit (per Chinese character for zh)
start/end times with zero recognition error. It is the primitive behind
per-character karaoke captions and beat-synced editing.

This suite covers:

* ``STTEngine._is_aligner`` detection (aligner vs whisper/parakeet).
* ``STTEngine.transcribe`` rejecting aligner models (no text to
  recognize) and ``STTEngine.align`` rejecting non-aligner models /
  empty text.
* ``STTEngine.align`` segment-shaping against a stubbed model that
  mirrors ``mlx_audio``'s ``ForcedAlignResult`` (``.items`` of
  ``ForcedAlignItem(text, start_time, end_time)`` + a ``.text``
  property) — verified against the installed library's call surface.
* The ``/v1/audio/transcriptions`` ``text`` extension: an aligner model
  + ``text`` routes to ``align`` and returns per-character segments in
  the verbose_json/srt serializer shapes; the two incoherent
  combinations 400; and omitting ``text`` still transcribes.

All tests stub the model/engine so no weights are downloaded.
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

pytestmark = pytest.mark.requires_mlx
from fastapi import FastAPI
from fastapi.testclient import TestClient

_ALIGNER_ID = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"


# ---------------------------------------------------------------------------
# Engine-level tests — no HTTP, no weights. The model object is stubbed so
# ``align()`` never touches mlx_audio.
# ---------------------------------------------------------------------------


class _FakeAlignItem:
    """Mirror of ``mlx_audio ... ForcedAlignItem`` (frozen fields)."""

    def __init__(self, text: str, start_time: float, end_time: float):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


class _FakeAlignResult:
    """Mirror of ``ForcedAlignResult``: ``.items`` + a joined ``.text``."""

    def __init__(self, items: list[_FakeAlignItem]):
        self.items = items

    @property
    def text(self) -> str:
        return " ".join(it.text for it in self.items)


class _RecordingAlignerModel:
    """Stub ``model`` whose ``generate`` returns a single result.

    Records the kwargs so the test can assert the engine calls the
    library's ``generate(audio=..., text=..., language=...)`` contract
    (single string audio -> single ForcedAlignResult).
    """

    def __init__(self, items: list[_FakeAlignItem]):
        self._items = items
        self.calls: list[dict] = []

    def generate(self, audio, text, language):
        self.calls.append({"audio": audio, "text": text, "language": language})
        return _FakeAlignResult(self._items)


def _make_aligner_engine(items: list[_FakeAlignItem]):
    """Return a loaded ``STTEngine`` bound to the aligner id + stub model."""
    from vllm_mlx.audio.stt import STTEngine

    eng = STTEngine(_ALIGNER_ID)
    eng._loaded = True  # bypass real weight load
    eng.model = _RecordingAlignerModel(items)
    return eng


class TestAlignerDetection:
    def test_aligner_id_detected(self):
        from vllm_mlx.audio.stt import STTEngine

        assert STTEngine(_ALIGNER_ID)._is_aligner is True
        # The short aliases resolve to an id containing "ForcedAligner".
        assert STTEngine("some/Qwen3-ForcedAligner-thing")._is_aligner is True

    def test_non_aligner_not_detected(self):
        from vllm_mlx.audio.stt import STTEngine

        assert STTEngine("mlx-community/whisper-large-v3-mlx")._is_aligner is False
        assert STTEngine("mlx-community/parakeet-tdt-0.6b-v2")._is_aligner is False


class TestTranscribeRejectsAligner:
    def test_transcribe_on_aligner_raises(self):
        from vllm_mlx.audio.stt import STTEngine

        eng = STTEngine(_ALIGNER_ID)
        eng._loaded = True  # skip load; the aligner guard fires first
        eng.model = object()
        with pytest.raises(ValueError, match="forced-alignment model"):
            eng.transcribe("/tmp/whatever.wav")

    def test_transcribe_guard_fires_before_load(self):
        # Codex MAJOR regression: the aligner guard must reject BEFORE
        # load() so an invalid call never downloads gigabytes of weights.
        from vllm_mlx.audio.stt import STTEngine

        eng = STTEngine(_ALIGNER_ID)  # _loaded is False

        def _boom():
            raise AssertionError("load() must not run for an invalid call")

        eng.load = _boom
        with pytest.raises(ValueError, match="forced-alignment model"):
            eng.transcribe("/tmp/whatever.wav")


class TestAlignGuards:
    def test_align_on_non_aligner_raises(self):
        from vllm_mlx.audio.stt import STTEngine

        eng = STTEngine("mlx-community/whisper-large-v3-mlx")
        eng._loaded = True
        eng.model = object()
        with pytest.raises(ValueError, match="requires a forced-aligner model"):
            eng.align("/tmp/a.wav", text="hello")

    def test_align_empty_text_raises(self):
        eng = _make_aligner_engine([])
        with pytest.raises(ValueError, match="non-empty known text"):
            eng.align("/tmp/a.wav", text="   ")

    def test_align_guards_fire_before_load(self):
        # Codex MAJOR regression: both the model-kind and empty-text
        # guards must reject BEFORE load() (no weight download on an
        # invalid call).
        from vllm_mlx.audio.stt import STTEngine

        def _boom():
            raise AssertionError("load() must not run for an invalid call")

        non_aligner = STTEngine("mlx-community/whisper-large-v3-mlx")
        non_aligner.load = _boom
        with pytest.raises(ValueError, match="requires a forced-aligner model"):
            non_aligner.align("/tmp/a.wav", text="hello")

        aligner = STTEngine(_ALIGNER_ID)
        aligner.load = _boom
        with pytest.raises(ValueError, match="non-empty known text"):
            aligner.align("/tmp/a.wav", text="  ")


class TestAlignSegmentShaping:
    def test_segments_are_per_unit_dicts(self):
        items = [
            _FakeAlignItem("你", 0.0, 0.5),
            _FakeAlignItem("好", 0.5, 1.1),
            _FakeAlignItem("世", 1.1, 1.6),
            _FakeAlignItem("界", 1.6, 2.0),
        ]
        eng = _make_aligner_engine(items)
        res = eng.align("/tmp/clip.wav", text="你好世界", language="Chinese")

        assert res.segments == [
            {"text": "你", "start": 0.0, "end": 0.5},
            {"text": "好", "start": 0.5, "end": 1.1},
            {"text": "世", "start": 1.1, "end": 1.6},
            {"text": "界", "start": 1.6, "end": 2.0},
        ]
        # Duration is the last unit's end; language passes through.
        assert res.duration == 2.0
        assert res.language == "Chinese"
        # ``text`` comes from the result (joined units) when present.
        assert res.text == "你 好 世 界"

    def test_generate_called_with_library_contract(self):
        eng = _make_aligner_engine([_FakeAlignItem("a", 0.0, 0.2)])
        eng.align("/tmp/clip.wav", text="a", language="English")
        assert eng.model.calls == [
            {"audio": "/tmp/clip.wav", "text": "a", "language": "English"}
        ]

    def test_start_end_coerced_to_float(self):
        # Library rounds to ms floats, but a stub returning ints must not
        # leak ints into the segment shape the serializers consume.
        eng = _make_aligner_engine([_FakeAlignItem("x", 0, 1)])
        res = eng.align("/tmp/clip.wav", text="x")
        seg = res.segments[0]
        assert isinstance(seg["start"], float) and isinstance(seg["end"], float)

    def test_empty_items_yield_empty_segments(self):
        eng = _make_aligner_engine([])
        res = eng.align("/tmp/clip.wav", text="anything")
        assert res.segments == []
        assert res.duration is None
        # Falls back to the supplied text when the result carries none.
        assert res.text == "anything"


# ---------------------------------------------------------------------------
# HTTP route tests — /v1/audio/transcriptions ``text`` -> forced alignment.
# The engine is fully stubbed (no weights, no mlx_audio load).
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


class _FakeTranscribeResult:
    text = "hello world goodbye world"
    language = "en"
    duration = 4.5
    segments = [
        {"start": 0.0, "end": 2.0, "text": "hello world"},
        {"start": 2.5, "end": 4.5, "text": "goodbye world"},
    ]


class _FakeAlignmentResult:
    text = "你好世界"
    language = "Chinese"
    duration = 2.0
    segments = [
        {"text": "你", "start": 0.0, "end": 0.5},
        {"text": "好", "start": 0.5, "end": 1.0},
        {"text": "世", "start": 1.0, "end": 1.5},
        {"text": "界", "start": 1.5, "end": 2.0},
    ]


# Module-level capture of the args the route hands to ``align`` — the
# fake engine instances are created inside the route and not otherwise
# reachable from the test.
_ALIGN_CALLS: list[dict] = []


class _FakeRouteEngine:
    """Mirrors the ``STTEngine`` surface the route depends on: both
    ``transcribe`` and ``align``. Returns distinguishable results so the
    test can tell which path the route chose from the response body."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def load(self):
        pass

    def transcribe(self, audio_path, language=None, task="transcribe"):
        return _FakeTranscribeResult()

    def align(self, audio_path, text, language="Chinese"):
        _ALIGN_CALLS.append({"text": text, "language": language})
        return _FakeAlignmentResult()


@pytest.fixture
def _stub_route_engine(monkeypatch):
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
    _ALIGN_CALLS.clear()

    monkeypatch.setattr("vllm_mlx.audio.stt.STTEngine", _FakeRouteEngine, raising=False)
    audio_stt_mod = sys.modules.get("vllm_mlx.audio.stt")
    if audio_stt_mod is not None:
        monkeypatch.setattr(audio_stt_mod, "STTEngine", _FakeRouteEngine)

    audio_route._stt_engine = None
    # The alignment lane keeps its OWN engine cache (see the module
    # comment on ``_aligner_engine``), so both must be cleared — leaving
    # the stub in ``_aligner_engine`` would leak into later tests.
    audio_route._aligner_engine = None
    yield
    audio_route._stt_engine = None
    audio_route._aligner_engine = None
    probe._reset_probe_cache()


def _mount_audio_app():
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
        files={"file": ("clip.wav", _make_tone_wav(), "audio/wav")},
    )


class TestAlignmentRoute:
    def test_verbose_json_returns_per_char_segments(self, _stub_route_engine):
        client, restore = _mount_audio_app()
        try:
            r = _post(
                client,
                {
                    "model": "qwen3-aligner",
                    "text": "你好世界",
                    "language": "Chinese",
                    "response_format": "verbose_json",
                },
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        body = r.json()
        # Alignment path chosen (distinct text from the transcribe stub).
        assert body["text"] == "你好世界"
        assert body["language"] == "Chinese"
        assert body["duration"] == 2.0
        assert len(body["segments"]) == 4, body["segments"]
        first = body["segments"][0]
        assert first["start"] == 0.0
        assert first["end"] == 0.5
        assert first["text"] == "你"

    def test_segment_granularity_keeps_forced_alignment_path(self, _stub_route_engine):
        """Rebased word-timestamp plumbing must not reroute aligner requests."""
        client, restore = _mount_audio_app()
        try:
            r = _post(
                client,
                {
                    "model": "qwen3-aligner",
                    "text": "你好世界",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                },
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["text"] == "你好世界"
        assert len(body["segments"]) == 4
        assert "words" not in body
        assert _ALIGN_CALLS[-1]["text"] == "你好世界"

    def test_word_granularity_rejects_forced_aligner(self, _stub_route_engine):
        client, restore = _mount_audio_app()
        try:
            r = _post(
                client,
                {
                    "model": "qwen3-aligner",
                    "text": "你好世界",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "word",
                },
            )
        finally:
            restore()
        assert r.status_code == 400, r.text
        error = r.json()["detail"]["error"]
        assert error["code"] == "invalid_model_for_word_timestamps"
        assert error["param"] == "timestamp_granularities"

    def test_srt_renders_character_cues(self, _stub_route_engine):
        client, restore = _mount_audio_app()
        try:
            r = _post(
                client,
                {
                    "model": "qwen3-aligner",
                    "text": "你好世界",
                    "response_format": "srt",
                },
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        body = r.text
        assert "00:00:00,000 --> 00:00:00,500" in body, body
        assert "你" in body and "界" in body

    def test_aligner_without_text_400(self, _stub_route_engine):
        client, restore = _mount_audio_app()
        try:
            r = _post(client, {"model": "qwen3-aligner"})
        finally:
            restore()
        assert r.status_code == 400, r.text
        err = r.json()["detail"]["error"]
        assert err["type"] == "invalid_request_error"
        assert err["code"] == "alignment_text_required"
        assert err["param"] == "text"

    def test_aligner_blank_text_400(self, _stub_route_engine):
        client, restore = _mount_audio_app()
        try:
            r = _post(client, {"model": "qwen3-aligner", "text": "   "})
        finally:
            restore()
        assert r.status_code == 400, r.text
        assert r.json()["detail"]["error"]["code"] == "alignment_text_required"

    def test_text_without_aligner_400(self, _stub_route_engine):
        client, restore = _mount_audio_app()
        try:
            r = _post(client, {"model": "whisper-large-v3", "text": "hi there"})
        finally:
            restore()
        assert r.status_code == 400, r.text
        err = r.json()["detail"]["error"]
        assert err["code"] == "alignment_model_required"
        assert err["param"] == "model"

    def test_route_passes_unstripped_text_to_align(self, _stub_route_engine):
        # Codex MINOR regression: the route must align the ORIGINAL
        # transcript, using strip only to decide the text was non-empty.
        client, restore = _mount_audio_app()
        try:
            r = _post(
                client,
                {"model": "qwen3-aligner", "text": "  你好  ", "language": "Chinese"},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert _ALIGN_CALLS, "align() was not called"
        assert _ALIGN_CALLS[-1]["text"] == "  你好  ", _ALIGN_CALLS
        assert _ALIGN_CALLS[-1]["language"] == "Chinese"

    def test_no_text_still_transcribes(self, _stub_route_engine):
        # Backward-compat: a normal transcription request (no text) must
        # keep hitting the transcribe path unchanged.
        client, restore = _mount_audio_app()
        try:
            r = _post(client, {"model": "whisper-large-v3"})
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert r.json()["text"] == "hello world goodbye world"
