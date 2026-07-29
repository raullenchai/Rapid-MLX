# SPDX-License-Identifier: Apache-2.0
"""Content-farm OpenAI-style routes: music, forced-alignment, video.

Hermetic — no real models / network / weights. Engines are faked at the
import boundary via monkeypatch (per repo convention), so these run fast
in CI:

* ``POST /v1/audio/music`` → ``MusicEngine`` (LIVE) — happy path + input
  validation.
* ``POST /v1/audio/transcriptions`` with a ``text`` field → forced
  alignment via ``STTEngine.align`` (LIVE) — happy path + the
  non-aligner-model 400.
* ``POST /v1/video/generations`` (CONTRACT-ONLY) — schema validates then
  the route returns a clean 501; a schema violation 422s at the boundary.
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


def _mount_audio_app() -> tuple[TestClient, callable]:
    """Mount the audio router on a bare FastAPI app, bypassing auth."""
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


def _mount_video_app() -> tuple[TestClient, callable]:
    """Mount the video router on a bare FastAPI app, bypassing auth."""
    from vllm_mlx.config import get_config
    from vllm_mlx.routes import video as video_route

    app = FastAPI()
    app.include_router(video_route.router)
    cfg = get_config()
    saved = cfg.api_key
    cfg.api_key = None

    def _restore():
        cfg.api_key = saved

    return TestClient(app), _restore


def _install_fake_mlx_audio(monkeypatch):
    """Make the STT-lane probe (``require_mlx_audio_stt``) pass without a
    real ``mlx_audio`` install."""
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


# ---------------------------------------------------------------------------
# Route 1: POST /v1/audio/music  (LIVE — MusicEngine)
# ---------------------------------------------------------------------------


class _FakeMusicEngine:
    """Stands in for ``vllm_mlx.audio.music.MusicEngine``.

    Records the call and writes a tiny WAV to ``out_path`` so the route's
    read-back-and-return path is exercised without SA3 weights.
    """

    last_call: dict | None = None

    def __init__(self, dit="medium", decoder="same-l"):
        self.dit = dit
        self.decoder = decoder

    def generate(
        self,
        prompt,
        out_path,
        seconds=30.0,
        steps=8,
        negative_prompt=None,
        seed=None,
        timeout=900,
    ):
        _FakeMusicEngine.last_call = {
            "prompt": prompt,
            "out_path": str(out_path),
            "seconds": seconds,
            "steps": steps,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "dit": self.dit,
            "decoder": self.decoder,
        }
        with open(out_path, "wb") as fh:
            fh.write(_make_tone_wav())
        return out_path


@pytest.fixture
def _stub_music_engine(monkeypatch):
    from vllm_mlx.routes import audio as audio_route

    _FakeMusicEngine.last_call = None
    monkeypatch.setattr(
        "vllm_mlx.audio.music.MusicEngine", _FakeMusicEngine, raising=False
    )
    music_mod = sys.modules.get("vllm_mlx.audio.music")
    if music_mod is not None:
        monkeypatch.setattr(music_mod, "MusicEngine", _FakeMusicEngine)
    audio_route._music_engine = None
    yield
    audio_route._music_engine = None


class TestMusicRoute:
    def test_happy_path_returns_wav(self, _stub_music_engine):
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/music",
                json={
                    "model": "medium",
                    "input": "epic cinematic war drums",
                    "seconds": 12,
                    "steps": 6,
                    "negative_prompt": "vocals",
                    "seed": 7,
                },
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("audio/wav"), r.headers
        # Real WAV bytes came back (RIFF header).
        assert r.content[:4] == b"RIFF", r.content[:16]
        # Request fields reached the engine, and model→(dit,decoder) mapped.
        call = _FakeMusicEngine.last_call
        assert call["prompt"] == "epic cinematic war drums"
        assert call["seconds"] == 12
        assert call["steps"] == 6
        assert call["negative_prompt"] == "vocals"
        assert call["seed"] == 7
        assert (call["dit"], call["decoder"]) == ("medium", "same-l")

    def test_model_alias_selects_small_variant(self, _stub_music_engine):
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/music",
                json={"model": "sm-music", "input": "lofi beat", "seconds": 5},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        call = _FakeMusicEngine.last_call
        assert (call["dit"], call["decoder"]) == ("sm-music", "same-s")

    def test_unknown_model_falls_back_to_defaults(self, _stub_music_engine):
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/music",
                json={"model": "no-such-variant", "input": "ambient pad"},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        call = _FakeMusicEngine.last_call
        assert (call["dit"], call["decoder"]) == ("medium", "same-l")
        # seconds default honoured.
        assert call["seconds"] == 30.0

    def test_blank_input_is_422(self, _stub_music_engine):
        client, restore = _mount_audio_app()
        try:
            r = client.post("/v1/audio/music", json={"input": "   ", "seconds": 5})
        finally:
            restore()
        assert r.status_code == 422, r.text
        # Engine was never invoked.
        assert _FakeMusicEngine.last_call is None

    def test_seconds_over_ceiling_is_422(self, _stub_music_engine):
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/music", json={"input": "long track", "seconds": 100}
            )
        finally:
            restore()
        assert r.status_code == 422, r.text
        assert _FakeMusicEngine.last_call is None

    def test_bad_response_format_is_422(self, _stub_music_engine):
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/music",
                json={"input": "track", "response_format": "mp3"},
            )
        finally:
            restore()
        assert r.status_code == 422, r.text
        assert _FakeMusicEngine.last_call is None


# ---------------------------------------------------------------------------
# Route 2: POST /v1/audio/transcriptions + text  (LIVE — forced alignment)
# ---------------------------------------------------------------------------


class _FakeAlignResult:
    def __init__(self, text, segments, language, duration):
        self.text = text
        self.segments = segments
        self.language = language
        self.duration = duration


class _FakeAlignerEngine:
    """Mirrors the ``STTEngine`` surface the alignment path touches."""

    last_call: dict | None = None

    def __init__(self, model_name):
        self.model_name = model_name

    def load(self):
        pass

    def align(self, audio_path, text, language="Chinese"):
        _FakeAlignerEngine.last_call = {
            "audio_path": str(audio_path),
            "text": text,
            "language": language,
        }
        segments = [
            {"text": ch, "start": round(i * 0.2, 3), "end": round((i + 1) * 0.2, 3)}
            for i, ch in enumerate(text)
        ]
        duration = segments[-1]["end"] if segments else 0.0
        return _FakeAlignResult(text, segments, language, duration)

    # Present so a mis-routed request would blow up loudly rather than
    # silently ASR — the route must NOT call this on the alignment path.
    def transcribe(
        self, audio_path, language=None, task="transcribe"
    ):  # pragma: no cover
        raise AssertionError("alignment path must not call transcribe()")


class _FakeNonAlignerEngine:
    """An ASR engine whose ``align`` refuses — mirrors the real
    ``STTEngine.align`` ValueError when the model isn't an aligner."""

    def __init__(self, model_name):
        self.model_name = model_name

    def load(self):
        pass

    def align(self, audio_path, text, language="Chinese"):
        raise ValueError(
            f"align() requires a forced-aligner model; {self.model_name!r} "
            "is not one (use a registry alias like 'qwen3-aligner')."
        )


@pytest.fixture
def _stub_aligner(monkeypatch):
    from vllm_mlx.audio import probe
    from vllm_mlx.routes import audio as audio_route

    _install_fake_mlx_audio(monkeypatch)
    probe._reset_probe_cache()

    monkeypatch.setattr(
        "vllm_mlx.audio.stt.STTEngine", _FakeAlignerEngine, raising=False
    )
    stt_mod = sys.modules.get("vllm_mlx.audio.stt")
    if stt_mod is not None:
        monkeypatch.setattr(stt_mod, "STTEngine", _FakeAlignerEngine)
    _FakeAlignerEngine.last_call = None
    audio_route._stt_engine = None
    yield
    audio_route._stt_engine = None
    probe._reset_probe_cache()


@pytest.fixture
def _stub_non_aligner(monkeypatch):
    from vllm_mlx.audio import probe
    from vllm_mlx.routes import audio as audio_route

    _install_fake_mlx_audio(monkeypatch)
    probe._reset_probe_cache()

    monkeypatch.setattr(
        "vllm_mlx.audio.stt.STTEngine", _FakeNonAlignerEngine, raising=False
    )
    stt_mod = sys.modules.get("vllm_mlx.audio.stt")
    if stt_mod is not None:
        monkeypatch.setattr(stt_mod, "STTEngine", _FakeNonAlignerEngine)
    audio_route._stt_engine = None
    yield
    audio_route._stt_engine = None
    probe._reset_probe_cache()


class TestForcedAlignment:
    def test_text_field_routes_to_alignment_verbose_json(self, _stub_aligner):
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"text": "临终前", "language": "Chinese"},
                files={"file": ("clip.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("application/json")
        body = r.json()
        # Defaults to verbose_json → segments present with per-char timing.
        assert body["text"] == "临终前"
        assert isinstance(body["segments"], list)
        assert len(body["segments"]) == 3
        assert body["segments"][0]["text"] == "临"
        assert body["segments"][0]["start"] == 0.0
        assert body["segments"][0]["end"] == 0.2
        # The known transcript was forwarded as an INPUT (no recognition),
        # and the aligner default model was picked (model omitted).
        call = _FakeAlignerEngine.last_call
        assert call["text"] == "临终前"
        assert call["language"] == "Chinese"

    def test_alignment_honors_srt_response_format(self, _stub_aligner):
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"text": "abc", "response_format": "srt"},
                files={"file": ("clip.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        ctype = r.headers["content-type"]
        assert ctype.startswith("text/srt") or ctype.startswith("text/plain")
        body = r.text
        assert "00:00:00,000 --> 00:00:00,200" in body, body
        assert "a" in body

    def test_no_text_field_is_still_asr(self, _stub_aligner):
        """Absent ``text`` → the alignment branch must NOT fire (the fake
        aligner's transcribe() asserts if reached, so a clean non-500
        here proves the ASR branch was taken)."""
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"model": "whisper-large-v3"},
                files={"file": ("clip.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        # transcribe() raises AssertionError → generic 500 envelope. The
        # point is the alignment path did NOT run (align() not called).
        assert _FakeAlignerEngine.last_call is None
        assert r.status_code == 500, r.text

    def test_non_aligner_model_with_text_is_400(self, _stub_non_aligner):
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"model": "whisper-large-v3", "text": "hello"},
                files={"file": ("clip.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        assert r.status_code == 400, r.text
        body = r.json()
        err = body.get("detail", {}).get("error") or body.get("error")
        assert err is not None, body
        assert err["type"] == "invalid_request_error", err
        assert err["code"] == "invalid_alignment_request", err


# ---------------------------------------------------------------------------
# Route 3: POST /v1/video/generations  (CONTRACT-ONLY → 501)
# ---------------------------------------------------------------------------


class TestVideoContract:
    def test_valid_request_returns_501(self):
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json={
                    "model": "ltx-2.3",
                    "prompt": "a red fox trotting through snow",
                    "height": 704,
                    "width": 1216,
                    "num_frames": 97,
                    "frame_rate": 25,
                },
            )
        finally:
            restore()
        assert r.status_code == 501, r.text
        body = r.json()
        err = body.get("detail", {}).get("error") or body.get("error")
        assert err is not None, body
        assert err["type"] == "not_implemented_error", err
        assert err["code"] == "video_backend_not_implemented", err
        assert "REQUIREMENTS_rapid.md B1" in err["message"], err

    def test_image_to_video_request_also_501(self):
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json={
                    "prompt": "animate this",
                    "image": "data:image/png;base64,iVBORw0KGgo=",
                },
            )
        finally:
            restore()
        assert r.status_code == 501, r.text

    def test_missing_prompt_is_422(self):
        client, restore = _mount_video_app()
        try:
            r = client.post("/v1/video/generations", json={"model": "ltx-2.3"})
        finally:
            restore()
        assert r.status_code == 422, r.text

    def test_zero_frames_is_422(self):
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json={"prompt": "x", "num_frames": 0},
            )
        finally:
            restore()
        assert r.status_code == 422, r.text

    def test_schema_round_trips(self):
        """The request/response models colleagues integrate against must
        validate a full payload and serialize the response envelope."""
        from vllm_mlx.api.models import (
            VideoGenerationRequest,
            VideoGenerationResponse,
            VideoGenerationResult,
        )

        req = VideoGenerationRequest(
            prompt="hello", image="https://example.com/frame.png", seed=1
        )
        assert req.num_frames == 97
        assert req.response_format == "mp4"

        resp = VideoGenerationResponse(
            created=123,
            model="ltx-2.3",
            data=[VideoGenerationResult(url="/tmp/out.mp4", width=1216, height=704)],
        )
        dumped = resp.model_dump()
        assert dumped["data"][0]["url"] == "/tmp/out.mp4"
        assert dumped["data"][0]["format"] == "mp4"
