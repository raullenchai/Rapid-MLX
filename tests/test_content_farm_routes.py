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
from pydantic import ValidationError

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


def _mount_video_app(with_handlers: bool = False) -> tuple[TestClient, callable]:
    """Mount the video router on a bare FastAPI app, bypassing auth.

    ``with_handlers=True`` additionally installs the server's global
    exception handlers, which is what turns a schema rejection into the
    documented 400 instead of stock FastAPI's 422 — see
    :meth:`TestVideoContract.test_schema_rejection_is_400_on_the_real_app`.
    """
    from vllm_mlx.config import get_config
    from vllm_mlx.routes import video as video_route

    app = FastAPI()
    app.include_router(video_route.router)
    if with_handlers:
        from vllm_mlx.middleware.exception_handlers import (
            install_exception_handlers,
        )

        install_exception_handlers(app)
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

    def test_over_long_input_is_422_not_argv_blowup(self, _stub_music_engine):
        """A giant prompt must 422 at the schema, not reach the engine.

        ``MusicEngine.generate`` passes ``input`` as an argv element to
        the vendored SA3 CLI, so an unbounded prompt hits the OS
        ``ARG_MAX`` ceiling and surfaces as an opaque 500 ``E2BIG``. The
        ``max_length`` bound turns that into an actionable 422.
        """
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/music",
                json={"input": "x" * 5000, "seconds": 5},
            )
        finally:
            restore()
        assert r.status_code == 422, r.text
        assert _FakeMusicEngine.last_call is None

    def test_engine_writing_nothing_is_500_not_empty_200(self, monkeypatch):
        """An engine that exits cleanly without writing must 500.

        The SA3 CLI can return success having produced no audio (a sampler
        that bailed after the header). Returning that as HTTP 200 with an
        empty ``audio/wav`` body reads to the caller as a successfully
        generated silent clip — the worst kind of failure. Assert we fail
        loudly with the documented ``music_generation_failed`` envelope.
        """

        class _SilentEngine(_FakeMusicEngine):
            def generate(self, prompt, out_path, **kwargs):  # noqa: D102
                # Exits "successfully" but leaves the temp file at 0 bytes.
                return out_path

        from vllm_mlx.routes import audio as audio_route

        monkeypatch.setattr(
            "vllm_mlx.audio.music.MusicEngine", _SilentEngine, raising=False
        )
        music_mod = sys.modules.get("vllm_mlx.audio.music")
        if music_mod is not None:
            monkeypatch.setattr(music_mod, "MusicEngine", _SilentEngine)
        audio_route._music_engine = None
        client, restore = _mount_audio_app()
        try:
            r = client.post("/v1/audio/music", json={"input": "silence", "seconds": 5})
        finally:
            restore()
            audio_route._music_engine = None
        assert r.status_code == 500, r.text
        assert r.json()["detail"]["error"]["code"] == "music_generation_failed", r.text

    def test_generation_does_not_block_the_event_loop(self, _stub_music_engine):
        """The blocking render must run off the event loop.

        ``MusicEngine.generate`` shells out and waits (up to 900 s by
        default). Calling it inline from the ``async def`` handler would
        stall every other request on the server for the whole render, so
        the handler hands it to ``asyncio.to_thread``. Detect that
        structurally: the engine must observe a DIFFERENT thread than the
        one running the event loop.
        """
        import asyncio
        import threading

        observed: dict[str, int] = {}

        class _ThreadRecordingEngine(_FakeMusicEngine):
            def generate(self, prompt, out_path, **kwargs):  # noqa: D102
                observed["engine_thread"] = threading.get_ident()
                with open(out_path, "wb") as fh:
                    fh.write(_make_tone_wav())
                return out_path

        from vllm_mlx.api.models import AudioMusicRequest
        from vllm_mlx.routes import audio as audio_route

        async def _drive():
            observed["loop_thread"] = threading.get_ident()
            return await audio_route.create_music(
                AudioMusicRequest(input="a march", seconds=5)
            )

        saved = audio_route._music_engine
        try:
            music_mod = sys.modules["vllm_mlx.audio.music"]
            saved_cls = music_mod.MusicEngine
            music_mod.MusicEngine = _ThreadRecordingEngine
            audio_route._music_engine = None
            try:
                resp = asyncio.run(_drive())
            finally:
                music_mod.MusicEngine = saved_cls
        finally:
            audio_route._music_engine = saved

        assert resp.body[:4] == b"RIFF", resp.body[:16]
        assert observed["engine_thread"] != observed["loop_thread"], observed


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

    #: Set when ``transcribe`` runs, so a test can assert the ASR branch
    #: was taken AND that it produced a real result — not merely that
    #: ``align`` was skipped.
    last_transcribe: dict | None = None

    def transcribe(self, audio_path, language=None, task="transcribe"):
        _FakeAlignerEngine.last_transcribe = {
            "audio_path": str(audio_path),
            "language": language,
            "task": task,
        }
        return _FakeAlignResult(
            "recognised speech",
            [{"text": "recognised speech", "start": 0.0, "end": 1.0}],
            language or "en",
            1.0,
        )


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
    _FakeAlignerEngine.last_transcribe = None
    audio_route._stt_engine = None
    audio_route._aligner_engine = None
    yield
    audio_route._stt_engine = None
    audio_route._aligner_engine = None
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
    audio_route._aligner_engine = None
    yield
    audio_route._stt_engine = None
    audio_route._aligner_engine = None
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
        """Absent ``text`` → unchanged ASR, positively verified.

        This asserts the ASR branch actually SUCCEEDS and returns the
        recognised text, not merely that ``align()`` was skipped. An
        earlier version expected a 500 (the fake's ``transcribe`` raised),
        which would have stayed green even if the whole ASR path were
        broken — it only ever proved "not alignment".
        """
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"model": "whisper-large-v3", "response_format": "verbose_json"},
                files={"file": ("clip.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert r.json()["text"] == "recognised speech", r.text
        # ASR ran; the alignment branch did not.
        assert _FakeAlignerEngine.last_transcribe is not None
        assert _FakeAlignerEngine.last_call is None

    def test_blank_text_is_400_not_a_silent_asr_fallback(self, _stub_aligner):
        """A present-but-blank ``text`` must 400, never quietly run ASR.

        Sending ``text`` says "I have the transcript, give me timings".
        Pre-fix a whitespace-only value fell through to speech
        recognition and returned 200 — answering a different question
        with no signal that the request had been reinterpreted.
        """
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"text": "   "},
                files={"file": ("clip.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        assert r.status_code == 400, r.text
        body = r.json()
        err = body.get("detail", {}).get("error") or body.get("error")
        assert err["code"] == "invalid_alignment_request", err
        assert err["param"] == "text", err
        # Neither engine path ran.
        assert _FakeAlignerEngine.last_call is None
        assert _FakeAlignerEngine.last_transcribe is None

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

    def test_omitted_model_resolves_to_the_registered_aligner(self, _stub_aligner):
        """No ``model`` → the ALIGNER alias, not the ASR default.

        ``whisper-large-v3`` is not a forced aligner, so defaulting the
        alignment branch to it would fail deep in ``STTEngine.align``.
        Assert the resolved repo is actually the aligner.
        """
        from vllm_mlx.routes import audio as audio_route

        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"text": "abc"},
                files={"file": ("clip.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        # The aligner lives in its own cache, deliberately NOT the shared
        # ``_stt_engine`` the ASR lane mutates from the event loop.
        assert audio_route._stt_engine is None, audio_route._stt_engine
        assert "ForcedAligner" in audio_route._aligner_engine.model_name, (
            audio_route._aligner_engine.model_name
        )

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_model_takes_the_aligner_default(self, _stub_aligner, blank):
        """A BLANK ``model`` must behave as omitted, not error.

        ``""`` is already absorbed by FastAPI (it coerces an empty form
        field to ``None`` for an ``Optional[str]``) — kept here to pin
        that boundary. ``"   "`` is the one that actually leaked: it
        arrives verbatim and pre-fix reached ``_resolve_stt_model`` as an
        explicit choice, 404-ing as a nonexistent alias, so "just send
        audio + text" from a form with a spaced-out field never aligned.
        """
        from vllm_mlx.routes import audio as audio_route

        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"text": "abc", "model": blank},
                files={"file": ("clip.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        # The aligner lives in its own cache, deliberately NOT the shared
        # ``_stt_engine`` the ASR lane mutates from the event loop.
        assert audio_route._stt_engine is None, audio_route._stt_engine
        assert "ForcedAligner" in audio_route._aligner_engine.model_name, (
            audio_route._aligner_engine.model_name
        )

    def test_blank_response_format_takes_verbose_json(self, _stub_aligner):
        """A whitespace-only ``response_format`` must fall back to verbose_json.

        Same shape as the ``model`` case above: pre-fix ``"   "`` counted
        as an explicit choice, reached the allowed-set check and 400'd —
        losing the timestamps the caller came for.
        """
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"text": "abc", "response_format": "   "},
                files={"file": ("clip.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert isinstance(r.json()["segments"], list), r.text

    def test_alignment_does_not_block_the_event_loop(self, _stub_aligner):
        """Weight load + align must run off the event loop.

        Both are seconds of blocking compute; inline in the ``async def``
        handler they stall every concurrent request on the server. Detect
        structurally — the engine must see a different thread than the
        loop.
        """
        import asyncio
        import threading

        from vllm_mlx.routes import audio as audio_route

        seen: dict[str, int] = {}
        real_align = _FakeAlignerEngine.align

        def _recording_align(self, audio_path, text, language="Chinese"):
            seen["engine_thread"] = threading.get_ident()
            return real_align(self, audio_path, text, language=language)

        async def _drive():
            seen["loop_thread"] = threading.get_ident()
            return await audio_route._run_alignment_request(
                file=_UploadLike(_make_tone_wav()),
                model="qwen3-aligner",
                text="abc",
                language=None,
                response_format="verbose_json",
            )

        _FakeAlignerEngine.align = _recording_align
        try:
            asyncio.run(_drive())
        finally:
            _FakeAlignerEngine.align = real_align
        assert seen["engine_thread"] != seen["loop_thread"], seen


class _UploadLike:
    """Minimal stand-in for ``UploadFile`` — only ``read`` is exercised by
    ``_stream_upload_to_tempfile``."""

    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)

    async def read(self, size: int = -1) -> bytes:
        return self._buf.read(size)


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
        assert "docs/content_farm_api.md" in err["message"], err

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

    @pytest.mark.parametrize(
        "body",
        [
            {"model": "ltx-2.3"},  # prompt missing
            {"prompt": "   "},  # blank prompt
            {"prompt": "x", "num_frames": 0},
            {"prompt": "x", "frame_rate": "NaN"},
            {"prompt": "x", "response_format": "webm"},
        ],
    )
    def test_schema_rejection_is_400_on_the_real_app(self, body):
        """Schema rejections reach the caller as 400, not FastAPI's 422.

        The bare-app tests above see 422 because they mount the router
        without the server's handlers. On the REAL server
        ``install_exception_handlers`` normalizes every
        ``RequestValidationError`` to a sanitized 400 — verified against a
        live ``rapid-mlx serve`` — so that is the number
        ``docs/content_farm_api.md`` documents and clients must code
        against. Pinned here so the two can't drift apart again.
        """
        client, restore = _mount_video_app(with_handlers=True)
        try:
            r = client.post("/v1/video/generations", json=body)
        finally:
            restore()
        assert r.status_code == 400, r.text

    @pytest.mark.parametrize(
        "image",
        [
            "file:///etc/passwd",
            "FILE:///etc/passwd",
            # Single-slash URIs are still URIs. urlsplit() sees scheme
            # "file" here, but a "://" substring test does not — that gap
            # would wave an arbitrary local-file read through as "bare
            # base64". Regression pin for codex round-2 finding 1.
            "file:/etc/passwd",
            "file:/../../etc/shadow",
            "gopher:/internal/x",
            "gopher://internal/x",
            "ftp://internal/frame.png",
            "data:text/html;base64,PHNjcmlwdD4=",
            "data:;base64,AAAA",
            "   ",
            # Scheme-less but not base64 — a bare path or host must not
            # sit in the field waiting for a backend to interpret it.
            "/etc/passwd",
            "../../secret.png",
            "internal.corp.example",
        ],
    )
    def test_unsafe_image_reference_is_rejected(self, image):
        """``image`` is the only field a backend DEREFERENCES.

        Left unconstrained it is a local-file-read (``file://``) and SSRF
        primitive the instant ``resolve_video_engine`` stops raising.
        Validate at the schema boundary so every future backend inherits
        the restriction rather than each having to remember it.
        """
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json={"prompt": "animate", "image": image},
            )
        finally:
            restore()
        assert r.status_code == 422, r.text

    @pytest.mark.parametrize(
        "image",
        [
            "data:image/png;base64,iVBORw0KGgo=",
            "https://example.com/frame.png",
            "http://example.com/frame.png",
            # Bare base64, no scheme — a real (truncated) PNG payload.
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGP4DwABAQEAGn0nsQAAAABJRU5ErkJggg==",
        ],
    )
    def test_documented_image_forms_are_accepted(self, image):
        """The three forms docs/content_farm_api.md advertises must pass
        validation (and then hit the contract-only 501, not a 422)."""
        client, restore = _mount_video_app()
        try:
            r = client.post(
                "/v1/video/generations",
                json={"prompt": "animate", "image": image},
            )
        finally:
            restore()
        assert r.status_code == 501, r.text

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

        # XOR invariant: both channels set, or neither, must be refused.
        with pytest.raises(ValidationError):
            VideoGenerationResult(b64_video="AAAA", url="https://x/y.mp4")
        with pytest.raises(ValidationError):
            VideoGenerationResult()

        resp = VideoGenerationResponse(
            created=123,
            model="ltx-2.3",
            data=[VideoGenerationResult(url="/tmp/out.mp4", width=1216, height=704)],
        )
        dumped = resp.model_dump()
        assert dumped["data"][0]["url"] == "/tmp/out.mp4"
        assert dumped["data"][0]["format"] == "mp4"
