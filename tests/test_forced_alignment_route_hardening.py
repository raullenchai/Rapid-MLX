# SPDX-License-Identifier: Apache-2.0
"""Forced-alignment HTTP lane hardening — ``/v1/audio/transcriptions`` + ``text``.

``tests/test_forced_alignment.py`` covers the engine surface and the
happy-path route contract. This suite pins the properties of the
dedicated alignment lane (``_run_alignment_request``) that are easy to
regress silently:

* the blocking weight load + ``align()`` run OFF the event loop, and a
  cancelled request drains its worker before the lock and the temp file
  are released;
* the aligner is cached in its OWN global, never the shared
  ``_stt_engine`` the ASR lane mutates from the event loop, and a failed
  ``load()`` is not published into that cache;
* error classification order — a corrupted upload is reported as a
  corrupted upload, an internal ``ValueError`` is a 500, and only the two
  documented ``align()`` rejections become a 400;
* the "just send audio + text" defaults (aligner model, ``verbose_json``)
  including whitespace-only ``model`` / ``response_format``;
* a present-but-blank ``text`` is a 400 rather than a silent ASR answer;
* calling the handler directly (no ASGI stack) tolerates the unresolved
  ``Form(None)`` / ``Query(None)`` sentinels.

Hermetic — engines are faked at the import boundary, no weights or
network.
"""

from __future__ import annotations

import asyncio
import importlib.machinery
import io
import math
import struct
import sys
import threading
import types
import wave

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="requires the MLX runtime available on Apple Silicon",
)

_ALIGNER_ID = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"


# ---------------------------------------------------------------------------
# Shared helpers
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


def _patch_engine(monkeypatch, engine_cls):
    """Swap ``STTEngine`` at the import boundary the route resolves."""
    monkeypatch.setattr("vllm_mlx.audio.stt.STTEngine", engine_cls, raising=False)
    stt_mod = sys.modules.get("vllm_mlx.audio.stt")
    if stt_mod is not None:
        monkeypatch.setattr(stt_mod, "STTEngine", engine_cls)


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


class _UploadLike:
    """Minimal stand-in for ``UploadFile`` — only ``read`` is exercised by
    ``_stream_upload_to_tempfile``."""

    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)

    async def read(self, size: int = -1) -> bytes:
        return self._buf.read(size)


def _patch_engine_direct(audio_route, engine_cls):
    """Swap ``STTEngine`` without monkeypatch, for use inside ``asyncio.run``.

    Mutates the real module, so the ``_fake_audio_env`` fixture restores the
    original on teardown — an earlier version of this helper did not, and it
    silently broke every later test file that imports ``STTEngine`` (six
    failures in ``test_forced_alignment.py`` that vanished when that file ran
    alone).

    Also clears both engine caches so the next call constructs the new class
    rather than reusing whatever a previous test left resident.
    """
    import vllm_mlx.audio.stt as stt_mod

    stt_mod.STTEngine = engine_cls
    audio_route._stt_engine = None
    audio_route._aligner_engine = None


class _FakeAlignResult:
    def __init__(self, text, segments, language, duration):
        self.text = text
        self.segments = segments
        self.language = language
        self.duration = duration


class _FakeAlignerEngine:
    """Mirrors the ``STTEngine`` surface the alignment path touches."""

    last_call: dict | None = None
    #: Set when ``transcribe`` runs, so a test can assert the ASR branch
    #: was taken AND that it produced a real result — not merely that
    #: ``align`` was skipped.
    last_transcribe: dict | None = None

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


@pytest.fixture
def _stub_aligner(monkeypatch):
    from vllm_mlx.audio import probe
    from vllm_mlx.routes import audio as audio_route

    _install_fake_mlx_audio(monkeypatch)
    probe._reset_probe_cache()
    _patch_engine(monkeypatch, _FakeAlignerEngine)
    _FakeAlignerEngine.last_call = None
    _FakeAlignerEngine.last_transcribe = None
    audio_route._stt_engine = None
    audio_route._aligner_engine = None
    yield
    audio_route._stt_engine = None
    audio_route._aligner_engine = None
    probe._reset_probe_cache()


@pytest.fixture
def _fake_audio_env(monkeypatch):
    """Probe + engine-cache reset without binding a specific engine — for
    tests that install their own failure-shaped engine."""
    import vllm_mlx.audio.stt as stt_mod
    from vllm_mlx.audio import probe
    from vllm_mlx.routes import audio as audio_route

    _install_fake_mlx_audio(monkeypatch)
    probe._reset_probe_cache()
    real_engine = stt_mod.STTEngine
    audio_route._stt_engine = None
    audio_route._aligner_engine = None
    yield audio_route
    # Restore the real class: _patch_engine_direct assigns it outright, and
    # leaking a fake here breaks every later test file that imports it.
    stt_mod.STTEngine = real_engine
    audio_route._stt_engine = None
    audio_route._aligner_engine = None
    probe._reset_probe_cache()


def _post(client, data, payload: bytes | None = None):
    return client.post(
        "/v1/audio/transcriptions",
        data=data,
        files={
            "file": (
                "clip.wav",
                _make_tone_wav() if payload is None else payload,
                "audio/wav",
            )
        },
    )


def _err(response) -> dict:
    body = response.json()
    return body.get("detail", {}).get("error") or body.get("error")


# ---------------------------------------------------------------------------
# Defaults: "just send audio + text"
# ---------------------------------------------------------------------------


class TestAlignmentDefaults:
    def test_text_field_alone_aligns_and_defaults_to_verbose_json(self, _stub_aligner):
        """``text`` with no ``model`` / ``response_format`` must still align.

        The plain ``json`` envelope drops ``segments``, so defaulting
        ``response_format`` to it would return a 200 with none of the
        timestamps the caller came for.
        """
        client, restore = _mount_audio_app()
        try:
            r = _post(client, {"text": "临终前", "language": "Chinese"})
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("application/json")
        body = r.json()
        assert body["text"] == "临终前"
        assert isinstance(body["segments"], list)
        assert len(body["segments"]) == 3, body["segments"]
        # Compare field-by-field: the verbose_json serializer also stamps an
        # ``id`` on every segment (part of the OpenAI shape), so an equality
        # assertion against a bare {text,start,end} dict is testing the
        # fixture rather than the contract.
        first = body["segments"][0]
        assert first["text"] == "临"
        assert first["start"] == 0.0
        assert first["end"] == 0.2
        assert first["id"] == 0
        # The known transcript was forwarded as an INPUT (no recognition).
        call = _FakeAlignerEngine.last_call
        assert call["text"] == "临终前"
        assert call["language"] == "Chinese"
        assert _FakeAlignerEngine.last_transcribe is None

    def test_omitted_model_resolves_to_the_registered_aligner(self, _stub_aligner):
        """No ``model`` → the ALIGNER alias, not the ASR default.

        ``whisper-large-v3`` is not a forced aligner, so defaulting the
        alignment branch to it would fail deep in ``STTEngine.align``.
        """
        from vllm_mlx.routes import audio as audio_route

        client, restore = _mount_audio_app()
        try:
            r = _post(client, {"text": "abc"})
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
        that boundary. ``"   "`` is the one that actually leaks: it
        arrives verbatim and would reach ``_resolve_stt_model`` as an
        explicit choice, 404-ing as a nonexistent alias, so "just send
        audio + text" from a form with a spaced-out field never aligns.
        """
        from vllm_mlx.routes import audio as audio_route

        client, restore = _mount_audio_app()
        try:
            r = _post(client, {"text": "abc", "model": blank})
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert audio_route._stt_engine is None, audio_route._stt_engine
        assert "ForcedAligner" in audio_route._aligner_engine.model_name, (
            audio_route._aligner_engine.model_name
        )

    def test_blank_response_format_takes_verbose_json(self, _stub_aligner):
        """A whitespace-only ``response_format`` must fall back to verbose_json.

        Same shape as the ``model`` case above: ``"   "`` would otherwise
        count as an explicit choice, reach the allowed-set check and 400 —
        losing the timestamps the caller came for.
        """
        client, restore = _mount_audio_app()
        try:
            r = _post(client, {"text": "abc", "response_format": "   "})
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert isinstance(r.json()["segments"], list), r.text

    def test_explicit_response_format_is_honoured(self, _stub_aligner):
        """An explicit srt must win over the alignment default."""
        client, restore = _mount_audio_app()
        try:
            r = _post(client, {"text": "abc", "response_format": "srt"})
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert "00:00:00,000 --> 00:00:00,200" in r.text, r.text

    def test_blank_model_does_not_leak_into_the_asr_path(self, _stub_aligner):
        """The blank-model default is scoped to the alignment branch.

        The ASR path's handling of a whitespace-only ``model`` is
        long-standing contract (a 404 for a nonexistent alias) and must
        not change just because the alignment branch relaxed it.
        """
        client, restore = _mount_audio_app()
        try:
            r = _post(client, {"model": "   "})
        finally:
            restore()
        assert r.status_code == 404, r.text
        assert _err(r)["code"] == "model_not_found", r.text


# ---------------------------------------------------------------------------
# A present-but-blank ``text`` is never a silent ASR answer
# ---------------------------------------------------------------------------


class TestBlankTextIsRejected:
    def test_blank_text_without_model_is_400_not_a_silent_asr_fallback(
        self, _stub_aligner
    ):
        """Sending ``text`` says "I have the transcript, give me timings".

        A whitespace-only value used to fall through to speech
        recognition whenever ``model`` was not an aligner, returning 200 —
        answering a different question with no signal that the request had
        been reinterpreted.
        """
        client, restore = _mount_audio_app()
        try:
            r = _post(client, {"text": "   "})
        finally:
            restore()
        assert r.status_code == 400, r.text
        err = _err(r)
        assert err["type"] == "invalid_request_error", err
        assert err["code"] == "alignment_text_required", err
        assert err["param"] == "text", err
        # Neither engine path ran.
        assert _FakeAlignerEngine.last_call is None
        assert _FakeAlignerEngine.last_transcribe is None

    def test_blank_text_with_asr_model_is_also_400(self, _stub_aligner):
        """Same rejection when an ASR model is named explicitly.

        This is the combination that previously slipped through: the
        blank-text guard only fired for aligner models.
        """
        client, restore = _mount_audio_app()
        try:
            r = _post(client, {"text": "  \t ", "model": "whisper-large-v3"})
        finally:
            restore()
        assert r.status_code == 400, r.text
        assert _err(r)["code"] == "alignment_text_required", r.text
        assert _FakeAlignerEngine.last_transcribe is None

    def test_absent_text_is_still_asr(self, _stub_aligner):
        """Absent ``text`` → unchanged ASR, positively verified.

        Asserts the ASR branch actually SUCCEEDS and returns recognised
        text, not merely that ``align()`` was skipped.
        """
        client, restore = _mount_audio_app()
        try:
            r = _post(
                client,
                {"model": "whisper-large-v3", "response_format": "verbose_json"},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert r.json()["text"] == "recognised speech", r.text
        assert _FakeAlignerEngine.last_transcribe is not None
        assert _FakeAlignerEngine.last_call is None


# ---------------------------------------------------------------------------
# Error classification — order matters
# ---------------------------------------------------------------------------


class TestAlignmentErrorClassification:
    def test_decode_error_is_invalid_audio_file_not_alignment_request(
        self, _fake_audio_env, monkeypatch
    ):
        """Decoder ValueErrors must not be mislabelled as a bad request.

        Some codec paths raise plain ``ValueError`` for undecodable audio.
        The decode check therefore has to run BEFORE the alignment-request
        classifier, or a corrupted upload comes back as
        ``invalid_alignment_request`` blaming ``model``/``text`` — sending
        the caller to fix a field that was never wrong.
        """

        class _DecodeFailingEngine:
            def __init__(self, model_name):
                self.model_name = model_name

            def load(self):
                pass

            def align(self, audio_path, text, language="Chinese"):
                # The shape a codec raises on a truncated/garbage file.
                raise ValueError(
                    "Error opening file: File contains data in an unknown format."
                )

        _patch_engine(monkeypatch, _DecodeFailingEngine)
        client, restore = _mount_audio_app()
        try:
            r = _post(
                client,
                {"text": "abc", "model": "qwen3-aligner"},
                payload=b"not-a-wav",
            )
        finally:
            restore()

        assert r.status_code == 400, r.text
        err = _err(r)
        # Assert the EXACT documented decode envelope, not merely
        # "something other than invalid_alignment_request" — a weaker
        # assertion would be satisfied by any unrelated error.
        assert err["type"] == "invalid_request_error", err
        assert err["code"] == "invalid_audio_file", err
        assert err["param"] == "file", err

    def test_unexpected_value_error_is_500_not_invalid_request(
        self, _fake_audio_env, monkeypatch
    ):
        """An internal ValueError must not be blamed on ``model`` / ``text``.

        ``align()`` raises ValueError for exactly two caller mistakes. A
        ValueError from anywhere else (weight loading, tokenizing, a
        reshape) is an internal fault; reporting it as
        ``invalid_alignment_request`` sends the caller to fix a field that
        was never wrong.
        """

        class _InternallyBrokenEngine:
            def __init__(self, model_name):
                self.model_name = model_name

            def load(self):
                pass

            def align(self, audio_path, text, language="Chinese"):
                # Nothing to do with the caller's model or text.
                raise ValueError("cannot reshape array of size 0 into shape (1,80)")

        _patch_engine(monkeypatch, _InternallyBrokenEngine)
        client, restore = _mount_audio_app()
        try:
            r = _post(client, {"text": "abc", "model": "qwen3-aligner"})
        finally:
            restore()

        assert r.status_code == 500, r.text
        err = _err(r)
        assert err["code"] == "alignment_failed", err
        assert err["type"] == "api_error", err

    def test_engine_side_rejection_is_a_400_backstop(
        self, _fake_audio_env, monkeypatch
    ):
        """A documented ``align()`` rejection stays a 400, not an envelope-less 500.

        The route's own ``_is_aligner_model`` guard is the primary path,
        but it is a substring heuristic over the resolved repo id. A repo
        that satisfies it while the engine's own predicate refuses (or any
        future divergence between the two) must still produce the
        documented client envelope rather than a generic 500.
        """

        class _RefusingEngine:
            def __init__(self, model_name):
                self.model_name = model_name

            def load(self):
                pass

            def align(self, audio_path, text, language="Chinese"):
                raise ValueError(
                    f"align() requires a forced-aligner model; {self.model_name!r} "
                    "is not one (use a registry alias like 'qwen3-aligner')."
                )

        _patch_engine(monkeypatch, _RefusingEngine)
        client, restore = _mount_audio_app()
        try:
            # Passes the route's substring guard, refused by the engine.
            r = _post(client, {"text": "abc", "model": "acme/pretend-aligner-0.6b"})
        finally:
            restore()

        assert r.status_code == 400, r.text
        err = _err(r)
        assert err["type"] == "invalid_request_error", err
        assert err["code"] == "invalid_alignment_request", err
        assert err["param"] == "model", err

    def test_non_aligner_model_with_text_400s_before_the_upload_drains(
        self, _fake_audio_env, monkeypatch
    ):
        """``text`` + an ASR model rejects without touching the engine."""

        class _Boom:
            def __init__(self, model_name):  # pragma: no cover - must not run
                raise AssertionError("engine must not be constructed")

        _patch_engine(monkeypatch, _Boom)
        client, restore = _mount_audio_app()
        try:
            r = _post(client, {"text": "hello", "model": "whisper-large-v3"})
        finally:
            restore()
        assert r.status_code == 400, r.text
        err = _err(r)
        assert err["code"] == "alignment_model_required", err
        assert err["param"] == "model", err


# ---------------------------------------------------------------------------
# Engine cache discipline
# ---------------------------------------------------------------------------


class TestAlignerEngineCache:
    def test_failed_load_is_not_published_to_the_cache(
        self, _fake_audio_env, monkeypatch
    ):
        """A load that raises must leave ``_aligner_engine`` untouched.

        Assigning the engine before ``load()`` would leave later requests
        matching on ``model_name`` against an object whose weights never
        loaded — a permanently broken cache entry that only a restart
        clears.
        """
        audio_route = _fake_audio_env
        attempts = []

        class _LoadFailingEngine:
            def __init__(self, model_name):
                self.model_name = model_name
                attempts.append(model_name)

            def load(self):
                raise RuntimeError("weights unavailable")

            def align(self, audio_path, text, language="Chinese"):
                raise AssertionError("align must not run after a failed load")

        _patch_engine(monkeypatch, _LoadFailingEngine)
        client, restore = _mount_audio_app()
        try:
            first = _post(client, {"text": "abc", "model": "qwen3-aligner"})
            assert first.status_code == 500, first.text
            assert audio_route._aligner_engine is None, audio_route._aligner_engine
            # A retry must construct + load again, not reuse a corpse.
            second = _post(client, {"text": "abc", "model": "qwen3-aligner"})
            assert second.status_code == 500, second.text
        finally:
            restore()
        assert len(attempts) == 2, attempts

    def test_lanes_use_distinct_caches_and_only_one_model_stays_resident(
        self, _stub_aligner
    ):
        """Two properties that sound contradictory but aren't.

        DISTINCT CACHES: the lanes must never share one global. Alignment
        runs on a worker thread while ASR runs on the event loop, so a shared
        global could be swapped between the alignment path's cache check and
        its ``align()`` call. Distinctness is what makes that impossible.

        ONE RESIDENT: distinct caches alone would leave BOTH multi-GB models
        in unified memory after alternating requests, on a server that can
        only use one at a time. So loading into one lane releases the other,
        which is safe precisely because the lane lock guarantees the released
        engine is idle.
        """
        from vllm_mlx.routes import audio as audio_route

        client, restore = _mount_audio_app()
        try:
            # Alignment populates the aligner cache only.
            assert _post(client, {"text": "abc"}).status_code == 200
            aligner = audio_route._aligner_engine
            assert aligner is not None
            assert audio_route._stt_engine is None

            # ASR loads into its OWN cache and releases the aligner.
            assert _post(client, {"model": "whisper-large-v3"}).status_code == 200
            asr = audio_route._stt_engine
            assert asr is not None
            assert asr is not aligner, "the lanes are sharing one cache"
            assert audio_route._aligner_engine is None, (
                "the aligner stayed resident alongside the ASR model"
            )

            # Back to alignment: a fresh aligner, and ASR released.
            assert _post(client, {"text": "def"}).status_code == 200
            assert audio_route._aligner_engine is not None
            assert audio_route._aligner_engine is not asr
            assert audio_route._stt_engine is None
        finally:
            restore()


# ---------------------------------------------------------------------------
# Concurrency: blocking work must leave the event loop
# ---------------------------------------------------------------------------


class TestAlignmentConcurrency:
    def test_alignment_does_not_block_the_event_loop(self, _stub_aligner):
        """Weight load + align must run off the event loop.

        Both are seconds of blocking compute; inline in the ``async def``
        handler they stall every concurrent request on the server. Detect
        structurally — the engine must see a different thread than the
        loop.
        """
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

    def test_lock_waits_asynchronously_not_on_a_thread_lock(self):
        """A contended lane lock does not stall unrelated loop work.

        This behavioral check fails if ``__aenter__`` calls
        ``threading.Lock.acquire`` inline, regardless of wrapper type.
        """
        from vllm_mlx.routes import audio as audio_route

        async def _drive():
            lock = audio_route._get_stt_lane_lock()
            entered = False

            async def _waiter():
                nonlocal entered
                async with lock:
                    entered = True

            async with lock:
                waiter = asyncio.create_task(_waiter())
                heartbeats = 0
                for _ in range(5):
                    await asyncio.sleep(0.01)
                    heartbeats += 1
                assert heartbeats == 5
                assert not entered
            await waiter
            assert entered

        asyncio.run(_drive())

    def test_cancellation_drains_the_worker_before_unwinding(self):
        """A client disconnect must not release the lock / temp file early.

        ``await asyncio.to_thread(...)`` is not cancellable — the worker
        runs on — but the await returns immediately when the task is
        cancelled. Without ``run_to_completion``'s shield-and-drain, the
        surrounding ``async with`` lock would admit another request and
        the ``finally`` would unlink the audio file while the abandoned
        worker was still using both.
        """
        from vllm_mlx.routes._async_utils import run_to_completion

        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def _slow_worker():
            started.set()
            release.wait(5.0)
            finished.set()

        async def _drive():
            task = asyncio.ensure_future(run_to_completion(_slow_worker))
            await asyncio.to_thread(started.wait, 5.0)
            task.cancel()
            # The worker is still running, so the cancellation must NOT
            # have completed yet.
            await asyncio.sleep(0.05)
            assert not task.done(), "cancellation unwound while the worker ran"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()

        asyncio.run(_drive())

    def test_direct_inner_task_cancel_still_drains_worker(self):
        """Cancelling the asyncio wrapper cannot outlive its worker thread."""
        from vllm_mlx.routes._async_utils import run_to_completion

        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def _slow_worker():
            started.set()
            release.wait(5.0)
            finished.set()

        async def _drive():
            outer = asyncio.create_task(run_to_completion(_slow_worker))
            await asyncio.to_thread(started.wait, 5.0)
            current = asyncio.current_task()
            inner = next(
                task
                for task in asyncio.all_tasks()
                if task not in {current, outer} and not task.done()
            )
            inner.cancel()
            await asyncio.sleep(0.05)
            assert not outer.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await outer
            assert finished.is_set()

        asyncio.run(_drive())


# ---------------------------------------------------------------------------
# Direct handler invocation (no ASGI stack)
# ---------------------------------------------------------------------------


def test_direct_handler_call_tolerates_unresolved_form_defaults(monkeypatch):
    """Calling ``create_transcription`` directly must not hit the ``text``
    logic with a FastAPI sentinel.

    Several existing tests (e.g. test_audio_upload_size_limit) invoke the
    handler as a plain coroutine rather than through the ASGI stack, so any
    parameter they don't pass arrives as its unresolved ``Form(None)`` /
    ``Query(None)`` object — truthy and non-None but NOT a string. The
    pre-existing params only ever compare against None, so they tolerate
    it; the alignment branch calls ``.strip()``, which would raise
    ``AttributeError: 'Form' object has no attribute 'strip'`` without the
    isinstance merge.
    """
    from vllm_mlx.audio import probe
    from vllm_mlx.routes import audio as audio_route

    _install_fake_mlx_audio(monkeypatch)
    probe._reset_probe_cache()

    class _Boom:
        """An engine that must never be reached — a bogus model 404s first."""

        def __init__(self, model_name):  # pragma: no cover - must not run
            raise AssertionError("engine must not be constructed")

    _patch_engine(monkeypatch, _Boom)

    # Only the pre-F-165 kwargs, exactly as the older direct-call tests do:
    # text_form / text_query are left at their unresolved defaults.
    try:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                audio_route.create_transcription(
                    file=_UploadLike(_make_tone_wav()),  # type: ignore[arg-type]
                    model_form="definitely-not-a-real-alias",
                    language_form=None,
                    response_format_form=None,
                    model_query=None,
                    language_query=None,
                    response_format_query=None,
                )
            )
    finally:
        probe._reset_probe_cache()

    # A 404 for the bogus alias proves we got through the ASR/alignment
    # branch selection without an AttributeError on the sentinel.
    assert exc_info.value.status_code == 404, exc_info.value.detail


class TestSttLaneMutualExclusion:
    """ASR and alignment must not run against the accelerator at once.

    The two lanes share no Python state (`_stt_engine` and `_aligner_engine`
    are separate caches on purpose) but they DO share unified memory and the
    GPU: each loads its own multi-GB model. Before alignment was offloaded,
    every audio lane ran inline on the event loop, so they were mutually
    exclusive by accident. Offloading removed that accident — hence one lock
    covering both lanes rather than one per lane.
    """

    def test_asr_and_alignment_take_the_same_lock(self):
        """Cheap source pin — the behavioural test below is the real check.

        Kept because it names the invariant at the point a future edit would
        break it, but it proves nothing on its own: a reference to
        `_get_stt_lane_lock()` outside an `async with` would satisfy it.
        """
        import inspect

        from vllm_mlx.routes import audio as audio_route

        for fn in (audio_route._run_stt_request, audio_route._run_alignment_request):
            src = inspect.getsource(fn)
            assert "async with _get_stt_lane_lock()" in src, (
                f"{fn.__name__} must hold the shared STT-lane lock, not just "
                f"reference it"
            )

    def test_alignment_cannot_enter_while_the_lane_is_held(self, _fake_audio_env):
        """A live ASR request must keep alignment out of its engine.

        Note ASR cannot be driven *concurrently* here: it still executes
        inline on the event loop, so a fake that parks inside `transcribe`
        blocks the loop itself and nothing else can be observed. So this
        holds the lane the way a running ASR request does — by taking the
        same lock ASR takes — and asserts alignment queues behind it,
        touching neither its engine nor the temp file until released.

        My first version of this test only invoked alignment, which meant it
        would have passed even with ASR entirely unlocked. This one fails if
        the alignment path stops taking the lane lock.
        """
        audio_route = _fake_audio_env
        order: list[str] = []

        class _RecordingAligner(_FakeAlignerEngine):
            def align(self, audio_path, text, language="Chinese"):
                order.append("align-enter")
                return super().align(audio_path, text, language=language)

        async def _drive():
            _patch_engine_direct(audio_route, _RecordingAligner)
            lock = audio_route._get_stt_lane_lock()

            async with lock:  # stands in for a running ASR request
                order.append("lane-held")
                align = asyncio.ensure_future(
                    audio_route._run_alignment_request(
                        file=_UploadLike(_make_tone_wav()),
                        model="qwen3-aligner",
                        text="abc",
                        language=None,
                        response_format="verbose_json",
                    )
                )
                # Several loop turns: a broken lane lock would let the
                # offloaded align() start in the executor by now.
                for _ in range(20):
                    await asyncio.sleep(0)
                await asyncio.sleep(0.05)
                assert "align-enter" not in order, (
                    f"alignment reached its engine while the lane was held: {order}"
                )
                assert not align.done()
                order.append("lane-released")

            await align
            assert order == ["lane-held", "lane-released", "align-enter"], order

        asyncio.run(_drive())

    def test_asr_engine_is_never_handed_an_alignment_request(self, _fake_audio_env):
        """A shared cache would hand `align()` to a resident ASR model.

        `STTEngine` rejects that, but only after a weight load and deep
        inside the engine. This asserts the alignment path constructs its
        own engine rather than reusing whatever ASR left behind.
        """
        audio_route = _fake_audio_env

        class _AsrOnlyEngine:
            def __init__(self, model_name):
                self.model_name = model_name

            def load(self):
                pass

            def transcribe(self, audio_path, language=None, task="transcribe"):
                return _FakeAlignResult("asr text", [], "en", 1.0)

            def align(self, audio_path, text, language="Chinese"):
                raise AssertionError(
                    "the ASR engine was handed an alignment request — the "
                    "lanes are sharing a cache"
                )

        async def _drive():
            _patch_engine_direct(audio_route, _AsrOnlyEngine)
            await audio_route._run_stt_request(
                file=_UploadLike(_make_tone_wav()),
                model="whisper-large-v3",
                language=None,
                response_format="json",
                task="transcribe",
            )
            assert isinstance(audio_route._stt_engine, _AsrOnlyEngine)

            # Alignment must build its own — if it reused the ASR engine the
            # AssertionError above would fire.
            import vllm_mlx.audio.stt as stt_mod

            stt_mod.STTEngine = _FakeAlignerEngine
            await audio_route._run_alignment_request(
                file=_UploadLike(_make_tone_wav()),
                model="qwen3-aligner",
                text="abc",
                language=None,
                response_format="verbose_json",
            )
            assert isinstance(audio_route._aligner_engine, _FakeAlignerEngine)

        asyncio.run(_drive())

    def test_asr_holds_the_lock_across_load_and_inference(self):
        """The lock must wrap the weight load too, not just inference.

        Acquiring it between load and `transcribe` would still let two
        models be loaded concurrently, which is half the point.
        """
        import inspect

        from vllm_mlx.routes import audio as audio_route

        src = inspect.getsource(audio_route._run_stt_request)
        lock_at = src.index("async with _get_stt_lane_lock()")
        load_at = src.index("stt_engine.load")
        transcribe_at = src.index("_stt_engine.transcribe")
        assert lock_at < load_at < transcribe_at, (
            "the lock must be acquired before the weight load"
        )


class TestDrainSurvivesRepeatedCancellation:
    """A second cancel must not abandon a running worker.

    The drain in `run_to_completion` has to be a SHIELDED LOOP. A bare
    `await task` is itself cancellable, so a second `Task.cancel()` — a
    shutdown signal, a supervisor giving up on a hung request — would
    interrupt the drain and hand control back while the thread is still
    running, releasing the lock and unlinking the temp file underneath it.
    That is the exact failure the helper exists to prevent, one cancel later.
    """

    def test_two_cancels_still_wait_for_the_worker(self):
        """The observable failure: does the CALLER's finally run too early?

        Asserting on the wrapper task alone is not enough — a bare-await
        drain also leaves it pending. What actually breaks is the caller
        resuming while the thread is live, so this reproduces the real
        shape: hold a lock and own a temp file across the call, cancel
        twice, and check whether cleanup happened before the worker
        finished.
        """
        from vllm_mlx.routes._async_utils import run_to_completion

        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        # Ordering evidence: what the caller did, and when.
        events: list[str] = []

        def _slow():
            started.set()
            release.wait(timeout=5)
            finished.set()
            events.append("worker-finished")
            return "done"

        async def _caller():
            """Mirrors the route: lock held + temp file owned across the call."""
            lock = asyncio.Lock()
            async with lock:
                try:
                    return await run_to_completion(_slow)
                finally:
                    # This is the line that must NOT run while the thread
                    # is still using the engine and the temp file.
                    events.append("caller-cleanup")

        async def _drive():
            task = asyncio.ensure_future(_caller())
            await asyncio.get_running_loop().run_in_executor(None, started.wait, 5)

            task.cancel()
            await asyncio.sleep(0)
            task.cancel()  # second cancel — must not break the drain
            # Give the loop several turns to let a broken drain unwind.
            for _ in range(20):
                await asyncio.sleep(0)
            await asyncio.sleep(0.05)

            assert not finished.is_set(), "test bug: worker released too early"
            assert "caller-cleanup" not in events, (
                "the caller's finally ran while the worker thread was still "
                f"alive — drain was interrupted. events={events}"
            )

            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert events == ["worker-finished", "caller-cleanup"], events

        asyncio.run(_drive())

    def test_uncancelled_call_returns_the_result(self):
        from vllm_mlx.routes._async_utils import run_to_completion

        async def _drive():
            return await run_to_completion(lambda: 42)

        assert asyncio.run(_drive()) == 42


class TestCrossLoopLock:
    def test_two_event_loops_share_one_exclusion_domain(self):
        """Separate event loops cannot drive the process-global engines together."""
        from vllm_mlx.routes import audio as audio_route

        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()

        async def _first():
            async with audio_route._get_stt_lane_lock():
                first_entered.set()
                await asyncio.to_thread(release_first.wait, 5.0)

        async def _second():
            async with audio_route._get_stt_lane_lock():
                second_entered.set()

        first = threading.Thread(target=lambda: asyncio.run(_first()))
        second = threading.Thread(target=lambda: asyncio.run(_second()))
        first.start()
        assert first_entered.wait(5.0)
        second.start()
        assert not second_entered.wait(0.1)
        release_first.set()
        first.join(5.0)
        second.join(5.0)
        assert second_entered.is_set()
