# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/audio/music`` — the text→music / text→SFX route.

Hermetic — no real models / network / SA3 weights. ``MusicEngine`` is
faked at the import boundary via monkeypatch (per repo convention), so
these run fast in CI.

Covered:

* happy path (WAV bytes out) + ``model`` → ``(dit, decoder)`` mapping;
* schema rejections (blank / over-long ``input``, ``seconds`` past the
  SA3 ceiling, unsupported ``response_format``) never reach the engine;
* the empty-output failure modes — no file, a 0-byte file, a valid RIFF
  header with zero sample frames, and unparseable bytes — all 500 with
  ``code="music_generation_failed"`` instead of a 200 that reads as a
  successfully generated silent clip;
* the blocking render runs OFF the event loop (structurally verified by
  thread identity), and survives cancellation without releasing the lock
  or unlinking the temp file under a live worker.

NOTE (CI): these import ``vllm_mlx.config``, whose import chain reaches
``engine_core``'s ``import mlx.core`` — so this file belongs in the
``test-apple-silicon`` job, not the mlx-free Linux matrix.
"""

from __future__ import annotations

import io
import math
import struct
import sys
import wave

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# This suite imports ``vllm_mlx.config``, whose import chain reaches
# ``engine_core`` and therefore requires MLX.  The regular Linux matrix and
# diff-aware PR validation deliberately run without MLX; the Apple Silicon
# job below exercises the full suite instead.
pytest.importorskip("mlx")

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


def _install_engine(monkeypatch, engine_cls):
    """Swap ``MusicEngine`` for ``engine_cls`` and clear the route cache.

    The route imports ``MusicEngine`` from ``vllm_mlx.audio.music`` INSIDE
    the worker function, so patch the attribute on the module object as
    well as by dotted path — the former is what a late import resolves.
    """
    from vllm_mlx.routes import audio as audio_route

    monkeypatch.setattr("vllm_mlx.audio.music.MusicEngine", engine_cls, raising=False)
    music_mod = sys.modules.get("vllm_mlx.audio.music")
    if music_mod is not None:
        monkeypatch.setattr(music_mod, "MusicEngine", engine_cls)
    audio_route._music_engine = None


@pytest.fixture
def _stub_music_engine(monkeypatch):
    from vllm_mlx.routes import audio as audio_route

    _FakeMusicEngine.last_call = None
    _install_engine(monkeypatch, _FakeMusicEngine)
    yield
    audio_route._music_engine = None


# ---------------------------------------------------------------------------
# Happy path + schema validation
# ---------------------------------------------------------------------------


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

    def test_unknown_model_is_rejected(self, _stub_music_engine):
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/music",
                json={"model": "no-such-variant", "input": "ambient pad"},
            )
        finally:
            restore()
        assert r.status_code == 400, r.text
        assert r.json()["detail"]["error"]["code"] == "invalid_model"
        assert r.json()["detail"]["error"]["param"] == "model"
        assert _FakeMusicEngine.last_call is None

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

    @pytest.mark.parametrize(
        "bad", [float("nan"), float("inf"), float("-inf"), "nan", "inf"]
    )
    def test_non_finite_seconds_is_rejected(self, bad):
        """A non-finite ``seconds`` must never reach the engine.

        Range checks are the classic NaN hole (every comparison against
        NaN is False), which is why the model carries an explicit
        ``math.isfinite`` guard alongside the ``gt=/le=`` bounds. This
        test pins the OUTCOME rather than which layer catches it, so it
        keeps holding whichever one fires first.

        Asserted at the model rather than over HTTP because JSON has no
        NaN literal — Python's ``json`` parses the non-standard ``NaN``
        token but then refuses to encode it back into the 422 body, so a
        bare-app round trip fails inside stock FastAPI's own error handler
        rather than at our validator.
        """
        from pydantic import ValidationError

        from vllm_mlx.api.models import AudioMusicRequest

        with pytest.raises(ValidationError):
            AudioMusicRequest(input="track", seconds=bad)

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

    def test_engine_error_is_500_envelope(self, monkeypatch):
        """An engine RuntimeError surfaces the documented envelope."""

        class _BoomEngine(_FakeMusicEngine):
            def generate(self, prompt, out_path, **kwargs):  # noqa: D102
                raise RuntimeError("SA3 exited 1: <no stderr>")

        _install_engine(monkeypatch, _BoomEngine)
        client, restore = _mount_audio_app()
        try:
            r = client.post("/v1/audio/music", json={"input": "boom", "seconds": 5})
        finally:
            restore()
        assert r.status_code == 500, r.text
        assert r.json()["detail"]["error"]["code"] == "music_generation_failed", r.text
        # No subprocess/filesystem internals leaked to the caller.
        assert "SA3 exited" not in r.text


# ---------------------------------------------------------------------------
# Empty-output detection — a "successful" render with no audio must 500
# ---------------------------------------------------------------------------


class TestMusicEmptyOutputDetection:
    """A "successful" render with no audio must not become a 200."""

    def _post_with_engine(self, monkeypatch, engine_cls):
        from vllm_mlx.routes import audio as audio_route

        _install_engine(monkeypatch, engine_cls)
        client, restore = _mount_audio_app()
        try:
            return client.post("/v1/audio/music", json={"input": "x", "seconds": 5})
        finally:
            restore()
            audio_route._music_engine = None

    def test_engine_writing_nothing_is_500_not_empty_200(self, monkeypatch):
        """An engine that exits cleanly without writing must 500.

        The SA3 CLI can return success having produced no audio (a sampler
        that bailed after the header). Returning that as HTTP 200 with an
        empty ``audio/wav`` body reads to the caller as a successfully
        generated silent clip — the worst kind of failure.
        """

        class _SilentEngine(_FakeMusicEngine):
            def generate(self, prompt, out_path, **kwargs):  # noqa: D102
                # Exits "successfully" but leaves the temp file at 0 bytes.
                return out_path

        r = self._post_with_engine(monkeypatch, _SilentEngine)
        assert r.status_code == 500, r.text
        assert r.json()["detail"]["error"]["code"] == "music_generation_failed", r.text

    def test_missing_output_file_is_500(self, monkeypatch):
        """``MusicEngine.generate`` unlinks the target before rendering.

        So a run that exits without writing leaves NO file at all, not an
        empty one — the route must handle the missing-path case too.
        """
        import os

        class _UnlinkingEngine(_FakeMusicEngine):
            def generate(self, prompt, out_path, **kwargs):  # noqa: D102
                os.unlink(out_path)
                return out_path

        r = self._post_with_engine(monkeypatch, _UnlinkingEngine)
        assert r.status_code == 500, r.text
        assert r.json()["detail"]["error"]["code"] == "music_generation_failed", r.text

    def test_header_only_wav_is_500_not_a_silent_clip(self, monkeypatch):
        """A valid RIFF header with ZERO sample frames must fail.

        A bare WAV header is ~44 bytes, so a non-empty-bytes check passes
        it and the caller receives a 200 with ``audio/wav`` that plays as
        silence — indistinguishable from a real quiet track.
        """

        class _HeaderOnlyEngine(_FakeMusicEngine):
            def generate(self, prompt, out_path, **kwargs):  # noqa: D102
                with wave.open(str(out_path), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(44100)
                    # No writeframes() call at all.
                return out_path

        r = self._post_with_engine(monkeypatch, _HeaderOnlyEngine)
        assert r.status_code == 500, r.text
        assert r.json()["detail"]["error"]["code"] == "music_generation_failed", r.text

    def test_unparseable_output_is_500_not_mislabelled_wav(self, monkeypatch):
        """Output that isn't a readable WAV must fail, not be mislabelled.

        Fail-closed is right here because the producer uses the SAME
        parser: SA3's ``save_wav`` writes 16-bit PCM through ``wave.open``
        (``audio/sa3/scripts/sa3_mlx.py``). So bytes ``wave`` can't read
        are not SA3 output, and returning them under
        ``Content-Type: audio/wav`` would be mislabelling.
        """

        class _OpaqueEngine(_FakeMusicEngine):
            def generate(self, prompt, out_path, **kwargs):  # noqa: D102
                with open(out_path, "wb") as fh:
                    fh.write(b"\x00\x01\x02\x03" * 64)
                return out_path

        r = self._post_with_engine(monkeypatch, _OpaqueEngine)
        assert r.status_code == 500, r.text
        assert r.json()["detail"]["error"]["code"] == "music_generation_failed", r.text


# ---------------------------------------------------------------------------
# Concurrency: off-loop execution + cancellation safety
# ---------------------------------------------------------------------------


class TestMusicConcurrencyShape:
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

    def test_lock_can_be_reused_across_event_loops(self):
        """A process-global lane remains usable when request loops change."""
        import asyncio

        from vllm_mlx.routes import audio as audio_route

        async def _take_once():
            async with audio_route._get_music_lock():
                await asyncio.sleep(0)

        asyncio.run(_take_once())
        asyncio.run(_take_once())

    def test_cancellation_drains_the_worker_before_unwinding(self):
        """A client disconnect must not free the lock / temp file early.

        ``await asyncio.to_thread(...)`` is NOT cancellable — the worker
        keeps running — yet the await returns immediately when the
        surrounding task is cancelled. Left unguarded, the handler's
        ``async with`` lock would unwind (admitting a second concurrent
        SA3 subprocess) and its ``finally`` would unlink the wav the
        abandoned worker is still writing into. ``run_to_completion``
        shields and drains instead. Assert the drain: the worker must have
        FINISHED by the time the CancelledError propagates.
        """
        import asyncio
        import threading

        from vllm_mlx.routes._async_utils import run_to_completion

        started = threading.Event()
        finished = threading.Event()

        def _slow_worker():
            started.set()
            # Long enough that the cancellation below lands mid-flight.
            threading.Event().wait(0.3)
            finished.set()

        async def _drive():
            task = asyncio.ensure_future(run_to_completion(_slow_worker))
            await asyncio.to_thread(started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # The drain is the whole point: the worker is done, so the
            # caller's lock release + temp-file unlink are now safe.
            return finished.is_set()

        assert asyncio.run(_drive()) is True
