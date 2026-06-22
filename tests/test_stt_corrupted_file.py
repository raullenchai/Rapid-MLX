# SPDX-License-Identifier: Apache-2.0
"""R6-H3 (Eva 0.8.7 dogfood) — STT must surface decode failures as 400.

Eva's r2 dogfood reproduced ``POST /v1/audio/transcriptions`` with 50
bytes of garbage as the audio body: the route returned 500
``transcription_failed`` because the decode exception from
mlx_audio (chained from ffmpeg / soundfile / librosa) fell through
the catch-all. The OpenAI contract maps a corrupted upload to
**400** ``invalid_request_error`` with ``param="file"`` — the r6-C fix
introspects the exception and re-maps it.

Tests stub the engine so the assertions don't require real audio
decoders.
"""

from __future__ import annotations

import io
import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def _stub_engine_raising(monkeypatch):
    """Install a fake STTEngine that raises a decode-shaped error
    when ``transcribe()`` runs.

    The route's exception classifier inspects both the exception class
    name AND the message for the decode-failure shape, so we use a
    plain ``RuntimeError`` whose message matches one of the hint
    substrings — same surface ``mlx_audio`` ends up presenting when
    the codec layer fails.
    """
    # Fake mlx_audio so the lane probe passes.
    import importlib.machinery

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

    class _FakeEngineRaisingDecode:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def load(self):
            pass

        def transcribe(self, audio_path, language=None, task="transcribe"):
            # Mirror the surface ``mlx_audio`` raises when libsndfile
            # rejects a malformed header — generic exception class,
            # decode-shaped message. The classifier matches the
            # substring case-insensitively.
            raise RuntimeError("Error opening 'audio.wav': Format not recognised.")

    monkeypatch.setattr(
        "vllm_mlx.audio.stt.STTEngine",
        _FakeEngineRaisingDecode,
        raising=False,
    )
    audio_stt_mod = sys.modules.get("vllm_mlx.audio.stt")
    if audio_stt_mod is not None:
        monkeypatch.setattr(audio_stt_mod, "STTEngine", _FakeEngineRaisingDecode)

    audio_route._stt_engine = None
    yield
    audio_route._stt_engine = None
    probe._reset_probe_cache()


def _mount_audio_app():
    from vllm_mlx.config import get_config
    from vllm_mlx.routes import audio as audio_route

    app = FastAPI()
    app.include_router(audio_route.router)
    cfg = get_config()
    saved = cfg.api_key
    cfg.api_key = None
    return TestClient(app), lambda: setattr(cfg, "api_key", saved)


def _garbage_bytes(n: int = 50) -> bytes:
    """Return ``n`` bytes of high-entropy garbage — no WAV/MP3/Ogg header."""
    return bytes(((i * 73) ^ 0x5A) & 0xFF for i in range(n))


class TestCorruptedUploadReturns400:
    """Decoder failures must return 400 ``invalid_request_error`` with
    ``param="file"`` — NOT 500 ``transcription_failed``."""

    def test_garbage_body_returns_400(self, _stub_engine_raising):
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"model": "whisper-large-v3"},
                files={
                    "file": (
                        "garbage.wav",
                        io.BytesIO(_garbage_bytes(50)),
                        "audio/wav",
                    ),
                },
            )
        finally:
            restore()
        # Pre-fix this was 500 ``transcription_failed`` — a client error
        # masquerading as a server error. Post-fix it's 400 with the
        # OpenAI-shape envelope.
        assert r.status_code == 400, (
            f"Expected 400 invalid_request_error for garbage upload, "
            f"got {r.status_code}: {r.text}"
        )
        body = r.json()
        err = body.get("detail", {}).get("error") or body.get("error")
        assert err is not None, body
        assert err["type"] == "invalid_request_error", err
        assert err["param"] == "file", err
        # Message must include the decode-failure phrase from the
        # underlying exception so the client sees the actual reason.
        msg = err["message"].lower()
        assert (
            "decode" in msg or "decode audio" in msg or "format not recognised" in msg
        ), f"Decode-error envelope must mention the failure: {err['message']!r}"

    def test_envelope_does_not_leak_traceback(self, _stub_engine_raising):
        """Sanity: the 400 envelope must NOT include a stack trace."""
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"model": "whisper-large-v3"},
                files={
                    "file": (
                        "garbage.wav",
                        io.BytesIO(_garbage_bytes(50)),
                        "audio/wav",
                    ),
                },
            )
        finally:
            restore()
        body_text = r.text
        for forbidden in (
            "Traceback",
            "/usr/",
            "site-packages",
            'File "',
        ):
            assert forbidden not in body_text, (
                f"R6-H3: 400 envelope leaked {forbidden!r}: {body_text!r}"
            )


class TestNonDecodeErrorStillReturns500:
    """A genuine backend bug (NOT a decode failure) must keep its 500
    ``transcription_failed`` envelope — the decode re-classifier must
    not over-eagerly relabel server errors as client errors."""

    def test_unrelated_runtime_error_returns_500(self, monkeypatch):
        # Fake mlx_audio for the probe.
        import importlib.machinery

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

        class _FakeEngineRaisingUnrelated:
            def __init__(self, model_name: str):
                self.model_name = model_name

            def load(self):
                pass

            def transcribe(self, audio_path, language=None, task="transcribe"):
                # A real backend bug — index error inside the decoder
                # stage. Nothing about this looks like a decode failure;
                # the classifier must NOT downgrade it to 400.
                raise RuntimeError("internal beam search state was None")

        monkeypatch.setattr(
            "vllm_mlx.audio.stt.STTEngine",
            _FakeEngineRaisingUnrelated,
            raising=False,
        )
        audio_stt_mod = sys.modules.get("vllm_mlx.audio.stt")
        if audio_stt_mod is not None:
            monkeypatch.setattr(audio_stt_mod, "STTEngine", _FakeEngineRaisingUnrelated)

        audio_route._stt_engine = None
        try:
            client, restore = _mount_audio_app()
            try:
                # Need a WAV-shaped body so the upload itself succeeds —
                # we want the failure to come from the engine.
                import math
                import struct
                import wave

                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    for i in range(4000):
                        sample = int(8000 * math.sin(2 * math.pi * 440 * i / 16000))
                        w.writeframes(struct.pack("<h", sample))
                wav = buf.getvalue()

                r = client.post(
                    "/v1/audio/transcriptions",
                    data={"model": "whisper-large-v3"},
                    files={"file": ("tone.wav", wav, "audio/wav")},
                )
            finally:
                restore()
        finally:
            audio_route._stt_engine = None
            probe._reset_probe_cache()

        assert r.status_code == 500, (
            f"Real backend bugs must stay 500, got {r.status_code}: "
            f"{r.text!r}. If this trips, the decode-error classifier "
            "is over-eagerly relabeling server errors as client errors."
        )
        body = r.json()
        err = body.get("detail", {}).get("error") or body.get("error")
        assert err is not None, body
        assert err["code"] == "transcription_failed", err
