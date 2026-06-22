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


def _install_fake_mlx_audio(monkeypatch):
    """Install minimal mlx_audio stub so the lane probe passes."""
    import importlib.machinery

    for name, is_pkg in (
        ("mlx_audio", True),
        ("mlx_audio.stt", True),
        ("mlx_audio.stt.utils", False),
    ):
        mod = types.ModuleType(name)
        if is_pkg:
            mod.__path__ = []
        mod.__spec__ = importlib.machinery.ModuleSpec(
            name, loader=None, is_package=is_pkg
        )
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["mlx_audio.stt.utils"].load_model = lambda *_a, **_kw: None


def _wav_bytes() -> bytes:
    """Return a tiny valid mono WAV so the upload itself succeeds."""
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
    return buf.getvalue()


class TestAudioModuleImports:
    """Codex r3 false-positive guard: the sanitiser uses ``re.compile``,
    and codex (which only sees the diff) repeatedly flagged this as a
    missing import. ``re`` IS imported at module top (line 6, untouched
    by r6-C), but a future refactor could drop it. This test catches
    that case at CI time so a broken module never reaches the server.
    """

    def test_audio_route_module_imports_cleanly(self):
        """Importing the module must succeed — proves ``re``
        (and every other required import) is wired."""
        import importlib

        # Force a fresh import in case a prior test imported it; if a
        # future refactor breaks the import chain (e.g. by deleting
        # ``import re``), ``importlib.reload`` raises NameError /
        # ModuleNotFoundError and this test fails loudly.
        import vllm_mlx.routes.audio as audio_route

        importlib.reload(audio_route)
        # Spot-check the sanitiser is callable and the regex compiled.
        assert callable(audio_route._sanitize_decode_reason)
        assert audio_route._PATH_LIKE_RE.search("/tmp/foo.wav") is not None


class TestDecodeErrorEnvelopeSanitisation:
    """Codex r2 BLOCKING: the 400 decode envelope must NOT leak server
    filesystem paths in the message. Librosa/soundfile/ffmpeg often
    echo the temp-file path the route created — we sanitise that out
    while keeping the format-shape phrase the client needs.
    """

    def test_sanitiser_strips_quoted_unix_paths(self):
        from vllm_mlx.routes.audio import _sanitize_decode_reason

        # Common librosa shape: "Error opening '/var/folders/.../tmpXYZ.wav': Format not recognised."
        msg = "Error opening '/var/folders/qz/T/tmpXYZ.wav': Format not recognised."
        out = _sanitize_decode_reason(msg)
        assert "/var/folders" not in out, out
        assert "tmpXYZ.wav" not in out, out
        assert "Format not recognised" in out, out

    def test_sanitiser_strips_bare_unix_paths(self):
        from vllm_mlx.routes.audio import _sanitize_decode_reason

        msg = "could not decode audio file: /tmp/audio_xyz.wav header is truncated"
        out = _sanitize_decode_reason(msg)
        assert "/tmp/audio_xyz.wav" not in out, out
        assert "header is truncated" in out, out

    def test_sanitiser_strips_quoted_windows_paths(self):
        from vllm_mlx.routes.audio import _sanitize_decode_reason

        msg = "Error opening 'C:\\Users\\srv\\tmp\\x.wav': Format not recognised."
        out = _sanitize_decode_reason(msg)
        assert "C:\\Users" not in out, out
        assert "Format not recognised" in out, out

    def test_sanitiser_caps_length(self):
        from vllm_mlx.routes.audio import _sanitize_decode_reason

        msg = "x" * 5000
        out = _sanitize_decode_reason(msg)
        assert len(out) <= 240, len(out)
        assert out.endswith("..."), out

    def test_route_envelope_does_not_leak_temp_path(self, monkeypatch):
        """End-to-end via the route: a decode error echoing the temp
        path must reach the client with the path redacted."""
        from vllm_mlx.audio import probe
        from vllm_mlx.routes import audio as audio_route

        _install_fake_mlx_audio(monkeypatch)
        probe._reset_probe_cache()

        class _FakeLeakyEngine:
            def __init__(self, model_name: str):
                self.model_name = model_name

            def load(self):
                pass

            def transcribe(self, audio_path, language=None, task="transcribe"):
                # Mirror what librosa actually raises — the temp path
                # the route created is echoed in the exception message.
                raise RuntimeError(
                    f"Error opening '{audio_path}': Format not recognised."
                )

        monkeypatch.setattr(
            "vllm_mlx.audio.stt.STTEngine", _FakeLeakyEngine, raising=False
        )
        audio_stt_mod = sys.modules.get("vllm_mlx.audio.stt")
        if audio_stt_mod is not None:
            monkeypatch.setattr(audio_stt_mod, "STTEngine", _FakeLeakyEngine)

        audio_route._stt_engine = None
        try:
            client, restore = _mount_audio_app()
            try:
                r = client.post(
                    "/v1/audio/transcriptions",
                    data={"model": "whisper-large-v3"},
                    files={"file": ("tone.wav", _wav_bytes(), "audio/wav")},
                )
            finally:
                restore()
        finally:
            audio_route._stt_engine = None
            probe._reset_probe_cache()

        assert r.status_code == 400, r.text
        body_text = r.text
        # The temp path was created in /var/folders/.../<...>.wav on
        # macOS, /tmp/<...>.wav on Linux. Neither must appear in the
        # envelope.
        for leaked in ("/var/folders", "/tmp/", "tmpXYZ"):
            assert leaked not in body_text, (
                f"Codex r2 BLOCKING regression: 400 envelope leaked "
                f"{leaked!r}: {body_text!r}"
            )
        # The format phrase must still be there for the client.
        body = r.json()
        err = body.get("detail", {}).get("error") or body.get("error")
        assert "Format not recognised" in err["message"], err


class TestServerMisconfigStays500:
    """Codex r2 BLOCKING: messages like "ffmpeg binary not found" or
    "libsndfile not installed" describe SERVER misconfiguration. They
    must NOT downgrade to 400 client-error — operators rely on the
    500 envelope to know the host audio stack is broken."""

    @pytest.mark.parametrize(
        "message",
        [
            "ffmpeg binary not found in PATH",
            "libsndfile not installed",
            "No module named 'audioread'",
            "ffmpeg: command not found",
            # decode-shape phrase but with a misconfig hint mixed in —
            # the misconfig hint wins (server problem, not client).
            "could not open audio: ffmpeg not found",
        ],
    )
    def test_misconfig_stays_500(self, monkeypatch, message):
        from vllm_mlx.audio import probe
        from vllm_mlx.routes import audio as audio_route

        _install_fake_mlx_audio(monkeypatch)
        probe._reset_probe_cache()

        class _FakeMisconfigEngine:
            def __init__(self, model_name: str):
                self.model_name = model_name

            def load(self):
                pass

            def transcribe(self, *_a, **_kw):
                raise RuntimeError(message)

        monkeypatch.setattr(
            "vllm_mlx.audio.stt.STTEngine", _FakeMisconfigEngine, raising=False
        )
        audio_stt_mod = sys.modules.get("vllm_mlx.audio.stt")
        if audio_stt_mod is not None:
            monkeypatch.setattr(audio_stt_mod, "STTEngine", _FakeMisconfigEngine)

        audio_route._stt_engine = None
        try:
            client, restore = _mount_audio_app()
            try:
                r = client.post(
                    "/v1/audio/transcriptions",
                    data={"model": "whisper-large-v3"},
                    files={"file": ("tone.wav", _wav_bytes(), "audio/wav")},
                )
            finally:
                restore()
        finally:
            audio_route._stt_engine = None
            probe._reset_probe_cache()

        assert r.status_code == 500, (
            f"Codex r2 BLOCKING regression: server misconfig "
            f"({message!r}) was downgraded to {r.status_code} client "
            f"error: {r.text!r}. Operators rely on 500 to notice the "
            f"host audio stack is broken."
        )
