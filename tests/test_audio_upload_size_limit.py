# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #193 — audio upload size cap.

The `/v1/audio/transcriptions` endpoint must reject uploads that exceed
`MAX_AUDIO_UPLOAD_SIZE` so that a malicious client cannot exhaust server
memory by streaming a multi-GB file. A normal-sized payload must continue
to flow through to the STT engine.
"""

from __future__ import annotations

import io
import sys
import types
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@dataclass
class _FakeResult:
    text: str = "hello"
    language: str = "en"
    duration: float = 1.0


class _FakeSTTEngine:
    """Stand-in for `vllm_mlx.audio.stt.STTEngine` that records the file it
    was handed but performs no real transcription."""

    instances: list[_FakeSTTEngine] = []

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.loaded = False
        self.transcribed_paths: list[str] = []
        _FakeSTTEngine.instances.append(self)

    def load(self) -> None:
        self.loaded = True

    def transcribe(self, path: str, language: str | None = None) -> _FakeResult:
        self.transcribed_paths.append(path)
        return _FakeResult()


@pytest.fixture
def audio_client(monkeypatch):
    """Build a TestClient mounting only the audio router, with the STT
    engine replaced by an in-process fake so no model is loaded."""

    # Reset the cached module-level engine in routes.audio between tests so
    # the second test does not reuse the first test's fake.
    from vllm_mlx.routes import audio as audio_route

    monkeypatch.setattr(audio_route, "_stt_engine", None, raising=False)

    # Stub the stt submodule import done lazily inside the handler.
    stt_mod = types.ModuleType("vllm_mlx.audio.stt")
    stt_mod.STTEngine = _FakeSTTEngine
    monkeypatch.setitem(sys.modules, "vllm_mlx.audio.stt", stt_mod)
    _FakeSTTEngine.instances.clear()

    app = FastAPI()
    app.include_router(audio_route.router)
    with TestClient(app) as client:
        yield client


def test_oversized_audio_upload_returns_413(audio_client, monkeypatch):
    """A payload above MAX_AUDIO_UPLOAD_SIZE must be rejected with HTTP 413
    *before* the STT engine is ever constructed or loaded.

    This is the regression test for issue #193 — DoS via memory exhaustion
    on the audio transcription endpoint."""

    from vllm_mlx.routes import audio as audio_route

    # Shrink the cap so the test stays fast and memory-light while still
    # exercising the streaming guard.
    monkeypatch.setattr(audio_route, "MAX_AUDIO_UPLOAD_SIZE", 1024, raising=True)

    oversized = io.BytesIO(b"\x00" * 4096)  # 4 KB, well above the 1 KB cap
    resp = audio_client.post(
        "/v1/audio/transcriptions",
        files={"file": ("big.wav", oversized, "audio/wav")},
        data={"model": "whisper-small"},
    )

    assert resp.status_code == 413, resp.text
    assert "too large" in resp.json()["detail"].lower()
    # No engine was constructed — the size check ran before the lazy import
    # and `STTEngine(model_name).load()` call. This is the property that
    # prevents an attacker from forcing model load just by advertising a
    # huge Content-Length.
    assert _FakeSTTEngine.instances == []
    assert audio_route._stt_engine is None


def test_streaming_cap_rejects_chunked_upload_before_engine_load(monkeypatch):
    """Direct unit test of the streaming cap — covers the chunked/no-
    Content-Length / understated-Content-Length attack vector that the
    TestClient-level test cannot exercise (TestClient always sets a
    truthful Content-Length).

    A fake UploadFile yields more bytes than the cap allows; we assert:
      * the handler raises HTTPException(413)
      * no STTEngine was ever constructed (no model load on the DoS path)
      * the temp file written so far was cleaned up
    """
    import os

    from fastapi import HTTPException
    from starlette.datastructures import Headers
    from starlette.requests import Request

    from vllm_mlx.routes import audio as audio_route

    monkeypatch.setattr(audio_route, "MAX_AUDIO_UPLOAD_SIZE", 1024, raising=True)
    monkeypatch.setattr(audio_route, "_stt_engine", None, raising=False)

    # Stub the engine import so a regression that *did* load the engine
    # would be visible via _FakeSTTEngine.instances.
    stt_mod = types.ModuleType("vllm_mlx.audio.stt")
    stt_mod.STTEngine = _FakeSTTEngine
    monkeypatch.setitem(sys.modules, "vllm_mlx.audio.stt", stt_mod)
    _FakeSTTEngine.instances.clear()

    class _LyingChunkedUpload:
        """Mimics the slice of `UploadFile` the handler touches. Reports
        `size = None` (chunked encoding semantics) but actually streams
        well past the cap when `.read()` is called."""

        size = None
        filename = "evil.wav"
        content_type = "audio/wav"

        def __init__(self, total_bytes: int, chunk: int = 512):
            self._remaining = total_bytes
            self._chunk = chunk
            self.read_calls = 0

        async def read(self, size: int = -1) -> bytes:
            self.read_calls += 1
            if self._remaining <= 0:
                return b""
            take = self._chunk if size < 0 else min(size, self._chunk)
            take = min(take, self._remaining)
            self._remaining -= take
            return b"\x00" * take

    fake_upload = _LyingChunkedUpload(total_bytes=8192)  # 8 KB > 1 KB cap

    # Build a Request whose Content-Length is absent / lying.
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/audio/transcriptions",
        "headers": Headers({}).raw,  # no content-length advertised
        "query_string": b"",
    }
    request = Request(scope)

    # Snapshot temp dir so we can assert no temp file leaked.
    import tempfile as _tf

    before = set(os.listdir(_tf.gettempdir()))

    import asyncio

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            audio_route.create_transcription(
                request=request,
                file=fake_upload,  # type: ignore[arg-type]
                model="whisper-small",
            )
        )

    assert exc_info.value.status_code == 413
    # The engine was never constructed — the streaming cap fired before
    # the import + load() block at the bottom of the handler.
    assert _FakeSTTEngine.instances == []
    assert audio_route._stt_engine is None

    # No leaked .wav temp file — the finally-block cleaned up.
    after = set(os.listdir(_tf.gettempdir()))
    leaked = [n for n in (after - before) if n.endswith(".wav")]
    assert leaked == [], f"temp .wav files leaked on rejection path: {leaked}"


def test_normal_audio_upload_succeeds(audio_client, monkeypatch):
    """A small payload (within the cap) must reach the STT engine and
    return a JSON transcription response. Positive control to confirm
    the size guard did not break the happy path."""

    from vllm_mlx.routes import audio as audio_route

    monkeypatch.setattr(audio_route, "MAX_AUDIO_UPLOAD_SIZE", 1024, raising=True)

    small = io.BytesIO(b"RIFFsmall-wav-bytes")  # 19 bytes, well under the cap
    resp = audio_client.post(
        "/v1/audio/transcriptions",
        files={"file": ("ok.wav", small, "audio/wav")},
        data={"model": "whisper-small"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["text"] == "hello"
    assert body["language"] == "en"
    # Exactly one fake engine was constructed, and it received the file.
    assert len(_FakeSTTEngine.instances) == 1
    assert len(_FakeSTTEngine.instances[0].transcribed_paths) == 1
