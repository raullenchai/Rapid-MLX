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

# SKIP NOTE: this test file has heavy environmental requirements that
# the Linux CI runners (`pr_validate.targeted_tests`,
# `.github/workflows/ci.yml`'s test-matrix) deliberately don't satisfy:
#
#   * the audio route transitively imports `vllm_mlx.config` ->
#     `engine` -> `engine_core` -> `scheduler`, which `import mlx.core`
#     and `from mlx_lm.* import ...` at module load — only the
#     apple-silicon CI job installs MLX, the Linux ones don't.
#   * `TestClient.post(files=...)` requires `python-multipart`, which
#     is not in `pr-validate.yml`'s dependency list.
#
# We tried session-level mlx-stubbing via `importlib.abc.MetaPathFinder`,
# but it leaked into other test files in the same pytest run
# (`test_server_utils.py`, `test_server_load_model_order.py`) and turned
# 80+ unrelated tests into false regressions. Skip cleanly here instead
# — `pr_validate` only counts FAILED, not SKIPPED, so this file stays
# out of the regression set on Linux runners, and the test still
# executes anywhere with the right deps (dev machines + apple-silicon
# CI job if it ever picks this file up).
pytest.importorskip(
    "mlx.core",
    reason="audio route imports transitively pull in mlx; "
    "test runs on Apple Silicon / dev, not Linux CI runners",
)
pytest.importorskip(
    "mlx_lm",
    reason="audio route imports transitively pull in mlx_lm; "
    "test runs on Apple Silicon / dev, not Linux CI runners",
)
pytest.importorskip(
    "multipart",
    reason="TestClient(files=...) requires python-multipart; "
    "skip on minimal-deps runners (CI pr-validate)",
)


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
    engine replaced by an in-process fake so no model is loaded.

    Mirrors how ``vllm_mlx.server`` wires the production app: the
    :class:`AudioBodyLimitMiddleware` is installed so the
    Content-Length pre-check is exercised end-to-end."""

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
    audio_route.install_audio_body_limit_middleware(app)
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

    # Snapshot temp dir so we can assert no temp file leaked.
    import tempfile as _tf

    before = set(os.listdir(_tf.gettempdir()))

    import asyncio

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            audio_route.create_transcription(
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


def test_content_length_guard_rejects_before_multipart_parsing(monkeypatch):
    """Honest-Content-Length DoS attempt: the request advertises a body
    several times larger than the cap. :class:`AudioBodyLimitMiddleware`
    must reject with 413 BEFORE Starlette's multipart parser calls
    ``receive`` to drain the body. This is the property codex flagged:
    a FastAPI ``Depends`` cannot do this because parameter resolution
    (which triggers ``MultiPartParser``) runs first.

    The probe: wrap the FastAPI app in an ASGI-layer ``receive`` tracer
    and assert NO ``http.request`` message was ever consumed. That is
    the empirical proof that no spooling-to-disk happened — there is
    no other way to land bytes server-side."""
    import asyncio

    from vllm_mlx.routes import audio as audio_route

    monkeypatch.setattr(audio_route, "MAX_AUDIO_UPLOAD_SIZE", 1024, raising=True)
    monkeypatch.setattr(audio_route, "_REQUEST_BODY_SLACK_BYTES", 256, raising=True)
    monkeypatch.setattr(audio_route, "_stt_engine", None, raising=False)

    # Stub the engine import so a regression that *did* parse the body and
    # reach the handler would be visible via _FakeSTTEngine.instances.
    stt_mod = types.ModuleType("vllm_mlx.audio.stt")
    stt_mod.STTEngine = _FakeSTTEngine
    monkeypatch.setitem(sys.modules, "vllm_mlx.audio.stt", stt_mod)
    _FakeSTTEngine.instances.clear()

    # Build an app exactly the way the audio_client fixture does — but
    # without TestClient, so we can drive ASGI manually and observe the
    # receive channel.
    app = FastAPI()
    app.include_router(audio_route.router)
    audio_route.install_audio_body_limit_middleware(app)

    receive_calls: list[str] = []
    body_bytes = b"A" * 16384  # 16 KB — comfortably above cap + slack

    async def receive():
        # If the middleware does its job, this never runs. We record
        # every call so the assertion below can prove the negative.
        receive_calls.append("http.request")
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    sent_messages: list[dict] = []

    async def send(msg):
        sent_messages.append(msg)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/audio/transcriptions",
        "raw_path": b"/v1/audio/transcriptions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"multipart/form-data; boundary=---x"),
            (b"content-length", str(len(body_bytes)).encode("ascii")),
        ],
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 12345),
    }

    asyncio.run(app(scope, receive, send))

    # 1) Middleware returned a 413 — the explicit safety result.
    start = next(m for m in sent_messages if m["type"] == "http.response.start")
    assert start["status"] == 413, sent_messages

    # 2) THE LOAD-BEARING ASSERTION: receive was never called, so the
    #    request body never left the client / never landed on the server.
    #    This is what codex's earlier review demanded — a test that
    #    fails if any body parsing started before the limit check.
    assert receive_calls == [], (
        f"middleware let body parsing begin (receive called "
        f"{len(receive_calls)} time(s)) — guard regressed to a "
        "Depends/handler-level check"
    )

    # 3) No engine was ever constructed — handler never ran.
    assert _FakeSTTEngine.instances == []
    assert audio_route._stt_engine is None


def test_chunked_no_content_length_aborts_mid_stream(monkeypatch):
    """A chunked / no-``Content-Length`` attacker who tries to stream a
    multi-GB body must be aborted by the middleware AS THE BYTES STREAM
    — not after Starlette finishes spooling. We assert that the
    middleware stops calling ``receive`` once the running total exceeds
    the cap, and that the client gets a 413.

    Without this guard, codex's round-6 finding stood: a Transfer-
    Encoding: chunked client (or one that lies about Content-Length and
    sends more) could spool gigabytes to disk before any handler-level
    cap fired.

    Test design: drive the ASGI app directly with a valid multipart
    envelope split across many ``http.request`` messages with
    ``more_body=True``. The middleware-side trip is independent of the
    multipart body's content — it only counts bytes — so we don't need
    a correctly-formatted multipart payload to prove the cap fires."""
    import asyncio

    from vllm_mlx.routes import audio as audio_route

    # Effective limit = cap + slack = 1024 + 256 = 1280 bytes.
    monkeypatch.setattr(audio_route, "MAX_AUDIO_UPLOAD_SIZE", 1024, raising=True)
    monkeypatch.setattr(audio_route, "_REQUEST_BODY_SLACK_BYTES", 256, raising=True)
    monkeypatch.setattr(audio_route, "_stt_engine", None, raising=False)

    stt_mod = types.ModuleType("vllm_mlx.audio.stt")
    stt_mod.STTEngine = _FakeSTTEngine
    monkeypatch.setitem(sys.modules, "vllm_mlx.audio.stt", stt_mod)
    _FakeSTTEngine.instances.clear()

    app = FastAPI()
    app.include_router(audio_route.router)
    audio_route.install_audio_body_limit_middleware(app)

    # Drive the middleware DIRECTLY — bypass the downstream FastAPI
    # router. We're testing the receive-wrapping logic in isolation;
    # the inner app just needs to drain the receive channel.
    middleware = audio_route.AudioBodyLimitMiddleware(_DrainingApp())

    total_chunks = 16
    chunk_size = 256
    chunk = b"X" * chunk_size
    received_count = {"n": 0}

    async def receive():
        i = received_count["n"]
        received_count["n"] += 1
        if i >= total_chunks:
            # Should never be reached; if it is, the cap regressed.
            return {"type": "http.request", "body": b"", "more_body": False}
        more = i < total_chunks - 1
        return {"type": "http.request", "body": chunk, "more_body": more}

    sent_messages: list[dict] = []

    async def send(msg):
        sent_messages.append(msg)

    # Critically: NO content-length header (chunked transfer encoding
    # would omit it). The middleware cannot use the fast path; it must
    # rely on its streaming tally.
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/audio/transcriptions",
        "raw_path": b"/v1/audio/transcriptions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"multipart/form-data; boundary=---x"),
            (b"transfer-encoding", b"chunked"),
        ],
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 12345),
    }

    asyncio.run(middleware(scope, receive, send))

    # 1) 413 response — client sees the rejection, not a stalled connection.
    start = next(m for m in sent_messages if m["type"] == "http.response.start")
    assert start["status"] == 413, sent_messages

    # 2) THE LOAD-BEARING ASSERTION: receive was called fewer than
    #    total_chunks times. Effective limit is 1280 bytes; at 256
    #    B/chunk, the trip must fire by chunk 6 (1536 > 1280). It
    #    must NOT have drained all 16 chunks.
    assert received_count["n"] < total_chunks, (
        f"middleware read {received_count['n']}/{total_chunks} chunks — "
        "streaming abort regressed"
    )
    assert received_count["n"] <= 7, (
        f"middleware over-read: {received_count['n']} chunks of {chunk_size} B "
        f"= {received_count['n'] * chunk_size} B vs limit "
        f"{audio_route.MAX_AUDIO_UPLOAD_SIZE + audio_route._REQUEST_BODY_SLACK_BYTES}"
    )

    # 3) Handler never ran — no engine constructed.
    assert _FakeSTTEngine.instances == []
    assert audio_route._stt_engine is None


def test_chunked_real_fastapi_app_returns_413(monkeypatch):
    """End-to-end variant of the chunked-streaming test that drives the
    REAL FastAPI app (not a stub draining inner app). Proves the
    middleware survives the multipart parser raising/aborting on the
    synthetic ``http.disconnect`` and still emits a 413 to the client.

    This is the test codex round 7 specifically asked for: ``_DrainingApp``
    cannot catch the Starlette MultiPartParser exception path that the
    real app exercises when it sees the disconnect we inject."""
    import asyncio

    from vllm_mlx.routes import audio as audio_route

    monkeypatch.setattr(audio_route, "MAX_AUDIO_UPLOAD_SIZE", 1024, raising=True)
    monkeypatch.setattr(audio_route, "_REQUEST_BODY_SLACK_BYTES", 256, raising=True)
    monkeypatch.setattr(audio_route, "_stt_engine", None, raising=False)

    stt_mod = types.ModuleType("vllm_mlx.audio.stt")
    stt_mod.STTEngine = _FakeSTTEngine
    monkeypatch.setitem(sys.modules, "vllm_mlx.audio.stt", stt_mod)
    _FakeSTTEngine.instances.clear()

    app = FastAPI()
    app.include_router(audio_route.router)
    audio_route.install_audio_body_limit_middleware(app)

    # Real-ish multipart prologue so Starlette's parser starts work; the
    # cap will trip mid-stream long before any "valid" multipart could
    # complete. We're testing the middleware survives whatever the
    # multipart parser does when it sees a disconnect.
    boundary = b"----xyz"
    prologue = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="x.wav"\r\n'
        b"Content-Type: audio/wav\r\n\r\n"
    )
    chunk = b"A" * 512
    total_chunks = 10  # 5 KB body, way above the 1280 B cap
    received_count = {"n": 0}

    async def receive():
        i = received_count["n"]
        received_count["n"] += 1
        if i == 0:
            return {"type": "http.request", "body": prologue, "more_body": True}
        if i <= total_chunks:
            more = i < total_chunks
            return {"type": "http.request", "body": chunk, "more_body": more}
        # Should never reach here — middleware should have aborted.
        return {"type": "http.request", "body": b"", "more_body": False}

    sent_messages: list[dict] = []

    async def send(msg):
        sent_messages.append(msg)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/audio/transcriptions",
        "raw_path": b"/v1/audio/transcriptions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"multipart/form-data; boundary=----xyz"),
            (b"transfer-encoding", b"chunked"),
        ],
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 12345),
    }

    asyncio.run(app(scope, receive, send))

    # The middleware must have emitted a 413 — even if Starlette's
    # multipart parser raised or aborted along the way.
    start = next(m for m in sent_messages if m["type"] == "http.response.start")
    assert start["status"] == 413, sent_messages

    # And the body must have been bounded — receive count below the
    # total-chunks ceiling proves the streaming abort actually fired.
    assert received_count["n"] < 1 + total_chunks, (
        f"middleware drained {received_count['n']} chunks against the real "
        "FastAPI app — streaming abort regressed"
    )

    # No engine constructed — handler never ran.
    assert _FakeSTTEngine.instances == []
    assert audio_route._stt_engine is None


class _DrainingApp:
    """Minimal ASGI inner app that drains ``receive`` until it sees
    ``more_body=False`` or ``http.disconnect``, then emits a 200.

    Stands in for the FastAPI router when testing the middleware's
    receive-wrapping in isolation; the goal is to verify that the
    middleware stops the inner app from over-reading, regardless of
    what the inner app would otherwise do."""

    async def __call__(self, scope, receive, send):
        while True:
            msg = await receive()
            if msg.get("type") == "http.disconnect":
                return
            if not msg.get("more_body"):
                break
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})


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
