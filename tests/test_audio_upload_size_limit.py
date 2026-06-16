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


def _ensure_mlx_stubs() -> None:
    """Insert minimal `mlx` / `mlx_lm` stubs so `vllm_mlx.routes.audio`
    can be imported on hosts without MLX installed (Linux CI runners).

    The audio route itself never touches mlx, but it transitively imports
    `vllm_mlx.config` -> `engine` -> `engine_core` -> `scheduler` etc.,
    which `import mlx.core` and `from mlx_lm.* import ...` at module top.
    We don't exercise any of that code in these tests — we only need the
    import chain to succeed so the streaming-cap branch of the audio
    handler runs.

    `pr_validate.targeted_tests` runs on Linux CI without MLX in scope,
    so without these stubs the test ERRORs at collection time and is
    misread as a regression."""
    import importlib.abc
    import importlib.machinery

    def _noop(*_a, **_kw):  # generic no-op stand-in
        return None

    class _StubBase:
        """Stand-in usable as both a callable and a base class.

        Some downstream sites do ``class Foo(<stub>):`` (e.g.
        ``vllm_mlx/utils/mamba_cache.py``) so the stub has to be a real
        class, not a function."""

        def __init__(self, *_a, **_kw):
            pass

        def __call__(self, *_a, **_kw):
            return self

        def __getattr__(self, name: str):  # noqa: D401
            if name.startswith("__"):
                raise AttributeError(name)
            return _StubBase()

    class _AutoModule(types.ModuleType):
        """Module whose attribute access auto-creates a stub class.

        Lets every ``from <stub_pkg>.X import Y`` resolve — including
        ``class Subclass(Y):`` patterns — without our having to
        enumerate every helper used downstream."""

        def __getattr__(self, name: str):  # noqa: D401
            if name.startswith("__"):
                raise AttributeError(name)
            stub = type(name, (_StubBase,), {})
            setattr(self, name, stub)
            return stub

    _STUB_ROOTS = ("mlx", "mlx_lm", "mlx_vlm", "mlx_audio")

    class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        """Resolve any ``mlx`` / ``mlx_lm`` / ``mlx_vlm`` / ``mlx_audio``
        submodule to an empty ``_AutoModule`` so submodule imports don't
        ``ModuleNotFoundError`` on Linux CI."""

        def find_spec(self, fullname, path, target=None):
            root = fullname.split(".", 1)[0]
            if root not in _STUB_ROOTS:
                return None
            # Don't override real installs (e.g. dev machine).
            if fullname in sys.modules:
                return None
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

        def create_module(self, spec):
            return _AutoModule(spec.name)

        def exec_module(self, module):
            module.__path__ = []  # mark as package so deeper imports work

    # Only install the finder if mlx truly isn't importable — dev
    # machines with real mlx installed should use the real package.
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        sys.meta_path.append(_StubFinder())

    # Pre-populate mlx.core with the specific attributes engine_core /
    # scheduler / _mlx_compat read at module load (auto-stub returns
    # _noop for everything; some sites want a real-ish object).
    def _ensure(name: str) -> types.ModuleType:
        mod = sys.modules.get(name)
        if mod is None:
            mod = _AutoModule(name)
            mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = mod
        return mod

    if "mlx" not in sys.modules:
        _ensure("mlx")
    if "mlx.core" not in sys.modules:
        mlx_core = _ensure("mlx.core")
        sys.modules["mlx"].core = mlx_core  # type: ignore[attr-defined]
    else:
        mlx_core = sys.modules["mlx.core"]

    class _Stream:  # placeholder for mx.Stream
        pass

    # Only set attributes we know are sampled at import time and need
    # a non-None value; everything else falls through to _noop via
    # _AutoModule.__getattr__.
    if not hasattr(mlx_core, "Stream"):
        mlx_core.Stream = _Stream
    if not hasattr(mlx_core, "metal"):
        mlx_core.metal = types.SimpleNamespace(
            set_memory_limit=_noop,
            set_cache_limit=_noop,
            get_active_memory=lambda: 0,
            get_peak_memory=lambda: 0,
            get_cache_memory=lambda: 0,
        )


_ensure_mlx_stubs()


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
