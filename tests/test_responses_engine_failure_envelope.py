# SPDX-License-Identifier: Apache-2.0
"""r6-A R6-C2 regression guard — engine failure must surface as
``status="failed"`` + populated ``error`` block, not silent
``status="incomplete"`` + ``usage=0/0/0``.

Background: Sasha R1 / R2 dogfood reports captured the same response
shape on every metal::malloc wedge:

  HTTP/1.1 200 OK
  {"status":"incomplete","output":[],"usage":{"input_tokens":0,
   "output_tokens":0,"total_tokens":0}}

SDK consumers could not distinguish that response from a legitimate
"tiny budget, model truncated mid-reply" response — the engine error
was being silently swallowed. This guard pins the failure-envelope
shape so a future regression that re-introduces the silent path
trips a CI failure rather than waiting for another dogfood report.
"""

import json
import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _Tokenizer:
    chat_template = ""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class _BaseEngine:
    pass


@dataclass
class _GenerationOutput:
    text: str = ""
    raw_text: str = ""
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = "length"  # the metal::malloc wedge shape
    new_text: str = ""
    finished: bool = True
    logprobs: Any = None
    channel: str | None = None
    tool_calls: list | None = None
    reasoning_text: str = ""


class _FailingEngine:
    """Engine shim that mimics the metal::malloc wedge: returns an empty
    GenerationOutput with no text / reasoning / tool_calls and zero
    completion_tokens. ``finish_reason="length"`` is what the scheduler
    surfaces when the request aborts before its first token (the
    ``num_tokens >= max_tokens`` check fires)."""

    preserve_native_tool_format = False

    def __init__(self):
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        return _GenerationOutput(
            text="",
            prompt_tokens=42,
            completion_tokens=0,
            finish_reason="length",
        )

    async def stream_chat(self, messages, **kwargs):
        """Stream variant: ONE chunk with no new text and zero
        completion tokens — the scheduler abort signature."""
        yield _GenerationOutput(
            text="",
            new_text="",
            prompt_tokens=42,
            completion_tokens=0,
            finish_reason="length",
            finished=True,
        )


class _HealthyEngine:
    """Control engine — produces actual output so the failure guard
    is proven to be narrow (only fires on degenerate outputs)."""

    preserve_native_tool_format = False

    def __init__(self):
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        return _GenerationOutput(
            text="ok",
            prompt_tokens=3,
            completion_tokens=1,
            finish_reason="stop",
        )

    async def stream_chat(self, messages, **kwargs):
        yield _GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=3,
            completion_tokens=1,
            finish_reason="stop",
        )


_IMPORTED = (
    "vllm_mlx.config",
    "vllm_mlx.config.server_config",
    "vllm_mlx.engine",
    "vllm_mlx.engine.base",
    "vllm_mlx.middleware.auth",
    "vllm_mlx.service.helpers",
    "vllm_mlx.routes.responses",
)
_PARENT_ATTRS = (
    ("vllm_mlx", "config"),
    ("vllm_mlx", "engine"),
    ("vllm_mlx.config", "server_config"),
    ("vllm_mlx.engine", "base"),
    ("vllm_mlx.middleware", "auth"),
    ("vllm_mlx.service", "helpers"),
    ("vllm_mlx.routes", "responses"),
)
_MISSING = object()


def _install_lightweight_engine_modules(monkeypatch):
    engine_pkg = types.ModuleType("vllm_mlx.engine")
    engine_pkg.BaseEngine = _BaseEngine
    engine_pkg.GenerationOutput = _GenerationOutput

    base_mod = types.ModuleType("vllm_mlx.engine.base")
    base_mod.BaseEngine = _BaseEngine
    base_mod.GenerationOutput = _GenerationOutput

    monkeypatch.setitem(sys.modules, "vllm_mlx.engine", engine_pkg)
    monkeypatch.setitem(sys.modules, "vllm_mlx.engine.base", base_mod)


def _build_client(monkeypatch, engine_factory):
    previous_modules = {n: sys.modules.get(n, _MISSING) for n in _IMPORTED}
    previous_attrs = {}
    for module_name, attr in _PARENT_ATTRS:
        module = sys.modules.get(module_name)
        previous_attrs[(module_name, attr)] = (
            getattr(module, attr, _MISSING) if module is not None else _MISSING
        )

    _install_lightweight_engine_modules(monkeypatch)

    from vllm_mlx.config import reset_config
    from vllm_mlx.middleware.auth import rate_limiter
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes.responses import router

    cfg = reset_config()
    cfg.api_key = "test-secret"
    cfg.engine = engine_factory()
    cfg.model_name = "test-model"
    cfg.model_registry = None

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    return SimpleNamespace(
        client=TestClient(app),
        engine=cfg.engine,
        cleanup=lambda: _cleanup(previous_modules, previous_attrs),
    )


def _cleanup(previous_modules, previous_attrs):
    from vllm_mlx.config import reset_config
    from vllm_mlx.middleware.auth import rate_limiter

    reset_config()
    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()
    for name, previous in previous_modules.items():
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    for (module_name, attr), previous in previous_attrs.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if previous is _MISSING:
            if hasattr(module, attr):
                delattr(module, attr)
        else:
            setattr(module, attr, previous)


@pytest.fixture
def failing_client(monkeypatch):
    holder = _build_client(monkeypatch, _FailingEngine)
    yield holder
    holder.cleanup()


@pytest.fixture
def healthy_client(monkeypatch):
    holder = _build_client(monkeypatch, _HealthyEngine)
    yield holder
    holder.cleanup()


HEADERS = {"Authorization": "Bearer test-secret"}
PAYLOAD = {"model": "test-model", "input": "Hi"}


def _parse_sse(body_text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name = None
        data_text = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_text = line[len("data:") :].strip()
        if event_name and data_text is not None:
            events.append((event_name, json.loads(data_text)))
    return events


# =============================================================================
# Non-stream — engine wedge → status="failed" + error block
# =============================================================================


class TestResponsesNonStreamFailureEnvelope:
    def test_engine_no_output_surfaces_status_failed(self, failing_client):
        """Pre-fix: HTTP 200 + ``status="incomplete"`` + zero usage. Post-fix:
        HTTP 200 + ``status="failed"`` + populated ``error`` block."""
        resp = failing_client.client.post(
            "/v1/responses", json=PAYLOAD, headers=HEADERS
        )
        # We deliberately return 200 (not 500) — the engine produced a
        # well-formed (if empty) GenerationOutput; the failure is
        # semantic, not transport-level. Matches the OpenAI Responses
        # cloud behaviour on stream-errored requests.
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "failed", (
            f"expected status='failed', got {body['status']!r}. Body: {body}"
        )

    def test_failed_envelope_carries_error_block(self, failing_client):
        """OpenAI Responses spec puts the failure detail in an ``error``
        block ``{code, message}``. Pre-fix, this field was empty even
        when the underlying engine error was unambiguous."""
        resp = failing_client.client.post(
            "/v1/responses", json=PAYLOAD, headers=HEADERS
        )
        body = resp.json()
        assert "error" in body, f"failed envelope missing 'error': {body}"
        err = body["error"]
        assert err["code"] == "engine_no_output", err
        assert err["message"], err

    def test_failed_envelope_preserves_prompt_token_usage(self, failing_client):
        """Even on engine failure, ``prompt_tokens`` is information the
        client paid for (the prompt was tokenized + sent through the
        pipeline). Mirror chat-completion's usage shape: ``input_tokens``
        carries the prompt count, ``output_tokens=0``."""
        resp = failing_client.client.post(
            "/v1/responses", json=PAYLOAD, headers=HEADERS
        )
        body = resp.json()
        usage = body["usage"]
        assert usage["input_tokens"] == 42, usage
        assert usage["output_tokens"] == 0, usage
        assert usage["total_tokens"] == 42, usage

    def test_healthy_engine_does_not_trip_failure_guard(self, healthy_client):
        """Narrowness check: the failure guard MUST NOT fire when the
        engine produced any user-visible output. A budget=1 reply
        returning a single ``"ok"`` token must round-trip as
        ``status="completed"``, not ``"failed"``."""
        resp = healthy_client.client.post(
            "/v1/responses", json=PAYLOAD, headers=HEADERS
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "completed", body
        assert "error" not in body, body
        assert body["usage"]["output_tokens"] >= 1, body


# =============================================================================
# Stream — engine wedge → response.failed instead of response.completed
# =============================================================================


class TestResponsesStreamFailureEnvelope:
    def test_stream_emits_response_failed_on_engine_no_output(self, failing_client):
        """When the stream produces no text deltas AND zero completion
        tokens, the terminal event must be ``response.failed`` (not
        ``response.completed`` with ``status="completed"``)."""
        with failing_client.client.stream(
            "POST",
            "/v1/responses",
            json={**PAYLOAD, "stream": True},
            headers=HEADERS,
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        names = [n for n, _ in events]
        assert "response.failed" in names, (
            f"response.failed missing on engine-wedge stream. Events: {names}"
        )
        assert "response.completed" not in names, (
            f"response.completed AND response.failed both emitted — "
            f"the terminal event must be the failure shape. Events: {names}"
        )

    def test_stream_failed_event_carries_error_block(self, failing_client):
        """``response.failed`` payload must echo the same ``{code,
        message}`` error block the non-stream surface uses."""
        with failing_client.client.stream(
            "POST",
            "/v1/responses",
            json={**PAYLOAD, "stream": True},
            headers=HEADERS,
        ) as resp:
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        failed = [d for (n, d) in events if n == "response.failed"]
        assert failed, "no response.failed event"
        envelope = failed[0]["response"]
        assert envelope["status"] == "failed", envelope
        assert envelope["error"]["code"] == "engine_no_output", envelope
        assert envelope["error"]["message"], envelope

    def test_healthy_stream_does_not_trip_failure_guard(self, healthy_client):
        """Same narrowness check on the streaming surface: a one-token
        healthy reply must close with ``response.completed``, not the
        failure event."""
        with healthy_client.client.stream(
            "POST",
            "/v1/responses",
            json={**PAYLOAD, "stream": True},
            headers=HEADERS,
        ) as resp:
            body = "".join(resp.iter_text())
        names = [n for n, _ in _parse_sse(body)]
        assert "response.completed" in names, names
        assert "response.failed" not in names, names
