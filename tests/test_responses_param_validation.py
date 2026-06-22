# SPDX-License-Identifier: Apache-2.0
"""r6-A R6-H8 regression guard — ``/v1/responses`` honours the same
``top_k`` / ``seed`` validators the chat completion surface uses.

Background: r5-E B-7 (PR #824) added an upper-bound cap on ``top_k``
(``_TOP_K_SENTINEL_CAP = 1 << 20``) and a negative-rejection on
``seed`` to ``ChatCompletionRequest`` + ``CompletionRequest``. The
Responses surface bypassed both — ``ResponsesRequest`` didn't even
declare ``top_k`` so Pydantic silently dropped pathological values
(``top_k=999_999_999``) and returned HTTP 200 with no validation
signal to the client.

This test pins the contract that the three OpenAI-shape surfaces
(/v1/chat/completions, /v1/responses, /v1/completions) share one
validator module — no copy-pasted thresholds, no surface-specific
escape hatches.
"""

import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Lightweight engine shim — mirrors the one in tests/test_responses_route.py
# (kept inline so a future refactor of either fixture doesn't break this
# regression guard).


class _Tokenizer:
    chat_template = ""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class _BaseEngine:
    pass


@dataclass
class _GenerationOutput:
    text: str
    raw_text: str = ""
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = "stop"
    new_text: str = ""
    finished: bool = True
    logprobs: Any = None
    channel: str | None = None
    tool_calls: list | None = None
    reasoning_text: str = ""


class _Engine:
    preserve_native_tool_format = False

    def __init__(self):
        self.calls: list[SimpleNamespace] = []
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        self.calls.append(SimpleNamespace(messages=messages, kwargs=kwargs))
        return _GenerationOutput(
            text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            finish_reason="stop",
        )

    async def stream_chat(self, messages, **kwargs):
        yield _GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=1,
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


@pytest.fixture
def responses_client(monkeypatch):
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
    cfg.engine = _Engine()
    cfg.model_name = "test-model"
    cfg.model_registry = None

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), engine=cfg.engine)

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


def _payload(**overrides):
    base = {"model": "test-model", "input": "Hello, world"}
    base.update(overrides)
    return base


HEADERS = {"Authorization": "Bearer test-secret"}


# =============================================================================
# top_k — upper bound + by-design escape hatches
# =============================================================================


class TestResponsesTopK:
    def test_pathological_top_k_rejected_with_400(self, responses_client):
        """r5-E B-7 / r6-A R6-H8: ``top_k=999_999_999`` must produce a
        Pydantic 400 the same way it does on /v1/chat/completions. Pre-fix,
        the Responses surface dropped the field silently (HTTP 200,
        ``ResponsesRequest`` had no ``top_k`` slot)."""
        client = responses_client.client
        resp = client.post(
            "/v1/responses",
            json=_payload(top_k=999_999_999),
            headers=HEADERS,
        )
        assert resp.status_code == 400, resp.text
        body = resp.json()
        # Error envelope shape matches the sanitized 400 path
        # (``install_exception_handlers`` converts Pydantic errors).
        assert "top_k" in resp.text, (
            f"400 response does not mention 'top_k': {resp.text}"
        )

    def test_top_k_zero_accepted_by_design(self, responses_client):
        """``top_k=0`` is the documented "disabled" sentinel on mlx-lm —
        chat surface preserves it as legal, Responses must mirror."""
        client = responses_client.client
        resp = client.post(
            "/v1/responses",
            json=_payload(top_k=0),
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text

    def test_top_k_in_normal_range_accepted(self, responses_client):
        """Sanity: a realistic ``top_k=50`` round-trips fine."""
        client = responses_client.client
        resp = client.post(
            "/v1/responses",
            json=_payload(top_k=50),
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text

    def test_top_k_negative_rejected(self, responses_client):
        """``top_k=-5`` is a serialization bug — mirror the chat surface
        rejection."""
        client = responses_client.client
        resp = client.post(
            "/v1/responses",
            json=_payload(top_k=-5),
            headers=HEADERS,
        )
        assert resp.status_code == 400, resp.text

    def test_top_k_boolean_rejected(self, responses_client):
        """``top_k=True`` would silently coerce to 1 under Pydantic v2
        without the shared validator (which explicitly rejects bool)."""
        client = responses_client.client
        resp = client.post(
            "/v1/responses",
            json=_payload(top_k=True),
            headers=HEADERS,
        )
        assert resp.status_code == 400, resp.text

    def test_top_k_at_sentinel_cap_accepted(self, responses_client):
        """The exact ``_TOP_K_SENTINEL_CAP`` value (``1 << 20`` =
        1,048,576) is INCLUSIVE — both surfaces accept it (just past
        the cap is the failure point)."""
        client = responses_client.client
        resp = client.post(
            "/v1/responses",
            json=_payload(top_k=1 << 20),
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text

    def test_top_k_just_past_cap_rejected(self, responses_client):
        """One past the inclusive cap must 400 — pins the boundary."""
        client = responses_client.client
        resp = client.post(
            "/v1/responses",
            json=_payload(top_k=(1 << 20) + 1),
            headers=HEADERS,
        )
        assert resp.status_code == 400, resp.text


# =============================================================================
# seed — negative-rejection parity with chat
# =============================================================================


class TestResponsesSeed:
    def test_negative_seed_rejected(self, responses_client):
        """``seed=-1`` must produce a Pydantic 400 — H-11 + r5-E B-8
        landed this on chat/responses already; this guard pins it on
        the Responses surface alongside the new top_k validator."""
        client = responses_client.client
        resp = client.post(
            "/v1/responses",
            json=_payload(seed=-1),
            headers=HEADERS,
        )
        assert resp.status_code == 400, resp.text

    def test_seed_zero_accepted(self, responses_client):
        """``seed=0`` is a legitimate PRNG key (eval harnesses); must
        round-trip 200."""
        client = responses_client.client
        resp = client.post(
            "/v1/responses",
            json=_payload(seed=0),
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text

    def test_seed_bool_rejected(self, responses_client):
        """``seed=True`` would coerce to 1 under Pydantic v2 — the
        shared validator rejects it."""
        client = responses_client.client
        resp = client.post(
            "/v1/responses",
            json=_payload(seed=True),
            headers=HEADERS,
        )
        assert resp.status_code == 400, resp.text


# =============================================================================
# top_k forwarding — once validated, it must reach the engine kwargs
# =============================================================================


class TestResponsesTopKForwarded:
    def test_top_k_threaded_to_engine_chat_kwargs(self, responses_client):
        """The Responses adapter must forward ``top_k`` into the
        ``ChatCompletionRequest`` it builds, so the value lands in the
        engine's sampling cascade. Pre-fix, ``responses_to_openai``
        didn't reference ``request.top_k`` at all (the field didn't
        exist) and the value vanished even when Pydantic dropped its
        validation."""
        client = responses_client.client
        engine = responses_client.engine
        resp = client.post(
            "/v1/responses",
            json=_payload(top_k=42),
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text
        assert len(engine.calls) == 1
        kwargs = engine.calls[0].kwargs
        # The sampling kwargs blob is merged into the engine call; top_k
        # rides in directly. ``build_extended_sampling_kwargs`` is the
        # function that surfaces it.
        assert kwargs.get("top_k") == 42, (
            f"top_k not threaded to engine.chat kwargs: {kwargs!r}"
        )
