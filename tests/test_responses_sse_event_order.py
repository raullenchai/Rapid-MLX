# SPDX-License-Identifier: Apache-2.0
"""r6-A R6-H7 regression guard — streaming ``/v1/responses`` emits the
OpenAI-spec lifecycle events IN ORDER, including
``response.in_progress`` between ``response.created`` and the first
``response.output_item.added``.

Background: Sasha R2 captured Codex CLI's stream parser entering a
half-initialized state because rapid-mlx jumped straight from
``response.created`` to ``response.output_item.added``, skipping the
spec-required ``response.in_progress`` event. The official
``openai-python`` SDK transitions internal state on each lifecycle
event, and consumers (Codex CLI, openai-agents, etc.) rely on the
ordering to drive their own UI / control-flow state machines.
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
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        return _GenerationOutput(
            text="hello",
            prompt_tokens=3,
            completion_tokens=1,
            finish_reason="stop",
        )

    async def stream_chat(self, messages, **kwargs):
        """Tiny stream: three text chunks so the message item opens."""
        chunks = ["Hello", " from", " rapid"]
        for i, c in enumerate(chunks):
            yield _GenerationOutput(
                text="".join(chunks[: i + 1]),
                new_text=c,
                prompt_tokens=3 if i == 0 else 0,
                completion_tokens=i + 1,
                finish_reason=None if i < len(chunks) - 1 else "stop",
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


HEADERS = {"Authorization": "Bearer test-secret"}


def _stream_payload(**overrides):
    base = {"model": "test-model", "input": "Hello, world", "stream": True}
    base.update(overrides)
    return base


# =============================================================================
# Event order — created → in_progress → output_item.added → text deltas →
# output_text.done → content_part.done → output_item.done → completed
# =============================================================================


class TestResponsesStreamEventOrder:
    def test_response_in_progress_event_between_created_and_first_item(
        self, responses_client
    ):
        """r6-A R6-H7: ``response.in_progress`` MUST land between
        ``response.created`` and the first ``response.output_item.added``.
        Pre-fix, the event was missing entirely — Sasha R2's Codex CLI
        capture jumped straight from ``created`` to ``output_item.added``.
        """
        client = responses_client.client
        with client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        names = [n for n, _ in events]

        assert "response.in_progress" in names, (
            f"response.in_progress missing from stream. Events seen: {names}"
        )

        created_idx = names.index("response.created")
        in_progress_idx = names.index("response.in_progress")
        first_item_idx = names.index("response.output_item.added")

        assert created_idx < in_progress_idx < first_item_idx, (
            f"Event ordering violation: created@{created_idx}, "
            f"in_progress@{in_progress_idx}, first_item@{first_item_idx}. "
            f"Spec requires created < in_progress < output_item.added."
        )

    def test_response_in_progress_payload_shape(self, responses_client):
        """The ``response.in_progress`` payload must echo the same
        envelope shape (``id`` / ``object`` / ``status="in_progress"`` /
        ``model``) so SDK consumers can populate their internal state
        from this event the same way they do from ``response.created``."""
        client = responses_client.client
        with client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        in_progress = [d for (n, d) in events if n == "response.in_progress"]
        assert in_progress, "no response.in_progress event"
        payload = in_progress[0]
        assert payload["type"] == "response.in_progress"
        assert payload["response"]["status"] == "in_progress"
        assert payload["response"]["object"] == "response"
        # ``id`` is shared with ``response.created`` so the SDK can
        # thread its state machine off a single response handle.
        created = [d for (n, d) in events if n == "response.created"][0]
        assert payload["response"]["id"] == created["response"]["id"]
        assert payload["response"]["model"] == created["response"]["model"]
        assert payload["response"]["created_at"] == created["response"]["created_at"]

    def test_full_spec_event_order(self, responses_client):
        """End-to-end ordering check against the OpenAI Responses SSE
        spec for a text-only reply:

          response.created
          response.in_progress
          response.output_item.added       (message item, lazy-open on first delta)
          response.content_part.added      (output_text part)
          response.output_text.delta       (one per chunk)
          response.output_text.done
          response.content_part.done
          response.output_item.done
          response.completed
        """
        client = responses_client.client
        with client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            body = "".join(resp.iter_text())
        names = [n for n, _ in _parse_sse(body)]

        # Each of these must appear at least once.
        expected_singletons = [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.output_text.done",
            "response.output_item.done",
            "response.completed",
        ]
        for ev in expected_singletons:
            assert ev in names, f"{ev} missing — events seen: {names}"

        # Ordering — pull the first index of each landmark and assert
        # they're monotonically increasing.
        landmarks = [
            ("response.created", names.index("response.created")),
            ("response.in_progress", names.index("response.in_progress")),
            (
                "response.output_item.added",
                names.index("response.output_item.added"),
            ),
            (
                "response.output_text.delta",
                names.index("response.output_text.delta"),
            ),
            (
                "response.output_text.done",
                names.index("response.output_text.done"),
            ),
            (
                "response.output_item.done",
                names.index("response.output_item.done"),
            ),
            ("response.completed", names.index("response.completed")),
        ]
        indices = [i for _, i in landmarks]
        assert indices == sorted(indices), (
            f"Event ordering violated. Landmarks (in expected order): "
            f"{landmarks}. Names: {names}"
        )
