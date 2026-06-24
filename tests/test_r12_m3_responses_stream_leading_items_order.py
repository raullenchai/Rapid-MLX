# SPDX-License-Identifier: Apache-2.0
"""R12-M3 regression guard — streaming ``/v1/responses`` MUST emit the
``reasoning`` output item BEFORE the ``message`` output item, even when
the reasoning text is empty.

Mira r12 dogfood (R-4): the streaming surface emits the empty/short
``reasoning`` item AFTER the message item, so SDK clients that expect
reasoning-before-message see message first and may discard the late
reasoning item or treat the stream as malformed. The OpenAI Responses
reference implementation always emits a ``reasoning`` item (possibly
with empty content) BEFORE the message item — this test pins that
invariant on the rapid-mlx streaming wire.

The invariant tested here:
  1. ``response.output_item.added`` events for "leading" items (currently
     just ``reasoning``) MUST land before the ``response.output_item.added``
     for the ``message`` item.
  2. Even when the model produces zero reasoning text, a ``reasoning``
     item with empty ``summary`` MUST still be emitted before the
     ``message`` item.
  3. Output indices in the wire events MUST be monotonically increasing
     and consistent with the final ``response.completed.response.output[]``
     array ordering.
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


class _EngineEmptyReasoning:
    """Engine that produces ONLY non-reasoning content (no reasoning bytes).

    Reproduces the M-3 bug: when reasoning is empty, the legacy code
    skipped the post-loop reasoning emit entirely, so the message item
    was the only ``output_item.added``. Post-fix, a reasoning item with
    empty summary MUST still be emitted BEFORE the message item.
    """

    preserve_native_tool_format = False

    def __init__(self):
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        return _GenerationOutput(
            text="hi",
            prompt_tokens=3,
            completion_tokens=1,
            finish_reason="stop",
        )

    async def stream_chat(self, messages, **kwargs):
        chunks = ["hi"]
        for i, c in enumerate(chunks):
            yield _GenerationOutput(
                text="".join(chunks[: i + 1]),
                new_text=c,
                prompt_tokens=3 if i == 0 else 0,
                completion_tokens=i + 1,
                finish_reason=None if i < len(chunks) - 1 else "stop",
            )


class _EngineWithReasoning:
    """Engine that emits reasoning via the harmony-style ``channel`` field."""

    preserve_native_tool_format = False

    def __init__(self):
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        return _GenerationOutput(
            text="hello",
            prompt_tokens=3,
            completion_tokens=2,
            finish_reason="stop",
        )

    async def stream_chat(self, messages, **kwargs):
        # First a reasoning chunk, then the actual answer.
        yield _GenerationOutput(
            text="",
            new_text="thinking step",
            prompt_tokens=3,
            completion_tokens=0,
            channel="reasoning",
            finish_reason=None,
        )
        yield _GenerationOutput(
            text="hello",
            new_text="hello",
            prompt_tokens=0,
            completion_tokens=2,
            channel="content",
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
def make_client(monkeypatch):
    """Yields a factory that swaps the engine and returns a TestClient."""
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
    cfg.model_name = "test-model"
    cfg.model_registry = None

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    client = TestClient(app)

    def _set(engine):
        cfg.engine = engine
        return client

    yield SimpleNamespace(set=_set, client=client)

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


HEADERS = {"Authorization": "Bearer test-secret"}


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


def _stream_payload(**overrides):
    base = {"model": "test-model", "input": "Hello, world", "stream": True}
    base.update(overrides)
    return base


def _stream_and_parse(client, payload):
    with client.stream(
        "POST",
        "/v1/responses",
        json=payload,
        headers=HEADERS,
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    return _parse_sse(body)


def _assert_leading_items_before_message(events):
    """Shared monotonic-ordering assertion helper.

    Walks the ``response.output_item.added`` events and asserts that any
    ``reasoning`` / ``function_call`` / ``computer_call`` item comes
    BEFORE the ``message`` item. Returns the ordered list of (type, idx)
    tuples so callers can do additional shape assertions.
    """
    added_items: list[tuple[str, int]] = []
    message_added_idx: int | None = None
    for idx, (name, data) in enumerate(events):
        if name != "response.output_item.added":
            continue
        item = data.get("item", {})
        item_type = item.get("type")
        added_items.append((item_type, idx))
        if item_type == "message" and message_added_idx is None:
            message_added_idx = idx

    for item_type, idx in added_items:
        if item_type in ("reasoning", "function_call", "computer_call"):
            if message_added_idx is not None:
                assert idx < message_added_idx, (
                    f"Ordering violation: leading {item_type!r} item at "
                    f"event#{idx} arrived AFTER message item at "
                    f"event#{message_added_idx}. OpenAI Responses spec "
                    f"requires all leading items before the message item."
                )
    return added_items


class TestLeadingItemOrdering:
    def test_empty_reasoning_still_emits_reasoning_before_message(self, make_client):
        """M-3 root case: model produces NO reasoning bytes. The fix must
        still emit an empty ``reasoning`` item BEFORE the message item,
        matching the OpenAI reference implementation."""
        client = make_client.set(_EngineEmptyReasoning())
        events = _stream_and_parse(client, _stream_payload())
        added = _assert_leading_items_before_message(events)

        added_types = [t for t, _ in added]
        assert "message" in added_types, (
            f"No message item emitted — events: {[n for n, _ in events]}"
        )
        assert "reasoning" in added_types, (
            "Empty-reasoning case: reasoning item MUST still be emitted "
            "before the message item per OpenAI Responses spec. "
            f"Items added (in order): {added_types}"
        )
        # Reasoning must be FIRST.
        assert added_types[0] == "reasoning", (
            f"reasoning must be the first output_item.added event; "
            f"got order: {added_types}"
        )

    def test_non_empty_reasoning_emitted_before_message(self, make_client):
        """Existing-behaviour parity: when the model DOES produce
        reasoning, reasoning-first ordering is also preserved."""
        client = make_client.set(_EngineWithReasoning())
        events = _stream_and_parse(client, _stream_payload())
        added = _assert_leading_items_before_message(events)
        added_types = [t for t, _ in added]
        assert added_types[0] == "reasoning"
        assert "message" in added_types

        # The reasoning summary must carry the model's chain-of-thought.
        reasoning_done = [
            d
            for (n, d) in events
            if n == "response.output_item.done"
            and d.get("item", {}).get("type") == "reasoning"
        ]
        assert reasoning_done
        summary = reasoning_done[0]["item"].get("summary", [])
        assert summary and summary[0]["text"]

    def test_completed_output_index_matches_wire_indices(self, make_client):
        """The ``output_index`` on every ``output_item.added`` event must
        match the position of that item in the terminal
        ``response.completed.response.output[]`` array."""
        client = make_client.set(_EngineEmptyReasoning())
        events = _stream_and_parse(client, _stream_payload())

        # Pull added events keyed by output_index. R12-M3 codex r1 NIT:
        # guard against duplicate output_index emissions — each wire
        # ``output_item.added`` must own a unique ``output_index``; the
        # terminal ``response.output[]`` is indexed in the same space,
        # so duplicates would silently mask wire bugs.
        added_by_idx: dict[int, str] = {}
        for name, data in events:
            if name != "response.output_item.added":
                continue
            idx = data["output_index"]
            assert idx not in added_by_idx, (
                f"duplicate output_index={idx} in response.output_item.added "
                f"events: existing={added_by_idx[idx]}, new={data['item']['type']}"
            )
            added_by_idx[idx] = data["item"]["type"]

        # Terminal completed payload.
        completed = [d for (n, d) in events if n == "response.completed"]
        assert completed, "missing response.completed event"
        output_arr = completed[0]["response"]["output"]
        # Index space must be 1:1 between wire-added and completed[].
        assert len(added_by_idx) == len(output_arr), (
            f"wire emitted {len(added_by_idx)} added events but completed "
            f"has {len(output_arr)} entries: added={added_by_idx}, "
            f"completed={[it['type'] for it in output_arr]}"
        )
        for i, item in enumerate(output_arr):
            assert added_by_idx.get(i) == item["type"], (
                f"Index mismatch at position {i}: added_by_idx={added_by_idx}, "
                f"output[i].type={item['type']}"
            )

    def test_assert_monotonic_helper(self, make_client):
        """The helper itself: a stream with reasoning-first ordering must
        pass the assertion. Sanity-check the helper so future tests can
        reuse it confidently."""
        client = make_client.set(_EngineWithReasoning())
        events = _stream_and_parse(client, _stream_payload())
        # Should not raise.
        _assert_leading_items_before_message(events)
