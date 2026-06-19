# SPDX-License-Identifier: Apache-2.0
"""Route-level regression for the /v1/messages thinking-block gate (#702).

Adapter-level coverage lives in ``tests/test_anthropic_adapter.py``; this
file walks the full FastAPI route through ``TestClient`` to lock in the
predicate that codex r1 BLOCKING called out — the route must consult
the per-request alias's reasoning capability via
``_resolve_reasoning_enabled``, not the process-global
``cfg.reasoning_parser`` singleton.

Two surfaces:

* Non-streaming: ``cfg.reasoning_parser = None`` (alias has
  ``reasoning_parser: null``). Engine returns ``reasoning_text`` anyway
  — the rescue duplication shape from #569 — and the route must surface
  only a ``text`` content block, never ``thinking``.
* Streaming: same alias config, engine emits a delta tagged with
  ``channel="reasoning"``. The route must demote it to ``text_delta``
  in the SSE stream so clients never see ``content_block_start`` for a
  ``thinking`` block.
"""

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.anthropic import router as anthropic_router
from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry
from vllm_mlx.service.helpers import _resolve_reasoning_enabled


class _NonStreamingEngineEmittingReasoning:
    """Engine whose non-stream output carries ``reasoning_text``.

    Mimics the post-#569 rescue shape: ``content`` and
    ``reasoning_text`` agree on the same string because the OpenAI-side
    response builder copied reasoning into content to avoid a silently
    empty assistant turn. On the Anthropic surface this used to leak
    BOTH a ``thinking`` block AND a ``text`` block carrying the same
    string — F-010.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, content: str, reasoning: str):
        self._content = content
        self._reasoning = reasoning
        self.chat_calls: list[dict[str, Any]] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text=self._content,
            raw_text=self._content,
            reasoning_text=self._reasoning,
            tokens=[1],
            prompt_tokens=4,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


class _StreamingEngineEmittingReasoningChannel:
    """Engine whose stream tags a delta with ``channel="reasoning"``.

    Models routed through the engine ``OutputRouter`` (gemma4 / harmony
    family) surface tokens with explicit channel tags. The Anthropic
    streaming route used to open a ``thinking`` content block on
    ``channel="reasoning"`` unconditionally, even for aliases that
    declared ``reasoning_parser: null`` in ``aliases.json``. Issue #702
    moved the gate to ``_resolve_reasoning_enabled`` so the same alias
    config now demotes those deltas to ``text``.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self):
        self.stream_calls: list[dict[str, Any]] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        # Reasoning-channel delta — would normally open a ``thinking``
        # block. With the gate, it must demote to text.
        yield GenerationOutput(
            text="hello ",
            new_text="hello ",
            tokens=[1],
            prompt_tokens=4,
            completion_tokens=1,
            finished=False,
            finish_reason=None,
            channel="reasoning",
        )
        yield GenerationOutput(
            text="hello world",
            new_text="world",
            tokens=[1, 2],
            prompt_tokens=4,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
            channel="content",
        )


def _make_client(engine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    # Non-thinking alias — the predicate ``_resolve_reasoning_enabled``
    # falls back to ``cfg.reasoning_parser is not None`` in single-model
    # mode, so ``None`` means "alias is not reasoning-capable" and the
    # ``thinking`` block must be suppressed on both surfaces.
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.tool_parser = None

    app = FastAPI()
    app.include_router(anthropic_router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_config()


def _parse_sse_events(body: str) -> list[dict]:
    events = []
    for raw_event in body.split("\n\n"):
        data_line = next(
            (line for line in raw_event.splitlines() if line.startswith("data: ")),
            None,
        )
        if not data_line:
            continue
        payload = data_line.removeprefix("data: ")
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


def test_non_stream_route_suppresses_thinking_for_non_thinking_alias():
    """Issue #702: when the served alias has ``reasoning_parser: null``
    (single-model mode → ``cfg.reasoning_parser is None``), the route
    must NOT emit a ``thinking`` block even if the engine surfaced
    ``reasoning_text``. The exact F-010 shape: ``content`` and
    ``reasoning_text`` carry the same string because the rescue (#569)
    copied reasoning into content.
    """
    duplicated = "I think this is the answer."
    engine = _NonStreamingEngineEmittingReasoning(
        content=duplicated,
        reasoning=duplicated,
    )
    client = _make_client(engine)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "say something"}],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    block_types = [b["type"] for b in body["content"]]
    assert "thinking" not in block_types, (
        f"non-thinking alias must NOT emit a thinking block; got blocks={block_types!r}"
    )
    # Text block survives so the assistant turn isn't silently empty.
    text_blocks = [b for b in body["content"] if b["type"] == "text"]
    assert len(text_blocks) == 1
    assert text_blocks[0]["text"] == duplicated


def test_resolve_reasoning_enabled_uses_registry_entry_not_global():
    """Codex r1 BLOCKING on #702: in multi-model mode the predicate
    must consult the per-request registry entry, not the process-global
    ``cfg.reasoning_parser`` singleton. Otherwise a non-thinking alias
    served alongside a thinking default would still emit a duplicate
    ``thinking`` block because the global parser is set.
    """
    cfg = reset_config()
    # Global says "reasoning parser is set" (matches the default model).
    cfg.reasoning_parser = object()
    cfg.reasoning_parser_name = "hermes"
    # Registry overrides for two aliases:
    #   - thinking-alias has reasoning_parser="hermes"
    #   - non-thinking-alias has reasoning_parser=None
    registry = ModelRegistry()
    registry.add(
        ModelEntry(
            engine=object(),
            model_name="thinking-alias",
            model_path="thinking-alias",
            reasoning_parser="hermes",
        ),
        is_default=True,
    )
    registry.add(
        ModelEntry(
            engine=object(),
            model_name="non-thinking-alias",
            model_path="non-thinking-alias",
            reasoning_parser=None,
        ),
    )
    cfg.model_registry = registry

    assert _resolve_reasoning_enabled("thinking-alias") is True
    assert _resolve_reasoning_enabled("non-thinking-alias") is False
    # Unknown alias falls back to registry's default entry — the
    # thinking-alias, so reasoning_enabled stays True. This matches
    # how ``get_entry`` and ``get_engine`` already behave for unknown
    # names in the registry.
    assert _resolve_reasoning_enabled("does-not-exist") is True

    reset_config()


def test_resolve_reasoning_enabled_falls_back_to_global_without_registry():
    """Single-model mode (``cfg.model_registry`` is None) keeps the
    legacy semantics: the gate uses the global
    ``cfg.reasoning_parser`` / ``cfg.reasoning_parser_name`` pair
    because there's no per-alias metadata to consult. Both are
    populated together by ``server.load_model`` so checking either is
    equivalent; the helper accepts both so unit-test fixtures that
    only set the name keep working.
    """
    cfg = reset_config()
    cfg.model_registry = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    assert _resolve_reasoning_enabled("any-name") is False
    cfg.reasoning_parser = object()
    cfg.reasoning_parser_name = "hermes"
    assert _resolve_reasoning_enabled("any-name") is True
    # Either field alone is enough — exercises the OR branch the
    # helper uses for test-fixture compatibility.
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = "hermes"
    assert _resolve_reasoning_enabled("any-name") is True
    cfg.reasoning_parser = object()
    cfg.reasoning_parser_name = None
    assert _resolve_reasoning_enabled("any-name") is True
    reset_config()


def test_stream_route_demotes_reasoning_channel_for_non_thinking_alias():
    """Issue #702 streaming variant: engine emits a delta tagged
    ``channel="reasoning"`` (e.g. tokenizer carries
    ``<|channel>thought`` but the alias opted out of the reasoning
    parser). Route must demote to ``text_delta`` so clients never see
    a ``content_block_start`` for ``type="thinking"``.
    """
    engine = _StreamingEngineEmittingReasoningChannel()
    client = _make_client(engine)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text

    events = _parse_sse_events(resp.text)
    # No content_block_start should announce a thinking block.
    starts = [e for e in events if e.get("type") == "content_block_start"]
    for start in starts:
        block = start.get("content_block", {})
        assert block.get("type") != "thinking", (
            "non-thinking alias must NOT open a thinking content block "
            f"in the SSE stream; got {start!r}"
        )
    # Conversely, the model's bytes still surface — at least one text
    # block must appear so the assistant turn isn't silently empty.
    text_starts = [
        e for e in starts if e.get("content_block", {}).get("type") == "text"
    ]
    assert text_starts, (
        f"expected at least one text content_block_start; got events={events!r}"
    )


class _StreamingEngineNoChannelTags:
    """Engine that streams plain text deltas with NO ``channel`` tag.

    This exercises the codex r2 BLOCKING path: a non-thinking alias is
    served beside a thinking GLOBAL parser (Qwen3 / hermes), so
    ``cfg.reasoning_parser_name`` is set. Without the parser-bypass
    fix, implicit-mode parsers would classify each delta as
    ``reasoning`` until ``finalize_streaming`` emits a correction at
    end-of-stream. The gate would demote per-delta pieces to text but
    the finalize emission goes through a separate code path
    (``content_block_start type='text'``) — same bytes would then
    appear twice in the stream.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self):
        self.stream_calls: list[dict[str, Any]] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        # Plain text answer, no channel tag, no <think> markers — a
        # non-thinking alias's normal output.
        yield GenerationOutput(
            text="hello ",
            new_text="hello ",
            tokens=[1],
            prompt_tokens=4,
            completion_tokens=1,
            finished=False,
            finish_reason=None,
            channel=None,
        )
        yield GenerationOutput(
            text="hello world",
            new_text="world",
            tokens=[1, 2],
            prompt_tokens=4,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


def test_stream_route_bypasses_implicit_parser_for_non_thinking_alias():
    """Codex r2 BLOCKING on #702: when the alias is non-thinking but
    ``cfg.reasoning_parser_name`` is set (thinking global default), the
    streaming path must bypass the reasoning parser entirely so an
    implicit-mode parser's ``finalize_streaming`` correction can't
    re-emit the demoted reasoning bytes as a second text block —
    visible duplication. The route must instead stream each delta
    through ``think_router`` (no <think> tags → all text), single
    block, no finalize re-emission.
    """
    engine = _StreamingEngineNoChannelTags()
    # Override the single-model fallback to "thinking global default".
    # The route must STILL gate because the per-request alias
    # (model_name="non-thinking-alias") is non-thinking via the
    # registry.
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "thinking-default"
    cfg.reasoning_parser = object()
    cfg.reasoning_parser_name = "qwen3"
    cfg.tool_parser = None

    registry = ModelRegistry()
    registry.add(
        ModelEntry(
            engine=engine,
            model_name="thinking-default",
            model_path="thinking-default",
            reasoning_parser="qwen3",
        ),
        is_default=True,
    )
    registry.add(
        ModelEntry(
            engine=engine,
            model_name="non-thinking-alias",
            model_path="non-thinking-alias",
            reasoning_parser=None,
        ),
    )
    cfg.model_registry = registry

    app = FastAPI()
    app.include_router(anthropic_router)
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "non-thinking-alias",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text

    events = _parse_sse_events(resp.text)
    # No thinking block opens.
    starts = [e for e in events if e.get("type") == "content_block_start"]
    for start in starts:
        block = start.get("content_block", {})
        assert block.get("type") != "thinking", (
            "non-thinking alias must NOT open a thinking content block "
            f"in the SSE stream; got {start!r}"
        )
    # And the model bytes appear EXACTLY ONCE in the stream — not
    # duplicated by finalize_streaming. Collect all text_delta payloads
    # and confirm the concatenation equals what the engine emitted.
    text_deltas = [
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "text_delta"
    ]
    assembled = "".join(text_deltas)
    # The engine emitted ``"hello "`` + ``"world"`` (two new_text
    # chunks). After the gate, each chunk emerges as one text_delta;
    # finalize_streaming MUST NOT re-emit the same bytes a second time
    # (the codex r2 BLOCKING regression shape).
    assert assembled == "hello world", (
        f"expected exactly 'hello world' (one copy); got {assembled!r} "
        f"from text_deltas={text_deltas!r}"
    )

    reset_config()
