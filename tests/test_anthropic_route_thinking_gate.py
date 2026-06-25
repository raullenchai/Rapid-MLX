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


# ──────────────────────────────────────────────────────────────────────
# Codex r3 MAJOR (probe 5): wire-format byte-level pinning.
#
# The five tests above parse SSE payloads into Python objects via
# ``_parse_sse_events`` and then check semantic invariants. That misses
# wire-format regressions Anthropic SDKs are sensitive to:
#   * spurious ``content_block_start`` for ``type="thinking"`` even when
#     no thinking delta follows (clients enter "extended thinking" UI);
#   * empty ``text_delta`` events (``"delta":{"text":""}``) — Anthropic's
#     reference client treats these as noise and some downstream tools
#     interrupt rendering;
#   * missing or duplicated ``content_block_stop`` for the demoted text
#     block.
#
# The tests below pin the raw event prefix bytes and assert exact
# framing so future regressions on the streaming gate (or on
# ``_emit_content_pieces``) surface immediately.
# ──────────────────────────────────────────────────────────────────────


def _split_raw_sse(body: str) -> list[str]:
    """Return raw ``event: ... \ndata: ...`` blocks (no trailing \n\n)."""
    return [chunk for chunk in body.split("\n\n") if chunk.strip()]


def test_stream_route_wire_format_no_thinking_start_byte_level():
    """No ``event: content_block_start`` byte sequence whose data
    payload contains ``"type": "thinking"`` may appear in the raw
    bytes. Stronger than the parsed-object check: a regression that
    emits a malformed JSON payload (e.g. wrong field order or extra
    keys) would still trip this assertion because we grep the data
    line directly.
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
    body = resp.text

    # Every event block carrying ``content_block_start`` must NOT
    # contain ``"thinking"`` ANYWHERE in its data line.
    for raw in _split_raw_sse(body):
        if "content_block_start" not in raw:
            continue
        data_line = next(
            (line for line in raw.splitlines() if line.startswith("data: ")),
            "",
        )
        assert '"type":"thinking"' not in data_line.replace(" ", ""), (
            f"non-thinking alias leaked a thinking content_block_start "
            f"into raw SSE: {raw!r}"
        )


def test_stream_route_wire_format_no_empty_text_delta():
    """No ``text_delta`` event in the raw stream may carry an empty
    string. Codex r3 MAJOR (probe 1) — strip-whitespace guards must
    prevent the streaming surface from opening an empty content
    block or emitting an empty delta to the client.
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
    body = resp.text

    for raw in _split_raw_sse(body):
        if "content_block_delta" not in raw:
            continue
        data_line = next(
            (line for line in raw.splitlines() if line.startswith("data: ")),
            "",
        )
        if "text_delta" not in data_line:
            continue
        payload = json.loads(data_line.removeprefix("data: "))
        delta_text = payload.get("delta", {}).get("text", None)
        assert delta_text != "", f"empty text_delta leaked into SSE stream: raw={raw!r}"


def test_stream_route_wire_format_event_prefix_and_terminator():
    """Each non-empty event chunk must follow the ``event: <name>\n``
    ``data: <json>\n\n`` shape exactly. Anthropic SDKs parse on this
    framing; a missing ``\n\n`` terminator or extra whitespace would
    silently break clients.
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
    body = resp.text

    # The body must end with a terminator. Trailing-newline tolerance
    # mirrors what Anthropic's reference SSE parser expects.
    assert body.endswith("\n\n"), (
        f"SSE body missing trailing terminator: tail={body[-20:]!r}"
    )

    for raw in _split_raw_sse(body):
        lines = raw.splitlines()
        # Required framing (codex r4 NIT — relaxed from "exactly 2
        # lines" to avoid over-pinning today's emitters): each chunk
        # MUST start with an ``event:`` line and MUST contain a JSON-
        # parseable ``data:`` line. Comments (``:`` lines) and
        # multi-line ``data:`` framing remain valid SSE shapes that a
        # future legitimate change might introduce.
        assert lines, f"empty SSE chunk: {raw!r}"
        # R15 #291: the SSE disconnect-guard injects a single
        # ``: keepalive`` comment line right after the first real
        # chunk (Vlad 8k-prompt boundary fix). Comments are no-ops
        # per WHATWG SSE §9.2.6 — skip them for the event-framing
        # check (they have neither ``event:`` nor ``data:`` lines).
        if all(line.startswith(":") for line in lines):
            continue
        assert lines[0].startswith("event: "), (
            f"first SSE line must start with 'event: ': {lines[0]!r}"
        )
        data_line = next((line for line in lines if line.startswith("data: ")), None)
        assert data_line is not None, f"SSE chunk missing 'data:' line: {lines!r}"
        json.loads(data_line.removeprefix("data: "))


def test_stream_route_wire_format_exactly_one_text_block_for_demoted_stream():
    """When the engine emits two reasoning deltas on a non-thinking
    alias the gate demotes them to text. The raw stream must contain
    exactly ONE ``content_block_start`` ``type="text"`` (consecutive
    same-type pieces merge in ``_emit_content_pieces``) and exactly
    ONE matching ``content_block_stop``. Catches regressions that
    would emit ``start/stop/start/stop`` instead.
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
    body = resp.text

    text_starts = 0
    text_stops = 0
    open_text_index: int | None = None
    for raw in _split_raw_sse(body):
        data_line = next(
            (line for line in raw.splitlines() if line.startswith("data: ")),
            "",
        )
        if not data_line:
            continue
        payload = json.loads(data_line.removeprefix("data: "))
        if payload.get("type") == "content_block_start":
            block_type = payload.get("content_block", {}).get("type")
            if block_type == "text":
                text_starts += 1
                open_text_index = payload.get("index")
        elif payload.get("type") == "content_block_stop":
            # Match against the most-recently-opened text block.
            if open_text_index is not None and payload.get("index") == open_text_index:
                text_stops += 1
                open_text_index = None

    assert text_starts == 1, (
        f"expected exactly 1 text content_block_start (merged); "
        f"got {text_starts} in body={body!r}"
    )
    assert text_stops == 1, (
        f"expected exactly 1 matching content_block_stop; "
        f"got {text_stops} in body={body!r}"
    )


def test_stream_route_drops_whitespace_only_reasoning_delta():
    """Codex r3 MAJOR (probe 1): a channel="reasoning" delta carrying
    pure whitespace must NOT open a thinking content_block_start, nor
    emit an empty/whitespace thinking_delta. Mirrors the non-stream
    ``openai_to_anthropic`` predicate (``reasoning_text.strip() != ""``).
    """

    class _ReasoningChannelWhitespaceOnly:
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
            # Whitespace-only reasoning — must be filtered.
            yield GenerationOutput(
                text="   \n",
                new_text="   \n",
                tokens=[1],
                prompt_tokens=4,
                completion_tokens=1,
                finished=False,
                finish_reason=None,
                channel="reasoning",
            )
            yield GenerationOutput(
                text="   \nanswer",
                new_text="answer",
                tokens=[1, 2],
                prompt_tokens=4,
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                channel="content",
            )

    # Switch to a thinking-capable alias so the gate is the no-op
    # branch — we want to confirm the WHITESPACE-DROP guard fires
    # even when the alias IS reasoning-capable.
    cfg = reset_config()
    engine = _ReasoningChannelWhitespaceOnly()
    cfg.engine = engine
    cfg.model_name = "thinking-model"
    cfg.model_registry = None
    cfg.reasoning_parser = object()
    cfg.reasoning_parser_name = "hermes"
    cfg.tool_parser = None

    app = FastAPI()
    app.include_router(anthropic_router)
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "thinking-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    events = _parse_sse_events(resp.text)
    thinking_starts = [
        e
        for e in events
        if e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "thinking"
    ]
    assert thinking_starts == [], (
        f"whitespace-only reasoning leaked a thinking content block; "
        f"got starts={thinking_starts!r}"
    )
    # The content delta still surfaces.
    text_deltas = [
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "text_delta"
    ]
    assembled = "".join(text_deltas)
    assert assembled == "answer", (
        f"expected 'answer' to surface; got {assembled!r} from {text_deltas!r}"
    )

    reset_config()


class _StreamingEngineInterleavedThinking:
    """Engine that streams `<think>first\n\nsecond</think>` to exercise
    the codex r4 MAJOR fix: intra-thinking whitespace separators must
    survive the gate so the rendered thinking block is `first\n\nsecond`
    (a paragraph break) and not `firstsecond` (visually concatenated).
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
        # Emit content in three reasoning-channel deltas: "first",
        # whitespace separator, "second", then a content delta.
        yield GenerationOutput(
            text="first",
            new_text="first",
            tokens=[1],
            prompt_tokens=4,
            completion_tokens=1,
            finished=False,
            finish_reason=None,
            channel="reasoning",
        )
        yield GenerationOutput(
            text="first\n\n",
            new_text="\n\n",
            tokens=[1, 2],
            prompt_tokens=4,
            completion_tokens=2,
            finished=False,
            finish_reason=None,
            channel="reasoning",
        )
        yield GenerationOutput(
            text="first\n\nsecond",
            new_text="second",
            tokens=[1, 2, 3],
            prompt_tokens=4,
            completion_tokens=3,
            finished=False,
            finish_reason=None,
            channel="reasoning",
        )
        yield GenerationOutput(
            text="first\n\nsecondANSWER",
            new_text="ANSWER",
            tokens=[1, 2, 3, 4],
            prompt_tokens=4,
            completion_tokens=4,
            finished=True,
            finish_reason="stop",
            channel="content",
        )


def test_stream_route_preserves_intra_thinking_whitespace():
    """Codex r4 MAJOR: whitespace between two non-empty thinking
    chunks must NOT be dropped by the gate — it is an intra-thinking
    paragraph separator the model emitted on purpose. The fix made
    ``_gate_thinking_pieces`` state-aware (current_block_type) so a
    whitespace piece is dropped only when it would OPEN a blank
    thinking block.
    """
    cfg = reset_config()
    engine = _StreamingEngineInterleavedThinking()
    cfg.engine = engine
    cfg.model_name = "thinking-model"
    cfg.model_registry = None
    cfg.reasoning_parser = object()
    cfg.reasoning_parser_name = "hermes"
    cfg.tool_parser = None

    app = FastAPI()
    app.include_router(anthropic_router)
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "thinking-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    events = _parse_sse_events(resp.text)

    # Assemble the full thinking content from thinking_delta events.
    thinking_text = "".join(
        e["delta"]["thinking"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "thinking_delta"
    )
    assert thinking_text == "first\n\nsecond", (
        "intra-thinking whitespace separator was dropped — "
        f"got {thinking_text!r}, expected 'first\\n\\nsecond'"
    )
    # And the answer still surfaces in a text block.
    text_text = "".join(
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "text_delta"
    )
    assert text_text == "ANSWER", f"answer text missing or wrong: got {text_text!r}"

    reset_config()


class _StreamingNonThinkingChannelHelloSpace:
    """Engine that streams ``("reasoning", "hello") -> ("reasoning", " ")
    -> ("content", "world")`` to exercise the codex r5 MAJOR fix.

    On a non-thinking alias the gate demotes the first delta to a text
    block. The second (whitespace-only) reasoning delta MUST also
    demote to text so the open text block continues to receive
    ``"hello "`` rather than truncating to ``"hello"``. The third
    (content) delta then appends ``"world"`` to the same block.
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
        yield GenerationOutput(
            text="hello",
            new_text="hello",
            tokens=[1],
            prompt_tokens=4,
            completion_tokens=1,
            finished=False,
            finish_reason=None,
            channel="reasoning",
        )
        yield GenerationOutput(
            text="hello ",
            new_text=" ",
            tokens=[1, 2],
            prompt_tokens=4,
            completion_tokens=2,
            finished=False,
            finish_reason=None,
            channel="reasoning",
        )
        yield GenerationOutput(
            text="hello world",
            new_text="world",
            tokens=[1, 2, 3],
            prompt_tokens=4,
            completion_tokens=3,
            finished=True,
            finish_reason="stop",
            channel="content",
        )


def test_stream_route_demoted_text_keeps_intra_block_whitespace():
    """Codex r5 MAJOR: on a non-thinking alias, a whitespace-only
    reasoning delta that arrives AFTER a non-empty reasoning delta
    must demote into the open text block (giving ``"hello world"``),
    not be dropped (which would give ``"helloworld"``).
    """
    engine = _StreamingNonThinkingChannelHelloSpace()
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

    # No thinking block should be opened.
    thinking_starts = [
        e
        for e in events
        if e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "thinking"
    ]
    assert thinking_starts == [], (
        f"non-thinking alias opened a thinking block: {thinking_starts!r}"
    )

    text_deltas = [
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "text_delta"
    ]
    assembled = "".join(text_deltas)
    assert assembled == "hello world", (
        f"expected 'hello world' (whitespace preserved into open text "
        f"block); got {assembled!r} from {text_deltas!r}"
    )
