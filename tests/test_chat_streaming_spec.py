# SPDX-License-Identifier: Apache-2.0
"""OpenAI streaming-spec invariants for the non-guided chat path.

Companion to ``tests/test_chat_streaming_guided.py``. PR #422 pinned these
invariants on the guided helper (``stream_chat_completion_guided``) but
the regular ``stream_chat_completion`` path retained two spec violations
that the 2026-05-20 ≥20B onboarding sweep caught on qwen3.5-35b-8bit
(see knowledge/guided_generation_gaps_2026-05-20.md, "Bug B"):

1. **``created`` drift** — content chunks share one timestamp but the
   finish/usage/terminator chunks build via ``ChatCompletionChunk(...)``
   default-factory'd a fresh ``int(time.time())`` per instantiation,
   producing a 5-7s gap between content and finish chunks on slow MoE
   models. Fixed by passing ``created=_sse_created`` to every
   constructor.

2. **Dual usage emission** — when ``stream_options.include_usage`` is
   True, the finish chunk AND the dedicated trailing chunk both carried
   the same usage payload, double-counting tokens for aggregating
   clients. Fixed by setting ``usage=None`` on the finish chunk when
   ``include_usage`` is True.

3. **D-SSE-USAGE (v0.8.2)** — when ``stream_options.include_usage`` is
   False / unset, the finish chunk still carried a populated ``usage``
   block. Per the OpenAI streaming spec, ``usage`` is opt-in and MUST
   be omitted from every chunk unless the caller passes
   ``include_usage=true``. LangChain / AI-SDK / vercel-ai-stream
   parsers double-count token totals when the field appears on
   non-opt-in streams. The old "usage on finish chunk for bare clients"
   accommodation has been removed — see
   ``tests/test_stream_include_usage_honored.py`` for the full matrix.

Bug A (streaming tool-parser coverage gap) is pinned by
``test_streaming_tool_fallback_catches_non_canonical_format``: the
``StreamingPostProcessor.finalize`` hook now runs ``extract_tool_calls``
unconditionally (no ``has_pending_tool_call`` gate) so the streaming
path inherits the full set of fallback patterns the non-stream parser
already supports.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import router as chat_router
from vllm_mlx.routes.chat import stream_chat_completion
from vllm_mlx.service.postprocessor import StreamingPostProcessor
from vllm_mlx.tool_parsers.abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
)


class _PlainStreamEngine:
    """Mock engine for the non-guided streaming path.

    ``stream_chat`` yields a sequence of small text deltas, then a final
    output with ``finished=True``. No guided-generation support — the
    chat route should dispatch directly to ``stream_chat_completion``.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, deltas: list[str] | None = None):
        self._deltas = deltas or ["Hello", " world", "."]
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        accumulated = ""
        for i, delta in enumerate(self._deltas):
            accumulated += delta
            is_last = i == len(self._deltas) - 1
            yield GenerationOutput(
                text=accumulated,
                new_text=delta,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=is_last,
                finish_reason="stop" if is_last else None,
                channel=None,
            )


def _make_client(engine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True

    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


def _parse_sse_events(text: str) -> tuple[list[dict], bool]:
    """Return ``(parsed_events, saw_done)``.

    ``parsed_events`` excludes the ``[DONE]`` sentinel.
    """
    events: list[dict] = []
    saw_done = False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events, saw_done


def test_non_guided_streaming_pins_single_created_timestamp(monkeypatch):
    """Bug B regression: every SSE chunk in one completion must share a
    single ``created`` value.

    Pre-fix: the fast-path content chunks baked ``_sse_created`` into the
    SSE prefix string, but Pydantic-built chunks (finish, fallback
    tool_call, dedicated usage) called ``ChatCompletionChunk(...)``
    without ``created=...`` and inherited the default factory's fresh
    ``time.time()`` per construction. On slow MoE models the gap
    between first content chunk and finish chunk was 5-7s (Agent A7,
    qwen3.5-35b-8bit, 2026-05-20 sweep).

    Patches ``time.time`` to advance one second per call so the bug is
    deterministically observable in a unit test (real wall-clock under
    a fast mock would collapse all timestamps to the same integer
    second and silently mask the drift, which is what the unfixed code
    used to coast on in CI).
    """
    # Force time to advance one logical second per consumer so each
    # ``time.time()`` invocation returns a strictly increasing integer.
    # Without the fix, ChatCompletionChunk's default factory will pull
    # a distinct value per construction and break the invariant.
    counter = {"t": 1_700_000_000}

    def _stepping_time():
        counter["t"] += 1
        return counter["t"]

    import time as _time_mod

    import vllm_mlx.api.models as _models_mod
    import vllm_mlx.routes.chat as _chat_mod

    monkeypatch.setattr(_time_mod, "time", _stepping_time)
    monkeypatch.setattr(_chat_mod.time, "time", _stepping_time, raising=False)
    monkeypatch.setattr(_models_mod.time, "time", _stepping_time, raising=False)

    engine = _PlainStreamEngine(deltas=["alpha", " beta", " gamma."])
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "say hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done

    created_values = {e["created"] for e in events if "created" in e}
    assert len(created_values) == 1, (
        f"non-guided streaming must share one created timestamp across "
        f"every chunk (role/content/finish/usage); saw {created_values}"
    )

    ids = {e["id"] for e in events if "id" in e}
    assert len(ids) == 1, f"all chunks must share one id; saw {ids}"


def test_non_guided_streaming_usage_only_in_dedicated_chunk_when_include_usage_true():
    """Bug B regression: with ``include_usage=True`` the finish chunk
    MUST have ``usage=null``; only the dedicated trailing chunk (with
    empty ``choices``) carries usage. Pre-fix both chunks carried the
    same payload, causing aggregating clients to double-count.
    """
    engine = _PlainStreamEngine(deltas=["one", " two."])
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "say hi"}],
            "stream_options": {"include_usage": True},
        },
    )
    assert resp.status_code == 200, resp.text
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done

    finish_events = [
        e for e in events for c in e.get("choices", []) if c.get("finish_reason")
    ]
    usage_only_events = [e for e in events if not e.get("choices") and e.get("usage")]

    assert len(finish_events) == 1, "exactly one finish chunk expected"
    assert finish_events[0].get("usage") is None, (
        "finish chunk MUST NOT carry usage when include_usage=True — "
        "double emission causes aggregating clients to double-count"
    )
    assert len(usage_only_events) == 1, (
        "expected exactly one dedicated usage chunk when include_usage=True"
    )


def test_non_guided_streaming_omits_usage_when_include_usage_unset():
    """D-SSE-USAGE regression: when ``stream_options`` is omitted (or
    ``include_usage=false``), the OpenAI streaming spec requires the
    ``usage`` field to be absent from EVERY SSE chunk — not just the
    dedicated trailing one. Pre-v0.8.2 the finish chunk carried a
    populated ``usage`` block on bare requests, which LangChain /
    AI-SDK / vercel-ai-stream parsers treated as the canonical totals
    AND then double-counted when a downstream proxy upgraded the
    request with ``include_usage=true`` and the dedicated chunk also
    landed.
    """
    engine = _PlainStreamEngine(deltas=["one", " two."])
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "say hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    events, _ = _parse_sse_events(resp.text)

    finish_events = [
        e for e in events for c in e.get("choices", []) if c.get("finish_reason")
    ]
    usage_only_events = [e for e in events if not e.get("choices") and e.get("usage")]

    assert len(finish_events) == 1
    assert finish_events[0].get("usage") is None, (
        "finish chunk MUST NOT carry usage when include_usage is unset — "
        "OpenAI streaming spec requires opt-in via stream_options"
    )
    assert usage_only_events == [], (
        "no dedicated usage chunk when include_usage is unset"
    )
    # Stronger: the ``usage`` KEY must be absent from every chunk —
    # a regression that serializes ``"usage": null`` would slip past a
    # truthiness check but is itself non-spec for the unset path
    # (codex review caught the same gap in
    # ``test_stream_include_usage_honored.py``).
    any_usage_key = [e for e in events if "usage" in e]
    assert any_usage_key == [], (
        f"no SSE chunk may carry the usage KEY when include_usage is "
        f"unset; got {len(any_usage_key)} chunk(s) with the key"
    )


# -----------------------------------------------------------------------
# Bug A — Streaming tool-parser coverage gap (family-wide)
# -----------------------------------------------------------------------


class _GapStreamParser(ToolParser):
    """Mock parser modeling the gap: streaming code can't see the tool
    call (returns plain content), but the non-stream ``extract_tool_calls``
    catches it via a fallback pattern.

    Mirrors the gemma-4-26b-4bit case from the 2026-05-20 sweep where
    streaming dropped a tool call that the non-stream parser handled.
    """

    # Use `{` so the cheap markup pre-check in finalize() (added in
    # response to DeepSeek's pr_validate finding on PR #424 — every real
    # tool-call format has at least one of `<`, `{`, or `[Calling`) lets
    # us reach extract_tool_calls. Plain-text responses without any
    # structural marker correctly skip the full parser.
    SENTINEL = '{"call":"get_weather"}'

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

    def reset(self):
        super().reset()

    def has_pending_tool_call(self, text: str) -> bool:
        # Deliberately understates coverage — same canonical-wrapper-only
        # gate every real parser has. Returns False for our SENTINEL so
        # the pre-fix finalize would skip extraction; the post-fix
        # finalize ignores this gate.
        return "<tool_call>" in text

    def extract_tool_calls(
        self, model_output: str, request: Any = None
    ) -> ExtractedToolCallInformation:
        # Non-stream side knows about the SENTINEL fallback format.
        if self.SENTINEL in model_output:
            content = model_output.replace(self.SENTINEL, "").strip() or None
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[
                    {
                        "id": "call_test123",
                        "name": "get_weather",
                        "arguments": '{"location": "SF"}',
                    }
                ],
                content=content,
            )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence = (),
        current_token_ids: Sequence = (),
        delta_token_ids: Sequence = (),
        request: dict[str, Any] | None = None,
    ) -> dict | None:
        # Streaming side does NOT recognize the SENTINEL — emits text
        # passthrough. This is the gap.
        return {"content": delta_text}


def test_finalize_fallback_runs_extract_tool_calls_without_pending_gate():
    """Bug A regression: ``StreamingPostProcessor.finalize`` must run
    ``extract_tool_calls`` on accumulated text even when
    ``has_pending_tool_call`` returns False.

    Rationale: every parser's ``has_pending_tool_call`` reuses the same
    canonical-wrapper check as the streaming code path. If the streaming
    path missed a tool call (because the model used a non-canonical
    format that only ``extract_tool_calls`` knows about), gating
    finalize on ``has_pending_tool_call`` would by construction also
    miss it. Dropping the gate gives streaming the same tolerance as
    non-streaming. See knowledge/guided_generation_gaps_2026-05-20.md
    "Bug A".
    """
    parser = _GapStreamParser()
    cfg = reset_config()
    processor = StreamingPostProcessor(cfg, tools_requested=True)
    # Inject our gap parser directly.
    processor.tool_parser = parser
    processor.reset()

    # Simulate streaming: feed the sentinel text but the streaming parser
    # returns content passthrough, so no tool_calls are detected.
    raw = f"Some preamble {_GapStreamParser.SENTINEL} trailing."
    output = GenerationOutput(
        text=raw,
        new_text=raw,
        prompt_tokens=4,
        completion_tokens=10,
        finished=True,
        finish_reason="stop",
        channel=None,
    )
    events_during = list(processor.process_chunk(output))
    # Streaming parser emitted only content/finish events (no tool_calls).
    assert not any(e.type == "tool_call" for e in events_during)
    assert not processor.tool_calls_detected, (
        "precondition: streaming code path must NOT have detected the "
        "tool call (that's the gap we're testing)"
    )

    # Pre-fix this returned [] because has_pending_tool_call(raw) is
    # False (no '<tool_call>' substring). Post-fix it runs
    # extract_tool_calls anyway and finds the SENTINEL fallback.
    finalize_events = processor.finalize()
    tool_events = [e for e in finalize_events if e.type == "tool_call"]
    assert len(tool_events) == 1, (
        f"finalize must catch tool calls that the streaming parser missed "
        f"via the non-stream parser's fallback patterns; got {finalize_events}"
    )
    assert tool_events[0].tool_calls[0]["function"]["name"] == "get_weather"
    assert tool_events[0].finish_reason == "tool_calls"
    assert processor.tool_calls_detected


def test_finalize_no_op_when_no_tool_calls_present():
    """Drop-of-gate must not generate spurious events when the model
    legitimately produced no tool calls. Bare passthrough text → empty
    finalize, no synthetic tool_call emission.
    """
    parser = _GapStreamParser()
    cfg = reset_config()
    processor = StreamingPostProcessor(cfg, tools_requested=True)
    processor.tool_parser = parser
    processor.reset()

    output = GenerationOutput(
        text="Hello, how can I help?",
        new_text="Hello, how can I help?",
        prompt_tokens=4,
        completion_tokens=6,
        finished=True,
        finish_reason="stop",
        channel=None,
    )
    list(processor.process_chunk(output))

    finalize_events = processor.finalize()
    assert finalize_events == [], (
        "no tool calls in text → finalize must return empty list, not spurious events"
    )
    assert not processor.tool_calls_detected


class _HermesHoldingEngine:
    """Mock engine that streams ``"abc<"`` so the hermes parser holds the
    trailing ``<`` (potential ``<tool_call>``/``<function=`` opener).
    The chat route must merge the released held byte into the terminal
    chunk's delta.content — without this fix the response would surface
    as ``abc`` (codex round-3 + round-4 CRITICAL).
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        for i, delta in enumerate("abc<"):
            text_so_far = "abc<"[: i + 1]
            yield GenerationOutput(
                text=text_so_far,
                new_text=delta,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=(i == 3),
                finish_reason="stop" if i == 3 else None,
                channel=None,
            )


def test_streaming_terminal_chunk_merges_prefix_held_content():
    """Codex round-4 CRITICAL: prefix-held tail must reach the client.

    Hermes/harmony streaming holds back partial tool-call sentinels
    (``<``, ``<|``, ``<func``...). If the stream ends with held bytes
    AND no tool call ever fires, those bytes are ordinary content and
    finalize()'s newly released ``content`` event MUST be merged into
    the terminal chunk's delta.content. Pre-fix the route only consumed
    ``tool_call`` events from finalize() and silently dropped the
    released suffix — user-facing result was ``abc`` instead of
    ``abc<`` (last char missing).
    """
    cfg = reset_config()
    cfg.engine = _HermesHoldingEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "hermes"

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "say abc<"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "dummy",
                        "description": "unused",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done

    content_parts: list[str] = []
    for ev in events:
        for choice in ev.get("choices", []):
            delta = choice.get("delta", {}) or {}
            piece = delta.get("content")
            if piece:
                content_parts.append(piece)
    full_content = "".join(content_parts)
    assert full_content == "abc<", (
        f"terminal chunk must carry the prefix-held tail; "
        f"expected 'abc<' got {full_content!r}"
    )


def test_streaming_terminal_chunk_does_not_duplicate_already_streamed_content():
    """Codex round-6 BLOCKING: when ``_detect_tool_calls`` prefix-holds
    the FINAL finished delta and returns None, the path used to skip
    the finish event entirely. Without ``buffered_finish``, the chat
    route fell back to the synthetic-chunk branch which re-emits
    ``accumulated_text + finalize_content`` — duplicating content the
    client already received via per-delta chunks (``abc<`` ⇒
    ``abc`` streamed + terminal ``abc<`` = ``abcabc<``).

    Fix: emit a finish event from the postprocessor even when the
    detected delta is suppressed, so the normal terminal-chunk path
    fires with the correct (un-duplicated) content + released held
    suffix.
    """
    cfg = reset_config()
    cfg.engine = _HermesHoldingEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "hermes"

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "say abc<"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "dummy",
                        "description": "unused",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    events, _ = _parse_sse_events(resp.text)

    content_parts: list[str] = []
    for ev in events:
        for choice in ev.get("choices", []):
            delta = choice.get("delta", {}) or {}
            piece = delta.get("content")
            if piece:
                content_parts.append(piece)

    full_content = "".join(content_parts)
    # The total wire content must match the input exactly — no doubling.
    assert full_content == "abc<", (
        f"client must receive content exactly once; got {full_content!r}. "
        f"A duplicated reply (e.g. 'abcabc<') means the synthetic-chunk "
        f"path was re-emitting accumulated_text on top of already-streamed "
        f"deltas (codex round-6 BLOCKING)."
    )


def test_terminal_finish_snapshot_does_not_replay_streamed_content(
    monkeypatch,
):
    """#1709: a parser finish snapshot must not become a second response.

    Auto tool-choice parser paths can attach the complete plain-text answer to
    the ``finish`` event (``finish_event.content``) after the same answer was
    emitted as ordinary content deltas — a cumulative snapshot of everything
    already on the wire. The terminal chunk must drop that snapshot. A
    genuinely parser-held suffix arriving via ``finalize`` is a different
    provenance and must NOT be suppressed — see
    ``test_terminal_finalize_suffix_is_not_suppressed_as_replay``.
    """
    from vllm_mlx.service import postprocessor as pp_mod

    original_process_chunk = pp_mod.StreamingPostProcessor.process_chunk

    def finish_with_full_response(self, output):
        for event in original_process_chunk(self, output):
            if event.type == "finish":
                event.content = output.text
            yield event

    monkeypatch.setattr(
        pp_mod.StreamingPostProcessor,
        "process_chunk",
        finish_with_full_response,
    )

    class _SeparateFinishEngine(_PlainStreamEngine):
        async def stream_chat(self, messages, **kwargs):
            accumulated = ""
            for i, delta in enumerate(["hello", " world"]):
                accumulated += delta
                yield GenerationOutput(
                    text=accumulated,
                    new_text=delta,
                    prompt_tokens=4,
                    completion_tokens=i + 1,
                    finished=False,
                    channel=None,
                )
            yield GenerationOutput(
                text=accumulated,
                new_text="",
                prompt_tokens=4,
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                channel=None,
            )

    cfg = reset_config()
    cfg.engine = _SeparateFinishEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "hermes"

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "say hello world"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "dummy",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done

    content_parts = [
        choice["delta"]["content"]
        for event in events
        for choice in event.get("choices", [])
        if (choice.get("delta") or {}).get("content")
    ]
    assert "".join(content_parts) == "hello world"

    terminal_choices = [
        choice
        for event in events
        for choice in event.get("choices", [])
        if choice.get("finish_reason") is not None
    ]
    assert len(terminal_choices) == 1
    assert "content" not in terminal_choices[0]["delta"]


def test_terminal_replay_guard_suppresses_snapshot_split_across_fields(monkeypatch):
    """#1709 (codex): a cumulative snapshot can arrive split across BOTH
    terminal fields — part on ``finish_event.content`` and part on the
    finalize flush — whose concatenation equals everything streamed. Checking
    the fields separately would miss it; the guard compares the COMBINED
    terminal payload. finish ``"hello"`` + finalize ``" world"`` over a fully
    streamed ``"hello world"`` → suppressed (no ``"hello worldhello world"``).
    """
    from vllm_mlx.domain.events import StreamEvent
    from vllm_mlx.service import postprocessor as pp_mod

    original_process_chunk = pp_mod.StreamingPostProcessor.process_chunk

    def finish_with_first_half(self, output):
        for event in original_process_chunk(self, output):
            if event.type == "finish":
                event.content = "hello"  # first half of the snapshot
            yield event

    monkeypatch.setattr(
        pp_mod.StreamingPostProcessor, "process_chunk", finish_with_first_half
    )
    monkeypatch.setattr(
        pp_mod.StreamingPostProcessor,
        "finalize",
        lambda self: [StreamEvent(type="content", content=" world")],
    )

    class _SplitFieldEngine(_PlainStreamEngine):
        async def stream_chat(self, messages, **kwargs):
            accumulated = ""
            for i, delta in enumerate(["hello", " world"]):
                accumulated += delta
                yield GenerationOutput(
                    text=accumulated,
                    new_text=delta,
                    prompt_tokens=4,
                    completion_tokens=i + 1,
                    finished=False,
                    channel=None,
                )
            yield GenerationOutput(
                text=accumulated,
                new_text="",
                prompt_tokens=4,
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                channel=None,
            )

    cfg = reset_config()
    cfg.engine = _SplitFieldEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "auto"

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "say hello world"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "dummy",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done

    content_parts = [
        choice["delta"]["content"]
        for event in events
        for choice in event.get("choices", [])
        if (choice.get("delta") or {}).get("content")
    ]
    assert "".join(content_parts) == "hello world", (
        "a snapshot split across finish + finalize must be suppressed as a "
        f"combined replay; got {''.join(content_parts)!r}"
    )


def test_terminal_finalize_flush_snapshot_does_not_replay_streamed_content(monkeypatch):
    """#1709 real manifestation (auto parser + reasoning model): the
    end-of-stream ``flush_held_content`` returns the ENTIRE content buffer
    because the auto parser's emitted-length counter is bypassed by the
    reasoning parser. That flush duplicates content already streamed as
    deltas. When the model's content buffer was fully streamed
    (``tool_accumulated_text == streamed_content``) the terminal flush must
    be suppressed, not replayed.
    """
    from vllm_mlx.domain.events import StreamEvent
    from vllm_mlx.service import postprocessor as pp_mod

    def finalize_flush_full_buffer(self):
        # Mirror the auto parser's stale-counter flush: buffer == everything
        # already streamed, released again as a terminal content event.
        self.tool_accumulated_text = "hello world"
        return [StreamEvent(type="content", content="hello world")]

    monkeypatch.setattr(
        pp_mod.StreamingPostProcessor, "finalize", finalize_flush_full_buffer
    )

    class _FullBufferEngine(_PlainStreamEngine):
        async def stream_chat(self, messages, **kwargs):
            accumulated = ""
            for i, delta in enumerate(["hello", " world"]):
                accumulated += delta
                yield GenerationOutput(
                    text=accumulated,
                    new_text=delta,
                    prompt_tokens=4,
                    completion_tokens=i + 1,
                    finished=False,
                    channel=None,
                )
            yield GenerationOutput(
                text=accumulated,
                new_text="",
                prompt_tokens=4,
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                channel=None,
            )

    cfg = reset_config()
    cfg.engine = _FullBufferEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "auto"

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "say hello world"}],
        },
    )
    assert resp.status_code == 200, resp.text
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done

    content_parts = [
        choice["delta"]["content"]
        for event in events
        for choice in event.get("choices", [])
        if (choice.get("delta") or {}).get("content")
    ]
    assert "".join(content_parts) == "hello world", (
        "a finalize flush that replays the fully-streamed content buffer must "
        f"be suppressed; got {''.join(content_parts)!r}"
    )


def test_terminal_suffix_survives_when_buffer_unavailable(monkeypatch):
    """#1709 (codex): an empty/unavailable content buffer is NOT proof that
    everything was streamed. Suppression requires AFFIRMATIVE provenance — a
    non-empty buffer that (sanitized) equals the streamed content. With no such
    proof a terminal suffix must survive, even one that happens to equal prior
    output. Streamed ``"ha"`` + finalize ``"ha"`` with an empty buffer →
    ``"haha"`` (not collapsed to ``"ha"``).
    """
    from vllm_mlx.domain.events import StreamEvent
    from vllm_mlx.service import postprocessor as pp_mod

    def finalize_empty_buffer_suffix(self):
        self.tool_accumulated_text = ""  # buffer unavailable at finalize
        return [StreamEvent(type="content", content="ha")]

    monkeypatch.setattr(
        pp_mod.StreamingPostProcessor, "finalize", finalize_empty_buffer_suffix
    )

    class _EmptyBufferEngine(_PlainStreamEngine):
        async def stream_chat(self, messages, **kwargs):
            yield GenerationOutput(
                text="ha",
                new_text="ha",
                prompt_tokens=4,
                completion_tokens=1,
                finished=False,
                channel=None,
            )
            yield GenerationOutput(
                text="ha",
                new_text="",
                prompt_tokens=4,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                channel=None,
            )

    cfg = reset_config()
    cfg.engine = _EmptyBufferEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "auto"

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "say ha"}],
        },
    )
    assert resp.status_code == 200, resp.text
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done

    content_parts = [
        choice["delta"]["content"]
        for event in events
        for choice in event.get("choices", [])
        if (choice.get("delta") or {}).get("content")
    ]
    assert "".join(content_parts) == "haha", (
        "with no affirmative buffer provenance an equal-valued terminal suffix "
        f"must survive; got {''.join(content_parts)!r}"
    )


def test_terminal_replay_guard_handles_sanitizer_sensitive_content(monkeypatch):
    """#1709 (codex): the ``fully-streamed`` provenance check must compare the
    content buffer on the SAME sanitized footing as the wire. The buffer holds
    raw text (e.g. a trailing ``<|im_end|>`` special token) while
    ``streamed_content`` is what ``sanitize_content_for_stream`` actually put
    on the wire. Comparing the raw buffer would fail the check for any
    sanitizer-sensitive content and replay it. Streamed ``"hello"`` (buffer
    ``"hello<|im_end|>"``) + finalize flush of the raw buffer → ``"hello"``.
    """
    from vllm_mlx.domain.events import StreamEvent
    from vllm_mlx.service import postprocessor as pp_mod

    raw = "hello<|im_end|>"  # sanitizes to "hello" on the wire

    def finalize_flush_raw_buffer(self):
        self.tool_accumulated_text = raw
        return [StreamEvent(type="content", content=raw)]

    monkeypatch.setattr(
        pp_mod.StreamingPostProcessor, "finalize", finalize_flush_raw_buffer
    )

    class _RawTokenEngine(_PlainStreamEngine):
        async def stream_chat(self, messages, **kwargs):
            yield GenerationOutput(
                text=raw,
                new_text=raw,
                prompt_tokens=4,
                completion_tokens=1,
                finished=False,
                channel=None,
            )
            yield GenerationOutput(
                text=raw,
                new_text="",
                prompt_tokens=4,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                channel=None,
            )

    cfg = reset_config()
    cfg.engine = _RawTokenEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "auto"

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "say hello"}],
        },
    )
    assert resp.status_code == 200, resp.text
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done

    content_parts = [
        choice["delta"]["content"]
        for event in events
        for choice in event.get("choices", [])
        if (choice.get("delta") or {}).get("content")
    ]
    assert "".join(content_parts) == "hello", (
        "sanitizer-sensitive content must be recognised as fully streamed and "
        f"its terminal replay suppressed; got {''.join(content_parts)!r}"
    )


def test_terminal_finalize_suffix_is_not_suppressed_as_replay(monkeypatch):
    """#1709 follow-up (codex BLOCKING): the replay guard keys on PROVENANCE
    — the model's content buffer was fully streamed — not on content equality
    alone. A genuinely parser-held suffix is the not-yet-streamed tail: the
    model produced ``"haha"`` (buffer), only ``"ha"`` reached the wire, and
    the held ``"ha"`` is released at finalize. Because the buffer (``"haha"``)
    is NOT equal to what was streamed (``"ha"``), the held tail must be
    emitted → ``"haha"``, never collapsed to ``"ha"``.
    """
    from vllm_mlx.domain.events import StreamEvent
    from vllm_mlx.service import postprocessor as pp_mod

    def finalize_release_held_tail(self):
        # Model actually produced "haha"; the parser streamed "ha" and held
        # the second "ha" behind a marker that never became a tool call.
        self.tool_accumulated_text = "haha"
        return [StreamEvent(type="content", content="ha")]

    monkeypatch.setattr(
        pp_mod.StreamingPostProcessor, "finalize", finalize_release_held_tail
    )

    class _HeldSuffixEngine(_PlainStreamEngine):
        async def stream_chat(self, messages, **kwargs):
            yield GenerationOutput(
                text="ha",
                new_text="ha",
                prompt_tokens=4,
                completion_tokens=1,
                finished=False,
                channel=None,
            )
            yield GenerationOutput(
                text="ha",
                new_text="",
                prompt_tokens=4,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                channel=None,
            )

    cfg = reset_config()
    cfg.engine = _HeldSuffixEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "auto"

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "say ha"}],
        },
    )
    assert resp.status_code == 200, resp.text
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done

    content_parts = [
        choice["delta"]["content"]
        for event in events
        for choice in event.get("choices", [])
        if (choice.get("delta") or {}).get("content")
    ]
    assert "".join(content_parts) == "haha", (
        "a parser-held finalize suffix (buffer != streamed) must be preserved, "
        f"not suppressed as a replay; got {''.join(content_parts)!r}"
    )


def test_synthetic_terminal_chunk_does_not_replay_accumulated_text(monkeypatch):
    """Defense-in-depth: the synthetic-chunk fallback (reached when
    ``buffered_finish`` is None but ``finalize_content`` is present)
    must NOT include ``processor.accumulated_text`` in its delta.content
    — the deltas were already written to the wire during the loop, so
    replaying them would duplicate the entire response.

    The round-6 postprocessor fix makes this branch unreachable in the
    common case (process_chunk always emits a finish event when the
    output is finished). This test forces the branch via monkeypatch
    to pin the defensive code's invariant: only emit material that has
    NOT yet been streamed (``finalize_content`` + ``fallback_tool_calls``).

    A regression here would mean a future code path that swallows the
    finish event (exception in middleware, new processor variant, etc.)
    would double-send the entire reply via the synthetic chunk.
    """
    from vllm_mlx.domain.events import StreamEvent
    from vllm_mlx.service import postprocessor as pp_mod

    # Force the defensive branch:
    #   * process_chunk yields ONLY content events (no finish event) →
    #     buffered_finish stays None
    #   * finalize yields a "content" event with the held tail →
    #     finalize_content is non-empty → branch fires
    original_process_chunk = pp_mod.StreamingPostProcessor.process_chunk

    def patched_process_chunk(self, output):
        # Drop finish events so the route's buffered_finish stays None
        for ev in original_process_chunk(self, output):
            if ev.type != "finish":
                yield ev

    def patched_finalize(self):
        # Emit a single content event with a sentinel "tail" string.
        # The defensive branch should emit ONLY this, not also the
        # accumulated_text from the loop.
        return [StreamEvent(type="content", content="<")]

    monkeypatch.setattr(
        pp_mod.StreamingPostProcessor, "process_chunk", patched_process_chunk
    )
    monkeypatch.setattr(pp_mod.StreamingPostProcessor, "finalize", patched_finalize)

    cfg = reset_config()
    cfg.engine = _PlainStreamEngine(deltas=["Hel", "lo", " world"])
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = False

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    events, _ = _parse_sse_events(resp.text)

    content_parts: list[str] = []
    for ev in events:
        for choice in ev.get("choices", []):
            delta = choice.get("delta", {}) or {}
            piece = delta.get("content")
            if piece:
                content_parts.append(piece)
    full_content = "".join(content_parts)

    # The postprocessor bundles content+finish into a single ``finish``
    # event on the finished chunk; the monkeypatch drops finish events,
    # so the last delta (`` world``) is consumed alongside the finish
    # event. That is harmless to the invariant under test — what matters
    # is that ``accumulated_text`` (which DOES include all three deltas)
    # is NOT replayed in the synthetic chunk.
    #
    # Post-fix behavior: client receives "Hel" + "lo" (loop content
    # events) + "<" (defensive synthetic chunk emitting only the
    # finalize tail) = "Hello<".
    #
    # Bug behavior (pre-fix): client would receive "Hel" + "lo"
    # (loop content events) + "Hello world<" (synthetic chunk replaying
    # accumulated_text + finalize tail) = "HelloHello world<".
    assert full_content == "Hello<", (
        f"defensive synthetic chunk must NOT replay accumulated_text; "
        f"expected 'Hello<' (loop deltas the client received + held tail "
        f"emitted once by the defensive branch), got {full_content!r}. "
        f"A duplicated prefix like 'HelloHello world<' means the synthetic "
        f"chunk re-emitted processor.accumulated_text on top of already-"
        f"streamed deltas (raised in Codex re-review)."
    )


# -----------------------------------------------------------------------
# R11-A: route-level exactly-one-finish-reason invariants
# -----------------------------------------------------------------------


class _InlineToolCallFinishEngine:
    """Mock engine that produces a SINGLE final chunk carrying a complete
    hermes tool_call. The hermes streaming parser will surface a
    ``tool_call`` StreamEvent with ``finish_reason="tool_calls"`` inline
    on the finished chunk — the codex r1 HIGH #1 path that, pre-fix,
    fell through to the new synthetic finish branch and emitted a SECOND
    terminal chunk with ``finish_reason="stop"``.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        body = (
            "<tool_call>\n"
            '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
            "</tool_call>"
        )
        yield GenerationOutput(
            text=body,
            new_text=body,
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


def test_r11a_inline_tool_call_finish_does_not_double_emit_terminal():
    """Codex r1 HIGH #1 regression: a single-chunk valid tool call ends
    with the postprocessor stamping ``finish_reason="tool_calls"`` on
    the tool_call event itself (no separate ``finish`` event). The
    route MUST NOT fall through to the R11-V2 synthetic-finish branch
    and emit a SECOND terminal chunk. The wire MUST carry exactly one
    chunk with a non-null finish_reason.
    """
    cfg = reset_config()
    cfg.engine = _InlineToolCallFinishEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "hermes"

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done

    # Invariant: exactly one chunk on the wire with a non-null
    # finish_reason, and that reason MUST be "tool_calls" (the inline
    # stamping wins, the synthetic-finish branch is gated off).
    finish_chunks: list = []
    for ev in events:
        for ch in ev.get("choices", []):
            if ch.get("finish_reason") is not None:
                finish_chunks.append(ch["finish_reason"])
    assert len(finish_chunks) == 1, (
        f"R11-A codex r1 HIGH #1: expected exactly one terminal "
        f"finish_reason on the wire, got {finish_chunks!r}. A second "
        f"finish_reason=stop chunk from the R11-V2 synthetic-finish "
        f"branch would split the spec-mandated 'exactly one finish' "
        f"contract."
    )
    assert finish_chunks[0] == "tool_calls"
    # And the tool_call delta MUST be on the wire (consistency check
    # — the inline-finish stamping only happens on tool_call events).
    tool_call_delta_count = 0
    for ev in events:
        for ch in ev.get("choices", []):
            delta = ch.get("delta") or {}
            if delta.get("tool_calls"):
                tool_call_delta_count += len(delta["tool_calls"])
    assert tool_call_delta_count >= 1


def test_r11a_v2_synthetic_finish_fires_when_no_inline_finish():
    """R11-V2 positive case: when the engine ends WITHOUT surfacing a
    finished chunk AND finalize() recovers nothing, the route's
    synthetic-finish branch MUST fire so the wire envelope is
    well-formed (exactly one chunk with finish_reason). Without the
    branch, spec-compliant clients stall at ``[DONE]`` with no
    terminal marker.
    """

    class _NoFinishEngine:
        preserve_native_tool_format = False
        is_mllm = False
        supports_guided_generation = False
        tokenizer = None

        def build_prompt(self, messages, tools=None, enable_thinking=None):
            return "PROMPT"

        async def stream_chat(self, messages, **kwargs):
            # Surface some text but never set finished=True. The
            # postprocessor will not emit a finish event, finalize()
            # has no tool material to recover, so the route's R11-V2
            # branch must synthesize the terminal chunk.
            yield GenerationOutput(
                text="hi",
                new_text="hi",
                prompt_tokens=4,
                completion_tokens=1,
                finished=False,
                finish_reason=None,
                channel=None,
            )

    cfg = reset_config()
    cfg.engine = _NoFinishEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done

    finish_chunks: list = []
    for ev in events:
        for ch in ev.get("choices", []):
            if ch.get("finish_reason") is not None:
                finish_chunks.append(ch["finish_reason"])
    # Exactly one synthesized "stop" — R11-V2 invariant met.
    assert finish_chunks == ["stop"], (
        f"R11-V2 regression: expected synthetic finish_reason=stop "
        f"when engine never emitted a finished chunk; got {finish_chunks!r}"
    )


def test_confirmed_disconnect_does_not_synthesize_missing_finish(caplog):
    """Expected client cancellation must not masquerade as a broken SSE stream.

    The disconnect guard cancels the in-flight engine iteration. Batched engines
    deliberately turn that cancellation into clean iterator exhaustion after
    aborting the scheduler request, which previously made the route emit the
    R11-V2 warning and construct a terminal chunk for an already-dead socket.
    """

    class _CancelledEngine:
        preserve_native_tool_format = False
        is_mllm = False
        supports_guided_generation = False
        tokenizer = None

        async def stream_chat(self, messages, **kwargs):
            yield GenerationOutput(
                text="partial",
                new_text="partial",
                prompt_tokens=4,
                completion_tokens=1,
                finished=False,
                finish_reason=None,
                channel=None,
            )

    cfg = reset_config()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    state = [True]
    request = ChatCompletionRequest(
        model="test-model",
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )

    async def _collect():
        return [
            chunk
            async for chunk in stream_chat_completion(
                _CancelledEngine(),
                request.messages,
                request,
                _client_disconnect_state=state,
            )
        ]

    chunks = asyncio.run(_collect())
    assert not any("finish_reason" in chunk for chunk in chunks)
    assert not any("[DONE]" in chunk for chunk in chunks)
    assert "SSE-MISSING-FINISH-R11V2" not in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
