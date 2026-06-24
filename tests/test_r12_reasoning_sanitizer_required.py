# SPDX-License-Identifier: Apache-2.0
"""R12-FIX-V2 — sanitize ``reasoning_content`` on the
``tool_choice="required"`` branch (Vlad r12 MED-2, /tmp/dogfood-0815/vlad-r12.md).

**Repro shape** (dogfood report, qwen3-0.6b-4bit on port 8090):

    body = {
      "model": "qwen3-0.6b-4bit",
      "messages": [{"role":"user","content":"What's the weather in Tokyo?"}],
      "tools": [...get_weather...],
      "tool_choice": "required",
      "max_tokens": 200,
      "temperature": 0
    }
    # Response: message.reasoning_content == "<|im_start|>"  (literal special token)

**Root cause**: ``vllm_mlx/routes/chat.py`` called ``sanitize_output()`` on
``final_content`` (line ~3277) but passed ``reasoning_text`` to
``AssistantMessage(reasoning_content=...)`` (line ~3380) WITHOUT
sanitization. The ``tool_choice="required"`` post-parse branch on
qwen3 emits a forced-prefix replay that, on certain prompts, drops a
residual ``<|im_start|>`` chat-template token into
``reasoning_text`` — the dedicated ``content`` sanitizer never sees
it. Streaming SSE and the non-stream Anthropic / Responses adapters
inherit the leak because they all read ``message.reasoning_content``.

**Systematic fix**: sanitize every user-visible string field at the
type boundary (``AssistantMessage`` + ``ChatCompletionChunkDelta``
field validators) + sanitize the ``_fast_sse_chunk`` streaming
hot path which bypasses pydantic serialization. The contract is
enforced at the message constructor so new call sites cannot reopen
the leak by forgetting to sanitize.

These tests pin:

1. The unit-level invariant — ``AssistantMessage(reasoning_content=...)``
   strips leaked special tokens regardless of how the field was set.
2. The streaming-side invariant — ``ChatCompletionChunkDelta`` and
   ``_fast_sse_chunk`` both strip the same set.
3. The end-to-end invariant — the chat route under
   ``tool_choice="required"`` returns clean ``reasoning_content``
   even when the raw model emission contained ``<|im_start|>`` etc.
4. Parametric coverage across ``tool_choice`` values (``"auto"``,
   ``"required"``, ``"none"``, named function) so the sanitizer
   applies uniformly — not just on the bug-repro path.
5. Streaming variant of (3): aggregated ``reasoning_content`` from
   delta chunks is also leak-free.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChunkDelta,
)
from vllm_mlx.api.utils import (
    sanitize_output,
    sanitize_reasoning_content,
    sanitize_reasoning_for_stream,
)
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
from vllm_mlx.routes.chat import router as chat_router

# Markers Vlad's report singled out, plus the canonical set every other
# parser-leak hardening test asserts.
_LEAK_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|channel|>",
    "<|message|>",
    "</think>",
    "</tool_call>",
)


# ──────────────────────────────────────────────────────────────────
# Unit-level invariant on the model envelopes
# ──────────────────────────────────────────────────────────────────


class TestSanitizeReasoningHelpers:
    """The two helpers exposed in ``vllm_mlx/api/utils.py`` are the
    single source of truth — pin their contract here so the field
    validators can rely on it without re-asserting per-call-site.
    """

    @pytest.mark.parametrize("marker", _LEAK_MARKERS)
    def test_sanitize_reasoning_content_strips_marker(self, marker):
        """A reasoning trace containing any known special-token marker
        must have the marker stripped — same semantic as
        ``sanitize_output`` for content."""
        text = f"thinking out loud {marker} answer is 42"
        out = sanitize_reasoning_content(text)
        assert out is not None
        assert marker not in out
        # And the surrounding text survives.
        assert "thinking out loud" in out
        assert "answer is 42" in out

    @pytest.mark.parametrize("marker", _LEAK_MARKERS)
    def test_sanitize_reasoning_content_collapses_pure_markup_to_none(self, marker):
        """When the trace is ENTIRELY special-token markup,
        ``sanitize_reasoning_content`` returns ``None`` so the field
        drops out under ``exclude_none`` serialization — same shape
        as ``sanitize_output``.
        """
        assert sanitize_reasoning_content(marker) is None

    def test_sanitize_reasoning_content_passes_through_plain_text(self):
        """No marker chars → no rewrite; the fast-path bypass kicks
        in and the input is returned unchanged (identity preservation
        matters for hot-path streaming)."""
        text = "Let me think step by step. The answer is 42."
        assert sanitize_reasoning_content(text) is text

    def test_sanitize_reasoning_content_handles_none_and_empty(self):
        assert sanitize_reasoning_content(None) is None
        assert sanitize_reasoning_content("") == ""

    @pytest.mark.parametrize("marker", _LEAK_MARKERS)
    def test_sanitize_reasoning_for_stream_collapses_to_empty_string(self, marker):
        """The streaming variant returns ``""`` (not ``None``) for a
        fully-stripped trace so per-delta JSON serialization stays
        type-stable (``delta.reasoning_content: ""`` is a valid
        zero-byte delta; ``null`` would change the field's wire
        type)."""
        assert sanitize_reasoning_for_stream(marker) == ""

    def test_sanitize_reasoning_for_stream_passes_through_plain_text(self):
        text = "step 1: figure out the city"
        assert sanitize_reasoning_for_stream(text) == text

    def test_sanitize_reasoning_for_stream_handles_none_and_empty(self):
        assert sanitize_reasoning_for_stream(None) == ""
        assert sanitize_reasoning_for_stream("") == ""

    def test_sanitize_reasoning_for_stream_preserves_leading_whitespace(self):
        """Codex r2 [P2] on R12-FIX-V2: streaming clients concatenate
        deltas verbatim. ``.strip()``-ing an individual delta corrupts
        cross-delta boundaries — e.g. prior delta ``"foo"`` + this
        delta ``" bar <|im_start|>"`` must arrive as ``"foo bar "`` not
        ``"foobar"``. The streaming sanitizer removes ONLY the marker
        bytes and leaves surrounding whitespace intact.
        """
        out = sanitize_reasoning_for_stream(" bar <|im_start|>")
        # Leading whitespace MUST survive so the concatenation with the
        # prior delta keeps the word boundary.
        assert out.startswith(" "), (
            f"streaming sanitizer must preserve leading whitespace; got {out!r}"
        )
        assert "<|im_start|>" not in out
        assert "bar" in out

    def test_sanitize_reasoning_for_stream_preserves_trailing_whitespace(self):
        """Mirror of the leading-whitespace test: ``"<|im_start|> next"``
        must NOT trim the leading space after marker removal — it's
        meaningful between the prior delta and the surviving prose."""
        out = sanitize_reasoning_for_stream("<|im_start|> next")
        assert out.endswith(" next"), (
            f"streaming sanitizer must preserve whitespace after marker "
            f"removal; got {out!r}"
        )
        assert "<|im_start|>" not in out

    def test_sanitize_reasoning_for_stream_preserves_newline_whitespace(self):
        """Newlines inside reasoning deltas are equally load-bearing —
        the streaming concatenation contract requires the sanitizer to
        leave newlines around stripped markers intact."""
        out = sanitize_reasoning_for_stream("step1\n<|im_end|>\nstep2")
        assert "<|im_end|>" not in out
        assert out == "step1\n\nstep2", (
            f"newlines surrounding marker must survive; got {out!r}"
        )

    def test_sanitize_reasoning_for_stream_pure_marker_collapses_to_empty(self):
        """When the delta is purely markup (no surrounding text),
        the result is ``""`` so the caller can suppress the empty
        delta without surprising clients."""
        for marker in _LEAK_MARKERS:
            assert sanitize_reasoning_for_stream(marker) == "", (
                f"pure-marker delta must collapse to empty; marker={marker!r}"
            )

    def test_sanitize_reasoning_for_stream_two_delta_concat_repro(self):
        """End-to-end of the codex r2 concern: ``"foo"`` + ``" bar
        <|im_start|>"`` must concatenate to ``"foo bar "`` — pre-fix
        the second delta sanitized to ``"bar"`` and clients saw
        ``"foobar"``.
        """
        prior = "foo"
        leaky = " bar <|im_start|>"
        sanitized = sanitize_reasoning_for_stream(leaky)
        joined = prior + sanitized
        assert "foobar" not in joined, (
            f"streaming concat regression: deltas {prior!r}+{leaky!r} "
            f"produced {joined!r}, must preserve the inter-word space"
        )
        assert "foo bar" in joined

    def test_helper_parity_with_sanitize_output_content_semantic(self):
        """The reasoning-side sanitizer MUST agree byte-for-byte with
        the content-side sanitizer. Single source of truth: drift
        between the two re-creates the original bug (sanitize content
        only, leak through reasoning)."""
        for text in [
            "<|im_start|>",
            "abc<|im_end|>def",
            "plain text",
            "",
            None,
            "mix <|channel|>final<|message|> with text",
        ]:
            assert sanitize_reasoning_content(text) == sanitize_output(text)

    def test_sanitize_reasoning_preserves_literal_tool_call_mention(self):
        """Codex r4 [P2] on R12-FIX-V2 considered and rejected:
        adding plain ``<tool_call>`` opener stripping to the global
        sanitizer would break the existing T1/T2/T3
        ``tool_choice="required"`` test suite, which intentionally
        pins that legitimate prose mentioning ``<tool_call>`` as text
        MUST survive (e.g. a model explaining the wire format). The
        route-level ``_scrub_visible_tool_wire_leaks`` already
        discriminates structural wire vs literal-token-mention via
        ``_contains_structural_tool_wire_leak`` and runs on
        ``reasoning_text`` for the forced/required path. Defense-
        in-depth at the global sanitizer would over-strip.

        Pin: ``sanitize_reasoning_content`` does NOT remove a literal
        ``<tool_call>`` mention from prose — the contract is
        preserved.
        """
        text = "I cannot call tools here, but the literal token is <tool_call>."
        # Marker char ``<`` triggers the regex pass but the result
        # still contains the bare opener — that's intentional, the
        # layered architecture handles structural leaks at the
        # route level.
        out = sanitize_reasoning_content(text)
        assert out is not None
        assert "<tool_call>" in out, (
            f"sanitize_output must NOT strip literal <tool_call> mention; "
            f"existing route-level scrubber owns structural detection. "
            f"got {out!r}"
        )


class TestAssistantMessageSanitizes:
    """The ``AssistantMessage`` field validator is the chokepoint.
    Every call site that constructs the message — chat route,
    Responses adapter input, Anthropic adapter input — funnels through
    here. Pin the contract so a future refactor that adds a fourth
    call site can't reopen the leak.
    """

    @pytest.mark.parametrize("marker", _LEAK_MARKERS)
    def test_reasoning_content_special_token_is_stripped_on_construction(self, marker):
        """The Vlad r12 repro shape: reasoning_content is set to a
        bare special token. The validator must strip it so the
        serialized envelope never carries the marker."""
        msg = AssistantMessage(content="ok", reasoning_content=marker)
        assert msg.reasoning_content is None, (
            f"AssistantMessage must strip pure-markup reasoning_content; "
            f"got {msg.reasoning_content!r}"
        )

    @pytest.mark.parametrize("marker", _LEAK_MARKERS)
    def test_content_special_token_is_stripped_on_construction(self, marker):
        """Parity: content side is sanitized identically (defense in
        depth — the chat route already sanitized content explicitly
        but other call sites must get the same guarantee)."""
        msg = AssistantMessage(content=marker, reasoning_content=None)
        assert msg.content is None

    @pytest.mark.parametrize("marker", _LEAK_MARKERS)
    def test_reasoning_content_mixed_text_only_strips_markers(self, marker):
        """When the trace mixes legitimate prose with a leaked
        marker, only the marker is removed — the actual reasoning
        survives."""
        text = f"working through the steps {marker} arrived at 42"
        msg = AssistantMessage(content="final", reasoning_content=text)
        assert msg.reasoning_content is not None
        assert marker not in msg.reasoning_content
        assert "working through the steps" in msg.reasoning_content
        assert "arrived at 42" in msg.reasoning_content

    def test_plain_reasoning_content_is_untouched(self):
        """Identity preservation on the happy path — no rewrite cost
        and no risk of accidentally mangling real reasoning prose."""
        text = "Let me work through this. 1 + 1 = 2. So the answer is 2."
        msg = AssistantMessage(content="2", reasoning_content=text)
        assert msg.reasoning_content == text

    def test_none_reasoning_content_stays_none(self):
        msg = AssistantMessage(content="ok", reasoning_content=None)
        assert msg.reasoning_content is None

    def test_dumped_envelope_omits_pure_markup_reasoning(self):
        """End-to-end serializer: ``model_dump_json(exclude_none=True)``
        must NOT carry ``reasoning_content`` when the input was pure
        markup. Pre-fix the field round-tripped to the wire."""
        msg = AssistantMessage(content="ok", reasoning_content="<|im_start|>")
        dumped = msg.model_dump(exclude_none=True)
        assert "reasoning_content" not in dumped


class TestChunkDeltaSanitizes:
    """Streaming-side parity with ``AssistantMessage``."""

    @pytest.mark.parametrize("marker", _LEAK_MARKERS)
    def test_reasoning_delta_special_token_is_stripped(self, marker):
        delta = ChatCompletionChunkDelta(reasoning_content=marker)
        assert delta.reasoning_content is None

    @pytest.mark.parametrize("marker", _LEAK_MARKERS)
    def test_content_delta_special_token_is_stripped(self, marker):
        delta = ChatCompletionChunkDelta(content=marker)
        assert delta.content is None

    def test_plain_reasoning_delta_is_untouched(self):
        delta = ChatCompletionChunkDelta(reasoning_content="next step:")
        assert delta.reasoning_content == "next step:"

    def test_content_delta_preserves_leading_whitespace(self):
        """Codex r3 [P2] on R12-FIX-V2: with ``logprobs`` enabled, the
        streaming loop serializes per-delta chunks through pydantic
        instead of the ``_fast_sse_chunk`` fast path — so the
        ``ChatCompletionChunkDelta`` validator must use the same
        whitespace-preserving sanitizer the fast path uses. Pre-fix
        the validator delegated to ``sanitize_reasoning_content`` which
        calls ``.strip()`` and would produce ``"foobar"`` from
        ``"foo"`` + sanitized(``" bar <|im_start|>"``).
        """
        delta = ChatCompletionChunkDelta(content=" bar <|im_start|>")
        assert delta.content is not None
        assert delta.content.startswith(" "), (
            f"ChatCompletionChunkDelta must preserve leading whitespace; "
            f"got {delta.content!r}"
        )
        assert "<|im_start|>" not in delta.content
        assert "bar" in delta.content

    def test_reasoning_delta_preserves_leading_whitespace(self):
        """Same contract on the reasoning_content side."""
        delta = ChatCompletionChunkDelta(reasoning_content=" thinking <|im_end|>")
        assert delta.reasoning_content is not None
        assert delta.reasoning_content.startswith(" ")
        assert "<|im_end|>" not in delta.reasoning_content

    def test_content_delta_two_delta_concat_repro(self):
        """End-to-end of the codex r3 scenario through the pydantic
        path (the path the logprobs streaming branch takes):
        ``"foo"`` + ``" bar <|im_start|>"`` must concatenate to
        ``"foo bar "``, NOT ``"foobar"``.
        """
        d1 = ChatCompletionChunkDelta(content="foo")
        d2 = ChatCompletionChunkDelta(content=" bar <|im_start|>")
        joined = (d1.content or "") + (d2.content or "")
        assert "foobar" not in joined, (
            f"pydantic delta validator must preserve cross-delta whitespace; "
            f"joined={joined!r}"
        )
        assert "foo bar" in joined


# ──────────────────────────────────────────────────────────────────
# End-to-end through the chat route — non-streaming
# ──────────────────────────────────────────────────────────────────


class _ReasoningLeakEngine:
    """Mock engine that surfaces a raw text shaped like the qwen3
    forced-prefix-replay leak Vlad observed: a closed ``<think>``
    block whose body carries a residual ``<|im_start|>`` token (the
    chat-template separator the engine didn't fully consume on the
    ``tool_choice="required"`` path), followed by a hermes
    ``<tool_call>{json}</tool_call>`` envelope.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None
    supports_tool_calls = True

    def __init__(self, raw: str):
        self.last_chat_kwargs: dict[str, Any] | None = None
        self.last_messages: Any = None
        self._text = raw
        self._raw_text = raw

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_chat_kwargs = kwargs
        return GenerationOutput(
            text=self._text,
            raw_text=self._raw_text,
            prompt_tokens=4,
            completion_tokens=20,
            finished=True,
            finish_reason="stop",
        )


_WEATHER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
]


# Mirrors the dogfood repro shape: closed ``<think>`` body with a
# leaked ``<|im_start|>`` mid-thought, followed by a clean hermes
# tool-call envelope so the parser DOES extract the call (the
# success path Vlad observed — tool_calls=[get_weather(...)],
# reasoning_content=<leak>).
_LEAKY_QWEN_RAW = (
    "<think>"
    "I need the weather in Tokyo. <|im_start|>"
    "Let me call the weather tool."
    "</think>"
    '<tool_call>{"name":"get_weather","arguments":{"city":"Tokyo"}}</tool_call>'
)


def _make_qwen3_required_client() -> tuple[TestClient, _ReasoningLeakEngine]:
    engine = _ReasoningLeakEngine(_LEAKY_QWEN_RAW)
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "qwen3-0.6b-4bit"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser = Qwen3ReasoningParser(tokenizer=None)
    cfg.reasoning_parser_name = "qwen3"
    cfg.tool_call_parser = "hermes"
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app), engine


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        "auto",
        "none",
        {"type": "function", "function": {"name": "get_weather"}},
    ],
    ids=["required", "auto", "none", "named-function"],
)
def test_chat_route_reasoning_content_sanitized_across_tool_choice(tool_choice):
    """The systematic invariant: ``reasoning_content`` MUST be
    leak-free regardless of ``tool_choice`` value. Pre-fix the
    sanitizer only fired on the explicit ``content`` channel — the
    ``required`` branch (and any other path that constructs an
    ``AssistantMessage``) leaked.

    Run the same leaky-raw fixture through every ``tool_choice`` value
    the OpenAI spec accepts; every code path must produce a clean
    ``reasoning_content``. ``"none"`` is included because it routes
    through a different post-parse branch (no tool-call extraction)
    so the sanitization gate has to fire there too.
    """
    client, _ = _make_qwen3_required_client()
    body: dict[str, Any] = {
        "model": "qwen3-0.6b-4bit",
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
        "max_tokens": 64,
    }
    if tool_choice != "none":
        body["tools"] = _WEATHER_TOOL
    body["tool_choice"] = tool_choice
    if tool_choice == "none":
        # ``tool_choice="none"`` with no tools array is the
        # canonical no-tools shape; OpenAI accepts both forms.
        body.pop("tools", None)

    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200, resp.text
    msg = resp.json()["choices"][0]["message"]

    reasoning = msg.get("reasoning_content") or ""
    for leak in _LEAK_MARKERS:
        assert leak not in reasoning, (
            f"R12-MED-2 leak: {leak!r} survived in reasoning_content "
            f"on tool_choice={tool_choice!r}: reasoning={reasoning!r}"
        )

    # And content stays clean too (defense-in-depth — the original
    # sanitize_output gate has not regressed).
    content = msg.get("content") or ""
    for leak in _LEAK_MARKERS:
        assert leak not in content, (
            f"R12-MED-2 leak: {leak!r} survived in content "
            f"on tool_choice={tool_choice!r}: content={content!r}"
        )


def test_chat_route_required_repro_exact_vlad_shape():
    """The exact shape from Vlad's r12 dogfood report:
    ``tool_choice="required"``, single ``get_weather`` tool, qwen3
    chat model. Asserts the bug-repro literal ``<|im_start|>`` is
    NOT in ``message.reasoning_content``.
    """
    client, _ = _make_qwen3_required_client()
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-0.6b-4bit",
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
            "tools": _WEATHER_TOOL,
            "tool_choice": "required",
            "max_tokens": 200,
            "temperature": 0,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    msg = body["choices"][0]["message"]

    # Tool call survived (the required-branch did its job).
    assert msg.get("tool_calls"), (
        f"tool_choice=required must produce tool_calls; got msg={msg!r}"
    )
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"

    # The bug Vlad caught: reasoning_content == "<|im_start|>".
    # Pre-fix this assertion would fail.
    reasoning = msg.get("reasoning_content") or ""
    assert "<|im_start|>" not in reasoning, (
        f"Vlad r12 MED-2 repro: <|im_start|> still leaks into "
        f"reasoning_content under tool_choice=required: {reasoning!r}"
    )


# ──────────────────────────────────────────────────────────────────
# End-to-end through the chat route — streaming
# ──────────────────────────────────────────────────────────────────


def _parse_sse_stream(raw: bytes) -> list[dict]:
    """Decode SSE ``data:`` frames into a list of parsed JSON chunks
    (the ``[DONE]`` sentinel is skipped). Mirrors the helper used in
    sibling streaming tests."""
    out: list[dict] = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]" or not payload:
            continue
        try:
            out.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return out


class _StreamingLeakEngine:
    """Streaming engine that yields raw deltas matching the Vlad
    repro shape — a leaked ``<|im_start|>`` arrives inside the
    ``<think>`` body during the stream, then the close + tool call
    follows. The chat route's streaming postprocessor routes the
    ``<think>`` body to ``reasoning_content`` deltas; the sanitizer
    must intercept the leaked token before the wire delta is
    emitted.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None
    supports_tool_calls = True

    def __init__(self, raw_pieces: list[str]):
        self._pieces = raw_pieces
        self.last_chat_kwargs: dict[str, Any] | None = None
        self.last_messages: Any = None

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_chat_kwargs = kwargs
        accumulated_raw = ""
        accumulated_text = ""
        n = len(self._pieces)
        for idx, piece in enumerate(self._pieces):
            accumulated_raw += piece
            accumulated_text += piece
            is_last = idx == n - 1
            yield GenerationOutput(
                text=accumulated_text,
                new_text=piece,
                raw_text=accumulated_raw,
                prompt_tokens=4,
                completion_tokens=idx + 1,
                finished=is_last,
                finish_reason="stop" if is_last else None,
            )


_LEAKY_STREAM_PIECES = [
    "<think>",
    "I need the weather in Tokyo. ",
    "<|im_start|>",
    "Let me call the weather tool.",
    "</think>",
    '<tool_call>{"name":"get_weather","arguments":{"city":"Tokyo"}}</tool_call>',
]


def test_chat_route_streaming_required_reasoning_is_sanitized():
    """Streaming variant of the Vlad r12 repro: aggregate every
    ``delta.reasoning_content`` SSE frame and assert no leaked
    special token survives. Pre-fix the streaming hot-path
    (``_fast_sse_chunk``) bypassed pydantic serialization so it
    leaked the same ``<|im_start|>`` directly into per-delta JSON.
    """
    engine = _StreamingLeakEngine(_LEAKY_STREAM_PIECES)
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "qwen3-0.6b-4bit"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser = Qwen3ReasoningParser(tokenizer=None)
    cfg.reasoning_parser_name = "qwen3"
    cfg.tool_call_parser = "hermes"
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-0.6b-4bit",
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
            "tools": _WEATHER_TOOL,
            "tool_choice": "required",
            "max_tokens": 200,
            "stream": True,
        },
    )
    assert resp.status_code == 200, resp.text
    chunks = _parse_sse_stream(resp.content)

    # Aggregate reasoning_content across every delta — same
    # canonical OpenAI SDK aggregation pattern.
    aggregated_reasoning = ""
    aggregated_content = ""
    for ch in chunks:
        for choice in ch.get("choices", []):
            delta = choice.get("delta", {}) or {}
            rc = delta.get("reasoning_content")
            if isinstance(rc, str):
                aggregated_reasoning += rc
            c = delta.get("content")
            if isinstance(c, str):
                aggregated_content += c

    for leak in _LEAK_MARKERS:
        assert leak not in aggregated_reasoning, (
            f"R12-MED-2 streaming leak: {leak!r} survived in "
            f"aggregated reasoning_content: {aggregated_reasoning!r}"
        )
        assert leak not in aggregated_content, (
            f"R12-MED-2 streaming leak: {leak!r} survived in "
            f"aggregated content: {aggregated_content!r}"
        )

    # The non-marker reasoning prose still made it through — we
    # didn't accidentally scrub the whole channel.
    assert "I need the weather in Tokyo." in aggregated_reasoning, (
        f"sanitizer over-strip: legitimate reasoning prose lost; "
        f"got reasoning={aggregated_reasoning!r}"
    )
