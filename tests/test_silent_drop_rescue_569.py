# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #569.

gemma-4-26b-4bit multi-turn tool calls silently drop content + tool_calls
when reasoning doesn't terminate. The model gets stuck inside
``<|channel>thought\n…`` and runs out of its token budget before
emitting any ``<|tool_call>`` or ``<|channel>content`` marker. The
engine's token-level ``OutputRouter`` correctly routes every token to
``reasoning``, but the route then emits an OpenAI-compat message with
``content=null`` and ``tool_calls=null`` while ``reasoning_content``
carries the entire stuck thought. Agentic clients (Cline, Cursor,
Codex CLI) read ``content`` and ``tool_calls`` only, see an empty
message, and either retry into the same trap or stall.

The fix: ``_rescue_silent_drop_from_reasoning`` runs at the route
layer AFTER tool-call parsing and AFTER the reasoning/content split.
When ``final_content`` is empty/None AND no ``tool_calls`` fired AND
``reasoning_text`` is non-empty, surface ``reasoning_text`` as
``content``. ``reasoning_content`` stays populated unchanged
(duplication between fields is the lesser evil vs. silent drop).
"""

from __future__ import annotations

import pytest

from vllm_mlx.reasoning.gemma4_parser import Gemma4ReasoningParser
from vllm_mlx.service.helpers import (
    _finalize_content_and_reasoning,
    _rescue_silent_drop_from_reasoning,
)

# ── Unit tests for the rescue helper ─────────────────────────────────


def test_rescue_fires_when_content_none_no_tools_with_reasoning():
    """Exact #569 production failure: ``final_content`` empty/None,
    ``tool_calls`` empty/None, ``reasoning_text`` non-empty. The
    rescue surfaces reasoning as content so the assistant message is
    non-empty for OpenAI-compat clients.
    """
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="The user wants weather for SF. I should call get_weather",
        tool_calls=None,
    )
    assert rescued == ("The user wants weather for SF. I should call get_weather"), (
        "rescue must surface reasoning trace when content+tool_calls are both empty"
    )


def test_rescue_fires_when_content_empty_string():
    """Empty-string ``final_content`` (downstream sanitization
    collapsed everything to empty) is treated the same as ``None`` —
    the user-visible field is silently empty either way.
    """
    rescued = _rescue_silent_drop_from_reasoning(
        final_content="",
        reasoning_text="Some reasoning trace",
        tool_calls=None,
    )
    assert rescued == "Some reasoning trace"


def test_rescue_noop_when_content_present():
    """Happy path: ``final_content`` is non-empty. Rescue must NOT
    overwrite the model's actual answer with the reasoning trace.
    This is the regression guard against the rescue firing on the
    properly-terminated gemma-4 multi-turn flow (the success case the
    bug report says must NOT regress).
    """
    rescued = _rescue_silent_drop_from_reasoning(
        final_content="The answer is 42.",
        reasoning_text="I thought about it carefully...",
        tool_calls=None,
    )
    assert rescued == "The answer is 42.", (
        "rescue must not clobber legitimate content with reasoning"
    )


def test_rescue_noop_when_tool_calls_fired():
    """When the model emitted a tool call, ``content`` is
    legitimately ``None`` per the OpenAI spec (the tool call IS the
    response). Rescue must NOT fire — surfacing the pre-call
    reasoning as content would confuse the OpenAI client into
    interpreting it as a text reply alongside the tool call.
    """
    fake_tool_calls = [{"id": "call_x", "type": "function"}]
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="I'll call get_weather for SF",
        tool_calls=fake_tool_calls,
    )
    assert rescued is None, (
        "tool-call turns must keep content=None; rescue must not fire"
    )


def test_rescue_noop_when_reasoning_also_empty():
    """If the model truly produced nothing (no content, no
    reasoning, no tool call), there's nothing to rescue with.
    ``None`` propagates — we do NOT fabricate content. The upstream
    bug (model silent) is the operator's problem to debug, but the
    rescue layer doesn't make it worse.
    """
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text=None,
        tool_calls=None,
    )
    assert rescued is None


def test_rescue_noop_when_reasoning_empty_string():
    """Empty-string ``reasoning_text`` (the engine populated the
    field but the value is empty) is treated the same as ``None``:
    no rescue, no fabrication.
    """
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="",
        tool_calls=None,
    )
    assert rescued is None


def test_rescue_noop_when_tool_calls_empty_list():
    """``tool_calls`` is sometimes an empty list rather than
    ``None`` (parser returned no matches). Treat it as falsy — the
    rescue fires when content is empty AND there are no actual
    calls, regardless of whether the field is ``None`` or ``[]``.
    """
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="Some thought",
        tool_calls=[],
    )
    assert rescued == "Some thought"


# ── Integration: parser-level repro of the truncated thought ─────────


def test_gemma4_parser_returns_no_reasoning_on_unterminated_thought():
    """Pins the upstream surface: the Gemma 4 reasoning parser's
    ``extract_reasoning`` returns ``(None, raw_text_with_marker)``
    when the thought channel never closed. This is what feeds the
    silent-drop path — the parser bails, the route then drops content.

    The parser behavior itself is correct (it can't infer where a
    missing close marker SHOULD have been); the route-layer rescue
    is what protects clients from the consequence. This test pins
    the parser's behavior so future parser changes don't silently
    bypass the rescue's reason for existing.
    """
    p = Gemma4ReasoningParser()
    truncated = (
        "<|channel>thought\nLet me think. The user asked about weather. "
        "I should call get_weather with city=SF. But first let me check"
    )
    reasoning, content = p.extract_reasoning(truncated)
    assert reasoning is None, (
        "parser must not pretend it extracted reasoning from an "
        f"unterminated thought; got {reasoning!r}"
    )
    # Content is the raw text (markers preserved). The route's
    # ``clean_output_text`` + ``strip_thinking_tags`` will collapse
    # this; the rescue handles the resulting empty content.
    assert content is not None and "Let me think" in content


# ── Integration: full _finalize + rescue chain on engine-routed input ──


def test_finalize_plus_rescue_recovers_stuck_gemma4_thought():
    """End-to-end stitching of the helpers chain that the chat route
    executes. Simulates the engine path for gemma-4-26b-4bit stuck
    mid-thought: the token-level ``OutputRouter`` populated
    ``engine_reasoning_text`` (token-routed reasoning trace) and
    cleared ``cleaned_text``. ``_finalize_content_and_reasoning``
    short-circuits on the non-empty ``engine_reasoning_text`` and
    returns ``("", reasoning)``. ``_rescue_silent_drop_from_reasoning``
    then surfaces the reasoning as content.
    """
    engine_reasoning = (
        "Need to figure out what weather tool wants. The city is SF. "
        "Let me think about format"
    )
    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text="<|channel>thought\n" + engine_reasoning,
        cleaned_text="",
        tool_calls=[],
        reasoning_parser=Gemma4ReasoningParser(),
        engine_reasoning_text=engine_reasoning,
    )
    # First leg: engine-routed reasoning wins, cleaned_text stays empty.
    assert cleaned_text == ""
    assert reasoning_text == engine_reasoning

    # Simulate the route's final_content compute path: empty
    # cleaned_text -> final_content stays None (mirror chat.py:1252-1257).
    final_content = None
    if cleaned_text:  # pragma: no cover — path not taken in this scenario
        final_content = cleaned_text

    # Apply the rescue.
    rescued = _rescue_silent_drop_from_reasoning(
        final_content, reasoning_text, tool_calls=None
    )
    assert rescued == engine_reasoning, (
        "end-to-end rescue must surface the engine-routed reasoning trace "
        "as content when the model got stuck mid-thought"
    )


def test_finalize_plus_rescue_preserves_happy_path():
    """Happy path: model emits properly-terminated thought + content.
    Engine routes both → ``cleaned_text`` carries the final answer,
    ``reasoning_text`` carries the thought trace. The rescue must
    NOT fire — the user's content is the model's actual answer, not
    its thought trace. Pins the no-regress contract from issue #569.
    """
    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text="<|channel>thought\nLet me think.<channel|>"
        "<|channel>content\nThe answer is 42.<channel|>",
        cleaned_text="The answer is 42.",
        tool_calls=[],
        reasoning_parser=Gemma4ReasoningParser(),
        engine_reasoning_text="Let me think.",
    )
    assert cleaned_text == "The answer is 42."
    assert reasoning_text == "Let me think."

    rescued = _rescue_silent_drop_from_reasoning(
        cleaned_text, reasoning_text, tool_calls=None
    )
    assert rescued == "The answer is 42.", (
        "happy-path content must NOT be overwritten by the reasoning trace"
    )


def test_finalize_plus_rescue_preserves_tool_call_path():
    """Tool-call path: model emits reasoning then a tool call. The
    OpenAI spec says ``content`` is ``None`` for tool-call turns
    (the tool call IS the response). Rescue must NOT fire — surfacing
    the pre-call reasoning as content would make the client see both
    a text reply AND a tool call, which violates the contract.
    """
    fake_tool_calls = [
        {
            "id": "call_x",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
        }
    ]
    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text="<|channel>thought\nNeed weather.<channel|>"
        '<|tool_call>call:get_weather{city:<|"|>SF<|"|>}<tool_call|>',
        cleaned_text="",  # tool parser stripped to empty
        tool_calls=fake_tool_calls,
        reasoning_parser=Gemma4ReasoningParser(),
        engine_reasoning_text="Need weather.",
    )
    assert reasoning_text == "Need weather."

    rescued = _rescue_silent_drop_from_reasoning(
        None, reasoning_text, tool_calls=fake_tool_calls
    )
    assert rescued is None, "tool-call turns must keep content=None"


# ── Integration: ChatCompletion response assembly ────────────────────


@pytest.fixture
def fake_chat_finalize():
    """Wraps the chat route's final assembly so the test exercises
    the exact sequence the production handler runs. Returns
    ``(final_content, reasoning_content)`` — the two fields that
    matter for the silent-drop regression.
    """

    def _finalize(
        *,
        cleaned_text: str,
        reasoning_text: str | None,
        tool_calls: list | None,
    ):
        from vllm_mlx.api.utils import (
            clean_output_text,
            sanitize_output,
            strip_thinking_tags,
        )

        final_content = None
        if cleaned_text:
            final_content = strip_thinking_tags(clean_output_text(cleaned_text))
            final_content = sanitize_output(final_content)

        final_content = _rescue_silent_drop_from_reasoning(
            final_content, reasoning_text, tool_calls
        )
        return final_content, reasoning_text

    return _finalize


def test_assistant_message_non_empty_when_only_reasoning_fired(fake_chat_finalize):
    """Pins the issue #569 contract at the assembly layer: when the
    model produced only a reasoning trace, the AssistantMessage MUST
    have ``content`` populated (so Cline/Cursor/Codex CLI don't see
    an empty turn). ``reasoning_content`` stays populated unchanged.
    """
    content, reasoning = fake_chat_finalize(
        cleaned_text="",
        reasoning_text="The user wants weather. I should call get_weather",
        tool_calls=None,
    )
    assert content is not None and content != "", (
        "issue #569: assistant turn must not be silently empty"
    )
    assert "weather" in content
    assert reasoning == "The user wants weather. I should call get_weather"


def test_assistant_message_truly_empty_when_nothing_fired(fake_chat_finalize):
    """Defensive: when the model produced NOTHING at all (no content,
    no reasoning, no tool call), the rescue does NOT fabricate
    content. ``content`` stays ``None`` so the route emits the
    OpenAI-spec empty-message shape. The upstream "model produced
    nothing" bug is a separate concern; rescue doesn't paper over it.
    """
    content, reasoning = fake_chat_finalize(
        cleaned_text="",
        reasoning_text=None,
        tool_calls=None,
    )
    assert content is None
    assert reasoning is None


# ── Streaming surface: SSE terminal-chunk rescue ─────────────────────


class _ReasoningOnlyStreamEngine:
    """Streaming engine that yields ONLY reasoning-channel chunks.

    Reproduces the gemma-4 stuck-thought streaming shape: per-token
    output emits on the reasoning channel, the stream ends without
    ever transitioning to a content channel or tool call. The
    streaming chat route's terminal chunk should rescue the
    accumulated reasoning trace into ``delta.content`` so OpenAI-compat
    clients reading only the content stream see something.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, reasoning_deltas: list[str]):
        self._deltas = reasoning_deltas
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        accumulated_reasoning = ""
        for i, delta in enumerate(self._deltas):
            accumulated_reasoning += delta
            is_last = i == len(self._deltas) - 1
            yield GenerationOutput(
                text="",
                new_text=delta,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=is_last,
                finish_reason="stop" if is_last else None,
                channel="reasoning",
                reasoning_text=accumulated_reasoning,
            )


def _parse_sse(text: str) -> list[dict]:
    """Extract JSON payloads from an SSE response body."""
    import json

    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events


def test_streaming_rescue_surfaces_reasoning_as_terminal_content():
    """SSE streaming surface: when the model emits only reasoning
    deltas during the loop AND no tool call fires, the terminal
    chunk's ``delta.content`` MUST carry the accumulated reasoning
    trace. The per-delta ``reasoning_content`` chunks already went
    out during the loop; this extra ``content`` in the terminal
    chunk is the no-silent-drop guarantee for OpenAI-compat agentic
    clients (Cline, Cursor, Codex CLI) reading the content stream
    only.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    cfg.engine = _ReasoningOnlyStreamEngine(
        reasoning_deltas=[
            "Need to think about ",
            "the weather query. ",
            "I should call get_weather but first",
        ]
    )
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None  # engine reasoning_text is authoritative

    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": True,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "weather?"}],
            },
        )
        assert resp.status_code == 200, resp.text
        events = _parse_sse(resp.text)
        assert events, "expected at least the terminal SSE chunk"

        terminal_events = [
            e
            for e in events
            if any(ch.get("finish_reason") is not None for ch in e.get("choices", []))
        ]
        assert terminal_events, "expected an SSE chunk with finish_reason set"
        terminal = terminal_events[-1]
        delta = terminal["choices"][0].get("delta", {})

        # #569: terminal chunk surfaces accumulated reasoning as content
        # so the content stream is non-empty for clients that ignore
        # reasoning_content.
        terminal_content = delta.get("content")
        assert terminal_content, (
            f"#569 streaming: terminal chunk MUST carry rescued content; "
            f"got {terminal_content!r}"
        )
        # The rescued content is the accumulated reasoning trace.
        assert "weather query" in terminal_content
    finally:
        reset_config()


def test_streaming_rescue_noop_when_content_was_streamed():
    """Streaming happy path: when content was emitted during the
    loop, the rescue MUST NOT fire and duplicate the accumulated
    text into the terminal chunk on TOP of what the normal terminal
    delta carries (which can legitimately be the last content delta,
    merged in by the postprocessor when ``output.finished=True``).

    The post-fix invariant we pin: the union of content deltas across
    all SSE chunks equals the original streamed text — no duplication
    of the WHOLE accumulated content via rescue.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.routes.chat import router as chat_router

    class _NormalEngine:
        preserve_native_tool_format = False
        is_mllm = False
        supports_guided_generation = False
        tokenizer = None

        def build_prompt(self, messages, tools=None, enable_thinking=None):
            return "PROMPT"

        async def stream_chat(self, messages, **kwargs):
            for i, txt in enumerate(["Hello ", "world."]):
                is_last = i == 1
                yield GenerationOutput(
                    text=txt,
                    new_text=txt,
                    prompt_tokens=4,
                    completion_tokens=i + 1,
                    finished=is_last,
                    finish_reason="stop" if is_last else None,
                    channel=None,
                )

    cfg = reset_config()
    cfg.engine = _NormalEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True

    try:
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
        events = _parse_sse(resp.text)

        # Concatenate every content delta across the whole stream.
        # The streamed content should equal the original engine
        # output exactly — no rescue duplication, no missing bytes.
        streamed_content = ""
        for ev in events:
            for choice in ev.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    streamed_content += delta["content"]
        assert streamed_content == "Hello world.", (
            f"happy path content stream must equal engine output; "
            f"got {streamed_content!r}"
        )
    finally:
        reset_config()
