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


def test_rescue_noop_when_reasoning_is_whitespace_only():
    """Codex round-1 NIT on #676: whitespace-only ``reasoning_text``
    (``"   \\n\\t  "``) must NOT pass the rescue predicate.
    Pre-fix the helper only checked truthiness, so a non-empty
    whitespace string was promoted to ``content`` and clients saw a
    technically-non-null but semantically-empty assistant turn.
    Post-fix the predicate uses ``.strip()`` so semantically-empty
    whitespace is treated identically to ``None`` / ``""`` — no
    rescue, no fabrication; the existing empty-response path fires.
    """
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="   \n\t  ",
        tool_calls=None,
    )
    assert rescued is None, (
        f"whitespace-only reasoning must not promote to content; got {rescued!r}"
    )


def test_rescue_noop_when_final_content_is_whitespace_only():
    """Codex round-3 NIT on #676: whitespace-only ``final_content``
    (``"   \\n"``) is semantically empty to OpenAI-compat clients —
    the assistant turn shows as blank even though the field is a
    non-empty string. Pre-fix the helper's early-exit predicate was
    a plain truthiness check (``if final_content: return …``), so a
    whitespace-only string blocked the rescue and the client saw an
    empty assistant turn despite the engine having produced a
    legitimate reasoning trace.

    Post-fix the predicate strips before returning: whitespace-only
    ``final_content`` is treated identically to ``None`` / ``""``,
    so the rescue FIRES and the reasoning trace surfaces as content.
    Pins the contract: any ``final_content`` that ``.strip()``s to
    empty must NOT block the rescue when reasoning is present.
    """
    rescued = _rescue_silent_drop_from_reasoning(
        final_content="   \n",
        reasoning_text="Real reasoning trace that should be rescued",
        tool_calls=None,
    )
    assert rescued == "Real reasoning trace that should be rescued", (
        "whitespace-only final_content must be treated as semantically "
        f"absent so rescue can fire; got {rescued!r}"
    )


def test_rescue_preserves_leading_trailing_whitespace_in_rescued_content():
    """The whitespace-only gate must use ``.strip()`` on the
    predicate only — the assigned content stays untouched so a
    legitimately-padded thought trace (``"  Thought.  "``) keeps its
    original framing. The predicate sees non-empty content via
    ``.strip()``; the returned value is the raw original.
    """
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="  Real thought.  ",
        tool_calls=None,
    )
    assert rescued == "  Real thought.  ", (
        "predicate must use strip() but the returned value must be "
        f"the original (un-stripped) reasoning text; got {rescued!r}"
    )


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


def test_gemma4_parser_routes_unterminated_thought_to_reasoning():
    """D-STOP-THINK (cross-cycle, cycle-6 F-CORR-2): the Gemma 4
    parser now detects the unterminated ``<|channel>thought`` opener
    and routes the body to ``reasoning`` instead of leaking the
    entire thought trace into ``content`` with the literal channel
    marker preserved.

    Pre-fix, ``extract_reasoning`` returned
    ``(None, "<|channel>thought\\n<full trace>")`` on the no-closer
    branch and the route's ``clean_output_text`` / ``strip_thinking_tags``
    pipeline did not fully strip the trace. The downstream
    silent-drop rescue masked the user-facing leak by promoting
    reasoning_text → content, but ``content`` AND the reasoning
    channel still ended up holding overlapping bytes on stuck-mid-
    thought outputs.

    Post-fix: reasoning carries the trace; content holds only the
    pre-opener prefix (typically empty). The silent-drop rescue
    still fires when content is empty AND reasoning is populated —
    its predicate is unchanged. Cross-cycle invariant for the
    D-STOP-THINK bundle is honoured (no byte duplication into
    user-visible content channel).
    """
    p = Gemma4ReasoningParser()
    truncated = (
        "<|channel>thought\nLet me think. The user asked about weather. "
        "I should call get_weather with city=SF. But first let me check"
    )
    reasoning, content = p.extract_reasoning(truncated)
    assert reasoning is not None and "Let me think" in reasoning, (
        f"unterminated thought must route to reasoning; got reasoning={reasoning!r}"
    )
    # Content holds only the (empty) pre-opener prefix — the trace
    # bytes are no longer leaked.
    assert not content or "Let me think" not in content, (
        f"thought trace bytes must not leak into content; got content={content!r}"
    )


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


def test_streaming_rescue_noop_when_reasoning_is_whitespace_only():
    """Codex round-3 BLOCKING on #676: the streaming rescue used to
    promote ``processor.accumulated_reasoning`` directly into
    ``delta.content`` on the terminal chunk, bypassing the helper's
    whitespace guard. A reasoning-only stream of ``"   \\n"`` would
    then emit a semantically empty ``delta.content`` while the
    non-streaming path correctly suppressed it via the helper.

    Post-fix the streaming rescue routes through
    ``_rescue_silent_drop_from_reasoning`` so both paths share the
    same predicate (whitespace + content + tool-call absence). A
    whitespace-only reasoning stream therefore yields NO
    ``delta.content`` on any chunk — same shape the non-streaming
    counterpart pins at the unit level. Mirrors
    ``test_rescue_noop_when_reasoning_is_whitespace_only`` for the
    SSE surface so the streaming/non-streaming parity is locked in.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    # Reasoning deltas that concatenate to whitespace-only — the
    # exact stuck-thought edge case where the helper's guard must
    # suppress the streaming rescue identically to non-streaming.
    cfg.engine = _ReasoningOnlyStreamEngine(
        reasoning_deltas=["   ", "\n", "\t  "],
    )
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None

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
        assert events, "expected at least one SSE chunk"

        # Codex round-4 BLOCKING on #676: collect EVERY non-null
        # ``delta.content`` value without stripping, so an
        # incorrectly-emitted whitespace-only payload (``"   \n\t  "``)
        # is caught instead of being silently coerced to empty by
        # ``.strip()``. Pre-fix this assertion was
        # ``not streamed_content.strip()``, which would still pass
        # under the exact regression we're guarding against — a
        # self-defeating test. The new shape lists every raw
        # ``delta.content`` value emitted on any chunk and asserts
        # the list is empty; any non-null emission at all
        # (whitespace, empty string, real text) is a regression.
        #
        # F-040: ``content: null`` is now an INTENTIONAL part of the
        # streaming shape on reasoning-only deltas (so clients reading
        # ``delta.content`` on the terminal chunk don't crash with a
        # KeyError). The original test caught any presence of the
        # ``content`` key including ``None``; we tighten the check to
        # only flag actual *promotions* (a non-null value), which is
        # the regression we care about here.
        content_values = [
            (choice.get("delta") or {}).get("content")
            for ev in events
            for choice in ev.get("choices", [])
            if "content" in (choice.get("delta") or {})
            and (choice.get("delta") or {}).get("content") is not None
        ]
        assert content_values == [], (
            "#676 round-4 BLOCKING (streaming): whitespace-only "
            "accumulated reasoning must NOT promote to delta.content "
            "on any SSE chunk; expected zero non-null content emissions "
            f"but got {content_values!r}"
        )
    finally:
        reset_config()


# ── response_format gate: codex round-1 BLOCKING on #676 ─────────────


class _ReasoningOnlyChatEngine:
    """Non-streaming engine that returns ONLY reasoning text.

    Reproduces the gemma-4 stuck-thought non-streaming shape: the
    token-level router populated ``reasoning_text`` but emitted no
    final/content channel and no tool call. The chat route's normal
    ``content`` extraction yields empty/None; the rescue would
    normally surface the reasoning trace as ``content`` — but a
    structured-output (``response_format`` = ``json_object`` /
    ``json_schema``) request MUST keep the rescue suppressed because
    reasoning prose is almost never valid JSON and would break the
    OpenAI-compat structured-output contract.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, reasoning_text: str):
        self._reasoning_text = reasoning_text
        self.chat_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text="",
            new_text="",
            prompt_tokens=4,
            completion_tokens=8,
            finished=True,
            finish_reason="stop",
            channel="reasoning",
            reasoning_text=self._reasoning_text,
        )


def _run_chat_route_with_response_format(response_format: dict) -> dict:
    """Helper: drive the chat route via TestClient with a
    reasoning-only engine and the given ``response_format``. Returns
    the parsed JSON response body so the test can assert on
    ``content`` / ``reasoning_content``.
    """
    import json as _json

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    cfg.engine = _ReasoningOnlyChatEngine(
        reasoning_text="Need to think about the JSON shape the user wants",
    )
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None

    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": False,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "give me JSON"}],
                "response_format": response_format,
            },
        )
        assert resp.status_code == 200, resp.text
        return _json.loads(resp.text)
    finally:
        reset_config()


def test_rescue_skipped_when_response_format_is_json_object():
    """Codex round-1 BLOCKING on #676: when a client requests
    ``response_format={"type": "json_object"}``, the route MUST NOT
    rescue reasoning text into ``content``. Reasoning prose is
    almost never valid JSON, so surfacing it would break the
    OpenAI-compat structured-output contract (clients expect either
    validated JSON or the existing empty/error path so they can
    retry, not surprise prose).

    Post-fix invariant: ``content`` is ``None`` (or empty), the
    reasoning trace stays in ``reasoning_content`` so it isn't
    lost, and the structured-output client sees the unchanged
    empty path.
    """
    body = _run_chat_route_with_response_format({"type": "json_object"})

    choice = body["choices"][0]
    msg = choice["message"]
    # The rescue must NOT fire — content stays None / absent.
    assert not msg.get("content"), (
        "#676 BLOCKING: rescue must be suppressed for "
        f"response_format=json_object; got content={msg.get('content')!r}"
    )
    # Reasoning is still surfaced via reasoning_content so the
    # trace isn't lost — operator can still debug, client gets the
    # existing structured-output empty/error path.
    assert "think" in (msg.get("reasoning_content") or "").lower()


def test_rescue_skipped_when_response_format_is_json_schema():
    """Same #676 BLOCKING contract for ``json_schema``: structured
    output requests MUST NOT have reasoning prose surfaced as
    ``content``. Validated JSON or the existing empty/error path —
    never surprise prose that the client will then fail to parse
    against the requested schema.
    """
    body = _run_chat_route_with_response_format(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        }
    )

    choice = body["choices"][0]
    msg = choice["message"]
    assert not msg.get("content"), (
        "#676 BLOCKING: rescue must be suppressed for "
        f"response_format=json_schema; got content={msg.get('content')!r}"
    )
    assert "think" in (msg.get("reasoning_content") or "").lower()


# ── streaming response_format gate: codex round-2 BLOCKING on #676 ──


def _run_streaming_chat_route_with_response_format(response_format: dict) -> list[dict]:
    """Helper: drive the streaming chat route via TestClient with a
    reasoning-only streaming engine and the given ``response_format``.
    Returns the parsed list of SSE event dicts so tests can assert
    that the terminal chunk does NOT carry rescued ``delta.content``
    (because structured output was requested).

    Mirrors ``_run_chat_route_with_response_format`` but drives
    ``stream=True`` — pins the streaming counterpart of the same
    contract. Factored so the ``json_object`` and ``json_schema``
    cases share the entire harness; only the request body differs.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    cfg.engine = _ReasoningOnlyStreamEngine(
        reasoning_deltas=[
            "Need to think about ",
            "the JSON shape ",
            "the user wants",
        ]
    )
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None

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
                "messages": [{"role": "user", "content": "give me JSON"}],
                "response_format": response_format,
            },
        )
        assert resp.status_code == 200, resp.text
        return _parse_sse(resp.text)
    finally:
        reset_config()


def test_streaming_rescue_skipped_when_response_format_is_json_object():
    """Codex round-2 BLOCKING on #676: the SSE/streaming rescue path
    must apply the same ``response_format`` gate as the non-streaming
    path. Pre-fix, ``stream=true`` requests with
    ``response_format={"type": "json_object"}`` would still get the
    reasoning trace surfaced in ``delta.content`` on the terminal
    chunk, breaking the OpenAI-compat structured-output contract for
    streaming clients exactly as the non-streaming path used to
    before round 1.

    Post-fix invariant: NO SSE chunk carries reasoning prose in
    ``delta.content``. Per-delta ``delta.reasoning_content`` chunks
    still go out during the loop (debuggability), and the terminal
    chunk's ``delta.content`` is empty / absent so structured-output
    clients see the existing empty path and can retry — never
    surprise prose.
    """
    events = _run_streaming_chat_route_with_response_format({"type": "json_object"})
    assert events, "expected at least one SSE chunk"

    # Aggregate every delta.content across the whole stream — none of
    # them should carry the reasoning trace. Per-delta reasoning
    # chunks SHOULD still flow on the reasoning_content channel so
    # operators can debug.
    streamed_content = ""
    streamed_reasoning = ""
    for ev in events:
        for choice in ev.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                streamed_content += delta["content"]
            if delta.get("reasoning_content"):
                streamed_reasoning += delta["reasoning_content"]

    assert not streamed_content, (
        "#676 BLOCKING (streaming): no SSE chunk may surface reasoning "
        "as delta.content when response_format=json_object; "
        f"got streamed_content={streamed_content!r}"
    )
    # Reasoning trace must still be preserved on the reasoning channel.
    assert "json shape" in streamed_reasoning.lower(), (
        "reasoning_content must still flow during the loop for "
        f"debuggability; got reasoning={streamed_reasoning!r}"
    )


def test_streaming_rescue_skipped_when_response_format_is_json_schema():
    """Same #676 BLOCKING contract for ``json_schema`` on the
    streaming path: structured output requests MUST NOT have
    reasoning prose surfaced as ``delta.content`` on the terminal
    chunk. Validated JSON or the existing empty path — never
    surprise prose the client will fail to parse against the
    requested schema.
    """
    events = _run_streaming_chat_route_with_response_format(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        }
    )
    assert events, "expected at least one SSE chunk"

    streamed_content = ""
    streamed_reasoning = ""
    for ev in events:
        for choice in ev.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                streamed_content += delta["content"]
            if delta.get("reasoning_content"):
                streamed_reasoning += delta["reasoning_content"]

    assert not streamed_content, (
        "#676 BLOCKING (streaming): no SSE chunk may surface reasoning "
        "as delta.content when response_format=json_schema; "
        f"got streamed_content={streamed_content!r}"
    )
    assert "json shape" in streamed_reasoning.lower()


# ── 2026-06-17 VibeThinker truncated-``<think>`` rescue gate ──────────


def test_rescue_skipped_when_truncated_think_with_finish_length():
    """2026-06-17 VibeThinker live-test repro: when ``finish_reason="length"``
    AND ``raw_text`` opens with an unclosed ``<think>``, the rescue must
    NOT fire. The reasoning trace is an in-progress thought, not a
    final answer — surfacing it as ``content`` would feed clients the
    SAME bytes as ``reasoning_content`` and break the "content is the
    final answer" contract.

    Live-test signature: content_len == reasoning_len (modulo the
    ``<think>`` opener), byte-identical. After fix: content stays
    ``None`` so clients can detect "model ran out of budget" via
    ``finish_reason="length"``.
    """
    raw = "<think>The user wants me to compute 17 * 23. Step 1: 17 * 20 = 340"
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="The user wants me to compute 17 * 23. Step 1: 17 * 20 = 340",
        tool_calls=None,
        finish_reason="length",
        raw_text=raw,
    )
    assert rescued is None, (
        "rescue must NOT fire on truncated-<think> + finish_reason=length; "
        f"got rescued={rescued!r}"
    )


def test_rescue_skipped_when_truncated_think_with_leading_whitespace():
    """The leading-position check uses ``lstrip().startswith`` so the
    gate still fires when ``raw_text`` opens with whitespace before
    ``<think>`` (the chat-template may emit a leading newline)."""
    raw = "\n  <think>thinking truncated"
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="thinking truncated",
        tool_calls=None,
        finish_reason="length",
        raw_text=raw,
    )
    assert rescued is None


def test_rescue_still_fires_on_length_when_raw_text_lacks_open_think():
    """Counter-test for the new gate: a ``finish_reason="length"``
    response WITHOUT an unclosed ``<think>`` opener (e.g. gemma-4
    stuck inside ``<|channel>thought\\n…`` — the original #569
    failure mode) must STILL rescue. The new gate only blocks the
    rescue when the model's raw text explicitly carries an unclosed
    ``<think>`` opener."""
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="reasoning that got truncated",
        tool_calls=None,
        finish_reason="length",
        raw_text="reasoning that got truncated",  # no <think> opener
    )
    assert rescued == "reasoning that got truncated", (
        "rescue MUST fire for #569 gemma-4 failure mode even when "
        "finish_reason=length, when raw_text does not start with <think>"
    )


def test_rescue_skipped_on_truncated_think_when_finish_is_stop():
    """D-STOP-THINK (cross-cycle bundle, cycle-3 / cycle-7 / cycle-11):
    when ``stop`` matches inside the unterminated ``<think>`` block,
    the trim path in the engine cuts the suffix before the closing
    ``</think>`` arrives. The pre-fix rescue gate only skipped this
    arm under ``finish_reason="length"``; ``finish_reason="stop"``
    fell through and the rescue surfaced the in-progress thought as
    ``content`` — yielding byte-identical ``content`` AND
    ``reasoning_content`` on the live wire (qwen3-0.6b-4bit live
    repro with ``stop=["STOP"]``).

    Post-fix the gate is symmetric across ``stop`` and ``length`` for
    the unclosed-``<think>`` arm. This counter-test pins the new
    behaviour against future regressions.

    The pre-fix rationale (``model voluntarily ended without
    producing a final answer``) is preserved on a different gate
    surface: callers that want to rescue genuinely orphaned
    reasoning traces can do so via ``finish_reason=None`` (legacy
    code path, see ``test_rescue_still_fires_on_truncated_think_when_finish_unknown``)
    — the conservative default still fires there.
    """
    raw = "<think>just a thought"
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="just a thought",
        tool_calls=None,
        finish_reason="stop",
        raw_text=raw,
    )
    assert rescued is None, (
        f"D-STOP-THINK regression — rescue surfaced trace as content "
        f"under stop-mid-think: rescued={rescued!r}"
    )


def test_rescue_still_fires_on_truncated_think_when_finish_unknown():
    """Counter-test: when ``finish_reason`` is ``None`` (legacy caller
    that doesn't thread the kwarg), the gate is conservative and
    rescue still fires. The kwarg defaults to ``None`` for back-
    compat with existing callers."""
    raw = "<think>just a thought"
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="just a thought",
        tool_calls=None,
    )
    assert rescued == "just a thought"


def test_rescue_skipped_when_closed_think_block_truncated_after():
    """Edge case: ``raw_text`` contains a CLOSED ``<think>...</think>``
    block plus partial answer truncated at length. The gate uses
    ``"</think>" not in raw_text`` so the closed block bypasses the
    rescue-skip and the rescue fires normally (the rescue's normal
    predicates still gate it appropriately)."""
    raw = "<think>complete thought</think>The ans"
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="complete thought",
        tool_calls=None,
        finish_reason="length",
        raw_text=raw,
    )
    # </think> is in raw_text, so the gate does NOT fire → rescue runs
    # normally and surfaces the reasoning.
    assert rescued == "complete thought"


# Codex r3 P1 — streaming rescue must skip truncated `<think>`.


def test_streaming_rescue_gate_skips_synthetic_truncated_think():
    """Codex r3 P1: when streaming, the parser consumes ``<think>``
    as a state transition so ``accumulated_reasoning`` doesn't carry
    the literal opener. The route synthesises a ``raw_text`` of
    ``"<think>" + accumulated_reasoning`` when the parser's
    ``_saw_any_tag`` flag indicates an unclosed opener was seen, then
    feeds that to the rescue. The rescue's existing
    ``finish=length + raw_text.lstrip().startswith("<think>") +
    "</think>" not in raw_text`` gate then suppresses the rescue
    uniformly with the non-streaming path.

    Pin the gate semantics: synthetic ``"<think>" + trace`` with
    ``finish="length"`` MUST suppress the rescue.
    """
    trace = "the model is in the middle of thinking..."
    synthetic_raw = "<think>" + trace
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text=trace,
        tool_calls=None,
        finish_reason="length",
        raw_text=synthetic_raw,
    )
    assert rescued is None, (
        "streaming truncated-<think> path must suppress rescue — got "
        f"rescued={rescued!r}"
    )


def test_streaming_rescue_still_fires_for_gemma4_stuck_thought_shape():
    """Counter-test: gemma-4 stuck-thought streaming shape — the
    original #569 failure — has no ``<think>`` opener in the
    accumulated reasoning. The route DOES NOT synthesise a
    ``<think>`` prefix for it (``_saw_any_tag`` is False on a
    non-``<think>`` parser like gemma4), so the rescue's gate stays
    OFF and the rescue still fires. Pin that path against drift."""
    trace = "the model is stuck inside the analysis channel"
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text=trace,
        tool_calls=None,
        finish_reason="length",
        # No ``<think>`` prefix — the route did not synthesise one
        # because ``_saw_any_tag`` was False on the (non-think) parser.
        raw_text=trace,
    )
    assert rescued == trace, (
        f"gemma-4 #569 failure mode must still rescue — got rescued={rescued!r}"
    )


# Bug C / Bug 3 cross-cut: PR #715 bundle live-test repro for
# ``reasoning_is_case4``.


def test_rescue_skipped_when_reasoning_is_case4_and_length():
    """PR #715 bundle, fuzz finding C / VibeThinker live-test repro:
    when the parser's Case-4 fallback fired (no tags in raw_text,
    ``enable_thinking=True``, parser routed the whole output to
    reasoning, helper blanked cleaned_text=""), the rescue MUST NOT
    surface that reasoning as content.

    Without this gate, the route mistakes the post-Case-4 empty
    content for a #569 silent drop and feeds the client byte-
    identical content + reasoning_content (live-test repro: model
    answered "What is 2+2?" with plain prose, no ``<think>`` token
    emitted at all, raw_text did NOT start with ``<think>`` — so the
    original narrow opener gate missed it).
    """
    trace = (
        'First, the user asked "What is 2+2?" This seems like a '
        "simple arithmetic question. Let me think..."
    )
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text=trace,
        tool_calls=None,
        finish_reason="length",
        # No ``<think>`` literal — model emitted plain prose. Without
        # the case4 gate the rescue would surface the trace.
        raw_text=trace,
        reasoning_is_case4=True,
    )
    assert rescued is None, (
        f"Case-4 length-truncated reasoning must NOT be surfaced as "
        f"content — got rescued={rescued!r}"
    )


def test_rescue_fires_when_case4_and_finish_is_stop_natural_eos():
    """D-STOP-THINK codex round-2 BLOCKING fix: scope the Case-4 +
    ``stop`` suppression to require explicit truncation evidence
    (``matched_stop`` set or unclosed-``<think>`` opener in raw_text).

    Natural-EOS Case-4 (``matched_stop=None``, no opener) is the
    VibeThinker / PR #715 fuzz finding C path: the model finishes
    its casual non-thinking answer with the model's own EOS token
    AND the parser routes the whole body to reasoning (Case-4
    blanking). The pre-fix rescue surfaced the reasoning trace as
    ``content`` — correct for #569. Suppressing on Case-4 alone
    would silently drop the assistant turn.
    """
    trace = "casual non-thinking answer that finished cleanly at EOS"
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text=trace,
        tool_calls=None,
        finish_reason="stop",
        # No ``<think>`` literal in raw_text — natural-EOS shape.
        raw_text=trace,
        reasoning_is_case4=True,
        matched_stop=None,  # natural-EOS — no user-supplied stop fired
    )
    assert rescued == trace, (
        f"#569 regression — rescue suppressed natural-EOS case-4 "
        f"reasoning trace under finish=stop with no opener evidence: "
        f"rescued={rescued!r}"
    )


def test_rescue_skipped_when_case4_stop_user_stop_matched_AND_prompt_thinking_active():
    """D-STOP-THINK codex round-5 BLOCKING fix: when the chat template
    pre-injects ``<think>\\n`` AND ``enable_thinking`` is non-False,
    a user-supplied stop firing mid-thought is the D-STOP-THINK leak
    shape. Require BOTH ``matched_stop`` AND ``prompt_thinking_active``
    — matched_stop alone is ambiguous (a casual stop-terminated
    answer ALSO has it set).
    """
    trace = "thought in progress when STOP fired"
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text=trace,
        tool_calls=None,
        finish_reason="stop",
        # No literal ``<think>`` in raw_text — prompt-injected shape.
        raw_text=trace,
        reasoning_is_case4=True,
        matched_stop="STOP",  # user-supplied stop fired mid-thought
        prompt_thinking_active=True,  # chat template injected <think>
    )
    assert rescued is None, (
        f"D-STOP-THINK regression — rescue surfaced prompt-injected "
        f"truncated thought as content: rescued={rescued!r}"
    )


def test_rescue_fires_when_case4_stop_matched_stop_but_thinking_not_active():
    """Codex round-5 BLOCKING counter-test: casual stop-terminated
    answer (matched_stop set, prompt_thinking_active=False) must
    still rescue per #569 — the OpenAI client must see the answer
    on ``message.content``.

    Example: ``"The answer is STOP"`` under ``stop=["STOP"]`` with
    a non-thinking model. The engine sets matched_stop and the
    parser may route the whole thing to reasoning (Case-4), but
    suppressing the rescue here would silently drop the answer.
    """
    answer = "The answer is 12"
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text=answer,
        tool_calls=None,
        finish_reason="stop",
        raw_text=answer,
        reasoning_is_case4=True,
        matched_stop="STOP",  # user-supplied stop fired
        prompt_thinking_active=False,  # chat template did NOT inject <think>
    )
    assert rescued == answer, (
        f"#569 regression — casual stop-terminated answer suppressed: "
        f"rescued={rescued!r}"
    )


def test_rescue_skipped_when_case4_stop_and_unclosed_think_opener():
    """D-STOP-THINK leak shape suppressed when raw_text carries the
    unclosed ``<think>`` opener — the opener-evidence arm picks this
    up symmetric with the prompt-injected shape above.
    """
    trace = "thought in progress when STOP fired"
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text=trace,
        tool_calls=None,
        finish_reason="stop",
        # Unclosed ``<think>`` opener — D-STOP-THINK explicit shape.
        raw_text="<think>\n" + trace,
        reasoning_is_case4=True,
    )
    assert rescued is None, (
        f"D-STOP-THINK regression — rescue surfaced unclosed-think "
        f"trace as content under finish=stop: rescued={rescued!r}"
    )


def test_rescue_fires_when_length_but_not_case4():
    """Counter-test: ``finish_reason="length"`` without the Case-4
    signal AND without the ``<think>`` opener still rescues — the
    original #569 gemma-4 stuck-thought failure shape. The case4
    gate is additive, not replacing, the narrow opener gate."""
    trace = "gemma-4 analysis channel content"
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text=trace,
        tool_calls=None,
        finish_reason="length",
        raw_text=trace,
        reasoning_is_case4=False,
    )
    assert rescued == trace
