# SPDX-License-Identifier: Apache-2.0
"""
Tests for Anthropic-to-OpenAI adapter conversion functions.

Tests all conversion functions in vllm_mlx/api/anthropic_adapter.py.
These are pure logic tests with no MLX dependency.
"""

import json

from vllm_mlx.api.anthropic_adapter import (
    _convert_message,
    _convert_stop_reason,
    _convert_tool,
    _convert_tool_choice,
    anthropic_to_openai,
    openai_to_anthropic,
    to_anthropic_tool_use_id,
)


def assert_tool_use_id_shape(id_str: str) -> None:
    """Single source-of-truth assertion for the Anthropic-spec
    ``tool_use.id`` prefix (F9). Every test that asserts on a
    ``tool_use.id`` shape across the codebase should use this helper
    so the prefix contract has one place to update if Anthropic ever
    widens it. ``toolu_`` is verbatim from Anthropic's public
    examples (see https://docs.anthropic.com/en/api/messages).

    The tail contract is intentionally PERMISSIVE — we only assert
    the prefix + non-empty tail. The ``to_anthropic_tool_use_id``
    helper has its OWN idempotency branch that passes a pre-existing
    ``toolu_*`` id through unchanged (so an engine that natively
    minted a non-hex tail flows through), and the mint path uses
    ``secrets.token_hex`` (locked separately by
    ``test_to_anthropic_tool_use_id_mints_fresh_when_missing`` in
    this file). Asserting hex here would force callers to filter
    their upstream-supplied ids, which the F9 codex r1 NIT flagged
    as over-tightening.
    """
    assert isinstance(id_str, str), (
        f"tool_use.id must be a string (got {type(id_str).__name__})"
    )
    assert id_str.startswith("toolu_"), (
        f"tool_use.id must start with 'toolu_' (Anthropic spec); "
        f"got {id_str!r}. OpenAI-style 'call_<hex>' ids leak the "
        f"underlying adapter and break clients that regex on the "
        f"prefix. See ``to_anthropic_tool_use_id`` for the single "
        f"source of truth."
    )
    tail = id_str[len("toolu_") :]
    assert tail, f"tool_use.id tail must be non-empty (got {id_str!r})"


from vllm_mlx.api.anthropic_models import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicRequest,
    AnthropicToolDef,
)
from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    FunctionCall,
    ToolCall,
    Usage,
)


class TestConvertStopReason:
    """Tests for _convert_stop_reason."""

    def test_stop_to_end_turn(self):
        assert _convert_stop_reason("stop") == "end_turn"

    def test_tool_calls_to_tool_use(self):
        assert _convert_stop_reason("tool_calls") == "tool_use"

    def test_length_to_max_tokens(self):
        assert _convert_stop_reason("length") == "max_tokens"

    def test_content_filter_to_end_turn(self):
        assert _convert_stop_reason("content_filter") == "end_turn"

    def test_none_to_end_turn(self):
        assert _convert_stop_reason(None) == "end_turn"

    def test_unknown_to_end_turn(self):
        assert _convert_stop_reason("something_else") == "end_turn"


class TestConvertToolChoice:
    """Tests for _convert_tool_choice."""

    def test_auto(self):
        assert _convert_tool_choice({"type": "auto"}) == "auto"

    def test_any_to_required(self):
        assert _convert_tool_choice({"type": "any"}) == "required"

    def test_none_type(self):
        assert _convert_tool_choice({"type": "none"}) == "none"

    def test_specific_tool(self):
        result = _convert_tool_choice({"type": "tool", "name": "search"})
        assert result == {
            "type": "function",
            "function": {"name": "search"},
        }

    def test_missing_type_defaults_to_auto(self):
        assert _convert_tool_choice({}) == "auto"

    def test_unknown_type_defaults_to_auto(self):
        assert _convert_tool_choice({"type": "unknown"}) == "auto"


class TestConvertTool:
    """Tests for _convert_tool."""

    def test_minimal_tool(self):
        tool = AnthropicToolDef(name="search")
        result = _convert_tool(tool)
        assert result.type == "function"
        assert result.function["name"] == "search"
        assert result.function["description"] == ""
        assert result.function["parameters"] == {"type": "object", "properties": {}}

    def test_full_tool(self):
        tool = AnthropicToolDef(
            name="get_weather",
            description="Get weather for a city",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        result = _convert_tool(tool)
        assert result.function["name"] == "get_weather"
        assert result.function["description"] == "Get weather for a city"
        assert result.function["parameters"]["required"] == ["city"]


class TestConvertMessage:
    """Tests for _convert_message."""

    def test_simple_user_string(self):
        msg = AnthropicMessage(role="user", content="hello")
        result = _convert_message(msg)
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "hello"

    def test_simple_assistant_string(self):
        msg = AnthropicMessage(role="assistant", content="hi there")
        result = _convert_message(msg)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content == "hi there"

    def test_user_with_text_blocks(self):
        msg = AnthropicMessage(
            role="user",
            content=[
                AnthropicContentBlock(type="text", text="first"),
                AnthropicContentBlock(type="text", text="second"),
            ],
        )
        result = _convert_message(msg)
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "first\nsecond"

    def test_user_with_tool_results(self):
        msg = AnthropicMessage(
            role="user",
            content=[
                AnthropicContentBlock(
                    type="tool_result",
                    tool_use_id="call_1",
                    content="sunny, 22C",
                ),
                AnthropicContentBlock(
                    type="tool_result",
                    tool_use_id="call_2",
                    content="rainy, 15C",
                ),
            ],
        )
        result = _convert_message(msg)
        assert len(result) == 2
        assert result[0].role == "tool"
        assert result[0].content == "sunny, 22C"
        assert result[0].tool_call_id == "call_1"
        assert result[1].role == "tool"
        assert result[1].content == "rainy, 15C"

    def test_user_with_text_and_tool_results(self):
        msg = AnthropicMessage(
            role="user",
            content=[
                AnthropicContentBlock(type="text", text="here are results"),
                AnthropicContentBlock(
                    type="tool_result",
                    tool_use_id="call_1",
                    content="done",
                ),
            ],
        )
        result = _convert_message(msg)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[0].content == "here are results"
        assert result[1].role == "tool"

    def test_tool_result_with_list_content(self):
        msg = AnthropicMessage(
            role="user",
            content=[
                AnthropicContentBlock(
                    type="tool_result",
                    tool_use_id="call_1",
                    content=[
                        {"type": "text", "text": "line one"},
                        {"type": "text", "text": "line two"},
                    ],
                ),
            ],
        )
        result = _convert_message(msg)
        assert result[0].role == "tool"
        assert result[0].content == "line one\nline two"

    def test_tool_result_with_none_content(self):
        msg = AnthropicMessage(
            role="user",
            content=[
                AnthropicContentBlock(
                    type="tool_result",
                    tool_use_id="call_1",
                    content=None,
                ),
            ],
        )
        result = _convert_message(msg)
        assert result[0].content == ""

    def test_assistant_with_tool_use(self):
        msg = AnthropicMessage(
            role="assistant",
            content=[
                AnthropicContentBlock(type="text", text="Let me check."),
                AnthropicContentBlock(
                    type="tool_use",
                    id="call_abc",
                    name="search",
                    input={"q": "weather"},
                ),
            ],
        )
        result = _convert_message(msg)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content == "Let me check."
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0]["function"]["name"] == "search"
        args = json.loads(result[0].tool_calls[0]["function"]["arguments"])
        assert args == {"q": "weather"}

    def test_assistant_empty_content(self):
        msg = AnthropicMessage(
            role="assistant",
            content=[],
        )
        result = _convert_message(msg)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content == ""

    def test_user_empty_content(self):
        msg = AnthropicMessage(
            role="user",
            content=[],
        )
        result = _convert_message(msg)
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == ""


class TestAnthropicToOpenai:
    """Tests for anthropic_to_openai conversion."""

    def _make_request(self, **kwargs):
        defaults = {
            "model": "default",
            "messages": [AnthropicMessage(role="user", content="hi")],
            "max_tokens": 100,
        }
        defaults.update(kwargs)
        return AnthropicRequest(**defaults)

    def test_simple_request(self):
        req = self._make_request()
        result = anthropic_to_openai(req)
        assert result.model == "default"
        assert result.max_tokens == 100
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "hi"

    def test_system_string(self):
        req = self._make_request(system="Be helpful.")
        result = anthropic_to_openai(req)
        assert len(result.messages) == 2
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "Be helpful."
        assert result.messages[1].role == "user"

    def test_system_list(self):
        req = self._make_request(system=[{"type": "text", "text": "Be concise."}])
        result = anthropic_to_openai(req)
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "Be concise."

    def test_temperature_default_forwards_none(self):
        """Adapter MUST forward None so the server-side cascade fires.

        Hard-coding 0.7 here would short-circuit
        ``service.helpers._resolve_temperature`` at layer 1, robbing
        Anthropic-compat clients of alias / generation_config overlays.
        """
        req = self._make_request()
        result = anthropic_to_openai(req)
        assert result.temperature is None

    def test_temperature_explicit(self):
        req = self._make_request(temperature=0.3)
        result = anthropic_to_openai(req)
        assert result.temperature == 0.3

    def test_top_p_default_forwards_none(self):
        """Same contract as temperature — see above."""
        req = self._make_request()
        result = anthropic_to_openai(req)
        assert result.top_p is None

    def test_top_p_explicit(self):
        req = self._make_request(top_p=0.5)
        result = anthropic_to_openai(req)
        assert result.top_p == 0.5

    def test_top_k_forwarded(self):
        """AnthropicRequest exposes top_k; the adapter must forward it."""
        req = self._make_request(top_k=20)
        result = anthropic_to_openai(req)
        assert result.top_k == 20

    def test_top_k_default_forwards_none(self):
        req = self._make_request()
        result = anthropic_to_openai(req)
        assert result.top_k is None

    def test_stop_sequences(self):
        req = self._make_request(stop_sequences=["END", "STOP"])
        result = anthropic_to_openai(req)
        assert result.stop == ["END", "STOP"]

    def test_stream_flag(self):
        req = self._make_request(stream=True)
        result = anthropic_to_openai(req)
        assert result.stream is True

    def test_tools_conversion(self):
        req = self._make_request(
            tools=[
                AnthropicToolDef(
                    name="search",
                    description="Search the web",
                    input_schema={
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                )
            ]
        )
        result = anthropic_to_openai(req)
        assert len(result.tools) == 1
        assert result.tools[0].function["name"] == "search"

    def test_tool_choice_conversion(self):
        # Anthropic ``tool_choice: any`` only makes sense alongside a
        # non-empty ``tools`` array — pre-F-034 the test passed without
        # tools, but ``ChatCompletionRequest`` now (correctly) rejects
        # ``tool_choice="required"`` with no tools at the schema layer.
        # Anthropic's own spec rejects ``any`` without tools, so feeding
        # a tool here brings the unit fixture in line with the real wire
        # contract.
        req = self._make_request(
            tools=[
                AnthropicToolDef(
                    name="search",
                    description="Search",
                    input_schema={"type": "object"},
                )
            ],
            tool_choice={"type": "any"},
        )
        result = anthropic_to_openai(req)
        assert result.tool_choice == "required"

    def test_no_tools(self):
        req = self._make_request()
        result = anthropic_to_openai(req)
        assert result.tools is None
        assert result.tool_choice is None

    def test_multiple_messages(self):
        msgs = [
            AnthropicMessage(role="user", content="hello"),
            AnthropicMessage(role="assistant", content="hi"),
            AnthropicMessage(role="user", content="how are you"),
        ]
        req = self._make_request(messages=msgs)
        result = anthropic_to_openai(req)
        assert len(result.messages) == 3
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"
        assert result.messages[2].role == "user"


class TestOpenaiToAnthropic:
    """Tests for openai_to_anthropic conversion."""

    def _make_response(
        self,
        content="hello",
        finish_reason="stop",
        tool_calls=None,
        reasoning_content=None,
    ):
        msg = AssistantMessage(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )
        choice = ChatCompletionChoice(message=msg, finish_reason=finish_reason)
        return ChatCompletionResponse(
            model="default",
            choices=[choice],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    def test_simple_text_response(self):
        resp = self._make_response(content="hi there")
        result = openai_to_anthropic(resp, "default")
        assert result.model == "default"
        assert result.type == "message"
        assert result.role == "assistant"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "hi there"
        assert result.stop_reason == "end_turn"

    def test_usage_mapping(self):
        resp = self._make_response()
        result = openai_to_anthropic(resp, "default")
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_usage_mapping_no_cache_leaves_cache_fields_unset(self):
        """When the OpenAI side reports no prefix-cache hit (the
        ``prompt_tokens_details`` field is omitted), the Anthropic
        cache fields stay ``None`` so downstream tools can distinguish
        "engine doesn't report cache" from "engine reported a hit".
        ``input_tokens`` falls back to the full prompt count since
        nothing is attributed to cache.
        """
        resp = self._make_response()
        result = openai_to_anthropic(resp, "default")
        assert result.usage.input_tokens == 10
        assert result.usage.cache_read_input_tokens is None
        assert result.usage.cache_creation_input_tokens is None

    def test_usage_mapping_with_cache_hit_preserves_anthropic_identity(self):
        """Per Anthropic's prompt-caching docs the three input fields
        are mutually exclusive:
            total_input_tokens = input_tokens
                + cache_read_input_tokens
                + cache_creation_input_tokens
        So ``input_tokens`` is the *non-cached* share, not the whole
        prompt. We populate ``cache_read_input_tokens`` with the
        local engine's hit count and leave
        ``cache_creation_input_tokens`` unset — Anthropic's "creation"
        means tokens written between explicit ``cache_control``
        breakpoints (billed 1.25x), which has no analog on a local
        KV-cache engine.
        """
        from vllm_mlx.api.models import PromptTokensDetails

        msg = AssistantMessage(content="hi")
        choice = ChatCompletionChoice(message=msg, finish_reason="stop")
        resp = ChatCompletionResponse(
            model="default",
            choices=[choice],
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=30),
            ),
        )
        result = openai_to_anthropic(resp, "default")
        assert result.usage.input_tokens == 70
        assert result.usage.output_tokens == 20
        assert result.usage.cache_read_input_tokens == 30
        assert result.usage.cache_creation_input_tokens is None
        # Anthropic-spec identity holds: 70 + 30 + 0 == 100
        assert (
            result.usage.input_tokens
            + (result.usage.cache_read_input_tokens or 0)
            + (result.usage.cache_creation_input_tokens or 0)
            == 100
        )

    def test_usage_mapping_full_cache_hit(self):
        """An exact re-run that hits 100% of the prompt prefix
        produces ``input_tokens=0`` and ``cache_read_input_tokens``
        equal to the full prompt — every input token is attributed
        to cache.
        """
        from vllm_mlx.api.models import PromptTokensDetails

        msg = AssistantMessage(content="hi")
        choice = ChatCompletionChoice(message=msg, finish_reason="stop")
        resp = ChatCompletionResponse(
            model="default",
            choices=[choice],
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=100),
            ),
        )
        result = openai_to_anthropic(resp, "default")
        assert result.usage.input_tokens == 0
        assert result.usage.cache_read_input_tokens == 100
        assert result.usage.cache_creation_input_tokens is None

    def test_tool_calls_response(self):
        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(
                name="search",
                arguments='{"q": "test"}',
            ),
        )
        resp = self._make_response(
            content="Let me search.",
            finish_reason="tool_calls",
            tool_calls=[tc],
        )
        result = openai_to_anthropic(resp, "default")
        assert len(result.content) == 2
        assert result.content[0].type == "text"
        assert result.content[0].text == "Let me search."
        assert result.content[1].type == "tool_use"
        assert result.content[1].name == "search"
        assert result.content[1].input == {"q": "test"}
        assert result.stop_reason == "tool_use"

    def test_tool_call_invalid_json_arguments(self):
        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="search", arguments="not json"),
        )
        resp = self._make_response(
            content=None, finish_reason="tool_calls", tool_calls=[tc]
        )
        result = openai_to_anthropic(resp, "default")
        tool_block = [b for b in result.content if b.type == "tool_use"][0]
        assert tool_block.input == {}

    def test_empty_choices(self):
        resp = ChatCompletionResponse(
            model="default",
            choices=[],
            usage=Usage(),
        )
        result = openai_to_anthropic(resp, "default")
        assert result.stop_reason == "end_turn"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == ""

    def test_no_content_adds_empty_text(self):
        resp = self._make_response(content=None)
        result = openai_to_anthropic(resp, "default")
        assert len(result.content) >= 1
        has_text = any(b.type == "text" for b in result.content)
        assert has_text

    def test_stop_reason_length(self):
        resp = self._make_response(finish_reason="length")
        result = openai_to_anthropic(resp, "default")
        assert result.stop_reason == "max_tokens"

    # H-03 — matched_stop surface
    def test_matched_stop_promotes_end_turn_to_stop_sequence(self):
        """When the engine surfaces a matched stop string the adapter
        must rewrite ``stop_reason="stop"`` (which would otherwise
        map to ``end_turn``) to Anthropic's dedicated
        ``stop_sequence`` value and populate the ``stop_sequence``
        field verbatim. Pre-fix the adapter always emitted
        ``end_turn`` with ``stop_sequence: None`` even when a stop
        fired."""
        resp = self._make_response(content="say ", finish_reason="stop")
        result = openai_to_anthropic(resp, "default", matched_stop="END")
        assert result.stop_reason == "stop_sequence"
        assert result.stop_sequence == "END"

    def test_matched_stop_none_keeps_end_turn(self):
        """No matched stop means EOS / length / no-stop; the legacy
        ``stop → end_turn`` mapping holds and ``stop_sequence`` stays
        ``None``."""
        resp = self._make_response(finish_reason="stop")
        result = openai_to_anthropic(resp, "default", matched_stop=None)
        assert result.stop_reason == "end_turn"
        assert result.stop_sequence is None

    def test_matched_stop_does_not_override_max_tokens(self):
        """A length cap MUST still win — Anthropic's stop_reason
        values are mutually exclusive, and the matched-stop rewrite
        only applies to the ``end_turn`` bucket. Otherwise a request
        that ran past max_tokens AND happened to also include the
        stop bytes mid-output would be mis-classified."""
        resp = self._make_response(finish_reason="length")
        result = openai_to_anthropic(resp, "default", matched_stop="END")
        assert result.stop_reason == "max_tokens"
        # ``stop_sequence`` MUST stay None — Anthropic spec invariant:
        # "stop_sequence is set iff stop_reason == 'stop_sequence'".
        assert result.stop_sequence is None

    def test_matched_stop_does_not_override_tool_use(self):
        """Same invariant for the tool_use bucket: a tool_calls
        finish must keep stop_reason='tool_use' even if a stop
        string happened to surface earlier in the response."""
        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="search", arguments='{"q":"x"}'),
        )
        resp = self._make_response(
            content="Calling.",
            finish_reason="tool_calls",
            tool_calls=[tc],
        )
        result = openai_to_anthropic(resp, "default", matched_stop="END")
        assert result.stop_reason == "tool_use"
        assert result.stop_sequence is None

    def test_response_has_id(self):
        resp = self._make_response()
        result = openai_to_anthropic(resp, "test-model")
        assert result.id.startswith("msg_")
        assert result.model == "test-model"

    def test_reasoning_content_becomes_thinking_block(self):
        """#413 fix: reasoning_content on the OpenAI response must appear
        as a ``thinking`` content block on the Anthropic response,
        placed BEFORE the text block to match Anthropic's
        extended-thinking SDK convention."""
        resp = self._make_response(
            content="Final answer.",
            reasoning_content="Let me think.",
        )
        result = openai_to_anthropic(resp, "default")
        assert len(result.content) == 2
        assert result.content[0].type == "thinking"
        assert result.content[0].thinking == "Let me think."
        assert result.content[1].type == "text"
        assert result.content[1].text == "Final answer."

    def test_reasoning_content_with_tool_calls(self):
        """Thinking block + text + tool_use coexist, in that order."""
        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="search", arguments='{"q":"x"}'),
        )
        resp = self._make_response(
            content="Calling search.",
            reasoning_content="I need to look this up.",
            finish_reason="tool_calls",
            tool_calls=[tc],
        )
        result = openai_to_anthropic(resp, "default")
        assert [b.type for b in result.content] == ["thinking", "text", "tool_use"]
        assert result.content[0].thinking == "I need to look this up."

    def test_no_reasoning_content_omits_thinking_block(self):
        """Absence of reasoning_content must NOT produce a thinking
        block (avoids leaking empty placeholders into clients that
        check ``content_block[0].type``)."""
        resp = self._make_response(content="hi", reasoning_content=None)
        result = openai_to_anthropic(resp, "default")
        assert all(b.type != "thinking" for b in result.content)

    def test_reasoning_disabled_omits_thinking_block(self):
        """Issue #702: when the served alias has ``reasoning_parser:
        null`` in ``aliases.json``, the route passes
        ``reasoning_enabled=False`` to the adapter. The adapter must
        suppress the ``thinking`` block regardless of what
        ``reasoning_content`` carries — Anthropic's public API never
        emits one for non-extended-thinking models, so emitting it
        would break client capability detection.
        """
        resp = self._make_response(
            content="hello world",
            reasoning_content="this trace must not leak",
        )
        result = openai_to_anthropic(resp, "default", reasoning_enabled=False)
        # No thinking block at all.
        assert all(b.type != "thinking" for b in result.content)
        # Text block survives.
        text_blocks = [b for b in result.content if b.type == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "hello world"

    def test_reasoning_equals_content_suppresses_thinking_block(self):
        """Issue #702: ``_rescue_silent_drop_from_reasoning`` (#569)
        deliberately promotes a stuck reasoning trace into ``content``
        so the OpenAI-side message isn't silently empty. The Anthropic
        adapter has no other way to know
        ``reasoning_content == content`` is a rescue artifact, so it
        would dutifully emit BOTH blocks carrying the same string.
        Claude Code / claude-cli / langchain-anthropic then render the
        same paragraph twice. Suppress the ``thinking`` block in that
        case and surface only ``text``.
        """
        duplicated = "I think this is the answer."
        resp = self._make_response(
            content=duplicated,
            reasoning_content=duplicated,
        )
        # ``reasoning_enabled=True`` matches a thinking-capable alias —
        # the dedup guard must fire even on those, because the rescue
        # path runs for thinking-capable aliases too.
        result = openai_to_anthropic(resp, "default", reasoning_enabled=True)
        assert all(b.type != "thinking" for b in result.content)
        text_blocks = [b for b in result.content if b.type == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == duplicated

    def test_reasoning_enabled_distinct_reasoning_still_emits_thinking(self):
        """Regression guard for #702: a genuinely thinking-capable
        alias that produced a real thought trace (distinct from the
        answer) must still get its ``thinking`` block.
        """
        resp = self._make_response(
            content="Final answer.",
            reasoning_content="Let me think.",
        )
        result = openai_to_anthropic(resp, "default", reasoning_enabled=True)
        assert len(result.content) == 2
        assert result.content[0].type == "thinking"
        assert result.content[0].thinking == "Let me think."
        assert result.content[1].type == "text"
        assert result.content[1].text == "Final answer."

    def test_reasoning_enabled_default_preserves_legacy_behavior(self):
        """``reasoning_enabled`` defaults to True so external callers
        that don't pass the kwarg (older tests, third-party imports)
        keep their existing pre-#702 behavior. Only the explicit
        equality dedup gate fires.
        """
        # Distinct reasoning + content → thinking block still appears
        # without passing the kwarg.
        resp = self._make_response(
            content="Final answer.",
            reasoning_content="Let me think.",
        )
        result = openai_to_anthropic(resp, "default")  # kwarg omitted
        assert any(b.type == "thinking" for b in result.content)
        assert any(b.type == "text" for b in result.content)

    def test_whitespace_only_reasoning_omits_thinking_block(self):
        """Codex r1 NIT on #702: whitespace-only ``reasoning_content``
        (``"   \\n"``) must NOT open a leading ``thinking`` block
        — Claude Code would render it as a blank thought. Mirrors the
        ``_rescue_silent_drop_from_reasoning`` whitespace guard so the
        two paths agree on what "semantically empty reasoning" means.
        """
        resp = self._make_response(
            content="real answer",
            reasoning_content="   \n  \t  ",
        )
        result = openai_to_anthropic(resp, "default", reasoning_enabled=True)
        assert all(b.type != "thinking" for b in result.content)
        text_blocks = [b for b in result.content if b.type == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "real answer"


# ===========================================================================
# F9 — Anthropic-spec ``toolu_`` prefix on every ``tool_use.id``.
# ===========================================================================


class TestAnthropicToolUseIdPrefix:
    """F9 — every ``tool_use.id`` emitted on the ``/v1/messages``
    surface must use Anthropic's ``toolu_<hex>`` prefix. Pre-fix the
    adapter passed OpenAI-style ``call_<hex>`` through verbatim,
    leaking the underlying conversion (Sergei evidence: id="call_fdc47973").

    The fix is at the adapter (single source of truth, not at every
    tool_parser emit site). Cross-route audit: ``/v1/chat/completions``
    and ``/v1/responses`` keep ``call_<hex>`` (OpenAI parity); only
    ``/v1/messages`` rewrites.
    """

    def test_to_anthropic_tool_use_id_rewrites_call_prefix(self):
        """The helper preserves the hex tail so call-id correlation
        across logs still works after the rewrite."""
        assert to_anthropic_tool_use_id("call_abcd1234") == "toolu_abcd1234"

    def test_to_anthropic_tool_use_id_passes_through_toolu_prefix(self):
        """An id that already carries ``toolu_`` survives unchanged —
        idempotency lets the helper be applied at multiple boundaries
        without double-rewriting.
        """
        assert to_anthropic_tool_use_id("toolu_already_set") == "toolu_already_set"

    def test_to_anthropic_tool_use_id_mints_fresh_when_missing(self):
        """Anthropic's public examples carry ~24 hex chars after the
        prefix; mint a fresh id when the caller hands us nothing
        usable so the wire response never carries an empty id."""
        out = to_anthropic_tool_use_id(None)
        assert_tool_use_id_shape(out)
        assert len(out[len("toolu_") :]) == 24

    def test_to_anthropic_tool_use_id_mints_fresh_for_unknown_prefix(self):
        """A non-OpenAI-shaped id (e.g. ``id="x_123"``) gets a fresh
        ``toolu_<hex>`` mint — we never echo an unknown prefix back
        on the Anthropic surface."""
        out = to_anthropic_tool_use_id("call_unknown_prefix_!!!")
        # Still has the ``call_`` prefix, so we DO rewrite — the
        # contract is "if it looks like call_<X>, return toolu_<X>".
        assert out.startswith("toolu_")
        # An id with a foreign prefix gets a fresh mint.
        foreign = to_anthropic_tool_use_id("x_abc")
        assert_tool_use_id_shape(foreign)

    def test_tool_use_id_uses_toolu_prefix_at_adapter(self):
        """Integration: ``openai_to_anthropic`` rewrites every
        ``tool_use.id`` to ``toolu_<hex>`` (single source of truth,
        F9). Locks the adapter-layer fix against regressions where
        a future change to ``openai_to_anthropic`` would leak the
        OpenAI prefix back to clients.
        """
        tc = ToolCall(
            id="call_fdc47973",
            type="function",
            function=FunctionCall(name="search", arguments='{"q": "x"}'),
        )
        choice = ChatCompletionChoice(
            message=AssistantMessage(content=None, tool_calls=[tc]),
            finish_reason="tool_calls",
        )
        resp = ChatCompletionResponse(model="default", choices=[choice], usage=Usage())
        result = openai_to_anthropic(resp, "default")
        tool_blocks = [b for b in result.content if b.type == "tool_use"]
        assert len(tool_blocks) == 1
        assert_tool_use_id_shape(tool_blocks[0].id)
        # Hex tail preserved across the rewrite — operator-side
        # correlation works.
        assert tool_blocks[0].id == "toolu_fdc47973"

    def test_tool_use_id_already_toolu_passes_through(self):
        """When upstream already mints ``toolu_<hex>`` (e.g. a future
        engine that natively produces Anthropic IDs), don't rewrite.
        """
        tc = ToolCall(
            id="toolu_already_set",
            type="function",
            function=FunctionCall(name="search", arguments="{}"),
        )
        choice = ChatCompletionChoice(
            message=AssistantMessage(content=None, tool_calls=[tc]),
            finish_reason="tool_calls",
        )
        resp = ChatCompletionResponse(model="default", choices=[choice], usage=Usage())
        result = openai_to_anthropic(resp, "default")
        tool_blocks = [b for b in result.content if b.type == "tool_use"]
        assert tool_blocks[0].id == "toolu_already_set"


# ===========================================================================
# F7 — ``tool_choice={"type":"none"}`` strips tools at the adapter so
# the chat template never injects tool definitions into the prompt.
# ===========================================================================


class TestToolChoiceNoneStripsTools:
    """F7 — when the client sets ``tool_choice={"type":"none"}``, the
    adapter MUST drop tools from the ``ChatCompletionRequest`` it
    constructs so the downstream chat template doesn't inject tool
    definitions into the prompt. Without this the model still sees
    the tool in the system prompt, decides to call it, and emits a
    raw ``<tool_call>{...}`` marker that leaks into the response
    text block (Sergei repro F7).

    Single source of truth: applied at the adapter, not at the
    parser / output layer. The fix is parser-independent.
    """

    def _make_request(self, **kwargs):
        defaults = {
            "model": "test",
            "messages": [AnthropicMessage(role="user", content="hi")],
            "max_tokens": 100,
        }
        defaults.update(kwargs)
        return AnthropicRequest(**defaults)

    def test_tool_choice_none_drops_tools(self):
        req = self._make_request(
            tools=[
                AnthropicToolDef(
                    name="get_weather",
                    description="weather",
                    input_schema={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                )
            ],
            tool_choice={"type": "none"},
        )
        result = anthropic_to_openai(req)
        assert result.tools is None, (
            f"tool_choice=none must strip tools; got {result.tools!r}"
        )
        # The tool_choice field is still forwarded so downstream
        # consumers branching on it can see the original signal.
        assert result.tool_choice == "none"

    def test_tool_choice_auto_keeps_tools(self):
        """Negative control: ``auto`` leaves tools intact."""
        req = self._make_request(
            tools=[
                AnthropicToolDef(
                    name="get_weather",
                    input_schema={"type": "object"},
                )
            ],
            tool_choice={"type": "auto"},
        )
        result = anthropic_to_openai(req)
        assert result.tools is not None
        assert len(result.tools) == 1

    def test_tool_choice_any_keeps_tools(self):
        """Negative control: ``any`` (Anthropic) → ``required``
        (OpenAI) leaves tools intact."""
        req = self._make_request(
            tools=[
                AnthropicToolDef(
                    name="get_weather",
                    input_schema={"type": "object"},
                )
            ],
            tool_choice={"type": "any"},
        )
        result = anthropic_to_openai(req)
        assert result.tools is not None
        assert result.tool_choice == "required"

    def test_tool_choice_specific_tool_keeps_tools(self):
        """Negative control: pinning a specific tool keeps the tool
        list intact (the downstream route filter handles the
        "pinned tool not in tools" 400 case)."""
        req = self._make_request(
            tools=[
                AnthropicToolDef(
                    name="get_weather",
                    input_schema={"type": "object"},
                )
            ],
            tool_choice={"type": "tool", "name": "get_weather"},
        )
        result = anthropic_to_openai(req)
        assert result.tools is not None


# ===========================================================================
# F6 — wire response excludes None-valued fields (no ``null`` leakage).
# ===========================================================================


class TestAnthropicResponseExcludesNullFields:
    """F6 — strict TypeScript clients / JSON-schema validators trip on
    ``null``-valued fields the spec says ``field?: T | undefined``.
    Anthropic's real API omits inapplicable fields entirely; we mirror
    that by serializing via ``model_dump_json(exclude_none=True)`` in
    the route.

    These tests pin the response model's shape so a future field
    addition (declared ``Optional`` and defaulting to ``None``) can't
    silently re-introduce the wire-level null leak Sergei's F6 repro
    flagged. Route-level serialization is also tested below.
    """

    def test_basic_response_serialization_excludes_none(self):
        """A response with only ``input_tokens`` + ``output_tokens`` in
        usage and a single text block must serialize to JSON with no
        ``null`` values anywhere."""
        msg = AssistantMessage(content="hi")
        choice = ChatCompletionChoice(message=msg, finish_reason="stop")
        resp = ChatCompletionResponse(
            model="default",
            choices=[choice],
            usage=Usage(prompt_tokens=5, completion_tokens=2),
        )
        result = openai_to_anthropic(resp, "default")
        body = result.model_dump(exclude_none=True)
        # No null top-level keys.
        for key, val in body.items():
            assert val is not None, (
                f"top-level key {key!r} serialized as None; "
                "exclude_none should have stripped it"
            )
        # No null usage fields.
        for key, val in body.get("usage", {}).items():
            assert val is not None, f"usage key {key!r} serialized as None"
        # No null content-block fields.
        for blk in body.get("content", []):
            for key, val in blk.items():
                assert val is not None, f"content block key {key!r} serialized as None"

    def test_wire_json_has_no_literal_null(self):
        """Belt-and-braces: dump as JSON and grep for the literal
        ``": null`` substring. A client running ``r.model_dump(
        exclude_unset=True)`` should see ONLY the fields the server
        actually populated."""
        msg = AssistantMessage(content="hi")
        choice = ChatCompletionChoice(message=msg, finish_reason="stop")
        resp = ChatCompletionResponse(
            model="default",
            choices=[choice],
            usage=Usage(prompt_tokens=5, completion_tokens=2),
        )
        result = openai_to_anthropic(resp, "default")
        wire_json = result.model_dump_json(exclude_none=True)
        assert ": null" not in wire_json and ":null" not in wire_json, (
            f"wire JSON leaked a literal null: {wire_json[:500]}"
        )

    def test_tool_use_response_has_no_null_fields(self):
        """Same invariant on a response that includes a ``tool_use``
        block — locks the F9 rewrite path against introducing
        spurious null fields on the way through."""
        tc = ToolCall(
            id="call_abc12345",
            type="function",
            function=FunctionCall(name="search", arguments='{"q": "x"}'),
        )
        choice = ChatCompletionChoice(
            message=AssistantMessage(content="Searching...", tool_calls=[tc]),
            finish_reason="tool_calls",
        )
        resp = ChatCompletionResponse(model="default", choices=[choice], usage=Usage())
        result = openai_to_anthropic(resp, "default")
        wire_json = result.model_dump_json(exclude_none=True)
        assert ": null" not in wire_json and ":null" not in wire_json, (
            f"tool_use wire JSON leaked a null: {wire_json[:500]}"
        )
