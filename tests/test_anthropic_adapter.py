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
)
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
        req = self._make_request(tool_choice={"type": "any"})
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
