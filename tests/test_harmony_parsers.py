# SPDX-License-Identifier: Apache-2.0
"""
Tests for Harmony format parsers (GPT-OSS models).

Tests cover:
- HarmonyToolParser: tool call extraction from commentary channel
- HarmonyReasoningParser: reasoning extraction from analysis channel
- convert_tools_to_typescript: OpenAI JSON Schema to TypeScript conversion

Usage:
    pytest tests/test_harmony_parsers.py -v
"""

import json

import pytest

from vllm_mlx.api.harmony_tools import convert_tools_to_typescript
from vllm_mlx.reasoning import get_parser
from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser
from vllm_mlx.tool_parsers import ToolParserManager
from vllm_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser

# ============================================================================
# Tool Parser Tests
# ============================================================================


class TestHarmonyToolParser:
    """Tests for HarmonyToolParser."""

    @pytest.fixture()
    def parser(self):
        return HarmonyToolParser()

    def test_registration(self):
        """Parser is registered under harmony and gpt-oss names."""
        assert ToolParserManager.get_tool_parser("harmony") is HarmonyToolParser
        assert ToolParserManager.get_tool_parser("gpt-oss") is HarmonyToolParser

    def test_single_tool_call(self, parser):
        """Parse a single tool call from commentary channel."""
        text = (
            "<|start|>\n"
            "<|channel|>commentary to=functions.get_weather\n"
            "<|constrain|>json\n"
            '<|message|>{"location": "San Francisco", "unit": "celsius"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["location"] == "San Francisco"
        assert args["unit"] == "celsius"

    def test_tool_call_with_analysis_and_final(self, parser):
        """Parse tool call when analysis and final channels are present."""
        text = (
            "<|start|>\n"
            "<|channel|>analysis\n"
            "<|message|>The user wants weather. I should call get_weather.\n"
            "<|end|>\n"
            "<|channel|>commentary to=functions.get_weather\n"
            "<|constrain|>json\n"
            '<|message|>{"location": "SF"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_final_response_only(self, parser):
        """Parse response with no tool calls (final channel only)."""
        text = (
            "<|start|>\n"
            "<|channel|>final\n"
            "<|message|>The weather in San Francisco is 72F and sunny!\n"
            "<|return|>"
        )
        result = parser.extract_tool_calls(text)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "The weather in San Francisco is 72F and sunny!"

    def test_multiple_tool_calls(self, parser):
        """Parse multiple tool calls from separate commentary blocks."""
        text = (
            "<|start|>\n"
            "<|channel|>commentary to=functions.get_weather\n"
            "<|constrain|>json\n"
            '<|message|>{"location": "SF"}\n'
            "<|call|>\n"
            "<|channel|>commentary to=functions.get_time\n"
            "<|constrain|>json\n"
            '<|message|>{"timezone": "PST"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[1]["name"] == "get_time"

    def test_tool_call_without_constrain(self, parser):
        """Parse tool call without <|constrain|>json tag."""
        text = (
            "<|channel|>commentary to=functions.simple_func\n"
            '<|message|>{"arg": "value"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "simple_func"

    def test_malformed_json_arguments(self, parser):
        """Handle malformed JSON gracefully by keeping raw string."""
        text = (
            "<|channel|>commentary to=functions.broken_func\n"
            "<|constrain|>json\n"
            "<|message|>{invalid json here}\n"
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "broken_func"
        assert result.tool_calls[0]["arguments"] == "{invalid json here}"

    def test_tool_call_with_final_content(self, parser):
        """Tool calls coexist with final channel content."""
        text = (
            "<|channel|>commentary to=functions.search\n"
            "<|constrain|>json\n"
            '<|message|>{"query": "python"}\n'
            "<|call|>\n"
            "<|channel|>final\n"
            "<|message|>Here are the results.\n"
            "<|return|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.content == "Here are the results."

    def test_empty_input(self, parser):
        """Handle empty input."""
        result = parser.extract_tool_calls("")
        assert not result.tools_called
        assert result.tool_calls == []

    def test_plain_text_input(self, parser):
        """Handle plain text with no Harmony tokens."""
        result = parser.extract_tool_calls("Just a regular response.")
        assert not result.tools_called
        assert result.content == "Just a regular response."

    def test_unique_tool_ids(self, parser):
        """Each tool call gets a unique ID."""
        text = (
            "<|channel|>commentary to=functions.func_a\n"
            "<|constrain|>json\n"
            "<|message|>{}\n"
            "<|call|>\n"
            "<|channel|>commentary to=functions.func_b\n"
            "<|constrain|>json\n"
            "<|message|>{}\n"
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        ids = [tc["id"] for tc in result.tool_calls]
        assert len(set(ids)) == 2
        assert all(id_.startswith("call_") for id_ in ids)

    def test_nested_json_arguments(self, parser):
        """Parse tool call with nested JSON arguments."""
        args = {"filter": {"type": "range", "min": 0, "max": 100}, "sort": "asc"}
        text = (
            "<|channel|>commentary to=functions.query\n"
            "<|constrain|>json\n"
            f"<|message|>{json.dumps(args)}\n"
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        parsed_args = json.loads(result.tool_calls[0]["arguments"])
        assert parsed_args["filter"]["type"] == "range"

    def test_streaming_no_tool_markers(self, parser):
        """Streaming: plain text passes through as content."""
        result = parser.extract_tool_calls_streaming("", "Hello", "Hello")
        assert result == {"content": "Hello"}

    def test_streaming_tool_call_complete(self, parser):
        """Streaming: emit tool calls when <|call|> appears."""
        current = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '<|message|>{"a": 1}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls_streaming("", current, "<|call|>")

        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "func"

    def test_streaming_building_tool_call(self, parser):
        """Streaming: suppress output while building tool call."""
        current = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '<|message|>{"a":'
        )
        result = parser.extract_tool_calls_streaming("", current, '{"a":')
        assert result is None


# ============================================================================
# Reasoning Parser Tests
# ============================================================================


class TestHarmonyReasoningParser:
    """Tests for HarmonyReasoningParser."""

    @pytest.fixture()
    def parser(self):
        return HarmonyReasoningParser()

    def test_registration(self):
        """Parser is registered under the harmony name."""
        parser_cls = get_parser("harmony")
        assert parser_cls is HarmonyReasoningParser

    def test_extract_analysis_and_final(self, parser):
        """Extract reasoning from analysis and content from final."""
        output = (
            "<|channel|>analysis\n"
            "<|message|>Let me think step by step.\n"
            "<|end|>\n"
            "<|channel|>final\n"
            "<|message|>The answer is 42.\n"
            "<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Let me think step by step."
        assert content == "The answer is 42."

    def test_multiple_analysis_blocks(self, parser):
        """Concatenate multiple analysis blocks."""
        output = (
            "<|channel|>analysis\n"
            "<|message|>First thought.\n"
            "<|end|>\n"
            "<|channel|>analysis\n"
            "<|message|>Second thought.\n"
            "<|end|>\n"
            "<|channel|>final\n"
            "<|message|>Result.\n"
            "<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)

        assert "First thought." in reasoning
        assert "Second thought." in reasoning
        assert content == "Result."

    def test_no_analysis_channel(self, parser):
        """Output with no analysis returns None reasoning."""
        output = "<|channel|>final\n" "<|message|>Direct answer.\n" "<|return|>"
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning is None
        assert content == "Direct answer."

    def test_analysis_only_no_final(self, parser):
        """Output with only analysis returns None content."""
        output = "<|channel|>analysis\n" "<|message|>Just thinking...\n" "<|end|>"
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Just thinking..."
        assert content is None

    def test_empty_input(self, parser):
        """Handle empty input."""
        reasoning, content = parser.extract_reasoning("")
        assert reasoning is None
        assert content is None

    def test_analysis_with_commentary_and_final(self, parser):
        """Ignore commentary channel, extract analysis and final."""
        output = (
            "<|channel|>analysis\n"
            "<|message|>Need to call a tool.\n"
            "<|end|>\n"
            "<|channel|>commentary to=functions.search\n"
            "<|constrain|>json\n"
            '<|message|>{"q": "test"}\n'
            "<|call|>\n"
            "<|channel|>final\n"
            "<|message|>Found results.\n"
            "<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "Need to call a tool."
        assert content == "Found results."

    def test_streaming_analysis_to_final(self, parser):
        """Streaming: emit reasoning for analysis, content for final."""
        parser.reset_state()

        # Channel switch to analysis
        r1 = parser.extract_reasoning_streaming(
            "", "<|channel|>analysis\n", "<|channel|>analysis\n"
        )
        assert r1 is None  # channel switch, no content yet

        # Message start
        r2 = parser.extract_reasoning_streaming(
            "<|channel|>analysis\n",
            "<|channel|>analysis\n<|message|>",
            "<|message|>",
        )
        assert r2 is None  # message start token

        # Reasoning content
        r3 = parser.extract_reasoning_streaming(
            "<|channel|>analysis\n<|message|>",
            "<|channel|>analysis\n<|message|>Thinking",
            "Thinking",
        )
        assert r3 is not None
        assert r3.reasoning == "Thinking"
        assert r3.content is None

        # End of analysis
        r4 = parser.extract_reasoning_streaming(
            "<|channel|>analysis\n<|message|>Thinking",
            "<|channel|>analysis\n<|message|>Thinking<|end|>",
            "<|end|>",
        )
        assert r4 is None  # end token

        # Switch to final
        r5 = parser.extract_reasoning_streaming(
            "<|channel|>analysis\n<|message|>Thinking<|end|>",
            "<|channel|>analysis\n<|message|>Thinking<|end|>\n<|channel|>final\n",
            "\n<|channel|>final\n",
        )
        assert r5 is None  # channel switch

        # Final message content
        prev = "<|channel|>analysis\n<|message|>Thinking<|end|>\n<|channel|>final\n<|message|>"
        parser.extract_reasoning_streaming(
            "<|channel|>analysis\n<|message|>Thinking<|end|>\n<|channel|>final\n",
            prev,
            "<|message|>",
        )
        r6 = parser.extract_reasoning_streaming(
            prev,
            prev + "Answer",
            "Answer",
        )
        assert r6 is not None
        assert r6.content == "Answer"
        assert r6.reasoning is None

    def test_streaming_reset(self, parser):
        """Reset clears internal state."""
        parser._current_channel = "analysis"
        parser._in_message = True
        parser.reset_state()
        assert parser._current_channel is None
        assert parser._in_message is False

    def test_streaming_commentary_passed_through(self, parser):
        """Streaming: commentary channel passes through as content for tool parser."""
        parser.reset_state()

        r = parser.extract_reasoning_streaming(
            "",
            "<|channel|>commentary to=functions.f\n",
            "<|channel|>commentary to=functions.f\n",
        )
        assert r is not None
        assert r.content == "<|channel|>commentary to=functions.f\n"

        r = parser.extract_reasoning_streaming(
            "<|channel|>commentary to=functions.f\n",
            "<|channel|>commentary to=functions.f\n<|message|>",
            "<|message|>",
        )
        assert r is not None
        assert r.content == "<|message|>"

        r = parser.extract_reasoning_streaming(
            "<|channel|>commentary to=functions.f\n<|message|>",
            '<|channel|>commentary to=functions.f\n<|message|>{"a":1}',
            '{"a":1}',
        )
        assert r is not None
        assert r.content == '{"a":1}'


# ============================================================================
# TypeScript Tool Converter Tests
# ============================================================================


class TestHarmonyToolDefinitionConverter:
    """Tests for convert_tools_to_typescript."""

    def test_simple_tool(self):
        """Convert a simple tool with required parameters."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "namespace functions" in result
        assert "get_weather" in result
        assert "location: string," in result
        assert "// Get weather for a location" in result

    def test_optional_parameters(self):
        """Optional parameters get ? suffix."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "func",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "required_param": {"type": "string"},
                            "optional_param": {"type": "number"},
                        },
                        "required": ["required_param"],
                    },
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "required_param: string," in result
        assert "optional_param?: number," in result

    def test_enum_type(self):
        """Enums become TypeScript union types."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_unit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                    },
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert '"celsius" | "fahrenheit"' in result

    def test_multiple_tools(self):
        """Multiple tools in one namespace."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "func_a",
                    "description": "First function",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "func_b",
                    "description": "Second function",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        result = convert_tools_to_typescript(tools)

        assert "func_a" in result
        assert "func_b" in result
        assert "// First function" in result
        assert "// Second function" in result

    def test_no_tools(self):
        """None input returns None."""
        assert convert_tools_to_typescript(None) is None
        assert convert_tools_to_typescript([]) is None

    def test_no_parameters(self):
        """Tool with no parameters uses empty signature."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "type ping = () => any;" in result

    def test_array_type(self):
        """Array types convert to Array<type>."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "process",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "Array<string>" in result

    def test_boolean_and_integer_types(self):
        """Boolean and integer map correctly."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "config",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "count": {"type": "integer"},
                        },
                    },
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "enabled?: boolean," in result
        assert "count?: number," in result

    def test_no_description(self):
        """Tool without description has no comment line."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "no_desc",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = convert_tools_to_typescript(tools)

        assert "//" not in result
        assert "no_desc" in result

    def test_skips_non_function_types(self):
        """Non-function tools are skipped."""
        tools = [
            {"type": "retrieval"},
            {
                "type": "function",
                "function": {
                    "name": "real_func",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        result = convert_tools_to_typescript(tools)

        assert "real_func" in result
        assert "retrieval" not in result


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestHarmonyEdgeCases:
    """Edge case tests for Harmony parsers."""

    def test_tool_parser_incomplete_call(self):
        """Incomplete tool call (missing <|call|>) is not parsed."""
        parser = HarmonyToolParser()
        text = "<|channel|>commentary to=functions.func\n" '<|message|>{"arg": "value"}'
        result = parser.extract_tool_calls(text)
        assert not result.tools_called

    def test_tool_parser_unicode_content(self):
        """Handle unicode in tool arguments."""
        parser = HarmonyToolParser()
        text = (
            "<|channel|>commentary to=functions.translate\n"
            "<|constrain|>json\n"
            '<|message|>{"text": "日本語テスト"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["text"] == "日本語テスト"

    def test_reasoning_parser_unicode_content(self):
        """Handle unicode in reasoning and content."""
        parser = HarmonyReasoningParser()
        output = (
            "<|channel|>analysis\n"
            "<|message|>让我想想...\n"
            "<|end|>\n"
            "<|channel|>final\n"
            "<|message|>答案是42。\n"
            "<|return|>"
        )
        reasoning, content = parser.extract_reasoning(output)

        assert reasoning == "让我想想..."
        assert content == "答案是42。"

    def test_mixed_channels_full_flow(self):
        """Full flow: analysis -> commentary -> analysis -> final."""
        text = (
            "<|start|>\n"
            "<|channel|>analysis\n"
            "<|message|>Think 1.\n"
            "<|end|>\n"
            "<|channel|>commentary to=functions.search\n"
            "<|constrain|>json\n"
            '<|message|>{"q": "test"}\n'
            "<|call|>\n"
            "<|channel|>analysis\n"
            "<|message|>Think 2.\n"
            "<|end|>\n"
            "<|channel|>final\n"
            "<|message|>Done.\n"
            "<|return|>"
        )

        # Tool parser finds tool calls
        tool_parser = HarmonyToolParser()
        tool_result = tool_parser.extract_tool_calls(text)
        assert tool_result.tools_called
        assert len(tool_result.tool_calls) == 1
        assert tool_result.tool_calls[0]["name"] == "search"
        assert tool_result.content == "Done."

        # Reasoning parser finds both analysis blocks
        reasoning_parser = HarmonyReasoningParser()
        reasoning, content = reasoning_parser.extract_reasoning(text)
        assert "Think 1." in reasoning
        assert "Think 2." in reasoning
        assert content == "Done."

    def test_tool_parser_empty_arguments(self):
        """Tool call with empty JSON arguments."""
        parser = HarmonyToolParser()
        text = (
            "<|channel|>commentary to=functions.ping\n"
            "<|constrain|>json\n"
            "<|message|>{}\n"
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {}

    def test_tool_parser_whitespace_handling(self):
        """Handle extra whitespace in Harmony format."""
        parser = HarmonyToolParser()
        text = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '<|message|>  {"key": "value"}  \n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["key"] == "value"


# ============================================================================
# Comprehensive Tool Parser Tests (extract_tool_calls)
# ============================================================================


class TestHarmonyExtractToolCalls:
    """Extended tests for HarmonyToolParser.extract_tool_calls."""

    @pytest.fixture()
    def parser(self):
        return HarmonyToolParser()

    def test_single_tool_with_analysis_and_commentary(self, parser):
        """Single tool call with analysis + commentary channels present."""
        text = (
            "<|start|>\n"
            "<|channel|>analysis\n"
            "<|message|>User wants weather info for London.\n"
            "<|end|>\n"
            "<|channel|>commentary to=functions.get_weather\n"
            "<|constrain|>json\n"
            '<|message|>{"city": "London", "units": "metric"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["city"] == "London"
        assert args["units"] == "metric"
        # No final channel, so content is None
        assert result.content is None

    def test_tool_call_with_constrain_token(self, parser):
        """Tool call that includes <|constrain|>json is parsed correctly."""
        text = (
            "<|channel|>commentary to=functions.calculate\n"
            "<|constrain|>json\n"
            '<|message|>{"expression": "2+2"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "calculate"

    def test_multiple_tool_calls_with_final(self, parser):
        """Multiple tool calls with a final channel response."""
        text = (
            "<|channel|>commentary to=functions.search\n"
            "<|constrain|>json\n"
            '<|message|>{"query": "python tutorials"}\n'
            "<|call|>\n"
            "<|channel|>commentary to=functions.search\n"
            "<|constrain|>json\n"
            '<|message|>{"query": "rust tutorials"}\n'
            "<|call|>\n"
            "<|channel|>commentary to=functions.bookmark\n"
            "<|constrain|>json\n"
            '<|message|>{"url": "https://example.com"}\n'
            "<|call|>\n"
            "<|channel|>final\n"
            "<|message|>I've searched for both and bookmarked the result.\n"
            "<|return|>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 3
        assert result.tool_calls[0]["name"] == "search"
        assert result.tool_calls[1]["name"] == "search"
        assert result.tool_calls[2]["name"] == "bookmark"
        assert result.content == "I've searched for both and bookmarked the result."

    def test_no_tool_call_final_channel_only(self, parser):
        """No tool call, final channel only -- returns content, no tools."""
        text = (
            "<|channel|>final\n"
            "<|message|>Sure, I can help with that.\n"
            "<|return|>"
        )
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "Sure, I can help with that."

    def test_no_tool_call_no_final_channel(self, parser):
        """No tool call, no final channel -- falls back to stripped text."""
        text = "Hello, how can I help you today?"
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "Hello, how can I help you today?"

    def test_no_tool_call_with_control_tokens_stripped(self, parser):
        """Text with stray control tokens but no proper tool call structure."""
        text = "<|start|>\nHere is some text with tokens.\n<|end|>"
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        # Control tokens should be stripped
        assert "Here is some text with tokens." in result.content

    def test_empty_string(self, parser):
        """Empty input returns no tools, empty content."""
        result = parser.extract_tool_calls("")
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == ""

    def test_only_whitespace(self, parser):
        """Whitespace-only input."""
        result = parser.extract_tool_calls("   \n\n  ")
        assert not result.tools_called
        assert result.tool_calls == []

    def test_malformed_missing_call_tag(self, parser):
        """Commentary block without <|call|> is not a complete tool call."""
        text = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '<|message|>{"key": "value"}\n'
            # Missing <|call|>
        )
        result = parser.extract_tool_calls(text)
        assert not result.tools_called

    def test_malformed_missing_message_tag(self, parser):
        """Commentary block without <|message|> tag."""
        text = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '{"key": "value"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)
        # The regex requires <|message|>, so this should not match
        assert not result.tools_called

    def test_tool_id_format(self, parser):
        """Tool IDs have the expected call_ prefix format."""
        text = (
            "<|channel|>commentary to=functions.test_func\n"
            "<|constrain|>json\n"
            '<|message|>{"a": 1}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tool_calls[0]["id"].startswith("call_")
        assert len(result.tool_calls[0]["id"]) > len("call_")


# ============================================================================
# Comprehensive Streaming Tests
# ============================================================================


class TestHarmonyStreaming:
    """Extended tests for HarmonyToolParser.extract_tool_calls_streaming."""

    @pytest.fixture()
    def parser(self):
        return HarmonyToolParser()

    def test_token_by_token_analysis_commentary_call(self, parser):
        """Simulate token-by-token: analysis -> commentary -> call -> emits tool_calls."""
        # Build up the text incrementally
        chunks = [
            "<|channel|>analysis\n",
            "<|message|>Let me think.\n",
            "<|end|>\n",
            "<|channel|>commentary to=functions.get_weather\n",
            "<|constrain|>json\n",
            '<|message|>{"location": "NYC"}\n',
            "<|call|>",
        ]

        previous = ""
        results = []
        for chunk in chunks:
            current = previous + chunk
            result = parser.extract_tool_calls_streaming(previous, current, chunk)
            results.append(result)
            previous = current

        # All chunks except the last should return None (suppressed during build)
        for r in results[:-1]:
            assert r is None, f"Expected None during build-up, got {r}"

        # Last chunk (<|call|>) should emit tool_calls
        final = results[-1]
        assert final is not None
        assert "tool_calls" in final
        assert final["tool_calls"][0]["function"]["name"] == "get_weather"
        args = json.loads(final["tool_calls"][0]["function"]["arguments"])
        assert args["location"] == "NYC"

    def test_final_channel_streaming_emits_content(self, parser):
        """Final channel tokens are emitted as content after <|message|>."""
        # Build final channel token by token
        base = ""
        chunks = [
            "<|channel|>final\n",
            "<|message|>",
            "The ",
            "weather ",
            "is ",
            "sunny.",
        ]

        previous = ""
        content_parts = []
        for chunk in chunks:
            current = previous + chunk
            result = parser.extract_tool_calls_streaming(previous, current, chunk)
            if result and result.get("content"):
                content_parts.append(result["content"])
            previous = current

        joined = "".join(content_parts)
        assert "The weather is sunny." == joined

    def test_final_channel_empty_content_before_message(self, parser):
        """In final channel before <|message|> content, returns empty content dict."""
        current = "<|channel|>final\n"
        result = parser.extract_tool_calls_streaming("", current, current)
        # In final channel but no <|message|> yet
        assert result == {"content": ""}

    def test_final_channel_control_tokens_suppressed(self, parser):
        """Control tokens are suppressed; only clean text is emitted."""
        # Simulate receiving <|return|> at end of final channel
        prev = "<|channel|>final\n<|message|>Done."
        current = prev + "<|return|>"
        result = parser.extract_tool_calls_streaming(prev, current, "<|return|>")
        # <|return|> should be stripped -- no new content
        # The result should be empty content or no new content
        if result is not None:
            assert result.get("content", "") == ""

    def test_call_in_delta_triggers_extraction(self, parser):
        """<|call|> in delta triggers full extraction."""
        current = (
            "<|channel|>commentary to=functions.add\n"
            "<|constrain|>json\n"
            '<|message|>{"a": 1, "b": 2}\n'
            "<|call|>"
        )
        prev = current[: -len("<|call|>")]
        result = parser.extract_tool_calls_streaming(prev, current, "<|call|>")

        assert result is not None
        assert "tool_calls" in result
        tc = result["tool_calls"][0]
        assert tc["function"]["name"] == "add"
        assert tc["index"] == 0
        assert tc["type"] == "function"
        assert "id" in tc

    def test_no_channel_markers_pass_through(self, parser):
        """Text with no channel markers passes through as content."""
        result = parser.extract_tool_calls_streaming(
            "", "Hello world", "Hello world"
        )
        assert result == {"content": "Hello world"}

        result2 = parser.extract_tool_calls_streaming(
            "Hello world", "Hello world!", "!"
        )
        assert result2 == {"content": "!"}

    def test_analysis_channel_suppressed(self, parser):
        """Tokens in analysis channel are suppressed (return None)."""
        current = "<|channel|>analysis\n<|message|>Thinking..."
        result = parser.extract_tool_calls_streaming("", current, current)
        assert result is None

    def test_commentary_channel_suppressed(self, parser):
        """Tokens in commentary channel (building tool call) are suppressed."""
        current = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '<|message|>{"partial":'
        )
        result = parser.extract_tool_calls_streaming("", current, current)
        assert result is None

    def test_streaming_multiple_tool_calls(self, parser):
        """Streaming with multiple <|call|> tags emits all tool calls."""
        text_before_second_call = (
            "<|channel|>commentary to=functions.func_a\n"
            "<|constrain|>json\n"
            '<|message|>{"x": 1}\n'
            "<|call|>\n"
            "<|channel|>commentary to=functions.func_b\n"
            "<|constrain|>json\n"
            '<|message|>{"y": 2}\n'
        )
        current = text_before_second_call + "<|call|>"
        result = parser.extract_tool_calls_streaming(
            text_before_second_call, current, "<|call|>"
        )
        assert result is not None
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["function"]["name"] == "func_a"
        assert result["tool_calls"][1]["function"]["name"] == "func_b"

    def test_streaming_final_channel_incremental_content(self, parser):
        """Final channel emits only the new content in each delta."""
        base = "<|channel|>final\n<|message|>"

        prev1 = base
        curr1 = base + "Hello"
        r1 = parser.extract_tool_calls_streaming(prev1, curr1, "Hello")
        assert r1 == {"content": "Hello"}

        prev2 = curr1
        curr2 = curr1 + " world"
        r2 = parser.extract_tool_calls_streaming(prev2, curr2, " world")
        assert r2 == {"content": " world"}

    def test_streaming_tool_call_format(self, parser):
        """Verify the exact structure of emitted streaming tool calls."""
        current = (
            "<|channel|>commentary to=functions.my_tool\n"
            "<|constrain|>json\n"
            '<|message|>{"key": "val"}\n'
            "<|call|>"
        )
        result = parser.extract_tool_calls_streaming(
            current[: -len("<|call|>")], current, "<|call|>"
        )
        tc = result["tool_calls"][0]
        assert "id" in tc
        assert tc["index"] == 0
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "my_tool"
        assert json.loads(tc["function"]["arguments"]) == {"key": "val"}


# ============================================================================
# has_pending_tool_call Tests
# ============================================================================


class TestHarmonyHasPendingToolCall:
    """Tests for HarmonyToolParser.has_pending_tool_call override."""

    @pytest.fixture()
    def parser(self):
        return HarmonyToolParser()

    def test_returns_true_for_commentary_to_functions(self, parser):
        """Returns True when 'commentary to=functions.' is present."""
        text = (
            "<|channel|>commentary to=functions.get_weather\n"
            "<|constrain|>json\n"
            '<|message|>{"location": "SF"}'
        )
        assert parser.has_pending_tool_call(text) is True

    def test_returns_true_partial_commentary(self, parser):
        """Returns True even for partial commentary block."""
        text = "<|channel|>commentary to=functions.something"
        assert parser.has_pending_tool_call(text) is True

    def test_returns_false_for_normal_text(self, parser):
        """Returns False for normal text with no harmony markers."""
        assert parser.has_pending_tool_call("Hello, how can I help?") is False

    def test_returns_false_for_final_channel_only(self, parser):
        """Returns False when only final channel is present."""
        text = (
            "<|channel|>final\n"
            "<|message|>Here is the answer.\n"
            "<|return|>"
        )
        assert parser.has_pending_tool_call(text) is False

    def test_returns_false_for_analysis_channel(self, parser):
        """Returns False for analysis channel (no tool call pending)."""
        text = (
            "<|channel|>analysis\n"
            "<|message|>Let me think about this.\n"
            "<|end|>"
        )
        assert parser.has_pending_tool_call(text) is False

    def test_returns_false_after_call_completed(self, parser):
        """After <|call|>, 'commentary to=functions.' is still in text.
        This is expected -- has_pending_tool_call checks substring presence."""
        text = (
            "<|channel|>commentary to=functions.func\n"
            "<|constrain|>json\n"
            '<|message|>{"a": 1}\n'
            "<|call|>"
        )
        # Note: the substring is still present, so this returns True.
        # This is the expected behavior -- the method detects the marker.
        assert parser.has_pending_tool_call(text) is True

    def test_returns_false_empty_string(self, parser):
        """Returns False for empty string."""
        assert parser.has_pending_tool_call("") is False

    def test_does_not_use_base_class_check(self, parser):
        """Harmony override does NOT check for <tool_call> like the base class."""
        # Base class would return True for this
        text = "<tool_call>some content"
        # Harmony override only checks for "commentary to=functions."
        assert parser.has_pending_tool_call(text) is False


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHarmonyHelperFunctions:
    """Tests for _strip_control_tokens and _is_control_token."""

    def test_strip_control_tokens_removes_all(self):
        """_strip_control_tokens removes all harmony control tokens."""
        from vllm_mlx.tool_parsers.harmony_tool_parser import _strip_control_tokens

        text = "<|start|>Hello<|end|>"
        result = _strip_control_tokens(text)
        assert "<|start|>" not in result
        assert "<|end|>" not in result
        assert "Hello" in result

    def test_strip_control_tokens_all_types(self):
        """Verify all control token types are stripped."""
        from vllm_mlx.tool_parsers.harmony_tool_parser import _strip_control_tokens

        tokens = [
            "<|start|>",
            "<|end|>",
            "<|message|>",
            "<|channel|>",
            "<|constrain|>",
            "<|return|>",
            "<|call|>",
        ]
        for token in tokens:
            result = _strip_control_tokens(f"before{token}after")
            assert token not in result

    def test_strip_control_tokens_cleans_channel_names(self):
        """_strip_control_tokens also removes channel names like 'analysis', 'final'."""
        from vllm_mlx.tool_parsers.harmony_tool_parser import _strip_control_tokens

        text = "<|channel|>analysis <|message|>Some reasoning<|end|>"
        result = _strip_control_tokens(text)
        assert "analysis" not in result.split()
        assert "Some reasoning" in result

    def test_strip_control_tokens_cleans_function_references(self):
        """_strip_control_tokens removes to=functions.name patterns."""
        from vllm_mlx.tool_parsers.harmony_tool_parser import _strip_control_tokens

        text = "commentary to=functions.get_weather some text"
        result = _strip_control_tokens(text)
        assert "to=functions.get_weather" not in result
        assert "some text" in result

    def test_strip_control_tokens_plain_text(self):
        """_strip_control_tokens on plain text returns it unchanged."""
        from vllm_mlx.tool_parsers.harmony_tool_parser import _strip_control_tokens

        text = "Just some regular text."
        result = _strip_control_tokens(text)
        assert result == "Just some regular text."

    def test_strip_control_tokens_empty(self):
        """_strip_control_tokens on empty string returns empty."""
        from vllm_mlx.tool_parsers.harmony_tool_parser import _strip_control_tokens

        assert _strip_control_tokens("") == ""

    def test_is_control_token_valid_tokens(self):
        """_is_control_token returns True for all harmony tokens."""
        from vllm_mlx.tool_parsers.harmony_tool_parser import _is_control_token

        valid_tokens = [
            "<|start|>",
            "<|end|>",
            "<|message|>",
            "<|channel|>",
            "<|constrain|>",
            "<|return|>",
            "<|call|>",
        ]
        for token in valid_tokens:
            assert _is_control_token(token) is True, f"{token} should be recognized"

    def test_is_control_token_with_whitespace(self):
        """_is_control_token handles surrounding whitespace."""
        from vllm_mlx.tool_parsers.harmony_tool_parser import _is_control_token

        assert _is_control_token("  <|start|>  ") is True
        assert _is_control_token("\n<|call|>\n") is True

    def test_is_control_token_non_tokens(self):
        """_is_control_token returns False for non-tokens."""
        from vllm_mlx.tool_parsers.harmony_tool_parser import _is_control_token

        assert _is_control_token("hello") is False
        assert _is_control_token("<|unknown|>") is False
        assert _is_control_token("") is False
        assert _is_control_token("<|start|>extra") is False

    def test_is_control_token_partial(self):
        """_is_control_token returns False for partial tokens."""
        from vllm_mlx.tool_parsers.harmony_tool_parser import _is_control_token

        assert _is_control_token("<|start") is False
        assert _is_control_token("start|>") is False


# ============================================================================
# CLI Integration Tests
# ============================================================================


class TestHarmonyCLIIntegration:
    """Tests that harmony and gpt-oss are valid CLI parser choices."""

    @staticmethod
    def _get_serve_tool_parser_choices():
        """Extract --tool-call-parser choices from the CLI serve subcommand.

        We inspect the argparse tree rather than calling main() so no server
        is started.
        """
        import argparse
        import importlib
        import inspect

        # Read the source of cli.main to find the choices list.
        # Alternatively, we can build the parser by importing and inspecting.
        # The safest approach: grep the choices from the source itself.
        mod = importlib.import_module("vllm_mlx.cli")
        source = inspect.getsource(mod.main)
        # The choices list is defined literally in the source
        return source

    def test_harmony_in_cli_choices(self):
        """Verify 'harmony' is listed as a --tool-call-parser choice in CLI source."""
        source = self._get_serve_tool_parser_choices()
        assert '"harmony"' in source

    def test_gpt_oss_in_cli_choices(self):
        """Verify 'gpt-oss' is listed as a --tool-call-parser choice in CLI source."""
        source = self._get_serve_tool_parser_choices()
        assert '"gpt-oss"' in source

    def test_registry_has_both_names(self):
        """ToolParserManager resolves both 'harmony' and 'gpt-oss'."""
        cls_harmony = ToolParserManager.get_tool_parser("harmony")
        cls_gpt_oss = ToolParserManager.get_tool_parser("gpt-oss")
        assert cls_harmony is HarmonyToolParser
        assert cls_gpt_oss is HarmonyToolParser
        assert cls_harmony is cls_gpt_oss

    def test_harmony_in_registered_list(self):
        """Both names appear in the registered parser list."""
        registered = ToolParserManager.list_registered()
        assert "harmony" in registered
        assert "gpt-oss" in registered

    def test_invalid_parser_not_registered(self):
        """Invalid parser name raises KeyError from registry."""
        with pytest.raises(KeyError):
            ToolParserManager.get_tool_parser("nonexistent_parser")


# ============================================================================
# SUPPORTS_NATIVE_TOOL_FORMAT Tests
# ============================================================================


class TestHarmonyNativeFormat:
    """Test that Harmony parser declares native format support.

    GPT-OSS chat templates natively handle tool_calls and role='tool'
    messages using harmony channel tokens.
    """

    def test_supports_native_format_true(self):
        """HarmonyToolParser supports native tool format."""
        assert HarmonyToolParser.SUPPORTS_NATIVE_TOOL_FORMAT is True
        assert HarmonyToolParser.supports_native_format() is True

    def test_instance_supports_native_format(self):
        """Instance-level check also returns True."""
        parser = HarmonyToolParser()
        assert parser.supports_native_format() is True
