# SPDX-License-Identifier: Apache-2.0
"""Tests for tool injection fallback when chat templates reject tools param."""

import copy

from vllm_mlx.utils.chat_template import (
    _build_tool_injection_text,
    _inject_tools_into_messages,
    apply_chat_template,
)

# Sample tool definitions in OpenAI format
SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["C", "F"]},
                },
                "required": ["location"],
            },
        },
    }
]

MESSAGES_WITH_SYSTEM = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Paris?"},
]

MESSAGES_NO_SYSTEM = [
    {"role": "user", "content": "What's the weather in Paris?"},
]


class TestBuildToolInjectionText:
    """Tests for _build_tool_injection_text."""

    def test_includes_tool_name(self):
        text = _build_tool_injection_text(SAMPLE_TOOLS)
        assert "get_weather" in text

    def test_includes_description(self):
        text = _build_tool_injection_text(SAMPLE_TOOLS)
        assert "Get current weather" in text

    def test_includes_parameters(self):
        text = _build_tool_injection_text(SAMPLE_TOOLS)
        assert '"location"' in text

    def test_includes_required(self):
        text = _build_tool_injection_text(SAMPLE_TOOLS)
        assert '["location"]' in text

    def test_includes_calling_instruction(self):
        text = _build_tool_injection_text(SAMPLE_TOOLS)
        assert '"name"' in text
        assert '"arguments"' in text

    def test_multiple_tools(self):
        tools = SAMPLE_TOOLS + [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        text = _build_tool_injection_text(tools)
        assert "get_weather" in text
        assert "search" in text


class TestInjectToolsIntoMessages:
    """Tests for _inject_tools_into_messages."""

    def test_appends_to_existing_system(self):
        msgs = _inject_tools_into_messages(MESSAGES_WITH_SYSTEM, SAMPLE_TOOLS)
        assert msgs[0]["role"] == "system"
        assert "You are a helpful assistant." in msgs[0]["content"]
        assert "get_weather" in msgs[0]["content"]

    def test_creates_system_when_none(self):
        msgs = _inject_tools_into_messages(MESSAGES_NO_SYSTEM, SAMPLE_TOOLS)
        assert msgs[0]["role"] == "system"
        assert "get_weather" in msgs[0]["content"]
        assert msgs[1]["role"] == "user"

    def test_does_not_mutate_original(self):
        original = copy.deepcopy(MESSAGES_WITH_SYSTEM)
        _inject_tools_into_messages(MESSAGES_WITH_SYSTEM, SAMPLE_TOOLS)
        assert original == MESSAGES_WITH_SYSTEM

    def test_preserves_message_count_with_system(self):
        msgs = _inject_tools_into_messages(MESSAGES_WITH_SYSTEM, SAMPLE_TOOLS)
        assert len(msgs) == len(MESSAGES_WITH_SYSTEM)

    def test_adds_one_message_without_system(self):
        msgs = _inject_tools_into_messages(MESSAGES_NO_SYSTEM, SAMPLE_TOOLS)
        assert len(msgs) == len(MESSAGES_NO_SYSTEM) + 1

    def test_handles_content_parts_format(self):
        """System message with content parts (multimodal) should append text part."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are helpful."}],
            },
            {"role": "user", "content": "Hi"},
        ]
        msgs = _inject_tools_into_messages(messages, SAMPLE_TOOLS)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "You are helpful."}
        assert content[1]["type"] == "text"
        assert "get_weather" in content[1]["text"]

    def test_content_parts_does_not_mutate_original(self):
        """Content parts list must not be mutated."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are helpful."}],
            },
            {"role": "user", "content": "Hi"},
        ]
        original_len = len(messages[0]["content"])
        _inject_tools_into_messages(messages, SAMPLE_TOOLS)
        assert len(messages[0]["content"]) == original_len


class TestApplyChatTemplateToolInjection:
    """Tests for apply_chat_template with tool injection fallback."""

    class MockTokenizerAcceptsTools:
        """Tokenizer that accepts tools param."""

        def apply_chat_template(self, messages, **kwargs):
            tools = kwargs.get("tools")
            if tools:
                return f"TOOLS:{len(tools)}|" + messages[-1]["content"]
            return messages[-1]["content"]

    class MockTokenizerRejectsTools:
        """Tokenizer that raises TypeError on tools/enable_thinking."""

        def apply_chat_template(self, messages, **kwargs):
            if "tools" in kwargs or "enable_thinking" in kwargs:
                raise TypeError("got an unexpected keyword argument 'tools'")
            # Return the system message content to verify injection
            for m in messages:
                if m["role"] == "system":
                    return f"SYSTEM:{m['content']}|USER:{messages[-1]['content']}"
            return f"USER:{messages[-1]['content']}"

    def test_native_tools_not_injected(self):
        """When template accepts tools, no injection occurs."""
        tok = self.MockTokenizerAcceptsTools()
        result = apply_chat_template(tok, MESSAGES_WITH_SYSTEM, tools=SAMPLE_TOOLS)
        assert result.startswith("TOOLS:1|")

    def test_tools_injected_on_typeerror(self):
        """When template rejects tools, definitions are injected into system."""
        tok = self.MockTokenizerRejectsTools()
        result = apply_chat_template(tok, MESSAGES_WITH_SYSTEM, tools=SAMPLE_TOOLS)
        assert "get_weather" in result

    def test_no_tools_no_injection(self):
        """When no tools provided, no injection occurs even on TypeError."""
        tok = self.MockTokenizerRejectsTools()
        result = apply_chat_template(tok, MESSAGES_WITH_SYSTEM, tools=None)
        assert "get_weather" not in result

    def test_injection_creates_system_if_needed(self):
        """When there's no system message, one is created for injection."""
        tok = self.MockTokenizerRejectsTools()
        result = apply_chat_template(tok, MESSAGES_NO_SYSTEM, tools=SAMPLE_TOOLS)
        assert "SYSTEM:" in result
        assert "get_weather" in result

    def test_original_messages_not_mutated(self):
        """Original messages list must not be modified."""
        tok = self.MockTokenizerRejectsTools()
        original = copy.deepcopy(MESSAGES_WITH_SYSTEM)
        apply_chat_template(tok, MESSAGES_WITH_SYSTEM, tools=SAMPLE_TOOLS)
        assert original == MESSAGES_WITH_SYSTEM


try:
    from vllm_mlx.tool_parsers.mistral_tool_parser import MistralToolParser

    _has_mistral_parser = True
except ImportError:
    _has_mistral_parser = False

import pytest


@pytest.mark.skipif(
    not _has_mistral_parser, reason="transformers not installed"
)
class TestMistralArgsStripping:
    """Tests for [ARGS] suffix stripping in Mistral parser."""

    def test_args_suffix_stripped_extract(self):
        parser = MistralToolParser(tokenizer=None)
        output = '[TOOL_CALLS]get_weather[ARGS]{"location": "Paris"}'
        result = parser.extract_tool_calls(output)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_no_args_suffix_still_works(self):
        parser = MistralToolParser(tokenizer=None)
        output = '[TOOL_CALLS]get_weather{"location": "Paris"}'
        result = parser.extract_tool_calls(output)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_args_suffix_stripped_streaming(self):
        parser = MistralToolParser(tokenizer=None)
        delta = parser._parse_streaming_tool_delta(
            'get_weather[ARGS]{"location": "Paris"}'
        )
        assert delta is not None
        assert delta["name"] == "get_weather"
