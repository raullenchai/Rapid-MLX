# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for tool call parsers."""

import json

import pytest

from vllm_mlx.api.tool_calling import parse_tool_calls
from vllm_mlx.tool_parsers import (
    AutoToolParser,
    DeepSeekToolParser,
    FunctionaryToolParser,
    GraniteToolParser,
    HermesToolParser,
    KimiToolParser,
    LlamaToolParser,
    MistralToolParser,
    NemotronToolParser,
    QwenToolParser,
    ToolParserManager,
    xLAMToolParser,
)


class TestToolParserManager:
    """Test the ToolParserManager registry."""

    def test_list_registered(self):
        """Test that all expected parsers are registered."""
        parsers = ToolParserManager.list_registered()
        expected = [
            "auto",
            "mistral",
            "qwen",
            "llama",
            "hermes",
            "deepseek",
            "kimi",
            "granite",
            "nemotron",
            "xlam",
            "functionary",
        ]
        for p in expected:
            assert p in parsers, f"Parser '{p}' not found"

    def test_get_tool_parser_by_name(self):
        """Test getting parsers by name."""
        test_cases = [
            ("mistral", MistralToolParser),
            ("qwen", QwenToolParser),
            ("qwen3", QwenToolParser),
            ("llama", LlamaToolParser),
            ("llama3", LlamaToolParser),
            ("llama4", LlamaToolParser),
            ("auto", AutoToolParser),
            ("deepseek", DeepSeekToolParser),
            ("deepseek_v3", DeepSeekToolParser),
            ("deepseek_r1", DeepSeekToolParser),
            ("kimi", KimiToolParser),
            ("kimi_k2", KimiToolParser),
            ("moonshot", KimiToolParser),
            ("granite", GraniteToolParser),
            ("granite3", GraniteToolParser),
            ("nemotron", NemotronToolParser),
            ("nemotron3", NemotronToolParser),
            ("xlam", xLAMToolParser),
            ("functionary", FunctionaryToolParser),
            ("meetkai", FunctionaryToolParser),
            ("hermes", HermesToolParser),
            ("nous", HermesToolParser),
        ]
        for name, expected_cls in test_cases:
            parser_cls = ToolParserManager.get_tool_parser(name)
            assert parser_cls == expected_cls, f"Parser '{name}' returned wrong class"

    def test_get_unknown_parser_raises(self):
        """Test that unknown parser raises KeyError."""
        with pytest.raises(KeyError):
            ToolParserManager.get_tool_parser("unknown_parser")

    def test_parser_instantiation(self):
        """Test that all parsers can be instantiated without tokenizer."""
        for name in [
            "auto",
            "mistral",
            "qwen",
            "llama",
            "hermes",
            "deepseek",
            "kimi",
            "granite",
            "nemotron",
            "xlam",
            "functionary",
        ]:
            parser_cls = ToolParserManager.get_tool_parser(name)
            parser = parser_cls()  # Should not raise
            assert parser is not None


class TestMistralToolParser:
    """Test the Mistral tool parser."""

    @pytest.fixture
    def parser(self):
        return MistralToolParser()

    def test_old_format_single(self, parser):
        """Test parsing old Mistral format with single tool call."""
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["city"] == "Paris"

    def test_old_format_multiple(self, parser):
        """Test parsing old Mistral format with multiple tool calls."""
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "get_time", "arguments": {"timezone": "UTC"}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[1]["name"] == "get_time"

    def test_new_format(self, parser):
        """Test parsing new Mistral format."""
        text = '[TOOL_CALLS]get_weather{"city": "London"}'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_no_tool_call(self, parser):
        """Test that regular text is not parsed as tool call."""
        text = "Hello, how can I help you today?"
        result = parser.extract_tool_calls(text)

        assert not result.tools_called
        assert result.content == text

    def test_content_with_tool_call(self, parser):
        """Test content before tool call is preserved."""
        text = 'Let me check the weather for you.[TOOL_CALLS] [{"name": "get_weather", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.content == "Let me check the weather for you."


class TestQwenToolParser:
    """Test the Qwen tool parser."""

    @pytest.fixture
    def parser(self):
        return QwenToolParser()

    def test_xml_format(self, parser):
        """Test parsing Qwen XML format."""
        text = '<tool_call>{"name": "calculate", "arguments": {"x": 1, "y": 2}}</tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "calculate"

    def test_bracket_format(self, parser):
        """Test parsing Qwen bracket format (Qwen3 style)."""
        text = '[Calling tool: add({"a": 5, "b": 3})]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add"

    def test_multiple_xml_calls(self, parser):
        """Test multiple XML tool calls."""
        text = '<tool_call>{"name": "func1", "arguments": {}}</tool_call><tool_call>{"name": "func2", "arguments": {}}</tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "I can help you with that question."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestGenericToolCallParsing:
    """Test generic OpenAI tool-call translation helpers."""

    def test_bare_calling_tool_is_parsed_and_removed(self):
        text = (
            "Thinking first.\n"
            'Calling tool: write({"filePath": "/tmp/app.tsx", "content": "ok"})'
        )

        cleaned_text, tool_calls = parse_tool_calls(text)

        assert cleaned_text == "Thinking first."
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "write"
        args = json.loads(tool_calls[0].function.arguments)
        assert args == {"filePath": "/tmp/app.tsx", "content": "ok"}

    def test_tool_arguments_decode_stringified_arrays(self):
        text = (
            'Calling tool: todowrite({"todos": '
            '"[{\\"content\\": \\"Initialize\\", \\"status\\": \\"in_progress\\"}]"'
            "})"
        )

        _, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        args = json.loads(tool_calls[0].function.arguments)
        assert isinstance(args["todos"], list)
        assert args["todos"][0]["content"] == "Initialize"


class TestLlamaToolParser:
    """Test the Llama tool parser."""

    @pytest.fixture
    def parser(self):
        return LlamaToolParser()

    def test_function_format(self, parser):
        """Test parsing Llama function format."""
        text = '<function=multiply>{"x": 3, "y": 4}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "multiply"

    def test_multiple_functions(self, parser):
        """Test parsing multiple function calls."""
        text = '<function=add>{"a": 1}</function><function=multiply>{"x": 3}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "add"
        assert result.tool_calls[1]["name"] == "multiply"

    def test_content_with_function(self, parser):
        """Test content before function call."""
        text = 'Computing result<function=calc>{"n": 5}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.content == "Computing result"


class TestHermesToolParser:
    """Test the Hermes tool parser."""

    @pytest.fixture
    def parser(self):
        return HermesToolParser()

    def test_tool_call_format(self, parser):
        """Test parsing Hermes format."""
        text = (
            '<tool_call>{"name": "search", "arguments": {"query": "test"}}</tool_call>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"

    def test_with_reasoning(self, parser):
        """Test with reasoning block."""
        text = '<tool_call_reasoning>I need to search for this</tool_call_reasoning><tool_call>{"name": "search", "arguments": {}}</tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert "Reasoning" in (result.content or "")


class TestDeepSeekToolParser:
    """Test the DeepSeek tool parser."""

    @pytest.fixture
    def parser(self):
        return DeepSeekToolParser()

    def test_deepseek_format(self, parser):
        """Test parsing DeepSeek V3 format."""
        text = """<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "Tokyo"}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_multiple_calls(self, parser):
        """Test multiple DeepSeek tool calls."""
        text = """<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>func1
```json
{"a": 1}
```<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>func2
```json
{"b": 2}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2

    def test_content_before_tools(self, parser):
        """Test content before tool calls is preserved."""
        text = """Let me help you with that.<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>search
```json
{}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.content == "Let me help you with that."

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "Here is my response without any tool calls."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestKimiToolParser:
    """Test the Kimi tool parser."""

    @pytest.fixture
    def parser(self):
        return KimiToolParser()

    def test_kimi_format(self, parser):
        """Test parsing Kimi K2 format."""
        text = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Beijing"}<|tool_call_end|>
<|tool_calls_section_end|>"""
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_simple_function_name(self, parser):
        """Test with simple function name (no functions. prefix)."""
        text = (
            "<|tool_call_begin|>search:0<|tool_call_argument_begin|>{}<|tool_call_end|>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "I'll answer your question directly."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestGraniteToolParser:
    """Test the Granite tool parser."""

    @pytest.fixture
    def parser(self):
        return GraniteToolParser()

    def test_granite_30_format(self, parser):
        """Test parsing Granite 3.0 format."""
        text = (
            '<|tool_call|>[{"name": "calculate", "arguments": {"expression": "2+2"}}]'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "calculate"

    def test_granite_31_format(self, parser):
        """Test parsing Granite 3.1 format."""
        text = '<tool_call>[{"name": "search", "arguments": {"query": "test"}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_multiple_calls(self, parser):
        """Test multiple tool calls."""
        text = '<|tool_call|>[{"name": "func1", "arguments": {}}, {"name": "func2", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "The answer is 42."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestNemotronToolParser:
    """Test the Nemotron tool parser."""

    @pytest.fixture
    def parser(self):
        return NemotronToolParser()

    def test_parameter_format(self, parser):
        """Test parsing Nemotron parameter format."""
        text = "<tool_call><function=get_weather><parameter=city>Paris</parameter><parameter=units>celsius</parameter></function></tool_call>"
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["city"] == "Paris"
        assert args["units"] == "celsius"

    def test_json_format(self, parser):
        """Test parsing Nemotron with JSON arguments."""
        text = '<tool_call><function=calculate>{"expression": "2*3"}</function></tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "calculate"

    def test_multiple_calls(self, parser):
        """Test multiple Nemotron tool calls."""
        text = "<tool_call><function=func1><parameter=a>1</parameter></function></tool_call><tool_call><function=func2><parameter=b>2</parameter></function></tool_call>"
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "Here is the information you requested."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestXLAMToolParser:
    """Test the xLAM tool parser."""

    @pytest.fixture
    def parser(self):
        return xLAMToolParser()

    def test_json_array(self, parser):
        """Test parsing JSON array format."""
        text = '[{"name": "search", "arguments": {"query": "AI"}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_code_block(self, parser):
        """Test parsing markdown code block."""
        text = '```json\n[{"name": "calculate", "arguments": {"x": 5}}]\n```'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "calculate"

    def test_after_think(self, parser):
        """Test parsing after </think> tag."""
        text = (
            '<think>Let me search for this</think>[{"name": "search", "arguments": {}}]'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_tool_calls_tag(self, parser):
        """Test [TOOL_CALLS] tag format."""
        text = '[TOOL_CALLS][{"name": "func", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "I don't need to use any tools for this."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestFunctionaryToolParser:
    """Test the Functionary tool parser."""

    @pytest.fixture
    def parser(self):
        return FunctionaryToolParser()

    def test_recipient_format(self, parser):
        """Test parsing Functionary v3 recipient format."""
        text = '<|from|>assistant\n<|recipient|>get_weather\n<|content|>{"city": "NYC"}'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_function_format(self, parser):
        """Test parsing function format."""
        text = '<function=search>{"query": "test"}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_json_array(self, parser):
        """Test parsing JSON array."""
        text = '[{"name": "func1", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "Let me explain that to you."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestAutoToolParser:
    """Test the auto-detecting tool parser."""

    @pytest.fixture
    def parser(self):
        return AutoToolParser()

    def test_detects_mistral(self, parser):
        """Test auto detection of Mistral format."""
        text = '[TOOL_CALLS] [{"name": "search", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_detects_qwen_xml(self, parser):
        """Test auto detection of Qwen XML format."""
        text = '<tool_call>{"name": "calculate", "arguments": {}}</tool_call>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "calculate"

    def test_detects_qwen_bracket(self, parser):
        """Test auto detection of Qwen bracket format."""
        text = '[Calling tool: add({"a": 1})]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "add"

    def test_detects_llama(self, parser):
        """Test auto detection of Llama format."""
        text = '<function=multiply>{"x": 2}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "multiply"

    def test_detects_nemotron(self, parser):
        """Test auto detection of Nemotron format."""
        text = "<tool_call><function=search><parameter=q>test</parameter></function></tool_call>"
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_detects_raw_json(self, parser):
        """Test auto detection of raw JSON format."""
        text = '{"name": "test_func", "arguments": {"key": "value"}}'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "test_func"

    def test_no_tool_call(self, parser):
        """Test text without tool calls."""
        text = "This is just a regular response."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input(self):
        """Test with empty input."""
        parsers = [
            MistralToolParser(),
            QwenToolParser(),
            LlamaToolParser(),
            DeepSeekToolParser(),
            AutoToolParser(),
        ]
        for parser in parsers:
            result = parser.extract_tool_calls("")
            assert not result.tools_called

    def test_malformed_json(self):
        """Test with malformed JSON."""
        parser = MistralToolParser()
        text = '[TOOL_CALLS] [{"name": "func", "arguments": {invalid json}]'
        result = parser.extract_tool_calls(text)
        # Should not crash, may or may not parse

    def test_nested_arguments(self):
        """Test with deeply nested arguments."""
        parser = AutoToolParser()
        args = {"level1": {"level2": {"level3": [1, 2, 3]}}}
        text = f'{{"name": "complex", "arguments": {json.dumps(args)}}}'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        parsed_args = json.loads(result.tool_calls[0]["arguments"])
        assert parsed_args["level1"]["level2"]["level3"] == [1, 2, 3]

    def test_unicode_in_arguments(self):
        """Test with unicode characters in arguments."""
        parser = MistralToolParser()
        text = '[TOOL_CALLS] [{"name": "translate", "arguments": {"text": "日本語"}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["text"] == "日本語"

    def test_special_characters_in_name(self):
        """Test function names with special characters."""
        parser = LlamaToolParser()
        text = '<function=get_user_info>{"user_id": 123}</function>'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_user_info"

    def test_tool_call_id_uniqueness(self):
        """Test that each tool call gets a unique ID."""
        parser = MistralToolParser()
        text = '[TOOL_CALLS] [{"name": "func1", "arguments": {}}, {"name": "func2", "arguments": {}}]'
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        ids = [tc["id"] for tc in result.tool_calls]
        assert len(ids) == len(set(ids)), "Tool call IDs should be unique"


class TestStreamingParsing:
    """Test streaming tool call parsing."""

    def test_mistral_streaming(self):
        """Test Mistral streaming parsing."""
        parser = MistralToolParser()

        # Simulate streaming
        result1 = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Let me",
            delta_text="Let me",
        )
        assert result1 == {"content": "Let me"}

        result2 = parser.extract_tool_calls_streaming(
            previous_text="Let me",
            current_text="Let me[TOOL_CALLS]",
            delta_text="[TOOL_CALLS]",
        )
        # Should start tool call parsing

    def test_auto_streaming(self):
        """Test auto parser streaming."""
        parser = AutoToolParser()

        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Hello world",
            delta_text="Hello world",
        )
        assert result == {"content": "Hello world"}


class TestThinkTagStripping:
    """Test <think> tag stripping in tool parsers (Issue #26)."""

    def test_strip_think_tags_utility(self):
        """Test the strip_think_tags static method."""
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        # Basic stripping
        text = "<think>Let me analyze this</think>The answer is 42"
        assert ToolParser.strip_think_tags(text) == "The answer is 42"

        # Multi-line thinking
        text = "<think>Step 1\nStep 2\nStep 3</think>Result"
        assert ToolParser.strip_think_tags(text) == "Result"

        # No think tags
        text = "Just regular text"
        assert ToolParser.strip_think_tags(text) == "Just regular text"

        # Empty think tags
        text = "<think></think>Content"
        assert ToolParser.strip_think_tags(text) == "Content"

    def test_hermes_with_think_tags(self):
        """Test Hermes parser strips think tags before parsing tool calls."""
        parser = HermesToolParser()

        # Model output with think tags AND tool call (Ring-Mini-Linear-2.0 style)
        output = """<think>Let me search for that information.</think>
<tool_call>{"name": "search", "arguments": {"query": "weather"}}</tool_call>"""

        result = parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"

    def test_qwen_with_think_tags(self):
        """Test Qwen parser strips think tags before parsing tool calls."""
        parser = QwenToolParser()

        # Model output with think tags AND tool call
        output = """<think>I need to get the weather data.</think>
[Calling tool: get_weather({"city": "Tokyo"})]"""

        result = parser.extract_tool_calls(output)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_think_tags_with_no_tool_call(self):
        """Test that think tags are stripped even when no tool call is present."""
        parser = HermesToolParser()

        output = "<think>Let me think about this</think>The answer is 42."
        result = parser.extract_tool_calls(output)

        assert result.tools_called is False
        assert result.content == "The answer is 42."


class TestQwen3CoderParser:
    """Test Qwen3-Coder tool call parsing (Issue #47)."""

    def test_qwen3_coder_alias_registered(self):
        """Test that qwen3_coder is registered as an alias for HermesToolParser."""
        parser_cls = ToolParserManager.get_tool_parser("qwen3_coder")
        assert parser_cls == HermesToolParser

    def test_qwen3_coder_xml_format(self):
        """Test parsing Qwen3-Coder XML format (Nemotron-style)."""
        parser = HermesToolParser()
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Paris</parameter>\n"
            "<parameter=units>celsius</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["city"] == "Paris"
        assert args["units"] == "celsius"

    def test_qwen3_coder_with_think_tags(self):
        """Test Qwen3-Coder XML format with think tags."""
        parser = HermesToolParser()
        text = (
            "<think>I need to read this file.</think>\n"
            "<tool_call>\n"
            "<function=read_file>\n"
            "<parameter=path>/src/main.py</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "read_file"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["path"] == "/src/main.py"

    def test_qwen3_coder_multiline_parameter(self):
        """Test Qwen3-Coder with multi-line parameter values (code)."""
        parser = HermesToolParser()
        text = (
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=path>/src/hello.py</parameter>\n"
            "<parameter=content>def hello():\n"
            "    print('hello')\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "write_file"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["path"] == "/src/hello.py"
        assert "def hello():" in args["content"]

    def test_bare_function_without_tool_call_wrapper(self):
        """Test bare <function=...> blocks without <tool_call> wrapper."""
        parser = HermesToolParser()
        text = "<function=get_weather><parameter=city>Berlin</parameter></function>"
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["city"] == "Berlin"

    def test_bare_multi_function_without_wrapper(self):
        """Test multiple bare <function=...> blocks without <tool_call> wrapper."""
        parser = HermesToolParser()
        text = (
            "<function=read_file>"
            "<parameter=path>/a.py</parameter>"
            "</function>\n"
            "<function=write_file>"
            "<parameter=path>/b.py</parameter>"
            "<parameter=content>hello</parameter>"
            "</function>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "read_file"
        assert result.tool_calls[1]["name"] == "write_file"

    def test_qwen3_coder_multiple_tool_calls(self):
        """Test multiple Nemotron XML tool calls (multi-tool scenario)."""
        parser = HermesToolParser()
        text = (
            "<tool_call>\n"
            "<function=read_file>\n"
            "<parameter=path>/src/main.py</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=list_files>\n"
            "<parameter=directory>/src</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "read_file"
        assert result.tool_calls[1]["name"] == "list_files"


class TestHermesStreamingFixes:
    """Test streaming fixes for Hermes parser (Issue #47)."""

    def test_streaming_complete_tool_call(self):
        """Test streaming with complete tool call in accumulated text."""
        parser = HermesToolParser()

        # Simulate token-by-token streaming
        # Token 1: regular content
        r = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Sure, let me check.",
            delta_text="Sure, let me check.",
        )
        assert r == {"content": "Sure, let me check."}

        # Token 2: opening tag
        r = parser.extract_tool_calls_streaming(
            previous_text="Sure, let me check.",
            current_text='Sure, let me check.<tool_call>{"name":',
            delta_text='<tool_call>{"name":',
        )
        assert r is None  # Inside tool call, suppress

        # Token 3: more JSON
        r = parser.extract_tool_calls_streaming(
            previous_text='Sure, let me check.<tool_call>{"name":',
            current_text='Sure, let me check.<tool_call>{"name": "search", "arguments": {"q": "test"}}',
            delta_text=' "search", "arguments": {"q": "test"}}',
        )
        assert r is None  # Still inside, no closing tag yet

        # Token 4: closing tag
        r = parser.extract_tool_calls_streaming(
            previous_text='Sure, let me check.<tool_call>{"name": "search", "arguments": {"q": "test"}}',
            current_text='Sure, let me check.<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>',
            delta_text="</tool_call>",
        )
        assert r is not None
        assert "tool_calls" in r
        assert r["tool_calls"][0]["function"]["name"] == "search"

    def test_streaming_split_closing_tag(self):
        """Test that </tool_call> split across deltas is detected via current_text."""
        parser = HermesToolParser()

        # Accumulated text has <tool_call> but </tool_call is split
        r = parser.extract_tool_calls_streaming(
            previous_text='<tool_call>{"name": "func", "arguments": {}}',
            current_text='<tool_call>{"name": "func", "arguments": {}}</tool_call',
            delta_text="</tool_call",
        )
        # Not yet complete (missing >)
        assert r is None

        # Now the > arrives, completing </tool_call>
        r = parser.extract_tool_calls_streaming(
            previous_text='<tool_call>{"name": "func", "arguments": {}}</tool_call',
            current_text='<tool_call>{"name": "func", "arguments": {}}</tool_call>',
            delta_text=">",
        )
        assert r is not None
        assert "tool_calls" in r
        assert r["tool_calls"][0]["function"]["name"] == "func"

    def test_streaming_content_after_tool_call(self):
        """Test that content after </tool_call> is not suppressed."""
        parser = HermesToolParser()

        # Complete tool call already in text
        full = '<tool_call>{"name": "func", "arguments": {}}</tool_call>'
        r = parser.extract_tool_calls_streaming(
            previous_text=full[:-1],  # everything except last >
            current_text=full,
            delta_text=">",
        )
        assert "tool_calls" in r

        # Now content comes after the tool call
        r = parser.extract_tool_calls_streaming(
            previous_text=full,
            current_text=full + "\nHere is the result.",
            delta_text="\nHere is the result.",
        )
        # Should NOT be suppressed
        assert r is not None
        assert r.get("content") == "\nHere is the result."

    def test_streaming_nemotron_xml_format(self):
        """Test streaming with Qwen3-Coder/Nemotron XML format."""
        parser = HermesToolParser()

        chunks = [
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>London</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        accumulated = ""
        tool_calls_found = False
        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            r = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=accumulated,
                delta_text=chunk,
            )
            if r is not None and "tool_calls" in r:
                tool_calls_found = True
                assert r["tool_calls"][0]["function"]["name"] == "get_weather"
                break

        assert tool_calls_found, "Tool call should have been detected"

    def test_streaming_no_false_positives(self):
        """Test that regular text with < doesn't trigger false positives."""
        parser = HermesToolParser()

        r = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Use x < 10 and y > 5",
            delta_text="Use x < 10 and y > 5",
        )
        assert r == {"content": "Use x < 10 and y > 5"}

    def test_streaming_multi_tool_calls(self):
        """Test streaming with two sequential tool calls (Issue #47)."""
        parser = HermesToolParser()

        # First tool call arrives token by token
        chunks = [
            '<tool_call>{"name": "read_file", "arguments": {"path": "/a.py"}}',
            "</tool_call>",
            "\n",
            '<tool_call>{"name": "list_dir", "arguments": {"dir": "/src"}}',
            "</tool_call>",
        ]

        accumulated = ""
        emitted_calls = []
        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            r = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=accumulated,
                delta_text=chunk,
            )
            if r is not None and "tool_calls" in r:
                emitted_calls.extend(r["tool_calls"])

        assert len(emitted_calls) == 2
        assert emitted_calls[0]["function"]["name"] == "read_file"
        assert emitted_calls[1]["function"]["name"] == "list_dir"
        # Verify indexes are correct
        assert emitted_calls[0]["index"] == 0
        assert emitted_calls[1]["index"] == 1

    def test_streaming_multi_tool_no_duplicates(self):
        """Test that completed tool calls are not re-emitted on second completion."""
        parser = HermesToolParser()

        first_complete = '<tool_call>{"name": "func1", "arguments": {}}</tool_call>'
        # After first tool call is emitted, content between calls passes through
        between = "\n"
        second_start = '<tool_call>{"name": "func2", "arguments": {}}'
        second_end = "</tool_call>"

        # First tool call completed
        r1 = parser.extract_tool_calls_streaming(
            previous_text=first_complete[:-1],
            current_text=first_complete,
            delta_text=">",
        )
        assert r1 is not None and "tool_calls" in r1
        assert len(r1["tool_calls"]) == 1
        assert r1["tool_calls"][0]["function"]["name"] == "func1"

        # Whitespace between
        r2 = parser.extract_tool_calls_streaming(
            previous_text=first_complete,
            current_text=first_complete + between,
            delta_text=between,
        )
        assert r2 == {"content": between}

        # Second tool call building (inside block, suppress)
        r3 = parser.extract_tool_calls_streaming(
            previous_text=first_complete + between,
            current_text=first_complete + between + second_start,
            delta_text=second_start,
        )
        assert r3 is None  # Inside incomplete second tool call

        # Second tool call completed
        r4 = parser.extract_tool_calls_streaming(
            previous_text=first_complete + between + second_start,
            current_text=first_complete + between + second_start + second_end,
            delta_text=second_end,
        )
        assert r4 is not None and "tool_calls" in r4
        assert len(r4["tool_calls"]) == 1  # Only the NEW tool call
        assert r4["tool_calls"][0]["function"]["name"] == "func2"
        assert r4["tool_calls"][0]["index"] == 1  # Second index

    def test_streaming_multi_nemotron_xml(self):
        """Test streaming with multiple Nemotron XML tool calls."""
        parser = HermesToolParser()

        chunks = [
            "<tool_call>\n<function=read_file>\n",
            "<parameter=path>/a.py</parameter>\n",
            "</function>\n</tool_call>\n",
            "<tool_call>\n<function=write_file>\n",
            "<parameter=path>/b.py</parameter>\n",
            "<parameter=content>hello</parameter>\n",
            "</function>\n</tool_call>",
        ]

        accumulated = ""
        emitted_calls = []
        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            r = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=accumulated,
                delta_text=chunk,
            )
            if r is not None and "tool_calls" in r:
                emitted_calls.extend(r["tool_calls"])

        assert len(emitted_calls) == 2
        assert emitted_calls[0]["function"]["name"] == "read_file"
        assert emitted_calls[1]["function"]["name"] == "write_file"

    def test_streaming_bare_function_blocks(self):
        """Test streaming with bare <function= blocks without <tool_call> wrapper."""
        parser = HermesToolParser()

        chunks = [
            "<function=read_file>",
            "<parameter=path>/src/main.py</parameter>",
            "</function>",
        ]

        accumulated = ""
        tool_calls_found = False
        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            r = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=accumulated,
                delta_text=chunk,
            )
            if r is not None and "tool_calls" in r:
                tool_calls_found = True
                assert r["tool_calls"][0]["function"]["name"] == "read_file"
                args = json.loads(r["tool_calls"][0]["function"]["arguments"])
                assert args["path"] == "/src/main.py"
                break

        assert tool_calls_found, "Bare function block should have been detected"

    def test_streaming_bare_multi_function_blocks(self):
        """Test streaming with multiple bare <function= blocks."""
        parser = HermesToolParser()

        chunks = [
            "<function=func1><parameter=a>1</parameter></function>",
            "\n",
            "<function=func2>",
            "<parameter=b>2</parameter>",
            "</function>",
        ]

        accumulated = ""
        emitted_calls = []
        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            r = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=accumulated,
                delta_text=chunk,
            )
            if r is not None and "tool_calls" in r:
                emitted_calls.extend(r["tool_calls"])

        assert len(emitted_calls) == 2
        assert emitted_calls[0]["function"]["name"] == "func1"
        assert emitted_calls[1]["function"]["name"] == "func2"


class TestTextFormatToolCallFallback:
    """Test text-format tool call fallback parser.

    Models at low quantization (e.g., 4-bit) sometimes degrade after multiple
    tool call rounds and output tool calls as plain text instead of structured
    format.  The base ToolParser class provides general detection and extraction
    for two common degradation patterns:

    Variant 1 (KV style):  [Calling tool="name" key="value" ...]
    Variant 2 (function call style):  [Calling tool: name({"key": "value"})]
    """

    # -- Fixtures --

    @pytest.fixture
    def minimax_parser(self):
        from vllm_mlx.tool_parsers import MiniMaxToolParser

        return MiniMaxToolParser()

    @pytest.fixture
    def hermes_parser(self):
        return HermesToolParser()

    # -- Helpers --

    def _assert_tool_call(self, tc, name, **expected_args):
        """Assert a tool call dict has the expected name and arguments."""
        assert tc["id"].startswith("call_"), (
            f"Tool call id should start with 'call_', got {tc['id']}"
        )
        assert tc["name"] == name
        args = json.loads(tc["arguments"])
        for k, v in expected_args.items():
            assert args[k] == v, f"Expected {k}={v!r}, got {args[k]!r}"

    # =================================================================
    # Variant 1 - KV style
    # =================================================================

    def test_variant1_simple_kv(self):
        """Test basic KV-style text-format tool call."""
        text = '[Calling tool="web_search" query="weather palo alto"]'
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 1
        self._assert_tool_call(calls[0], "web_search", query="weather palo alto")

    def test_variant1_multiple_params(self):
        """Test KV-style with multiple parameters."""
        text = '[Calling tool="exec" command="ls -la" timeout="5000"]'
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 1
        self._assert_tool_call(calls[0], "exec", command="ls -la")
        args = json.loads(calls[0]["arguments"])
        # timeout should be parsed as int by json.loads
        assert args["timeout"] == 5000

    def test_variant1_escaped_quotes(self):
        """Test KV-style with escaped quotes in value."""
        text = r'[Calling tool="exec" command="curl -s \"https://example.com\""]'
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "exec"
        args = json.loads(calls[0]["arguments"])
        assert "https://example.com" in args["command"]

    def test_variant1_single_param(self):
        """Test KV-style with a single parameter."""
        text = '[Calling tool="read" path="/tmp/file.txt"]'
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 1
        self._assert_tool_call(calls[0], "read", path="/tmp/file.txt")

    # =================================================================
    # Variant 2 - Function call style
    # =================================================================

    def test_variant2_json_args(self):
        """Test function-call style with JSON arguments."""
        text = '[Calling tool: process({"action":"poll", "sessionId":"clear-haven", "timeout":5000})]'
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 1
        self._assert_tool_call(
            calls[0],
            "process",
            action="poll",
            sessionId="clear-haven",
            timeout=5000,
        )

    def test_variant2_simple_json(self):
        """Test function-call style with simple JSON."""
        text = '[Calling tool: web_search({"query":"weather tonight"})]'
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 1
        self._assert_tool_call(calls[0], "web_search", query="weather tonight")

    def test_variant2_single_key(self):
        """Test function-call style with single key."""
        text = '[Calling tool: exec({"command":"python3 --version"})]'
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 1
        self._assert_tool_call(calls[0], "exec", command="python3 --version")

    # =================================================================
    # Edge cases
    # =================================================================

    def test_inside_think_tags(self):
        """Text-format tool call embedded inside <think>...</think> tags."""
        text = '<think>I should search for this.\n[Calling tool="web_search" query="test"]</think>'
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        # The raw extraction should still find it inside think tags
        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 1
        self._assert_tool_call(calls[0], "web_search", query="test")

    def test_content_before(self):
        """Text-format tool call with content BEFORE it."""
        text = 'Let me check the weather for you. [Calling tool="web_search" query="weather"]'
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 1
        self._assert_tool_call(calls[0], "web_search", query="weather")

    def test_content_after(self):
        """Text-format tool call with content AFTER it."""
        text = '[Calling tool="web_search" query="weather"] I will get the results.'
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 1
        self._assert_tool_call(calls[0], "web_search", query="weather")

    def test_multiple_text_format_calls(self):
        """Multiple text-format tool calls in same output."""
        text = (
            '[Calling tool="web_search" query="weather"]\n'
            '[Calling tool="exec" command="date"]'
        )
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "web_search"
        assert calls[1]["name"] == "exec"

    def test_mixed_xml_and_text_format(self):
        """Mixed: one XML tool call + one text-format tool call in same output."""
        # The text-format extractor only extracts text-format calls,
        # not XML ones.  Verify it does not pick up the XML call.
        text = (
            '<minimax:tool_call><invoke name="func1"><parameter name="a">1</parameter></invoke></minimax:tool_call>\n'
            '[Calling tool="func2" b="2"]'
        )
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "func2"

    def test_unicode_in_arguments(self):
        """Unicode in arguments (Chinese, emoji)."""
        text = '[Calling tool="translate" text="\u4f60\u597d\u4e16\u754c"]'
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0]["arguments"])
        assert args["text"] == "\u4f60\u597d\u4e16\u754c"

    def test_unicode_emoji_in_arguments(self):
        """Emoji characters in arguments."""
        text = '[Calling tool="react" emoji="\U0001f680\U0001f525"]'
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0]["arguments"])
        assert args["emoji"] == "\U0001f680\U0001f525"

    def test_variant2_nested_json(self):
        """Nested JSON in variant 2."""
        text = '[Calling tool: configure({"key": {"nested": true}})]'
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0]["arguments"])
        assert args["key"]["nested"] is True

    def test_has_text_format_tool_call_true(self):
        """has_text_format_tool_call() returns True for matching text."""
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        assert ToolParser.has_text_format_tool_call('[Calling tool="web_search" q="x"]')
        assert ToolParser.has_text_format_tool_call('[Calling tool: func({"a":1})]')

    def test_has_text_format_tool_call_false(self):
        """has_text_format_tool_call() returns False for non-matching text."""
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        assert not ToolParser.has_text_format_tool_call("Hello, world!")
        assert not ToolParser.has_text_format_tool_call("[Calling out to the void]")
        assert not ToolParser.has_text_format_tool_call(
            '<tool_call>{"name":"f"}</tool_call>'
        )

    def test_has_pending_tool_call_text_format(self):
        """has_pending_tool_call() on base ToolParser returns True for text-format."""
        # Use a concrete subclass (HermesToolParser inherits from ToolParser)
        parser = HermesToolParser()
        assert parser.has_pending_tool_call('[Calling tool="search" q="test"]')

    def test_variant1_no_params_should_not_match(self):
        """Variant 1 with no params (just tool name) should NOT produce a tool call."""
        # The pattern requires at least one key="value" param, and extract
        # also checks `if arguments:` before appending.
        text = '[Calling tool="web_search"]'
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 0

    def test_partial_incomplete_should_not_match(self):
        """Partial/incomplete text-format (missing closing ']') should NOT match."""
        text = '[Calling tool="web_search" query="test"'
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 0

    def test_both_variants_in_same_text(self):
        """Both variant 1 and variant 2 in the same text."""
        text = (
            '[Calling tool="read" path="/tmp/data.txt"]\n'
            '[Calling tool: exec({"command":"cat /tmp/data.txt"})]'
        )
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 2
        names = {c["name"] for c in calls}
        assert names == {"read", "exec"}
        # Verify unique IDs
        ids = [c["id"] for c in calls]
        assert len(ids) == len(set(ids))

    # =================================================================
    # MiniMax-specific integration
    # =================================================================

    def test_minimax_extract_falls_back_to_text_format(self, minimax_parser):
        """MiniMax extract_tool_calls() falls back to text-format when no XML found."""
        text = '[Calling tool="web_search" query="weather"]'
        result = minimax_parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        self._assert_tool_call(result.tool_calls[0], "web_search", query="weather")

    def test_minimax_extract_text_format_variant2(self, minimax_parser):
        """MiniMax extract_tool_calls() handles variant 2 text-format."""
        text = '[Calling tool: exec({"command":"ls -la"})]'
        result = minimax_parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        self._assert_tool_call(result.tool_calls[0], "exec", command="ls -la")

    def test_minimax_extract_text_format_with_content(self, minimax_parser):
        """MiniMax extracts text-format and preserves surrounding content."""
        text = 'Let me check that for you. [Calling tool="web_search" query="test"]'
        result = minimax_parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.content == "Let me check that for you."

    def test_minimax_has_pending_text_format(self, minimax_parser):
        """MiniMax has_pending_tool_call() detects text-format."""
        assert minimax_parser.has_pending_tool_call('[Calling tool="search" q="x"]')
        assert not minimax_parser.has_pending_tool_call("Just regular text.")

    def test_minimax_has_tool_start_text_format(self, minimax_parser):
        """MiniMax _has_tool_start() detects [Calling tool="..."."""
        assert minimax_parser._has_tool_start('[Calling tool="web_search" q="x"]')
        assert not minimax_parser._has_tool_start("Regular text without tool calls.")

    def test_minimax_has_tool_end_text_format(self, minimax_parser):
        """MiniMax _has_tool_end() detects text-format completion."""
        previous = 'Some text [Calling tool="search" q="te'
        current = '[Calling tool="search" q="test"]'
        assert minimax_parser._has_tool_end(current, previous)

    def test_minimax_has_tool_end_no_new_match(self, minimax_parser):
        """MiniMax _has_tool_end() returns False when no new match appeared."""
        text = '[Calling tool="search" q="test"]'
        assert not minimax_parser._has_tool_end(text, text)

    def test_minimax_streaming_text_format(self, minimax_parser):
        """MiniMax streaming: text-format tool call suppressed until complete, then emitted."""
        chunks = [
            "Let me check. ",
            '[Calling tool="web_search"',
            ' query="weather palo alto"',
            "]",
        ]

        accumulated = ""
        content_parts = []
        tool_calls_found = []
        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            r = minimax_parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=accumulated,
                delta_text=chunk,
            )
            if r is not None:
                if "content" in r:
                    content_parts.append(r["content"])
                if "tool_calls" in r:
                    tool_calls_found.extend(r["tool_calls"])

        # Content before the tool call should have been emitted
        assert any("Let me check." in c for c in content_parts)
        # Tool call should have been emitted once complete
        assert len(tool_calls_found) == 1
        assert tool_calls_found[0]["function"]["name"] == "web_search"
        args = json.loads(tool_calls_found[0]["function"]["arguments"])
        assert args["query"] == "weather palo alto"

    def test_minimax_streaming_text_format_variant2(self, minimax_parser):
        """MiniMax streaming: variant 2 text-format is detected when complete."""
        chunks = [
            "[Calling tool: exec(",
            '{"command":"python3 --version"}',
            ")]",
        ]

        accumulated = ""
        tool_calls_found = []
        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            r = minimax_parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=accumulated,
                delta_text=chunk,
            )
            if r is not None and "tool_calls" in r:
                tool_calls_found.extend(r["tool_calls"])

        assert len(tool_calls_found) == 1
        assert tool_calls_found[0]["function"]["name"] == "exec"

    # =================================================================
    # General ToolParser integration
    # =================================================================

    def test_hermes_inherits_has_pending_text_format(self, hermes_parser):
        """HermesToolParser inherits has_pending_tool_call detecting text-format."""
        assert hermes_parser.has_pending_tool_call(
            '[Calling tool="search" query="test"]'
        )

    def test_hermes_has_pending_false_for_plain_text(self, hermes_parser):
        """HermesToolParser has_pending_tool_call returns False for plain text."""
        assert not hermes_parser.has_pending_tool_call("Just a regular message.")

    def test_variant2_empty_json_should_not_match(self):
        """Variant 2 with empty JSON object should NOT produce a tool call."""
        text = "[Calling tool: func({})]"
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        # Empty dict check: `if isinstance(arguments, dict) and arguments:`
        assert len(calls) == 0

    def test_tool_call_ids_are_unique(self):
        """Each extracted tool call gets a unique ID."""
        text = (
            '[Calling tool="func1" a="1"]\n'
            '[Calling tool="func2" b="2"]\n'
            '[Calling tool: func3({"c":"3"})]'
        )
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        calls = ToolParser.extract_text_format_tool_calls(text)
        assert len(calls) == 3
        ids = [c["id"] for c in calls]
        assert len(ids) == len(set(ids)), "All tool call IDs should be unique"

    def test_arguments_are_valid_json(self):
        """Extracted arguments field is always valid JSON."""
        texts = [
            '[Calling tool="read" path="/tmp/file.txt"]',
            '[Calling tool: search({"query":"hello world"})]',
            '[Calling tool="exec" command="echo hello" timeout="30"]',
        ]
        from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

        for text in texts:
            calls = ToolParser.extract_text_format_tool_calls(text)
            for call in calls:
                parsed = json.loads(call["arguments"])
                assert isinstance(parsed, dict)
