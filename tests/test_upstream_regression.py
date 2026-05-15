# SPDX-License-Identifier: Apache-2.0
"""
Upstream regression tests — test cases ported from vLLM (vllm-project/vllm)
to verify our tool parser forks haven't broken correctness.

Sources:
  - tests/tool_parsers/test_glm4_moe_tool_parser.py
  - tests/tool_parsers/test_mistral_tool_parser.py
  - tests/tool_parsers/test_seed_oss_tool_parser.py
  - tests/tool_parsers/test_deepseekv31_tool_parser.py
  - tests/tool_parsers/test_qwen3coder_tool_parser.py
"""

import json

import pytest

from vllm_mlx.tool_parsers import ToolParserManager

# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def glm47_parser():
    cls = ToolParserManager.get_tool_parser("glm47")
    return cls(tokenizer=None)


@pytest.fixture
def mistral_parser():
    cls = ToolParserManager.get_tool_parser("mistral")
    return cls(tokenizer=None)


@pytest.fixture
def glm47_request():
    """Minimal request dict with tools (GLM47 uses tool names for validation)."""
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "state": {"type": "string"},
                            "unit": {"type": "string"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "date": {"type": "string"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "send_message",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recipient": {"type": "string"},
                            "message": {"type": "string"},
                            "priority": {"type": "string"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {"type": "string"},
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                            "enabled": {"type": "boolean"},
                        },
                    },
                },
            },
        ]
    }


# ═══════════════════════════════════════════════════════════════════════
# GLM-4.7 (glm47) — ported from vLLM test_glm4_moe_tool_parser.py
# ═══════════════════════════════════════════════════════════════════════


class TestGlm47UpstreamNonStreaming:
    """Non-streaming tests ported from upstream vLLM."""

    def test_no_tools(self, glm47_parser, glm47_request):
        """Plain text → no tool calls."""
        result = glm47_parser.extract_tool_calls("This is a test", glm47_request)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "This is a test"

    def test_single_tool_call(self, glm47_parser, glm47_request):
        """Single tool with 3 args."""
        output = """<tool_call>get_current_weather
<arg_key>city</arg_key><arg_value>Dallas</arg_value>
<arg_key>state</arg_key><arg_value>TX</arg_value>
<arg_key>unit</arg_key><arg_value>fahrenheit</arg_value>
</tool_call>"""
        result = glm47_parser.extract_tool_calls(output, glm47_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["name"] == "get_current_weather"
        args = json.loads(tc["arguments"])
        assert args == {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}

    def test_multiple_tool_calls(self, glm47_parser, glm47_request):
        """Two tool calls in sequence."""
        output = """<tool_call>get_current_weather
<arg_key>city</arg_key><arg_value>Dallas</arg_value>
<arg_key>state</arg_key><arg_value>TX</arg_value>
<arg_key>unit</arg_key><arg_value>fahrenheit</arg_value>
</tool_call>
<tool_call>get_current_weather
<arg_key>city</arg_key><arg_value>Orlando</arg_value>
<arg_key>state</arg_key><arg_value>FL</arg_value>
<arg_key>unit</arg_key><arg_value>fahrenheit</arg_value>
</tool_call>"""
        result = glm47_parser.extract_tool_calls(output, glm47_request)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        args0 = json.loads(result.tool_calls[0]["arguments"])
        args1 = json.loads(result.tool_calls[1]["arguments"])
        assert args0["city"] == "Dallas"
        assert args1["city"] == "Orlando"

    def test_tool_call_with_content_before(self, glm47_parser, glm47_request):
        """Content before tool call — upstream expects content preserved."""
        output = """I'll help you check the weather. <tool_call>get_current_weather
<arg_key>city</arg_key><arg_value>Seattle</arg_value>
<arg_key>state</arg_key><arg_value>WA</arg_value>
<arg_key>unit</arg_key><arg_value>celsius</arg_value>
</tool_call>"""
        result = glm47_parser.extract_tool_calls(output, glm47_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_current_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"city": "Seattle", "state": "WA", "unit": "celsius"}

    def test_tool_call_with_chinese_content(self, glm47_parser, glm47_request):
        """Chinese content before tool call + date argument."""
        output = """I will help you get the weather.<tool_call>get_weather
<arg_key>city</arg_key><arg_value>Beijing</arg_value>
<arg_key>date</arg_key><arg_value>2025-08-01</arg_value>
</tool_call>"""
        result = glm47_parser.extract_tool_calls(output, glm47_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"city": "Beijing", "date": "2025-08-01"}

    def test_thinking_tags(self, glm47_parser, glm47_request):
        """Tool call after <think>...</think> block."""
        output = """<think>I want to get the weather.</think>

I will help you get the weather.
<tool_call>get_weather
<arg_key>city</arg_key><arg_value>Beijing</arg_value>
<arg_key>date</arg_key><arg_value>2025-08-01</arg_value>
</tool_call>"""
        result = glm47_parser.extract_tool_calls(output, glm47_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_empty_arguments(self, glm47_parser, glm47_request):
        """Tool call with no arguments."""
        output = """<tool_call>get_current_time
</tool_call>"""
        result = glm47_parser.extract_tool_calls(output, glm47_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_current_time"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {}

    def test_special_characters(self, glm47_parser, glm47_request):
        """Tool call with special characters in values."""
        output = """<tool_call>send_message
<arg_key>recipient</arg_key><arg_value>Amy</arg_value>
<arg_key>message</arg_key><arg_value>It is a nice day</arg_value>
<arg_key>priority</arg_key><arg_value>high</arg_value>
</tool_call>"""
        result = glm47_parser.extract_tool_calls(output, glm47_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["recipient"] == "Amy"
        assert args["message"] == "It is a nice day"
        assert args["priority"] == "high"

    def test_incomplete_tool_call(self, glm47_parser, glm47_request):
        """Missing </tool_call> → should NOT extract."""
        output = """<tool_call>get_weather
<arg_key>city</arg_key><arg_value>Beijing</arg_value>
<arg_key>date</arg_key><arg_value>2025-08-01</arg_value>"""
        result = glm47_parser.extract_tool_calls(output, glm47_request)
        assert not result.tools_called
        assert result.tool_calls == []

    def test_numeric_deserialization(self, glm47_parser, glm47_request):
        """Integer, float, and boolean arg values should deserialize to correct types."""
        output = """<tool_call>calculate
<arg_key>operation</arg_key><arg_value>add</arg_value>
<arg_key>a</arg_key><arg_value>42</arg_value>
<arg_key>b</arg_key><arg_value>3.14</arg_value>
<arg_key>enabled</arg_key><arg_value>true</arg_value>
</tool_call>"""
        result = glm47_parser.extract_tool_calls(output, glm47_request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["operation"] == "add"
        assert isinstance(args["operation"], str)
        assert args["a"] == 42
        assert isinstance(args["a"], int)
        assert args["b"] == 3.14
        assert isinstance(args["b"], float)
        assert args["enabled"] is True
        assert isinstance(args["enabled"], bool)

    def test_mixed_content_between_tools(self, glm47_parser, glm47_request):
        """Content between two tool calls — both should be extracted."""
        output = """I will help you get the weather info.

<tool_call>get_weather
<arg_key>city</arg_key><arg_value>Beijing</arg_value>
<arg_key>date</arg_key><arg_value>2025-08-01</arg_value>
</tool_call>

meanwhile, I will also check the weather in Shanghai.

<tool_call>get_weather
<arg_key>city</arg_key><arg_value>Shanghai</arg_value>
<arg_key>date</arg_key><arg_value>2025-08-01</arg_value>
</tool_call>"""
        result = glm47_parser.extract_tool_calls(output, glm47_request)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        args0 = json.loads(result.tool_calls[0]["arguments"])
        args1 = json.loads(result.tool_calls[1]["arguments"])
        assert args0["city"] == "Beijing"
        assert args1["city"] == "Shanghai"

    def test_malformed_xml_graceful(self, glm47_parser, glm47_request):
        """Malformed XML (missing </arg_key>) — should not crash."""
        output = """<tool_call>get_weather
<arg_key>city</arg_key><arg_value>Seattle</arg_value>
<arg_key>incomplete_arg
<arg_value>value</arg_value>
</tool_call>"""
        # Should not raise; may or may not extract
        result = glm47_parser.extract_tool_calls(output, glm47_request)
        assert isinstance(result.tools_called, bool)
        assert isinstance(result.tool_calls, list)


class TestGlm47UpstreamStreaming:
    """Streaming tests ported from upstream vLLM."""

    def test_streaming_no_tool_calls(self, glm47_parser, glm47_request):
        """Regular text in streaming → content delta."""
        result = glm47_parser.extract_tool_calls_streaming(
            previous_text="Hello",
            current_text="Hello world",
            delta_text=" world",
            request=glm47_request,
        )
        assert result is not None
        assert result["content"] == " world"

    def test_streaming_buffers_during_tool_call(self, glm47_parser, glm47_request):
        """While inside <tool_call> but before </tool_call>, returns None."""
        result = glm47_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="<tool_call>get_weather\n<arg_key>city</arg_key>",
            delta_text="<arg_key>city</arg_key>",
            request=glm47_request,
        )
        assert result is None

    def test_streaming_emits_on_close(self, glm47_parser, glm47_request):
        """When </tool_call> arrives, tool calls should be emitted."""
        full = """<tool_call>get_weather
<arg_key>city</arg_key><arg_value>Beijing</arg_value>
</tool_call>"""
        result = glm47_parser.extract_tool_calls_streaming(
            previous_text=full.replace("</tool_call>", ""),
            current_text=full,
            delta_text="</tool_call>",
            request=glm47_request,
        )
        assert result is not None
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        args = json.loads(tc["function"]["arguments"])
        assert args["city"] == "Beijing"

    def test_streaming_multiple_tools_on_close(self, glm47_parser, glm47_request):
        """Two tool calls, second </tool_call> emits both."""
        full = """<tool_call>get_weather
<arg_key>city</arg_key><arg_value>Beijing</arg_value>
</tool_call>
<tool_call>get_weather
<arg_key>city</arg_key><arg_value>Shanghai</arg_value>
</tool_call>"""
        result = glm47_parser.extract_tool_calls_streaming(
            previous_text=full.replace("</tool_call>", "", 1).rsplit("</tool_call>", 1)[
                0
            ],
            current_text=full,
            delta_text="</tool_call>",
            request=glm47_request,
        )
        # Our GLM parser re-parses the full text on close, so both should appear
        assert result is not None
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 2


# ═══════════════════════════════════════════════════════════════════════
# Mistral — ported from vLLM test_mistral_tool_parser.py
# ═══════════════════════════════════════════════════════════════════════


class TestMistralUpstreamNonStreaming:
    """Non-streaming tests ported from upstream vLLM."""

    def test_no_tools(self, mistral_parser):
        """Plain text → no tool calls."""
        result = mistral_parser.extract_tool_calls("This is a test", request=None)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "This is a test"

    # --- Old format (pre v11): [TOOL_CALLS] [{...}] ---

    @pytest.mark.parametrize(
        "model_output, expected_name, expected_args",
        [
            (
                '[TOOL_CALLS][{"name": "add", "arguments":{"a": 3.5, "b": 4}}]',
                "add",
                {"a": 3.5, "b": 4},
            ),
            (
                '[TOOL_CALLS] [{"name": "get_current_weather", "arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}}]',
                "get_current_weather",
                {"city": "San Francisco", "state": "CA", "unit": "celsius"},
            ),
            (
                '[TOOL_CALLS] [{"arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}, "name": "get_current_weather"}]',
                "get_current_weather",
                {"city": "San Francisco", "state": "CA", "unit": "celsius"},
            ),
            (
                '[TOOL_CALLS] [{"arguments":{"name": "John Doe"}, "name": "get_age"}]',
                "get_age",
                {"name": "John Doe"},
            ),
        ],
        ids=[
            "single_tool_add",
            "single_tool_weather",
            "argument_before_name",
            "argument_before_name_and_name_in_argument",
        ],
    )
    def test_old_format_single(
        self, mistral_parser, model_output, expected_name, expected_args
    ):
        """Old Mistral format: [TOOL_CALLS] [{"name": ..., "arguments": ...}]"""
        result = mistral_parser.extract_tool_calls(model_output, request=None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["name"] == expected_name
        assert len(tc["id"]) == 9  # Mistral IDs are 9-char alphanumeric
        args = json.loads(tc["arguments"])
        assert args == expected_args

    def test_old_format_multiple(self, mistral_parser):
        """Old format with two tools in one JSON array."""
        output = '[TOOL_CALLS] [{"name": "add", "arguments": {"a": 3.5, "b": 4}}, {"name": "get_current_weather", "arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}}]'
        result = mistral_parser.extract_tool_calls(output, request=None)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "add"
        assert result.tool_calls[1]["name"] == "get_current_weather"

    # --- New format (>= v11): [TOOL_CALLS]func_name{...} ---

    @pytest.mark.parametrize(
        "model_output, expected_name, expected_args",
        [
            (
                '[TOOL_CALLS]add_this_and_that{"a": 3.5, "b": 4}',
                "add_this_and_that",
                {"a": 3.5, "b": 4},
            ),
            (
                '[TOOL_CALLS]get_current_weather{"city": "San Francisco", "state": "CA", "unit": "celsius"}',
                "get_current_weather",
                {"city": "San Francisco", "state": "CA", "unit": "celsius"},
            ),
        ],
        ids=[
            "new_format_add",
            "new_format_weather",
        ],
    )
    def test_new_format_single(
        self, mistral_parser, model_output, expected_name, expected_args
    ):
        """New Mistral format: [TOOL_CALLS]func_name{...}"""
        result = mistral_parser.extract_tool_calls(model_output, request=None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["name"] == expected_name
        args = json.loads(tc["arguments"])
        assert args == expected_args

    def test_new_format_multiple(self, mistral_parser):
        """New format with two [TOOL_CALLS] in one output."""
        output = '[TOOL_CALLS]add{"a": 3.5, "b": 4}[TOOL_CALLS]multiply{"a": 3, "b": 6}'
        result = mistral_parser.extract_tool_calls(output, request=None)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "add"
        assert result.tool_calls[1]["name"] == "multiply"
        args0 = json.loads(result.tool_calls[0]["arguments"])
        args1 = json.loads(result.tool_calls[1]["arguments"])
        assert args0 == {"a": 3.5, "b": 4}
        assert args1 == {"a": 3, "b": 6}

    def test_content_before_tool_call(self, mistral_parser):
        """Content before [TOOL_CALLS] should be preserved."""
        output = 'hi{hi[TOOL_CALLS]bash{"command": "print(\\"hello world!\\")\\nre.compile(r\'{}\')"}'
        result = mistral_parser.extract_tool_calls(output, request=None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "bash"
        assert result.content == "hi{hi"

    def test_complex_escaped_json(self, mistral_parser):
        """Complex JSON with escaped quotes and newlines."""
        output = '[TOOL_CALLS]bash{"command": "print(\\"hello world!\\")\\nre.compile(r\'{}\')"}'
        result = mistral_parser.extract_tool_calls(output, request=None)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "bash"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert "print" in args["command"]
        assert "re.compile" in args["command"]


# ─── Fixtures for new parsers ────────────────────────────────────────


@pytest.fixture
def seed_oss_parser():
    cls = ToolParserManager.get_tool_parser("seed_oss")
    return cls(tokenizer=None)


@pytest.fixture
def deepseekv31_parser():
    cls = ToolParserManager.get_tool_parser("deepseek_v31")
    return cls(tokenizer=None)


@pytest.fixture
def qwen3coder_parser():
    cls = ToolParserManager.get_tool_parser("qwen3_coder_xml")
    return cls(tokenizer=None)


@pytest.fixture
def seed_oss_request():
    """Request with tools for Seed-OSS type conversion tests."""
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "number"},
                            "op": {"type": "string"},
                            "enabled": {"type": "boolean"},
                            "config": {"type": "object"},
                        },
                    },
                },
            },
        ]
    }


@pytest.fixture
def qwen3coder_request():
    """Request with tools for Qwen3-Coder type conversion tests."""
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "state": {"type": "string"},
                            "unit": {"type": "string"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_area",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "shape": {"type": "string"},
                            "dimensions": {"type": "object"},
                            "precision": {"type": "integer"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "test_types",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "int_param": {"type": "integer"},
                            "float_param": {"type": "float"},
                            "bool_param": {"type": "boolean"},
                            "str_param": {"type": "string"},
                            "obj_param": {"type": "object"},
                        },
                    },
                },
            },
        ]
    }


# ═══════════════════════════════════════════════════════════════════════
# Seed-OSS — ported from vLLM test_seed_oss_tool_parser.py
# ═══════════════════════════════════════════════════════════════════════


class TestSeedOssUpstreamNonStreaming:
    """Non-streaming tests ported from upstream vLLM."""

    def test_no_tools(self, seed_oss_parser):
        """Plain text → no tool calls."""
        result = seed_oss_parser.extract_tool_calls(
            "This is a test response without any tool calls", request=None
        )
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "This is a test response without any tool calls"

    def test_single_tool_call(self, seed_oss_parser, seed_oss_request):
        """Single tool call with <seed:tool_call> wrapper."""
        output = (
            "<seed:tool_call>\n<function=get_weather>\n"
            "<parameter=location>Barcelona, Spain</parameter>\n"
            "</function>\n</seed:tool_call>"
        )
        result = seed_oss_parser.extract_tool_calls(output, seed_oss_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["name"] == "get_weather"
        args = json.loads(tc["arguments"])
        assert args == {"location": "Barcelona, Spain"}

    def test_tool_call_with_two_params(self, seed_oss_parser, seed_oss_request):
        """Tool call with two parameters."""
        output = (
            "<seed:tool_call>\n<function=get_weather>\n"
            "<parameter=location>Barcelona, Spain</parameter>\n"
            "<parameter=unit>celsius</parameter>\n"
            "</function>\n</seed:tool_call>"
        )
        result = seed_oss_parser.extract_tool_calls(output, seed_oss_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args == {"location": "Barcelona, Spain", "unit": "celsius"}

    def test_tool_call_with_thinking(self, seed_oss_parser, seed_oss_request):
        """Tool call after <seed:think>...</seed:think> block."""
        output = (
            "<seed:think>I should check the weather.</seed:think>\n"
            "<seed:tool_call>\n<function=get_weather>\n"
            "<parameter=location>Barcelona, Spain</parameter>\n"
            "</function>\n</seed:tool_call>"
        )
        result = seed_oss_parser.extract_tool_calls(output, seed_oss_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        # Content should include the thinking block
        assert result.content is not None
        assert "<seed:think>" in result.content

    def test_content_before_tool_call(self, seed_oss_parser, seed_oss_request):
        """Content before tool call is preserved."""
        output = (
            "Let me check that for you.\n"
            "<seed:tool_call>\n<function=get_weather>\n"
            "<parameter=location>Paris, France</parameter>\n"
            "</function>\n</seed:tool_call>"
        )
        result = seed_oss_parser.extract_tool_calls(output, seed_oss_request)
        assert result.tools_called
        assert result.content is not None
        assert "Let me check that" in result.content

    def test_multiple_tool_calls(self, seed_oss_parser, seed_oss_request):
        """Multiple tool calls in sequence."""
        output = (
            "<seed:tool_call>\n<function=get_weather>\n"
            "<parameter=location>Paris</parameter>\n"
            "</function>\n</seed:tool_call>\n"
            "<seed:tool_call>\n<function=get_weather>\n"
            "<parameter=location>London</parameter>\n"
            "</function>\n</seed:tool_call>"
        )
        result = seed_oss_parser.extract_tool_calls(output, seed_oss_request)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        args0 = json.loads(result.tool_calls[0]["arguments"])
        args1 = json.loads(result.tool_calls[1]["arguments"])
        assert args0["location"] == "Paris"
        assert args1["location"] == "London"

    def test_type_conversion_integer(self, seed_oss_parser, seed_oss_request):
        """Integer parameter type conversion."""
        output = (
            "<seed:tool_call>\n<function=calculate>\n"
            "<parameter=a>42</parameter>\n"
            "<parameter=b>3.14</parameter>\n"
            "<parameter=op>add</parameter>\n"
            "<parameter=enabled>true</parameter>\n"
            "</function>\n</seed:tool_call>"
        )
        result = seed_oss_parser.extract_tool_calls(output, seed_oss_request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["a"] == 42
        assert isinstance(args["a"], int)
        assert args["b"] == 3.14
        assert isinstance(args["b"], float)
        assert args["op"] == "add"
        assert args["enabled"] is True

    def test_type_conversion_object(self, seed_oss_parser, seed_oss_request):
        """Object parameter type conversion."""
        output = (
            "<seed:tool_call>\n<function=calculate>\n"
            '<parameter=config>{"key": "value"}</parameter>\n'
            "<parameter=op>test</parameter>\n"
            "</function>\n</seed:tool_call>"
        )
        result = seed_oss_parser.extract_tool_calls(output, seed_oss_request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["config"] == {"key": "value"}


class TestSeedOssUpstreamStreaming:
    """Streaming tests ported from upstream vLLM."""

    def test_streaming_no_tools(self, seed_oss_parser):
        """Regular text → content delta."""
        result = seed_oss_parser.extract_tool_calls_streaming(
            previous_text="Hello",
            current_text="Hello world",
            delta_text=" world",
        )
        assert result is not None
        assert result["content"] == " world"

    def test_streaming_buffers_during_tool(self, seed_oss_parser):
        """Inside tool call but before close → returns None or tool header."""
        # First delta starts the tool call
        result = seed_oss_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="<seed:tool_call>\n<function=get_weather>",
            delta_text="<seed:tool_call>\n<function=get_weather>",
        )
        # Should either return None (buffering) or a tool_calls header — not content
        assert result is None or "tool_calls" in result

    def test_streaming_content_before_tool(self, seed_oss_parser):
        """Content before tool call is streamed as content."""
        result = seed_oss_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Let me check",
            delta_text="Let me check",
        )
        assert result is not None
        assert result["content"] == "Let me check"

    def test_streaming_thinking_content(self, seed_oss_parser):
        """Thinking content before seed:think end is streamed."""
        result = seed_oss_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="<seed:think>thinking...",
            delta_text="<seed:think>thinking...",
        )
        assert result is not None
        assert "content" in result

    def test_streaming_full_tool_call_multistep(
        self, seed_oss_parser, seed_oss_request
    ):
        """Multi-step streaming: header → { → param → } across calls.

        Streaming parsers emit one piece per call; callers must invoke
        extract_tool_calls_streaming once per token/delta (fine-grained).
        """
        deltas = [
            "<seed:tool_call>",
            "\n<function=get_weather>",
            "\n",
            "<parameter=location>Paris</parameter>",
            "\n</function>",
            "\n</seed:tool_call>",
        ]
        text = ""
        collected = []
        for d in deltas:
            prev = text
            text += d
            r = seed_oss_parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=text,
                delta_text=d,
                request=seed_oss_request,
            )
            if r:
                collected.append(r)

        # Should have: header (name), opening {, param fragment, closing }
        names = [
            c["tool_calls"][0]["function"].get("name")
            for c in collected
            if "tool_calls" in c and "name" in c["tool_calls"][0].get("function", {})
        ]
        assert "get_weather" in names

        # Concatenate all argument fragments
        arg_parts = [
            c["tool_calls"][0]["function"]["arguments"]
            for c in collected
            if "tool_calls" in c
            and "arguments" in c["tool_calls"][0].get("function", {})
        ]
        full_args = "".join(arg_parts)
        assert full_args.startswith("{")
        assert full_args.endswith("}")
        parsed = json.loads(full_args)
        assert parsed["location"] == "Paris"

    def test_streaming_coarse_deltas_complete(self, seed_oss_parser, seed_oss_request):
        """Two coarse deltas: header + complete body → full args emitted.

        Reproduces the scenario where the function body is already complete
        when the header is first detected (e.g. fast model, large chunk).
        """
        deltas = [
            "<seed:tool_call>\n<function=get_weather>"
            "\n<parameter=location>Paris</parameter>\n</function>"
            "\n</seed:tool_call>",
        ]
        text = ""
        collected = []
        for d in deltas:
            prev = text
            text += d
            r = seed_oss_parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=text,
                delta_text=d,
                request=seed_oss_request,
            )
            if r:
                collected.append(r)

        # Must have at least one tool_calls chunk with non-empty arguments
        tc_chunks = [c for c in collected if "tool_calls" in c]
        assert len(tc_chunks) >= 1
        # First chunk should have complete arguments (fast-path)
        first_tc = tc_chunks[0]["tool_calls"][0]
        assert first_tc["function"]["name"] == "get_weather"
        args = first_tc["function"]["arguments"]
        assert args  # not empty
        parsed = json.loads(args)
        assert parsed["location"] == "Paris"


# ═══════════════════════════════════════════════════════════════════════
# DeepSeek V3.1 — ported from vLLM test_deepseekv31_tool_parser.py
# ═══════════════════════════════════════════════════════════════════════


class TestDeepSeekV31UpstreamNonStreaming:
    """Non-streaming tests ported from upstream vLLM."""

    def test_no_tools(self, deepseekv31_parser):
        """Plain text → no tool calls."""
        result = deepseekv31_parser.extract_tool_calls("This is a test", request=None)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "This is a test"

    def test_single_tool_call(self, deepseekv31_parser):
        """Single tool call in V3.1 format (no code fence, no type prefix)."""
        output = (
            "normal text"
            "<｜tool▁calls▁begin｜>"
            '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
        )
        result = deepseekv31_parser.extract_tool_calls(output, request=None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "foo"
        assert result.tool_calls[0]["arguments"] == '{"x":1}'
        assert result.content == "normal text"

    def test_multiple_tool_calls(self, deepseekv31_parser):
        """Multiple tool calls in V3.1 format."""
        output = (
            "some prefix text"
            "<｜tool▁calls▁begin｜>"
            '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
            '<｜tool▁call▁begin｜>bar<｜tool▁sep｜>{"y":2}<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
        )
        result = deepseekv31_parser.extract_tool_calls(output, request=None)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "foo"
        assert result.tool_calls[0]["arguments"] == '{"x":1}'
        assert result.tool_calls[1]["name"] == "bar"
        assert result.tool_calls[1]["arguments"] == '{"y":2}'
        assert result.content == "some prefix text"

    def test_content_preserved(self, deepseekv31_parser):
        """Content before tool calls is preserved."""
        output = (
            "I'll help with that!"
            "<｜tool▁calls▁begin｜>"
            '<｜tool▁call▁begin｜>search<｜tool▁sep｜>{"q":"test"}<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
        )
        result = deepseekv31_parser.extract_tool_calls(output, request=None)
        assert result.tools_called
        assert result.content == "I'll help with that!"

    def test_no_tool_calls_start(self, deepseekv31_parser):
        """Without tool_calls_begin token, treat as content."""
        output = "Just some regular text without any special tokens"
        result = deepseekv31_parser.extract_tool_calls(output, request=None)
        assert not result.tools_called
        assert result.content == output

    def test_complex_json_args(self, deepseekv31_parser):
        """Tool call with nested JSON arguments."""
        output = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>create_event<｜tool▁sep｜>"
            '{"title":"Meeting","details":{"time":"3pm","room":"A1"}}'
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = deepseekv31_parser.extract_tool_calls(output, request=None)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "create_event"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["title"] == "Meeting"
        assert args["details"]["room"] == "A1"


class TestDeepSeekV31UpstreamStreaming:
    """Streaming tests ported from upstream vLLM."""

    def test_streaming_no_tools(self, deepseekv31_parser):
        """Regular text → content delta."""
        result = deepseekv31_parser.extract_tool_calls_streaming(
            previous_text="Hello",
            current_text="Hello world",
            delta_text=" world",
        )
        assert result is not None
        assert result["content"] == " world"

    def test_streaming_content_before_tools(self, deepseekv31_parser):
        """Content before tool calls start token."""
        result = deepseekv31_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Some text",
            delta_text="Some text",
        )
        assert result is not None
        assert result["content"] == "Some text"


# ═══════════════════════════════════════════════════════════════════════
# Qwen3-Coder XML — ported from vLLM test_qwen3coder_tool_parser.py
# ═══════════════════════════════════════════════════════════════════════


class TestQwen3CoderUpstreamNonStreaming:
    """Non-streaming tests ported from upstream vLLM."""

    def test_no_tools(self, qwen3coder_parser):
        """Plain text → no tool calls."""
        result = qwen3coder_parser.extract_tool_calls(
            "This is a test response without any tool calls", request=None
        )
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "This is a test response without any tool calls"

    def test_single_tool_call(self, qwen3coder_parser, qwen3coder_request):
        """Single tool call with <tool_call> wrapper."""
        output = (
            "<tool_call>\n<function=get_current_weather>\n"
            "<parameter=city>\nDallas\n</parameter>\n"
            "<parameter=state>\nTX\n</parameter>\n"
            "<parameter=unit>\nfahrenheit\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["name"] == "get_current_weather"
        args = json.loads(tc["arguments"])
        assert args == {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}

    def test_single_tool_with_content(self, qwen3coder_parser, qwen3coder_request):
        """Content before tool call is preserved."""
        output = (
            "Sure! Let me check the weather for you."
            "<tool_call>\n<function=get_current_weather>\n"
            "<parameter=city>\nDallas\n</parameter>\n"
            "<parameter=state>\nTX\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.content == "Sure! Let me check the weather for you."

    def test_parallel_tools(self, qwen3coder_parser, qwen3coder_request):
        """Multiple parallel tool calls."""
        output = (
            "<tool_call>\n<function=get_current_weather>\n"
            "<parameter=city>\nDallas\n</parameter>\n"
            "<parameter=state>\nTX\n</parameter>\n"
            "</function>\n</tool_call>\n"
            "<tool_call>\n<function=get_current_weather>\n"
            "<parameter=city>\nOrlando\n</parameter>\n"
            "<parameter=state>\nFL\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        args0 = json.loads(result.tool_calls[0]["arguments"])
        args1 = json.loads(result.tool_calls[1]["arguments"])
        assert args0["city"] == "Dallas"
        assert args1["city"] == "Orlando"

    def test_type_conversion(self, qwen3coder_parser, qwen3coder_request):
        """Parameter type conversion based on tool schema."""
        output = (
            "<tool_call>\n<function=test_types>\n"
            "<parameter=int_param>\n42\n</parameter>\n"
            "<parameter=float_param>\n3.14\n</parameter>\n"
            "<parameter=bool_param>\ntrue\n</parameter>\n"
            "<parameter=str_param>\nhello world\n</parameter>\n"
            '<parameter=obj_param>\n{"key": "value"}\n</parameter>\n'
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["int_param"] == 42
        assert isinstance(args["int_param"], int)
        assert args["float_param"] == 3.14
        assert isinstance(args["float_param"], float)
        assert args["bool_param"] is True
        assert args["str_param"] == "hello world"
        assert args["obj_param"] == {"key": "value"}

    def test_object_with_single_quotes(self, qwen3coder_parser, qwen3coder_request):
        """Object parameter with single-quote JSON (Python literal)."""
        output = (
            "<tool_call>\n<function=test_types>\n"
            "<parameter=obj_param>\n{'key': 'value'}\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["obj_param"] == {"key": "value"}

    def test_array_parameter_double_encoded_json_string(self, qwen3coder_parser):
        """Array parameters may arrive as double-encoded JSON strings."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "todowrite",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "todos": {
                                    "type": "array",
                                    "items": {"type": "object"},
                                },
                            },
                        },
                    },
                }
            ]
        }
        output = (
            "<tool_call>\n<function=todowrite>\n"
            "<parameter=todos>\n"
            '"[{\\"content\\": \\"Initialize\\", \\"status\\": \\"in_progress\\"}]"\n'
            "</parameter>\n"
            "</function>\n</tool_call>"
        )

        result = qwen3coder_parser.extract_tool_calls(output, request)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert isinstance(args["todos"], list)
        assert args["todos"][0]["content"] == "Initialize"

    def test_array_parameter_nullable_type_list(self, qwen3coder_parser):
        """Schemas may encode nullable arrays as type lists."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "todowrite",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "todos": {
                                    "type": ["array", "null"],
                                    "items": {"type": "object"},
                                },
                            },
                        },
                    },
                }
            ]
        }
        output = (
            "<tool_call>\n<function=todowrite>\n"
            "<parameter=todos>\n"
            '"[{\\"content\\": \\"Initialize\\", \\"status\\": \\"in_progress\\"}]"\n'
            "</parameter>\n"
            "</function>\n</tool_call>"
        )

        result = qwen3coder_parser.extract_tool_calls(output, request)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert isinstance(args["todos"], list)
        assert args["todos"][0]["content"] == "Initialize"

    def test_fallback_no_tool_call_tags(self, qwen3coder_parser, qwen3coder_request):
        """Bare <function=...> without <tool_call> wrapper also works."""
        output = (
            "<function=get_current_weather>\n"
            "<parameter=city>\nDallas\n</parameter>\n"
            "<parameter=state>\nTX\n</parameter>\n"
            "</function>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_current_weather"

    def test_missing_closing_parameter_tag(self, qwen3coder_parser, qwen3coder_request):
        """Missing </parameter> tag — graceful handling."""
        output = (
            "<tool_call>\n<function=get_current_weather>\n"
            "<parameter=city>\nDallas\n"
            "<parameter=state>\nTX\n</parameter>\n"
            "<parameter=unit>\nfahrenheit\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0]["arguments"])
        assert "city" in args
        assert args["state"] == "TX"
        assert args["unit"] == "fahrenheit"

    def test_multiline_object_param(self, qwen3coder_parser, qwen3coder_request):
        """Object parameter spanning multiple lines."""
        output = (
            "<tool_call>\n<function=calculate_area>\n"
            "<parameter=shape>\nrectangle\n</parameter>\n"
            '<parameter=dimensions>\n{"width": 10, \n "height": 20}\n</parameter>\n'
            "<parameter=precision>\n2\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["shape"] == "rectangle"
        assert args["dimensions"] == {"width": 10, "height": 20}
        assert args["precision"] == 2

    def test_tool_with_content_and_typed_params(
        self, qwen3coder_parser, qwen3coder_request
    ):
        """Content before tool call with typed parameters."""
        output = (
            "Let me calculate that area for you."
            "<tool_call>\n<function=calculate_area>\n"
            "<parameter=shape>\ncircle\n</parameter>\n"
            '<parameter=dimensions>\n{"radius": 15.5}\n</parameter>\n'
            "<parameter=precision>\n3\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = qwen3coder_parser.extract_tool_calls(output, qwen3coder_request)
        assert result.tools_called
        assert result.content == "Let me calculate that area for you."
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["shape"] == "circle"
        assert args["dimensions"] == {"radius": 15.5}
        assert args["precision"] == 3


class TestQwen3CoderUpstreamStreaming:
    """Streaming tests ported from upstream vLLM."""

    def test_streaming_no_tools(self, qwen3coder_parser):
        """Regular text → content delta."""
        result = qwen3coder_parser.extract_tool_calls_streaming(
            previous_text="Hello",
            current_text="Hello world",
            delta_text=" world",
        )
        assert result is not None
        assert result["content"] == " world"

    def test_streaming_content_before_tool(self, qwen3coder_parser):
        """Content before tool call is streamed."""
        result = qwen3coder_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Let me check",
            delta_text="Let me check",
        )
        assert result is not None
        assert result["content"] == "Let me check"

    def test_streaming_full_tool_call_multistep(
        self, qwen3coder_parser, qwen3coder_request
    ):
        """Multi-step streaming: header → { → param → } across calls."""
        deltas = [
            "<tool_call>",
            "\n<function=get_current_weather>",
            "\n",
            "<parameter=city>Dallas</parameter>",
            "\n</function>",
            "\n</tool_call>",
        ]
        text = ""
        collected = []
        for d in deltas:
            prev = text
            text += d
            r = qwen3coder_parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=text,
                delta_text=d,
                request=qwen3coder_request,
            )
            if r:
                collected.append(r)

        names = [
            c["tool_calls"][0]["function"].get("name")
            for c in collected
            if "tool_calls" in c and "name" in c["tool_calls"][0].get("function", {})
        ]
        assert "get_current_weather" in names

        arg_parts = [
            c["tool_calls"][0]["function"]["arguments"]
            for c in collected
            if "tool_calls" in c
            and "arguments" in c["tool_calls"][0].get("function", {})
        ]
        full_args = "".join(arg_parts)
        assert full_args.startswith("{")
        assert full_args.endswith("}")
        parsed = json.loads(full_args)
        assert parsed["city"] == "Dallas"

    def test_streaming_array_parameter_nullable_type_list(self, qwen3coder_parser):
        """Streaming conversion also handles nullable array schemas."""
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "todowrite",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "todos": {
                                    "type": ["array", "null"],
                                    "items": {"type": "object"},
                                },
                            },
                        },
                    },
                }
            ]
        }
        deltas = [
            "<tool_call>\n<function=todowrite>\n",
            "<parameter=todos>\n",
            '"[{\\"content\\": \\"Initialize\\", \\"status\\": \\"in_progress\\"}]"\n'
            "</parameter>\n",
            "</function>\n</tool_call>",
        ]
        text = ""
        collected = []
        for delta in deltas:
            previous = text
            text += delta
            result = qwen3coder_parser.extract_tool_calls_streaming(
                previous_text=previous,
                current_text=text,
                delta_text=delta,
                request=request,
            )
            if result:
                collected.append(result)

        arg_parts = [
            chunk["tool_calls"][0]["function"]["arguments"]
            for chunk in collected
            if "tool_calls" in chunk
            and "arguments" in chunk["tool_calls"][0].get("function", {})
        ]
        args = json.loads("".join(arg_parts))
        assert isinstance(args["todos"], list)
        assert args["todos"][0]["content"] == "Initialize"

    def test_streaming_coarse_deltas_complete(
        self, qwen3coder_parser, qwen3coder_request
    ):
        """Single coarse delta with complete tool call → full args emitted."""
        deltas = [
            "<tool_call>\n<function=get_current_weather>"
            "\n<parameter=city>Dallas</parameter>\n</function>"
            "\n</tool_call>",
        ]
        text = ""
        collected = []
        for d in deltas:
            prev = text
            text += d
            r = qwen3coder_parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=text,
                delta_text=d,
                request=qwen3coder_request,
            )
            if r:
                collected.append(r)

        tc_chunks = [c for c in collected if "tool_calls" in c]
        assert len(tc_chunks) >= 1
        first_tc = tc_chunks[0]["tool_calls"][0]
        assert first_tc["function"]["name"] == "get_current_weather"
        args = first_tc["function"]["arguments"]
        assert args
        parsed = json.loads(args)
        assert parsed["city"] == "Dallas"


# ═══════════════════════════════════════════════════════════════════════
# Registration tests — verify all new parsers are discoverable
# ═══════════════════════════════════════════════════════════════════════


class TestNewParserRegistration:
    """Verify new parsers are registered and discoverable."""

    @pytest.mark.parametrize(
        "name",
        [
            "seed_oss",
            "seed",
            "gpt_oss",
            "deepseek_v31",
            "deepseek_r1_0528",
            "qwen3_coder_xml",
            "qwen3_xml",
        ],
    )
    def test_parser_registered(self, name):
        """Parser name should be in the registry."""
        cls = ToolParserManager.get_tool_parser(name)
        assert cls is not None

    @pytest.mark.parametrize(
        "name",
        ["seed_oss", "deepseek_v31", "qwen3_coder_xml"],
    )
    def test_parser_instantiation(self, name):
        """Parser should instantiate without tokenizer."""
        cls = ToolParserManager.get_tool_parser(name)
        parser = cls(tokenizer=None)
        assert parser is not None

    @pytest.mark.parametrize(
        "name",
        ["seed_oss", "deepseek_v31", "qwen3_coder_xml"],
    )
    def test_parser_supports_native_format(self, name):
        """All new parsers should support native tool format."""
        cls = ToolParserManager.get_tool_parser(name)
        assert cls.supports_native_format() is True
