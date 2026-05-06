# SPDX-License-Identifier: Apache-2.0
"""Tests for tool_calling.py"""

import json
from unittest.mock import MagicMock

from vllm_mlx.api.tool_calling import (
    _is_tool_call_json,
    _parse_raw_json_tool_calls,
    convert_tools_for_template,
    format_tool_call_for_message,
    parse_tool_calls,
    validate_json_schema,
)


class TestIsToolCallJson:
    """Tests for _is_tool_call_json function."""

    def test_valid_tool_call_with_dict_arguments(self):
        """Test valid tool call with dict arguments."""
        obj = {"name": "get_weather", "arguments": {"city": "NYC"}}
        assert _is_tool_call_json(obj) is True

    def test_valid_tool_call_with_string_arguments(self):
        """Test valid tool call with string arguments."""
        obj = {"name": "get_weather", "arguments": '{"city": "NYC"}'}
        assert _is_tool_call_json(obj) is True

    def test_valid_tool_call_with_empty_arguments(self):
        """Test valid tool call with empty dict arguments."""
        obj = {"name": "get_weather", "arguments": {}}
        assert _is_tool_call_json(obj) is True

    def test_missing_name_key(self):
        """Test object missing 'name' key."""
        obj = {"arguments": {"city": "NYC"}}
        assert _is_tool_call_json(obj) is False

    def test_missing_arguments_key(self):
        """Test object missing 'arguments' key."""
        obj = {"name": "get_weather"}
        assert _is_tool_call_json(obj) is False

    def test_name_not_string(self):
        """Test object with non-string name."""
        obj = {"name": 123, "arguments": {"city": "NYC"}}
        assert _is_tool_call_json(obj) is False

    def test_name_empty_string(self):
        """Test object with empty string name."""
        obj = {"name": "", "arguments": {"city": "NYC"}}
        assert _is_tool_call_json(obj) is False

    def test_name_whitespace_only(self):
        """Test object with whitespace-only name."""
        obj = {"name": "   ", "arguments": {"city": "NYC"}}
        assert _is_tool_call_json(obj) is False

    def test_arguments_not_dict_or_string(self):
        """Test object with invalid arguments type."""
        obj = {"name": "get_weather", "arguments": [1, 2, 3]}
        assert _is_tool_call_json(obj) is False

    def test_arguments_is_list(self):
        """Test object with list arguments."""
        obj = {"name": "get_weather", "arguments": ["arg1", "arg2"]}
        assert _is_tool_call_json(obj) is False

    def test_arguments_is_number(self):
        """Test object with number arguments."""
        obj = {"name": "get_weather", "arguments": 123}
        assert _is_tool_call_json(obj) is False

    def test_input_not_dict(self):
        """Test non-dict input."""
        assert _is_tool_call_json("string") is False
        assert _is_tool_call_json([1, 2, 3]) is False
        assert _is_tool_call_json(None) is False
        assert _is_tool_call_json(123) is False

    def test_regular_json_not_tool_call(self):
        """Test that regular JSON with name field is not treated as tool call."""
        obj = {"name": "John", "age": 30}
        assert _is_tool_call_json(obj) is False


class TestParseRawJsonToolCalls:
    """Tests for _parse_raw_json_tool_calls function."""

    def test_empty_string(self):
        """Test empty string returns None."""
        assert _parse_raw_json_tool_calls("") is None

    def test_none_input(self):
        """Test None input returns None."""
        assert _parse_raw_json_tool_calls(None) is None

    def test_single_json_object(self):
        """Test single JSON object tool call."""
        text = '{"name": "get_weather", "arguments": {"city": "NYC"}}'
        result = _parse_raw_json_tool_calls(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["arguments"] == {"city": "NYC"}

    def test_json_array(self):
        """Test JSON array with multiple tool calls."""
        text = '[{"name": "func1", "arguments": {"a": 1}}, {"name": "func2", "arguments": {"b": 2}}]'
        result = _parse_raw_json_tool_calls(text)
        assert result is not None
        assert len(result) == 2
        assert result[0]["name"] == "func1"
        assert result[1]["name"] == "func2"

    def test_json_array_with_non_tool_call_objects(self):
        """Test JSON array filters to only valid tool calls."""
        text = '[{"name": "John", "age": 30}, {"name": "func", "arguments": {}}]'
        result = _parse_raw_json_tool_calls(text)
        # Only valid tool call objects are returned
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "func"

    def test_multiple_objects_separated_by_comma(self):
        """Test multiple JSON objects separated by commas."""
        text = '{"name": "func1", "arguments": {"a": 1}}, {"name": "func2", "arguments": {"b": 2}}'
        result = _parse_raw_json_tool_calls(text)
        assert result is not None
        assert len(result) == 2

    def test_text_with_leading_whitespace(self):
        """Test text with leading/trailing whitespace."""
        text = '   {"name": "func", "arguments": {}}   '
        result = _parse_raw_json_tool_calls(text)
        assert result is not None
        assert len(result) == 1

    def test_text_with_regular_json(self):
        """Test text with regular JSON (not tool calls) returns None."""
        text = '{"name": "John", "age": 30}'
        result = _parse_raw_json_tool_calls(text)
        assert result is None

    def test_text_with_mixed_content(self):
        """Test text with mixed valid and invalid JSON objects."""
        text = (
            '{"name": "func1", "arguments": {}} some text {"name": "John", "age": 30}'
        )
        result = _parse_raw_json_tool_calls(text)
        # Should only return the valid tool call
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "func1"

    def test_text_with_json_file_content(self):
        """Tool JSON with braces inside a string argument is still extracted."""
        text = (
            'Now write this file: {"name": "write_file", "arguments": {'
            '"path": "/tmp/tsconfig.json", '
            '"content": "{\\n  \\"compilerOptions\\": {\\n    \\"strict\\": true\\n  }\\n}\\n"'
            "}}"
        )
        result = _parse_raw_json_tool_calls(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "write_file"
        assert '"compilerOptions"' in result[0]["arguments"]["content"]

    def test_arguments_extracted_as_dict(self):
        """Test that arguments are extracted as dict when present."""
        text = '{"name": "func", "arguments": {"key": "value", "num": 42}}'
        result = _parse_raw_json_tool_calls(text)
        assert result is not None
        assert result[0]["arguments"] == {"key": "value", "num": 42}

    def test_empty_arguments(self):
        """Test tool call with empty arguments dict."""
        text = '{"name": "func", "arguments": {}}'
        result = _parse_raw_json_tool_calls(text)
        assert result is not None
        assert result[0]["arguments"] == {}


class TestParseToolCalls:
    """Tests for parse_tool_calls function."""

    def test_qwen_style_tool_call(self):
        """Test Qwen-style XML tool call."""
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>Some text'
        cleaned, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert "city" in tool_calls[0].function.arguments

    def test_llama_style_tool_call(self):
        """Test Llama-style XML tool call."""
        text = '<function=get_weather>{"city": "NYC"}</function>Some text'
        cleaned, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"

    def test_nemotron_style_tool_call(self):
        """Test Nemotron-style XML tool call."""
        text = "<tool_call><function=get_weather><parameter=city>NYC</parameter></function></tool_call>"
        cleaned, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert "city" in tool_calls[0].function.arguments

    def test_qwen3_bracket_style(self):
        """Test Qwen3 bracket-style tool call."""
        text = '[Calling tool: get_weather({"city": "NYC"})] Some response'
        cleaned, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert "city" in tool_calls[0].function.arguments

    def test_multiple_tool_calls(self):
        """Test multiple tool calls in same text."""
        text = '<tool_call>{"name": "func1", "arguments": {"a": 1}}</tool_call><tool_call>{"name": "func2", "arguments": {"b": 2}}</tool_call>'
        cleaned, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert len(tool_calls) == 2

    def test_no_tool_calls(self):
        """Test text with no tool calls returns None."""
        text = "This is just regular text without any tool calls."
        cleaned, tool_calls = parse_tool_calls(text)

        assert tool_calls is None
        assert cleaned == text

    def test_cleaned_text_removes_tags(self):
        """Test that cleaned text has tool call tags removed."""
        text = '<tool_call>{"name": "func", "arguments": {}}</tool_call>Hello world'
        cleaned, tool_calls = parse_tool_calls(text)

        assert "Hello world" in cleaned
        assert "<tool_call>" not in cleaned

    def test_raw_json_fallback(self):
        """Test raw JSON fallback when no XML tags matched."""
        text = '{"name": "get_weather", "arguments": {"city": "NYC"}}'
        cleaned, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"

    def test_raw_command_json_infers_matching_tool(self):
        """Bare command JSON maps to the requested shell tool schema."""
        text = '[{"command": "bun install express @types/express cors"}]'
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                        },
                    },
                }
            ]
        }

        _, tool_calls = parse_tool_calls(text, request=request)

        assert tool_calls is not None
        assert tool_calls[0].function.name == "bash"
        assert "bun install express" in tool_calls[0].function.arguments

    def test_function_equals_tool_call_variant(self):
        """Malformed function=name JSON calls are still parsed."""
        text = '<tool_call>function=bash({"command":"bun init\\n", "timeout": 30})]'

        _, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert tool_calls[0].function.name == "bash"
        assert "bun init" in tool_calls[0].function.arguments

    def test_function_equals_line_command_variant(self):
        """Malformed line-based function calls are still parsed."""
        text = "<tool_call>function=bash\nparameter=bash\ncommand=bun --version\n</parameter>"

        _, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert tool_calls[0].function.name == "bash"
        assert "bun --version" in tool_calls[0].function.arguments

    def test_tool_call_function_parameter_command_variant(self):
        """Malformed XML-ish function headers are still parsed."""
        text = (
            "<tool_call>function=bash>\n"
            "<parameter=command>\n"
            "ls -la\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

        _, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert tool_calls[0].function.name == "bash"
        assert "ls -la" in tool_calls[0].function.arguments

    def test_tool_call_function_parameter_command_without_angle_variant(self):
        """Malformed XML-ish function headers may omit the header terminator."""
        text = (
            "<tool_call>function=bash\n"
            "</parameter>\n"
            "<parameter=command>\n"
            "bun init --yes\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

        _, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert tool_calls[0].function.name == "bash"
        assert "bun init --yes" in tool_calls[0].function.arguments

    def test_parameter_name_commands_variant(self):
        """Malformed parameter-name command arrays are still parsed."""
        text = (
            "<parameter_name>\nbash\n</parameter_name>\n"
            "<parameter=commands>\n"
            '["bun add express @types/express"]\n'
            "</parameter>"
        )

        _, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert tool_calls[0].function.name == "bash"
        assert "bun add express" in tool_calls[0].function.arguments

    def test_parameter_name_command_variant(self):
        """Malformed parameter=name command blocks are still parsed."""
        text = (
            "<parameter=name>\n"
            "bash\n"
            "</parameter>\n"
            "<parameter=command>\n"
            "bun add express && bun add -d @types/express typescript tsx\n"
            "</parameter>"
        )

        _, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert tool_calls[0].function.name == "bash"
        assert "bun add express" in tool_calls[0].function.arguments

    def test_parameter_name_attribute_command_variant(self):
        """Malformed parameter name attributes are still parsed."""
        text = (
            "<tool_call>user\n"
            '<parameter name="bash">\n'
            '<parameter name="command">\n'
            "bun init --yes\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

        _, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert tool_calls[0].function.name == "bash"
        assert "bun init --yes" in tool_calls[0].function.arguments

    def test_parameter_name_attribute_edit_content_uses_write(self):
        """Malformed edit-with-content calls are routed to write when available."""
        text = (
            "[user\n"
            '<parameter name="edit">\n'
            "<parameter=path>\n"
            "/tmp/index.ts\n"
            "</parameter>\n"
            "<parameter=content>\n"
            'console.log("ok");\n'
            "</parameter>\n"
        )
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "edit",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "write",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                },
            ]
        }

        _, tool_calls = parse_tool_calls(text, request=request)

        assert tool_calls is not None
        assert tool_calls[0].function.name == "write"
        arguments = json.loads(tool_calls[0].function.arguments)
        assert arguments["path"] == "/tmp/index.ts"
        assert arguments["content"] == 'console.log("ok");'

    def test_tool_call_missing_parameter_terminators_variant(self):
        """XML-ish tool calls with missing tag terminators are still parsed."""
        text = (
            "<tool_call>function=write\n"
            "</parameter\n"
            "<parameter=path\n"
            "/tmp/tsconfig.json\n"
            "</parameter\n"
            "<parameter=content>\n"
            '{"compilerOptions":{"strict":true}}\n'
            "</parameter\n"
            "</tool_call>"
        )

        _, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert tool_calls[0].function.name == "write"
        arguments = json.loads(tool_calls[0].function.arguments)
        assert arguments["path"] == "/tmp/tsconfig.json"
        assert arguments["content"]["compilerOptions"]["strict"] is True

    def test_call_prefixed_tool_name_is_normalized(self):
        """Generated call_<tool> names are normalized to the requested tool."""
        text = "<tool_call>function=call_bash\n<parameter=command>\nls\n</parameter>"
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                        },
                    },
                }
            ]
        }

        _, tool_calls = parse_tool_calls(text, request=request)

        assert tool_calls is not None
        assert tool_calls[0].function.name == "bash"

    def test_calling_tool_parameter_variant(self):
        """Calling-tool text with parameter tags is parsed."""
        text = (
            "[Calling tool: edit\n"
            "<parameter=path>\n"
            "/tmp/index.ts\n"
            "</parameter>\n"
            "<parameter=edits>\n"
            '[{"oldText":"a","newText":"b"}]\n'
            "</parameter>\n"
        )

        _, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert tool_calls[0].function.name == "edit"
        arguments = json.loads(tool_calls[0].function.arguments)
        assert arguments["path"] == "/tmp/index.ts"
        assert arguments["edits"][0]["newText"] == "b"

    def test_prefixed_tool_write_variant(self):
        """Malformed _tool-prefixed calls are still parsed."""
        text = '_tool: write({"path": "/tmp/index.ts", "content": "console.log(\\"ok\\")"})]'

        _, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert tool_calls[0].function.name == "write"
        assert "/tmp/index.ts" in tool_calls[0].function.arguments

    def test_invalid_json_in_tags_not_parsed_as_tool_call(self):
        """Test that JSON without name+arguments is not parsed as a tool call."""
        text = '<tool_call>{"invalid": "json"}</tool_call>'
        cleaned, tool_calls = parse_tool_calls(text)

        # JSON missing "name" and "arguments" keys is not a valid tool call
        assert tool_calls is None

    def test_with_request_context(self):
        """Test parse_tool_calls with request context (currently unused but tests the param)."""
        text = '<tool_call>{"name": "func", "arguments": {}}</tool_call>'
        request = {"temperature": 0.5}
        cleaned, tool_calls = parse_tool_calls(text, request=request)

        assert tool_calls is not None

    def test_schema_string_arguments_serialize_objects(self):
        """Object values for string parameters are serialized before OpenAI output."""
        text = (
            '{"name": "write_file", "arguments": '
            '{"path": "/tmp/tsconfig.json", "content": {"compilerOptions": {"strict": true}}}}'
        )
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                }
            ]
        }

        _, tool_calls = parse_tool_calls(text, request=request)

        assert tool_calls is not None
        arguments = tool_calls[0].function.arguments
        assert '"content": "{\\"compilerOptions\\": {\\"strict\\": true}}"' in arguments

    def test_nested_schema_extra_properties_are_pruned(self):
        """Nested generated extras are dropped before OpenAI output."""
        text = (
            '<tool_call>{"name":"edit","arguments":{"path":"/tmp/index.ts",'
            '"edits":[{"oldText":"a","newText":"b","path":"/tmp/index.ts"}]}}'
            "</tool_call>"
        )
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "edit",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "edits": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "oldText": {"type": "string"},
                                            "newText": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                }
            ]
        }

        _, tool_calls = parse_tool_calls(text, request=request)

        assert tool_calls is not None
        arguments = json.loads(tool_calls[0].function.arguments)
        assert arguments["path"] == "/tmp/index.ts"
        assert arguments["edits"] == [{"oldText": "a", "newText": "b"}]


class TestConvertToolsForTemplate:
    """Tests for convert_tools_for_template function."""

    def test_none_input(self):
        """Test None input returns None."""
        assert convert_tools_for_template(None) is None

    def test_empty_list(self):
        """Test empty list returns None."""
        assert convert_tools_for_template([]) is None

    def test_openai_format_dict(self):
        """Test conversion from OpenAI format dict."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        result = convert_tools_for_template(tools)

        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather for a city"

    def test_openai_format_with_multiple_tools(self):
        """Test conversion with multiple tools."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "func1",
                    "description": "Description 1",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "func2",
                    "description": "Description 2",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        result = convert_tools_for_template(tools)

        assert result is not None
        assert len(result) == 2

    def test_non_function_type_filtered(self):
        """Test that non-function types are filtered out."""
        tools = [
            {
                "type": "function",
                "function": {"name": "func1", "description": "", "parameters": {}},
            },
            {"type": "unknown", "data": "something"},
        ]
        result = convert_tools_for_template(tools)

        assert result is not None
        assert len(result) == 1

    def test_pydantic_model_input(self):
        """Test conversion with Pydantic model-like objects."""
        # Create mock Pydantic-like objects
        mock_function = MagicMock()
        mock_function.name = "get_weather"
        mock_function.description = "Get weather"
        mock_function.parameters = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        }

        mock_tool = MagicMock()
        mock_tool.type = "function"
        mock_tool.function = mock_function

        result = convert_tools_for_template([mock_tool])

        assert result is not None
        assert len(result) == 1

    def test_missing_function_key(self):
        """Test tool dict without function key is handled."""
        tools = [{"type": "function"}]
        result = convert_tools_for_template(tools)

        # Should return None or empty list since no valid function
        assert result is None or result == []

    def test_default_parameters(self):
        """Test that default parameters are used when missing."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "func"
                    # No description or parameters
                },
            }
        ]
        result = convert_tools_for_template(tools)

        assert result is not None
        assert result[0]["function"]["name"] == "func"
        assert result[0]["function"]["description"] == ""


class TestFormatToolCallForMessage:
    """Tests for format_tool_call_for_message function."""

    def test_format_tool_call(self):
        """Test basic ToolCall formatting."""
        # Create mock ToolCall
        mock_function = MagicMock()
        mock_function.name = "get_weather"
        mock_function.arguments = '{"city": "NYC"}'

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_abc123"
        mock_tool_call.type = "function"
        mock_tool_call.function = mock_function

        result = format_tool_call_for_message(mock_tool_call)

        assert result["id"] == "call_abc123"
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["arguments"] == '{"city": "NYC"}'

    def test_format_with_complex_arguments(self):
        """Test formatting with complex arguments."""
        mock_function = MagicMock()
        mock_function.name = "search"
        mock_function.arguments = (
            '{"query": "weather", "limit": 10, "filters": {"type": "news"}}'
        )

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_xyz789"
        mock_tool_call.type = "function"
        mock_tool_call.function = mock_function

        result = format_tool_call_for_message(mock_tool_call)

        assert result["function"]["name"] == "search"
        assert "query" in result["function"]["arguments"]


class TestValidateJsonSchema:
    """Tests for validate_json_schema function."""

    def test_valid_simple_object(self):
        """Test valid JSON against simple schema."""
        data = {"name": "John", "age": 30}
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        }
        is_valid, error = validate_json_schema(data, schema)

        assert is_valid is True
        assert error is None

    def test_valid_nested_object(self):
        """Test valid JSON against nested schema."""
        data = {"user": {"name": "John", "address": {"city": "NYC", "zip": "10001"}}}
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "zip": {"type": "string"},
                            },
                        },
                    },
                }
            },
        }
        is_valid, error = validate_json_schema(data, schema)

        assert is_valid is True

    def test_invalid_type(self):
        """Test invalid type fails validation."""
        data = {"name": "John", "age": "thirty"}  # age should be number
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        }
        is_valid, error = validate_json_schema(data, schema)

        assert is_valid is False
        assert error is not None
