# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Llama tool parser's bare-JSON + python-tag
support — fixes issue #700 ("F-008").

The Llama 3.1/3.2 official chat template trains the model to emit
assistant tool calls as bare ``{"name": "X", "parameters": {...}}``
JSON, with an optional ``<|python_tag|>`` prefix when the system header
advertises ``Environment: ipython``. Before the fix, the parser only
matched the Llama-4-style ``<function=name>...</function>`` XML wrapper
and therefore let the entire JSON leak into ``message.content`` — see
the issue body for the full repro.

These tests pin the three shapes (XML / python-tag / bare JSON), the
``parameters`` ↔ ``arguments`` key alias, the streaming contract, and a
small set of negative cases that previously could have been misrouted
as tool calls (prose JSON, partial fragments, etc.).
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.tool_parsers import LlamaToolParser


@pytest.fixture
def parser() -> LlamaToolParser:
    return LlamaToolParser()


# ---------------------------------------------------------------------------
# Shape 1 — XML wrapper (Llama 4 / vLLM style), preserved by the fix.
# ---------------------------------------------------------------------------


class TestXmlWrapperShape:
    def test_single_call(self, parser: LlamaToolParser):
        text = '<function=multiply>{"x": 3, "y": 4}</function>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "multiply"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"x": 3, "y": 4}

    def test_multiple_calls(self, parser: LlamaToolParser):
        text = (
            '<function=add>{"a": 1}</function>'
            '<function=multiply>{"x": 3}</function>'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert [tc["name"] for tc in result.tool_calls] == ["add", "multiply"]

    def test_content_before(self, parser: LlamaToolParser):
        text = 'Computing result<function=calc>{"n": 5}</function>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.content == "Computing result"


# ---------------------------------------------------------------------------
# Shape 2 — <|python_tag|>{...} (Llama 3.1 ipython mode).
# ---------------------------------------------------------------------------


class TestPythonTagShape:
    def test_basic(self, parser: LlamaToolParser):
        text = (
            '<|python_tag|>{"name": "web_search", '
            '"parameters": {"query": "你好"}}'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "web_search"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"query": "你好"}
        assert result.content is None

    def test_tag_with_preceding_prose(self, parser: LlamaToolParser):
        text = (
            'Sure, let me search.<|python_tag|>'
            '{"name": "web_search", "parameters": {"query": "weather"}}'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.content == "Sure, let me search."

    def test_tag_with_arguments_alias(self, parser: LlamaToolParser):
        # Some quantizations drift to OpenAI's "arguments" key under
        # fine-tuning. We accept both so users don't lose tool routing.
        text = (
            '<|python_tag|>{"name": "get_weather", '
            '"arguments": {"city": "Tokyo"}}'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Tokyo"}


# ---------------------------------------------------------------------------
# Shape 3 — bare JSON (Llama 3.1/3.2 default). This is the F-008 root
# cause: every test here would have failed against the pre-fix parser.
# ---------------------------------------------------------------------------


class TestBareJsonShape:
    def test_f008_repro_chinese_greeting(self, parser: LlamaToolParser):
        """Exact wire string from issue #700 / F-008 screenshot."""
        text = '{"name": "web_search", "parameters": {"query": "你好"}}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called, (
            "Bare JSON tool-call leaked as content — F-008 regression"
        )
        assert result.tool_calls[0]["name"] == "web_search"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"query": "你好"}
        assert result.content is None

    def test_bare_json_with_arguments_alias(self, parser: LlamaToolParser):
        text = '{"name": "calc", "arguments": {"n": 5}}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {"n": 5}

    def test_bare_json_no_args(self, parser: LlamaToolParser):
        # No-arg tool calls are legitimate (``get_current_time()``).
        text = '{"name": "get_current_time"}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {}

    def test_bare_json_with_nested_args(self, parser: LlamaToolParser):
        args = {"filter": {"city": "Tokyo", "limit": 10, "tags": ["a", "b"]}}
        text = '{"name": "search", "parameters": ' + json.dumps(args["filter"]) + "}"
        # Re-form full payload
        text = json.dumps({"name": "search", "parameters": args["filter"]})
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == args["filter"]

    def test_bare_json_preserves_prefix_content(self, parser: LlamaToolParser):
        text = (
            'Let me look that up. '
            '{"name": "web_search", "parameters": {"query": "weather"}}'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.content == "Let me look that up."

    def test_string_with_brace_inside(self, parser: LlamaToolParser):
        # The brace-balancer must be string-aware so a "}" inside a JSON
        # string doesn't terminate the object early.
        text = '{"name": "echo", "parameters": {"msg": "hello } world"}}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "msg": "hello } world"
        }


# ---------------------------------------------------------------------------
# Negative cases — must NOT misroute as tool calls.
# ---------------------------------------------------------------------------


class TestNoFalsePositives:
    def test_plain_prose_is_not_a_tool_call(self, parser: LlamaToolParser):
        text = "Hello! How can I help you today?"
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.content == text

    def test_prose_json_without_name_key(self, parser: LlamaToolParser):
        # Model handed back an object literal as an example — must stay
        # as content, not be routed to a tool call.
        text = 'Here is an example: {"city": "Tokyo", "temp": 22}'
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.content == text

    def test_json_with_empty_name(self, parser: LlamaToolParser):
        text = '{"name": "", "parameters": {}}'
        result = parser.extract_tool_calls(text)
        assert not result.tools_called

    def test_malformed_json_left_as_content(self, parser: LlamaToolParser):
        text = '{"name": "broken", "parameters": {oops'
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.content == text

    def test_empty_input(self, parser: LlamaToolParser):
        result = parser.extract_tool_calls("")
        assert not result.tools_called


# ---------------------------------------------------------------------------
# Streaming contract — clients see content tokens until the JSON closes,
# then a single tool_calls delta.
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_bare_json_streams_content_until_close(
        self, parser: LlamaToolParser
    ):
        # Mid-stream fragments must not emit tool_calls yet.
        partial = '{"name": "web_search", "parameters": {"query": "wea'
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=partial,
            delta_text=partial,
        )
        assert result is None or "tool_calls" not in result

    def test_bare_json_emits_tool_calls_on_close(
        self, parser: LlamaToolParser
    ):
        previous = '{"name": "web_search", "parameters": {"query": "weather"'
        delta = "}}"
        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
        )
        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "web_search"

    def test_python_tag_emits_on_close(self, parser: LlamaToolParser):
        previous = (
            '<|python_tag|>{"name": "web_search", '
            '"parameters": {"query": "weather"'
        )
        delta = "}}"
        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
        )
        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "web_search"

    def test_xml_wrapper_still_works_streaming(self, parser: LlamaToolParser):
        previous = '<function=add>{"a": 1}'
        delta = "</function>"
        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
        )
        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "add"

    def test_plain_content_streams_as_content(self, parser: LlamaToolParser):
        delta = "Hello there"
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=delta,
            delta_text=delta,
        )
        assert result == {"content": delta}


# ---------------------------------------------------------------------------
# Wire-format declaration — keep ``test_tool_parser_wire_formats`` happy
# and document the upgrade explicitly.
# ---------------------------------------------------------------------------


def test_expected_wire_formats_declared():
    fmts = LlamaToolParser.EXPECTED_WIRE_FORMATS
    assert "function_bare" in fmts
    assert "llama_python_tag" in fmts
    assert "raw_json" in fmts
