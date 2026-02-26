# SPDX-License-Identifier: Apache-2.0
"""Tests for MiniMax tool call parser."""

import json

import pytest

from vllm_mlx.tool_parsers import MiniMaxToolParser, ToolParserManager


class TestMiniMaxRegistration:
    """Test that MiniMax parser is registered correctly."""

    def test_registered_as_minimax(self):
        parser_cls = ToolParserManager.get_tool_parser("minimax")
        assert parser_cls is MiniMaxToolParser

    def test_registered_as_minimax_m2(self):
        parser_cls = ToolParserManager.get_tool_parser("minimax_m2")
        assert parser_cls is MiniMaxToolParser

    def test_instantiation(self):
        parser = MiniMaxToolParser()
        assert parser is not None


class TestExtractToolCalls:
    """Test non-streaming tool call extraction."""

    @pytest.fixture
    def parser(self):
        return MiniMaxToolParser()

    # -- Wrapped format: <minimax:tool_call>...<invoke>...</minimax:tool_call> --

    def test_single_tool_wrapped(self, parser):
        text = (
            '<minimax:tool_call>\n'
            '<invoke name="get_weather">\n'
            '<parameter name="city">Tokyo</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["city"] == "Tokyo"

    def test_multiple_params_wrapped(self, parser):
        text = (
            '<minimax:tool_call>\n'
            '<invoke name="search_web">\n'
            '<parameter name="query">MLX benchmarks</parameter>\n'
            '<parameter name="max_results">5</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search_web"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["query"] == "MLX benchmarks"
        assert args["max_results"] == 5  # JSON-parsed as int

    def test_multiple_invokes_in_one_block(self, parser):
        text = (
            '<minimax:tool_call>\n'
            '<invoke name="get_weather">\n'
            '<parameter name="city">Tokyo</parameter>\n'
            '</invoke>\n'
            '<invoke name="get_time">\n'
            '<parameter name="timezone">JST</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[1]["name"] == "get_time"

    def test_multiple_blocks(self, parser):
        text = (
            '<minimax:tool_call>\n'
            '<invoke name="func1">\n'
            '<parameter name="a">1</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>\n'
            '<minimax:tool_call>\n'
            '<invoke name="func2">\n'
            '<parameter name="b">2</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "func1"
        assert result.tool_calls[1]["name"] == "func2"

    def test_content_before_tool_call(self, parser):
        text = (
            'Let me check the weather for you.\n'
            '<minimax:tool_call>\n'
            '<invoke name="get_weather">\n'
            '<parameter name="city">Paris</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.content == "Let me check the weather for you."

    def test_think_tags_stripped_from_content(self, parser):
        text = (
            '<think>The user wants weather info</think>\n'
            '<minimax:tool_call>\n'
            '<invoke name="get_weather">\n'
            '<parameter name="city">NYC</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        # Think tags should be stripped from content
        assert result.content is None or "<think>" not in (result.content or "")

    # -- Bare invoke format (no <minimax:tool_call> wrapper) --

    def test_bare_invoke(self, parser):
        text = (
            '<invoke name="run_python">\n'
            '<parameter name="code">print("hello")</parameter>\n'
            '</invoke>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "run_python"
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["code"] == 'print("hello")'

    def test_bare_invoke_inside_think(self, parser):
        """Model sometimes emits tool calls inside <think> without wrapper."""
        text = (
            '<think>I should call this tool\n'
            '<invoke name="search">\n'
            '<parameter name="query">test</parameter>\n'
            '</invoke>\n'
            '</think>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    # -- Truncated / partial formats --

    def test_truncated_invoke_no_closing_tag(self, parser):
        """Streaming may end before </invoke> arrives."""
        text = (
            '<minimax:tool_call>\n'
            '<invoke name="get_weather">\n'
            '<parameter name="city">Tokyo</parameter>\n'
            # Missing </invoke> and </minimax:tool_call>
        )
        # Should still not crash — wrapper regex won't match but bare invoke partial will
        result = parser.extract_tool_calls(text)
        # The wrapped block regex won't match (no closing </minimax:tool_call>)
        # But bare invoke partial should kick in
        if result.tools_called:
            assert result.tool_calls[0]["name"] == "get_weather"

    def test_truncated_parameter_no_closing_tag(self, parser):
        """Parameter value without closing </parameter>."""
        text = (
            '<invoke name="search">\n'
            '<parameter name="query">partial value'
            # Missing </parameter> and </invoke>
        )
        result = parser.extract_tool_calls(text)

        if result.tools_called:
            args = json.loads(result.tool_calls[0]["arguments"])
            assert "partial value" in args["query"]

    # -- Edge cases --

    def test_no_tool_call(self, parser):
        text = "Hello, how can I help you today?"
        result = parser.extract_tool_calls(text)

        assert not result.tools_called
        assert result.content == text

    def test_empty_input(self, parser):
        result = parser.extract_tool_calls("")
        assert not result.tools_called

    def test_json_value_parameter(self, parser):
        """Parameter values that are valid JSON should be parsed."""
        text = (
            '<minimax:tool_call>\n'
            '<invoke name="func">\n'
            '<parameter name="count">42</parameter>\n'
            '<parameter name="flag">true</parameter>\n'
            '<parameter name="items">["a", "b"]</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["count"] == 42
        assert args["flag"] is True
        assert args["items"] == ["a", "b"]

    def test_unicode_in_parameters(self, parser):
        text = (
            '<minimax:tool_call>\n'
            '<invoke name="translate">\n'
            '<parameter name="text">日本語テスト</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert args["text"] == "日本語テスト"

    def test_empty_parameter_value_skipped(self, parser):
        """Empty parameter values should be skipped."""
        text = (
            '<invoke name="func">\n'
            '<parameter name="empty"></parameter>\n'
            '<parameter name="real">value</parameter>\n'
            '</invoke>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        args = json.loads(result.tool_calls[0]["arguments"])
        assert "empty" not in args
        assert args["real"] == "value"

    def test_invoke_without_params_skipped(self, parser):
        """Bare invoke with no parameters (hallucinated junk) should be skipped."""
        text = '<invoke name="hallucinated"></invoke>'
        result = parser.extract_tool_calls(text)

        assert not result.tools_called

    def test_tool_call_id_uniqueness(self, parser):
        text = (
            '<minimax:tool_call>\n'
            '<invoke name="func1">\n'
            '<parameter name="a">1</parameter>\n'
            '</invoke>\n'
            '<invoke name="func2">\n'
            '<parameter name="b">2</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>'
        )
        result = parser.extract_tool_calls(text)

        ids = [tc["id"] for tc in result.tool_calls]
        assert len(ids) == len(set(ids)), "Tool call IDs should be unique"
        assert all(id.startswith("call_") for id in ids)

    def test_whitespace_in_function_name(self, parser):
        """Function name with leading/trailing whitespace should be stripped."""
        text = (
            '<invoke name="  get_weather  ">\n'
            '<parameter name="city">NYC</parameter>\n'
            '</invoke>'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_escape_sequence_cleanup(self, parser):
        """Content after tool call should have [e~[ junk cleaned."""
        text = (
            'Some content\n'
            '<minimax:tool_call>\n'
            '<invoke name="func">\n'
            '<parameter name="x">1</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>\n'
            '[e~[extra junk'
        )
        result = parser.extract_tool_calls(text)

        assert result.tools_called
        assert "[e~[" not in (result.content or "")


class TestHasPendingToolCall:
    """Test has_pending_tool_call detection."""

    @pytest.fixture
    def parser(self):
        return MiniMaxToolParser()

    def test_has_tool_call_start(self, parser):
        assert parser.has_pending_tool_call("<minimax:tool_call>")
        assert parser.has_pending_tool_call('text <invoke name="func">')

    def test_no_tool_call(self, parser):
        assert not parser.has_pending_tool_call("regular text")
        assert not parser.has_pending_tool_call("")


class TestStreamingExtraction:
    """Test streaming tool call extraction."""

    @pytest.fixture
    def parser(self):
        return MiniMaxToolParser()

    def test_content_before_tool_call(self, parser):
        """Content before tool call start should pass through."""
        r = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Let me check that.",
            delta_text="Let me check that.",
        )
        assert r == {"content": "Let me check that."}

    def test_suppress_during_tool_call(self, parser):
        """Output suppressed while inside tool call block."""
        r = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text='<minimax:tool_call>\n<invoke name="func">',
            delta_text='<minimax:tool_call>\n<invoke name="func">',
        )
        assert r is None  # Suppressed

    def test_tool_call_emitted_on_close(self, parser):
        """Tool call emitted when closing tag arrives."""
        full = (
            '<minimax:tool_call>\n'
            '<invoke name="get_weather">\n'
            '<parameter name="city">Tokyo</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>'
        )
        prev = full.replace("</minimax:tool_call>", "")
        r = parser.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=full,
            delta_text="</minimax:tool_call>",
        )
        assert r is not None
        assert "tool_calls" in r
        assert r["tool_calls"][0]["function"]["name"] == "get_weather"
        args = json.loads(r["tool_calls"][0]["function"]["arguments"])
        assert args["city"] == "Tokyo"

    def test_streaming_token_by_token(self, parser):
        """Simulate realistic token-by-token streaming."""
        chunks = [
            "<minimax:tool_call>\n",
            '<invoke name="search">\n',
            '<parameter name="query">',
            "MLX performance",
            "</parameter>\n",
            "</invoke>\n",
            "</minimax:tool_call>",
        ]

        accumulated = ""
        result = None
        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            r = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=accumulated,
                delta_text=chunk,
            )
            if r is not None and "tool_calls" in r:
                result = r

        assert result is not None
        assert result["tool_calls"][0]["function"]["name"] == "search"

    def test_bare_invoke_streaming(self, parser):
        """Bare invoke (no wrapper) should also work in streaming."""
        chunks = [
            '<invoke name="run_code">\n',
            '<parameter name="code">print(1)</parameter>\n',
            '</invoke>',
        ]

        accumulated = ""
        result = None
        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            r = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=accumulated,
                delta_text=chunk,
            )
            if r is not None and "tool_calls" in r:
                result = r

        assert result is not None
        assert result["tool_calls"][0]["function"]["name"] == "run_code"

    def test_streaming_multiple_tool_calls(self, parser):
        """Two tool calls in sequence."""
        chunks = [
            '<minimax:tool_call>\n<invoke name="func1">\n',
            '<parameter name="a">1</parameter>\n</invoke>\n</minimax:tool_call>\n',
            '<minimax:tool_call>\n<invoke name="func2">\n',
            '<parameter name="b">2</parameter>\n</invoke>\n</minimax:tool_call>',
        ]

        accumulated = ""
        emitted = []
        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            r = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=accumulated,
                delta_text=chunk,
            )
            if r is not None and "tool_calls" in r:
                emitted.extend(r["tool_calls"])

        assert len(emitted) >= 2
        names = [tc["function"]["name"] for tc in emitted]
        assert "func1" in names
        assert "func2" in names
