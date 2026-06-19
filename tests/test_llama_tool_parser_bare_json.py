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

    def test_bare_json_no_arg_uses_empty_parameters(
        self, parser: LlamaToolParser
    ):
        # No-arg tool calls render an explicit ``"parameters": {}`` in
        # the Llama 3.1/3.2 chat template (``arguments | tojson`` on an
        # empty dict). Bare ``{"name": "X"}`` with no args key is *not*
        # a tool call — it's prose JSON that happens to have a ``name``
        # field; see TestNoFalsePositives.
        text = '{"name": "get_current_time", "parameters": {}}'
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

    def test_prose_json_with_name_but_no_args(self, parser: LlamaToolParser):
        # Regression for codex r1 P2: a JSON object that *happens* to
        # carry a ``name`` field but no ``parameters``/``arguments`` key
        # must NOT be routed as a no-arg tool call — that's almost
        # always prose, not a Llama-template tool call (the template
        # always emits ``parameters``).
        text = 'Here is a user: {"name": "Alice", "age": 30}'
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

    def test_first_brace_token_is_pending_not_content(
        self, parser: LlamaToolParser
    ):
        """Regression for codex r1 P1: the very first ``{`` token of a
        streamed bare-JSON tool call must be treated as pending — we
        cannot wait for the ``"name"`` key to appear because that would
        leak the opening bytes as assistant content."""
        for delta in ("{", '{"', '{"na'):
            result = parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=delta,
                delta_text=delta,
            )
            # Must NOT pass through as content. Either None (still
            # buffering) or already a structured tool_calls dict — but
            # crucially not ``{"content": "{"}``.
            assert result is None or "content" not in result, (
                f"streaming leaked partial JSON prefix as content: "
                f"delta={delta!r} -> {result!r}"
            )

    def test_streaming_buffered_prefix_flushed_if_not_tool(
        self, parser: LlamaToolParser
    ):
        """Companion to test_first_brace_token_is_pending_not_content:
        if we buffered a ``{``-prefixed stream on the bet that it was a
        tool call, and the eventual close reveals it wasn't (no
        ``name``/``parameters`` pair), the buffered text MUST be
        flushed back to the client as content so we don't drop it."""
        # Mid-stream: still pending (no emit).
        previous = '{"city": "Tokyo", "temp"'
        mid = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=previous,
            delta_text=previous,
        )
        assert mid is None or "content" not in mid

        # Closing brace arrives — not a tool call → flush as content.
        delta = ": 22}"
        current = previous + delta
        final = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
        )
        assert final is not None
        assert "content" in final
        assert final["content"] == current

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

    def test_prose_prefix_then_bare_json_tool_call(
        self, parser: LlamaToolParser
    ):
        """Regression for codex r2 P2 (1/2): a streamed response of
        ``Let me check. {"name": "X", "parameters": {}}`` must NOT leak
        the JSON tail as content. Earlier prose chunks pass through; the
        JSON anchor onward is buffered and emitted as ``tool_calls``."""
        # Step 1: prose preface — passes through.
        d1 = "Let me check. "
        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=d1, delta_text=d1
        )
        assert r1 == {"content": d1}

        # Step 2: opening brace + name/params arrive in one go.
        d2 = '{"name": "search", "parameters": {}}'
        cur = d1 + d2
        r2 = parser.extract_tool_calls_streaming(
            previous_text=d1, current_text=cur, delta_text=d2
        )
        # The JSON closes in this delta → tool_calls emitted, no leak.
        assert r2 is not None
        assert "tool_calls" in r2
        assert r2["tool_calls"][0]["function"]["name"] == "search"

    def test_post_json_content_streams_through(self, parser: LlamaToolParser):
        """Regression for codex r2 P2 (2/2): after a (non-tool) JSON
        object is flushed as content, subsequent prose deltas MUST
        continue to stream through. Previously the parser kept seeing
        the buffered ``{`` prefix and returned ``None`` for every
        trailing token, dropping the rest of the reply."""
        # Step 1: closed prose JSON — flushed as content on close.
        d1 = '{"city": "Tokyo"}'
        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=d1, delta_text=d1
        )
        assert r1 is not None
        assert "content" in r1
        assert r1["content"] == d1

        # Step 2: trailing prose — must pass through.
        d2 = " is nice."
        cur = d1 + d2
        r2 = parser.extract_tool_calls_streaming(
            previous_text=d1, current_text=cur, delta_text=d2
        )
        assert r2 == {"content": d2}


# ---------------------------------------------------------------------------
# Wire-format declaration — keep ``test_tool_parser_wire_formats`` happy
# and document the upgrade explicitly.
# ---------------------------------------------------------------------------


def test_expected_wire_formats_declared():
    fmts = LlamaToolParser.EXPECTED_WIRE_FORMATS
    assert "function_bare" in fmts
    assert "llama_python_tag" in fmts
    assert "raw_json" in fmts
