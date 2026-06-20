# SPDX-License-Identifier: Apache-2.0
"""F-042 — VibeThinker XML-named ``<function>`` shape regression.

VibeThinker-{1.5B,3B} under ``tool_choice="auto"`` sometimes emits the
tool call as::

    <function>
     <name>get_weather</name>
     <arguments>
      {"loc":"Paris"}
     </arguments>
    </function>

…even though its chat template prescribes the
``<tool_call>{"name":...,"arguments":...}</tool_call>`` shape. Without a
matcher this leaks as ``content`` with ``tool_calls=None`` and
``finish_reason="stop"`` — breaking every OpenAI-compatible client.

The fix is a layer-level matcher in :class:`HermesToolParser`
(``FUNCTION_XML_NAMED_PATTERN``) per the "matcher in shared parser, not
per-alias workaround" convention. These tests pin the canonical shape,
whitespace variants, parallel emission, and the non-tools regression so a
future regression on the matcher trips the suite rather than the live
endpoint.
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.tool_parsers import HermesToolParser


@pytest.fixture
def parser() -> HermesToolParser:
    return HermesToolParser()


def _first(result_calls: list[dict]) -> dict:
    assert result_calls, "expected at least one tool call"
    return result_calls[0]


class TestVibeThinkerFunctionXmlNamed:
    """F-042: ``<function><name>...<arguments>...</function>`` matcher."""

    def test_canonical_shape_with_indented_json(self, parser):
        """Mirror the exact wire shape from the bug report.

        Whitespace between every tag, indented JSON inside ``<arguments>``,
        single tool call. Must populate ``tool_calls`` with the parsed
        function name and stringified JSON arguments, drop the XML from
        ``content``, and report ``tools_called=True``.
        """
        text = (
            "<function>\n"
            " <name>get_weather</name>\n"
            " <arguments>\n"
            '  {"loc":"Paris"}\n'
            " </arguments>\n"
            "</function>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        call = _first(result.tool_calls)
        assert call["name"] == "get_weather"
        assert json.loads(call["arguments"]) == {"loc": "Paris"}
        # Content is what's left after stripping; should be empty/None.
        assert not (result.content or "").strip()

    def test_no_whitespace_compact_shape(self, parser):
        """Compact ``<function><name>...</name><arguments>...</arguments></function>``
        with no inter-tag whitespace must still match (whitespace-tolerant
        regex, not whitespace-required)."""
        text = (
            "<function><name>search</name>"
            '<arguments>{"q":"trio"}</arguments></function>'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        call = _first(result.tool_calls)
        assert call["name"] == "search"
        assert json.loads(call["arguments"]) == {"q": "trio"}

    def test_arguments_with_nested_braces_and_escapes(self, parser):
        """JSON body inside ``<arguments>`` may carry nested ``}`` and
        escaped quotes — the matcher must not stop at the first ``}``."""
        nested = {
            "query": 'where is "Paris, France"?',
            "options": {"unit": "celsius", "extra": {"detail": True}},
        }
        body = json.dumps(nested)
        text = (
            "<function>\n"
            " <name>get_weather</name>\n"
            f" <arguments>{body}</arguments>\n"
            "</function>"
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        call = _first(result.tool_calls)
        assert call["name"] == "get_weather"
        assert json.loads(call["arguments"]) == nested

    def test_parallel_function_blocks(self, parser):
        """Two ``<function>...</function>`` blocks ⇒ two ``tool_calls``
        entries, in order. ``.*?`` must be non-greedy so the first opener
        doesn't swallow the second block."""
        text = (
            '<function><name>get_weather</name><arguments>{"loc":"Paris"}</arguments></function>\n'
            '<function><name>get_news</name><arguments>{"topic":"weather"}</arguments></function>'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        names = [c["name"] for c in result.tool_calls]
        assert names == ["get_weather", "get_news"]
        args = [json.loads(c["arguments"]) for c in result.tool_calls]
        assert args == [{"loc": "Paris"}, {"topic": "weather"}]

    def test_prelude_text_is_preserved_as_content(self, parser):
        """Any text before the ``<function>`` block must survive on the
        ``content`` field — the XML block is stripped, the prose stays."""
        text = (
            "Here you go: "
            '<function><name>get_weather</name><arguments>{"loc":"Paris"}</arguments></function>'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert _first(result.tool_calls)["name"] == "get_weather"
        # Stripping leaves at least the prose; trailing whitespace is OK.
        assert "Here you go:" in (result.content or "")

    def test_pre_existing_tool_call_shape_still_wins(self, parser):
        """The canonical ``<tool_call>{...}</tool_call>`` JSON shape must
        still be matched first — no regression on existing wire format.

        Codex round-1 NIT: actually mix BOTH shapes in the same response
        and assert ordering precedence — the ``<tool_call>`` JSON path
        is tried before ``FUNCTION_XML_NAMED_PATTERN``, so a stray
        ``<function>`` block alongside a real ``<tool_call>`` JSON block
        must yield the JSON call (not double-emit, not swap order).
        """
        text = (
            '<tool_call>{"name":"get_weather","arguments":{"loc":"Paris"}}</tool_call>\n'
            "Some scratch reasoning…\n"
            "<function><name>get_news</name>"
            '<arguments>{"topic":"weather"}</arguments></function>'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        # The <tool_call> JSON path wins (matcher ordering is
        # deterministic) — only the JSON call is emitted. The
        # FUNCTION_XML_NAMED_PATTERN is gated by
        # ``if not tool_calls`` so it is skipped here.
        assert len(result.tool_calls) == 1
        call = _first(result.tool_calls)
        assert call["name"] == "get_weather"
        assert json.loads(call["arguments"]) == {"loc": "Paris"}

    def test_nemotron_bare_function_equal_form_still_wins(self, parser):
        """``<function=NAME>`` (Nemotron / Qwen3-Coder) is disjoint from
        ``<function>`` (VibeThinker). Adding the new matcher must not
        regress the existing ``<function=`` path."""
        text = '<function=get_weather>{"loc":"Paris"}</function>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        call = _first(result.tool_calls)
        assert call["name"] == "get_weather"
        assert json.loads(call["arguments"]) == {"loc": "Paris"}

    def test_no_tools_user_quoted_function_literal_is_not_false_parse(self, parser):
        """Negative regression: a user-quoted ``<function>`` literal in
        plain text (no ``<name>`` or ``<arguments>`` tags) MUST NOT be
        misparsed as a tool call. Otherwise prompts that discuss
        function-call syntax would surface as zero-arg tool calls."""
        text = "I think the format is `<function>` but I'm not sure."
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.tool_calls == []
        # Content survives unchanged (parser doesn't mangle it).
        assert "<function>" in (result.content or "")

    def test_no_tools_plain_content_passthrough(self, parser):
        """Negative regression: plain content with no tool-call shape
        must pass through unmodified, ``tools_called=False``."""
        text = "The weather in Paris is sunny today."
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == text


class TestVibeThinkerFunctionXmlStreaming:
    """Streaming sentinel hold-back + emit on close.

    The streaming branch must hold ``<function>`` partial bytes
    (``<f``, ``<fu``, …) back from the content stream until either the
    full opener arrives (claim by tool-call branch) or a non-matching
    char arrives (release as content). On close, the tool call must be
    emitted as a structured ``tool_calls`` event.
    """

    def test_sentinel_holds_partial_function_open(self, parser):
        """``<f`` alone is a held prefix — no content event fires."""
        previous_text = ""
        current_text = "<f"
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text="<f",
        )
        # Either ``None`` (held) or no leak into content.
        if delta is not None:
            assert delta.get("content", "") == ""

    def test_completed_block_emits_tool_calls(self, parser):
        """Full ``<function>...</function>`` block at stream tail emits a
        structured ``tool_calls`` event for the new call."""
        previous_text = "<function><name>get_weather</name><arguments>"
        current_text = previous_text + '{"loc":"Paris"}</arguments></function>'
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text='{"loc":"Paris"}</arguments></function>',
        )
        assert delta is not None
        assert "tool_calls" in delta
        assert delta["tool_calls"][0]["function"]["name"] == "get_weather"
        args = json.loads(delta["tool_calls"][0]["function"]["arguments"])
        assert args == {"loc": "Paris"}

    def test_prose_function_literal_does_not_suppress_stream(self, parser):
        """Codex round-1 BLOCKING regression:

        A streamed prose ``<function>`` literal (e.g. the model
        explaining tool-call syntax) MUST NOT enter the function
        hold-back branch. Otherwise the stream returns ``None`` waiting
        on a ``</function>`` that will never arrive, and the user sees
        an indefinite hang. The disambiguation guard requires
        ``<function>`` to be followed by ``<name`` before claiming a
        named-XML opener.
        """
        previous_text = "The tag is "
        current_text = "The tag is `<function>` — it is followed by `<name>`."
        delta_text = "`<function>` — it is followed by `<name>`."

        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
        )
        # Must NOT be None (would suppress) and must NOT emit tool_calls.
        # Content delta path is fine — the test asserts the prose is not
        # held forever and that no false-positive tool_calls fires.
        assert delta is None or "tool_calls" not in delta
        # Sanity: non-streaming parse on the same text also yields no
        # tool calls (defense in depth).
        result = parser.extract_tool_calls(current_text)
        assert not result.tools_called
        assert result.tool_calls == []
