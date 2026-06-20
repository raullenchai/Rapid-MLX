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

    def test_mixed_tool_call_json_and_named_xml_both_emitted(self, parser):
        """Codex round-4 BLOCKING: ``<tool_call>{...}</tool_call>`` and
        ``<function><name>...</name>...</function>`` are structurally
        disjoint and both are valid tool-call shapes. A response that
        contains both must emit BOTH calls in wire order — silently
        dropping one is a correctness bug.

        The previous version of this test asserted only the JSON call
        was emitted (encoding the silent-drop bug as expected
        behavior); round-4 fix makes the named-XML matcher additive
        rather than gated by ``if not tool_calls``.
        """
        text = (
            '<tool_call>{"name":"get_weather","arguments":{"loc":"Paris"}}</tool_call>\n'
            "Some scratch reasoning…\n"
            "<function><name>get_news</name>"
            '<arguments>{"topic":"weather"}</arguments></function>'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 2, (
            "Both tool_call JSON and named-XML calls must be emitted "
            "(codex round-4 BLOCKING regression)."
        )
        names = [c["name"] for c in result.tool_calls]
        assert names == ["get_weather", "get_news"]
        args = [json.loads(c["arguments"]) for c in result.tool_calls]
        assert args == [{"loc": "Paris"}, {"topic": "weather"}]

    def test_mixed_bare_function_and_named_xml_both_emitted(self, parser):
        """Codex round-4 BLOCKING (companion): the bare-function
        Nemotron shape ``<function=NAME>...</function>`` and the
        VibeThinker named-XML shape ``<function><name>...</name>...
        </function>`` are also structurally disjoint and can co-occur.
        Both must be emitted."""
        text = (
            '<function=get_weather>{"loc":"Paris"}</function>\n'
            "<function><name>get_news</name>"
            '<arguments>{"topic":"weather"}</arguments></function>'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 2, (
            "Both bare-function and named-XML calls must be emitted "
            "(codex round-4 BLOCKING regression)."
        )
        names = [c["name"] for c in result.tool_calls]
        assert set(names) == {"get_weather", "get_news"}

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

        Codex round-2 BLOCKING refinement: the prior assertion accepted
        ``delta is None`` as "OK", which would have stayed green for the
        exact suppression bug we're catching. Tighten to require a
        non-``None`` content-only delta — that is the only correct
        behavior for streamed prose containing ``<function>``.
        """
        previous_text = "The tag is "
        current_text = "The tag is `<function>` — it is followed by `<name>`."
        delta_text = "`<function>` — it is followed by `<name>`."

        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
        )
        # Must NOT be None — suppression is the bug, content release is
        # the correct behavior. And must NOT contain ``tool_calls`` —
        # the prose is not a tool call.
        assert delta is not None, (
            "Streaming branch suppressed prose <function> literal — "
            "F-042 codex round-2 BLOCKING regression."
        )
        assert "tool_calls" not in delta, (
            "Streaming branch false-positive tool_calls on prose <function> literal."
        )
        # Sanity: non-streaming parse on the same text also yields no
        # tool calls (defense in depth).
        result = parser.extract_tool_calls(current_text)
        assert not result.tools_called
        assert result.tool_calls == []

    def test_partial_named_opener_chunks_held_until_disambiguation(self, parser):
        """Codex round-2 BLOCKING: ``<function>`` plus a partial ``<name``
        prefix (e.g. ``<function><n``, ``<function>\\n <na``) must hold
        the bytes back so they don't leak as content before the
        opener-claim branch fires.

        Once the next chunk completes ``<name`` the opener-claim branch
        takes over and the suppression ends. Once the next chunk
        disambiguates as prose (e.g. ``<function>!``) the partial-opener
        regex stops matching and the bytes are released via the
        safe-content path.
        """
        # Stage 1: ``<function><n`` — partial named opener. Suppress.
        delta = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="<function><n",
            delta_text="<function><n",
        )
        assert delta is None, (
            "Partial named opener <function><n must be held; leaking "
            "now would expose tool-call XML as content."
        )

        # Stage 2: ``<function>\n <na`` — partial named opener with
        # whitespace between tags. Still suppress.
        delta = parser.extract_tool_calls_streaming(
            previous_text="<function>\n <n",
            current_text="<function>\n <na",
            delta_text="a",
        )
        assert delta is None

        # Stage 3: ``<function>!`` — clearly NOT a named opener. Release
        # via the safe-content path (delta is non-None or content
        # eventually flushes; assert no tool_calls false positive).
        delta = parser.extract_tool_calls_streaming(
            previous_text="<function>",
            current_text="<function>!",
            delta_text="!",
        )
        if delta is not None:
            assert "tool_calls" not in delta

    def test_stream_ending_with_partial_opener_flushes_on_end(self, parser):
        """Codex round-3 BLOCKING regression:

        If a stream legitimately ends with a literal ``<function>`` (e.g.
        a doc paragraph about tool-call syntax that just so happens to
        finish on the tag), the partial-hold mechanism MUST release the
        held bytes at end-of-stream via ``flush_held_content``. Otherwise
        the user sees the closing bytes silently dropped.

        The round-3 fix folds the partial-opener regex into
        ``_safe_content_prefix`` so the existing
        ``flush_held_content`` (already in place for the literal
        sentinels) covers the variable-length partial-opener case too.
        """
        full_text = "The tag is `<function>"
        # During streaming, ``_emit_safe_content`` holds back the
        # ``<function>`` tail (it is a viable opener prefix). Once the
        # stream ends, ``flush_held_content`` must release the held
        # suffix — losing it would be a silent content drop.
        flushed = parser.flush_held_content(full_text)
        assert flushed == "<function>", (
            f"flush_held_content dropped the held <function> suffix; got "
            f"{flushed!r} — F-042 codex round-3 BLOCKING regression."
        )

    def test_stream_ending_with_partial_name_chunk_flushes_on_end(self, parser):
        """Variant of the round-3 BLOCKING regression: the stream ends
        mid-``<name`` token. The standard ``flush_held_content`` must
        release the held bytes — both the ``<function>`` portion and the
        partial ``<n``/``<na``/etc. suffix."""
        full_text = "Streaming ended early at <function><n"
        flushed = parser.flush_held_content(full_text)
        assert flushed == "<function><n", (
            f"flush_held_content dropped the held partial-named-opener "
            f"suffix; got {flushed!r}."
        )
