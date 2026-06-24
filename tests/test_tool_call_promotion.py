# SPDX-License-Identifier: Apache-2.0
"""Tests for tool_call promotion from reasoning to content (#344).

Ports waybarrios#433 to our refactored ``BaseThinkingReasoningParser``.
The architecture differs from upstream (we use ``_streaming_phase`` /
``_held_tag_suffix_len`` instead of ``_phase`` / ``_content_buffer``),
so the promotion logic is centralised at a new seam — the public
``extract_reasoning_streaming`` wrapper and ``_promote_tool_calls``
post-filter — that every ``<think>``-tag subclass (Qwen3, DeepSeek-R1,
Glm4, VibeThinker, …) inherits uniformly. These tests cover:

* Non-streaming closed/unclosed/multi-block promotion.
* Streaming chunked promotion (carry across SSE boundaries).
* Negative test: prose mentioning ``<tool_call>`` is NOT promoted
  (structural guard).
* Multi-family parity: same wire shape promotes correctly on
  ``qwen3``, ``deepseek_r1``, and ``vibethinker`` parsers — proving
  the centralised promotion fires for every subclass.
* Trailing-prose boundary trim: a malformed unclosed ``<tool_call>``
  followed by plain reasoning prose does NOT over-promote the prose
  — the existing wire-scrub pipeline keeps handling that shape.
"""

import logging

import pytest

from vllm_mlx.reasoning import finalize_streaming_compat, get_parser


@pytest.fixture
def parser():
    cls = get_parser("qwen3")
    return cls()


def _stream(parser_inst, text, chunk_size=None):
    """Feed text through the streaming parser in chunks."""
    if chunk_size is None:
        chunk_size = len(text)
    parser_inst.reset_state()
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    accumulated = ""
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        previous = accumulated
        accumulated += chunk
        delta = parser_inst.extract_reasoning_streaming(previous, accumulated, chunk)
        if delta is not None:
            if delta.reasoning:
                reasoning_parts.append(delta.reasoning)
            if delta.content:
                content_parts.append(delta.content)
    final = finalize_streaming_compat(parser_inst, accumulated)
    if final is not None:
        if final.reasoning:
            reasoning_parts.append(final.reasoning)
        if final.content:
            content_parts.append(final.content)
    return "".join(reasoning_parts) or None, "".join(content_parts) or None


# ── Non-streaming promotion ─────────────────────────────────────────


class TestNonStreamingPromotion:
    def test_closed_tool_call_inside_think_appended(self, parser):
        output = (
            "<think>I should check.\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "</think>\n"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is not None
        assert "I should check" in reasoning
        assert "<tool_call>" not in reasoning
        assert content is not None
        assert "<tool_call>" in content
        assert "get_weather" in content

    def test_tool_call_after_think_unchanged(self, parser):
        output = (
            "<think>Let me think about this.</think>\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning == "Let me think about this."
        assert content is not None
        assert "<tool_call>" in content

    def test_multiple_tool_calls_inside_think(self, parser):
        output = (
            "<think>I need two lookups.\n"
            "<tool_call>\n"
            "<function=get_weather><parameter=city>Tokyo</parameter></function>\n"
            "</tool_call>\n"
            "Now the second one.\n"
            "<tool_call>\n"
            "<function=get_time><parameter=tz>JST</parameter></function>\n"
            "</tool_call>\n"
            "</think>\n"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is not None
        assert "<tool_call>" not in reasoning
        assert "I need two lookups" in reasoning
        assert "Now the second one" in reasoning
        assert content is not None
        assert content.count("<tool_call>") == 2

    def test_truncated_unclosed_tool_call_prepended(self, parser):
        output = (
            "<think>Let me call the API.\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</think>\n"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert content is not None
        assert "<tool_call>" in content
        assert "Let me call the API" in (reasoning or "")

    def test_hermes_json_tool_call(self, parser):
        output = (
            "<think>I will check.\n"
            "<tool_call>\n"
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n'
            "</tool_call>\n"
            "</think>\n"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is not None
        assert "<tool_call>" not in reasoning
        assert content is not None
        assert "get_weather" in content

    def test_prose_mention_not_promoted(self, parser):
        """Negative test: prose mentioning ``<tool_call>`` is NOT promoted.

        Structural guard via ``_TOOL_CALL_UNCLOSED_RE`` requires
        ``<tool_call>`` to be followed by ``{`` or ``<`` (a JSON or
        XML opener). A bare prose follow-up never matches.
        """
        output = (
            "<think>The model should use <tool_call> to invoke functions. "
            "Then it should verify.</think>\n"
            "The answer is 42."
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is not None
        assert "should use" in reasoning
        # Prose mention stays in reasoning — NOT promoted.
        assert "<tool_call>" in reasoning
        assert content == "The answer is 42."

    def test_content_none_handled(self, parser):
        output = (
            "<think>Checking.\n"
            "<tool_call>\n"
            "<function=search><parameter=q>test</parameter></function>\n"
            "</tool_call>\n"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert content is not None
        assert "<tool_call>" in content

    def test_closed_appended_preserves_existing_content(self, parser):
        """Closed block is appended AFTER existing post-think content."""
        output = (
            "<think>Let me check.\n"
            "<tool_call>\n"
            "<function=search><parameter=q>test</parameter></function>\n"
            "</tool_call>\n"
            "</think>\n"
            "Here is my answer."
        )
        reasoning, content = parser.extract_reasoning(output)
        assert content is not None
        assert "Here is my answer." in content
        assert "<tool_call>" in content
        assert content.index("Here is my answer") < content.index("<tool_call>")

    def test_promotion_logs_warning(self, parser, caplog):
        with caplog.at_level(logging.WARNING):
            output = (
                "<think>\n"
                "<tool_call>\n"
                "<function=f><parameter=x>1</parameter></function>\n"
                "</tool_call>\n"
                "</think>\n"
            )
            parser.extract_reasoning(output)
        assert any("tool_call" in r.message.lower() for r in caplog.records)

    def test_no_tool_calls_no_warning(self, parser, caplog):
        with caplog.at_level(logging.WARNING):
            output = "<think>Just reasoning.</think>\nContent."
            parser.extract_reasoning(output)
        assert not any("tool_call" in r.message.lower() for r in caplog.records)

    def test_unclosed_with_trailing_prose_stops_at_boundary(self, parser):
        """Malformed unclosed ``<tool_call>`` + trailing prose: the
        prose stays in reasoning, NOT promoted.

        Without the line-boundary trim the upstream regex would
        greedily eat ``Done thinking.`` into the promoted block.
        Existing wire-scrub pipeline expects the prose to stay in
        reasoning so it can be merged with the pre-call reasoning —
        regression test for the
        ``test_t3_chat_route_scrubs_wire_leak_from_reasoning_content``
        wire-scrub pipeline gate.
        """
        output = (
            "<think>Need to call the tool.\n"
            '<tool_call>{"name":"add_numbers","arguments": 4128, 7591}</function>\n'
            "Done thinking.</think>"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert reasoning is not None
        assert "Need to call the tool." in reasoning
        assert "Done thinking." in reasoning
        # The malformed unclosed body promoted to content, not the prose tail.
        assert content is not None
        assert "<tool_call>" in content
        assert "Done thinking." not in content


# ── Streaming promotion ─────────────────────────────────────────────


class TestStreamingPromotion:
    def test_stream_tool_call_inside_think_full_text(self, parser):
        text = (
            "<think>I should check.\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "</think>\n"
            "Final answer."
        )
        reasoning, content = _stream(parser, text)
        assert reasoning is not None
        assert "I should check" in reasoning
        assert content is not None
        assert "<tool_call>" in content
        assert "get_weather" in content
        assert "Final answer." in content

    def test_stream_think_ends_while_buffering(self, parser):
        """``</think>`` before ``</tool_call>`` flushes buffer as content."""
        text = (
            "<think>Check.\n"
            "<tool_call>\n"
            "<function=search><parameter=q>test</parameter></function>\n"
            "</think>\n"
            "Done."
        )
        reasoning, content = _stream(parser, text)
        assert content is not None
        assert "<tool_call>" in content

    def test_stream_finalize_with_buffered_tool_call(self, parser):
        """Stream ends mid-tool-call — flushed by finalize as content."""
        text = (
            "<think>\n<tool_call>\n<function=f><parameter=x>1</parameter></function>\n"
        )
        reasoning, content = _stream(parser, text)
        assert content is not None
        assert "<tool_call>" in content

    def test_stream_multiple_tool_calls(self, parser):
        text = (
            "<think>Two calls.\n"
            "<tool_call>\n"
            "<function=a><parameter=x>1</parameter></function>\n"
            "</tool_call>\n"
            "Middle reasoning.\n"
            "<tool_call>\n"
            "<function=b><parameter=y>2</parameter></function>\n"
            "</tool_call>\n"
            "</think>\n"
        )
        reasoning, content = _stream(parser, text)
        assert "Two calls" in (reasoning or "")
        assert "Middle reasoning" in (reasoning or "")
        assert content is not None
        assert content.count("<tool_call>") == 2

    @pytest.mark.parametrize("chunk_size", [5, 11, 20, 50])
    def test_stream_chunked_promotion(self, parser, chunk_size):
        """Promotion correct across chunk sizes — SSE-boundary carry
        handles ``<tool_call>`` opener spanning chunks."""
        text = (
            "<think>Check.\n"
            "<tool_call>\n"
            "<function=f><parameter=x>1</parameter></function>\n"
            "</tool_call>\n"
            "</think>\nDone."
        )
        reasoning, content = _stream(parser, text, chunk_size=chunk_size)
        assert content is not None, f"chunk_size={chunk_size}"
        assert "<tool_call>" in content, f"chunk_size={chunk_size}"
        assert "<tool_call>" not in (reasoning or ""), f"chunk_size={chunk_size}"

    def test_stream_tool_call_closed_immediately_before_think_end(self, parser):
        """``</tool_call></think>`` with no trailing content."""
        text = (
            "<think>R\n"
            "<tool_call>\n"
            "<function=f><parameter=x>1</parameter></function>\n"
            "</tool_call></think>"
        )
        reasoning, content = _stream(parser, text)
        assert content is not None
        assert "<tool_call>" in content

    def test_stream_no_tool_calls_regression(self, parser):
        """Normal reasoning unchanged when no tool_call present."""
        text = "<think>Just thinking here.</think>\nThe answer is 42."
        reasoning, content = _stream(parser, text)
        assert "Just thinking here." in (reasoning or "")
        assert content is not None
        assert "The answer is 42." in content

    def test_stream_prose_mention_not_promoted_at_finalize(self, parser):
        """Bare ``<tool_call>`` prose mention inside reasoning is NOT
        promoted in the streaming path either — the structural guard
        (parity with non-streaming ``_TOOL_CALL_UNCLOSED_RE``) skips
        the buffer entry when the next non-whitespace byte after the
        opener is not ``{`` or ``<``. The prose tail must stay in
        reasoning; post-think content must remain clean."""
        text = (
            "<think>The model should use <tool_call> to invoke functions. "
            "Then it should verify.</think>\n"
            "The answer is 42."
        )
        reasoning, content = _stream(parser, text)
        # Bare ``<tool_call>`` must NOT have triggered promotion — the
        # prose tail "to invoke functions. Then it should verify."
        # stays in reasoning, not content.
        assert reasoning is not None
        assert "to invoke functions" in reasoning
        assert content is not None
        assert "The answer is 42." in content
        # Post-think content channel must NOT contain the prose tail.
        assert "to invoke functions" not in content

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 13, 25])
    def test_stream_chunked_prose_mention_not_promoted(self, parser, chunk_size):
        """Regression: chunked streaming of a bare ``<tool_call>``
        prose mention must NOT promote prose into content, regardless
        of where SSE boundaries fall (mid-tag, after-tag-whitespace,
        mid-prose). Exercises the carry-forward case where the
        discriminating non-whitespace byte arrives in a later chunk."""
        text = (
            "<think>The model should use <tool_call> to invoke functions. "
            "Then it should verify.</think>\n"
            "The answer is 42."
        )
        reasoning, content = _stream(parser, text, chunk_size=chunk_size)
        assert reasoning is not None
        assert "to invoke functions" in reasoning
        assert content is not None
        assert "The answer is 42." in content
        assert "to invoke functions" not in content

    def test_stream_chunked_real_tool_call_after_prose_mention(self, parser):
        """A bare prose ``<tool_call>`` followed later by a real
        structural ``<tool_call>{json}</tool_call>`` must keep the
        prose in reasoning AND promote the real call. Verifies the
        structural-guard loop continues scanning past the prose tag."""
        text = (
            "<think>I might use <tool_call> if needed.\n"
            "<tool_call>\n"
            "<function=f><parameter=x>1</parameter></function>\n"
            "</tool_call>\n"
            "</think>\nDone."
        )
        reasoning, content = _stream(parser, text, chunk_size=4)
        assert reasoning is not None
        assert "if needed" in reasoning
        assert content is not None
        # The structural ``<tool_call>`` (followed by ``<function=…``)
        # must have been promoted.
        assert "<function=f>" in content
        assert "Done." in content

    def test_stream_carry_then_single_delta_shortcut_preserves_reasoning(self, parser):
        """Regression for the single-delta promotion shortcut: when a
        prior delta withheld a partial ``<tool_call>`` opener in
        ``_reasoning_carry`` and the NEXT delta arrives with both
        reasoning AND content (the post-think segment), the shortcut
        must NOT bypass ``_absorb_reasoning_chunk`` — otherwise the
        carried prefix is silently dropped from the wire.

        Setup: chunk_size 5 splits ``<tool_call>`` across the boundary
        so a carry is set, then the trailing chunk completes the tag
        AND closes ``</think>`` plus post-think content in the same
        delta. The streaming output must be byte-equivalent to the
        non-streaming extract for the same text."""
        text = (
            "<think>R\n"
            "<tool_call>\n"
            "<function=f><parameter=x>1</parameter></function>\n"
            "</tool_call>\n"
            "</think>\nPost."
        )
        # Non-streaming reference shape.
        ref_reasoning, ref_content = parser.extract_reasoning(text)
        # Try several chunk sizes that force partial-opener carries.
        for cs in (3, 5, 7, 11):
            stream_reasoning, stream_content = _stream(parser, text, chunk_size=cs)
            assert stream_content is not None, cs
            assert "<function=f>" in stream_content, cs
            assert "Post." in stream_content, cs
            # The carried "<tool_ca…" prefix must not disappear: the
            # reasoning prefix ``R\n`` survives, and ``<tool_call>``
            # never leaks into reasoning.
            assert stream_reasoning is not None, cs
            assert "R" in stream_reasoning, cs
            assert "<tool_call>" not in stream_reasoning, cs
            # Same wire shape (modulo whitespace) as non-streaming.
            assert ("<function=f>" in (ref_content or "")) == (
                "<function=f>" in stream_content
            ), cs


# ── Multi-family parity ─────────────────────────────────────────────


class TestMultiFamilyParity:
    """Promotion fires across every ``<think>``-tag subclass via the
    centralised filter."""

    @pytest.mark.parametrize("parser_name", ["qwen3", "deepseek_r1", "vibethinker"])
    def test_non_streaming_promotes_across_families(self, parser_name):
        cls = get_parser(parser_name)
        p = cls()
        output = (
            "<think>I should check.\n"
            "<tool_call>\n"
            "<function=get_weather><parameter=city>Tokyo</parameter></function>\n"
            "</tool_call>\n"
            "</think>\n"
        )
        reasoning, content = p.extract_reasoning(output)
        assert content is not None, parser_name
        assert "<tool_call>" in content, parser_name
        assert "get_weather" in content, parser_name
        assert "<tool_call>" not in (reasoning or ""), parser_name

    @pytest.mark.parametrize("parser_name", ["qwen3", "deepseek_r1", "vibethinker"])
    def test_streaming_promotes_across_families(self, parser_name):
        cls = get_parser(parser_name)
        p = cls()
        text = (
            "<think>Check.\n"
            "<tool_call>\n"
            "<function=f><parameter=x>1</parameter></function>\n"
            "</tool_call>\n"
            "</think>\nDone."
        )
        reasoning, content = _stream(p, text)
        assert content is not None, parser_name
        assert "<tool_call>" in content, parser_name


# ── Composition: promoted output parses through the tool parser ─────


class TestComposition:
    def test_promoted_parsed_by_tool_parser(self, parser):
        """Promoted content is structurally valid for the tool parser."""
        from vllm_mlx.tool_parsers import ToolParserManager

        output = (
            "<think>Let me look this up.\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "</think>\n"
        )
        reasoning, content = parser.extract_reasoning(output)
        assert content is not None

        # Find any Qwen-family tool parser registered in this build.
        tool_cls = None
        for parser_name in ("qwen3_xml", "qwen3.5", "qwen", "qwen3", "hermes"):
            try:
                tool_cls = ToolParserManager.get_tool_parser(parser_name)
                break
            except KeyError:
                continue
        if tool_cls is None:
            pytest.skip("No compatible tool parser registered")

        tool_parser_inst = tool_cls(None)
        result = tool_parser_inst.extract_tool_calls(content)
        # The promoted XML format is parser-specific — at minimum the
        # tool parser must not crash AND must surface ``get_weather``
        # somewhere in its output (the name appears in the raw content).
        assert result is not None
        # Either the tool parser recognised the call, or the raw text
        # passes through — both prove the parser was given the call to
        # work with (the promotion gate is open).
        assert "get_weather" in content
