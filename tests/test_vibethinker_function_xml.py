# SPDX-License-Identifier: Apache-2.0
"""F-042 redo — VibeThinker ``<function><name>...</name><arguments>...</arguments></function>``
parser regression suite.

These tests cover the redo design where the parser uses a single
left-to-right scan over all three structured wire shapes
(``<tool_call>``, ``<function=NAME>``, ``<function><name>``) so that
``tool_calls`` always appear in wire order — fixing the two P2 issues
codex flagged on the original F-042 (PR #746):

* **P2-1 (wire order):** mixed-shape responses now preserve the order
  the model emitted the calls in, regardless of which shape went first.
* **P2-2 (streaming mixed leak):** a stream carrying shape #1 then
  shape #4 (or vice versa) routes BOTH through ``tool_calls`` deltas
  instead of leaking the second opener as content.

13-case matrix per the F-042 v2 spec — 8 non-streaming + 5 streaming.
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser


@pytest.fixture
def parser() -> HermesToolParser:
    return HermesToolParser()


# ---------------------------------------------------------------------
# Non-streaming matrix (cases 1–8)
# ---------------------------------------------------------------------


def test_case01_function_xml_alone(parser: HermesToolParser) -> None:
    """Case 1: ``<function>a</function>`` alone → ``[a]``."""
    text = '<function><name>a</name><arguments>{"x": 1}</arguments></function>'
    result = parser.extract_tool_calls(text)
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "a"
    assert json.loads(result.tool_calls[0]["arguments"]) == {"x": 1}


def test_case02_tool_call_alone(parser: HermesToolParser) -> None:
    """Case 2: ``<tool_call>b</tool_call>`` alone → ``[b]`` (regression check)."""
    text = '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    result = parser.extract_tool_calls(text)
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "b"
    assert json.loads(result.tool_calls[0]["arguments"]) == {"y": 2}


def test_case03_function_then_tool_call_wire_order(parser: HermesToolParser) -> None:
    """Case 3: ``<function>a</function><tool_call>b</tool_call>`` → ``[a, b]``.

    **P2-1 fix:** the prior implementation appended named-XML matches
    AFTER existing ``<tool_call>`` matches regardless of wire position,
    so this case returned ``[b, a]``. The unified scan fixes that.
    """
    text = (
        "<function><name>a</name>"
        '<arguments>{"x": 1}</arguments></function>'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    )
    result = parser.extract_tool_calls(text)
    assert result.tools_called is True
    assert len(result.tool_calls) == 2
    assert [c["name"] for c in result.tool_calls] == ["a", "b"], (
        f"Expected wire order [a, b]; got {[c['name'] for c in result.tool_calls]}"
    )


def test_case04_tool_call_then_function_wire_order(parser: HermesToolParser) -> None:
    """Case 4: ``<tool_call>b</tool_call><function>a</function>`` → ``[b, a]``.

    **P2-1 sanity check (reverse order):** swapping wire order MUST
    swap emitted order. This is the case the prior code coincidentally
    handled correctly because additive append matched wire order; the
    redo must keep it working.
    """
    text = (
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
        "<function><name>a</name>"
        '<arguments>{"x": 1}</arguments></function>'
    )
    result = parser.extract_tool_calls(text)
    assert result.tools_called is True
    assert len(result.tool_calls) == 2
    assert [c["name"] for c in result.tool_calls] == ["b", "a"], (
        f"Expected [b, a]; got {[c['name'] for c in result.tool_calls]}"
    )


def test_case05_function_then_bare_function_mixed_shapes(
    parser: HermesToolParser,
) -> None:
    """Case 5: named-XML then bare-Nemotron mixed → wire order preserved."""
    text = (
        "<function><name>a</name>"
        '<arguments>{"x": 1}</arguments></function>'
        '<function=c>{"z": 3}</function>'
    )
    result = parser.extract_tool_calls(text)
    assert result.tools_called is True
    assert len(result.tool_calls) == 2
    assert [c["name"] for c in result.tool_calls] == ["a", "c"]


def test_case06_two_function_xml_blocks(parser: HermesToolParser) -> None:
    """Case 6: two named-XML blocks in sequence → both emitted in order."""
    text = (
        "<function><name>a</name>"
        '<arguments>{"x": 1}</arguments></function>'
        "<function><name>b</name>"
        '<arguments>{"y": 2}</arguments></function>'
    )
    result = parser.extract_tool_calls(text)
    assert result.tools_called is True
    assert len(result.tool_calls) == 2
    assert [c["name"] for c in result.tool_calls] == ["a", "b"]


def test_case07_plain_text_no_tool_calls(parser: HermesToolParser) -> None:
    """Case 7: plain text → ``tools_called=False``, content preserved."""
    text = "Hello, this is a regular reply with no tool call markers."
    result = parser.extract_tool_calls(text)
    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content is not None
    assert "regular reply" in result.content


def test_case08_literal_function_in_prose_no_false_positive(
    parser: HermesToolParser,
) -> None:
    """Case 8: literal ``<function>`` mention in prose without ``<name>``
    follower must NOT be parsed as a tool call."""
    text = (
        "Tool calls can use the <function> XML tag, like "
        "<function=NAME>...</function> for Nemotron-style calls. "
        "This is just prose, not a real call."
    )
    # The closing prose has a bare-function literal call mid-paragraph,
    # which WILL match BARE_FUNCTION_PATTERN. Use a version with no
    # closing tag so this is purely prose.
    text = (
        "Models emit calls inside <function>...</function> tags. "
        "This explanatory sentence is plain content."
    )
    result = parser.extract_tool_calls(text)
    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content is not None
    assert "<function>" in result.content


# ---------------------------------------------------------------------
# Streaming matrix (cases 9–13)
# ---------------------------------------------------------------------


def _drive_stream(
    parser: HermesToolParser, chunks: list[str]
) -> tuple[list[dict], list[str]]:
    """Replay a list of chunks through the streaming API.

    Returns ``(tool_call_deltas, content_deltas)`` aggregated across
    all chunk callbacks.
    """
    previous = ""
    tool_call_deltas: list[dict] = []
    content_deltas: list[str] = []
    for chunk in chunks:
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=chunk,
        )
        if delta is not None:
            if "tool_calls" in delta:
                tool_call_deltas.extend(delta["tool_calls"])
            if "content" in delta and delta["content"]:
                content_deltas.append(delta["content"])
        previous = current
    return tool_call_deltas, content_deltas


def test_case09_stream_function_xml_alone(parser: HermesToolParser) -> None:
    """Case 9: stream a named-XML block → tool_calls delta, no content."""
    chunks = [
        "<function><name>",
        "a",
        '</name><arguments>{"x": 1}</arguments></function>',
    ]
    tc_deltas, content_deltas = _drive_stream(parser, chunks)
    assert len(tc_deltas) == 1
    assert tc_deltas[0]["function"]["name"] == "a"
    assert content_deltas == []


def test_case10_stream_tool_call_then_function_xml_wire_order(
    parser: HermesToolParser,
) -> None:
    """Case 10: stream ``<tool_call>b</tool_call>`` then ``<function>a</function>``
    → BOTH appear as ``tool_calls`` deltas in wire order (P2-2 fix)."""
    chunks = [
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>',
        '<function><name>a</name><arguments>{"x": 1}</arguments></function>',
    ]
    tc_deltas, content_deltas = _drive_stream(parser, chunks)
    names = [d["function"]["name"] for d in tc_deltas]
    assert names == ["b", "a"], (
        f"Expected [b, a] in tool_calls deltas; got {names}. "
        f"Stray content: {content_deltas!r}"
    )
    # P2-2: the second opener must NOT have leaked as content.
    leaked = "".join(content_deltas)
    assert "<function>" not in leaked, f"Named-XML opener leaked as content: {leaked!r}"
    assert "<name>" not in leaked


def test_case11_stream_function_xml_then_tool_call_wire_order(
    parser: HermesToolParser,
) -> None:
    """Case 11: reverse — ``<function>a</function>`` then ``<tool_call>b</tool_call>``
    → BOTH appear in tool_calls in wire order ``[a, b]``."""
    chunks = [
        '<function><name>a</name><arguments>{"x": 1}</arguments></function>',
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>',
    ]
    tc_deltas, content_deltas = _drive_stream(parser, chunks)
    names = [d["function"]["name"] for d in tc_deltas]
    assert names == ["a", "b"], (
        f"Expected [a, b] in tool_calls deltas; got {names}. "
        f"Stray content: {content_deltas!r}"
    )
    leaked = "".join(content_deltas)
    assert "<tool_call>" not in leaked
    assert "<function>" not in leaked


def test_case12_stream_function_literal_no_name_follower_is_content(
    parser: HermesToolParser,
) -> None:
    """Case 12: stream a literal ``<function>`` not followed by ``<name>``
    → emitted as content, NOT held back as a tool-call opener.

    Reaches the parser via end-of-stream flush so the partial-hold path
    is exercised. ``flush_held_content`` releases any prefix-held bytes.
    """
    chunks = ["The <function> tag is one form.", ""]
    tc_deltas, content_deltas = _drive_stream(parser, chunks)
    assert tc_deltas == []
    # Some content should have been emitted as the stream progressed.
    # ``<function>`` is a sentinel so it'll be held until disambiguated;
    # at the next chunk (" tag is one form.") the bytes following the
    # `<` disambiguate as prose and are released.
    full_text = "The <function> tag is one form."
    flushed = parser.flush_held_content(full_text)
    # Either content_deltas covered the entire safe-emit text, or some
    # bytes were held until flush. The union must reconstruct the input.
    reconstructed = "".join(content_deltas) + flushed
    assert reconstructed == full_text, (
        f"Stream content + flush ({reconstructed!r}) didn't reconstruct "
        f"original input ({full_text!r})"
    )


def test_case13_stream_partial_opener_at_tail_holds_back(
    parser: HermesToolParser,
) -> None:
    """Case 13: partial opener at tail (``<func``) is held back, NOT
    leaked as content; resolves when ``<function><name>...`` completes."""
    # Stage 1: "<func" — held, no emit.
    delta1 = parser.extract_tool_calls_streaming(
        previous_text="prefix ",
        current_text="prefix <func",
        delta_text="<func",
    )
    # delta1 should be either None (held entirely) or content="" (no new safe bytes).
    if delta1 is not None and "content" in delta1:
        assert "<func" not in delta1["content"], (
            f"Partial opener leaked as content: {delta1['content']!r}"
        )

    # Stage 2: completion to a real named-XML opener should claim the
    # held bytes as part of the tool-call block, NOT emit them as content.
    full = 'prefix <function><name>a</name><arguments>{"x": 1}</arguments></function>'
    delta2 = parser.extract_tool_calls_streaming(
        previous_text="prefix <func",
        current_text=full,
        delta_text=full[len("prefix <func") :],
    )
    # Should be a tool_calls delta with name "a".
    assert delta2 is not None
    assert "tool_calls" in delta2
    assert delta2["tool_calls"][0]["function"]["name"] == "a"
