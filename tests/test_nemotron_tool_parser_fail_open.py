# SPDX-License-Identifier: Apache-2.0
"""Fail-open regression tests for :class:`NemotronToolParser`.

Nemotron's *nominal* tool-call wire shape is fully wrapped::

    <tool_call><function=NAME>...</function></tool_call>

In practice the model (especially at low quantization / after several tool
rounds) degrades into several near-miss variants. The historical parser
keyed on the whole ``<tool_call>...</tool_call>`` wrapper, so every one of
these leaked the call through as plain assistant ``content`` instead of a
structured tool call:

    (a) a missing / truncated ``</tool_call>``
    (b) a bare ``<function=..>..</function>`` with no wrapper at all
    (d) stray text between ``</function>`` and ``</tool_call>``
    (e) prose between ``<tool_call>`` and ``<function=``

The load-bearing signature of a call is ``<function=NAME>...</function>``, so
the parser now keys on that and treats the wrapper as optional. Prose that
does not contain ``<function=..>..</function>`` still never matches, so this
cannot manufacture a tool call out of plain text.

This module imports only the parser class (no MLX / model runtime), so it is
runnable in isolation.
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser


@pytest.fixture
def parser() -> NemotronToolParser:
    return NemotronToolParser()


def _only_call(result):
    assert result.tools_called
    assert len(result.tool_calls) == 1
    return result.tool_calls[0]


# ---------------------------------------------------------------------------
# Canonical regression — the fully-wrapped shape must keep working.
# ---------------------------------------------------------------------------


def test_canonical_parameter_wrapper(parser):
    text = (
        "<tool_call><function=get_weather>"
        "<parameter=city>Paris</parameter>"
        "</function></tool_call>"
    )
    tc = _only_call(parser.extract_tool_calls(text))
    assert tc["name"] == "get_weather"
    assert json.loads(tc["arguments"]) == {"city": "Paris"}


def test_canonical_json_body_wrapper(parser):
    text = '<tool_call><function=calculate>{"expression": "2*3"}</function></tool_call>'
    tc = _only_call(parser.extract_tool_calls(text))
    assert tc["name"] == "calculate"
    assert json.loads(tc["arguments"]) == {"expression": "2*3"}


# ---------------------------------------------------------------------------
# Degraded variants that previously leaked as text.
# ---------------------------------------------------------------------------


def test_variant_a_missing_closing_tool_call(parser):
    """(a) ``</tool_call>`` truncated away — the call must still parse."""
    text = (
        "<tool_call><function=get_weather><parameter=city>Paris</parameter></function>"
    )
    tc = _only_call(parser.extract_tool_calls(text))
    assert tc["name"] == "get_weather"
    assert json.loads(tc["arguments"]) == {"city": "Paris"}


def test_variant_b_bare_function_no_wrapper(parser):
    """(b) No ``<tool_call>`` wrapper at all."""
    text = "<function=get_weather><parameter=city>Paris</parameter></function>"
    tc = _only_call(parser.extract_tool_calls(text))
    assert tc["name"] == "get_weather"
    assert json.loads(tc["arguments"]) == {"city": "Paris"}


def test_variant_d_stray_text_before_close(parser):
    """(d) Stray text between ``</function>`` and ``</tool_call>``."""
    text = (
        "<tool_call><function=get_weather>"
        "<parameter=city>Paris</parameter></function>"
        " some junk </tool_call>"
    )
    result = parser.extract_tool_calls(text)
    tc = _only_call(result)
    assert tc["name"] == "get_weather"
    assert json.loads(tc["arguments"]) == {"city": "Paris"}
    # The stray text survives as content; residual wrapper tags do not leak.
    assert result.content == "some junk"
    assert "tool_call" not in (result.content or "")


def test_variant_e_prose_before_function(parser):
    """(e) Prose between ``<tool_call>`` and ``<function=``."""
    text = (
        "<tool_call>Let me check the weather."
        "<function=get_weather><parameter=city>Paris</parameter></function></tool_call>"
    )
    result = parser.extract_tool_calls(text)
    tc = _only_call(result)
    assert tc["name"] == "get_weather"
    assert result.content == "Let me check the weather."
    assert "tool_call" not in (result.content or "")


def test_surrounding_prose_bare_call_no_leak(parser):
    """A bare call embedded in prose parses; no XML leaks into content."""
    text = (
        "Sure, I'll do that.\n"
        "<function=get_weather><parameter=city>Paris</parameter></function>\n"
        "Done."
    )
    result = parser.extract_tool_calls(text)
    tc = _only_call(result)
    assert tc["name"] == "get_weather"
    assert "<function=" not in (result.content or "")
    assert "</function>" not in (result.content or "")


# ---------------------------------------------------------------------------
# Zero false positives — prose without a function signature never matches.
# ---------------------------------------------------------------------------


def test_zero_false_positive_plain_prose(parser):
    text = "Here is the information you requested. No tools needed."
    result = parser.extract_tool_calls(text)
    assert not result.tools_called
    assert result.content == text


def test_zero_false_positive_mentions_function_word(parser):
    text = "You can call the function get_weather to retrieve the data."
    result = parser.extract_tool_calls(text)
    assert not result.tools_called
    assert result.content == text


def test_marker_present_but_unparseable_logs_and_fails_open(parser, caplog):
    """A ``<tool_call>`` marker with no parseable function must fail open
    (return the raw text unchanged) and emit a diagnostic warning so the
    unhandled shape can be captured."""
    text = "<tool_call>garbled, no function here</tool_call>"
    with caplog.at_level(
        "WARNING", logger="vllm_mlx.tool_parsers.nemotron_tool_parser"
    ):
        result = parser.extract_tool_calls(text)
    assert not result.tools_called
    assert result.content == text
    assert any("no tool call extracted" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Streaming.
# ---------------------------------------------------------------------------


def _stream(parser, chunks):
    """Feed ``chunks`` incrementally; return the list of per-delta results."""
    previous = ""
    results = []
    for chunk in chunks:
        current = previous + chunk
        results.append(parser.extract_tool_calls_streaming(previous, current, chunk))
        previous = current
    return results


def test_streaming_passthrough_without_marker(parser):
    assert parser.extract_tool_calls_streaming("", "Hello", "Hello") == {
        "content": "Hello"
    }


def test_streaming_closes_on_function_when_tool_call_truncated(parser):
    """A truncated variant only ever emits ``</function>`` (no
    ``</tool_call>``); the streaming path must still emit the call."""
    chunks = [
        "<tool_call>",
        "<function=get_weather>",
        "<parameter=city>Paris</parameter>",
        "</function>",
    ]
    deltas = _stream(parser, chunks)
    emitted = [d for d in deltas if d and "tool_calls" in d]
    assert len(emitted) == 1
    tc = emitted[0]["tool_calls"][0]
    assert tc["index"] == 0
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}


def test_streaming_closes_on_function_bare_no_wrapper(parser):
    chunks = [
        "<function=get_weather>",
        "<parameter=city>Paris</parameter>",
        "</function>",
    ]
    deltas = _stream(parser, chunks)
    emitted = [d for d in deltas if d and "tool_calls" in d]
    assert len(emitted) == 1
    assert emitted[0]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_streaming_dedup_across_separate_close_deltas(parser):
    """When ``</function>`` and ``</tool_call>`` arrive in *separate* deltas,
    the call must be emitted exactly once, not twice."""
    chunks = [
        "<tool_call>",
        "<function=get_weather><parameter=city>Paris</parameter></function>",
        "</tool_call>",
    ]
    deltas = _stream(parser, chunks)
    emitted = [d for d in deltas if d and "tool_calls" in d]
    assert len(emitted) == 1
    assert emitted[0]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_streaming_close_tag_split_across_two_chunks(parser):
    """The tokenizer can split ``</function>`` across deltas (``"</fun"`` then
    ``"ction>"``). No single ``delta_text`` ever contains the whole close tag,
    but the accumulated ``current_text`` does once both fragments arrive, so
    the completed call must still be emitted exactly once."""
    chunks = [
        "<tool_call><function=get_weather><parameter=city>Paris</parameter>",
        "</fun",
        "ction>",
    ]
    deltas = _stream(parser, chunks)
    emitted = [d for d in deltas if d and "tool_calls" in d]
    assert len(emitted) == 1
    tc = emitted[0]["tool_calls"][0]
    assert tc["index"] == 0
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}


def test_streaming_no_reemit_on_trailing_deltas_after_close(parser):
    """Once a call has closed, later deltas that add no new close tag must not
    re-emit it (the completion trigger keys on a NEW close tag, so a call is
    emitted exactly once even as trailing tokens keep arriving). The trailing
    text passes through as content events, not swallowed."""
    chunks = [
        "<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>",
        " and",
        " that",
        " is all",
    ]
    deltas = _stream(parser, chunks)
    tool_deltas = [d for d in deltas if d and "tool_calls" in d]
    assert len(tool_deltas) == 1
    # The trailing content deltas produce no further tool-call emissions ...
    assert all("tool_calls" not in (d or {}) for d in deltas[1:])
    # ... and the trailing prose is streamed through as content, not dropped.
    trailing_content = "".join(d["content"] for d in deltas[1:] if d and "content" in d)
    assert trailing_content == " and that is all"


def test_streaming_bare_call_then_trailing_prose(parser):
    """A bare ``<function=..></function>`` (no wrapper) followed by trailing
    prose: the call is emitted once and the trailing prose is streamed as
    content (regression for the dropped-trailing-content finding)."""
    chunks = [
        "<function=get_weather>",
        "<parameter=city>Paris</parameter>",
        "</function>",
        " Anything",
        " else?",
    ]
    deltas = _stream(parser, chunks)
    tool_deltas = [d for d in deltas if d and "tool_calls" in d]
    assert len(tool_deltas) == 1
    assert tool_deltas[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    trailing_content = "".join(d["content"] for d in deltas if d and "content" in d)
    assert trailing_content == " Anything else?"
    # No XML markup ever leaks into the content stream.
    assert "<function=" not in trailing_content
    assert "</function>" not in trailing_content


def test_streaming_close_tag_and_trailing_prose_same_delta(parser):
    """The tokenizer can emit the close tag AND trailing prose in ONE delta
    ("</function> done"). The call must be emitted and the trailing prose must
    ride out on the same delta (combined content+tool_calls), never dropped."""
    chunks = [
        "<function=get_weather><parameter=city>Paris</parameter>",
        "</function> done",
    ]
    deltas = _stream(parser, chunks)
    tool_deltas = [d for d in deltas if d and "tool_calls" in d]
    assert len(tool_deltas) == 1
    combined = tool_deltas[0]
    assert combined["tool_calls"][0]["function"]["name"] == "get_weather"
    # Trailing prose preserved on the same delta, with no markup leak.
    assert combined.get("content") == " done"
    all_content = "".join(d["content"] for d in deltas if d and "content" in d)
    assert all_content == " done"
    assert "<" not in all_content and ">" not in all_content


def test_streaming_no_markup_leak_when_close_tag_split(parser):
    """When ``</function>`` is split across deltas, the fragment that carries
    the tag ("ction>") must never surface as a content event."""
    chunks = [
        "<function=get_weather><parameter=city>Paris</parameter>",
        "</fun",
        "ction>",
        " done",
    ]
    deltas = _stream(parser, chunks)
    contents = [d["content"] for d in deltas if d and "content" in d]
    # Only the genuine trailing prose is content; no tag fragment leaks.
    assert contents == [" done"]
    assert all("fun" not in c and "<" not in c and ">" not in c for c in contents)


def test_streaming_two_sequential_calls_increment_index(parser):
    chunks = [
        "<tool_call><function=f1><parameter=a>1</parameter></function></tool_call>",
        "<tool_call><function=f2><parameter=b>2</parameter></function></tool_call>",
    ]
    deltas = _stream(parser, chunks)
    emitted = [d for d in deltas if d and "tool_calls" in d]
    assert len(emitted) == 2
    assert emitted[0]["tool_calls"][0]["index"] == 0
    assert emitted[0]["tool_calls"][0]["function"]["name"] == "f1"
    assert emitted[1]["tool_calls"][0]["index"] == 1
    assert emitted[1]["tool_calls"][0]["function"]["name"] == "f2"
