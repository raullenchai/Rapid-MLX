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

from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser


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
        "WARNING", logger="rapid_mlx.tool_parsers.nemotron_tool_parser"
    ):
        result = parser.extract_tool_calls(text)
    assert not result.tools_called
    assert result.content == text
    assert any("no tool call extracted" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Streaming.
# ---------------------------------------------------------------------------


def _stream(parser, chunks, request=None):
    """Feed ``chunks`` incrementally; return the list of per-delta results."""
    previous = ""
    results = []
    for chunk in chunks:
        current = previous + chunk
        results.append(
            parser.extract_tool_calls_streaming(
                previous, current, chunk, request=request
            )
        )
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


_DECLARED_WEATHER = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
}
_COMPLETED_FUNCTION = (
    "<function=get_weather><parameter=city>Paris</parameter></function>"
)
_COMPLETED_WRAPPER = "<tool_call>" + _COMPLETED_FUNCTION + "</tool_call>"


@pytest.mark.parametrize("closer", ["</function>", "</tool_call>"])
@pytest.mark.parametrize("prefix", [_COMPLETED_FUNCTION, _COMPLETED_WRAPPER])
def test_buffered_completed_call_preserves_literal_closer(prefix, closer):
    # chat's _parse_tool_calls_with_parser -> _run_tool_parser creates a fresh parser,
    # calls extract_tool_calls on the full response, and returns result.content.
    # No streaming watermark or EOF flush participates in that buffered path.
    tail = f"Explain {closer} literally."
    result = NemotronToolParser().extract_tool_calls(
        prefix + " " + tail, request=_DECLARED_WEATHER
    )
    call = _only_call(result)
    assert json.loads(call["arguments"]) == {"city": "Paris"}
    assert result.content == tail


_PARAMETER_CLOSER_PROSE = " Use </parameter> and </function> literally."
_PENDING_WRAPPER_PROSE = "</tool_call> Explain </function> literally."


def _assert_weather_stream(chunks, expected_content):
    parser = NemotronToolParser()
    text = ""
    content = ""
    calls = []
    for chunk in chunks:
        if not chunk:
            continue
        current = text + chunk
        delta = (
            parser.extract_tool_calls_streaming(
                text, current, chunk, request=_DECLARED_WEATHER
            )
            or {}
        )
        content += delta.get("content", "")
        calls.extend(delta.get("tool_calls", []))
        text = current
    content += parser.flush_held_content(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert content == expected_content


@pytest.mark.parametrize("prefix", [_COMPLETED_FUNCTION, _COMPLETED_WRAPPER])
def test_post_call_parameter_closer_buffered(prefix):
    result = NemotronToolParser().extract_tool_calls(
        prefix + _PARAMETER_CLOSER_PROSE, request=_DECLARED_WEATHER
    )
    assert json.loads(_only_call(result)["arguments"]) == {"city": "Paris"}
    assert result.content == _PARAMETER_CLOSER_PROSE.strip()


@pytest.mark.parametrize("prefix", [_COMPLETED_FUNCTION, _COMPLETED_WRAPPER])
@pytest.mark.parametrize(
    "split", range(len(_COMPLETED_WRAPPER + _PARAMETER_CLOSER_PROSE) + 1)
)
def test_post_call_parameter_closer_streamed(prefix, split):
    text = prefix + _PARAMETER_CLOSER_PROSE
    _assert_weather_stream([text[:split], text[split:]], _PARAMETER_CLOSER_PROSE)


def test_pending_wrapper_and_literal_closer_buffered():
    result = NemotronToolParser().extract_tool_calls(
        "<tool_call>" + _COMPLETED_FUNCTION + _PENDING_WRAPPER_PROSE,
        request=_DECLARED_WEATHER,
    )
    assert json.loads(_only_call(result)["arguments"]) == {"city": "Paris"}
    assert result.content == "Explain </function> literally."


@pytest.mark.parametrize("split", range(len(_PENDING_WRAPPER_PROSE) + 1))
def test_pending_wrapper_and_literal_closer_streamed(split):
    _assert_weather_stream(
        [
            "<tool_call>" + _COMPLETED_FUNCTION,
            _PENDING_WRAPPER_PROSE[:split],
            _PENDING_WRAPPER_PROSE[split:],
        ],
        " Explain </function> literally.",
    )


@pytest.mark.parametrize("closer", ["</function>", "</tool_call>"])
@pytest.mark.parametrize("json_body", [False, True])
def test_buffered_payload_closers_and_post_call_prose_are_distinct(closer, json_body):
    value = "literal </parameter> </function> </tool_call> inside"
    body = (
        json.dumps({"city": value})
        if json_body
        else f"<parameter=city>{value}</parameter>"
    )
    prefix = f"<tool_call><function=get_weather>{body}</function></tool_call>"
    tail = f"Explain {closer} literally."
    result = NemotronToolParser().extract_tool_calls(
        prefix + " " + tail, request=_DECLARED_WEATHER
    )
    assert json.loads(_only_call(result)["arguments"]) == {"city": value}
    assert result.content == tail


def test_trailing_narration_does_not_rescan_accounted_prefix(parser, monkeypatch):
    parser.extract_tool_calls_streaming(
        "", _COMPLETED_WRAPPER, _COMPLETED_WRAPPER, request=_DECLARED_WEATHER
    )
    scans = 0
    scan = parser._scan_unmatched_opener

    def counted_scan(prefix, calls=None):
        nonlocal scans
        scans += 1
        return scan(prefix, calls)

    monkeypatch.setattr(parser, "_scan_unmatched_opener", counted_scan)
    previous = _COMPLETED_WRAPPER
    chunks = [" ordinary prose", " literal </function> and </tool_call>"] * 200
    for chunk in chunks:
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous, current, chunk, request=_DECLARED_WEATHER
        )
        assert delta == {"content": chunk}
        previous = current
    assert scans <= 1


@pytest.mark.parametrize("ending", ["", "</tool_call> visible", " literal"])
def test_wrapped_tail_bytewise_whitespace_has_linear_scan_work(
    parser, monkeypatch, ending
):
    scanned_bytes = 0
    scans = 0
    boundary = parser._wrapped_tail_boundary

    def counted_boundary(held, start=0, stop=None):
        nonlocal scanned_bytes, scans
        scanned_bytes += (stop or len(held)) - start
        scans += 1
        return boundary(held, start, stop)

    monkeypatch.setattr(parser, "_wrapped_tail_boundary", counted_boundary)
    whitespace = " \n" * 4096
    tail = " held prose </function>" + whitespace + ending
    previous = ""
    content = ""
    calls = []
    for chunk in [_COMPLETED_WRAPPER, *tail]:
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(previous, current, chunk) or {}
        calls.extend(delta.get("tool_calls", []))
        content += delta.get("content", "")
        previous = current
    content += parser.flush_held_content(previous)
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert content == (
        " held prose  visible" if ending.startswith("</tool_call>") else tail
    )
    assert scanned_bytes <= 32 * len(previous), (scanned_bytes, scans)
    assert scans <= 64


@pytest.mark.parametrize("reset", [False, True])
def test_opener_scan_cache_does_not_hide_next_turn_wrapper(parser, reset):
    parser.extract_tool_calls_streaming(
        "", _COMPLETED_WRAPPER, _COMPLETED_WRAPPER, request=_DECLARED_WEATHER
    )
    parser.flush_held_content(_COMPLETED_WRAPPER + " old narration")
    if reset:
        parser.reset()
    prefix = "<tool_call>" + _COMPLETED_FUNCTION
    first = parser.extract_tool_calls_streaming(
        "", prefix, prefix, request=_DECLARED_WEATHER
    )
    assert not first.get("content")
    assert parser.flush_held_content(prefix + "</tool") == ""


@pytest.mark.parametrize("closer", ["</function>", "</tool_call>"])
@pytest.mark.parametrize("prefix", [_COMPLETED_FUNCTION, _COMPLETED_WRAPPER])
def test_completed_call_flush_preserves_literal_closer(parser, prefix, closer):
    """EOF flush sees only buffered prose after an already-accounted call."""
    first = parser.extract_tool_calls_streaming(
        "", prefix, prefix, request=_DECLARED_WEATHER
    )
    assert len(first["tool_calls"]) == 1
    tail = f" Explain {closer} literally."
    assert parser.flush_held_content(prefix + tail) == tail
    assert parser.flush_held_content(prefix + tail) == ""


@pytest.mark.parametrize("closer", ["</function>", "</tool_call>"])
@pytest.mark.parametrize("prefix", [_COMPLETED_FUNCTION, _COMPLETED_WRAPPER])
def test_completed_call_stream_preserves_literal_closer_every_split(prefix, closer):
    tail = f" Explain {closer} literally."
    for split in range(len(tail) + 1):
        parser = NemotronToolParser()
        previous = ""
        content = ""
        calls = []
        for chunk in [prefix, tail[:split], tail[split:]]:
            if not chunk:
                continue
            current = previous + chunk
            delta = (
                parser.extract_tool_calls_streaming(
                    previous, current, chunk, request=_DECLARED_WEATHER
                )
                or {}
            )
            content += delta.get("content", "")
            calls.extend(delta.get("tool_calls", []))
            previous = current
        content += parser.flush_held_content(previous)
        assert content == tail, (split, content)
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}


@pytest.mark.parametrize("split", range(len("</tool_call>") + 1))
def test_pending_wrapper_close_stays_off_content(parser, split):
    prefix = "<tool_call>" + _COMPLETED_FUNCTION
    first = parser.extract_tool_calls_streaming(
        "", prefix, prefix, request=_DECLARED_WEATHER
    )
    assert len(first["tool_calls"]) == 1
    previous = prefix
    for chunk in ["</tool_call>"[:split], "</tool_call>"[split:]]:
        if not chunk:
            continue
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous, current, chunk, request=_DECLARED_WEATHER
        )
        assert not (delta or {}).get("content")
        assert parser.flush_held_content(current) == ""
        previous = current


@pytest.mark.parametrize("prefix", ["<function=get_weather>", "<tool_call>"])
@pytest.mark.parametrize("closer", ["</function>", "</tool_call>"])
def test_unclosed_consumed_opener_keeps_closers_held(parser, prefix, closer):
    parser._content_upto = len(prefix)
    assert parser.flush_held_content(prefix + closer[:5]) == ""
    assert parser.flush_held_content(prefix + closer) == ""


@pytest.mark.parametrize("literal", ["</function>", "</tool_call>"])
def test_payload_literal_close_does_not_close_pending_wrapper(parser, literal):
    prefix = (
        "<tool_call><function=get_weather><parameter=city>"
        + literal
        + "</parameter></function>"
    )
    parser._content_upto = len(prefix)
    assert parser.flush_held_content(prefix + "</tool") == ""


@pytest.mark.parametrize("closer", ["</function>", "</tool_call>"])
def test_multiple_completed_calls_then_literal_closer_bytewise(parser, closer):
    prefix = _COMPLETED_WRAPPER + _COMPLETED_FUNCTION + _COMPLETED_WRAPPER
    tail = f" Explain {closer} literally."
    previous = ""
    content = ""
    calls = []
    for byte in prefix + tail:
        current = previous + byte
        delta = (
            parser.extract_tool_calls_streaming(
                previous, current, byte, request=_DECLARED_WEATHER
            )
            or {}
        )
        content += delta.get("content", "")
        calls.extend(delta.get("tool_calls", []))
        previous = current
    content += parser.flush_held_content(previous)
    assert content == tail
    assert len(calls) == 3
    assert [call["index"] for call in calls] == [0, 1, 2]
    assert all(
        json.loads(call["function"]["arguments"]) == {"city": "Paris"} for call in calls
    )


@pytest.mark.parametrize("wrapper_closed", [False, True])
def test_consumed_payload_markers_do_not_determine_wrapper_state(
    parser, wrapper_closed
):
    prefix = (
        "<tool_call><function=get_weather><parameter=city>"
        "literal </function> </tool_call> <tool_call> <function=example>"
        "</parameter></function>"
    )
    if wrapper_closed:
        prefix += "</tool_call>"
    parser._content_upto = len(prefix)
    tail = " Mention </tool_call> literally."
    assert parser.flush_held_content(prefix + tail) == (
        tail if wrapper_closed else " Mention "
    )


@pytest.mark.parametrize("opener", ["<function=get_weather>", "<tool_call>"])
def test_completed_call_does_not_clear_later_unmatched_opener(parser, opener):
    prefix = _COMPLETED_WRAPPER + opener
    parser._content_upto = len(prefix)
    assert parser.flush_held_content(prefix + "</fun") == ""


@pytest.mark.parametrize("mode", ["buffered", "streamed", "split_wrapper"])
@pytest.mark.parametrize("body_format", ["raw", "parameters", "json"])
@pytest.mark.parametrize(
    "value",
    [
        "literal </function> inside",
        "literal </tool_call> inside",
        "literal </function> then </tool_call> inside",
        "literal </tool_call> then </function> inside",
        "literal </function> then </tool_call> then </function> inside",
    ],
)
def test_wrapped_payload_literal_closers_main_parity(mode, body_format, value):
    # These complete payloads preserve their literal bytes on origin/main,
    # both buffered and when the payload/function and wrapper close separately.
    body = {
        "raw": value,
        "parameters": f"<parameter=city>{value}</parameter>",
        "json": json.dumps({"city": value}),
    }[body_format]
    function = f"<tool_call><function=get_weather>{body}</function>"
    text = function + "</tool_call>"
    parser = NemotronToolParser()
    if mode == "buffered":
        result = parser.extract_tool_calls(text, request=_DECLARED_WEATHER)
        call = _only_call(result)
        arguments = call["arguments"]
        assert not result.content
    else:
        chunks = [text] if mode == "streamed" else [function, "</tool_call>"]
        deltas = _stream(parser, chunks, request=_DECLARED_WEATHER)
        calls = [tc for delta in deltas if delta for tc in delta.get("tool_calls", [])]
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"
        arguments = calls[0]["function"]["arguments"]
        assert not "".join(delta.get("content", "") for delta in deltas if delta)
        assert not parser.flush_held_content(text)
    if body_format == "raw":
        assert arguments == value
    else:
        assert json.loads(arguments) == {"city": value}


@pytest.mark.parametrize("size", [4096, 8192])
def test_streaming_large_payload_and_pending_wrapper_tail_scan_work(
    parser, monkeypatch, size
):
    value = "x" * size
    tail = " ordinary prose " * (size // 16)
    text = (
        "<tool_call><function=get_weather><parameter=city>"
        + value
        + "</parameter></function>"
        + tail
        + "</tool_call>"
    )
    scans = 0
    scanned_bytes = 0
    scan = NemotronToolParser._scan_unmatched_opener

    def counted_scan(prefix, calls=None):
        nonlocal scans, scanned_bytes
        scans += 1
        scanned_bytes += len(prefix)
        return scan(prefix, calls)

    # Patch the class so direct class-level calls cannot bypass the counter.
    monkeypatch.setattr(
        NemotronToolParser, "_scan_unmatched_opener", staticmethod(counted_scan)
    )
    deltas = _stream(parser, text, request=_DECLARED_WEATHER)
    content = "".join(d.get("content", "") for d in deltas if d)
    content += parser.flush_held_content(text)
    calls = [call for d in deltas if d for call in d.get("tool_calls", [])]
    assert content == tail
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": value}
    # Bytewise delivery must not rescan the completed payload per prose byte.
    assert scans <= 32, (scans, scanned_bytes)
    assert scanned_bytes <= 16 * len(text), (scans, scanned_bytes)


def _capture_stream(chunks):
    parser = NemotronToolParser()
    previous = ""
    calls = []
    content = ""
    for chunk in chunks:
        if not chunk:
            continue
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(previous, current, chunk) or {}
        calls.extend(
            json.loads(call["function"]["arguments"])
            for call in delta.get("tool_calls", [])
        )
        content += delta.get("content", "")
        previous = current
    return calls, content + parser.flush_held_content(previous)


_AMBIGUOUS_VALUE = "literal </parameter> </function> </tool_call> inside"
_AMBIGUOUS_WRAPPED = (
    f"<tool_call><function=f><parameter=a>{_AMBIGUOUS_VALUE}"
    "</parameter></function></tool_call>"
)


@pytest.mark.parametrize("split", range(len(_AMBIGUOUS_WRAPPED) + 1))
def test_wrapped_continuation_never_leaks_more_than_main(split):
    buffered = NemotronToolParser().extract_tool_calls(_AMBIGUOUS_WRAPPED)
    expected = {"a": _AMBIGUOUS_VALUE}
    assert json.loads(_only_call(buffered)["arguments"]) == expected
    assert not buffered.content
    calls, content = _capture_stream(
        [_AMBIGUOUS_WRAPPED[:split], _AMBIGUOUS_WRAPPED[split:]]
    )
    # Documented origin/main result: only splits 68..99 disagree (32/124).
    # Those splits emit the first call too soon; its arguments cannot retract.
    # Visible content may improve on main, but must never extend its leakage.
    if 68 <= split <= 99:
        assert calls in ([expected], [{"a": "literal "}])
        baseline_content = ""
        if 82 <= split <= 88 or 91 <= split <= 99:
            baseline_content = _AMBIGUOUS_WRAPPED[81:split]
        elif split in (89, 90):
            baseline_content = " inside"
        assert baseline_content.startswith(content)
    else:
        assert calls == [expected]
        assert content == ""


@pytest.mark.parametrize("whitespace", ["", "  ", "\n\t"])
@pytest.mark.parametrize("delivery", ["whole", "every_split", "bytewise"])
def test_wrapped_post_call_pair_in_prose_is_payload(whitespace, delivery):
    tail = f" Use </function>{whitespace}</tool_call> literally"
    text = _COMPLETED_WRAPPER + tail
    buffered = NemotronToolParser().extract_tool_calls(text)
    assert json.loads(_only_call(buffered)["arguments"]) == {"city": "Paris"}
    assert buffered.content == "literally"
    # Accepted ambiguity: the LAST pair consumes even prose before that pair.
    # Streaming cannot recover prose already sent, but withholds all held bytes
    # through the pair. Splitting inside the pair must never leak its markup.
    deliveries = (
        [[text]]
        if delivery == "whole"
        else [list(text)]
        if delivery == "bytewise"
        else [[text[:split], text[split:]] for split in range(len(text) + 1)]
    )
    for chunks in deliveries:
        calls, content = _capture_stream(chunks)
        assert calls == [{"city": "Paris"}]
        assert content in (" literally", " Use  literally") or (
            content.endswith(" literally")
            and " Use ".startswith(content[: -len(" literally")])
        )
        assert "<" not in content and ">" not in content


@pytest.mark.parametrize(
    "partial", ["</function>", "</function>  ", "</function></tool_"]
)
def test_wrapped_post_call_partial_pair_holds_entire_tail(partial):
    parser = NemotronToolParser()
    parser.extract_tool_calls_streaming("", _COMPLETED_WRAPPER, _COMPLETED_WRAPPER)
    text = _COMPLETED_WRAPPER + " held prose " + partial
    assert not (
        parser.extract_tool_calls_streaming(
            _COMPLETED_WRAPPER, text, text[len(_COMPLETED_WRAPPER) :]
        )
        or {}
    ).get("content")
    # EOF would release this partial pair; continue streaming before EOF here.
    # Non-whitespace text disambiguates a lone function close as literal prose.
    suffix = " literal" if partial != "</function></tool_" else "x"
    resolved = text + suffix
    delta = parser.extract_tool_calls_streaming(text, resolved, suffix) or {}
    assert delta.get("content") == " held prose " + partial + suffix


@pytest.mark.parametrize(
    "pair", ["</function></tool_call>", "</function> \n</tool_call>"]
)
@pytest.mark.parametrize("next_call", [False, True])
def test_wrapped_held_tail_consumed_through_last_pair_before_next_opener(
    pair, next_call
):
    tail = " payload " + pair + " more payload " + pair + " visible"
    suffix = _COMPLETED_WRAPPER + " Explain </function> literally." if next_call else ""
    calls, content = _capture_stream([_COMPLETED_WRAPPER, tail + suffix])
    assert calls == [{"city": "Paris"}] * (2 if next_call else 1)
    assert content == " visible" + (
        " Explain </function> literally." if next_call else ""
    )


@pytest.mark.parametrize(
    "ending", ["</function>", "</function>  ", "</function></tool_", "</tool_call>"]
)
def test_post_call_literal_closer_at_eof_every_split(ending):
    tail = " Close it with " + ending
    text = _COMPLETED_WRAPPER + tail
    buffered = NemotronToolParser().extract_tool_calls(text)
    assert buffered.content == tail.strip()
    for chunks in [list(text)] + [
        [text[:split], text[split:]] for split in range(len(text) + 1)
    ]:
        calls, content = _capture_stream(chunks)
        assert calls == [{"city": "Paris"}]
        # Streaming preserves narration whitespace; buffered extraction strips it.
        assert content == tail
        assert content.strip() == buffered.content


def test_complete_call_emits_before_possible_continuation_resolves(parser):
    text = _COMPLETED_WRAPPER + " held prose </function>"
    delta = parser.extract_tool_calls_streaming("", text, text)
    assert len(delta["tool_calls"]) == 1
    assert not delta.get("content")
    assert parser.flush_held_content(text) == " held prose </function>"


@pytest.mark.parametrize("mode", ["buffered", "whole", "split_calls", "bytewise"])
@pytest.mark.parametrize("body_format", ["parameters", "json"])
def test_earlier_payload_wrapper_opener_does_not_wrap_later_bare_call(
    mode, body_format
):
    value = "literal <tool_call> inside"
    first_body = (
        f"<parameter=a>{value}</parameter>"
        if body_format == "parameters"
        else json.dumps({"a": value})
    )
    first = f"<function=f>{first_body}</function>"
    second = "<function=f><parameter=a>second</parameter></function>"
    tail = " Explain </function></tool_call> literally."
    text = first + second + tail
    expected = [{"a": value}, {"a": "second"}]
    if mode == "buffered":
        result = NemotronToolParser().extract_tool_calls(text)
        assert [json.loads(call["arguments"]) for call in result.tool_calls] == expected
        assert result.content == tail.strip()
    else:
        chunks = (
            [text]
            if mode == "whole"
            else [first, second, tail]
            if mode == "split_calls"
            else list(text)
        )
        calls, content = _capture_stream(chunks)
        assert calls == expected
        assert content == tail


@pytest.mark.parametrize("streamed", [False, True])
def test_bare_malformed_json_body_preserves_raw_arguments(streamed):
    function = "<function=f>{broken</function>"
    tail = " Done."
    parser = NemotronToolParser()
    if streamed:
        deltas = _stream(parser, [function, tail])
        calls = [call for d in deltas if d for call in d.get("tool_calls", [])]
        assert len(calls) == 1
        assert calls[0]["function"]["arguments"] == "{broken"
        assert "".join(d.get("content", "") for d in deltas if d) == tail
        assert not parser.flush_held_content(function + tail)
    else:
        result = parser.extract_tool_calls(function + tail)
        assert _only_call(result)["arguments"] == "{broken"
        assert result.content == tail.strip()


@pytest.mark.parametrize("streamed", [False, True])
def test_many_wrapped_calls_bound_tail_scan_and_slice_work(monkeypatch, streamed):
    """Bound each scan; origin/main already reparses each streaming close event.

    Sequential streaming is still quadratic overall. Require exactly one span
    pass per close event so summed growing prefixes cannot mask an extra pass.
    """
    scanned_bytes = sliced_bytes = span_passes = 0
    boundary = NemotronToolParser._wrapped_tail_boundary
    spans = NemotronToolParser._function_spans

    class CountedText(str):
        def __getitem__(self, key):
            nonlocal sliced_bytes
            result = super().__getitem__(key)
            if isinstance(key, slice):
                sliced_bytes += len(result)
                return CountedText(result)
            return result

    def counted_boundary(text, start=0, stop=None):
        nonlocal scanned_bytes
        scanned_bytes += (len(text) if stop is None else stop) - start
        if start == 0 and stop is None:
            return boundary(text)
        return boundary(text, start, stop)

    def counted_spans(cls, text, valid_names=None):
        nonlocal span_passes
        span_passes += 1
        return spans(CountedText(text), valid_names)

    monkeypatch.setattr(
        NemotronToolParser, "_wrapped_tail_boundary", staticmethod(counted_boundary)
    )
    monkeypatch.setattr(
        NemotronToolParser, "_function_spans", classmethod(counted_spans)
    )
    call = "<tool_call><function=f><parameter=a>v</parameter></function></tool_call>"
    if streamed:
        parser = NemotronToolParser()
        previous = ""
        for index in range(500):
            current = previous + call
            before = span_passes
            before_scanned, before_sliced = scanned_bytes, sliced_bytes
            delta = parser.extract_tool_calls_streaming(previous, current, call)
            assert span_passes - before == 1
            assert scanned_bytes - before_scanned <= 2 * len(current)
            assert sliced_bytes - before_sliced <= 8 * len(current)
            assert delta and not delta.get("content")
            emitted = delta["tool_calls"]
            assert len(emitted) == 1
            assert emitted[0]["index"] == index
            assert json.loads(emitted[0]["function"]["arguments"]) == {"a": "v"}
            previous = current
        assert not parser.flush_held_content(previous)
        assert span_passes == 500
    else:
        result = NemotronToolParser().extract_tool_calls(call * 500)
        assert [json.loads(c["arguments"]) for c in result.tool_calls] == [
            {"a": "v"}
        ] * 500
        assert not result.content
        assert span_passes == 1
        input_bytes = len(call) * 500
        assert scanned_bytes <= 2 * input_bytes, (scanned_bytes, input_bytes)
        assert sliced_bytes <= 8 * input_bytes, (sliced_bytes, input_bytes)
    print(f"{streamed=}: {span_passes=}, {scanned_bytes=}, {sliced_bytes=}")


@pytest.mark.parametrize("change", ["text", "names", "reset"])
def test_span_reuse_requires_exact_input_and_declared_names(monkeypatch, change):
    parser = NemotronToolParser()
    text = "<function=f><parameter=a>v</parameter></function>"
    request = {"tools": [{"function": {"name": "f"}}]}
    assert _only_call(parser.extract_tool_calls(text, request))["name"] == "f"
    scans = 0
    spans = parser._function_spans

    def counted_spans(text, valid_names=None):
        nonlocal scans
        scans += 1
        return spans(text, valid_names)

    monkeypatch.setattr(parser, "_function_spans", counted_spans)
    assert parser._visible_content_between(text, 0, len(text), request) == ""
    assert scans == 0
    if change == "text":
        text = text.replace("function=f", "function=g")
    elif change == "names":
        request["tools"][0]["function"]["name"] = "g"
    else:
        parser.reset()
    expected = "" if change == "reset" else text
    assert parser._visible_content_between(text, 0, len(text), request) == expected
    assert scans == 1


def test_narration_does_not_copy_accounted_prefix_per_chunk():
    sliced_bytes = 0

    class CountedText(str):
        def __getitem__(self, key):
            nonlocal sliced_bytes
            result = super().__getitem__(key)
            if isinstance(key, slice):
                sliced_bytes += len(result)
            return result

    parser = NemotronToolParser()
    prefix = (
        "<tool_call><function=f><parameter=a>"
        + "x" * 8192
        + "</parameter></function></tool_call>"
    )
    parser.extract_tool_calls_streaming("", prefix, prefix)
    previous = prefix
    chunk = " Explain </function> literally."
    content = ""
    for _ in range(500):
        current = CountedText(previous + chunk)
        delta = parser.extract_tool_calls_streaming(previous, current, chunk) or {}
        content += delta.get("content", "")
        previous = current
    content += parser.flush_held_content(previous)
    assert content == chunk * 500
    assert sliced_bytes <= 8 * len(previous), (sliced_bytes, len(previous))
