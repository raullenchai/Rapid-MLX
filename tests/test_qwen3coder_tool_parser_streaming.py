# SPDX-License-Identifier: Apache-2.0
"""Incremental-delta streaming for ``Qwen3CoderToolParser``.

Closes the bug surfaced in #479 (kenizhou): the parser used to buffer the
entire string-typed parameter value until ``</parameter>`` arrived, then
dump it as a single ``function.arguments`` delta. CopilotKit / LangChain
inspectors that render argument values live stalled for multiple seconds
on long summaries / key-points.

The three tests below pin the new behavior:

* ``test_long_string_param_emits_multiple_deltas`` — granularity guard.
* ``test_close_tag_never_leaks_into_emitted_fragment`` — the
  ``len("</parameter>")`` tail-buffer guarantees no half-tag escapes
  into a JSON fragment.
* ``test_streaming_json_matches_non_streaming`` — concatenating all
  streamed argument fragments and parsing the result yields the same
  Python object as ``extract_tool_calls`` returns for the full text.
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

_LONG_SUMMARY = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut "
    "enim ad minim veniam, quis nostrud exercitation ullamco laboris "
    "nisi ut aliquip ex ea commodo consequat."
)


def _request_with_tool(name: str, properties: dict) -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {"type": "object", "properties": properties},
                },
            }
        ]
    }


def _feed(parser: Qwen3CoderToolParser, chunks: list[str], request: dict | None):
    """Stream ``chunks`` through the parser; return non-None delta dicts."""
    parser.reset()
    deltas: list[dict] = []
    previous = ""
    for chunk in chunks:
        if not chunk:
            continue
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=chunk,
            request=request,
        )
        if delta is not None:
            deltas.append(delta)
        previous = current
    return deltas


def _argument_fragments(deltas: list[dict]) -> list[str]:
    """Flatten ``function.arguments`` strings out of streamed tool_call deltas."""
    out: list[str] = []
    for d in deltas:
        for tc in d.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if args:
                out.append(args)
    return out


def test_long_string_param_emits_multiple_deltas():
    """For a long string param, at least 2 ``function.arguments`` deltas with
    non-empty content must arrive BEFORE the ``</parameter>`` close.

    Without #479's incremental emit the parser only produced a single
    delta containing the whole value once the close tag arrived.
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool(
        "summarize",
        {"summary": {"type": "string"}},
    )

    head = [
        "<tool_call>\n",
        "<function=summarize>\n",
        "<parameter=summary>\n",
    ]
    # Split the long summary into 32-char body chunks so the in-flight
    # branch fires several times before the close tag arrives.
    value_chunks = [_LONG_SUMMARY[i : i + 32] for i in range(0, len(_LONG_SUMMARY), 32)]
    pre_close_chunks = head + value_chunks

    parser.reset()
    deltas_before_close: list[dict] = []
    previous = ""
    for chunk in pre_close_chunks:
        current = previous + chunk
        d = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=chunk,
            request=request,
        )
        if d is not None:
            deltas_before_close.append(d)
        previous = current

    arg_fragments = [
        f
        for f in _argument_fragments(deltas_before_close)
        # Skip the structural openers ("{", "") so we only count real
        # value-bearing fragments — those are what the UI renders live.
        if f not in ("{", "")
    ]

    assert len(arg_fragments) >= 2, (
        "#479 regression: long string params should stream incrementally; "
        f"got {len(arg_fragments)} value-bearing argument deltas before "
        f"</parameter>. fragments={arg_fragments!r}"
    )


def test_close_tag_never_leaks_into_emitted_fragment():
    """Feeding ``...value</par`` then ``ameter>`` across two chunks must
    never produce a ``function.arguments`` fragment containing the literal
    substring ``</par`` or ``</parameter>``.

    Guards the tail-buffer: the parser must hold back the last
    ``len("</parameter>")`` chars of unread tail so a partial close tag
    straddling a chunk boundary can't be flushed prematurely.
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool("echo", {"value": {"type": "string"}})

    # Long enough that incremental emission will fire before the close.
    value = "A" * 200
    # Split mid-close-tag (``</par`` | ``ameter>``) to exercise the
    # tail-buffer guard at a chunk boundary.
    chunks = [
        "<tool_call>\n",
        "<function=echo>\n",
        "<parameter=value>\n",
        value[:80],
        value[80:160],
        value[160:] + "\n</par",
        "ameter>\n",
        "</function>\n",
        "</tool_call>",
    ]

    deltas = _feed(parser, chunks, request)
    fragments = _argument_fragments(deltas)
    for frag in fragments:
        assert "</par" not in frag, f"close-tag leaked into streamed fragment: {frag!r}"
        assert "</parameter>" not in frag, (
            f"close-tag leaked into streamed fragment: {frag!r}"
        )

    # Belt + braces: concatenate everything, parse the JSON, and assert
    # the decoded value is exactly the original — catches escaped /
    # split-across-fragments leaks that a substring scan alone would miss.
    combined = "".join(fragments)
    decoded = json.loads(combined)
    assert decoded == {"value": value}, (
        f"streamed args decoded to {decoded!r}, expected {{'value': {value!r}}}"
    )


def test_streaming_json_matches_non_streaming():
    """Concatenating all streamed ``function.arguments`` fragments and
    ``json.loads``-ing the result must equal the arguments dict
    ``extract_tool_calls`` returns for the same complete input.

    Covers mixed param types — string (mid-flight emit), int (buffered
    emit), and a second short string (back-to-back string).
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool(
        "report",
        {
            "summary": {"type": "string"},
            "score": {"type": "integer"},
            "owner": {"type": "string"},
        },
    )

    full_text = (
        "<tool_call>\n"
        "<function=report>\n"
        f"<parameter=summary>\n{_LONG_SUMMARY}\n</parameter>\n"
        "<parameter=score>\n42\n</parameter>\n"
        "<parameter=owner>\nken\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    # Non-streaming reference
    ns_result = parser.extract_tool_calls(full_text, request=request)
    assert ns_result.tools_called, "non-stream extract should detect tool call"
    expected_args = json.loads(ns_result.tool_calls[0]["arguments"])

    # Streaming run with small body chunks so the in-flight emit fires
    # repeatedly and back-to-back string params exercise param-separator
    # logic in ``_close_string_increment``.
    chunks = [
        "<tool_call>\n",
        "<function=report>\n",
        "<parameter=summary>\n",
    ]
    summary_body = _LONG_SUMMARY + "\n"
    chunks.extend(summary_body[i : i + 24] for i in range(0, len(summary_body), 24))
    chunks.extend(
        [
            "</parameter>\n",
            "<parameter=score>\n42\n</parameter>\n",
            "<parameter=owner>\nken\n</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
    )
    deltas = _feed(parser, chunks, request)
    fragments = _argument_fragments(deltas)
    combined = "".join(fragments)

    streamed_args = json.loads(combined)
    assert streamed_args == expected_args, (
        f"streamed JSON does not match non-streamed JSON.\n"
        f"  combined raw    = {combined!r}\n"
        f"  streamed parsed = {streamed_args!r}\n"
        f"  expected        = {expected_args!r}"
    )


def test_same_chunk_close_and_trailing_param_not_dropped():
    """When one chunk batches ``...tail</parameter><parameter=other>val</parameter>``
    the parser must emit BOTH the closing tail of the in-flight string
    AND the complete trailing param in the same call — and finalize.

    Without resuming the complete-param loop after closing the in-flight
    string, the trailing ``other`` param would be silently dropped (a
    regression caught by adversarial review on PR #648).
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool(
        "report",
        {
            "summary": {"type": "string"},
            "score": {"type": "integer"},
        },
    )

    # Multi-chunk stream where the FINAL chunk batches
    # ``<rest_of_summary></parameter><parameter=score>42</parameter></function></tool_call>``
    # so the in-flight close and the trailing complete param land in
    # the same parser call.
    head_value = _LONG_SUMMARY[:120]
    tail_value = _LONG_SUMMARY[120:]
    chunks = [
        "<tool_call>\n",
        "<function=report>\n",
        "<parameter=summary>\n",
        head_value,
        tail_value
        + "\n</parameter>\n<parameter=score>\n42\n</parameter>\n</function>\n</tool_call>",
    ]

    deltas = _feed(parser, chunks, request)
    fragments = _argument_fragments(deltas)
    combined = "".join(fragments)
    # Trailing ``}`` arrives on the next streaming call after function-end
    # is observed; close manually for the parity assert.
    if not combined.endswith("}"):
        combined += "}"
    decoded = json.loads(combined)
    assert decoded == {"summary": _LONG_SUMMARY, "score": 42}, (
        f"trailing param dropped on same-chunk close. decoded={decoded!r}"
    )


@pytest.mark.parametrize(
    "param_type",
    ["string", "str", "text", "enum"],
)
def test_string_aliases_all_stream_incrementally(param_type):
    """Schema ``type`` aliases that the parser treats as strings (per
    ``_convert_param_value``) must all trigger incremental emission.
    Prevents drift if the alias set changes.
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool("echo", {"value": {"type": param_type}})
    head = [
        "<tool_call>\n",
        "<function=echo>\n",
        "<parameter=value>\n",
    ]
    value_chunks = [_LONG_SUMMARY[i : i + 32] for i in range(0, len(_LONG_SUMMARY), 32)]
    pre_close_chunks = head + value_chunks

    parser.reset()
    deltas: list[dict] = []
    previous = ""
    for chunk in pre_close_chunks:
        current = previous + chunk
        d = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=chunk,
            request=request,
        )
        if d is not None:
            deltas.append(d)
        previous = current

    value_frags = [f for f in _argument_fragments(deltas) if f not in ("{", "")]
    assert len(value_frags) >= 2, (
        f"type={param_type!r}: expected incremental emission, got "
        f"{len(value_frags)} value-bearing fragments"
    )
