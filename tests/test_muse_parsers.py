# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer ATEM tool parser + recipient-channel reasoning parser.

Wire shapes are pinned against ``chat_template.jinja`` on
``meta-models/Muse-Glimmer-30B`` (2026-08-10): tool calls are
Anthropic-style ``<atem:...>`` XML blocks, reasoning rides a
``to=self`` channel, and the template itself warns the output "is not
expected to be valid XML and is parsed with regular expressions" — so
the delimiter-ambiguity cases here (#1730's bug class) are not
paranoia, they are the documented contract.

Streaming cases run per-character where it matters: every sentinel in
this format spans many deltas at char granularity, which is exactly
where prefix-leak bugs (#444/#480) live.
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.reasoning import get_parser
from vllm_mlx.tool_parsers import ToolParserManager

from .parsers.dispatch import run_reasoning_extraction, run_tool_extraction

BOTH_MODES = pytest.mark.parametrize(
    "streaming", [False, True], ids=["nonstream", "stream"]
)


def _tool_parser():
    parser = ToolParserManager.get_tool_parser("muse")(None)
    parser.reset()
    return parser


def _reasoning_parser():
    return get_parser("muse")()


def _chars(text: str) -> list[str]:
    return list(text)


def _block(*invokes: str) -> str:
    return "<atem:function_calls>\n" + "\n".join(invokes) + "\n</atem:function_calls>"


def _invoke(name: str, params: dict[str, str]) -> str:
    lines = [f'<atem:invoke name="{name}">']
    for k, v in params.items():
        lines.append(f'<atem:parameter name="{k}">{v}</atem:parameter>')
    lines.append("</atem:invoke>")
    return "\n".join(lines)


def _request(name: str, properties: dict) -> dict:
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


# ---------------------------------------------------------------------------
# Tool parser — extraction
# ---------------------------------------------------------------------------


@BOTH_MODES
def test_single_call_bare_string_value(streaming):
    text = _block(_invoke("get_weather", {"city": "Paris"}))
    content, calls = run_tool_extraction(
        _tool_parser(), _chars(text), streaming=streaming
    )
    assert [(c.name, json.loads(c.arguments)) for c in calls] == [
        ("get_weather", {"city": "Paris"})
    ]
    assert not (content or "").strip()


@BOTH_MODES
def test_multiple_invokes_in_one_block(streaming):
    text = _block(
        _invoke("read_file", {"path": "/a"}),
        _invoke("read_file", {"path": "/b"}),
    )
    # Both calls close with the same ``</atem:function_calls>`` delta, so
    # the stream finalizes them together — the documented parallel-tool
    # finalization shape.
    _, calls = run_tool_extraction(
        _tool_parser(),
        _chars(text),
        streaming=streaming,
        assert_one_tool_per_delta=False,
    )
    assert [json.loads(c.arguments)["path"] for c in calls] == ["/a", "/b"]


@BOTH_MODES
def test_content_before_block_survives(streaming):
    text = "Let me check.\n" + _block(_invoke("get_weather", {"city": "Oslo"}))
    content, calls = run_tool_extraction(
        _tool_parser(), _chars(text), streaming=streaming
    )
    assert len(calls) == 1
    assert (content or "").strip() == "Let me check."


def test_string_whitespace_is_preserved():
    # The template: "spaces for string values are not stripped."
    parser = _tool_parser()
    text = _block(_invoke("echo", {"text": "  padded  "}))
    result = parser.extract_tool_calls(
        text, request=_request("echo", {"text": {"type": "string"}})
    )
    assert json.loads(result.tool_calls[0]["arguments"]) == {"text": "  padded  "}


def test_multiline_string_value_kept_verbatim():
    value = 'line one\ncan span\n"multiple" lines\n'
    parser = _tool_parser()
    text = _block(_invoke("echo", {"text": value}))
    result = parser.extract_tool_calls(
        text, request=_request("echo", {"text": {"type": "string"}})
    )
    assert json.loads(result.tool_calls[0]["arguments"]) == {"text": value}


def test_schema_typing_scalars_and_containers():
    parser = _tool_parser()
    props = {
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
        "items": {"type": "array"},
        "meta": {"type": "object"},
        "note": {"type": "string"},
    }
    text = _block(
        _invoke(
            "configure",
            {
                "count": "5",
                "ratio": "0.5",
                "flag": "true",
                "items": '["a", "b"]',
                "meta": '{"k": 1}',
                "note": "true",
            },
        )
    )
    result = parser.extract_tool_calls(text, request=_request("configure", props))
    args = json.loads(result.tool_calls[0]["arguments"])
    assert args == {
        "count": 5,
        "ratio": 0.5,
        "flag": True,
        "items": ["a", "b"],
        "meta": {"k": 1},
        # A declared string stays a string even when it spells a boolean.
        "note": "true",
    }


def test_undeclared_param_stays_string_unless_container():
    # No schema: bare scalars must stay strings ("5" could be a real
    # string), containers self-identify as JSON per the template.
    parser = _tool_parser()
    text = _block(_invoke("f", {"a": "5", "b": '{"x": 1}'}))
    result = parser.extract_tool_calls(text, request=None)
    assert json.loads(result.tool_calls[0]["arguments"]) == {"a": "5", "b": {"x": 1}}


def test_literal_closer_inside_string_value():
    # #1730's bug class: a value containing a literal closer must not be
    # truncated at the first one — the value runs to the LAST closer.
    parser = _tool_parser()
    value = "before </atem:parameter> after"
    text = _block(_invoke("echo", {"text": value}))
    result = parser.extract_tool_calls(
        text, request=_request("echo", {"text": {"type": "string"}})
    )
    assert json.loads(result.tool_calls[0]["arguments"]) == {"text": value}


def test_fake_opener_inside_value_filtered_by_schema():
    # An opener whose name the schema does not declare is value text.
    parser = _tool_parser()
    value = 'x <atem:parameter name="fake">y</atem:parameter> z'
    text = _block(_invoke("echo", {"text": value}))
    result = parser.extract_tool_calls(
        text, request=_request("echo", {"text": {"type": "string"}})
    )
    assert json.loads(result.tool_calls[0]["arguments"]) == {"text": value}


def test_null_for_nullable_non_string():
    parser = _tool_parser()
    text = _block(_invoke("f", {"limit": "null"}))
    result = parser.extract_tool_calls(
        text, request=_request("f", {"limit": {"type": "integer"}})
    )
    assert json.loads(result.tool_calls[0]["arguments"]) == {"limit": None}


@BOTH_MODES
def test_no_tools_plain_text_passthrough(streaming):
    content, calls = run_tool_extraction(
        _tool_parser(), _chars("Just an answer."), streaming=streaming
    )
    assert calls == []
    assert content == "Just an answer."


@BOTH_MODES
def test_channel_wrapped_call_extracts_and_strips_plumbing(streaming):
    # Standalone use (no reasoning parser): raw channel output.
    text = (
        " to=self<|message|>Consider the request.<|eom|>"
        "<|start|>assistant to=get_weather<|message|>"
        + _block(_invoke("get_weather", {"city": "Lima"}))
        + "<|eot|>"
    )
    content, calls = run_tool_extraction(
        _tool_parser(), _chars(text), streaming=streaming
    )
    assert [(c.name,) for c in calls] == [("get_weather",)]
    # Neither plumbing nor the to=self reasoning may leak into content —
    # exact comparison, because a substring check let a 2-byte " to"
    # header fragment through during development.
    assert not (content or "").strip()


def test_truncated_block_keeps_bytes_as_content():
    # An opener with no parseable invoke must not vanish silently.
    parser = _tool_parser()
    text = '<atem:function_calls>\n<atem:invoke name="get_w'
    result = parser.extract_tool_calls(text)
    assert not result.tools_called
    assert "atem:invoke" in (result.content or "")


def test_streaming_emits_no_partial_sentinel_bytes():
    parser = _tool_parser()
    text = "Hello " + _block(_invoke("f", {"a": "1"}))
    emitted: list[str] = []
    prev = ""
    for ch in text:
        curr = prev + ch
        delta = parser.extract_tool_calls_streaming(prev, curr, ch)
        if delta and delta.get("content"):
            emitted.append(delta["content"])
        prev = curr
    assert "".join(emitted) == "Hello "


# ---------------------------------------------------------------------------
# Reasoning parser
# ---------------------------------------------------------------------------


@BOTH_MODES
def test_reasoning_then_answer(streaming):
    text = (
        " to=self<|message|>Two plus two is four.<|eom|>"
        "<|start|>assistant to=user<|message|>4<|eot|>"
    )
    reasoning, content = run_reasoning_extraction(
        _reasoning_parser(), _chars(text), streaming=streaming
    )
    assert reasoning == "Two plus two is four."
    assert content == "4"


@BOTH_MODES
def test_bare_message_header_is_user_content(streaming):
    text = "<|message|>Plain answer.<|eot|>"
    reasoning, content = run_reasoning_extraction(
        _reasoning_parser(), _chars(text), streaming=streaming
    )
    assert reasoning is None
    assert content == "Plain answer."


@BOTH_MODES
def test_no_channel_markers_degrades_to_content(streaming):
    reasoning, content = run_reasoning_extraction(
        _reasoning_parser(), _chars("No plumbing at all."), streaming=streaming
    )
    assert reasoning is None
    assert content == "No plumbing at all."


@BOTH_MODES
def test_tool_segment_passes_through_as_content(streaming):
    # The ATEM block must SURVIVE the reasoning split — the tool parser
    # downstream consumes it from the content channel (harmony division).
    block = _block(_invoke("get_weather", {"city": "Rome"}))
    text = (
        " to=self<|message|>Need the weather.<|eom|>"
        "<|start|>assistant to=get_weather<|message|>" + block + "<|eot|>"
    )
    reasoning, content = run_reasoning_extraction(
        _reasoning_parser(), _chars(text), streaming=streaming
    )
    assert reasoning == "Need the weather."
    assert content is not None and block in content


def test_streaming_never_leaks_header_or_terminator_bytes():
    text = (
        " to=self<|message|>thinking<|eom|>"
        "<|start|>assistant to=user<|message|>answer<|eot|>"
    )
    parser = _reasoning_parser()
    parser.reset_state()
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    prev = ""
    for ch in text:
        curr = prev + ch
        msg = parser.extract_reasoning_streaming(prev, curr, ch)
        if msg is not None:
            if msg.reasoning:
                reasoning_parts.append(msg.reasoning)
            if msg.content:
                content_parts.append(msg.content)
        prev = curr
    assert "".join(reasoning_parts) == "thinking"
    assert "".join(content_parts) == "answer"
