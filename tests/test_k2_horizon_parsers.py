# SPDX-License-Identifier: Apache-2.0
"""Protocol contracts for K2 Horizon reasoning and tool output."""

import json

import pytest

from vllm_mlx.reasoning import get_parser
from vllm_mlx.reasoning.k2_horizon_parser import K2HorizonReasoningParser
from vllm_mlx.tool_parsers import ToolParserManager
from vllm_mlx.tool_parsers.k2_horizon_tool_parser import K2HorizonToolParser

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ping",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _request(wire_format="xml", tool_choice="auto"):
    return {
        "tools": TOOLS,
        "tool_choice": tool_choice,
        "chat_template_kwargs": {"tool_call_format": wire_format},
    }


@pytest.mark.parametrize(
    "start,end",
    K2HorizonReasoningParser.EFFORT_TOKENS,
)
def test_reasoning_all_effort_markers(start, end):
    parser = K2HorizonReasoningParser()
    assert parser.extract_reasoning(f"{start}plan{end}answer") == ("plan", "answer")


def test_reasoning_hands_tool_group_to_content():
    parser = K2HorizonReasoningParser()
    tool = "<ifm|tool_calls><ifm|tool_call>ping</ifm|tool_call></ifm|tool_calls>"
    assert parser.extract_reasoning(f"plan{tool}", enable_thinking=True) == (
        "plan",
        tool,
    )


def test_reasoning_truncated_implicit_thought_fails_closed():
    parser = K2HorizonReasoningParser()
    assert parser.extract_reasoning("private plan", enable_thinking=True) == (
        "private plan",
        None,
    )


def test_reasoning_stream_character_boundaries_do_not_leak_markers():
    parser = K2HorizonReasoningParser()
    reasoning: list[str] = []
    content: list[str] = []
    output = "plan</ifm|think_fast>answer"
    previous = ""
    for char in output:
        current = previous + char
        delta = parser.extract_reasoning_streaming(previous, current, char)
        previous = current
        if delta:
            reasoning.append(delta.reasoning or "")
            content.append(delta.content or "")
    tail = parser.finish_stream()
    if tail:
        reasoning.append(tail.reasoning or "")
        content.append(tail.content or "")
    assert "".join(reasoning) == "plan"
    assert "".join(content) == "answer"


def test_reasoning_parser_is_registered():
    assert get_parser("k2_horizon") is K2HorizonReasoningParser


def _group(*calls: str, prefix="", suffix="") -> str:
    return prefix + "<ifm|tool_calls>" + "".join(calls) + "</ifm|tool_calls>" + suffix


def _xml_call(name: str, args=(), typed=False) -> str:
    parts = [f"<ifm|tool_call>{name}"]
    for key, value in args:
        parts.append(f"<ifm|arg_key>{key}</ifm|arg_key>")
        if typed:
            parts.append(
                f"<ifm|arg_type>{'integer' if key == 'limit' else 'string'}</ifm|arg_type>"
            )
        parts.append(f"<ifm|arg_value>{value}</ifm|arg_value>")
    parts.append("</ifm|tool_call>")
    return "".join(parts)


def _json_call(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"<ifm|tool_call>{payload}</ifm|tool_call>"


def test_xml_default_parses_multiple_calls_and_schema_types():
    parser = K2HorizonToolParser()
    result = parser.extract_tool_calls(
        _group(
            _xml_call("lookup", (("query", "123"), ("limit", "3"))),
            _xml_call("ping"),
            prefix="Before ",
            suffix=" after",
        ),
        _request(),
    )
    assert result.tools_called
    assert result.content == "Before  after"
    assert [call["name"] for call in result.tool_calls] == ["lookup", "ping"]
    assert json.loads(result.tool_calls[0]["arguments"]) == {"query": "123", "limit": 3}


@pytest.mark.parametrize(
    "wire_format,call,expected",
    [
        ("json", _json_call("lookup", {"limit": "2"}), {"limit": 2}),
        (
            "xml_typed",
            _xml_call("lookup", (("query", "weather"), ("limit", "2")), typed=True),
            {"query": "weather", "limit": 2},
        ),
    ],
)
def test_other_documented_formats(wire_format, call, expected):
    result = K2HorizonToolParser().extract_tool_calls(
        _group(call), _request(wire_format)
    )
    assert result.tools_called
    assert json.loads(result.tool_calls[0]["arguments"]) == expected


@pytest.mark.parametrize(
    "output",
    [
        "<ifm|tool_calls><ifm|tool_call>ping",
        _group("junk"),
        _group(_xml_call("unknown")),
        _group(_xml_call("lookup", (("limit", "2"),), typed=True)),
    ],
)
def test_malformed_or_mismatched_calls_fail_closed_as_content(output):
    result = K2HorizonToolParser().extract_tool_calls(output, _request())
    assert not result.tools_called
    assert result.content == output


def test_named_choice_rejects_other_declared_tool():
    result = K2HorizonToolParser().extract_tool_calls(
        _group(_xml_call("ping")),
        _request(tool_choice={"type": "function", "function": {"name": "lookup"}}),
    )
    assert not result.tools_called


def test_streaming_emits_prefix_once_and_call_on_close():
    parser = K2HorizonToolParser()
    request = _request()
    output = _group(_xml_call("lookup", (("limit", "3"),)), prefix="Before ")
    content: list[str] = []
    calls: list[dict] = []
    previous = ""
    for char in output:
        current = previous + char
        delta = parser.extract_tool_calls_streaming(
            previous, current, char, request=request
        )
        previous = current
        if delta:
            content.append(delta.get("content") or "")
            calls.extend(delta.get("tool_calls") or [])
    assert "".join(content) == "Before "
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "lookup"
    assert json.loads(calls[0]["function"]["arguments"]) == {"limit": 3}


def test_partial_marker_flushes_without_silent_byte_loss():
    parser = K2HorizonToolParser()
    text = "ordinary <ifm|tool_"
    previous = ""
    emitted: list[str] = []
    for char in text:
        current = previous + char
        delta = parser.extract_tool_calls_streaming(
            previous, current, char, request=_request()
        )
        previous = current
        if delta:
            emitted.append(delta.get("content") or "")
    emitted.append(parser.flush_held_content(text))
    assert "".join(emitted) == text


def test_tool_parser_is_registered():
    assert ToolParserManager.get_tool_parser("k2_horizon") is K2HorizonToolParser
