# SPDX-License-Identifier: Apache-2.0
"""Protocol contracts for K2 Horizon reasoning and tool output."""

import json
from unittest.mock import MagicMock

import pytest

from vllm_mlx.reasoning import get_parser
from vllm_mlx.reasoning.k2_horizon_parser import K2HorizonReasoningParser
from vllm_mlx.service.postprocessor import StreamingPostProcessor
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


def test_reasoning_preserves_visible_whitespace_verbatim():
    parser = K2HorizonReasoningParser()
    assert parser.extract_reasoning(" private plan </ifm|think>\n  answer  \n") == (
        " private plan ",
        "\n  answer  \n",
    )


def test_reasoning_truncated_implicit_thought_fails_closed():
    parser = K2HorizonReasoningParser()
    for compatibility_flag in (True, False, None):
        assert parser.extract_reasoning(
            "private plan", enable_thinking=compatibility_flag
        ) == (
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


def test_reasoning_parser_stays_active_for_lowest_effort_compatibility():
    assert K2HorizonReasoningParser.sanitize_when_thinking_disabled is True


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
    "closer",
    ["</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"],
)
def test_tool_call_hides_prompt_primed_reasoning_prefix(closer):
    result = K2HorizonToolParser().extract_tool_calls(
        _group(_xml_call("ping"), prefix=f"private plan{closer}"),
        _request(),
    )
    assert result.tools_called
    assert result.content is None


def test_tool_call_preserves_plain_visible_prefix_without_reasoning_boundary():
    result = K2HorizonToolParser().extract_tool_calls(
        _group(_xml_call("ping"), prefix="Visible preface. "),
        _request(),
    )
    assert result.tools_called
    assert result.content == "Visible preface. "


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


def test_tool_call_without_declared_tools_fails_closed():
    output = _group(_xml_call("ping"))
    result = K2HorizonToolParser().extract_tool_calls(output, {"tools": []})
    assert not result.tools_called
    assert result.content == output


def test_typed_argument_must_match_declared_schema():
    output = _group(
        "<ifm|tool_call>lookup"
        "<ifm|arg_key>limit</ifm|arg_key>"
        "<ifm|arg_type>string</ifm|arg_type>"
        "<ifm|arg_value>2</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    result = K2HorizonToolParser().extract_tool_calls(output, _request("xml_typed"))
    assert not result.tools_called
    assert result.content == output


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


@pytest.mark.parametrize("closer", K2HorizonToolParser.REASONING_ENDS)
def test_streaming_complete_group_hides_prompt_primed_reasoning(closer):
    parser = K2HorizonToolParser()
    output = _group(
        _xml_call("ping"), prefix=f"private plan{closer}", suffix="Visible suffix"
    )
    content: list[str] = []
    calls: list[dict] = []
    previous = ""
    for char in output:
        current = previous + char
        delta = parser.extract_tool_calls_streaming(
            previous, current, char, request=_request()
        )
        previous = current
        if delta:
            content.append(delta.get("content") or "")
            calls.extend(delta.get("tool_calls") or [])
    content.append(parser.flush_held_content(output))
    assert "".join(content) == "Visible suffix"
    assert len(calls) == 1


def test_streaming_parses_two_complete_groups_in_one_chunk():
    parser = K2HorizonToolParser()
    output = _group(_xml_call("ping")) + " between " + _group(_xml_call("lookup"))
    delta = parser.extract_tool_calls_streaming("", output, output, request=_request())
    assert delta is not None
    assert delta["content"] == " between "
    assert [call["index"] for call in delta["tool_calls"]] == [0, 1]
    assert [call["function"]["name"] for call in delta["tool_calls"]] == [
        "ping",
        "lookup",
    ]


def test_partial_marker_flushes_without_silent_byte_loss():
    parser = K2HorizonToolParser()
    parser.set_reasoning_sanitized(True)
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


def test_streaming_low_effort_compatibility_separates_reasoning_and_tool_call():
    """K2 ignores enable_thinking=False; the stream must still be sanitized."""
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser_name = None
    cfg.reasoning_parser = K2HorizonReasoningParser()
    cfg.tool_call_parser = None
    cfg.tool_parser_instance = K2HorizonToolParser()
    cfg.enable_auto_tool_choice = True
    processor = StreamingPostProcessor(
        cfg,
        tools_requested=True,
        enable_thinking=False,
        request=_request(),
    )
    processor.reset()

    output = "private plan</ifm|think_faster>" + _group(_xml_call("ping"))
    reasoning: list[str] = []
    content: list[str] = []
    calls: list[dict] = []
    for char in output:
        chunk = MagicMock()
        chunk.new_text = char
        chunk.finished = False
        chunk.channel = None
        chunk.finish_reason = None
        chunk.tool_calls = None
        for event in processor.process_chunk(chunk):
            if event.type == "reasoning":
                reasoning.append(event.reasoning)
            elif event.type == "content":
                content.append(event.content)
            elif event.type == "tool_call":
                calls.extend(event.tool_calls)

    assert "".join(reasoning) == "private plan"
    assert "".join(content) == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "ping"


def test_tool_parser_is_registered():
    assert ToolParserManager.get_tool_parser("k2_horizon") is K2HorizonToolParser
