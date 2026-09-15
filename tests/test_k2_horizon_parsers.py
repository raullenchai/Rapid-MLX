# SPDX-License-Identifier: Apache-2.0
"""Protocol contracts for K2 Horizon reasoning and tool output."""

import json
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from vllm_mlx.reasoning import get_parser
from vllm_mlx.reasoning.k2_horizon_parser import K2HorizonReasoningParser
from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
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
        "tools": deepcopy(TOOLS),
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


def test_reasoning_protocol_properties_match_default_effort():
    parser = K2HorizonReasoningParser()
    assert parser.reasoning_start_str == "<ifm|think>"
    assert parser.reasoning_end_str == "</ifm|think>"
    assert parser.end_token == "</ifm|think>"


def test_reasoning_stream_accepts_split_and_prefixed_start_markers():
    parser = K2HorizonReasoningParser()
    previous = ""
    marker = "<ifm|think_fast>"
    for char in marker:
        current = previous + char
        assert parser.extract_reasoning_streaming(previous, current, char) is None
        previous = current
    delta = parser.extract_reasoning_streaming(previous, previous + "plan", "plan")
    assert delta is not None and delta.reasoning == "plan"

    parser = K2HorizonReasoningParser()
    delta = parser.extract_reasoning_streaming("", marker + "plan", marker + "plan")
    assert delta is not None and delta.reasoning == "plan"


def test_reasoning_finish_stream_and_implicit_open_state():
    parser = K2HorizonReasoningParser()
    partial = "</ifm|thi"
    assert parser.extract_reasoning_streaming("", partial, partial) is None
    tail = parser.finish_stream()
    assert tail is not None and tail.reasoning == partial
    assert parser.finish_stream() is None
    assert K2HorizonReasoningParser().is_open_in_think("private plan")


@pytest.mark.parametrize("start,end", K2HorizonReasoningParser.EFFORT_TOKENS)
def test_reasoning_open_state_includes_a_bare_generated_start(start, end):
    parser = K2HorizonReasoningParser()
    assert parser.is_open_in_think(start)
    assert parser.is_open_in_think(f"{start}plan")
    assert not parser.is_open_in_think(f"{start}plan{end}")


def test_reasoning_mismatched_generated_closer_fails_closed():
    parser = K2HorizonReasoningParser()
    output = "<ifm|think>private</ifm|think_faster>not visible"
    assert parser.extract_reasoning(output) == (
        "private</ifm|think_faster>not visible",
        None,
    )
    assert parser.is_open_in_think(output)


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
            suffix="private retry</ifm|think> after",
        ),
        _request(),
    )
    assert result.tools_called
    assert result.content == "Before  after"
    assert [call["name"] for call in result.tool_calls] == ["lookup", "ping"]
    assert json.loads(result.tool_calls[0]["arguments"]) == {"query": "123", "limit": 3}


def test_nonstreaming_parses_multiple_groups_without_markup_leak():
    output = _group(
        _xml_call("ping"),
        prefix="Before ",
        suffix="private retry</ifm|think> between ",
    ) + _group(
        _xml_call("lookup"),
        suffix="private final</ifm|think_fast> after",
    )
    result = K2HorizonToolParser().extract_tool_calls(output, _request())
    assert result.tools_called
    assert result.content == "Before  between  after"
    assert [call["name"] for call in result.tool_calls] == ["ping", "lookup"]


def test_nonstreaming_redacts_reasoning_between_tool_groups():
    output = (
        _group(_xml_call("ping"))
        + "private retry</ifm|think_fast>Visible between"
        + _group(_xml_call("lookup"))
    )
    result = K2HorizonToolParser().extract_tool_calls(output, _request())
    assert result.tools_called
    assert result.content == "Visible between"


def test_nonstreaming_discards_unclosed_post_tool_reasoning():
    output = _group(_xml_call("ping"), suffix="private retry without a closer")
    result = K2HorizonToolParser().extract_tool_calls(output, _request())
    assert result.tools_called
    assert result.content is None


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


def test_json_string_may_contain_ifm_closing_delimiters():
    query = "literal </ifm|tool_call> inside value and </ifm|tool_calls> inside value"
    result = K2HorizonToolParser().extract_tool_calls(
        _group(_json_call("lookup", {"query": query})),
        _request("json"),
    )
    assert result.tools_called
    assert json.loads(result.tool_calls[0]["arguments"]) == {"query": query}


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


def test_malformed_call_never_exposes_prompt_primed_reasoning():
    output = "private plan</ifm|think>" + _group(_xml_call("unknown"))
    result = K2HorizonToolParser().extract_tool_calls(output, _request())
    assert not result.tools_called
    assert result.content == _group(_xml_call("unknown"))


def test_streaming_malformed_call_never_exposes_prompt_primed_reasoning():
    parser = K2HorizonToolParser()
    output = "private plan</ifm|think>" + _group(_xml_call("unknown"))
    delta = parser.extract_tool_calls_streaming("", output, output, request=_request())
    assert delta == {"content": _group(_xml_call("unknown"))}


def test_streaming_malformed_group_does_not_strand_later_visible_text():
    parser = K2HorizonToolParser()
    malformed = _group(_xml_call("unknown"))
    first = parser.extract_tool_calls_streaming(
        "", malformed, malformed, request=_request()
    )
    assert first == {"content": malformed}

    current = malformed + "Visible suffix"
    assert parser.extract_tool_calls_streaming(
        malformed, current, "Visible suffix", request=_request()
    ) == {"content": "Visible suffix"}
    assert parser.flush_held_content(current) == ""


def test_streaming_invalid_group_does_not_hide_later_valid_group_in_same_chunk():
    parser = K2HorizonToolParser()
    malformed = _group("junk")
    valid = _group(_xml_call("ping"), prefix=" between ")
    output = malformed + valid
    delta = parser.extract_tool_calls_streaming("", output, output, request=_request())
    assert delta is not None
    assert delta["content"] == malformed + " between "
    assert [call["function"]["name"] for call in delta["tool_calls"]] == ["ping"]
    assert parser.flush_held_content(output) == ""


def test_named_choice_rejects_other_declared_tool():
    result = K2HorizonToolParser().extract_tool_calls(
        _group(_xml_call("ping")),
        _request(tool_choice={"type": "function", "function": {"name": "lookup"}}),
    )
    assert not result.tools_called


def test_tool_choice_none_strips_native_envelope_without_dispatch():
    result = K2HorizonToolParser().extract_tool_calls(
        _group(
            _xml_call("ping"),
            prefix="Visible before ",
            suffix="private retry</ifm|think> after",
        ),
        _request(tool_choice="none"),
    )
    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content == "Visible before  after"


def test_tool_call_without_declared_tools_fails_closed():
    output = _group(_xml_call("ping"))
    result = K2HorizonToolParser().extract_tool_calls(output, {"tools": []})
    assert not result.tools_called
    assert result.content == output


def test_plain_output_without_tool_group_is_untouched():
    result = K2HorizonToolParser().extract_tool_calls("plain answer", _request())
    assert not result.tools_called
    assert result.content == "plain answer"


def test_request_value_uses_default_for_non_mapping_request():
    assert K2HorizonToolParser._request_value(None, "missing", "fallback") == "fallback"


def test_declared_tool_without_parameter_schema_accepts_empty_call():
    request = _request()
    request["tools"][1]["function"].pop("parameters")
    result = K2HorizonToolParser().extract_tool_calls(
        _group(_xml_call("ping")), request
    )
    assert result.tools_called


@pytest.mark.parametrize(
    "body",
    [
        "[]",
        '{"name":"lookup","arguments":[]}',
    ],
    ids=["non-object-payload", "non-object-arguments"],
)
def test_invalid_json_call_shapes_fail_closed(body):
    output = _group(
        K2HorizonToolParser.CALL_START + body + K2HorizonToolParser.CALL_END
    )
    result = K2HorizonToolParser().extract_tool_calls(output, _request("json"))
    assert not result.tools_called
    assert result.content == output


def test_typed_unknown_argument_uses_its_explicit_wire_type():
    call = (
        "<ifm|tool_call>lookup"
        "<ifm|arg_key>extra</ifm|arg_key>"
        "<ifm|arg_type>integer</ifm|arg_type>"
        "<ifm|arg_value>7</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    result = K2HorizonToolParser().extract_tool_calls(
        _group(call), _request("xml_typed")
    )
    assert result.tools_called
    assert json.loads(result.tool_calls[0]["arguments"]) == {"extra": 7}


def test_xml_gap_between_complete_arguments_fails_closed():
    call = (
        "<ifm|tool_call>lookup"
        "<ifm|arg_key>query</ifm|arg_key><ifm|arg_value>x</ifm|arg_value>"
        "junk"
        "<ifm|arg_key>limit</ifm|arg_key><ifm|arg_value>2</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    output = _group(call)
    result = K2HorizonToolParser().extract_tool_calls(output, _request())
    assert not result.tools_called
    assert result.content == output


@pytest.mark.parametrize(
    "group",
    [
        K2HorizonToolParser.GROUP_START + "junk" + K2HorizonToolParser.GROUP_END,
        K2HorizonToolParser.GROUP_START
        + K2HorizonToolParser.CALL_START
        + '{"name":}'
        + K2HorizonToolParser.CALL_END
        + K2HorizonToolParser.GROUP_END,
        K2HorizonToolParser.GROUP_START
        + K2HorizonToolParser.CALL_START
        + '{"name":"ping","arguments":{}}'
        + K2HorizonToolParser.GROUP_END,
        K2HorizonToolParser.GROUP_START + K2HorizonToolParser.GROUP_END,
    ],
    ids=["missing-call-start", "invalid-json", "missing-call-end", "empty"],
)
def test_json_group_framing_rejects_malformed_shapes(group):
    with pytest.raises(ValueError):
        K2HorizonToolParser._parse_group(group, _request("json"))


def test_json_group_framing_accepts_whitespace_and_escapes():
    query = 'quote " and slash \\'
    payload = json.dumps({"name": "lookup", "arguments": {"query": query}})
    group = (
        K2HorizonToolParser.GROUP_START
        + "  "
        + K2HorizonToolParser.CALL_START
        + "  "
        + payload
        + "  "
        + K2HorizonToolParser.CALL_END
        + "  "
        + K2HorizonToolParser.GROUP_END
    )
    result = K2HorizonToolParser().extract_tool_calls(group, _request("json"))
    assert result.tools_called
    assert json.loads(result.tool_calls[0]["arguments"]) == {"query": query}


def test_json_scanner_preserves_split_group_delimiter():
    partial = K2HorizonToolParser.GROUP_END[:-2]
    assert K2HorizonToolParser._scan_json_group_end(
        partial, 0, in_string=False, escape=False
    ) == (None, 0, False, False)


def test_tool_choice_none_strips_incomplete_and_malformed_groups():
    incomplete = "Visible " + K2HorizonToolParser.GROUP_START + "partial"
    result = K2HorizonToolParser().extract_tool_calls(
        incomplete, _request(tool_choice="none")
    )
    assert not result.tools_called
    assert result.content == "Visible "

    malformed = "Visible " + _group("junk")
    result = K2HorizonToolParser().extract_tool_calls(
        malformed, _request(tool_choice="none")
    )
    assert not result.tools_called
    assert result.content == "Visible "


def test_without_tool_groups_handles_plain_incomplete_and_multiple_groups():
    parser = K2HorizonToolParser
    assert parser._without_tool_groups("plain") == "plain"
    assert parser._without_tool_groups("before " + parser.GROUP_START) == "before "
    text = (
        "before "
        + _group(_xml_call("ping"))
        + "private</ifm|think> between "
        + _group(_xml_call("ping"))
        + "private</ifm|think_fast> after"
    )
    assert parser._without_tool_groups(text) == "before  between  after"


def test_unsanitized_direct_stream_flushes_only_proven_public_suffix():
    parser = K2HorizonToolParser()
    assert parser.flush_held_content("private plan") == ""
    assert parser.flush_held_content("private plan</ifm|think>Visible") == "Visible"


@pytest.mark.parametrize(
    "wire_format,call",
    [
        ("xml", _xml_call("bad name")),
        (
            "xml",
            "<ifm|tool_call>lookup<ifm|arg_key>query</ifm|arg_key>junk"
            "<ifm|arg_value>x</ifm|arg_value></ifm|tool_call>",
        ),
        ("xml", _xml_call("lookup", (("query", "a"), ("query", "b")))),
        ("xml_typed", _xml_call("lookup", (("query", "a"),), typed=False)),
        (
            "xml",
            _xml_call("lookup", (("query", "a"),))[: -len("</ifm|tool_call>")]
            + "junk</ifm|tool_call>",
        ),
    ],
    ids=["invalid-name", "gap", "duplicate", "missing-type", "trailing-junk"],
)
def test_additional_malformed_xml_shapes_fail_closed(wire_format, call):
    output = _group(call)
    result = K2HorizonToolParser().extract_tool_calls(output, _request(wire_format))
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


@pytest.mark.parametrize(
    "arguments",
    [{"unexpected": "value"}, {}],
    ids=["additional-properties", "required"],
)
def test_declared_json_schema_contract_fails_closed(arguments):
    request = _request("json")
    request["tools"][0]["function"]["parameters"].update(
        {"additionalProperties": False, "required": ["query"]}
    )
    output = _group(_json_call("lookup", arguments))
    result = K2HorizonToolParser().extract_tool_calls(output, request)
    assert not result.tools_called
    assert result.content == output


def test_declared_xml_required_argument_fails_closed_when_call_is_empty():
    request = _request("xml")
    request["tools"][0]["function"]["parameters"]["required"] = ["query"]
    output = _group(_xml_call("lookup"))
    result = K2HorizonToolParser().extract_tool_calls(output, request)
    assert not result.tools_called
    assert result.content == output


@pytest.mark.parametrize(
    "wire_format,call",
    [
        ("json", _json_call("lookup", {"limit": "not-an-integer"})),
        ("xml", _xml_call("lookup", (("limit", "not-an-integer"),))),
    ],
)
def test_schema_type_violation_after_coercion_fails_closed(wire_format, call):
    output = _group(call)
    result = K2HorizonToolParser().extract_tool_calls(output, _request(wire_format))
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


def test_incomplete_stream_searches_only_the_new_closer_window():
    class ObservedText(str):
        starts: list[tuple[str, int]] = []

        def find(self, sub, start=0, end=None):
            self.starts.append((sub, start))
            return (
                super().find(sub, start)
                if end is None
                else super().find(sub, start, end)
            )

    parser = K2HorizonToolParser()
    request = _request()
    opening = ObservedText(
        "<ifm|tool_calls><ifm|tool_call>lookup<ifm|arg_key>query</ifm|arg_key>"
        "<ifm|arg_value>"
    )
    assert (
        parser.extract_tool_calls_streaming("", opening, str(opening), request=request)
        is None
    )

    previous = str(opening)
    for chunk in ("x" * 10_000, "y" * 10_000):
        current = ObservedText(previous + chunk)
        ObservedText.starts.clear()
        assert (
            parser.extract_tool_calls_streaming(
                previous, current, chunk, request=request
            )
            is None
        )
        closer_searches = [
            start for marker, start in ObservedText.starts if marker == parser.GROUP_END
        ]
        assert closer_searches
        assert closer_searches[0] >= len(current) - len(chunk) - len(parser.GROUP_END)
        previous = str(current)


def test_prompt_primed_stream_searches_only_new_tool_marker_window():
    class ObservedText(str):
        starts: list[tuple[str, int]] = []

        def find(self, sub, start=0, end=None):
            self.starts.append((sub, start))
            return (
                super().find(sub, start)
                if end is None
                else super().find(sub, start, end)
            )

    parser = K2HorizonToolParser()
    previous = ""
    for chunk in ("x" * 10_000, "y" * 10_000):
        current = ObservedText(previous + chunk)
        ObservedText.starts.clear()
        assert (
            parser.extract_tool_calls_streaming(
                previous, current, chunk, request=_request()
            )
            is None
        )
        opener_searches = [
            start
            for marker, start in ObservedText.starts
            if marker == parser.GROUP_START
        ]
        assert opener_searches
        assert opener_searches[0] >= len(previous) - len(parser.GROUP_START)
        previous = str(current)


def test_long_json_stream_advances_incremental_framing_cursor():
    parser = K2HorizonToolParser()
    request = _request("json")
    previous = parser.GROUP_START + parser.CALL_START
    for chunk in (
        '{"name":"lookup","arguments":{"query":"',
        "x" * 10_000,
        "y" * 10_000,
    ):
        current = previous + chunk
        assert (
            parser.extract_tool_calls_streaming(
                previous, current, chunk, request=request
            )
            is None
        )
        assert parser._json_scan_upto >= len(current) - len(parser.GROUP_END) + 1
        previous = current
    closing = '"}}' + parser.CALL_END + parser.GROUP_END
    current = previous + closing
    delta = parser.extract_tool_calls_streaming(
        previous, current, closing, request=request
    )
    assert delta is not None
    assert json.loads(delta["tool_calls"][0]["function"]["arguments"])["query"] == (
        "x" * 10_000 + "y" * 10_000
    )


def test_malformed_unterminated_json_stream_is_visible_at_eof():
    parser = K2HorizonToolParser()
    parser.set_reasoning_sanitized(True)
    output = (
        parser.GROUP_START
        + parser.CALL_START
        + '{"name":"lookup","arguments":{"query":"unterminated}'
        + parser.CALL_END
        + parser.GROUP_END
    )
    assert (
        parser.extract_tool_calls_streaming(
            "", output, output, request=_request("json")
        )
        is None
    )
    assert parser.flush_held_content(output) == output


@pytest.mark.parametrize("closer", K2HorizonToolParser.REASONING_ENDS)
def test_streaming_complete_group_hides_prompt_primed_reasoning(closer):
    parser = K2HorizonToolParser()
    output = _group(
        _xml_call("ping"),
        prefix=f"private plan{closer}",
        suffix=f"private retry{closer}Visible suffix",
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


def test_streaming_discards_unclosed_post_tool_reasoning_at_eof():
    parser = K2HorizonToolParser()
    group = _group(_xml_call("ping"))
    first = parser.extract_tool_calls_streaming("", group, group, request=_request())
    assert first is not None and len(first["tool_calls"]) == 1

    truncated = group + "private retry without a closer"
    assert (
        parser.extract_tool_calls_streaming(
            group, truncated, "private retry without a closer", request=_request()
        )
        is None
    )
    assert parser.flush_held_content(truncated) == ""


def test_streaming_parses_two_complete_groups_in_one_chunk():
    parser = K2HorizonToolParser()
    output = (
        _group(_xml_call("ping"))
        + "private retry</ifm|think> between "
        + _group(_xml_call("lookup"))
    )
    delta = parser.extract_tool_calls_streaming("", output, output, request=_request())
    assert delta is not None
    assert delta["content"] == " between "
    assert [call["index"] for call in delta["tool_calls"]] == [0, 1]
    assert [call["function"]["name"] for call in delta["tool_calls"]] == [
        "ping",
        "lookup",
    ]


def test_streaming_redacts_reasoning_between_tool_groups():
    parser = K2HorizonToolParser()
    output = (
        _group(_xml_call("ping"))
        + "private retry</ifm|think_faster>Visible between"
        + _group(_xml_call("lookup"))
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
    assert "".join(content) == "Visible between"
    assert len(calls) == 2
    assert all(len(call["id"]) == len("call_") + 32 for call in calls)


def test_streaming_buffers_post_tool_content_until_eof():
    parser = K2HorizonToolParser()
    group = _group(_xml_call("ping"))
    first = parser.extract_tool_calls_streaming("", group, group, request=_request())
    assert first is not None and len(first["tool_calls"]) == 1

    private = group + "private retry"
    assert (
        parser.extract_tool_calls_streaming(
            group, private, "private retry", request=_request()
        )
        is None
    )

    boundary = private + "</ifm|think_fast>"
    assert (
        parser.extract_tool_calls_streaming(
            private, boundary, "</ifm|think_fast>", request=_request()
        )
        is None
    )

    visible = boundary + "Visible"
    assert (
        parser.extract_tool_calls_streaming(
            boundary, visible, "Visible", request=_request()
        )
        is None
    )
    more = visible + " now"
    assert (
        parser.extract_tool_calls_streaming(visible, more, " now", request=_request())
        is None
    )
    assert parser.flush_held_content(more) == "Visible now"


def test_streaming_repeated_post_tool_reasoning_reveals_only_final_suffix():
    parser = K2HorizonToolParser()
    group = _group(_xml_call("ping"))
    first = parser.extract_tool_calls_streaming("", group, group, request=_request())
    assert first is not None and len(first["tool_calls"]) == 1

    tail = "private one</ifm|think>Visible oneprivate two</ifm|think_fast>Visible final"
    full = group + tail
    assert (
        parser.extract_tool_calls_streaming(
            group,
            full,
            tail,
            request=_request(),
        )
        is None
    )
    assert parser.flush_held_content(full) == "Visible final"


def test_post_tool_marker_search_advances_with_each_delta():
    class ObservedText(str):
        starts: list[tuple[str, int]] = []

        def find(self, sub, start=0, end=None):
            self.starts.append((sub, start))
            return (
                super().find(sub, start)
                if end is None
                else super().find(sub, start, end)
            )

    parser = K2HorizonToolParser()
    group = _group(_xml_call("ping"))
    first = parser.extract_tool_calls_streaming("", group, group, request=_request())
    assert first is not None and first.get("tool_calls")

    previous = group
    for chunk in ("x" * 10_000, "y" * 10_000):
        current = ObservedText(previous + chunk)
        ObservedText.starts.clear()
        assert (
            parser.extract_tool_calls_streaming(
                previous, current, chunk, request=_request()
            )
            is None
        )
        opener_searches = [
            start
            for marker, start in ObservedText.starts
            if marker == parser.GROUP_START
        ]
        assert opener_searches
        assert opener_searches[0] >= len(current) - len(chunk) - len(parser.GROUP_START)
        previous = str(current)


def test_streaming_tool_choice_none_strips_envelope_incrementally():
    parser = K2HorizonToolParser()
    parser.set_reasoning_sanitized(True)
    output = _group(
        _xml_call("ping"),
        prefix="Before ",
        suffix="private retry</ifm|think> after",
    )
    content: list[str] = []
    calls: list[dict] = []
    previous = ""
    for char in output:
        current = previous + char
        delta = parser.extract_tool_calls_streaming(
            previous,
            current,
            char,
            request=_request(tool_choice="none"),
        )
        previous = current
        if delta:
            content.append(delta.get("content") or "")
            calls.extend(delta.get("tool_calls") or [])
    content.append(parser.flush_held_content(output))
    assert "".join(content) == "Before  after"
    assert calls == []


def test_streaming_tool_choice_none_drops_incomplete_envelope_at_eof():
    parser = K2HorizonToolParser()
    parser.set_reasoning_sanitized(True)
    partial = "Visible before <ifm|tool_calls><ifm|tool_call>ping"
    emitted: list[str] = []
    previous = ""
    for char in partial:
        current = previous + char
        delta = parser.extract_tool_calls_streaming(
            previous,
            current,
            char,
            request=_request(tool_choice="none"),
        )
        previous = current
        if delta:
            emitted.append(delta.get("content") or "")
    emitted.append(parser.flush_held_content(partial))
    assert "".join(emitted) == "Visible before "


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


def test_postprocessor_only_trusts_matching_reasoning_protocol():
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser_name = None
    cfg.reasoning_parser = Qwen3ReasoningParser()
    cfg.tool_call_parser = None
    cfg.tool_parser_instance = K2HorizonToolParser()
    cfg.enable_auto_tool_choice = True
    processor = StreamingPostProcessor(
        cfg,
        tools_requested=True,
        enable_thinking=True,
        request=_request(),
    )
    assert isinstance(processor.tool_parser, K2HorizonToolParser)
    assert processor.tool_parser._input_reasoning_sanitized is False


def test_postprocessor_trusts_matching_k2_reasoning_protocol():
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
        enable_thinking=True,
        request=_request(),
    )
    assert isinstance(processor.tool_parser, K2HorizonToolParser)
    assert processor.tool_parser._input_reasoning_sanitized is True


def test_postprocessor_never_leaks_reasoning_after_a_tool_group():
    """The upstream reasoning parser stops at the first native tool group."""
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
        enable_thinking=True,
        request=_request(),
    )
    processor.reset()

    output = (
        "initial plan</ifm|think>"
        + _group(_xml_call("ping"))
        + "private retry</ifm|think_fast>Visible recovery"
    )
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
            if event.type == "content":
                content.append(event.content)
            elif event.type == "tool_call":
                calls.extend(event.tool_calls)

    for event in processor.finalize():
        if event.type == "content":
            content.append(event.content)

    assert "".join(content) == "Visible recovery"
    assert "private retry" not in "".join(content)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "ping"


def test_tool_parser_is_registered():
    assert ToolParserManager.get_tool_parser("k2_horizon") is K2HorizonToolParser
