# SPDX-License-Identifier: Apache-2.0
"""Character-boundary tool-argument parity across registered parser families.

The client concatenates argument deltas by ``index``. A parsed JSON value in
each individual delta is neither required nor sufficient; the concatenation
must equal the non-streaming parser's final JSON string byte for byte.
"""

import json
from unittest.mock import MagicMock

import pytest

from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
from rapid_mlx.service.postprocessor import StreamingPostProcessor
from rapid_mlx.tool_parsers import ToolParserManager
from rapid_mlx.tool_parsers.qwen3coder_tool_parser import (
    Qwen3CoderToolParser,
)
from rapid_mlx.tool_parsers.qwen3coder_tool_parser import (
    _ObjectPairs as QwenObjectPairs,
)
from rapid_mlx.tool_parsers.qwen3coder_tool_parser import (
    _restore_json_value as restore_qwen_json,
)
from rapid_mlx.tool_parsers.seed_oss_tool_parser import (
    _ObjectPairs as SeedObjectPairs,
)
from rapid_mlx.tool_parsers.seed_oss_tool_parser import (
    _restore_json_value as restore_seed_json,
)


def _request(*names):
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "nested": {"type": "object"},
                        },
                    },
                },
            }
            for name in names
        ]
    }


def _run(
    parser_name, wire, request, chunks, *, with_reasoning=False, return_content=False
):
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = Qwen3ReasoningParser() if with_reasoning else None
    cfg.reasoning_parser_name = None
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = parser_name
    cfg.tool_parser_instance = None
    processor = StreamingPostProcessor(cfg, tools_requested=True, request=request)
    processor.reset()
    events = []
    offset = 0
    for chunk in chunks:
        offset += len(chunk)
        events.extend(
            processor.process_chunk(
                GenerationOutput(
                    text=wire[:offset],
                    new_text=chunk,
                    finished=offset == len(wire),
                )
            )
        )
    events.extend(processor.finalize())
    if return_content:
        return "".join(event.content or "" for event in events)
    names = {}
    arguments = {}
    for event in events:
        for call in event.tool_calls or []:
            index = call["index"]
            function = call["function"]
            if function.get("name"):
                names[index] = function["name"]
            arguments[index] = arguments.get(index, "") + function.get("arguments", "")
    return [(names[i], arguments[i]) for i in sorted(arguments)]


HOSTILE_VALUES = (
    'quotes: "hello" and \\slashes',
    "Unicode: 你好 🧪\nnext line\tindented",
    'JSON-looking: {"deep": [true, null, {"n": 1}]}',
    "text with a literal </tool_call> inside",
    "text with a literal </function> inside",
    "  leading and trailing  \n",
)


def _json_wire(name, value):
    return (
        "<tool_call>"
        + json.dumps({"name": name, "arguments": {"text": value}}, ensure_ascii=False)
        + "</tool_call>"
    )


def _xml_wire(name, value):
    return (
        f"<tool_call>\n<function={name}>\n<parameter=text>\n{value}\n"
        "</parameter>\n</function>\n</tool_call>"
    )


@pytest.mark.parametrize("parser_name", ["hermes", "qwen3", "qwen3_coder_xml"])
@pytest.mark.parametrize("value", HOSTILE_VALUES)
def test_argument_deltas_match_nonstream_at_every_character(parser_name, value):
    wire = (
        _xml_wire("lookup", value)
        if parser_name == "qwen3_coder_xml"
        else _json_wire("lookup", value)
    )
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser(parser_name)(None).extract_tool_calls(
        wire, request
    )
    assert expected.tools_called
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    for chunks in (
        list(wire),
        *([wire[:cut], wire[cut:]] for cut in range(1, len(wire))),
    ):
        actual = _run(parser_name, wire, request, chunks)
        assert actual == expected_calls, (parser_name, value, chunks, actual)
        for _, arguments in actual:
            assert isinstance(json.loads(arguments), dict)


@pytest.mark.parametrize("parser_name", ["hermes", "qwen3", "qwen3_coder_xml"])
def test_parallel_calls_after_text(parser_name):
    render = _xml_wire if parser_name == "qwen3_coder_xml" else _json_wire
    wire = (
        "Working. "
        + render("first", HOSTILE_VALUES[0])
        + render("second", HOSTILE_VALUES[1])
    )
    request = _request("first", "second")
    expected = ToolParserManager.get_tool_parser(parser_name)(None).extract_tool_calls(
        wire, request
    )
    assert expected.tools_called and len(expected.tool_calls) == 2
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    for chunks in (list(wire), [wire], [wire[:20], wire[20:]]):
        assert _run(parser_name, wire, request, chunks) == expected_calls


def test_hermes_nested_marker_inside_json_string_does_not_hide_later_call():
    wire = _json_wire("first", "plain") + _json_wire(
        "second", "quoted <tool_call>{ marker inside a JSON string"
    )
    request = _request("first", "second")
    expected = ToolParserManager.get_tool_parser("hermes")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert len(expected_calls) == 2
    for chunks in (list(wire), [wire[: len(wire) // 2], wire[len(wire) // 2 :]]):
        assert _run("hermes", wire, request, chunks) == expected_calls


def test_hermes_nested_xml_opener_inside_json_string():
    wire = _json_wire("first", "plain") + _json_wire(
        "second", "literal <tool_call><function= marker"
    )
    request = _request("first", "second")
    expected = ToolParserManager.get_tool_parser("hermes")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert len(expected_calls) == 2
    for chunks in (list(wire), [wire[: len(wire) // 2], wire[len(wire) // 2 :]]):
        assert _run("hermes", wire, request, chunks) == expected_calls


def test_nemotron_nested_marker_inside_xml_value_does_not_hide_later_call():
    wire = _xml_wire("first", "plain") + _xml_wire(
        "second", "literal <tool_call> inside a value"
    )
    request = _request("first", "second")
    expected = ToolParserManager.get_tool_parser("nemotron")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert len(expected_calls) == 2
    for chunks in (list(wire), [wire[: len(wire) // 2], wire[len(wire) // 2 :]]):
        assert _run("nemotron", wire, request, chunks) == expected_calls


def test_nemotron_bare_call_after_wrapped_call():
    first = _xml_wire("first", "plain")
    second = (
        _xml_wire("second", "plain")
        .replace("<tool_call>", "")
        .replace("</tool_call>", "")
    )
    wire = first + second
    request = _request("first", "second")
    expected = ToolParserManager.get_tool_parser("nemotron")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert len(expected_calls) == 2
    assert _run("nemotron", wire, request, [first, second]) == expected_calls


@pytest.mark.parametrize(
    "parser_name", ["deepseek", "deepseek_v31", "qwen3", "qwen3_coder_xml"]
)
def test_partial_tool_opener_at_end_is_content(parser_name):
    wire = "Compare x <"
    assert (
        _run(parser_name, wire, _request("lookup"), list(wire), return_content=True)
        == wire
    )


def test_qwen_json_tool_after_prose_with_split_opener():
    wire = "Working. " + _json_wire("lookup", "hello")
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser("qwen3")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    chunks = ["Working. <", wire[len("Working. <") :]]
    assert _run("qwen3", wire, request, chunks) == expected_calls
    assert _run("qwen3", wire, request, chunks, return_content=True) == "Working. "


def test_deepseek_tool_after_prose_in_same_chunk():
    wire = "Working. " + CANONICAL_WIRES["deepseek_v31"]
    assert (
        _run("deepseek_v31", wire, _request("lookup"), [wire], return_content=True)
        == "Working. "
    )


def test_deepseek_repeated_close_does_not_replay_arguments():
    first = CANONICAL_WIRES["deepseek"]
    wire = first + "<｜tool▁call▁end｜>"
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser("deepseek")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert (
        _run("deepseek", wire, request, [first, wire[len(first) :]]) == expected_calls
    )


def test_completed_and_partial_opener_flush_behavior():
    for parser_name, complete in (
        ("deepseek", CANONICAL_WIRES["deepseek"]),
        ("seed_oss", CANONICAL_WIRES["seed_oss"]),
    ):
        parser = ToolParserManager.get_tool_parser(parser_name)(None)
        assert parser.flush_held_content(complete) == ""
        assert parser.flush_held_content("Compare x <") == "<"


def test_hermes_escaped_json_string_can_contain_literal_wrapper_close():
    value = 'escaped \\ slash and \\" quote, then </tool_call> inside'
    wire = _json_wire("lookup", value)
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser("hermes")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert _run("hermes", wire, request, list(wire)) == expected_calls


def test_hermes_does_not_close_xml_function_inside_parameter_value():
    parser = ToolParserManager.get_tool_parser("hermes")(None)
    wire = "<tool_call><function=lookup><parameter=text>abc</function>def</parameter>"
    assert (
        parser.extract_tool_calls_streaming("", wire, wire, request=_request("lookup"))
        is None
    )


def test_qwen3_coder_incremental_raw_string_and_invalid_quoted_prefix():
    parser = Qwen3CoderToolParser(None)
    fragment = parser._emit_string_increment("text", 'hello "world"' + "x" * 20)
    assert fragment.startswith('"text": "hello \\"world\\"')
    assert Qwen3CoderToolParser._decoded_json_string_prefix('"bad\nvalue') == ""


def test_qwen3_coder_detects_special_token_opener():
    parser = Qwen3CoderToolParser(None)
    parser.tool_call_start_token_id = 123
    assert parser._new_opener_position("ready", "ready", [123]) == len("ready")


def test_xml_duplicate_key_recovery_preserves_nested_arrays():
    raw = '{"nested":{"items":[{"x":1},2]}}'
    expected = {"nested": {"items": [{"x": 1}, 2]}}
    for pairs_type, restore in (
        (QwenObjectPairs, restore_qwen_json),
        (SeedObjectPairs, restore_seed_json),
    ):
        parsed = json.loads(raw, object_pairs_hook=pairs_type)
        assert restore(parsed) == expected


def test_qwen3_coder_finalizer_avoids_reclosing_covered_parameter():
    parser = Qwen3CoderToolParser(None)
    parser.in_function = True
    parser.in_param = True
    parser.param_count = 1
    parser.json_closed = False
    wire = CANONICAL_WIRES["qwen3_coder_xml"]
    assert parser.finalize_legacy_raw_stream(wire, _request("lookup")) is None


def test_xml_parsers_reenter_completed_header_with_final_arguments():
    for parser_name in ("qwen3_coder_xml", "seed_oss"):
        wire = CANONICAL_WIRES[parser_name]
        if parser_name == "qwen3_coder_xml":
            wire += wire.replace("lookup", "second")
        request = _request("lookup", "second")
        parser = ToolParserManager.get_tool_parser(parser_name)(None)
        parser.is_tool_call_started = True
        parser.is_thinking_end = True
        expected = parser.extract_tool_calls(wire, request)
        delta = parser.extract_tool_calls_streaming(
            wire, wire + "x", "x", request=request
        )
        assert delta and delta["tool_calls"]
        calls = delta["tool_calls"]
        assert [
            (call["function"]["name"], call["function"]["arguments"]) for call in calls
        ] == [(call["name"], call["arguments"]) for call in expected.tool_calls]


def test_seed_oss_valid_call_before_malformed_coalesced_close():
    first = CANONICAL_WIRES["seed_oss"]
    second = (
        "<seed:tool_call><function=second><parameter=text>unfinished"
        "</function></seed:tool_call>"
    )
    wire = first + second
    request = _request("lookup", "second")
    expected = ToolParserManager.get_tool_parser("seed_oss")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    cut = first.index("<parameter=")
    for chunks in (
        [wire[:cut], wire[cut:]],
        list(wire),
        *([wire[:i], wire[i:]] for i in range(1, len(wire))),
    ):
        assert _run("seed_oss", wire, request, chunks) == expected_calls


def test_seed_oss_single_incomplete_parameter_recovers_at_every_boundary():
    wire = (
        "<seed:tool_call><function=lookup><parameter=text>unfinished"
        "</function></seed:tool_call>"
    )
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser("seed_oss")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert json.loads(expected_calls[0][1]) == {"text": "unfinished</function>"}
    for chunks in (list(wire), *([wire[:i], wire[i:]] for i in range(1, len(wire)))):
        assert _run("seed_oss", wire, request, chunks) == expected_calls


@pytest.mark.parametrize(
    "suffix",
    ["", "</parameter>", "</parameter></function>"],
)
def test_seed_oss_truncated_xml_still_closes_streamed_json(suffix):
    wire = "<seed:tool_call><function=lookup><parameter=text>hello" + suffix
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser("seed_oss")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert expected_calls
    for chunks in (list(wire), [wire]):
        actual = _run("seed_oss", wire, request, chunks)
        assert actual == expected_calls
        json.loads(actual[0][1])


def test_seed_oss_wrapped_call_followed_by_bare_function_close():
    first = CANONICAL_WIRES["seed_oss"]
    wire = first + "<function=second></function>"
    request = _request("lookup", "second")
    expected = ToolParserManager.get_tool_parser("seed_oss")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    cut = first.index("<parameter=")
    assert _run("seed_oss", wire, request, [wire[:cut], wire[cut:]]) == expected_calls


def test_seed_oss_malformed_wrapper_does_not_duplicate_next_call():
    wire = (
        "<seed:tool_call><parameter=x>a</parameter></function></seed:tool_call>"
        "<seed:tool_call><function=g><parameter=y>b</parameter>"
        "</function></seed:tool_call>"
    )
    request = _request("g")
    expected = ToolParserManager.get_tool_parser("seed_oss")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert expected_calls == [("g", '{"y": "b"}')]
    for chunks in (list(wire), [wire]):
        assert _run("seed_oss", wire, request, chunks) == expected_calls


def test_seed_oss_missing_parameter_close_keeps_next_function():
    wire = (
        "<seed:tool_call><function=f><parameter=x>a</function>"
        "<function=g><parameter=y>b</parameter></function></seed:tool_call>"
    )
    request = _request("f", "g")
    expected = ToolParserManager.get_tool_parser("seed_oss")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert [call[0] for call in expected_calls] == ["f", "g"]
    for chunks in (list(wire), [wire]):
        assert _run("seed_oss", wire, request, chunks) == expected_calls


def test_seed_oss_literal_wrapper_opener_stays_inside_parameter():
    wire = (
        "<seed:tool_call><function=f><parameter=x>a<seed:tool_call>b"
        "</parameter></function></seed:tool_call>"
    )
    request = _request("f")
    expected = ToolParserManager.get_tool_parser("seed_oss")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert json.loads(expected_calls[0][1]) == {"x": "a<seed:tool_call>b"}
    for chunks in (list(wire), [wire]):
        assert _run("seed_oss", wire, request, chunks) == expected_calls


def test_seed_oss_coalesced_close_keeps_one_available_fragment():
    first = CANONICAL_WIRES["seed_oss"]
    second = first.replace("lookup", "second")
    wire = first + second
    cut = first.index("<parameter=")
    parser = ToolParserManager.get_tool_parser("seed_oss")(None)
    parser.in_function = True
    parser.is_tool_call_started = True
    parser.is_thinking_end = True
    parser.json_started = True
    parser.json_closed = True
    parser.header_sent = True
    delta = parser.extract_tool_calls_streaming(
        wire[:cut], wire, wire[cut:], request=_request("lookup", "second")
    )
    assert delta and delta["tool_calls"][0]["function"]["arguments"]


def test_seed_oss_finalizer_does_not_replay_completed_call():
    parser = ToolParserManager.get_tool_parser("seed_oss")(None)
    wire = CANONICAL_WIRES["seed_oss"]
    parser.prev_tool_call_arr = parser.extract_tool_calls(wire).tool_calls
    assert parser.finalize_legacy_raw_stream(wire) is None


def test_qwen3_coder_plain_content_is_not_replayed_at_end():
    wire = "Hello world"
    assert (
        _run(
            "qwen3_coder_xml", wire, _request("lookup"), list(wire), return_content=True
        )
        == wire
    )


def test_deepseek_v31_large_arguments_survive_suppression_budget():
    wire = (
        _DEEPSEEK_OPEN
        + "lookup<｜tool▁sep｜>"
        + json.dumps({"text": "x" * 70000})
        + _DEEPSEEK_CLOSE
    )
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser("deepseek_v31")(
        None
    ).extract_tool_calls(wire, request)
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    chunks = [wire[i : i + 1024] for i in range(0, len(wire), 1024)]
    assert _run("deepseek_v31", wire, request, chunks) == expected_calls


def test_deepseek_v31_marker_inside_json_is_not_a_tool_name():
    first = _DEEPSEEK_OPEN + "first<｜tool▁sep｜>{}<｜tool▁call▁end｜>"
    payload = json.dumps({"text": "<｜tool▁call▁begin｜>ghost<｜tool▁sep｜>"})
    second = "<｜tool▁call▁begin｜>second<｜tool▁sep｜>" + payload
    wire = first + second + _DEEPSEEK_CLOSE
    request = _request("first", "second", "ghost")
    expected = ToolParserManager.get_tool_parser("deepseek_v31")(
        None
    ).extract_tool_calls(wire, request)
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert (
        _run("deepseek_v31", wire, request, [first, second, _DEEPSEEK_CLOSE])
        == expected_calls
    )


def test_seed_oss_unicode_parameter_name_preserves_json_bytes():
    chunks = [
        "<seed:tool_call><function=lookup>",
        "<parameter=文字>",
        "hello</parameter></function></seed:tool_call>",
    ]
    wire = "".join(chunks)
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser("seed_oss")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert _run("seed_oss", wire, request, chunks) == expected_calls


@pytest.mark.parametrize("parser_name", ["qwen3_coder_xml", "seed_oss"])
def test_parallel_calls_with_distinct_names_in_separate_chunks(parser_name):
    first = CANONICAL_WIRES[parser_name]
    second = first.replace("lookup", "second")
    wire = first + second
    request = _request("lookup", "second")
    expected = ToolParserManager.get_tool_parser(parser_name)(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert len(expected_calls) == 2
    for chunks in ([first, second], [wire]):
        assert _run(parser_name, wire, request, chunks) == expected_calls


@pytest.mark.parametrize(
    "parser_name",
    ["qwen3_coder_xml", "seed_oss", "harmony", "functionary", "glm47", "kimi"],
)
def test_second_call_header_cannot_emit_empty_arguments(parser_name):
    first = CANONICAL_WIRES[parser_name]
    second = first.replace("lookup", "second")
    wire = first + second
    request = _request("lookup", "second")
    expected = ToolParserManager.get_tool_parser(parser_name)(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert len(expected_calls) == 2
    header_end = second.find(">", second.find("second")) + 1
    if parser_name == "harmony":
        header_end = second.find("<|message|>") + len("<|message|>")
    elif parser_name == "glm47":
        header_end = second.find("\n", second.find("second")) + 1
    elif parser_name == "kimi":
        header_end = second.find("<|tool_call_argument_begin|>") + len(
            "<|tool_call_argument_begin|>"
        )
    chunks = [first + second[:header_end], second[header_end:]]
    assert _run(parser_name, wire, request, chunks) == expected_calls


@pytest.mark.parametrize("parser_name", ["qwen3_coder_xml", "seed_oss"])
def test_complete_call_with_leading_text_preserves_content(parser_name):
    wire = "Working. " + CANONICAL_WIRES[parser_name]
    assert (
        _run(parser_name, wire, _request("lookup"), [wire], return_content=True)
        == "Working. "
    )


@pytest.mark.parametrize("parser_name", ["qwen3_coder_xml", "seed_oss"])
def test_finishing_started_call_and_next_call_in_same_chunk(parser_name):
    first = CANONICAL_WIRES[parser_name]
    second = first.replace("lookup", "second")
    split = first.find("<parameter=")
    wire = first + second
    request = _request("lookup", "second")
    expected = ToolParserManager.get_tool_parser(parser_name)(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert (
        _run(parser_name, wire, request, [first[:split], first[split:] + second])
        == expected_calls
    )


def test_qwen3_coder_quoted_value_then_coalesced_call():
    first = _xml_wire("lookup", '"hello"')
    second = _xml_wire("second", '"world"')
    split = first.index("<parameter=")
    wire = first + second
    request = _request("lookup", "second")
    expected = ToolParserManager.get_tool_parser("qwen3_coder_xml")(
        None
    ).extract_tool_calls(wire, request)
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert (
        _run("qwen3_coder_xml", wire, request, [first[:split], first[split:] + second])
        == expected_calls
    )


@pytest.mark.parametrize("key", ['a"b', "文字"])
def test_qwen3_coder_parameter_names_are_json_escaped(key):
    value = 'quoted "value" and 你好' * 3
    wire = (
        f"<tool_call><function=lookup><parameter={key}>"
        + json.dumps(value, ensure_ascii=False)
        + "</parameter></function></tool_call>"
    )
    request = _request("lookup")
    request["tools"][0]["function"]["parameters"]["properties"] = {
        key: {"type": "string"}
    }
    expected = ToolParserManager.get_tool_parser("qwen3_coder_xml")(
        None
    ).extract_tool_calls(wire, request)
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert json.loads(expected_calls[0][1]) == {key: value}
    for chunks in (list(wire), [wire[: len(wire) // 2], wire[len(wire) // 2 :]]):
        assert _run("qwen3_coder_xml", wire, request, chunks) == expected_calls


def test_qwen3_coder_keeps_fragments_before_raw_string_fallback():
    chunks = [
        "<tool_call><function=lookup>",
        '<parameter=a>"x"</parameter><parameter=b>y</parameter></function></tool_call>',
    ]
    wire = "".join(chunks)
    request = _request("lookup")
    request["tools"][0]["function"]["parameters"]["properties"] = {
        "a": {"type": "string"},
        "b": {"type": "string"},
    }
    expected = ToolParserManager.get_tool_parser("qwen3_coder_xml")(
        None
    ).extract_tool_calls(wire, request)
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert _run("qwen3_coder_xml", wire, request, chunks) == expected_calls


def test_qwen3_coder_quoted_parameter_marker_is_value_text():
    wire = (
        '<tool_call><function=lookup><parameter=text>"<parameter=nested>x"'
        "</parameter></function></tool_call>"
    )
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser("qwen3_coder_xml")(
        None
    ).extract_tool_calls(wire, request)
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert json.loads(expected_calls[0][1]) == {"text": "<parameter=nested>x"}
    for chunks in (list(wire), *([wire[:i], wire[i:]] for i in range(1, len(wire)))):
        assert _run("qwen3_coder_xml", wire, request, chunks) == expected_calls


@pytest.mark.parametrize("second", ['"y"', "y"])
def test_qwen3_coder_repeated_parameters_keep_valid_json(second):
    wire = (
        '<tool_call><function=lookup><parameter=text>"x"</parameter>'
        f"<parameter=text>{second}</parameter></function></tool_call>"
    )
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser("qwen3_coder_xml")(
        None
    ).extract_tool_calls(wire, request)
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert json.loads(expected_calls[0][1]) == {"text": "y"}
    for chunks in (list(wire), *([wire[:i], wire[i:]] for i in range(1, len(wire)))):
        actual = _run("qwen3_coder_xml", wire, request, chunks)
        assert actual == expected_calls
        assert json.loads(actual[0][1]) == {"text": "y"}


def test_qwen3_coder_malformed_quoted_suffix_keeps_stream_parity():
    wire = (
        '<tool_call><function=lookup><parameter=text>"abc"junk'
        "</parameter></function></tool_call>"
    )
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser("qwen3_coder_xml")(
        None
    ).extract_tool_calls(wire, request)
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert json.loads(expected_calls[0][1]) == {"text": "abcjunk"}
    for chunks in (list(wire), *([wire[:i], wire[i:]] for i in range(1, len(wire)))):
        assert _run("qwen3_coder_xml", wire, request, chunks) == expected_calls


@pytest.mark.parametrize(
    "value",
    ['"a\\nb\\q"', ' "abc"junk', '"abc'],
)
def test_qwen3_coder_malformed_quoted_value_recovers_at_every_boundary(value):
    wire = (
        f"<tool_call><function=lookup><parameter=text>{value}"
        "</parameter></function></tool_call>"
    )
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser("qwen3_coder_xml")(
        None
    ).extract_tool_calls(wire, request)
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert len(expected_calls) == 1
    json.loads(expected_calls[0][1])
    for chunks in (list(wire), *([wire[:i], wire[i:]] for i in range(1, len(wire)))):
        assert _run("qwen3_coder_xml", wire, request, chunks) == expected_calls


def test_qwen3_coder_missing_parameter_close_keeps_later_declared_parameter():
    wire = (
        "<tool_call><function=lookup><parameter=text>Dallas"
        '<parameter=nested>{"x":1}</parameter></function></tool_call>'
    )
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser("qwen3_coder_xml")(
        None
    ).extract_tool_calls(wire, request)
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert json.loads(expected_calls[0][1])["nested"] == {"x": 1}
    for chunks in (list(wire), *([wire[:i], wire[i:]] for i in range(1, len(wire)))):
        assert _run("qwen3_coder_xml", wire, request, chunks) == expected_calls


def test_seed_oss_literal_wrapper_closer_in_value():
    chunks = [
        "<seed:tool_call><function=lookup>",
        "<parameter=text>a",
        "</seed:tool_call>b</parameter></function></seed:tool_call>",
    ]
    wire = "".join(chunks)
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser("seed_oss")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert json.loads(expected_calls[0][1]) == {"text": "a</seed:tool_call>b"}
    for stream_chunks in (
        chunks,
        list(wire),
        *([wire[:i], wire[i:]] for i in range(1, len(wire))),
    ):
        assert _run("seed_oss", wire, request, stream_chunks) == expected_calls


def test_seed_oss_repeated_parameter_after_nested_json():
    chunks = [
        "<seed:tool_call><function=lookup>",
        '<parameter=text>"x"</parameter>',
        '<parameter=nested>{"x":[1,{"y":2}]}</parameter>',
        "<parameter=text>y</parameter>",
        "</function></seed:tool_call>",
    ]
    wire = "".join(chunks)
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser("seed_oss")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert json.loads(expected_calls[0][1]) == {
        "text": "y",
        "nested": {"x": [1, {"y": 2}]},
    }
    for stream_chunks in (
        chunks,
        list(wire),
        *([wire[:i], wire[i:]] for i in range(1, len(wire))),
    ):
        assert _run("seed_oss", wire, request, stream_chunks) == expected_calls


def test_seed_oss_literal_function_closer_before_parallel_call():
    chunks = [
        '<seed:tool_call><function=lookup><parameter=text>"a</function>b"'
        "</parameter></function></seed:tool_call><seed:tool_call><function=second>",
        "<parameter=text>done</parameter></function></seed:tool_call>",
    ]
    wire = "".join(chunks)
    request = _request("lookup", "second")
    expected = ToolParserManager.get_tool_parser("seed_oss")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert len(expected_calls) == 2
    assert json.loads(expected_calls[0][1]) == {"text": '"a</function>b"'}
    assert json.loads(expected_calls[1][1]) == {"text": "done"}
    for stream_chunks in (
        chunks,
        list(wire),
        *([wire[:i], wire[i:]] for i in range(1, len(wire))),
    ):
        assert _run("seed_oss", wire, request, stream_chunks) == expected_calls


@pytest.mark.parametrize("value", ["<parameter=text>x", "null"])
def test_qwen3_coder_quoted_string_across_parameter_boundaries(value):
    encoded = json.dumps(value)
    chunks = [
        "<tool_call><function=lookup>",
        '<parameter=text>"',
        encoded[1:] + "</parameter></function></tool_call>",
    ]
    wire = "".join(chunks)
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser("qwen3_coder_xml")(
        None
    ).extract_tool_calls(wire, request)
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert json.loads(expected_calls[0][1]) == {"text": value}
    for stream_chunks in (chunks, [chunks[0], chunks[1] + chunks[2]]):
        assert _run("qwen3_coder_xml", wire, request, stream_chunks) == expected_calls


def test_seed_oss_parallel_functions_in_single_wrapper():
    first = CANONICAL_WIRES["seed_oss"]
    second = first.replace("lookup", "second").replace("hello", "world")
    wire = (first + second).replace("</seed:tool_call><seed:tool_call>", "")
    request = _request("lookup", "second")
    expected = ToolParserManager.get_tool_parser("seed_oss")(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert len(expected_calls) == 2
    assert _run("seed_oss", wire, request, [wire]) == expected_calls


def test_seed_oss_split_opener_does_not_leak_into_content():
    first = CANONICAL_WIRES["seed_oss"]
    wire = "Working. " + first
    assert (
        _run(
            "seed_oss",
            wire,
            _request("lookup"),
            ["Working. <", first[1:]],
            return_content=True,
        )
        == "Working. "
    )


@pytest.mark.parametrize("parser_name", ["deepseek_v3", "granite", "xlam"])
def test_parallel_calls_in_shared_envelope(parser_name):
    if parser_name == "deepseek_v3":
        first = CANONICAL_WIRES[parser_name]
        block = first[len("<｜tool▁calls▁begin｜>") : -len("<｜tool▁calls▁end｜>")]
        wire = "<｜tool▁calls▁begin｜>" + block + block + "<｜tool▁calls▁end｜>"
    else:
        first = json.loads("[" + CANONICAL_WIRES[parser_name].split("[", 1)[1])
        # The parser's outer marker, where present, precedes this JSON list.
        wire = ("<|tool_call|>" if parser_name == "granite" else "") + json.dumps(
            first + first
        )
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser(parser_name)(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert len(expected_calls) == 2
    for chunks in (list(wire), [wire]):
        assert _run(parser_name, wire, request, chunks) == expected_calls


@pytest.mark.parametrize("parser_name", ["hermes", "qwen3", "qwen3_coder_xml"])
def test_reasoning_before_tool_call(parser_name):
    render = _xml_wire if parser_name == "qwen3_coder_xml" else _json_wire
    wire = "<think>Plan the tool arguments.</think>" + render(
        "lookup", HOSTILE_VALUES[0]
    )
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser(parser_name)(None).extract_tool_calls(
        wire, request
    )
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    for chunks in (list(wire), [wire[: len(wire) // 2], wire[len(wire) // 2 :]]):
        assert (
            _run(parser_name, wire, request, chunks, with_reasoning=True)
            == expected_calls
        )


# One canonical wire example for each concrete registered parser class. The
# registry check below makes a newly shipped parser add a case here.
_ARGS = '{"text":"hello"}'
_DEEPSEEK_OPEN = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
_DEEPSEEK_CLOSE = "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
CANONICAL_WIRES = {
    "deepseek": (
        _DEEPSEEK_OPEN
        + "function<｜tool▁sep｜>lookup\n```json\n"
        + _ARGS
        + "\n```"
        + _DEEPSEEK_CLOSE
    ),
    "deepseek_v3": (
        _DEEPSEEK_OPEN
        + "function<｜tool▁sep｜>lookup\n```json\n"
        + _ARGS
        + "\n```"
        + _DEEPSEEK_CLOSE
    ),
    "deepseek_v31": (_DEEPSEEK_OPEN + "lookup<｜tool▁sep｜>" + _ARGS + _DEEPSEEK_CLOSE),
    "deepseek_v4_0731": (
        '<｜DSML｜tool_calls><｜DSML｜invoke name="lookup">'
        '<｜DSML｜parameter name="text" string="true">hello'
        "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"
    ),
    "functionary": "<function=lookup>" + _ARGS + "</function>",
    "gemma4": '<|tool_call>call:lookup{text:<|"|>hello<|"|>}<tool_call|>',
    "glm47": (
        "<tool_call>lookup\n<arg_key>text</arg_key>"
        "<arg_value>hello</arg_value></tool_call>"
    ),
    "granite": ('<|tool_call|>[{"name":"lookup","arguments":' + _ARGS + "}]"),
    "harmony": (
        "<|channel|>commentary to=functions.lookup\n"
        "<|constrain|>json\n<|message|>" + _ARGS + "<|call|>"
    ),
    "hermes": _json_wire("lookup", "hello"),
    "hy_v3": (
        "<tool_call:opensource>lookup<tool_sep:opensource>"
        + _ARGS
        + "<end_of_tool_call:opensource>"
    ),
    "k2_horizon": (
        "<ifm|tool_calls><ifm|tool_call>lookup"
        "<ifm|arg_key>text</ifm|arg_key>"
        "<ifm|arg_value>hello</ifm|arg_value>"
        "</ifm|tool_call></ifm|tool_calls>"
    ),
    "kimi": (
        "<|tool_calls_section_begin|><|tool_call_begin|>lookup:0"
        "<|tool_call_argument_begin|>"
        + _ARGS
        + "<|tool_call_end|><|tool_calls_section_end|>"
    ),
    "lfm": '[lookup(text="hello")]',
    "llama": '{"name":"lookup","parameters":' + _ARGS + "}",
    "minicpm": '<function name="lookup"><param name="text">hello</param></function>',
    "minimax": (
        '<minimax:tool_call><invoke name="lookup">'
        '<parameter name="text">hello</parameter>'
        "</invoke></minimax:tool_call>"
    ),
    "mistral": "[TOOL_CALLS]lookup" + _ARGS,
    "muse": (
        '<atem:function_calls><atem:invoke name="lookup">'
        '<atem:parameter name="text">hello</atem:parameter>'
        "</atem:invoke></atem:function_calls>"
    ),
    "nemotron": _xml_wire("lookup", "hello"),
    "qwen3_coder_xml": _xml_wire("lookup", "hello"),
    "qwen3": _json_wire("lookup", "hello"),
    "seed_oss": (
        "<seed:tool_call><function=lookup><parameter=text>hello</parameter>"
        "</function></seed:tool_call>"
    ),
    "ui_tars": "Thought: Click search.\nAction: click(point='<point>200 300</point>')",
    "xlam": '[{"name":"lookup","arguments":' + _ARGS + "}]",
}


def test_canonical_cases_cover_parser_registry():
    registered = {cls.__name__ for cls in ToolParserManager.tool_parsers.values()} - {
        "AutoToolParser"
    }
    covered = {
        ToolParserManager.get_tool_parser(name).__name__ for name in CANONICAL_WIRES
    }
    assert covered == registered


@pytest.mark.parametrize("parser_name,wire", CANONICAL_WIRES.items())
def test_registered_parser_character_boundaries(parser_name, wire):
    request = _request("lookup", "computer")
    request["chat_template_kwargs"] = {"tool_call_format": "xml"}
    expected = ToolParserManager.get_tool_parser(parser_name)(None).extract_tool_calls(
        wire, request
    )
    assert expected.tools_called, parser_name
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    for chunks in (
        list(wire),
        *([wire[:cut], wire[cut:]] for cut in range(1, len(wire))),
    ):
        actual = _run(parser_name, wire, request, chunks)
        assert actual == expected_calls, (parser_name, chunks, actual, expected_calls)
        for _, arguments in actual:
            assert isinstance(json.loads(arguments), dict)


@pytest.mark.parametrize("parser_name,wire", CANONICAL_WIRES.items())
def test_registered_parser_parallel_calls_when_nonstream_supports_them(
    parser_name, wire
):
    request = _request("lookup", "computer")
    combined = wire + wire
    expected = ToolParserManager.get_tool_parser(parser_name)(None).extract_tool_calls(
        combined, request
    )
    if len(expected.tool_calls) != 2:
        pytest.skip("This wire shape does not represent two calls for this parser")
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    for chunks in (list(combined), [wire, wire], [combined]):
        assert _run(parser_name, combined, request, chunks) == expected_calls


_JSON_WIRE_PARSERS = (
    "deepseek",
    "deepseek_v3",
    "deepseek_v31",
    "functionary",
    "granite",
    "harmony",
    "hy_v3",
    "kimi",
    "llama",
    "mistral",
    "xlam",
)


@pytest.mark.parametrize("parser_name", _JSON_WIRE_PARSERS)
@pytest.mark.parametrize("value", HOSTILE_VALUES[:3])
def test_json_wires_preserve_nested_and_escaped_arguments(parser_name, value):
    arguments = {"text": value, "nested": {"items": [1, True, None]}}
    payload = json.dumps(arguments, ensure_ascii=False)
    wire = CANONICAL_WIRES[parser_name].replace(_ARGS, payload)
    request = _request("lookup")
    expected = ToolParserManager.get_tool_parser(parser_name)(None).extract_tool_calls(
        wire, request
    )
    assert expected.tools_called
    expected_calls = [(call["name"], call["arguments"]) for call in expected.tool_calls]
    assert json.loads(expected_calls[0][1]) == arguments
    for chunks in (
        list(wire),
        [wire],
        [wire[: len(wire) // 2], wire[len(wire) // 2 :]],
    ):
        assert _run(parser_name, wire, request, chunks) == expected_calls
