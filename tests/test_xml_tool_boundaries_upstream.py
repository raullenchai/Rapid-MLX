# SPDX-License-Identifier: Apache-2.0
"""Boundary regressions for SGLang's qwen3_coder_detector.py rules.

Reference: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/qwen3_coder_detector.py
The first parameter close, next parameter opener, or function close wins.
"""

import json

import pytest

from rapid_mlx.tool_call_scan import split_marked_parameters
from rapid_mlx.tool_parsers import ToolParserManager


@pytest.mark.parametrize("parser_name", ["hermes", "qwen3_coder_xml", "nemotron"])
@pytest.mark.parametrize("size", [None, 1, 7, 10000])
def test_duplicate_parameter_closer(parser_name, size):
    text = (
        "<tool_call><function=edit><parameter=new_string>\nhi-second\n"
        "</parameter>\n</parameter>\n</function></tool_call>"
    )
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "edit",
                    "parameters": {
                        "type": "object",
                        "properties": {"new_string": {"type": "string"}},
                    },
                },
            }
        ]
    }
    parser = ToolParserManager.get_tool_parser(parser_name)(None)
    if size is None:
        calls = parser.extract_tool_calls(text, request).tool_calls
        assert len(calls) == 1
        assert json.loads(calls[0]["arguments"]) == {"new_string": "hi-second"}
    else:
        names, arguments = {}, {}
        previous = ""
        for offset in range(0, len(text), size):
            current = text[: offset + size]
            delta = parser.extract_tool_calls_streaming(
                previous, current, text[offset : offset + size], request=request
            )
            previous = current
            for call in (delta or {}).get("tool_calls", []):
                index = call["index"]
                function = call["function"]
                if function.get("name"):
                    names[index] = function["name"]
                arguments[index] = arguments.get(index, "") + function.get(
                    "arguments", ""
                )
        assert names == {0: "edit"}
        assert json.loads(arguments[0]) == {"new_string": "hi-second"}


def test_duplicate_parameter_closer_scanner():
    assert split_marked_parameters(
        "<parameter=new_string>\nhi-second\n</parameter>\n</parameter>\n",
        r"<parameter=([^>]+)>",
        "</parameter>",
    ) == [("new_string", "hi-second")]


@pytest.mark.parametrize(
    "block,expected",
    [
        ("<parameter=a>one<parameter=b>two</parameter>", [("a", "one"), ("b", "two")]),
        ("<parameter=a>one</function>ignored</parameter>", [("a", "one")]),
        ("<parameter=a>one<parameter=", [("a", "one")]),
        ("<parameter=a>one<parameter=broken</parameter>", [("a", "one")]),
        ("<parameter=broken</parameter><parameter=b>two", [("b", "two")]),
        ("<parameter=a>\n\n one \n\n</parameter>", [("a", "\n one \n")]),
        ("<parameter=a>one</tool_call>two</parameter>", [("a", "one</tool_call>two")]),
        ('<parameter=a>"one</parameter>two"</parameter>', [("a", '"one')]),
    ],
)
def test_scanner_upstream_parameter_boundaries(block, expected):
    assert (
        split_marked_parameters(block, r"<parameter=([^>]+)>", "</parameter>")
        == expected
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "<function=read>first</function><function=read>second</function><function=read>third</function>",
            [("read", "first"), ("read", "second"), ("read", "third")],
        ),
        (
            "<function=read>first<function=other>nested</function>",
            [("read", "first<function=other>nested")],
        ),
        ("<function=read>remaining", [("read", "remaining")]),
        ("<function= read >x</function>", [(" read ", "x")]),
        ("<function=read</function>", []),
        ("<function=read|>x</function>", [("read|", "x")]),
    ],
)
def test_scanner_upstream_call_boundaries(text, expected):
    from rapid_mlx.tool_call_scan import split_marked_calls

    calls = split_marked_calls(text, r"<function=([^>]+)>", "</function>")
    assert [(name, body) for name, body, _, _ in calls] == expected


@pytest.mark.parametrize("parser_name", ["hermes", "qwen3_coder_xml", "nemotron"])
def test_function_closer_cannot_complete_a_partial_function_name(parser_name):
    parser = ToolParserManager.get_tool_parser(parser_name)(None)
    result = parser.extract_tool_calls(
        "<tool_call><function=read</function></tool_call>"
    )
    assert not result.tools_called


@pytest.mark.parametrize("size", [0, 1, 7])
def test_streaming_skips_malformed_parameter_header(size):
    from tests.qwen3coder_stream_harness import non_stream, stream

    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "edit",
                    "parameters": {
                        "type": "object",
                        "properties": {"new_string": {"type": "string"}},
                    },
                },
            }
        ]
    }
    text = "<tool_call><function=edit><parameter=broken</parameter><parameter=new_string>hi-second</parameter></function></tool_call>"
    calls, _ = stream(text, request, size)
    buffered, _ = non_stream(text, request)
    assert calls == buffered
    assert len(calls) == 1
    assert json.loads(calls[0][1]) == {"new_string": "hi-second"}


@pytest.mark.parametrize("parser_name", ["hermes", "nemotron", "qwen3_coder_xml"])
@pytest.mark.parametrize("undeclared", [False, True])
def test_streaming_wrapper_close_does_not_commit_xml_call(parser_name, undeclared):
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "edit",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "string"}},
                    },
                },
            }
        ]
    }
    chunks = [
        "<tool_call><function=edit><parameter=a>"
        + ("x" if undeclared else "before ")
        + "</tool_call>",
        ("</parameter><parameter=undeclared>y" if undeclared else "\n after")
        + "</parameter></function></tool_call>",
    ]
    parser = ToolParserManager.get_tool_parser(parser_name)(None)
    arguments = ""
    previous = ""
    for chunk in chunks:
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous, current, chunk, request=request
        )
        previous = current
        for call in (delta or {}).get("tool_calls", []):
            arguments += call["function"].get("arguments", "")
        if chunk == chunks[0] or undeclared:
            # Qwen may send a provisional header; it must not be executable.
            with pytest.raises(json.JSONDecodeError):
                json.loads(arguments)
    buffered = parser.extract_tool_calls(previous, request)
    if undeclared:
        assert not buffered.tools_called
    else:
        assert len(buffered.tool_calls) == 1
        assert (
            json.loads(arguments)
            == json.loads(buffered.tool_calls[0]["arguments"])
            == {"a": "before </tool_call>\n after"}
        )


@pytest.mark.parametrize("closed", [False, True])
def test_legacy_toolcall_parser_uses_first_close_or_eof(closed):
    from rapid_mlx.api.tool_calling import parse_tool_calls

    text = (
        "<toolcall>edit\n<parameter=a>before <toolcall>nested\n after</parameter>"
        + ("</toolcall>" if closed else "")
    )
    content, calls = parse_tool_calls(text)
    assert not content
    assert len(calls) == 1
    assert calls[0].function.name == "edit"
    assert json.loads(calls[0].function.arguments) == {
        "a": "before <toolcall>nested\n after"
    }
