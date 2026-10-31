# SPDX-License-Identifier: Apache-2.0
"""Prose before a tool call is delivered once across chunk boundaries."""

import pytest

from tests.test_streaming_tool_arguments_property import _request, _run
from tests.test_tool_call_streaming_parity import PARITY_FIXTURES

PROSE = "Working on it. "
OPENERS = {
    "minicpm": "<function name=",
    "muse": "<atem:function_calls>",
    "hermes": "<tool_call>",
    "qwen3_xml": "<tool_call>",
    "qwen3_coder_xml": "<tool_call>",
    "hy_v3": "<tool_call:opensource>",
    "ui_tars": "Action:",
    "lfm": "[get_current_weather(",
    "deepseek_v4_flash": "<｜DSML｜tool_calls>",
    "deepseek_v4_0731": "<｜DSML｜tool_calls>",
}


@pytest.mark.parametrize("parser_name,wire_label,text,expected", PARITY_FIXTURES)
@pytest.mark.parametrize("mode", ["one", "prose_then_call", "char"])
def test_prose_before_call(parser_name, wire_label, text, expected, mode, request):
    if parser_name == "hermes" and mode == "one":
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="hermes streaming is pinned to the pre-#5267 rollback (tests/test_truncated_tool_markup_buffered.py::test_streaming_unchanged_against_base); prose + complete call in one chunk drops the prose - follow-up",
            )
        )
    wire = PROSE + text
    chunks = {"one": [wire], "prose_then_call": [PROSE, text], "char": list(wire)}[mode]
    tool_request = _request(*(name for name, _ in expected))
    content = _run(parser_name, wire, tool_request, chunks, return_content=True)
    calls = _run(parser_name, wire, tool_request, chunks)
    assert content.count("Working on it.") == 1
    assert content.lstrip().startswith("Working on it.")
    assert OPENERS[parser_name] not in content
    assert len(calls) == len(expected)


@pytest.mark.parametrize(
    "parser_name,closed",
    [
        ("deepseek_v4_0731", "<｜DSML｜tool_calls></｜DSML｜tool_calls>"),
        ("qwen3_xml", "<tool_call>not a call</tool_call>"),
        (
            "qwen3_xml",
            '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
            "<tool_call>not a call</tool_call>",
        ),
    ],
)
def test_prose_kept_when_closed_block_has_no_call(parser_name, closed):
    from rapid_mlx.tool_parsers import ToolParserManager

    parser = ToolParserManager.get_tool_parser(parser_name)(None)
    previous = PROSE + closed[:-1]
    delta = parser.extract_tool_calls_streaming(
        previous, previous + closed[-1], closed[-1]
    )
    assert not (delta or {}).get("tool_calls")
