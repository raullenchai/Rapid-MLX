"""Pin restored v0.16.9 parser branches exercised by the streaming revert."""

import json

import pytest

from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
from rapid_mlx.tool_parsers.hermes_tool_parser import HermesToolParser


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_qwen_empty_think_opener_has_no_finalization_delta(finish_reason):
    parser = Qwen3ReasoningParser()

    # An opener alone contains neither answer nor reasoning to replay at EOF.
    assert parser.finalize_streaming("<think>", finish_reason=finish_reason) is None


@pytest.mark.parametrize("suffix", ["}", "} \n"])
def test_hermes_streaming_raw_json_emits_declared_call_at_closing_brace(suffix):
    parser = HermesToolParser(None)
    request = {"tools": [{"type": "function", "function": {"name": "get_value"}}]}
    prefix = '{"name": "get_value", "arguments": {"key": "first"}'

    assert (
        parser.extract_tool_calls_streaming(
            "", prefix[:-1], prefix[:-1], request=request
        )
        is None
    )
    result = parser.extract_tool_calls_streaming(
        prefix[:-1], prefix + suffix, "}" + suffix, request=request
    )

    assert result is not None
    assert set(result) == {"tool_calls"}
    assert len(result["tool_calls"]) == 1
    call = result["tool_calls"][0]
    assert call["index"] == 0
    assert call["id"]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_value"
    assert json.loads(call["function"]["arguments"]) == {"key": "first"}
