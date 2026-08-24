"""Malformed Hermes calls stay in the tool loop without being executed."""

import json
from types import SimpleNamespace

import pytest

from tests.test_postprocessor import _make_cfg, _make_output
from vllm_mlx.reasoning.deepseek_r1_parser import DeepSeekR1DistillReasoningParser
from vllm_mlx.service.postprocessor import StreamingPostProcessor
from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browse",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    }
]
REQUEST = {"tools": TOOLS}
MALFORMED = (
    "<tool_call>\n"
    "<function=browse>\n"
    "<parameter=url>\n"
    "https://example.com/guide\n"
    "</tool_call>"
)
MALFORMED_JSON = (
    '<tool_call>{"name":"browse","arguments":{"url":'
    '"https://example.com/guide"</tool_call>'
)


def test_parser_surfaces_declared_malformed_call_with_unparseable_arguments():
    result = HermesToolParser(None).extract_tool_calls(MALFORMED, request=REQUEST)

    assert result.tools_called is True
    assert result.content is None
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["name"] == "browse"
    # The orchestration layer must reject this before dispatch and return its
    # normal parse-error tool result to the model. Do not silently repair and
    # execute a request whose structure the model failed to close.
    try:
        json.loads(call["arguments"])
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("malformed arguments unexpectedly became executable JSON")


def test_parser_accepts_typed_request_tool_models():
    request = SimpleNamespace(
        tools=[SimpleNamespace(function=SimpleNamespace(name="browse"))]
    )

    result = HermesToolParser(None).extract_tool_calls(MALFORMED, request=request)

    assert result.tools_called is True
    assert result.tool_calls[0]["name"] == "browse"


def test_parser_does_not_promote_undeclared_malformed_function():
    result = HermesToolParser(None).extract_tool_calls(
        MALFORMED.replace("function=browse", "function=delete_everything"),
        request=REQUEST,
    )
    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == MALFORMED.replace(
        "function=browse", "function=delete_everything"
    )


def test_stream_does_not_wedge_on_undeclared_malformed_function():
    raw = MALFORMED.replace("function=browse", "function=delete_everything")
    parser = HermesToolParser(None)

    result = parser.extract_tool_calls_streaming("", raw, raw, request=REQUEST)

    assert result == {"content": raw}


def test_parser_surfaces_declared_malformed_json_call_for_executor_feedback():
    result = HermesToolParser(None).extract_tool_calls(MALFORMED_JSON, request=REQUEST)

    assert result.tools_called is True
    assert result.content is None
    assert result.tool_calls[0]["name"] == "browse"
    try:
        json.loads(result.tool_calls[0]["arguments"])
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("malformed JSON unexpectedly became executable")


def test_parser_does_not_promote_name_nested_inside_malformed_arguments():
    nested_name = (
        '<tool_call>{"arguments":{"payload":{"name":"browse"}},'
        '"other":"truncated"</tool_call>'
    )

    result = HermesToolParser(None).extract_tool_calls(nested_name, request=REQUEST)

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == nested_name


def test_stream_finalize_emits_standard_tool_call_not_raw_xml():
    cfg = _make_cfg(
        enable_auto_tool_choice=True, tool_parser_instance=HermesToolParser(None)
    )
    processor = StreamingPostProcessor(cfg, tools_requested=True, request=REQUEST)
    processor.reset()

    streamed = processor.process_chunk(_make_output(MALFORMED))
    finalized = processor.finalize()
    events = [*streamed, *finalized]

    tool_events = [event for event in events if event.type == "tool_call"]
    assert len(tool_events) == 1
    call = tool_events[0].tool_calls[0]
    assert call["function"]["name"] == "browse"
    assert MALFORMED not in "".join(
        event.content or "" for event in events if event.type in {"content", "finish"}
    )


def test_stream_preserves_trailing_prose_once_when_it_shares_the_close_chunk():
    cfg = _make_cfg(
        enable_auto_tool_choice=True, tool_parser_instance=HermesToolParser(None)
    )
    processor = StreamingPostProcessor(cfg, tools_requested=True, request=REQUEST)
    processor.reset()

    events = processor.process_chunk(_make_output(MALFORMED + "\nAfterward."))
    events.extend(processor.finalize())

    assert len([event for event in events if event.type == "tool_call"]) == 1
    content = "".join(
        event.content or "" for event in events if event.type in {"content", "finish"}
    )
    assert content == "\nAfterward."
    assert MALFORMED not in content


def test_stream_preserves_trailing_prose_once_when_it_arrives_later():
    cfg = _make_cfg(
        enable_auto_tool_choice=True, tool_parser_instance=HermesToolParser(None)
    )
    processor = StreamingPostProcessor(cfg, tools_requested=True, request=REQUEST)
    processor.reset()

    events = processor.process_chunk(_make_output(MALFORMED))
    events.extend(processor.process_chunk(_make_output("\nAfterward.")))
    events.extend(processor.finalize())

    assert len([event for event in events if event.type == "tool_call"]) == 1
    content = "".join(
        event.content or "" for event in events if event.type in {"content", "finish"}
    )
    assert content == "\nAfterward."
    assert MALFORMED not in content


def test_stream_valid_call_after_malformed_call_restores_prose_suppression():
    cfg = _make_cfg(
        enable_auto_tool_choice=True, tool_parser_instance=HermesToolParser(None)
    )
    processor = StreamingPostProcessor(cfg, tools_requested=True, request=REQUEST)
    processor.reset()
    valid = (
        '\n<tool_call>{"name":"browse","arguments":'
        '{"url":"https://example.com"}}</tool_call>'
    )

    events = processor.process_chunk(_make_output(MALFORMED))
    events.extend(processor.process_chunk(_make_output(valid)))
    events.extend(processor.process_chunk(_make_output("\nSuppressed.")))
    events.extend(processor.finalize())

    assert (
        sum(
            len(event.tool_calls or []) for event in events if event.type == "tool_call"
        )
        == 2
    )
    content = "".join(
        event.content or "" for event in events if event.type in {"content", "finish"}
    )
    assert "Suppressed." not in content


def test_stream_valid_call_does_not_preserve_same_chunk_trailing_prose():
    cfg = _make_cfg(
        enable_auto_tool_choice=True, tool_parser_instance=HermesToolParser(None)
    )
    processor = StreamingPostProcessor(cfg, tools_requested=True, request=REQUEST)
    processor.reset()
    valid_with_prose = (
        '<tool_call>{"name":"browse","arguments":'
        '{"url":"https://example.com"}}</tool_call>\nSuppressed.'
    )

    events = processor.process_chunk(_make_output(valid_with_prose))
    events.extend(processor.finalize())

    assert len([event for event in events if event.type == "tool_call"]) == 1
    content = "".join(
        event.content or "" for event in events if event.type in {"content", "finish"}
    )
    assert "Suppressed." not in content


def test_stream_malformed_then_valid_in_one_chunk_suppresses_only_final_prose():
    cfg = _make_cfg(
        enable_auto_tool_choice=True, tool_parser_instance=HermesToolParser(None)
    )
    processor = StreamingPostProcessor(cfg, tools_requested=True, request=REQUEST)
    processor.reset()
    valid = (
        '<tool_call>{"name":"browse","arguments":'
        '{"url":"https://example.com"}}</tool_call>'
    )

    events = processor.process_chunk(
        _make_output(MALFORMED + "\nBetween.\n" + valid + "\nSuppressed.")
    )
    events.extend(processor.finalize())

    assert (
        sum(
            len(event.tool_calls or []) for event in events if event.type == "tool_call"
        )
        == 2
    )
    content = "".join(
        event.content or "" for event in events if event.type in {"content", "finish"}
    )
    assert content == "\nBetween.\n"


@pytest.mark.parametrize("mode", ["standard", "channel", "reasoning"])
@pytest.mark.parametrize("finished", [False, True])
def test_stream_preserves_later_prose_across_processing_modes(mode, finished):
    cfg_args = {
        "enable_auto_tool_choice": True,
        "tool_parser_instance": HermesToolParser(None),
    }
    channel = None
    if mode == "channel":
        channel = "content"
    elif mode == "reasoning":
        cfg_args["reasoning_parser"] = DeepSeekR1DistillReasoningParser(None)
    processor = StreamingPostProcessor(
        _make_cfg(**cfg_args), tools_requested=True, request=REQUEST
    )
    processor.reset()

    events = processor.process_chunk(_make_output(MALFORMED, channel=channel))
    events.extend(
        processor.process_chunk(
            _make_output("\nAfterward.", finished=finished, channel=channel)
        )
    )
    events.extend(processor.finalize())

    content = "".join(
        event.content or "" for event in events if event.type in {"content", "finish"}
    )
    assert content == "\nAfterward."
