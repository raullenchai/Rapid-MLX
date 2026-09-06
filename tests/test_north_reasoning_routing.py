# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for Cohere North reasoning routing and wire-token hygiene.

North reasoning itself is parsed by ``CohereCommand4ReasoningParser`` (the
protocol detector registered as ``cohere_command4``/``north``); these tests
cover how that detector composes with ``NorthToolParser`` and the streaming
postprocessor, plus the final-sanitizer exception for North's uppercase
sentinels.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from vllm_mlx.api.utils import sanitize_output
from vllm_mlx.reasoning import get_parser
from vllm_mlx.service.helpers import _finalize_content_and_reasoning
from vllm_mlx.service.postprocessor import StreamingPostProcessor
from vllm_mlx.tool_parsers.cohere_tool_parser import NorthToolParser


def test_uppercase_north_control_tokens_are_stripped_last_mile() -> None:
    assert (
        sanitize_output(
            "<|END_THINKING|><|START_TEXT|>READY<|END_TEXT|>"
            "<|START_ACTION|><|END_ACTION|>"
        )
        == "READY"
    )


def test_unrelated_uppercase_angle_token_is_preserved() -> None:
    text = "Document the literal <|NOT_A_NORTH_TOKEN|> marker."
    assert sanitize_output(text) == text


def test_tool_call_turn_does_not_duplicate_reasoning_as_content() -> None:
    raw = (
        "The user asked for weather, so I should call the tool."
        "<|END_THINKING|>"
        '<|START_ACTION|>{"tool_name":"get_weather",'
        '"parameters":{"city":"Toronto","unit":"celsius"}}<|END_ACTION|>'
    )
    parsed = NorthToolParser(None).extract_tool_calls(raw)

    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=parsed.content or "",
        tool_calls=parsed.tool_calls,
        reasoning_parser=get_parser("north")(),
    )

    assert reasoning == "The user asked for weather, so I should call the tool."
    assert cleaned == ""


def test_non_north_equal_content_and_reasoning_is_not_erased() -> None:
    class NonNorthParser:
        def extract_reasoning(self, _text, **_kwargs):
            return "same legitimate text", None

    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text="same legitimate text",
        cleaned_text="same legitimate text",
        tool_calls=[{"name": "some_tool", "arguments": "{}"}],
        reasoning_parser=NonNorthParser(),
    )

    assert reasoning == "same legitimate text"
    assert cleaned == "same legitimate text"


def test_streaming_postprocessor_keeps_north_parser_active_when_flag_is_false() -> (
    None
):
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = "north"
    cfg.enable_auto_tool_choice = False
    cfg.tool_call_parser = None
    cfg.tool_parser_instance = None
    processor = StreamingPostProcessor(cfg, enable_thinking=False)
    processor.reset()

    def output(text: str, *, finished: bool = False) -> MagicMock:
        chunk = MagicMock()
        chunk.new_text = text
        chunk.finished = finished
        chunk.channel = None
        chunk.finish_reason = "stop" if finished else None
        chunk.tool_calls = None
        return chunk

    reasoning_events = processor.process_chunk(output("Plan."))
    boundary_events = processor.process_chunk(output("<|END_THINKING|>"))
    content_events = processor.process_chunk(output("<|START_TEXT|>READY"))
    finish_events = processor.process_chunk(output("<|END_TEXT|>", finished=True))

    assert [(event.type, event.reasoning) for event in reasoning_events] == [
        ("reasoning", "Plan.")
    ]
    assert boundary_events == []
    assert [(event.type, event.content) for event in content_events] == [
        ("content", "READY")
    ]
    assert len(finish_events) == 1
    assert finish_events[0].type == "finish"
    assert finish_events[0].content is None
    assert finish_events[0].finish_reason == "stop"


def test_streaming_promotes_north_action_emitted_inside_reasoning() -> None:
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = "north"
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "north"
    cfg.tool_parser_instance = None
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object"},
                },
            }
        ]
    }
    processor = StreamingPostProcessor(
        cfg, tools_requested=True, enable_thinking=False, request=request
    )
    processor.reset()
    raw = (
        "plan"
        '<|START_ACTION|>{"tool_name":"read_file","parameters":{}}'
        "<|END_ACTION|>"
        "more plan<|END_THINKING|><|START_TEXT|>READY<|END_TEXT|>"
    )
    output = MagicMock()
    output.new_text = raw
    output.finished = True
    output.channel = None
    output.finish_reason = "stop"
    output.prompt_tokens = 10
    output.completion_tokens = 5
    output.tokens = []
    output.logprobs = None
    output.tool_calls = None

    events = processor.process_chunk(output) + processor.finalize()
    calls = [call for event in events for call in (event.tool_calls or [])]
    visible_content = "".join(event.content or "" for event in events)

    assert [call["function"]["name"] for call in calls] == ["read_file"]
    assert visible_content == "READY"
    assert "plan" not in visible_content
    assert "<|START_ACTION|>" not in visible_content
    assert processor.tool_calls_detected is True


def test_streaming_promotes_split_north_reasoning_action_with_quoted_end() -> None:
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = "north"
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "north"
    cfg.tool_parser_instance = None
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "note_write",
                    "parameters": {"type": "object"},
                },
            }
        ]
    }
    processor = StreamingPostProcessor(
        cfg, tools_requested=True, enable_thinking=False, request=request
    )
    processor.reset()

    def output(text: str, *, finished: bool = False) -> MagicMock:
        chunk = MagicMock()
        chunk.new_text = text
        chunk.finished = finished
        chunk.channel = None
        chunk.finish_reason = "stop" if finished else None
        chunk.prompt_tokens = 10
        chunk.completion_tokens = 5
        chunk.tokens = []
        chunk.logprobs = None
        chunk.tool_calls = None
        return chunk

    chunks = [
        'plan<|START_ACTION|>{"tool_name":"note_write","parameters":{"body":"literal ',
        '<|END_ACTION|> marker"}}<|END_ACTION|>more plan',
        "<|END_THINKING|><|START_TEXT|>READY<|END_TEXT|>",
    ]
    events = []
    for index, chunk in enumerate(chunks):
        events.extend(processor.process_chunk(output(chunk, finished=index == 2)))
    events.extend(processor.finalize())

    calls = [call for event in events for call in (event.tool_calls or [])]
    visible_content = "".join(event.content or "" for event in events)

    assert [call["function"]["name"] for call in calls] == ["note_write"]
    assert json.loads(calls[0]["function"]["arguments"])["body"] == (
        "literal <|END_ACTION|> marker"
    )
    assert visible_content == "READY"


def _north_processor(request):
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = "north"
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "north"
    cfg.tool_parser_instance = None
    processor = StreamingPostProcessor(
        cfg, tools_requested=True, enable_thinking=False, request=request
    )
    processor.reset()
    return processor


def _chunk(text: str, *, finished: bool = False) -> MagicMock:
    chunk = MagicMock()
    chunk.new_text = text
    chunk.finished = finished
    chunk.channel = None
    chunk.finish_reason = "stop" if finished else None
    chunk.prompt_tokens = 10
    chunk.completion_tokens = 5
    chunk.tokens = []
    chunk.logprobs = None
    chunk.tool_calls = None
    return chunk


_READ_FILE_REQUEST = {
    "tools": [
        {
            "type": "function",
            "function": {"name": "read_file", "parameters": {"type": "object"}},
        }
    ]
}


def test_text_only_turn_is_not_replayed_at_finalize() -> None:
    """Marker-free deltas take the fast path without engaging the tool
    parser; ``flush_held_content`` must not replay them at stream end."""
    processor = _north_processor(_READ_FILE_REQUEST)

    events = []
    events.extend(
        processor.process_chunk(
            _chunk("plan<|END_THINKING|><|START_TEXT|>Hello world<|END_TEXT|>")
        )
    )
    events.extend(processor.process_chunk(_chunk("", finished=True)))
    events.extend(processor.finalize())

    visible_content = "".join(event.content or "" for event in events)
    assert visible_content == "Hello world"


def test_plain_prefix_then_action_emits_prefix_once() -> None:
    """A fast-path prefix followed by an action must not be re-emitted when
    the tool parser engages mid-stream with a stale cursor.

    Configured without a reasoning parser so the prefix streams as plain
    content through the marker-free fast path — the exact seam where the
    parser's cursor would otherwise be stale.
    """
    processor = _north_processor(_READ_FILE_REQUEST)
    processor.reasoning_parser = None

    events = []
    events.extend(processor.process_chunk(_chunk("Checking.")))
    events.extend(
        processor.process_chunk(
            _chunk(
                '<|START_ACTION|>{"tool_name":"read_file","parameters":{}}'
                "<|END_ACTION|>",
                finished=True,
            )
        )
    )
    events.extend(processor.finalize())

    calls = [call for event in events for call in (event.tool_calls or [])]
    visible_content = "".join(event.content or "" for event in events)

    assert [call["function"]["name"] for call in calls] == ["read_file"]
    assert visible_content.count("Checking.") == 1
    assert "<|START_ACTION|>" not in visible_content
