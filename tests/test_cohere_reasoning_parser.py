# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for Cohere North reasoning and wire-token hygiene."""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm_mlx.api.utils import sanitize_output
from vllm_mlx.reasoning import DeltaMessage, get_parser
from vllm_mlx.reasoning.cohere_parser import CohereReasoningParser
from vllm_mlx.service.helpers import _finalize_content_and_reasoning
from vllm_mlx.service.postprocessor import StreamingPostProcessor
from vllm_mlx.tool_parsers.cohere_tool_parser import CohereToolParser


def test_cohere_reasoning_parser_is_registered() -> None:
    assert get_parser("cohere") is CohereReasoningParser


def test_implicit_start_splits_reasoning_from_final_text() -> None:
    parser = CohereReasoningParser()

    reasoning, content = parser.extract_reasoning(
        "Check the instruction carefully."
        "<|END_THINKING|><|START_TEXT|>READY<|END_TEXT|>"
    )

    assert reasoning == "Check the instruction carefully."
    assert sanitize_output(content) == "READY"


def test_explicit_start_splits_reasoning_from_final_text() -> None:
    parser = CohereReasoningParser()

    reasoning, content = parser.extract_reasoning(
        "<|START_THINKING|>Plan.<|END_THINKING|><|START_TEXT|>Done.<|END_TEXT|>"
    )

    assert reasoning == "Plan."
    assert sanitize_output(content) == "Done."


def test_streaming_routes_implicit_reasoning_then_content() -> None:
    parser = CohereReasoningParser()
    parser.reset_state()

    reasoning = parser.extract_reasoning_streaming("", "Plan.", "Plan.")
    boundary = parser.extract_reasoning_streaming(
        "Plan.",
        "Plan.<|END_THINKING|>",
        "<|END_THINKING|>",
    )
    content = parser.extract_reasoning_streaming(
        "Plan.<|END_THINKING|>",
        "Plan.<|END_THINKING|><|START_TEXT|>READY",
        "<|START_TEXT|>READY",
    )

    assert reasoning == DeltaMessage(reasoning="Plan.")
    assert boundary is None
    assert content is not None
    assert sanitize_output(content.content) == "READY"


def test_uppercase_north_control_tokens_are_stripped_last_mile() -> None:
    assert (
        sanitize_output(
            "<|END_THINKING|><|START_TEXT|>READY<|END_TEXT|>"
            "<|START_ACTION|><|END_ACTION|>"
        )
        == "READY"
    )


def test_tool_call_turn_does_not_duplicate_reasoning_as_content() -> None:
    raw = (
        "The user asked for weather, so I should call the tool."
        "<|END_THINKING|>"
        '<|START_ACTION|>{"tool_name":"get_weather",'
        '"parameters":{"city":"Toronto","unit":"celsius"}}<|END_ACTION|>'
    )
    parsed = CohereToolParser(None).extract_tool_calls(raw)

    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=parsed.content or "",
        tool_calls=parsed.tool_calls,
        reasoning_parser=CohereReasoningParser(),
    )

    assert reasoning == "The user asked for weather, so I should call the tool."
    assert cleaned == ""


def test_streaming_postprocessor_keeps_cohere_parser_active_when_flag_is_false() -> (
    None
):
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = "cohere"
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
