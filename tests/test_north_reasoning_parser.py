# SPDX-License-Identifier: Apache-2.0
"""Tests for the Cohere North reasoning parser (North-Mini-Code).

Regression (2026-08-20 release dogfood): serving
``mlx-community/North-Mini-Code-1.0-4bit`` shipped the raw chain of
thought and the literal ``<|END_THINKING|><|START_TEXT|>`` markers inside
``message.content`` because no parser understood North's format.
"""

import json
from pathlib import Path

import pytest

from vllm_mlx.reasoning import get_parser
from vllm_mlx.reasoning.north_parser import NorthReasoningParser

END_THINK = "<|END_THINKING|>"
START_TEXT = "<|START_TEXT|>"
END_TEXT = "<|END_TEXT|>"


@pytest.fixture
def parser() -> NorthReasoningParser:
    return NorthReasoningParser()


class TestRegistry:
    def test_north_is_registered(self):
        assert get_parser("north") is NorthReasoningParser

    def test_alias_profiles_wire_north(self):
        aliases = json.loads(
            (Path(__file__).parent.parent / "vllm_mlx" / "aliases.json").read_text()
        )
        for alias in ("north-mini-code-4bit", "north-mini-code-bf16"):
            assert aliases[alias]["reasoning_parser"] == "north"

    def test_auto_config_wires_north_for_raw_hf_paths(self):
        from vllm_mlx.model_auto_config import detect_model_config

        cfg = detect_model_config("mlx-community/North-Mini-Code-1.0-4bit")
        assert cfg is not None
        assert cfg.reasoning_parser == "north"
        assert cfg.tool_call_parser is None


class TestPromptThinkingPredicate:
    def test_north_template_detected_as_prompt_thinking(self):
        from vllm_mlx.service.helpers import _should_start_in_thinking

        template = (
            "{% if add_generation_prompt %}"
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|>"
            "{% endif %}"
        )
        assert _should_start_in_thinking(template, None) is True
        # Explicitly disabled thinking still short-circuits to False.
        assert _should_start_in_thinking(template, False) is False

    def test_think_tag_templates_still_detected(self):
        from vllm_mlx.service.helpers import _should_start_in_thinking

        template = "{% if add_generation_prompt %}<think>\n{% endif %}"
        assert _should_start_in_thinking(template, None) is True


class TestExtractReasoning:
    def test_implicit_think_shape(self, parser):
        # The live dogfood shape: opener lives in the prompt, output is
        # cot + END_THINKING + wrapped answer.
        out = f"This is simple. Answer: 4.{END_THINK}{START_TEXT}4{END_TEXT}"
        reasoning, content = parser.extract_reasoning(out)
        assert reasoning == "This is simple. Answer: 4."
        assert content == "4"

    def test_explicit_both_markers(self, parser):
        out = f"<|START_THINKING|>plan{END_THINK}{START_TEXT}done{END_TEXT}"
        reasoning, content = parser.extract_reasoning(out)
        assert reasoning == "plan"
        assert content == "done"

    def test_no_markers_routes_to_reasoning(self, parser):
        # North templates end the prompt inside <|START_THINKING|>, so a
        # marker-free output is a truncated thought trace.
        reasoning, content = parser.extract_reasoning("half a thought")
        assert reasoning == "half a thought"
        assert content is None

    def test_direct_answer_without_thinking_block(self, parser):
        out = f"{START_TEXT}direct answer{END_TEXT}"
        reasoning, content = parser.extract_reasoning(out)
        assert reasoning is None
        assert content == "direct answer"

    def test_unterminated_text_wrapper(self, parser):
        # Truncated mid-answer: END_TEXT never arrived.
        out = f"cot{END_THINK}{START_TEXT}partial answer"
        reasoning, content = parser.extract_reasoning(out)
        assert reasoning == "cot"
        assert content == "partial answer"

    def test_no_marker_leakage_in_either_channel(self, parser):
        out = f"think{END_THINK}{START_TEXT}answer{END_TEXT}"
        reasoning, content = parser.extract_reasoning(out)
        for channel in (reasoning, content):
            assert channel is not None
            assert "<|" not in channel


class TestTruncationContract:
    def test_open_in_think_before_end_marker(self, parser):
        assert parser.is_open_in_think("some unfinished thought") is True

    def test_not_open_after_end_marker(self, parser):
        assert parser.is_open_in_think(f"thought{END_THINK}answer") is False

    def test_not_open_in_direct_answer_shape(self, parser):
        assert parser.is_open_in_think(f"{START_TEXT}answer") is False

    def test_empty_is_not_open(self, parser):
        assert parser.is_open_in_think("") is False


def _stream(parser, deltas):
    parser.reset_state()
    accumulated = ""
    results = []
    for delta in deltas:
        prev = accumulated
        accumulated += delta
        msg = parser.extract_reasoning_streaming(prev, accumulated, delta)
        if msg:
            results.append(msg)
    reasoning = "".join(m.reasoning for m in results if m.reasoning)
    content = "".join(m.content for m in results if m.content)
    return reasoning, content


class TestStreaming:
    def test_streaming_simple_flow(self, parser):
        reasoning, content = _stream(
            parser,
            ["think", "ing", END_THINK, START_TEXT, "ans", "wer", END_TEXT],
        )
        assert reasoning == "thinking"
        assert content == "answer"

    def test_streaming_text_marker_split_across_deltas(self, parser):
        # START_TEXT arrives split over three deltas glued to answer bytes.
        reasoning, content = _stream(
            parser,
            ["cot", END_THINK, "<|STA", "RT_TE", "XT|>an", "swer", END_TEXT],
        )
        assert reasoning == "cot"
        assert content == "answer"

    def test_streaming_end_text_split_with_glued_bytes(self, parser):
        reasoning, content = _stream(
            parser,
            ["cot", END_THINK, START_TEXT, "answer<|END_", "TEXT|>"],
        )
        assert reasoning == "cot"
        assert content == "answer"

    def test_streaming_no_marker_bytes_leak(self, parser):
        parser.reset_state()
        deltas = ["cot", END_THINK, "<|STA", "RT_TE", "XT|>a", END_TEXT]
        accumulated = ""
        for delta in deltas:
            prev = accumulated
            accumulated += delta
            msg = parser.extract_reasoning_streaming(prev, accumulated, delta)
            if msg and msg.content:
                assert "<|" not in msg.content

    def test_streaming_lone_angle_in_answer_survives(self, parser):
        # A genuine "<" in the answer that never becomes a marker must be
        # flushed by the following delta, not swallowed.
        reasoning, content = _stream(
            parser,
            ["cot", END_THINK, START_TEXT, "a <", " b", END_TEXT],
        )
        assert content == "a < b"
