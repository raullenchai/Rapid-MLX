"""Integration test for the Anthropic streaming pipeline.

Tests the full flow: raw model output → StreamingToolCallFilter → StreamingThinkRouter
→ Anthropic SSE events, verifying block transitions, tool call extraction, and
prompt_tokens tracking work together correctly.
"""

import json
import unittest

from vllm_mlx.api.utils import StreamingThinkRouter, StreamingToolCallFilter
from vllm_mlx.routes.anthropic import _should_start_in_thinking
from vllm_mlx.server import _emit_content_pieces


class TestEmitContentPieces(unittest.TestCase):
    """Test the refactored _emit_content_pieces helper."""

    def test_single_text_block(self):
        events, block_type, index = _emit_content_pieces([("text", "hello")], None, 0)
        assert len(events) == 2  # block_start + delta
        assert block_type == "text"
        assert index == 0
        # Verify block_start
        start_data = json.loads(events[0].split("data: ")[1])
        assert start_data["type"] == "content_block_start"
        assert start_data["content_block"]["type"] == "text"
        # Verify delta
        delta_data = json.loads(events[1].split("data: ")[1])
        assert delta_data["delta"]["text"] == "hello"

    def test_single_thinking_block(self):
        events, block_type, index = _emit_content_pieces(
            [("thinking", "reasoning")], None, 0
        )
        assert block_type == "thinking"
        delta_data = json.loads(events[1].split("data: ")[1])
        assert delta_data["delta"]["thinking"] == "reasoning"

    def test_transition_thinking_to_text(self):
        events, block_type, index = _emit_content_pieces(
            [("thinking", "reason"), ("text", "answer")], None, 0
        )
        assert block_type == "text"
        assert index == 1  # incremented on block transition
        # Should have: start_thinking, delta_thinking, stop_thinking, start_text, delta_text
        assert len(events) == 5
        stop_data = json.loads(events[2].split("data: ")[1])
        assert stop_data["type"] == "content_block_stop"

    def test_continues_existing_block(self):
        """If current_block_type matches, no start/stop emitted."""
        events, block_type, index = _emit_content_pieces([("text", "more")], "text", 0)
        assert len(events) == 1  # just delta, no start
        assert block_type == "text"

    def test_empty_pieces(self):
        events, block_type, index = _emit_content_pieces([], None, 0)
        assert events == []
        assert block_type is None
        assert index == 0


class TestAnthropicThinkingStartDecision(unittest.TestCase):
    """Test when Anthropic streaming should start inside a thinking block."""

    THINKING_TEMPLATE = "{% if add_generation_prompt %}<think>{% endif %}"

    def test_thinking_template_starts_in_thinking_by_default(self):
        """Thinking-capable templates keep implicit thinking support by default."""
        assert _should_start_in_thinking(self.THINKING_TEMPLATE, None) is True

    def test_thinking_template_starts_in_thinking_when_explicitly_enabled(self):
        """Explicit thinking preserves the implicit-template routing behavior."""
        assert _should_start_in_thinking(self.THINKING_TEMPLATE, True) is True

    def test_thinking_template_does_not_start_in_thinking_when_disabled(self):
        """No-thinking mode must emit direct answer tokens as text, not thinking."""
        assert _should_start_in_thinking(self.THINKING_TEMPLATE, False) is False

    def test_1570_unconditional_distill_template_ignores_disabled_flag(self):
        assert (
            _should_start_in_thinking(self.THINKING_TEMPLATE, False, unconditional=True)
            is True
        )

    def test_1570_conditional_distill_template_respects_disabled_flag(self):
        conditional = (
            "{% if add_generation_prompt and enable_thinking %}<think>{% endif %}"
        )
        assert (
            _should_start_in_thinking(conditional, False, unconditional=True) is False
        )

    def test_1570_conditional_distill_template_uses_enabled_default(self):
        conditional = (
            "{% if add_generation_prompt and enable_thinking %}<think>{% endif %}"
        )
        assert _should_start_in_thinking(conditional, None, unconditional=True) is True

    def test_1570_nested_conditional_distill_template_respects_disabled_flag(self):
        conditional = (
            "{% if enable_thinking %}"
            "{% if add_generation_prompt %}<think>{% endif %}"
            "{% endif %}"
        )
        assert (
            _should_start_in_thinking(conditional, False, unconditional=True) is False
        )

    def test_1570_whitespace_control_condition_respects_disabled_flag(self):
        conditional = (
            "{%- if enable_thinking %}"
            "{%- if add_generation_prompt %}<think>{%- endif %}"
            "{%- endif %}"
        )
        assert (
            _should_start_in_thinking(conditional, False, unconditional=True) is False
        )

    def test_1570_false_branch_can_still_prime_implicit_thinking(self):
        conditional = (
            "{% if enable_thinking %}plain{% else %}"
            "{% if add_generation_prompt %}<think>{% endif %}"
            "{% endif %}"
        )
        assert _should_start_in_thinking(conditional, False, unconditional=True) is True

    def test_1570_inactive_false_branch_does_not_prime_when_enabled(self):
        conditional = (
            "{% if enable_thinking %}plain{% else %}"
            "{% if add_generation_prompt %}<think>{% endif %}"
            "{% endif %}"
        )
        assert _should_start_in_thinking(conditional, True, unconditional=True) is False

    def test_1570_inactive_elif_marker_does_not_prime_thinking(self):
        conditional = (
            "{% if not enable_thinking %}plain"
            "{% elif add_generation_prompt %}<think>"
            "{% endif %}"
        )
        assert (
            _should_start_in_thinking(conditional, False, unconditional=True) is False
        )

    def test_1570_jinja_assignment_can_activate_marker(self):
        template = (
            "{% set prime = true %}"
            "{% if prime and add_generation_prompt %}<think>{% endif %}"
        )
        assert _should_start_in_thinking(template, False, unconditional=True) is True

    def test_1570_marker_in_empty_loop_is_inactive(self):
        template = (
            "{% for item in [] %}<think>{% endfor %}"
            "{% if add_generation_prompt %}plain{% endif %}"
        )
        assert _should_start_in_thinking(template, False, unconditional=True) is False

    def test_1570_marker_in_unused_macro_is_inactive(self):
        template = (
            "{% macro prime() %}<think>{% endmacro %}"
            "{% if add_generation_prompt %}plain{% endif %}"
        )
        assert _should_start_in_thinking(template, False, unconditional=True) is False

    def test_1570_marker_in_jinja_comment_is_inactive(self):
        template = "{# <think> #}{% if add_generation_prompt %}plain{% endif %}"
        assert _should_start_in_thinking(template, False, unconditional=True) is False

    def test_1570_unrenderable_template_does_not_suppress_content(self):
        template = (
            "{% if missing_global() %}<think>{% endif %}"
            "{% if add_generation_prompt %}plain{% endif %}"
        )
        assert _should_start_in_thinking(template, False, unconditional=True) is False

    def test_1570_rendered_marker_does_not_require_generation_variable(self):
        assert _should_start_in_thinking("prefix<think>", False, unconditional=True)

    def test_1570_closed_rendered_block_does_not_prime_generation(self):
        template = "prefix<think>private</think>answer-prefix"
        assert _should_start_in_thinking(template, False, unconditional=True) is False

    def test_1570_unrelated_enable_variable_does_not_hide_unconditional_marker(self):
        template = (
            "{% set enable_thinking = false %}"
            "{% if add_generation_prompt %}<think>{% endif %}"
        )
        assert _should_start_in_thinking(template, False, unconditional=True) is True

    def test_1570_named_default_template_is_resolved(self):
        templates = {"default": self.THINKING_TEMPLATE, "tool_use": "plain"}
        assert _should_start_in_thinking(templates, None) is True

    def test_1570_named_tool_template_is_selected_when_tools_requested(self):
        templates = {"default": self.THINKING_TEMPLATE, "tool_use": "plain"}
        assert (
            _should_start_in_thinking(
                templates, None, unconditional=True, tools_requested=True
            )
            is False
        )

    def test_1570_named_tool_template_can_enable_implicit_thinking(self):
        templates = {"default": "plain", "tool_use": self.THINKING_TEMPLATE}
        assert (
            _should_start_in_thinking(
                templates, None, unconditional=True, tools_requested=True
            )
            is True
        )

    def test_plain_template_never_starts_in_thinking(self):
        """Templates without an implicit think marker should start as text."""
        assert _should_start_in_thinking("{{ add_generation_prompt }}", None) is False


class TestStreamingPipelineIntegration(unittest.TestCase):
    """Integration test for the full streaming pipeline."""

    def _run_pipeline(self, deltas, start_in_thinking=False):
        """Run deltas through tool_filter → think_router → emit, return events."""
        tool_filter = StreamingToolCallFilter()
        think_router = StreamingThinkRouter(start_in_thinking=start_in_thinking)
        current_block_type = None
        block_index = 0
        all_events = []
        accumulated_text = ""

        for delta in deltas:
            accumulated_text += delta
            filtered = tool_filter.process(delta)
            if not filtered:
                continue
            pieces = think_router.process(filtered)
            events, current_block_type, block_index = _emit_content_pieces(
                pieces, current_block_type, block_index
            )
            all_events.extend(events)

        # Flush
        remaining = tool_filter.flush()
        if remaining:
            pieces = think_router.process(remaining)
            events, current_block_type, block_index = _emit_content_pieces(
                pieces, current_block_type, block_index
            )
            all_events.extend(events)

        flush_pieces = think_router.flush()
        if flush_pieces:
            events, current_block_type, block_index = _emit_content_pieces(
                flush_pieces, current_block_type, block_index
            )
            all_events.extend(events)

        # Close final block
        if current_block_type is not None:
            all_events.append(
                f"event: content_block_stop\ndata: "
                f"{json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
            )
            block_index += 1

        return all_events, accumulated_text, block_index

    def _parse_events(self, events):
        """Parse SSE events into structured data."""
        parsed = []
        for event in events:
            data_line = event.split("data: ", 1)[1].split("\n")[0]
            parsed.append(json.loads(data_line))
        return parsed

    def test_pure_text_response(self):
        """Simple text response - one text block."""
        events, _, block_index = self._run_pipeline(["Hello ", "world!"])
        parsed = self._parse_events(events)

        # block_start, 2 deltas, block_stop
        types = [p["type"] for p in parsed]
        assert types[0] == "content_block_start"
        assert parsed[0]["content_block"]["type"] == "text"
        assert types[-1] == "content_block_stop"
        assert block_index == 1

    def test_thinking_then_text(self):
        """Model thinks then responds."""
        events, _, block_index = self._run_pipeline(
            ["<think>Let me think", " about this</think>", "The answer is 42"]
        )
        parsed = self._parse_events(events)

        block_starts = [p for p in parsed if p["type"] == "content_block_start"]
        assert len(block_starts) == 2
        assert block_starts[0]["content_block"]["type"] == "thinking"
        assert block_starts[1]["content_block"]["type"] == "text"
        assert block_index == 2

    def test_start_in_thinking_then_text(self):
        """Model starts in thinking mode (template injects <think>)."""
        events, _, _ = self._run_pipeline(
            ["reasoning here", "</think>", "The answer"],
            start_in_thinking=True,
        )
        parsed = self._parse_events(events)

        block_starts = [p for p in parsed if p["type"] == "content_block_start"]
        assert len(block_starts) == 2
        assert block_starts[0]["content_block"]["type"] == "thinking"
        assert block_starts[1]["content_block"]["type"] == "text"

    def test_no_thinking_template_plain_answer_streams_as_text(self):
        """Disabled thinking should not turn direct answer tokens into thinking."""
        start_in_thinking = _should_start_in_thinking(
            TestAnthropicThinkingStartDecision.THINKING_TEMPLATE,
            enable_thinking=False,
        )
        events, _, block_index = self._run_pipeline(
            ["Direct ", "answer"],
            start_in_thinking=start_in_thinking,
        )
        parsed = self._parse_events(events)

        block_starts = [p for p in parsed if p["type"] == "content_block_start"]
        text_deltas = [
            p["delta"]["text"]
            for p in parsed
            if p["type"] == "content_block_delta"
            and p["delta"].get("type") == "text_delta"
        ]

        assert len(block_starts) == 1
        assert block_starts[0]["content_block"]["type"] == "text"
        assert "".join(text_deltas) == "Direct answer"
        assert block_index == 1

    def test_text_then_tool_call(self):
        """Text followed by tool call - tool markup suppressed from text."""
        events, accumulated, _ = self._run_pipeline(
            [
                "I'll search for that. ",
                "<minimax:tool_call>",
                '<invoke name="bash">',
                '<parameter name="command">ls /tmp</parameter>',
                "</invoke>",
                "</minimax:tool_call>",
            ]
        )
        parsed = self._parse_events(events)

        # Only text block should appear (tool call is suppressed from streaming)
        text_deltas = [
            p
            for p in parsed
            if p["type"] == "content_block_delta"
            and p["delta"].get("type") == "text_delta"
        ]
        text_content = "".join(d["delta"]["text"] for d in text_deltas)
        assert "I'll search for that." in text_content
        assert "<minimax:tool_call>" not in text_content

        # But accumulated text has the full tool call for parsing
        assert "<minimax:tool_call>" in accumulated

    def test_thinking_then_tool_call(self):
        """Thinking followed by tool call - both properly routed."""
        events, accumulated, _ = self._run_pipeline(
            [
                "<think>I need to search</think>",
                "<minimax:tool_call>",
                '<invoke name="search">',
                '<parameter name="q">test</parameter>',
                "</invoke>",
                "</minimax:tool_call>",
            ]
        )
        parsed = self._parse_events(events)

        block_starts = [p for p in parsed if p["type"] == "content_block_start"]
        # Only thinking block (tool call is suppressed)
        assert len(block_starts) == 1
        assert block_starts[0]["content_block"]["type"] == "thinking"

    def test_mixed_thinking_text_and_tool_call(self):
        """Full scenario: thinking → text → tool call."""
        events, accumulated, block_index = self._run_pipeline(
            [
                "<think>analyzing request</think>",
                "Let me help. ",
                "<minimax:tool_call>",
                '<invoke name="bash"><parameter name="cmd">echo hi</parameter></invoke>',
                "</minimax:tool_call>",
            ]
        )
        parsed = self._parse_events(events)

        block_starts = [p for p in parsed if p["type"] == "content_block_start"]
        # thinking block + text block (tool call suppressed)
        assert len(block_starts) == 2
        assert block_starts[0]["content_block"]["type"] == "thinking"
        assert block_starts[1]["content_block"]["type"] == "text"

        # Accumulated has everything for post-stream tool parsing
        assert "<minimax:tool_call>" in accumulated

    def test_block_index_increments_correctly(self):
        """Block indices should increment on each transition."""
        events, _, final_index = self._run_pipeline(
            ["<think>t1</think>text<think>t2</think>end"]
        )
        parsed = self._parse_events(events)

        starts = [p for p in parsed if p["type"] == "content_block_start"]
        assert [s["index"] for s in starts] == [0, 1, 2, 3]
        assert final_index == 4


if __name__ == "__main__":
    unittest.main()
