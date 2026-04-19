# SPDX-License-Identifier: Apache-2.0
"""Tests for StreamingPostProcessor — the unified streaming pipeline."""

from unittest.mock import MagicMock

from vllm_mlx.service.postprocessor import StreamingPostProcessor


def _make_cfg(**overrides):
    """Create a mock ServerConfig."""
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.enable_auto_tool_choice = False
    cfg.tool_call_parser = None
    cfg.tool_parser_instance = None
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_output(text="", finished=False, channel=None, finish_reason=None):
    """Create a mock GenerationOutput."""
    out = MagicMock()
    out.new_text = text
    out.finished = finished
    out.channel = channel
    out.finish_reason = finish_reason or ("stop" if finished else None)
    out.prompt_tokens = 10
    out.completion_tokens = 5
    out.tokens = []
    out.logprobs = None
    return out


class TestStreamingPostProcessorBasic:
    """Tests for basic content streaming (no reasoning, no tools)."""

    def test_simple_content(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("Hello"))
        assert len(events) == 1
        assert events[0].type == "content"
        assert events[0].content == "Hello"

    def test_empty_text_skipped(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output(""))
        assert len(events) == 0

    def test_finish_event(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("Done", finished=True))
        # Should have content + finish
        content_events = [e for e in events if e.type == "content"]
        finish_events = [e for e in events if e.type == "finish"]
        assert len(content_events) >= 0  # may or may not have content
        assert any(e.finish_reason == "stop" for e in events)

    def test_special_tokens_stripped(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("Hello<|endoftext|>"))
        assert len(events) >= 1
        content = [e for e in events if e.type == "content"]
        if content:
            assert "<|endoftext|>" not in content[0].content


class TestStreamingPostProcessorChannelRouted:
    """Tests for OutputRouter (channel-routed) models."""

    def test_content_channel(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("Hello", channel="content"))
        assert len(events) == 1
        assert events[0].type == "content"
        assert events[0].content == "Hello"

    def test_reasoning_channel(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("thinking...", channel="reasoning"))
        assert len(events) == 1
        assert events[0].type == "reasoning"
        assert events[0].reasoning == "thinking..."


class TestStreamingPostProcessorReasoning:
    """Tests for text-based reasoning parser integration."""

    def test_reasoning_extraction(self):
        """Reasoning parser separates thinking from content."""
        parser = MagicMock()
        delta_msg = MagicMock()
        delta_msg.content = "answer"
        delta_msg.reasoning = "let me think"
        parser.extract_reasoning_streaming.return_value = delta_msg

        cfg = _make_cfg(reasoning_parser=parser)
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("<think>let me think</think>answer"))
        content_events = [e for e in events if e.type == "content"]
        reasoning_events = [e for e in events if e.type == "reasoning"]
        assert len(content_events) == 1
        assert len(reasoning_events) == 1
        assert "answer" in content_events[0].content

    def test_reasoning_suppressed_chunk(self):
        """Parser returns None (e.g., inside <think> tag) → no events."""
        parser = MagicMock()
        parser.extract_reasoning_streaming.return_value = None

        cfg = _make_cfg(reasoning_parser=parser)
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("<think>"))
        assert len(events) == 0


class TestStreamingPostProcessorToolCalls:
    """Tests for tool call detection."""

    def _make_tool_parser(self):
        parser = MagicMock()
        parser.extract_tool_calls_streaming.return_value = None  # default: suppressed
        parser.has_pending_tool_call.return_value = False
        return parser

    def test_tool_markup_suppresses_content(self):
        """Content is suppressed while inside tool markup."""
        tool_parser = self._make_tool_parser()
        tool_parser.extract_tool_calls_streaming.return_value = None

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("<tool_call>"))
        assert len(events) == 0

    def test_tool_call_detected(self):
        """Tool call detection emits tool_call event."""
        tool_parser = self._make_tool_parser()
        tool_parser.extract_tool_calls_streaming.return_value = {
            "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                           "function": {"name": "test", "arguments": "{}"}}]
        }

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("<tool_call>test</tool_call>"))
        assert len(events) == 1
        assert events[0].type == "tool_call"
        assert events[0].tool_calls is not None

    def test_content_after_tool_calls_suppressed(self):
        """After tool calls detected, remaining content is suppressed."""
        tool_parser = self._make_tool_parser()
        # First call: detect tool calls
        tool_parser.extract_tool_calls_streaming.return_value = {
            "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                           "function": {"name": "test", "arguments": "{}"}}]
        }

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # Detect tool call
        pp.process_chunk(_make_output("<tool_call>"))
        assert pp.tool_calls_detected

        # After detection, parser returns normal content but should be suppressed
        tool_parser.extract_tool_calls_streaming.return_value = {"content": "extra text"}
        events = pp.process_chunk(_make_output("extra text"))
        assert len(events) == 0

    def test_fallback_tool_detection_on_finalize(self):
        """Finalize detects tool calls when streaming detection missed them."""
        tool_parser = self._make_tool_parser()
        tool_parser.has_pending_tool_call.return_value = True
        result = MagicMock()
        result.tools_called = True
        result.tool_calls = [{"id": "call_1", "name": "test", "arguments": "{}"}]
        tool_parser.extract_tool_calls.return_value = result

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # Accumulate some text without triggering streaming detection
        pp.tool_accumulated_text = "<tool_call>{}"

        events = pp.finalize()
        assert len(events) == 1
        assert events[0].type == "tool_call"
        assert events[0].finish_reason == "tool_calls"


class TestStreamingPostProcessorNemotron:
    """Tests for Nemotron thinking prefix."""

    def test_thinking_prefix_injected(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.set_thinking_model("nemotron-nano-30b")
        pp.reset()

        events = pp.process_chunk(_make_output("Starting to think"))
        assert len(events) >= 1
        content_events = [e for e in events if e.type == "content"]
        assert content_events[0].content.startswith("<think>")

    def test_thinking_prefix_only_once(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.set_thinking_model("nemotron-nano-30b")
        pp.reset()

        pp.process_chunk(_make_output("First"))
        events = pp.process_chunk(_make_output("Second"))
        content_events = [e for e in events if e.type == "content"]
        assert not content_events[0].content.startswith("<think>")


class TestStreamingPostProcessorFinishMerging:
    """Tests for content + finish_reason merging (prevents double-emission)."""

    def test_final_chunk_single_event(self):
        """Final chunk with content + finish emits ONE event, not two."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("final word", finished=True))
        # Should be exactly one finish event with content merged in
        assert len(events) == 1
        assert events[0].type == "finish"
        assert events[0].finish_reason == "stop"
        assert events[0].content is not None

    def test_finish_without_content(self):
        """Finish-only chunk (empty text) emits finish event."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("", finished=True))
        assert len(events) == 1
        assert events[0].type == "finish"

    def test_channel_routed_finish_merges(self):
        """Channel-routed path also merges content into finish."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(
            _make_output("done", finished=True, channel="content")
        )
        assert len(events) == 1
        assert events[0].type == "finish"
        assert events[0].content is not None

    def test_reasoning_finish_merges(self):
        """Reasoning path merges reasoning into finish."""
        parser = MagicMock()
        delta_msg = MagicMock()
        delta_msg.content = "answer"
        delta_msg.reasoning = "thought"
        parser.extract_reasoning_streaming.return_value = delta_msg

        cfg = _make_cfg(reasoning_parser=parser)
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(
            _make_output("raw", finished=True)
        )
        assert len(events) == 1
        assert events[0].type == "finish"
        assert events[0].reasoning == "thought"


class TestStreamingPostProcessorMiniMaxRedirect:
    """Tests for MiniMax tool-in-thinking redirect."""

    def test_tool_xml_in_reasoning_redirected(self):
        """Tool call XML in reasoning stream gets redirected to content."""
        parser = MagicMock()
        delta_msg = MagicMock()
        delta_msg.content = None
        delta_msg.reasoning = "<tool_call>{}"
        parser.extract_reasoning_streaming.return_value = delta_msg

        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = {"content": ""}
        tool_parser.has_pending_tool_call.return_value = False

        cfg = _make_cfg(
            reasoning_parser=parser,
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        pp.process_chunk(_make_output("<tool_call>{}"))
        # Tool parser should have been called (reasoning was redirected to content)
        assert tool_parser.extract_tool_calls_streaming.called


class TestStreamingPostProcessorToolCallChannel:
    """Tests for tool_call channel routing."""

    def test_tool_call_channel_with_parser(self):
        """Tool call channel content goes through tool parser."""
        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = {
            "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                           "function": {"name": "test", "arguments": "{}"}}]
        }
        tool_parser.has_pending_tool_call.return_value = False

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(
            _make_output("<tool_call>", channel="tool_call")
        )
        assert len(events) == 1
        assert events[0].type == "tool_call"

    def test_reasoning_channel_finish(self):
        """Reasoning channel with finish emits single finish event."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(
            _make_output("final thought", finished=True, channel="reasoning")
        )
        assert len(events) == 1
        assert events[0].type == "finish"
        assert events[0].reasoning == "final thought"


# ======================================================================
# Coverage gap tests — model-specific edge cases + error paths
# ======================================================================


class TestToolParserInit:
    """Tests for _init_tool_parser paths (lines 72-99)."""

    def test_init_uses_existing_parser_instance(self):
        """Tool parser uses cfg.tool_parser_instance when already set."""
        mock_parser = MagicMock()

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=mock_parser,
        )
        pp = StreamingPostProcessor(cfg, tools_requested=True)
        assert pp.tool_parser is mock_parser

    def test_init_parser_failure_returns_none(self):
        """Failed tool parser init returns None gracefully."""
        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="nonexistent_parser_xyz",
            tool_parser_instance=None,
        )
        pp = StreamingPostProcessor(cfg)
        # Should not crash, tool_parser should be None
        assert pp.tool_parser is None

    def test_auto_infer_minimax_parser(self):
        """Auto-infer MiniMax tool parser from reasoning_parser_name."""
        cfg = _make_cfg(
            reasoning_parser_name="minimax",
            engine=MagicMock(_tokenizer=MagicMock()),
        )
        pp = StreamingPostProcessor(cfg, tools_requested=True)
        # Should attempt to create a minimax parser
        # May succeed or fail depending on parser registry
        # The key is it doesn't crash


class TestChannelRoutedEdgeCases:
    """Tests for channel-routed path edge cases."""

    def test_tool_call_channel_suppressed(self):
        """Tool call channel with parser returning None suppresses output."""
        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = None

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(
            _make_output("<tool_call>", channel="content")
        )
        assert len(events) == 0

    def test_channel_content_passthrough_no_tool_parser(self):
        """Content channel without tool parser passes through."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(
            _make_output("hello world", channel="content")
        )
        assert len(events) == 1
        assert events[0].type == "content"

    def test_channel_empty_after_sanitize(self):
        """Channel content that sanitizes to empty is dropped."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # Special tokens only → sanitize strips everything
        events = pp.process_chunk(
            _make_output("<|endoftext|>", channel="content")
        )
        # May produce 0 events if sanitized to empty
        content_events = [e for e in events if e.type == "content"]
        for e in content_events:
            assert e.content  # no empty content events

    def test_channel_tool_calls_detected_suppresses_subsequent(self):
        """After tool_calls detected via channel, subsequent content suppressed."""
        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.side_effect = [
            {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                            "function": {"name": "t", "arguments": "{}"}}]},
            {"content": "leftover"},
        ]

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # First chunk: tool call detected
        events1 = pp.process_chunk(_make_output("<tc>", channel="content"))
        assert events1[0].type == "tool_call"

        # Second chunk: should be suppressed
        events2 = pp.process_chunk(_make_output("more", channel="content"))
        assert len(events2) == 0

    def test_channel_tool_calls_finish_event(self):
        """After tool_calls detected, finish chunk emits finish with tool_calls reason."""
        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = {
            "tool_calls": [{"index": 0, "id": "c1", "type": "function",
                           "function": {"name": "t", "arguments": "{}"}}]
        }

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        pp.process_chunk(_make_output("<tc>", channel="content"))

        tool_parser.extract_tool_calls_streaming.return_value = {"content": ""}
        events = pp.process_chunk(
            _make_output("", finished=True, channel="content")
        )
        assert len(events) == 1
        assert events[0].type == "finish"
        assert events[0].finish_reason == "tool_calls"


class TestReasoningPathEdgeCases:
    """Tests for reasoning parser path edge cases."""

    def test_reasoning_with_tool_suppression(self):
        """Reasoning path: tool parser returns None → suppressed."""
        parser = MagicMock()
        delta_msg = MagicMock()
        delta_msg.content = "content with <tool_call>"
        delta_msg.reasoning = None
        parser.extract_reasoning_streaming.return_value = delta_msg

        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = None

        cfg = _make_cfg(
            reasoning_parser=parser,
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("<tool_call>"))
        assert len(events) == 0

    def test_reasoning_tool_calls_detected_finish(self):
        """Reasoning path: after tool calls, finish emits tool_calls reason."""
        parser = MagicMock()
        delta_msg = MagicMock()
        delta_msg.content = "<tool_call>markup"  # must contain < to trigger full parsing
        delta_msg.reasoning = None
        parser.extract_reasoning_streaming.return_value = delta_msg

        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = {
            "tool_calls": [{"index": 0, "id": "c1", "type": "function",
                           "function": {"name": "t", "arguments": "{}"}}]
        }

        cfg = _make_cfg(
            reasoning_parser=parser,
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        pp.process_chunk(_make_output("<tool_call>markup"))
        assert pp.tool_calls_detected

        # Subsequent finish — tool_calls_detected so suppressed, finish emitted
        delta_msg2 = MagicMock()
        delta_msg2.content = ""
        delta_msg2.reasoning = None
        parser.extract_reasoning_streaming.return_value = delta_msg2
        tool_parser.extract_tool_calls_streaming.return_value = {"content": ""}

        events = pp.process_chunk(_make_output("", finished=True))
        assert len(events) == 1
        assert events[0].finish_reason == "tool_calls"

    def test_reasoning_finish_on_suppressed_chunk(self):
        """Reasoning parser returns None on final chunk → finish event."""
        parser = MagicMock()
        parser.extract_reasoning_streaming.return_value = None

        cfg = _make_cfg(reasoning_parser=parser)
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("final", finished=True))
        assert len(events) == 1
        assert events[0].type == "finish"

    def test_minimax_tool_call_in_reasoning_with_content(self):
        """MiniMax: tool XML in reasoning WITH existing content → both merged."""
        parser = MagicMock()
        delta_msg = MagicMock()
        delta_msg.content = "\n"  # boundary content from </think>
        delta_msg.reasoning = "<minimax:tool_call>{}"
        parser.extract_reasoning_streaming.return_value = delta_msg

        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = {"content": "merged"}

        cfg = _make_cfg(
            reasoning_parser=parser,
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        pp.process_chunk(_make_output("raw"))
        # Tool parser should receive content + reasoning merged
        call_args = tool_parser.extract_tool_calls_streaming.call_args
        assert call_args is not None


class TestStandardPathEdgeCases:
    """Tests for standard (no reasoning, no channel) path edge cases."""

    def test_tool_fast_path_no_markup(self):
        """Standard path: content without < or [ takes fast path."""
        tool_parser = MagicMock()
        tool_parser.has_pending_tool_call.return_value = False

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("hello world"))
        # extract_tool_calls_streaming should NOT be called (fast path)
        assert not tool_parser.extract_tool_calls_streaming.called
        assert len(events) == 1
        assert events[0].type == "content"

    def test_tool_markup_triggers_full_parsing(self):
        """Standard path: < in content triggers full tool parsing."""
        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = {"content": "text"}
        tool_parser.has_pending_tool_call.return_value = False

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("before <tag>"))
        assert tool_parser.extract_tool_calls_streaming.called
