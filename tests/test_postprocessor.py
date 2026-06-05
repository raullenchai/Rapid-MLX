# SPDX-License-Identifier: Apache-2.0
"""Tests for StreamingPostProcessor — the unified streaming pipeline."""

import json
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


def _make_output(
    text="", finished=False, channel=None, finish_reason=None, tool_calls=None
):
    """Create a mock GenerationOutput.

    ``tool_calls`` defaults to None so the postprocessor's structured
    fast-path (PR #515) stays inert in tests that don't opt in. Tests
    exercising HarmonyStreamingRouter passthrough must set this explicitly.
    """
    out = MagicMock()
    out.new_text = text
    out.finished = finished
    out.channel = channel
    out.finish_reason = finish_reason or ("stop" if finished else None)
    out.prompt_tokens = 10
    out.completion_tokens = 5
    out.tokens = []
    out.logprobs = None
    out.tool_calls = tool_calls
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
        assert len(finish_events) == 1
        assert finish_events[0].finish_reason == "stop"

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

    def test_channel_bypasses_legacy_reasoning_parser(self):
        parser = MagicMock()
        cfg = _make_cfg(reasoning_parser=parser)
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("Hello", channel="content"))

        assert len(events) == 1
        assert events[0].type == "content"
        assert events[0].content == "Hello"
        parser.extract_reasoning_streaming.assert_not_called()

    def test_channel_routed_accumulators_populated(self):
        """OutputRouter path must update accumulated_text + accumulated_reasoning.

        Regression for v0.6.66 onboarding sweep finding: the streaming
        usage chunk dropped ``completion_tokens_details.reasoning_tokens``
        entirely for Gemma 4 / harmony because ``_process_channel_routed``
        emitted events to the client but never updated the per-processor
        accumulators that ``_build_usage`` reads to compute the
        reasoning/content split. Confirmed by parallel onboarding agents
        on gemma-4-26b and gpt-oss-20b.
        """
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # Interleave reasoning + content chunks across multiple calls
        # (the real stream emits one channel-tagged token at a time).
        pp.process_chunk(_make_output("Let ", channel="reasoning"))
        pp.process_chunk(_make_output("me ", channel="reasoning"))
        pp.process_chunk(_make_output("think.", channel="reasoning"))
        pp.process_chunk(_make_output("The ", channel="content"))
        pp.process_chunk(_make_output("answer ", channel="content"))
        pp.process_chunk(_make_output("is 42.", channel="content"))

        assert pp.accumulated_reasoning == "Let me think."
        assert pp.accumulated_text == "The answer is 42."

    def test_channel_routed_accumulators_skip_empty_after_sanitize(self):
        """Sanitized-empty chunks must NOT accumulate (avoid phantom tokens).

        ``sanitize_output`` can return None when a chunk is entirely
        special tokens; the accumulator must respect that so the usage
        char-ratio doesn't include suppressed content.
        """
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        pp.process_chunk(_make_output("real ", channel="content"))
        pp.process_chunk(_make_output("<|endoftext|>", channel="content"))
        pp.process_chunk(_make_output("text", channel="content"))

        # Special-token-only chunk is dropped from accumulation.
        assert "<|endoftext|>" not in pp.accumulated_text
        assert pp.accumulated_text == "real text"


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

    def test_enable_thinking_false_bypasses_reasoning_parser(self):
        """When enable_thinking=False, the chat template suppresses the <think>
        gen prompt and the model answers directly. Qwen3's reasoning parser
        treats untagged tokens as reasoning (implicit-think heuristic), which
        misroutes the answer to reasoning_content and leaves content empty.
        Setting enable_thinking=False on the postprocessor must bypass the
        parser entirely so the answer flows to delta.content. Repro for the
        eval-suite irrelevance/missing-params empty-content failures.
        """
        parser = MagicMock()
        # If the parser were called, it would return reasoning (the bug shape).
        delta_msg = MagicMock()
        delta_msg.content = ""
        delta_msg.reasoning = "The capital of France is Paris."
        parser.extract_reasoning_streaming.return_value = delta_msg

        cfg = _make_cfg(reasoning_parser=parser)
        pp = StreamingPostProcessor(cfg, enable_thinking=False)
        pp.reset()

        events = pp.process_chunk(_make_output("The capital of France is Paris."))
        # Parser must NOT have been consulted.
        parser.extract_reasoning_streaming.assert_not_called()
        content_events = [e for e in events if e.type == "content"]
        reasoning_events = [e for e in events if e.type == "reasoning"]
        assert len(content_events) == 1
        assert content_events[0].content == "The capital of France is Paris."
        assert not reasoning_events

    def test_enable_thinking_none_uses_reasoning_parser(self):
        """Default (None) preserves the existing reasoning-parser path."""
        parser = MagicMock()
        delta_msg = MagicMock()
        delta_msg.content = "answer"
        delta_msg.reasoning = "thinking"
        parser.extract_reasoning_streaming.return_value = delta_msg

        cfg = _make_cfg(reasoning_parser=parser)
        pp = StreamingPostProcessor(cfg)  # enable_thinking defaults to None
        pp.reset()

        events = pp.process_chunk(_make_output("<think>thinking</think>answer"))
        parser.extract_reasoning_streaming.assert_called_once()
        assert any(e.type == "reasoning" for e in events)
        assert any(e.type == "content" for e in events)

    def test_enable_thinking_true_uses_reasoning_parser(self):
        """Explicit True also keeps the reasoning-parser path on."""
        parser = MagicMock()
        delta_msg = MagicMock()
        delta_msg.content = "answer"
        delta_msg.reasoning = "thinking"
        parser.extract_reasoning_streaming.return_value = delta_msg

        cfg = _make_cfg(reasoning_parser=parser)
        pp = StreamingPostProcessor(cfg, enable_thinking=True)
        pp.reset()

        events = pp.process_chunk(_make_output("<think>thinking</think>answer"))
        parser.extract_reasoning_streaming.assert_called_once()
        assert any(e.type == "reasoning" for e in events)
        assert any(e.type == "content" for e in events)


class TestStreamingPostProcessorToolCalls:
    """Tests for tool call detection."""

    def _make_tool_parser(self):
        parser = MagicMock()
        parser.extract_tool_calls_streaming.return_value = None  # default: suppressed
        parser.has_pending_tool_call.return_value = False
        # Default to empty string so finalize()'s held-content release path
        # (postprocessor.py — codex round-3 CRITICAL fix) is a no-op unless
        # a test explicitly opts into a non-empty held suffix.
        parser.flush_held_content.return_value = ""
        return parser

    def test_tool_markup_suppresses_content(self):
        """Content is suppressed while inside tool markup."""
        tool_parser = self._make_tool_parser()
        tool_parser.extract_tool_calls_streaming.return_value = None

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
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
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                }
            ]
        }

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
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
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                }
            ]
        }

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # Detect tool call
        pp.process_chunk(_make_output("<tool_call>"))
        assert pp.tool_calls_detected

        # After detection, parser returns normal content but should be suppressed
        tool_parser.extract_tool_calls_streaming.return_value = {
            "content": "extra text"
        }
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

    def test_finalize_cross_format_fallback_recovers_xml_tool_call(self):
        """Configured parser is wire-format-specific; finalize must fall back to
        the multi-format ``parse_tool_calls`` when the configured parser fails.

        Reproduces #425: ``--tool-call-parser qwen3_xml`` resolves to
        ``QwenToolParser`` (JSON inside ``<tool_call>``) but vanilla
        Qwen3.6-35B-A3B emits the XML-bodied variant
        (``<function=...><parameter=...>``). The configured parser returns
        ``tools_called=False`` on the wire-mismatched text; without a
        cross-format fallback the stream emits zero ``tool_calls`` deltas
        while the same text non-streaming returns a structured call via the
        ``parse_tool_calls`` fallback in ``service/helpers.py``.
        """
        tool_parser = self._make_tool_parser()
        # Configured parser fails to extract (wire-format mismatch)
        tool_parser.extract_tool_calls.return_value = MagicMock(
            tools_called=False, tool_calls=[]
        )

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # Qwen3.6 vanilla wire format: XML body inside <tool_call>
        pp.tool_accumulated_text = (
            "<tool_call>\n"
            "<function=read_file>\n"
            "<parameter=path>\n"
            "/etc/hostname\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

        events = pp.finalize()
        assert len(events) == 1
        assert events[0].type == "tool_call"
        assert events[0].finish_reason == "tool_calls"
        assert events[0].tool_calls is not None
        assert len(events[0].tool_calls) == 1
        tc = events[0].tool_calls[0]
        assert tc["function"]["name"] == "read_file"
        # arguments is a JSON string; parse to assert the value
        assert json.loads(tc["function"]["arguments"]) == {"path": "/etc/hostname"}
        assert pp.tool_calls_detected is True

    def test_finalize_cross_format_fallback_noop_on_plain_text(self):
        """Plain text in the accumulator must NOT trigger a spurious tool_call
        from the cross-format fallback. Guards against the structural pre-check
        passing on bare ``<`` characters in normal prose."""
        tool_parser = self._make_tool_parser()
        tool_parser.extract_tool_calls.return_value = MagicMock(
            tools_called=False, tool_calls=[]
        )

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # Plain prose that just happens to contain a literal "<" character
        pp.tool_accumulated_text = "the value of x < 10 is fine"

        events = pp.finalize()
        assert events == []
        assert pp.tool_calls_detected is False

    def test_finalize_cross_format_fallback_both_parsers_fail(self):
        """Structural markup is present (passes the ``<`` pre-check) but the text
        is not a parseable tool call in any known format. Both the configured
        parser and ``parse_tool_calls`` return no calls. finalize() must return
        ``[]`` cleanly and leave ``tool_calls_detected`` False — not raise, not
        emit a spurious event."""
        tool_parser = self._make_tool_parser()
        tool_parser.extract_tool_calls.return_value = MagicMock(
            tools_called=False, tool_calls=[]
        )

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # XML-shaped but not actually a tool call — no <function=...> or
        # [Calling tool: ...] or {"name":...} structure parse_tool_calls
        # recognises.
        pp.tool_accumulated_text = "<tool_call>garbled junk no actual call</tool_call>"

        events = pp.finalize()
        assert events == []
        assert pp.tool_calls_detected is False

    def test_finalize_cross_format_fallback_swallows_parser_exception(self):
        """``parse_tool_calls`` must not be allowed to abort the stream. If the
        multi-format scanner raises (regex pathology, adversarial input), the
        fallback logs a warning and returns ``[]`` rather than propagating.
        Mirrors the defensive ``try/except`` in ``service/helpers.py:605-607``."""
        tool_parser = self._make_tool_parser()
        tool_parser.extract_tool_calls.return_value = MagicMock(
            tools_called=False, tool_calls=[]
        )

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()
        pp.tool_accumulated_text = "<tool_call>anything</tool_call>"

        # Force the fallback parser to raise
        from unittest.mock import patch

        with patch(
            "vllm_mlx.service.postprocessor.parse_tool_calls",
            side_effect=RuntimeError("boom"),
        ):
            events = pp.finalize()

        assert events == []
        assert pp.tool_calls_detected is False

    def test_finalize_releases_held_content_when_no_tool_calls(self):
        """Codex round-3 CRITICAL: streaming parsers that prefix-hold
        partial tool-call sentinels can end the stream still holding
        bytes. If no tool call ever fires, those bytes are ordinary
        content and finalize() must release them so the user doesn't
        see a truncated reply (e.g. ``abc<`` shown as ``abc``)."""
        tool_parser = self._make_tool_parser()
        tool_parser.extract_tool_calls.return_value = MagicMock(
            tools_called=False, tool_calls=[]
        )
        tool_parser.flush_held_content.return_value = "<"

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()
        pp.tool_accumulated_text = "abc<"

        events = pp.finalize()
        content_events = [e for e in events if e.type == "content"]
        assert len(content_events) == 1
        assert content_events[0].content == "<"

    def test_finalize_does_not_release_held_content_when_tool_call_fires(self):
        """If a tool call DID fire mid-stream, the held bytes are part
        of the tool-call body — not ordinary content. The release path
        must stay silent so the held tail doesn't leak alongside the
        structured tool_call event."""
        tool_parser = self._make_tool_parser()
        tool_parser.extract_tool_calls.return_value = MagicMock(
            tools_called=False, tool_calls=[]
        )
        tool_parser.flush_held_content.return_value = "<"

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()
        pp.tool_accumulated_text = "abc<"
        pp.tool_calls_detected = True  # simulate a mid-stream tool call

        events = pp.finalize()
        content_events = [e for e in events if e.type == "content"]
        assert content_events == []

    def test_finalize_rejects_non_string_held_content(self):
        """Defensive: a buggy ``flush_held_content`` override returning
        non-string (or None) must not propagate as a malformed
        StreamEvent. The strict-string gate in finalize() skips it
        instead."""
        tool_parser = self._make_tool_parser()
        tool_parser.extract_tool_calls.return_value = MagicMock(
            tools_called=False, tool_calls=[]
        )
        tool_parser.flush_held_content.return_value = None  # buggy override

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()
        pp.tool_accumulated_text = "abc<"

        events = pp.finalize()
        assert [e for e in events if e.type == "content"] == []


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

        events = pp.process_chunk(_make_output("raw", finished=True))
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
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                }
            ]
        }
        tool_parser.has_pending_tool_call.return_value = False

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("<tool_call>", channel="tool_call"))
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


class TestStructuredToolCallStreaming:
    """Tests for the engine-surfaced structured tool-call fast-path
    introduced by PR #515 (HarmonyStreamingRouter bypass).

    Two invariants the legacy text-parser path enforced must be
    preserved on the structured path — codex round-15 BLOCKINGs:

    1. ``tool_calls[*].index`` is monotonically increasing across
       distinct router chunks (clients merge deltas on ``index``;
       colliding indices silently overwrite).
    2. ``parallel_tool_calls=false`` caps the response at one call
       (matching the post-parse trim in ``routes/chat.py``).
    """

    def test_index_monotonic_across_chunks(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        first = pp.process_chunk(
            _make_output(
                "search(...)",
                channel="tool_call",
                tool_calls=[{"name": "search", "arguments": '{"q":"a"}'}],
            )
        )
        second = pp.process_chunk(
            _make_output(
                "lookup(...)",
                channel="tool_call",
                tool_calls=[{"name": "lookup", "arguments": '{"q":"b"}'}],
            )
        )

        assert len(first) == 1 and first[0].type == "tool_call"
        assert len(second) == 1 and second[0].type == "tool_call"
        assert first[0].tool_calls[0]["index"] == 0
        assert second[0].tool_calls[0]["index"] == 1

    def test_index_monotonic_within_single_chunk(self):
        """Multiple calls in one router event still receive distinct indices."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(
            _make_output(
                "two(...)",
                channel="tool_call",
                tool_calls=[
                    {"name": "a", "arguments": "{}"},
                    {"name": "b", "arguments": "{}"},
                ],
            )
        )
        assert len(events) == 1
        indices = [tc["index"] for tc in events[0].tool_calls]
        assert indices == [0, 1]

    def test_parallel_tool_calls_false_caps_across_chunks(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, request={"parallel_tool_calls": False})
        pp.reset()

        first = pp.process_chunk(
            _make_output(
                "search(...)",
                channel="tool_call",
                tool_calls=[{"name": "search", "arguments": '{"q":"a"}'}],
            )
        )
        second = pp.process_chunk(
            _make_output(
                "lookup(...)",
                channel="tool_call",
                tool_calls=[{"name": "lookup", "arguments": '{"q":"b"}'}],
            )
        )

        assert len(first) == 1
        assert first[0].type == "tool_call"
        assert first[0].tool_calls[0]["function"]["name"] == "search"
        # Second chunk is suppressed entirely (no finish marker since
        # the engine output isn't ``finished``).
        assert second == []
        assert pp.tool_calls_detected is True

    def test_parallel_tool_calls_false_caps_within_single_chunk(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, request={"parallel_tool_calls": False})
        pp.reset()

        events = pp.process_chunk(
            _make_output(
                "two(...)",
                channel="tool_call",
                tool_calls=[
                    {"name": "a", "arguments": "{}"},
                    {"name": "b", "arguments": "{}"},
                ],
            )
        )
        assert len(events) == 1
        assert len(events[0].tool_calls) == 1
        assert events[0].tool_calls[0]["function"]["name"] == "a"

    def test_parallel_tool_calls_true_does_not_cap(self):
        """Explicit ``True`` must behave like unset (no cap)."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, request={"parallel_tool_calls": True})
        pp.reset()

        first = pp.process_chunk(
            _make_output(
                "search(...)",
                channel="tool_call",
                tool_calls=[{"name": "search", "arguments": "{}"}],
            )
        )
        second = pp.process_chunk(
            _make_output(
                "lookup(...)",
                channel="tool_call",
                tool_calls=[{"name": "lookup", "arguments": "{}"}],
            )
        )
        assert len(first) == 1 and len(second) == 1
        assert second[0].tool_calls[0]["index"] == 1

    def test_cap_suppression_preserves_finish_semantics(self):
        """When the suppressed chunk is also the finished chunk we must
        still emit a finish event so the route's buffered_finish gate
        fires.
        """
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, request={"parallel_tool_calls": False})
        pp.reset()

        pp.process_chunk(
            _make_output(
                "first(...)",
                channel="tool_call",
                tool_calls=[{"name": "first", "arguments": "{}"}],
            )
        )
        events = pp.process_chunk(
            _make_output(
                "second(...)",
                channel="tool_call",
                finished=True,
                finish_reason="stop",
                tool_calls=[{"name": "second", "arguments": "{}"}],
            )
        )
        assert len(events) == 1
        assert events[0].type == "finish"
        assert events[0].finish_reason == "tool_calls"

    def test_structured_event_assigns_id_when_missing(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(
            _make_output(
                "search(...)",
                channel="tool_call",
                tool_calls=[{"name": "search", "arguments": "{}"}],
            )
        )
        assert events[0].tool_calls[0]["id"].startswith("call_")
        assert events[0].tool_calls[0]["type"] == "function"

    def test_structured_event_preserves_caller_id(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(
            _make_output(
                "search(...)",
                channel="tool_call",
                tool_calls=[{"id": "call_abc", "name": "search", "arguments": "{}"}],
            )
        )
        assert events[0].tool_calls[0]["id"] == "call_abc"

    def test_reset_clears_structured_counter(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        pp.process_chunk(
            _make_output(
                "first(...)",
                channel="tool_call",
                tool_calls=[{"name": "a", "arguments": "{}"}],
            )
        )
        pp.reset()
        events = pp.process_chunk(
            _make_output(
                "second(...)",
                channel="tool_call",
                tool_calls=[{"name": "b", "arguments": "{}"}],
            )
        )
        # New stream → index restarts at 0.
        assert events[0].tool_calls[0]["index"] == 0


# ======================================================================
# Coverage gap tests — model-specific edge cases + error paths
# ======================================================================


class TestToolParserInit:
    """Tests for _init_tool_parser paths (lines 72-99)."""

    def test_init_creates_fresh_parser_not_singleton(self):
        """Tool parser is a fresh per-request instance, NOT cfg singleton."""
        singleton = MagicMock()

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
            tool_parser_instance=singleton,
        )
        pp = StreamingPostProcessor(cfg, tools_requested=True)
        # Must NOT be the singleton — should be a fresh HermesToolParser
        assert pp.tool_parser is not singleton
        assert pp.tool_parser is not None

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
        # MiniMax parser should be auto-inferred from reasoning_parser_name
        assert pp.tool_parser is not None


class TestChannelRoutedEdgeCases:
    """Tests for channel-routed path edge cases."""

    def test_tool_call_channel_suppressed(self):
        """Tool call channel with parser returning None suppresses output."""
        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = None

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("<tool_call>", channel="content"))
        assert len(events) == 0

    def test_channel_content_passthrough_no_tool_parser(self):
        """Content channel without tool parser passes through."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("hello world", channel="content"))
        assert len(events) == 1
        assert events[0].type == "content"

    def test_channel_empty_after_sanitize(self):
        """Channel content that sanitizes to empty is dropped."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # Special tokens only → sanitize strips everything
        events = pp.process_chunk(_make_output("<|endoftext|>", channel="content"))
        # May produce 0 events if sanitized to empty
        content_events = [e for e in events if e.type == "content"]
        for e in content_events:
            assert e.content  # no empty content events

    def test_channel_tool_calls_detected_suppresses_subsequent(self):
        """After tool_calls detected via channel, subsequent content suppressed."""
        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.side_effect = [
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ]
            },
            {"content": "leftover"},
        ]

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
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
            "tool_calls": [
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "t", "arguments": "{}"},
                }
            ]
        }

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        pp.process_chunk(_make_output("<tc>", channel="content"))

        tool_parser.extract_tool_calls_streaming.return_value = {"content": ""}
        events = pp.process_chunk(_make_output("", finished=True, channel="content"))
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
        delta_msg.content = (
            "<tool_call>markup"  # must contain < to trigger full parsing
        )
        delta_msg.reasoning = None
        parser.extract_reasoning_streaming.return_value = delta_msg

        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "t", "arguments": "{}"},
                }
            ]
        }

        cfg = _make_cfg(
            reasoning_parser=parser,
            enable_auto_tool_choice=True,
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
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        events = pp.process_chunk(_make_output("before <tag>"))
        assert tool_parser.extract_tool_calls_streaming.called


# =====================================================================
# Per-request parser isolation (P1 singleton fix)
# =====================================================================


class TestPerRequestParserIsolation:
    """Verify that each PostProcessor gets its own parser instances,
    not references to the global singleton from ServerConfig."""

    def test_reasoning_parser_is_per_request(self):
        """Two PostProcessors must have different reasoning parser instances."""
        cfg = _make_cfg(reasoning_parser_name="qwen3")
        # cfg.reasoning_parser is the singleton — should NOT be used
        cfg.reasoning_parser = MagicMock()

        pp1 = StreamingPostProcessor(cfg)
        pp2 = StreamingPostProcessor(cfg)

        # Each must have its OWN parser, not the singleton
        assert pp1.reasoning_parser is not cfg.reasoning_parser
        assert pp2.reasoning_parser is not cfg.reasoning_parser
        assert pp1.reasoning_parser is not pp2.reasoning_parser

    def test_reasoning_parser_none_when_no_name(self):
        """No parser created when reasoning_parser_name is not set."""
        cfg = _make_cfg(reasoning_parser_name=None)
        pp = StreamingPostProcessor(cfg)
        assert pp.reasoning_parser is None

    def test_tool_parser_is_per_request(self):
        """Two PostProcessors must have different tool parser instances."""
        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
        )
        # cfg.tool_parser_instance is the singleton — should NOT be used
        cfg.tool_parser_instance = MagicMock()

        pp1 = StreamingPostProcessor(cfg, tools_requested=True)
        pp2 = StreamingPostProcessor(cfg, tools_requested=True)

        assert pp1.tool_parser is not cfg.tool_parser_instance
        assert pp2.tool_parser is not cfg.tool_parser_instance
        assert pp1.tool_parser is not pp2.tool_parser

    def test_tool_parser_auto_infer_is_per_request(self):
        """Auto-inferred tool parsers are also per-request."""
        cfg = _make_cfg(reasoning_parser_name="minimax")
        pp1 = StreamingPostProcessor(cfg, tools_requested=True)
        pp2 = StreamingPostProcessor(cfg, tools_requested=True)

        if pp1.tool_parser is not None:
            assert pp1.tool_parser is not pp2.tool_parser

    def test_concurrent_reset_does_not_corrupt(self):
        """Simulating concurrent usage: reset on one doesn't affect other."""
        cfg = _make_cfg(reasoning_parser_name="qwen3")

        pp1 = StreamingPostProcessor(cfg)
        pp2 = StreamingPostProcessor(cfg)

        # Simulate pp1 accumulating state
        pp1.reset()
        pp1.process_chunk(_make_output("Hello"))
        assert pp1.accumulated_text == "Hello"

        # pp2 resets independently
        pp2.reset()
        pp2.process_chunk(_make_output("World"))

        # pp1's state should be untouched
        assert pp1.accumulated_text == "Hello"
        assert pp2.accumulated_text == "World"

    def test_reasoning_parser_state_isolated(self):
        """Reasoning parser internal state is isolated between instances."""
        cfg = _make_cfg(reasoning_parser_name="qwen3")

        pp1 = StreamingPostProcessor(cfg)
        pp2 = StreamingPostProcessor(cfg)

        pp1.reset()
        pp2.reset()

        # Process a thinking chunk on pp1 — mutates pp1's parser state
        pp1.process_chunk(_make_output("<think>reasoning"))

        # pp2's parser should NOT have any accumulated state from pp1
        assert pp2.reasoning_parser is not pp1.reasoning_parser
        # Verify pp2's parser has clean internal state
        if hasattr(pp2.reasoning_parser, "_buffer"):
            assert pp2.reasoning_parser._buffer == ""

    def test_graceful_fallback_on_bad_parser_name(self):
        """Invalid parser name results in None, not crash."""
        cfg = _make_cfg(reasoning_parser_name="nonexistent_parser_xyz")
        pp = StreamingPostProcessor(cfg)
        assert pp.reasoning_parser is None

    def test_graceful_fallback_on_bad_tool_parser(self):
        """Invalid tool parser name results in None, not crash."""
        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_call_parser="nonexistent_parser_xyz",
        )
        pp = StreamingPostProcessor(cfg, tools_requested=True)
        assert pp.tool_parser is None


# =====================================================================
# Coverage gap tests — edge cases in processing paths
# =====================================================================


class TestCoverageGaps:
    """Tests targeting specific uncovered lines for 100% coverage."""

    def test_create_reasoning_parser_name_not_set(self):
        """Line 89: _create_reasoning_parser returns None when no name."""
        result = StreamingPostProcessor._create_reasoning_parser(
            _make_cfg(reasoning_parser_name=None)
        )
        assert result is None

    def test_auto_infer_tool_parser_failure(self):
        """Lines 124-125: auto-infer tool parser exception path."""
        from unittest.mock import patch

        cfg = _make_cfg(reasoning_parser_name="minimax")
        # Make ToolParserManager.get_tool_parser raise for "minimax"
        with patch(
            "vllm_mlx.tool_parsers.ToolParserManager.get_tool_parser",
            side_effect=KeyError("minimax not found"),
        ):
            pp = StreamingPostProcessor(cfg, tools_requested=True)
            assert pp.tool_parser is None  # graceful fallback

    def test_channel_routed_tool_detected_then_finish(self):
        """Lines 199, 276-279: tool_calls_detected + finish in channel mode."""
        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ]
        }
        cfg = _make_cfg(tool_parser_instance=tool_parser)
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # First chunk: tool detected via channel routing (needs "<" to trigger parsing)
        out1 = _make_output("<tool_call>test", channel="content")
        events1 = pp.process_chunk(out1)
        assert any(e.type == "tool_call" for e in events1)
        assert pp.tool_calls_detected

        # After tool detected, content with text should be suppressed (line 197-201)
        # Return content (not None/suppressed) so we reach the tool_calls_detected check
        tool_parser.extract_tool_calls_streaming.return_value = {"content": "trailing"}
        out2 = _make_output("<more>text", channel="content")
        events2 = pp.process_chunk(out2)
        assert len(events2) == 0  # suppressed by tool_calls_detected

        # Finish chunk with text after tool detected (line 199)
        out3 = _make_output("<final>", finished=True, channel="content")
        events3 = pp.process_chunk(out3)
        assert any(
            e.type == "finish" and e.finish_reason == "tool_calls" for e in events3
        )

    def test_channel_routed_sanitize_empty(self):
        """Line 216: content becomes None after sanitize_output."""
        from unittest.mock import patch

        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # sanitize_output returns empty → content becomes None
        with patch("vllm_mlx.service.postprocessor.sanitize_output", return_value=""):
            out = _make_output("some text", channel="content")
            events = pp.process_chunk(out)
            content_events = [e for e in events if e.type == "content"]
            assert len(content_events) == 0

    def test_reasoning_path_tool_detected_then_finish(self):
        """Lines 276-279: tool_calls_detected then finish in reasoning path."""
        parser = MagicMock()
        delta_msg = MagicMock()
        delta_msg.content = "<tool_call>test"
        delta_msg.reasoning = None
        parser.extract_reasoning_streaming.return_value = delta_msg

        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ]
        }

        cfg = _make_cfg(reasoning_parser=parser, tool_parser_instance=tool_parser)
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # Detect tool calls (content has "<" to trigger full parsing)
        events1 = pp.process_chunk(_make_output("<tool_call>test"))
        assert any(e.type == "tool_call" for e in events1)

        # After tool detected, content should be suppressed (lines 276-279)
        tool_parser.extract_tool_calls_streaming.return_value = {"content": "trailing"}
        events2 = pp.process_chunk(_make_output("<more>text"))
        assert len(events2) == 0

        # Finish with text after tool detection (line 276-277)
        out = _make_output("<final>", finished=True)
        events3 = pp.process_chunk(out)
        assert any(e.finish_reason == "tool_calls" for e in events3)

    def test_reasoning_path_sanitize_to_none(self):
        """Line 294: content sanitizes to empty in reasoning path."""
        from unittest.mock import patch

        parser = MagicMock()
        delta_msg = MagicMock()
        delta_msg.content = "text that sanitizes away"
        delta_msg.reasoning = None
        parser.extract_reasoning_streaming.return_value = delta_msg

        cfg = _make_cfg(reasoning_parser=parser)
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        with patch("vllm_mlx.service.postprocessor.sanitize_output", return_value=""):
            events = pp.process_chunk(_make_output("text"))
            content_events = [e for e in events if e.type == "content"]
            assert len(content_events) == 0

    def test_standard_path_sanitize_to_none(self):
        """Line 350: content sanitizes to empty in standard path."""
        from unittest.mock import patch

        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        with patch("vllm_mlx.service.postprocessor.sanitize_output", return_value=""):
            events = pp.process_chunk(_make_output("text"))
            content_events = [e for e in events if e.type == "content"]
            assert len(content_events) == 0

    def test_standard_path_tool_detected_then_finish(self):
        """Lines 334-335: tool_calls_detected + finish in standard path."""
        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ]
        }

        cfg = _make_cfg(tool_parser_instance=tool_parser)
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # Detect tool call (needs "<" to trigger full parsing)
        events1 = pp.process_chunk(_make_output("<tool_call>test"))
        assert pp.tool_calls_detected
        assert any(e.type == "tool_call" for e in events1)

        # After detection, content suppressed (line 332-336)
        tool_parser.extract_tool_calls_streaming.return_value = {"content": "trailing"}
        events2 = pp.process_chunk(_make_output("<more>text"))
        assert len(events2) == 0

        # Finish with text after tool detection (line 334)
        out = _make_output("<final>", finished=True)
        events3 = pp.process_chunk(out)
        assert any(
            e.type == "finish" and e.finish_reason == "tool_calls" for e in events3
        )

    def test_standard_path_empty_content_filtered(self):
        """Lines 340, 345: empty string content filtered out."""
        from unittest.mock import patch

        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # strip_special_tokens returns empty string → content=None, no finish → []
        with patch(
            "vllm_mlx.service.postprocessor.strip_special_tokens", return_value=""
        ):
            events = pp.process_chunk(_make_output("some_special_token"))
            assert len(events) == 0

    def test_standard_path_content_then_empty_return(self):
        """Line 361: return [] when no content and no finish."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        # Text that sanitizes to nothing
        from unittest.mock import patch

        with patch("vllm_mlx.service.postprocessor.sanitize_output", return_value=""):
            events = pp.process_chunk(_make_output("some text"))
            # sanitize returned empty, no finish → empty list
            content_events = [e for e in events if e.type == "content"]
            assert len(content_events) == 0


# =====================================================================
# JSON mode preamble stripping (#46)
# =====================================================================


class TestJsonModePreambleStripping:
    """Verify that json_mode=True strips thinking preamble in streaming."""

    def test_preamble_stripped_before_json(self):
        """Thinking text before JSON is suppressed."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()

        # Model outputs thinking preamble
        events1 = pp.process_chunk(_make_output("Let me think about this..."))
        assert len(events1) == 0  # suppressed

        events2 = pp.process_chunk(_make_output(" The answer is "))
        assert len(events2) == 0  # still suppressed

        # JSON starts
        events3 = pp.process_chunk(_make_output('{"result": 42}'))
        content_events = [e for e in events3 if e.type == "content"]
        assert len(content_events) == 1
        assert '{"result": 42}' in content_events[0].content

    def test_json_starts_immediately(self):
        """No preamble — JSON starts in first chunk."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()

        events = pp.process_chunk(_make_output('{"key": "value"}'))
        content_events = [e for e in events if e.type == "content"]
        assert len(content_events) == 1
        assert content_events[0].content == '{"key": "value"}'

    def test_json_array_start(self):
        """JSON array start also triggers emission."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()

        events1 = pp.process_chunk(_make_output("thinking... "))
        assert len(events1) == 0

        events2 = pp.process_chunk(_make_output('[{"item": 1}]'))
        content_events = [e for e in events2 if e.type == "content"]
        assert len(content_events) == 1
        assert content_events[0].content.startswith("[")

    def test_preamble_with_think_tags(self):
        """<think>...</think> before JSON is stripped."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()

        events1 = pp.process_chunk(_make_output("<think>Let me reason"))
        assert len(events1) == 0

        events2 = pp.process_chunk(_make_output("</think>"))
        assert len(events2) == 0

        events3 = pp.process_chunk(_make_output('{"answer": true}'))
        content_events = [e for e in events3 if e.type == "content"]
        assert len(content_events) == 1
        assert '{"answer": true}' in content_events[0].content

    def test_json_mode_false_passes_through(self):
        """Without json_mode, preamble is not stripped."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=False)
        pp.reset()

        events = pp.process_chunk(_make_output("thinking text"))
        content_events = [e for e in events if e.type == "content"]
        assert len(content_events) == 1

    def test_json_mode_with_reasoning_parser_skips_stripping(self):
        """When reasoning parser is active, json_mode stripping is skipped."""
        parser = MagicMock()
        delta_msg = MagicMock()
        delta_msg.content = "thinking preamble"
        delta_msg.reasoning = "reasoning"
        parser.extract_reasoning_streaming.return_value = delta_msg

        cfg = _make_cfg(reasoning_parser=parser)
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()

        # Should go through reasoning parser path, not standard path
        events = pp.process_chunk(_make_output("thinking"))
        # Content comes from reasoning parser, not json stripping
        assert any(e.type in ("content", "reasoning") for e in events)

    def test_json_delimiter_mid_chunk(self):
        """JSON delimiter in the middle of a chunk — emit from delimiter."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()

        events = pp.process_chunk(_make_output('preamble text {"key": 1}'))
        content_events = [e for e in events if e.type == "content"]
        assert len(content_events) == 1
        assert content_events[0].content.startswith("{")
        assert "preamble" not in content_events[0].content

    def test_after_json_start_normal_streaming(self):
        """After JSON start, subsequent chunks pass through normally."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()

        pp.process_chunk(_make_output('{"key": '))
        events = pp.process_chunk(_make_output('"value"}'))
        content_events = [e for e in events if e.type == "content"]
        assert len(content_events) == 1
        assert content_events[0].content == '"value"}'

    def test_response_format_text_does_not_activate(self):
        """response_format type=text should NOT activate json_mode."""
        cfg = _make_cfg()
        # Simulate: json_mode should be False when type is "text"
        pp = StreamingPostProcessor(cfg, json_mode=False)
        pp.reset()

        events = pp.process_chunk(_make_output("Normal text without JSON"))
        content_events = [e for e in events if e.type == "content"]
        assert len(content_events) == 1
        assert "Normal text" in content_events[0].content

    def test_stream_ends_during_preamble(self):
        """Model never outputs JSON delimiter — finalize returns empty."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()

        # All chunks are preamble, no JSON delimiter
        events1 = pp.process_chunk(_make_output("thinking..."))
        assert len(events1) == 0

        events2 = pp.process_chunk(_make_output("still thinking"))
        assert len(events2) == 0

        # Stream ends — finalize has no tool calls, no JSON
        final = pp.finalize()
        assert len(final) == 0

    def test_json_mode_does_not_corrupt_accumulated_text(self):
        """json_mode preamble buffer is separate from accumulated_text."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()

        # Preamble phase — accumulated_text should be unaffected
        pp.process_chunk(_make_output("preamble "))
        assert pp._json_preamble_buffer == "preamble "
        assert pp.accumulated_text == ""  # NOT mutated by preamble

        pp.process_chunk(_make_output('{"key": 1}'))
        assert pp._json_preamble_stripped is True

    def test_braces_inside_think_tags_ignored(self):
        """{ inside <think> tags should NOT trigger JSON start."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()

        # Think block with braces inside
        events1 = pp.process_chunk(_make_output("<think>if x > 0 { return }</think>"))
        assert len(events1) == 0  # still in preamble, { was inside <think>

        # Actual JSON after think block
        events2 = pp.process_chunk(_make_output('{"result": true}'))
        content_events = [e for e in events2 if e.type == "content"]
        assert len(content_events) == 1
        assert content_events[0].content == '{"result": true}'

    def test_unclosed_think_tag_suppresses(self):
        """Unclosed <think> should suppress until closed + JSON found."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()

        events1 = pp.process_chunk(_make_output("<think>still thinking {"))
        assert len(events1) == 0

        events2 = pp.process_chunk(_make_output("more </think>"))
        assert len(events2) == 0  # no JSON delimiter yet

        events3 = pp.process_chunk(_make_output('{"answer": 42}'))
        content_events = [e for e in events3 if e.type == "content"]
        assert len(content_events) == 1
        assert '{"answer": 42}' in content_events[0].content


class TestRequestForwardedToToolParser:
    """#171 regression: streaming parsers (qwen3_coder) need request.tools
    for schema-driven type conversion. Without it, raw XML leaks to delta.content."""

    def test_request_forwarded_to_streaming_parser(self):
        """request kwarg is passed to extract_tool_calls_streaming."""
        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = {"content": ""}

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_parser_instance=tool_parser,
        )
        request_dict = {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "read", "parameters": {}},
                }
            ]
        }
        pp = StreamingPostProcessor(cfg, request=request_dict)
        pp.reset()

        pp.process_chunk(_make_output("<tool_call>"))
        kwargs = tool_parser.extract_tool_calls_streaming.call_args.kwargs
        assert kwargs.get("request") is request_dict

    def test_request_forwarded_to_finalize_fallback(self):
        """finalize() fallback also forwards request to extract_tool_calls."""
        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = {"content": ""}
        tool_parser.has_pending_tool_call.return_value = True
        tool_parser.extract_tool_calls.return_value = MagicMock(tools_called=False)

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_parser_instance=tool_parser,
        )
        request_dict = {"tools": [{"type": "function", "function": {"name": "x"}}]}
        pp = StreamingPostProcessor(cfg, request=request_dict)
        pp.reset()

        pp.process_chunk(_make_output("<tool_call>incomplete"))
        pp.finalize()

        kwargs = tool_parser.extract_tool_calls.call_args.kwargs
        assert kwargs.get("request") is request_dict

    def test_request_defaults_to_none(self):
        """No request → None is forwarded (preserves prior behavior)."""
        tool_parser = MagicMock()
        tool_parser.extract_tool_calls_streaming.return_value = {"content": ""}

        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_parser_instance=tool_parser,
        )
        pp = StreamingPostProcessor(cfg)
        pp.reset()

        pp.process_chunk(_make_output("<tool_call>"))
        kwargs = tool_parser.extract_tool_calls_streaming.call_args.kwargs
        assert kwargs.get("request") is None

    def test_qwen3_coder_streaming_with_request_extracts_tool_call(self):
        """End-to-end: real qwen3_coder parser + request → structured tool_calls,
        not raw XML in content. Reproduces #171."""
        from vllm_mlx.tool_parsers.qwen3coder_tool_parser import (
            Qwen3CoderToolParser,
        )

        parser = Qwen3CoderToolParser(tokenizer=None)
        cfg = _make_cfg(
            enable_auto_tool_choice=True,
            tool_parser_instance=parser,
        )
        request_dict = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                }
            ]
        }
        pp = StreamingPostProcessor(cfg, tools_requested=True, request=request_dict)
        pp.reset()

        # Feed the canonical Qwen3-Coder XML in tokens-ish chunks.
        full_xml = (
            "<tool_call>\n"
            "<function=read>\n"
            "<parameter=path>\n"
            "HEARTBEAT.md\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        all_events = []
        for piece in [
            "<tool_call>\n",
            "<function=read>\n",
            "<parameter=path>\n",
            "HEARTBEAT.md\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]:
            all_events.extend(pp.process_chunk(_make_output(piece)))
        all_events.extend(pp.finalize())

        # Must produce at least one tool_call event with the function name.
        tool_events = [e for e in all_events if e.type == "tool_call"]
        assert tool_events, (
            f"#171 regression: no tool_call events for {full_xml!r}; "
            f"events={[(e.type, getattr(e, 'content', None)) for e in all_events]}"
        )
        # Tool name should appear in at least one event.
        names_seen = []
        for e in tool_events:
            for tc in e.tool_calls or []:
                fn = tc.get("function", {}).get("name")
                if fn:
                    names_seen.append(fn)
        assert "read" in names_seen, (
            f"#171: tool_call emitted but function name 'read' missing; "
            f"names_seen={names_seen}"
        )
