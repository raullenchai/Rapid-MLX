# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1359 — streaming tool-call suppression must be
bounded.

A streaming ``/v1/chat/completions`` request whose model opens a tool-call
block that never closes (a literal ``<tool_call>``/``<function=`` in generated
code that skews the open/close balance, oversized/degenerate arguments, or a
repetition loop) used to have every subsequent content delta suppressed
forever — the tool parser returns ``None`` for each chunk, the SSE generator
yields nothing, and the keepalive layer emits only ``: keepalive`` comments,
so no client timeout can fire (2207 tokens generated, 1 chunk on the wire,
518s). The postprocessor now bounds the suppression and releases the buffered
content as text once the budget is exceeded.

Pure-logic: a stub tool parser drives ``StreamingPostProcessor.process_chunk``.
No model, no GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm_mlx.service.postprocessor import (
    _MAX_TOOL_SUPPRESSION_BYTES,
    StreamingPostProcessor,
)


def _make_cfg(**overrides):
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = None
    cfg.tool_parser_instance = None
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_output(text="", finished=False):
    out = MagicMock()
    out.new_text = text
    out.finished = finished
    out.channel = None
    out.finish_reason = "stop" if finished else None
    out.prompt_tokens = 10
    out.completion_tokens = 5
    out.tokens = []
    out.logprobs = None
    out.tool_calls = None
    return out


class _NeverClosingToolParser:
    """Once it sees a ``<tool_call>`` opener it suppresses (returns ``None``)
    for every subsequent delta, simulating a block whose close tag never
    arrives (issue #1359)."""

    def __init__(self):
        self._opened = False

    def extract_tool_calls_streaming(
        self, previous_text, current_text, delta_text, request=None
    ):
        if "<tool_call>" in current_text:
            self._opened = True
        if self._opened:
            return None  # never closes → suppress forever (pre-fix wedge)
        return {"content": delta_text}

    def has_pending_tool_call(self, text):
        return "<tool_call>" in text

    def extract_tool_calls(self, text, request=None):
        r = MagicMock()
        r.tools_called = False
        r.tool_calls = []
        return r

    def flush_held_content(self, text):
        return ""


def _pp():
    pp = StreamingPostProcessor(_make_cfg())
    pp.reset()
    pp.tool_parser = _NeverClosingToolParser()
    pp.tool_markup_possible = True  # skip the no-markup fast path
    return pp


def _content(events):
    return "".join(e.content or "" for e in events if e.type == "content")


def test_unclosed_block_suppressed_under_budget():
    pp = _pp()
    # Content before the opener flows normally.
    assert "Intro" in _content(pp.process_chunk(_make_output("Intro. ")))
    # Opener → enter suppression.
    assert _content(pp.process_chunk(_make_output("<tool_call>"))) == ""
    # A body smaller than the budget stays suppressed (real tool calls close
    # well under it), and nothing is released yet.
    body = "The quick brown fox. " * 100  # ~2 KB
    assert _content(pp.process_chunk(_make_output(body))) == ""
    assert pp._tool_suppression_released is False


def test_unclosed_block_released_after_budget():
    pp = _pp()
    pp.process_chunk(_make_output("<tool_call>"))
    # One body past the budget must release the withheld text as content.
    big = "The quick brown fox. " * (_MAX_TOOL_SUPPRESSION_BYTES // 20 + 10)
    emitted = _content(pp.process_chunk(_make_output(big)))
    assert pp._tool_suppression_released is True
    assert "fox" in emitted  # the withheld content reached the wire


def test_stream_resumes_as_content_after_release():
    pp = _pp()
    pp.process_chunk(_make_output("<tool_call>"))
    pp.process_chunk(_make_output("x" * (_MAX_TOOL_SUPPRESSION_BYTES + 32)))
    assert pp._tool_suppression_released is True
    # Once released, tool detection is off for the turn: later deltas stream
    # straight through as content instead of wedging.
    assert "recovered" in _content(pp.process_chunk(_make_output("recovered tail")))


def test_reset_clears_release_latch():
    pp = _pp()
    pp.process_chunk(_make_output("<tool_call>"))
    pp.process_chunk(_make_output("y" * (_MAX_TOOL_SUPPRESSION_BYTES + 32)))
    assert pp._tool_suppression_released is True
    pp.reset()
    assert pp._tool_suppression_released is False
    assert pp._tool_suppressed_buffer == ""


def test_legit_short_tool_call_not_released():
    """A real tool call closes well under the budget → the stub that *does*
    close must never trip the release path (no false positive)."""

    class _ClosingToolParser(_NeverClosingToolParser):
        def extract_tool_calls_streaming(
            self, previous_text, current_text, delta_text, request=None
        ):
            if "</tool_call>" in current_text:
                return {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ]
                }
            if "<tool_call>" in current_text:
                return None  # buffering the small args body
            return {"content": delta_text}

    pp = StreamingPostProcessor(_make_cfg())
    pp.reset()
    pp.tool_parser = _ClosingToolParser()
    pp.tool_markup_possible = True

    pp.process_chunk(_make_output('<tool_call>{"name":"f",'))
    pp.process_chunk(_make_output('"arguments":{}}'))
    pp.process_chunk(_make_output("</tool_call>"))
    assert pp._tool_suppression_released is False
    assert pp.tool_calls_detected is True


def test_no_release_once_a_tool_call_reached_the_wire():
    """The ``_tool_calls_emitted_to_wire == 0`` guard: once a tool call has
    streamed to the wire, a large following buffered block is a legitimate
    continuation (e.g. a second call whose args stream), not a wedge — the
    budget must NOT release, and nothing is accumulated into the suppression
    buffer at all."""

    class _EmitThenBufferParser(_NeverClosingToolParser):
        def __init__(self):
            super().__init__()
            self._emitted = False

        def extract_tool_calls_streaming(
            self, previous_text, current_text, delta_text, request=None
        ):
            if "<tool_call>" in current_text and not self._emitted:
                self._emitted = True
                return {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ]
                }
            if self._emitted:
                return None  # now buffering a large second block forever
            return {"content": delta_text}

    pp = StreamingPostProcessor(_make_cfg())
    pp.reset()
    pp.tool_parser = _EmitThenBufferParser()
    pp.tool_markup_possible = True

    pp.process_chunk(_make_output("<tool_call>"))
    assert pp._tool_calls_emitted_to_wire > 0
    pp.process_chunk(_make_output("z" * (_MAX_TOOL_SUPPRESSION_BYTES + 100)))
    assert pp._tool_suppression_released is False  # guard held
    assert pp._tool_suppressed_buffer == ""  # never accumulated past the guard
