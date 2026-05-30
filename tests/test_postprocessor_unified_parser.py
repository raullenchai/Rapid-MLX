# SPDX-License-Identifier: Apache-2.0
"""
Phase 2 parity tests for the unified ``Parser.parse_delta`` path in
``StreamingPostProcessor``.

When ``RAPID_MLX_UNIFIED_PARSER=1`` the post-processor takes its
text-based reasoning + tool flow through the new orchestrator instead of
the legacy two-call (``extract_reasoning_streaming`` →
``_detect_tool_calls``) sequence. These tests pin:

1. The opt-in is a no-op when the env flag is unset.
2. With the flag set, a Qwen3-style ``<think>...</think>...<tool_call>...``
   stream produces the same StreamEvent shapes the legacy path would
   have produced (reasoning → content boundary, tool_call emission).
3. The flag does NOT activate for channel-routed (Gemma 4 / harmony)
   outputs — those still route through ``_process_channel_routed`` since
   OutputRouter is token-level and the orchestrator is text-level.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from vllm_mlx.service.postprocessor import StreamingPostProcessor


def _make_cfg(**overrides):
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


# ----------------------- opt-in is no-op by default -----------------------


def test_unified_parser_is_none_without_env_flag(monkeypatch):
    """Default OFF: ``unified_parser`` stays None so the legacy path is
    used. This is the Phase 1→2 safety contract — landing the new code
    must not change existing behavior for any deployed install."""
    monkeypatch.delenv("RAPID_MLX_UNIFIED_PARSER", raising=False)
    cfg = _make_cfg(
        reasoning_parser_name="qwen3",
        tool_call_parser="hermes",
        enable_auto_tool_choice=True,
    )
    pp = StreamingPostProcessor(cfg, tools_requested=True)
    assert pp.unified_parser is None


def test_unified_parser_not_built_without_reasoning_parser(monkeypatch):
    """The unified path is only an improvement over the legacy
    ``_process_with_reasoning``. With no reasoning parser the standard
    path is correct and the orchestrator gives nothing — don't pay the
    construction cost."""
    monkeypatch.setenv("RAPID_MLX_UNIFIED_PARSER", "1")
    cfg = _make_cfg(tool_call_parser="hermes", enable_auto_tool_choice=True)
    pp = StreamingPostProcessor(cfg, tools_requested=True)
    assert pp.unified_parser is None


# ----------------------- unified path activation -----------------------


def test_unified_parser_built_when_env_and_reasoning_parser_set(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_UNIFIED_PARSER", "1")
    cfg = _make_cfg(
        reasoning_parser_name="qwen3",
        tool_call_parser="hermes",
        enable_auto_tool_choice=True,
    )
    pp = StreamingPostProcessor(cfg, tools_requested=True)
    assert pp.unified_parser is not None
    # Sanity: orchestrator holds the same sub-parser instances the
    # post-processor already built, not freshly-rebuilt ones.
    assert pp.unified_parser._reasoning_parser is pp.reasoning_parser
    assert pp.unified_parser._tool_parser is pp.tool_parser


# ----------------------- streaming behavior parity -----------------------


def _stream_chunks(pp, chunks):
    """Feed a list of text chunks through ``process_chunk`` and return the
    aggregated event list (with the final chunk marked finished)."""
    events: list = []
    last = len(chunks) - 1
    for i, c in enumerate(chunks):
        out = _make_output(c, finished=(i == last))
        events.extend(pp.process_chunk(out))
    return events


def _event_types(events):
    return [(e.type, e.content, e.reasoning) for e in events]


def test_qwen3_think_then_content_unified_path(monkeypatch):
    """Classic Qwen3 stream: think block then answer. Boundary must
    transition cleanly from reasoning → content with no leak in either
    direction."""
    monkeypatch.setenv("RAPID_MLX_UNIFIED_PARSER", "1")
    cfg = _make_cfg(reasoning_parser_name="qwen3")
    pp = StreamingPostProcessor(cfg, tools_requested=False)
    pp.reset()

    events = _stream_chunks(
        pp,
        ["<think>", "Working on it.", "</think>", "Answer is 42."],
    )

    # We expect at least one reasoning event from "Working on it." and
    # one content event for "Answer is 42." — interleaved with a finish.
    has_reasoning = any(e.type == "reasoning" and e.reasoning for e in events)
    has_content = any(
        (e.type in ("content", "finish"))
        and (
            e.content == "Answer is 42." or (e.content or "").endswith("Answer is 42.")
        )
        for e in events
    )
    assert has_reasoning, f"expected a reasoning event, got: {_event_types(events)}"
    assert has_content, f"expected the answer in content, got: {_event_types(events)}"


def test_qwen3_plus_hermes_emits_tool_call_through_unified_path(monkeypatch):
    """Composed stream: ``<think>...</think>`` then a Hermes
    ``<tool_call>{...}</tool_call>`` block. The orchestrator hands off
    to the hermes parser when reasoning ends; the tool_call delta must
    appear as a ``tool_call`` StreamEvent (not as content)."""
    monkeypatch.setenv("RAPID_MLX_UNIFIED_PARSER", "1")
    cfg = _make_cfg(
        reasoning_parser_name="qwen3",
        tool_call_parser="hermes",
        enable_auto_tool_choice=True,
    )
    pp = StreamingPostProcessor(cfg, tools_requested=True)
    pp.reset()

    tool_call_payload = json.dumps(
        {"name": "get_weather", "arguments": {"city": "Paris"}}
    )
    events = _stream_chunks(
        pp,
        [
            "<think>",
            "plan",
            "</think>",
            "<tool_call>",
            tool_call_payload,
            "</tool_call>",
        ],
    )

    # A tool_call event must appear somewhere in the stream.
    tool_events = [e for e in events if e.type == "tool_call" and e.tool_calls]
    assert tool_events, f"expected tool_call event, got: {_event_types(events)}"


# ----------------------- channel-routed bypass -----------------------


def test_channel_routed_bypasses_unified_path(monkeypatch):
    """Even with the env flag set, OutputRouter outputs (Gemma 4,
    harmony) must still go through ``_process_channel_routed`` — the
    orchestrator is text-level and would lose the token-level channel
    metadata the engine already attached."""
    monkeypatch.setenv("RAPID_MLX_UNIFIED_PARSER", "1")
    cfg = _make_cfg(reasoning_parser_name="qwen3")
    pp = StreamingPostProcessor(cfg, tools_requested=False)
    pp.reset()

    out = _make_output("answer", channel="content")
    events = pp.process_chunk(out)

    # The route picked ``_process_channel_routed`` because output.channel
    # was set. Verify a content event surfaced.
    assert any(e.type == "content" and e.content == "answer" for e in events), (
        f"expected content event from channel-routed path, got: {_event_types(events)}"
    )


# ----------------------- reset clears orchestrator state -----------------------


def test_reset_clears_stream_state_between_requests(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_UNIFIED_PARSER", "1")
    cfg = _make_cfg(reasoning_parser_name="qwen3")
    pp = StreamingPostProcessor(cfg, tools_requested=False)

    pp.reset()
    pp.process_chunk(_make_output("<think>old"))
    pp.process_chunk(_make_output("</think>"))
    assert pp.unified_parser._stream_state.reasoning_ended is True

    pp.reset()
    assert pp.unified_parser._stream_state.reasoning_ended is False
    assert pp.unified_parser._stream_state.previous_text == ""


# ----------------------- safety: legacy path still callable -----------------------


def test_legacy_path_still_used_with_flag_off(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_UNIFIED_PARSER", raising=False)
    cfg = _make_cfg(reasoning_parser_name="qwen3")
    pp = StreamingPostProcessor(cfg, tools_requested=False)
    pp.reset()

    # The legacy path must still produce events for the same stream.
    events = _stream_chunks(
        pp,
        ["<think>", "thought", "</think>", "ok"],
    )
    assert events, "legacy path must still work with flag off"
