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

    # Codex round-1 regression: no content event may carry the raw
    # ``<tool_call>`` / closing-tag markup. The orchestrator must
    # suppress those chunks while the tool parser is buffering.
    leaked = [
        e
        for e in events
        if e.type == "content"
        and e.content
        and ("<tool_call" in e.content or "</tool_call>" in e.content)
    ]
    assert not leaked, (
        f"tool-call markup leaked as content: {[(e.type, e.content) for e in leaked]}"
    )

    # Codex round-1 regression: tool_calls_detected must be set on the
    # post-processor so finalize() doesn't re-parse accumulated_text and
    # duplicate the call (or stamp the finish event as 'stop').
    assert pp.tool_calls_detected, (
        "expected tool_calls_detected=True after unified path emitted tool_calls"
    )


def test_unified_path_suppresses_partial_tool_call_chunks(monkeypatch):
    """Codex round-1 regression: while the tool parser buffers an
    incomplete ``<tool_call>{...`` body it returns ``None``. The
    orchestrator must NOT fabricate a content chunk from that delta —
    doing so leaks the raw markup to the SSE stream before the
    structured ``tool_calls`` delta arrives."""
    monkeypatch.setenv("RAPID_MLX_UNIFIED_PARSER", "1")
    cfg = _make_cfg(
        reasoning_parser_name="qwen3",
        tool_call_parser="hermes",
        enable_auto_tool_choice=True,
    )
    pp = StreamingPostProcessor(cfg, tools_requested=True)
    pp.reset()

    # Stream up to the OPENING ``<tool_call>`` marker only — no closing
    # tag yet, so the hermes parser will buffer (return None each time).
    out = _make_output("<think>")
    pp.process_chunk(out)
    out = _make_output("</think>")
    pp.process_chunk(out)
    partial_events = pp.process_chunk(_make_output("<tool_call>"))
    partial_events += pp.process_chunk(_make_output('{"name": "x"'))

    # Zero content events should carry the raw markup.
    leaked = [
        e
        for e in partial_events
        if e.type == "content" and e.content and "<tool_call" in e.content
    ]
    assert not leaked, (
        f"<tool_call> markup leaked as content while buffering: "
        f"{[(e.type, e.content) for e in leaked]}"
    )


def test_unified_path_finalize_with_empty_tool_scope_does_not_fall_back(monkeypatch):
    """Codex round-9 #1 regression. Stream ends with ALL output inside
    the reasoning block (e.g. ``<think>...{"name":"x","arguments":{}}``
    with no closing ``</think>`` and no post-boundary content).
    ``tool_phase_text`` is the empty string. The legacy fallback would
    treat ``""`` as falsey and use ``self.accumulated_text`` (the
    full reasoning body), then run ``extract_tool_calls`` over it and
    emit a bogus tool_call. The unified-scope flag must short-circuit
    the ``or`` fallback so empty-on-purpose stays empty.
    """
    monkeypatch.setenv("RAPID_MLX_UNIFIED_PARSER", "1")
    cfg = _make_cfg(
        reasoning_parser_name="qwen3",
        tool_call_parser="hermes",
        enable_auto_tool_choice=True,
    )
    pp = StreamingPostProcessor(cfg, tools_requested=True)
    pp.reset()

    # Stream is all reasoning, no post-boundary content. Note the
    # absence of any ``</think>`` so reasoning_ended never flips —
    # tool_phase_text is "".
    fake_call = '{"name":"x","arguments":{}}'
    _stream_chunks(pp, [f"<think>plan to call x: {fake_call}"])

    assert pp.tool_accumulated_text == "", (
        f"expected empty tool scope, got {pp.tool_accumulated_text!r}"
    )

    final_events = pp.finalize()
    tool_events = [e for e in final_events if e.type == "tool_call"]
    assert not tool_events, (
        f"finalize fell back to accumulated_text and parsed reasoning: "
        f"{[(e.type, e.tool_calls) for e in tool_events]}"
    )


def test_unified_path_finalize_does_not_parse_reasoning_as_tool(monkeypatch):
    """Codex round-7 regression. The model can mention a bare JSON
    tool-call shape inside its ``<think>`` block:

      <think>I should call x with {"name":"x","arguments":{}}</think>
      The answer is 42.

    finalize() runs a fallback ``extract_tool_calls`` on the
    accumulated text when no streaming tool call surfaced — that
    fallback must be scoped to post-reasoning text only, otherwise
    the reasoning mention would synthesize a bogus tool_call event
    at end-of-stream.
    """
    monkeypatch.setenv("RAPID_MLX_UNIFIED_PARSER", "1")
    cfg = _make_cfg(
        reasoning_parser_name="qwen3",
        tool_call_parser="hermes",
        enable_auto_tool_choice=True,
    )
    pp = StreamingPostProcessor(cfg, tools_requested=True)
    pp.reset()

    fake_call = '{"name":"x","arguments":{}}'
    events = _stream_chunks(
        pp,
        [
            f"<think>I should call x with {fake_call}",
            "</think>",
            "The answer is 42.",
        ],
    )
    events.extend(pp.finalize())

    tool_events = [e for e in events if e.type == "tool_call"]
    assert not tool_events, (
        f"finalize parsed bare JSON from reasoning as a bogus tool call: "
        f"{[(e.type, e.tool_calls) for e in tool_events]}"
    )
    # tool_accumulated_text must NOT contain the reasoning prefix.
    assert fake_call not in pp.tool_accumulated_text or (
        "<think>" not in pp.tool_accumulated_text
        and "I should call x" not in pp.tool_accumulated_text
    ), f"tool_accumulated_text leaked reasoning prefix: {pp.tool_accumulated_text!r}"


def test_unified_path_suppresses_tool_prefix_on_split_boundary_chunk(monkeypatch):
    """Codex round-6 regression. Chunk split pattern:

      chunk 1: ``<think>plan``     (reasoning body)
      chunk 2: ``</think><tool_call>``  (boundary + opening tag)
      chunk 3: ``{...}</tool_call>``   (tool-call body)

    On chunk 2 the reasoning parser emits ``content=<tool_call>``
    while the tool parser returns None (buffering until the closing
    tag arrives in chunk 3). The orchestrator must NOT let the
    reasoning parser's content side flow through — the raw
    ``<tool_call>`` markup must stay suppressed.
    """
    monkeypatch.setenv("RAPID_MLX_UNIFIED_PARSER", "1")
    cfg = _make_cfg(
        reasoning_parser_name="qwen3",
        tool_call_parser="hermes",
        enable_auto_tool_choice=True,
    )
    pp = StreamingPostProcessor(cfg, tools_requested=True)
    pp.reset()

    payload = json.dumps({"name": "x", "arguments": {}})
    events = _stream_chunks(
        pp,
        [
            "<think>plan",
            "</think><tool_call>",
            f"{payload}</tool_call>",
        ],
    )

    leaked = [
        e
        for e in events
        if e.type == "content" and e.content and "<tool_call" in e.content
    ]
    assert not leaked, (
        f"tool-call prefix leaked on split boundary chunk: "
        f"{[(e.type, e.content) for e in leaked]}"
    )
    # The real tool call still must surface.
    tool_events = [e for e in events if e.type == "tool_call" and e.tool_calls]
    assert tool_events, f"tool_call dropped: {_event_types(events)}"


def test_unified_path_does_not_leak_reasoning_into_tool_parser(monkeypatch):
    """Codex round-5 regression (exact POC reproduction).

    When the model thinks aloud about tool-call syntax inside its
    reasoning block (e.g. ``<think>...the closing tag is
    ``</tool_call>``...</think>``), the orchestrator must NOT feed
    those mentions to the tool parser. Otherwise the tool parser's
    open/close counter goes negative (sees ``</tool_call>`` before
    any ``<tool_call>``) and the next real tool-call chunk gets
    treated as pass-through content instead of buffered + emitted.

    Symptom under the bug: legacy path emits a ``tool_call``
    StreamEvent; unified path emits the raw ``<tool_call>{...}``
    JSON as ``content``.
    """
    monkeypatch.setenv("RAPID_MLX_UNIFIED_PARSER", "1")
    cfg = _make_cfg(
        reasoning_parser_name="qwen3",
        tool_call_parser="hermes",
        enable_auto_tool_choice=True,
    )
    pp = StreamingPostProcessor(cfg, tools_requested=True)
    pp.reset()

    payload = json.dumps({"name": "x", "arguments": {}})
    events = _stream_chunks(
        pp,
        [
            "<think>remember closing tag is </tool_call>",
            "</think>",
            f"<tool_call>{payload}</tool_call>",
        ],
    )

    # The real tool call must surface as a structured tool_call event.
    tool_events = [e for e in events if e.type == "tool_call" and e.tool_calls]
    assert tool_events, (
        f"tool_call dropped — reasoning mention of </tool_call> "
        f"poisoned the tool parser's counter. events: {_event_types(events)}"
    )

    # And no content event may carry raw ``<tool_call>`` JSON.
    leaked = [
        e
        for e in events
        if e.type == "content" and e.content and "<tool_call" in e.content
    ]
    assert not leaked, (
        f"tool-call JSON leaked as content: {[(e.type, e.content) for e in leaked]}"
    )


def test_unified_path_suppresses_lone_think_tag_without_tools(monkeypatch):
    """Codex round-2 regression: when a reasoning parser is wired but
    no tool parser exists, the reasoning parser intentionally returns
    ``None`` for standalone special tokens (e.g. ``<think>``,
    ``</think>``). The orchestrator must NOT fabricate a content delta
    from those — that would leak the marker to the SSE stream.
    """
    monkeypatch.setenv("RAPID_MLX_UNIFIED_PARSER", "1")
    cfg = _make_cfg(reasoning_parser_name="qwen3")
    pp = StreamingPostProcessor(cfg, tools_requested=False)
    pp.reset()

    events: list = []
    for chunk in ["<think>", "</think>"]:
        events.extend(pp.process_chunk(_make_output(chunk)))

    leaked = [
        e
        for e in events
        if e.type == "content" and e.content and ("<think" in e.content)
    ]
    assert not leaked, (
        f"<think>/</think> markers leaked as content with no tool parser: "
        f"{[(e.type, e.content) for e in leaked]}"
    )


def test_unified_path_coalesced_boundary_preserves_reasoning_with_tool_call(
    monkeypatch,
):
    """Codex round-2 regression: when a single chunk closes reasoning
    AND completes a tool call (e.g. ``final thought</think><tool_call>
    {...}</tool_call>``), the unified path must surface the trailing
    reasoning text before / alongside the tool_call event. Previously
    it dropped reasoning and content because the early return on
    ``delta_msg.tool_calls`` skipped the reasoning emit.
    """
    monkeypatch.setenv("RAPID_MLX_UNIFIED_PARSER", "1")
    cfg = _make_cfg(
        reasoning_parser_name="qwen3",
        tool_call_parser="hermes",
        enable_auto_tool_choice=True,
    )
    pp = StreamingPostProcessor(cfg, tools_requested=True)
    pp.reset()

    payload = json.dumps({"name": "x", "arguments": {}})
    # Coalesced boundary: reasoning body, </think> closes reasoning,
    # tool_call opens + closes — all in one chunk sequence.
    events = _stream_chunks(
        pp,
        [
            "<think>",
            "final thought",
            f"</think><tool_call>{payload}</tool_call>",
        ],
    )

    reasoning_events = [
        e for e in events if e.type == "reasoning" and (e.reasoning or "")
    ]
    tool_events = [e for e in events if e.type == "tool_call" and e.tool_calls]

    # Both the reasoning body and the tool call must materialize.
    assert reasoning_events, (
        f"reasoning text was dropped on coalesced boundary chunk: "
        f"{_event_types(events)}"
    )
    assert tool_events, (
        f"tool_call event missing on coalesced boundary chunk: {_event_types(events)}"
    )

    # accumulated_reasoning must include the pre-boundary thinking text.
    assert "final thought" in pp.accumulated_reasoning, (
        f"accumulated_reasoning lost pre-boundary text: {pp.accumulated_reasoning!r}"
    )


def test_minimax_unified_path_transitions_to_tool_phase(monkeypatch):
    """Codex round-1 regression: MiniMax doesn't carry a
    ``reasoning_ended`` attribute, so the default
    ``is_reasoning_end_streaming`` (which reads that attr) would keep
    the orchestrator stuck in the reasoning phase forever. MiniMax now
    overrides the method to read ``_decided and not _is_reasoning``."""
    from vllm_mlx.reasoning.minimax_parser import MiniMaxReasoningParser

    p = MiniMaxReasoningParser()
    # Initial state: not decided yet.
    assert p.is_reasoning_end_streaming("", "") is False

    # Force into the "decided to content" state — the unified
    # orchestrator should now consider reasoning ended.
    p._decided = True
    p._is_reasoning = False
    assert p.is_reasoning_end_streaming("", "any content") is True

    # Force into the "decided to reasoning" state — still in the
    # reasoning phase.
    p._is_reasoning = True
    assert p.is_reasoning_end_streaming("", "still thinking") is False


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
