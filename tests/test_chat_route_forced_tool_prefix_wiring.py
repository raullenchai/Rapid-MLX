# SPDX-License-Identifier: Apache-2.0
"""Route-level wiring test for the forced ``tool_choice`` assistant-prefix
injection lever.

The chat route MUST pass ``forced_assistant_prefix`` to the engine's
``chat()`` / ``stream_chat()`` whenever:

  * ``tool_choice == {"type":"function","function":{"name":X}}``, OR
  * ``tool_choice == "required"`` AND there is exactly one tool (the
    named-tool form's unambiguous sibling).

Conversely, the prefix MUST NOT be passed when:
  * ``tool_choice`` is "auto" / "none" / unset.
  * The parser is channel-routed (harmony / gemma4) — those publish tool
    calls via the OutputRouter and the prefix would break the channel
    state machine.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import router as chat_router


class _RecordingEngine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self):
        self.last_chat_kwargs: dict[str, Any] | None = None

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        self.last_chat_kwargs = kwargs
        # Return synthesized tool call so the post-parse enforcement
        # gate is satisfied (we only care about ``forced_assistant_prefix``).
        return GenerationOutput(
            text='<tool_call>\n{"name": "get_weather", "arguments": {}}</tool_call>',
            raw_text='<tool_call>\n{"name": "get_weather", "arguments": {}}</tool_call>',
            prompt_tokens=4,
            completion_tokens=8,
            finished=True,
            finish_reason="stop",
        )


def _make_client(engine, parser="hermes"):
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = parser
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    },
]


def test_forced_named_function_sets_assistant_prefix():
    engine = _RecordingEngine()
    client = _make_client(engine, parser="hermes")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    assert engine.last_chat_kwargs is not None
    prefix = engine.last_chat_kwargs.get("forced_assistant_prefix")
    assert prefix is not None
    assert prefix.startswith("<tool_call>")
    assert '"name": "get_weather"' in prefix


def test_forced_required_with_solo_tool_sets_prefix():
    engine = _RecordingEngine()
    client = _make_client(engine, parser="hermes")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_TOOLS[0]],
            "tool_choice": "required",
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    prefix = engine.last_chat_kwargs.get("forced_assistant_prefix")
    assert prefix is not None
    assert '"name": "get_weather"' in prefix


def test_required_with_multiple_tools_no_prefix():
    """``required`` with multiple tools is ambiguous — we don't pick a
    function for the model. Post-parse enforcement still fires."""
    engine = _RecordingEngine()
    client = _make_client(engine, parser="hermes")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
            "tool_choice": "required",
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    assert engine.last_chat_kwargs.get("forced_assistant_prefix") is None


def test_auto_choice_no_prefix():
    engine = _RecordingEngine()
    client = _make_client(engine, parser="hermes")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
            "tool_choice": "auto",
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    assert engine.last_chat_kwargs.get("forced_assistant_prefix") is None


def test_channel_routed_parser_no_prefix():
    """``harmony`` / ``gemma4`` are channel-routed — no prefix even when
    a function is forced. Prefix would confuse the channel state machine."""
    engine = _RecordingEngine()
    client = _make_client(engine, parser="harmony")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    assert engine.last_chat_kwargs.get("forced_assistant_prefix") is None


# ---------------------------------------------------------------------------
# Postprocessor swallow tests — PR #716 codex r9 BLOCKING #1.
#
# The synthetic prefix chunk yielded by ``BatchedEngine.stream_chat`` must be
# swallowed by ``StreamingPostProcessor`` BEFORE the reasoning parser sees it.
# Without the swallow, ``BaseThinkingReasoningParser`` Case-3 (no ``<think>``
# seen yet → treat as reasoning) routes the raw ``<tool_call>{"name":...``
# bytes into ``accumulated_reasoning`` AND emits a ``reasoning_content`` SSE
# delta to the client when the MiniMax tool-markup redirect at
# ``_process_with_reasoning`` doesn't catch the prefix (chunk-boundary splits,
# future parser variants).
# ---------------------------------------------------------------------------


from unittest.mock import MagicMock  # noqa: E402

from vllm_mlx.service.postprocessor import StreamingPostProcessor  # noqa: E402


def _swallow_cfg(**overrides):
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


def _swallow_output(text: str, finished: bool = False):
    out = MagicMock()
    out.new_text = text
    out.text = text
    out.finished = finished
    out.channel = None
    out.finish_reason = "stop" if finished else None
    out.prompt_tokens = 0
    out.completion_tokens = 0
    out.tokens = []
    out.logprobs = None
    out.tool_calls = None
    return out


_PREFIX = '<tool_call>\n{"name": "get_weather", "arguments":'


def test_forced_prefix_swallow_single_chunk_no_leak():
    """Whole-prefix synthetic chunk emits NO content / reasoning event."""
    pp = StreamingPostProcessor(_swallow_cfg())
    pp.reset()
    pp.seed_forced_assistant_prefix(_PREFIX)

    events = pp.process_chunk(_swallow_output(_PREFIX))
    assert events == [], (
        f"prefix chunk must be fully swallowed; got events: "
        f"{[(e.type, getattr(e, 'content', None), getattr(e, 'reasoning', None)) for e in events]}"
    )
    assert pp._forced_prefix_pending == ""
    # Tool parser context was seeded so a downstream tool parser sees
    # the complete envelope opener.
    assert pp.tool_accumulated_text == _PREFIX


def test_forced_prefix_swallow_split_across_chunks():
    """Engine splits the synthetic prefix across two chunks (shorter than
    the prefix on the first call). The swallow must drain incrementally
    and STILL emit nothing — neither half of the prefix may leak as
    content or reasoning. Regression for the codex r9 BLOCKING scenario
    where chunk-boundary swallow logic only handled "drop first N bytes
    from chunk" but not "remember to drop the rest from the next chunk".
    """
    pp = StreamingPostProcessor(_swallow_cfg())
    pp.reset()
    pp.seed_forced_assistant_prefix(_PREFIX)

    half = len(_PREFIX) // 2
    first_half = _PREFIX[:half]
    second_half = _PREFIX[half:]

    events_a = pp.process_chunk(_swallow_output(first_half))
    assert events_a == [], (
        f"first half of split prefix must be fully swallowed; got: "
        f"{[(e.type, getattr(e, 'content', None), getattr(e, 'reasoning', None)) for e in events_a]}"
    )
    # Pending bytes shrank by exactly the consumed amount — confirms
    # the swallow is byte-count stateful, not positional-once.
    assert pp._forced_prefix_pending == second_half

    events_b = pp.process_chunk(_swallow_output(second_half))
    assert events_b == [], (
        f"second half of split prefix must be fully swallowed; got: "
        f"{[(e.type, getattr(e, 'content', None), getattr(e, 'reasoning', None)) for e in events_b]}"
    )
    assert pp._forced_prefix_pending == ""


def test_forced_prefix_swallow_overshoot_emits_tail_only():
    """Engine merges the prefix with a trailing model token (single chunk
    carries ``prefix + tail``). The swallow must strip the prefix and emit
    the tail through the normal pipeline — neither over-stripping (lost
    model output) nor under-stripping (raw prefix leak)."""
    pp = StreamingPostProcessor(_swallow_cfg())
    pp.reset()
    pp.seed_forced_assistant_prefix(_PREFIX)

    tail = ' {"city": "Tokyo"}}'
    events = pp.process_chunk(_swallow_output(_PREFIX + tail))

    # Exactly one content event, content == tail, no prefix bytes in it.
    content_events = [e for e in events if e.type == "content"]
    assert len(content_events) == 1, (
        f"expected exactly one content event with the tail; got: "
        f"{[(e.type, getattr(e, 'content', None)) for e in events]}"
    )
    assert content_events[0].content == tail
    assert "<tool_call>" not in content_events[0].content
    assert pp._forced_prefix_pending == ""


def test_forced_prefix_no_pollution_of_accumulated_reasoning():
    """The Case-3 reasoning misclassification surface: when a reasoning
    parser is active, the swallow must run BEFORE the parser so prefix
    bytes never pollute ``accumulated_reasoning`` (which feeds
    ``_build_usage`` reasoning-token accounting AND the silent-drop rescue
    path)."""
    from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    cfg = _swallow_cfg(
        reasoning_parser=Qwen3ReasoningParser(),
        reasoning_parser_name=None,  # use pre-built instance
    )
    pp = StreamingPostProcessor(cfg, enable_thinking=True)
    pp.reset()
    pp.seed_forced_assistant_prefix(_PREFIX)

    events = pp.process_chunk(_swallow_output(_PREFIX))
    assert events == []
    # Critical invariant: ``accumulated_reasoning`` must NOT carry the
    # prefix bytes. Without the swallow, Qwen3 Case-3 would classify the
    # whole prefix as reasoning and the MiniMax redirect only zeros the
    # delta — it does NOT roll back the buffer update at line ~1043.
    assert "<tool_call>" not in pp.accumulated_reasoning
    assert pp.accumulated_reasoning == ""
