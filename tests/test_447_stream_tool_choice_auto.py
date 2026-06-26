# SPDX-License-Identifier: Apache-2.0
"""rapid-desktop#447 — streaming + tool_choice=auto/required must emit
tool_call deltas on hermes parser + qwen3 model family.

The reported failure mode (v0.8.18 / v0.9.6): a model that streams
the Nemotron-shape envelope

    <tool_call>
    <function=NAME>
    <parameter=K>V</parameter>
    </function>
    </tool_call>

under ``enable_thinking=False`` (the auto-disable injected when ``tools``
are present and the client did not pin a thinking preference) used to
finish with zero ``delta.tool_calls`` and ``finish_reason="stop"`` — the
default OpenAI SDK / LangChain / LiteLLM streaming-tool-call pattern saw
"the model never says anything".

Root cause: ``StreamingPostProcessor._should_route_through_reasoning``
tentatively re-routed any chunk whose head was a strict prefix of
``<think>`` into the reasoning lane (R8-M2 safety net for SSE-split
``<think>`` tags). With ``enable_thinking=False`` and an in-flight
``<tool_call>`` envelope, the next ``<`` byte (the opener of an inner
``<function=...>`` / ``<parameter=...>`` tag) matched ``<`` as a strict
prefix of ``<think>``, was eaten into the reasoning lane, and never
reached the tool parser's ``tool_accumulated_text``. The assembled body
read ``<tool_call>\nfunction=...\n<parameter=...`` — outer ``<function=``
corrupted, Nemotron regex failed, ``has_incomplete`` stayed True until
end-of-stream, and the stream finished with ``finish_reason="stop"``.

Fix: when the tool parser is mid-block on an unclosed
``<tool_call>``-style envelope, skip the split-prefix ``<think>``
rescue. The tool parser owns its own held-back machinery
(``hermes._safe_content_prefix``) so the ``<`` lands in the tool parser
and is correctly assembled with the rest of the inner tag.

Secondary fix: ``tool_choice="required"`` (single tool) and
``tool_choice={"type":"function",...}`` inject a JSON-shape
``forced_assistant_prefix`` (``<tool_call>\n{"name":..., "arguments": ``)
that misaligns with qwen3-family Nemotron-native output and produces a
hybrid wire shape neither parser branch can recover. The non-stream
chat route falls back to ``_synthesize_forced_tool_call`` for parity
with channel-routed paths; the streaming route now does the same so
``stream=true`` clients see the OpenAI tool_call-guaranteed contract
honored on both surfaces.
"""

from __future__ import annotations

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


def _make_output(text: str = "", finished: bool = False):
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


def _drive(pp: StreamingPostProcessor, deltas: list[str]) -> dict:
    """Drive deltas through the postprocessor and return the SSE-level
    reasoning / content / tool_calls aggregate."""
    all_events = []
    for i, d in enumerate(deltas):
        finished = i == len(deltas) - 1
        all_events.extend(pp.process_chunk(_make_output(d, finished=finished)))
    reasoning = "".join(
        getattr(e, "reasoning", "") or "" for e in all_events if e.type == "reasoning"
    )
    content = "".join(
        getattr(e, "content", "") or ""
        for e in all_events
        if e.type in ("content", "finish")
    )
    tool_calls: list = []
    for e in all_events:
        if e.type == "tool_call" and e.tool_calls:
            tool_calls.extend(e.tool_calls)
    return {
        "reasoning": reasoning,
        "content": content,
        "tool_calls": tool_calls,
        "events": all_events,
    }


def _pp(enable_thinking=False):
    cfg = _make_cfg(
        reasoning_parser_name="qwen3",
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
    )
    pp = StreamingPostProcessor(
        cfg, tools_requested=True, enable_thinking=enable_thinking
    )
    pp.reset()
    return pp


# =====================================================================
# Primary repro — auto + Nemotron-shape envelope (no forced_prefix)
# =====================================================================


class TestNemotronShapeEnvelopeReachesParser:
    """The bug-#447 wire shape: qwen3.5-4b under ``enable_thinking=False``
    + ``tool_choice="auto"`` streams a Nemotron-shape envelope. Each
    inner ``<`` (of ``<function=`` / ``<parameter=``) used to be eaten
    by the reasoning-lane split-prefix rescue. After the fix the
    envelope reaches the tool parser intact and the structured call
    surfaces on ``delta.tool_calls``."""

    def test_full_nemotron_envelope_emits_tool_call(self):
        """Exact chunk sequence from the live qwen3.5-4b-4bit repro
        (BatchedEngine emits one tokenizer-token per delta)."""
        pp = _pp(enable_thinking=False)
        result = _drive(
            pp,
            [
                "<tool_call>",
                "\n",
                "<",
                "function",
                "=get",
                "_weather",
                ">",
                "\n",
                "<",
                "parameter",
                "=",
                "location",
                ">",
                "\n",
                "SF",
                "\n",
                "</",
                "parameter",
                ">",
                "\n",
                "</",
                "function",
                ">",
                "\n",
                "</tool_call>",
            ],
        )
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        # arguments JSON encodes the recovered parameter dict.
        assert "location" in tc["function"]["arguments"]
        assert "SF" in tc["function"]["arguments"]
        # The wrapper bytes must NOT have leaked into content or
        # reasoning channels.
        assert "<tool_call>" not in result["content"]
        assert "<function=" not in result["content"]
        assert "<parameter=" not in result["content"]
        assert "<tool_call>" not in result["reasoning"]


# =====================================================================
# Regression guards — pre-existing happy paths
# =====================================================================


class TestPreFixHappyPathsPreserved:
    def test_explicit_think_wrapper_still_routed_to_reasoning(self):
        """The R8-M2 explicit-wrapper rescue must STILL fire when no
        tool envelope is in flight."""
        pp = _pp(enable_thinking=False)
        result = _drive(
            pp,
            [
                "<think>",
                "I will call get_weather.",
                "</think>",
                '<tool_call>{"name":"get_weather","arguments":{"location":"NYC"}}</tool_call>',
            ],
        )
        assert "I will call" in result["reasoning"]
        assert "<think>" not in result["content"]
        # The JSON-shape tool_call still extracts.
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_split_think_tag_no_leak_still_works(self):
        """The SSE-split ``<th`` + ``ink>`` opener case must still
        route correctly when there is no preceding tool envelope —
        the gate only skips the split-prefix rescue when an envelope
        is in flight."""
        pp = _pp(enable_thinking=False)
        result = _drive(
            pp,
            [
                "<th",
                "ink>",
                "thinking step one.",
                "</think>",
                '<tool_call>{"name":"foo","arguments":{}}</tool_call>',
            ],
        )
        assert "thinking step one" in result["reasoning"]
        assert "<th" not in result["content"]
        assert "ink>" not in result["content"]

    def test_plain_prose_with_lt_byte_unchanged(self):
        """A plain-text answer with a bare ``<`` byte must continue to
        flow through ``_process_standard`` unchanged (no envelope ⇒
        no in-flight gate)."""
        pp = _pp(enable_thinking=False)
        result = _drive(pp, ["The answer is ", "x < y, ", "always."])
        assert "x < y" in result["content"]
        assert result["reasoning"] == ""


# =====================================================================
# Direct unit-level coverage for the new helper
# =====================================================================


class TestToolEnvelopeInFlightHelper:
    def test_empty_buffer_returns_false(self):
        pp = _pp(enable_thinking=False)
        assert pp._tool_envelope_in_flight() is False

    def test_open_envelope_returns_true(self):
        pp = _pp(enable_thinking=False)
        pp.tool_accumulated_text = "<tool_call>\n{"
        assert pp._tool_envelope_in_flight() is True

    def test_closed_envelope_returns_false(self):
        pp = _pp(enable_thinking=False)
        pp.tool_accumulated_text = (
            '<tool_call>{"name":"foo","arguments":{}}</tool_call>'
        )
        assert pp._tool_envelope_in_flight() is False

    def test_minimax_envelope_in_flight(self):
        pp = _pp(enable_thinking=False)
        pp.tool_accumulated_text = '<minimax:tool_call><invoke name="x">'
        assert pp._tool_envelope_in_flight() is True
