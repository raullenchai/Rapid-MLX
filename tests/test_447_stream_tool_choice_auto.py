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

    def test_split_outer_opener_does_not_leak(self):
        """Codex r2 MAJOR (PR #948): when the tokenizer chunks the
        outer ``<tool_call>`` opener into ``<`` then ``tool_call>``,
        the bare ``<`` used to be eaten by the reasoning-lane
        split-prefix rescue (``<`` is a strict prefix of ``<think>``).
        The next chunk (``tool_call>``) would then arrive without a
        leading ``<``, miss the standard-path tool detection, and the
        reasoning parser would release the buffered ``<tool_call>``
        as plain ``delta.content`` once the tag failed to complete.

        After the round-2 fix, ambiguous prefixes (``<``, ``<t``) that
        also start a tool envelope opener are deferred to the standard
        path so the tool parser's held-back machinery can buffer the
        partial sentinel safely. The full envelope reaches the parser
        intact and the call surfaces on ``delta.tool_calls`` — no
        ``tool_call>`` leak in either content or reasoning channel."""
        pp = _pp(enable_thinking=False)
        result = _drive(
            pp,
            [
                "<",
                "tool_call>",
                "\n",
                '{"name":"get_weather","arguments":{"location":"SF"}}',
                "\n",
                "</tool_call>",
            ],
        )
        # The outer opener must not have leaked.
        assert "tool_call>" not in result["content"], (
            f"split outer opener leaked into content channel: {result['content']!r}"
        )
        assert "tool_call>" not in result["reasoning"], (
            f"split outer opener leaked into reasoning channel: {result['reasoning']!r}"
        )
        # And the structured call must surface.
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_split_think_opener_still_routes_to_reasoning_with_tools(self):
        """Codex r3 MAJOR (PR #948): the round-2 unconditional defer
        of ``<`` / ``<t`` to the standard path regressed the R8-M2
        split-``<think>`` rescue. After the round-3 hold-forward fix,
        a stream that splits the ``<think>`` opener as ``<`` +
        ``think>`` (or ``<t`` + ``hink>``) under
        ``enable_thinking=False`` + tools enabled must STILL route
        the wrapper to the reasoning lane and not leak into
        ``delta.content``."""
        for first, second in (("<", "think>"), ("<t", "hink>")):
            pp = _pp(enable_thinking=False)
            result = _drive(
                pp,
                [
                    first,
                    second,
                    "abc",
                    "</think>",
                    "answer",
                ],
            )
            # The opener and the reasoning body must not have leaked.
            assert "<think>" not in result["content"], (
                f"split={first!r}+{second!r}: <think> leaked into content: "
                f"{result['content']!r}"
            )
            assert "abc" not in result["content"], (
                f"split={first!r}+{second!r}: reasoning body 'abc' leaked: "
                f"{result['content']!r}"
            )
            # And the reasoning lane must carry the body.
            assert "abc" in result["reasoning"], (
                f"split={first!r}+{second!r}: reasoning body missing from "
                f"reasoning channel: {result['reasoning']!r}"
            )

    def test_held_ambiguous_prefix_flushed_on_empty_finish_only_chunk(self):
        """Codex r4 NIT (PR #948): if the stream emits an ambiguous
        head (``<`` / ``<t``) on a non-finished chunk and then an
        EMPTY finish-only chunk, the held byte was previously dropped
        because the ``not delta_text`` early-return ran before the
        hold-buffer replay. After the round-4 finish-flush, the held
        prefix is replayed through the normal routing path so it
        reaches the wire as content (the only correct outcome for a
        terminal sequence with no disambiguating second chunk)."""
        pp = _pp(enable_thinking=False)
        # Drive deltas manually so we can emit an empty-text finish
        # chunk; the ``_drive`` helper marks the last delta as
        # ``finished=True`` instead, which is a different shape.
        events = list(pp.process_chunk(_make_output("<", finished=False)))
        # Held — no events.
        assert events == [], f"expected ambiguous head to be held, got {events!r}"
        # Now an empty finish-only chunk — the held ``<`` must flush.
        events += list(pp.process_chunk(_make_output("", finished=True)))
        # Find any content that reached the wire.
        wire_content = "".join(
            (getattr(e, "content", "") or "")
            for e in events
            if e.type in ("content", "finish")
        )
        # The held ``<`` must have surfaced — either as a content event
        # or merged into the finish event — and NOT been silently
        # dropped.
        assert "<" in wire_content, (
            f"held ambiguous prefix was dropped on empty finish-only "
            f"chunk; wire_content={wire_content!r} events={events!r}"
        )

    def test_split_outer_opener_two_byte_prefix(self):
        """Same race, but the tokenizer chunks as ``<t`` + ``ool_call>``.
        ``<t`` is a strict prefix of BOTH ``<think>`` AND ``<tool_call>``
        so it must also be deferred to the standard path."""
        pp = _pp(enable_thinking=False)
        result = _drive(
            pp,
            [
                "<t",
                "ool_call>",
                "\n",
                '{"name":"foo","arguments":{}}',
                "\n",
                "</tool_call>",
            ],
        )
        assert "ool_call>" not in result["content"]
        assert "ool_call>" not in result["reasoning"]
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "foo"

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


# =====================================================================
# Streaming synth fallback — chat.py route-level branch (codex r1 NIT #2)
# =====================================================================
#
# Direct unit-level coverage for the new ``stream_chat_completion``
# synthesis gate: forced ``tool_choice`` finishes with zero wire
# tool_calls AND the parser saw nothing → synthesise; forced
# ``tool_choice`` finishes with zero wire tool_calls but the parser DID
# see a different tool (dropped by the forced-name filter) → do NOT
# synthesise (mirrors non-stream ``_mismatched`` 422 case rather than
# silently replacing the model's intent — codex r1 MAJOR #1).
#
# Drives ``stream_chat_completion`` end-to-end against a fake engine so
# the SSE-level shape is asserted (not just the postprocessor state).


import asyncio
import json

import pytest

from vllm_mlx.api.models import ChatCompletionRequest


class _FakeStreamingOutput:
    """Minimal ``GenerationOutput`` shim for the streaming loop."""

    def __init__(self, new_text: str, finished: bool):
        self.new_text = new_text
        self.text = new_text
        self.finished = finished
        self.finish_reason = "stop" if finished else None
        self.channel = None
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.cached_tokens = 0
        self.tokens = []
        self.logprobs = None
        self.tool_calls = None
        self.matched_stop = None
        self.raw_text = new_text


class _FakeEngine:
    """Minimal engine shim — yields a fixed delta sequence."""

    def __init__(self, deltas: list[str]):
        self._deltas = deltas
        self.tokenizer = None
        self.is_mllm = False
        self.supports_tool_calls = True
        self.supports_guided_generation = False

    async def stream_chat(self, **kwargs):
        for i, d in enumerate(self._deltas):
            yield _FakeStreamingOutput(d, finished=(i == len(self._deltas) - 1))

    def build_prompt(self, *args, **kwargs):
        return "prompt"

    def estimate_new_tokens(self, *args, **kwargs):
        return (10, 5)


def _drive_stream(engine, request) -> tuple[list[dict], str | None]:
    """Run ``stream_chat_completion`` against the fake engine, collect
    parsed SSE chunks (skipping ``[DONE]`` and keepalives).

    Returns ``(chunks, final_finish_reason)``.
    """
    from vllm_mlx.routes.chat import stream_chat_completion

    chunks: list[dict] = []

    async def _run():
        gen = stream_chat_completion(
            engine, [{"role": "user", "content": "hi"}], request
        )
        async for sse in gen:
            line = sse.strip()
            if not line.startswith("data: "):
                continue
            body = line[len("data: ") :]
            if body == "[DONE]":
                break
            try:
                chunks.append(json.loads(body))
            except json.JSONDecodeError:
                continue

    asyncio.run(_run())
    finish = None
    for c in reversed(chunks):
        choices = c.get("choices") or []
        if choices and choices[0].get("finish_reason"):
            finish = choices[0]["finish_reason"]
            break
    return chunks, finish


class TestStreamSynthForcedToolChoice:
    """Route-level: forced tool_choice produced no parser-detected call
    → synth fires; forced tool_choice produced a DIFFERENT call (filter
    dropped it) → synth must NOT fire (non-stream parity)."""

    def _request(self, tool_choice):
        return ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "x",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            tool_choice=tool_choice,
            stream=True,
            max_tokens=50,
            chat_template_kwargs={"enable_thinking": False},
        )

    @pytest.fixture(autouse=True)
    def _patch_cfg(self, monkeypatch):
        """Wire a minimal ServerConfig + StreamingPostProcessor that
        matches the qwen3 + hermes production shape."""
        from vllm_mlx.config import server_config

        cfg = server_config.get_config()
        # The cfg singleton is mutated to reflect the qwen3 + hermes
        # path; restore the pre-test value at teardown via monkeypatch.
        monkeypatch.setattr(cfg, "tool_call_parser", "hermes", raising=False)
        monkeypatch.setattr(cfg, "reasoning_parser_name", "qwen3", raising=False)
        monkeypatch.setattr(cfg, "enable_auto_tool_choice", True, raising=False)
        monkeypatch.setattr(cfg, "cloud_router", None, raising=False)
        monkeypatch.setattr(cfg, "gc_control", False, raising=False)
        yield

    def test_synth_fires_when_parser_saw_nothing(self):
        """``required`` + 1 tool + zero parser-detected call shapes →
        terminal chunk carries a synthesised ``delta.tool_calls`` with
        ``finish_reason="tool_calls"``."""
        # The model emits a hybrid wire shape (Nemotron closers after a
        # JSON prefix injection) that neither shape regex matches; the
        # parser never sets ``tool_calls_detected``.
        deltas = [
            "0",
            "}",
            "\n",
            "</parameter>",
            "\n",
            "</function>",
            "\n",
            "</tool_call>",
        ]
        chunks, finish = _drive_stream(_FakeEngine(deltas), self._request("required"))
        # Look for a tool_calls delta in any chunk.
        emitted_tcs = []
        for c in chunks:
            for ch in c.get("choices") or []:
                tcs = (ch.get("delta") or {}).get("tool_calls")
                if tcs:
                    emitted_tcs.extend(tcs)
        assert emitted_tcs, "synth must emit a delta.tool_calls"
        assert emitted_tcs[0]["function"]["name"] == "get_weather"
        assert finish == "tool_calls"

    def test_synth_does_not_fire_when_parser_saw_a_call(self):
        """Codex r1 MAJOR #1: a parser that detected a tool call
        (``tool_calls_detected=True``) but had every entry dropped by
        the forced-name filter / parallel cap (``_tool_calls_emitted_to
        _wire == 0``) must NOT trigger the synth — that case mirrors
        the non-stream ``_mismatched`` 422 path, not the
        ``not _names`` synth path. Synth here would silently replace
        the model's wrong-tool call with the pinned target."""
        from vllm_mlx.service import postprocessor as pp_mod

        # Simulate the filter-dropped state by wrapping ``reset()`` (the
        # route calls ``processor.reset()`` before the stream loop, so an
        # ``__init__`` patch would be wiped). After ``reset`` only
        # ``process_chunk`` runs on plain text (which won't touch the
        # flag here because no tool envelope appears in the wire).
        original_reset = pp_mod.StreamingPostProcessor.reset

        def _patched_reset(self):
            original_reset(self)
            # Pre-set the filter-dropped state: parser detected a call
            # but every entry was filtered out before reaching the wire.
            self.tool_calls_detected = True

        pp_mod.StreamingPostProcessor.reset = _patched_reset
        try:
            deltas = ["plain ", "answer", "."]
            chunks, finish = _drive_stream(
                _FakeEngine(deltas),
                self._request(
                    {"type": "function", "function": {"name": "get_weather"}}
                ),
            )
        finally:
            pp_mod.StreamingPostProcessor.reset = original_reset
        emitted_tcs = []
        for c in chunks:
            for ch in c.get("choices") or []:
                tcs = (ch.get("delta") or {}).get("tool_calls")
                if tcs:
                    emitted_tcs.extend(tcs)
        # Synth MUST NOT fire when the parser already saw a (different) call.
        assert not emitted_tcs, (
            "synth must not fire when tool_calls_detected=True; would "
            "silently replace the model's intended call"
        )
        # And the wire envelope must still close cleanly.
        assert finish in ("stop", "tool_calls", None) or finish is None
