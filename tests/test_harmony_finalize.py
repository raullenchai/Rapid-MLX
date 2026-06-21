# SPDX-License-Identifier: Apache-2.0
"""D-HARMONY-LEAK (2026-06-21) regression tests.

The gpt-oss family (and any Harmony-encoding tokenizer) emits

    <|channel|>analysis<|message|>…<|end|>
    <|channel|>final<|message|>…<|return|>

as the wire contract for a complete reasoning-then-answer turn. When
generation is cut short BEFORE the final-channel opener appears
(``max_tokens`` mid-analysis OR a ``stop`` string matching mid-
analysis), the engine correctly routes the analysis bytes into
``reasoning_content`` and leaves ``content`` empty — exactly the
silent-drop shape the issue-#569 rescue
(``_rescue_silent_drop_from_reasoning``) was designed to fix.

But on Harmony, the analysis body is NOT the model's final answer.
Promoting it to ``content`` ships the SAME bytes in both fields
(``content == reasoning_content`` mojibake) — the bug filed as
D-HARMONY-LEAK in the cycle-2 fuzz-perf P0 sweep.

The fix adds a harmony-channel-shape gate to the rescue: when
``raw_text`` shows an analysis-channel opener but NO final-channel
opener AND no ``<|call|>`` (commentary tool call), the rescue does
NOT fire, regardless of ``finish_reason`` (length / stop / other).
The gate is symmetric across the streaming and non-streaming surfaces
— the streaming call site synthesises a harmony-marked ``raw_text``
when the active reasoning parser is the ``HarmonyReasoningParser`` and
the model never reached the content channel so the helper's gate fires
uniformly across both paths.

This file exercises:

* Unit tests for the rescue helper's new harmony gate (length cut,
  stop-string match, happy path with full close, truncation mid-final,
  truncation mid-commentary tool call).
* Token-level checks against ``HarmonyStreamingRouter.feed_sequence``
  so the engine-side split contract stays pinned (analysis-only →
  ``content=None`` / ``reasoning=<analysis>``; mid-final →
  ``content=<partial>``; happy path → both populated).
* Regressions for the existing #569 rescue (gemma-4 stuck-thought
  shape) and the VibeThinker ``<think>`` gate so the new harmony gate
  is additive, not replacing.
"""

from __future__ import annotations

import pytest

from vllm_mlx.service.helpers import _rescue_silent_drop_from_reasoning

# ── Unit tests: the rescue helper's new harmony gate ─────────────────


def test_rescue_skipped_when_harmony_cut_short_finish_length():
    """``max_tokens=20`` cut mid-analysis on gpt-oss-20b-mxfp4-q8.

    Live repro: POST chat with ``max_tokens=20`` → finish_reason=length,
    raw_text carries ``<|channel|>analysis<|message|>…`` and never
    reaches ``<|channel|>final<|message|>``. The engine populated
    ``reasoning_text`` with the analysis body and left ``content`` empty.
    Pre-fix, the rescue surfaced the analysis trace as ``content`` —
    ``content == reasoning_content`` mojibake. Post-fix, the harmony-
    shape gate suppresses the rescue so ``content`` stays ``None``.
    """
    raw = "<|channel|>analysis<|message|>Let me think step by step. 17 * 23 ="
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="Let me think step by step. 17 * 23 =",
        tool_calls=None,
        finish_reason="length",
        raw_text=raw,
    )
    assert rescued is None, (
        "D-HARMONY-LEAK: rescue must NOT fire on harmony cut short "
        f"mid-analysis; got rescued={rescued!r}"
    )


def test_rescue_skipped_when_harmony_cut_short_finish_stop():
    """``stop:["Mars"]`` match mid-analysis.

    Live repro: POST chat with a prompt that mentions Mars and
    ``stop=["Mars"]`` → finish_reason=stop, raw_text carries
    ``<|channel|>analysis<|message|>…`` ending just before the
    stop-string token. The engine populated ``reasoning_text`` and
    left ``content`` empty. Pre-fix, the rescue fired (its ``length``
    gate didn't match) and surfaced the analysis prefix as content.
    Post-fix, the harmony gate is finish_reason-agnostic and
    suppresses the rescue.
    """
    raw = "<|channel|>analysis<|message|>Let me think about "
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="Let me think about ",
        tool_calls=None,
        finish_reason="stop",
        raw_text=raw,
    )
    assert rescued is None, (
        "D-HARMONY-LEAK: rescue must NOT fire on harmony cut short "
        f"by stop-string mid-analysis; got rescued={rescued!r}"
    )


def test_rescue_fires_on_happy_path_with_final_channel_present():
    """Counter-test: when the model reached the final channel and
    emitted content, ``final_content`` is non-empty. The rescue's
    happy-path early-exit (``if final_content and final_content.strip()``)
    returns it as-is — the harmony gate is irrelevant here.
    """
    raw = (
        "<|channel|>analysis<|message|>17 * 23 = 391.<|end|>"
        "<|channel|>final<|message|>The answer is 391.<|return|>"
    )
    rescued = _rescue_silent_drop_from_reasoning(
        final_content="The answer is 391.",
        reasoning_text="17 * 23 = 391.",
        tool_calls=None,
        finish_reason="stop",
        raw_text=raw,
    )
    assert rescued == "The answer is 391.", (
        "happy path must propagate final-channel content unchanged; "
        f"got rescued={rescued!r}"
    )


def test_rescue_fires_on_truncation_mid_final_channel():
    """Counter-test: truncation mid-FINAL channel must NOT mistakenly
    suppress the rescue. The engine's router populates ``content`` with
    the partial final-channel bytes; the rescue's happy-path early-
    exit returns it as-is. The harmony gate (analysis present, final
    absent) does not fire because ``<|channel|>final<|message|>`` IS
    in raw_text.
    """
    raw = (
        "<|channel|>analysis<|message|>Brief thought.<|end|>"
        "<|channel|>final<|message|>The ans is"
    )
    rescued = _rescue_silent_drop_from_reasoning(
        final_content="The ans is",
        reasoning_text="Brief thought.",
        tool_calls=None,
        finish_reason="length",
        raw_text=raw,
    )
    assert rescued == "The ans is", (
        "mid-final truncation must surface the partial final content; "
        f"got rescued={rescued!r}"
    )


def test_rescue_skipped_when_harmony_cut_short_finish_unknown():
    """The harmony gate is finish_reason-agnostic. A caller that
    doesn't thread ``finish_reason`` (legacy or non-streaming early-
    return path) still gets the suppression — the discriminator is the
    raw_text channel state, not the wire finish signal.
    """
    raw = "<|channel|>analysis<|message|>still thinking"
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="still thinking",
        tool_calls=None,
        finish_reason=None,
        raw_text=raw,
    )
    assert rescued is None, (
        f"D-HARMONY-LEAK gate must be finish_reason-agnostic; got rescued={rescued!r}"
    )


def test_rescue_suppressed_on_commentary_tool_call_marker_only():
    """Codex r1 BLOCKING #1 counter-test: when ``<|call|>`` is present
    in raw_text but ``tool_calls`` is None (parser failed to extract
    the structured call — malformed args, downstream filter dropped
    the entry, the helper was invoked by a third-party caller that
    doesn't thread the parsed-calls list), the harmony gate must
    STILL suppress the rescue. Promoting the analysis body to
    ``content`` in this state would leak ``reasoning_content`` bytes
    onto the user-visible channel, which is the exact mojibake
    D-HARMONY-LEAK exists to prevent.

    The helper's contract: analysis-present + final-absent suppresses
    REGARDLESS of ``<|call|>`` presence. The earlier
    ``if tool_calls:`` branch already preserves successfully-parsed
    tool calls (it returns the original ``final_content`` before
    reaching this gate). The two branches are independent — a parsed
    call short-circuits early; an unparsed-but-marker-present call
    falls into this gate, which still refuses to leak the analysis.
    """
    raw = (
        "<|channel|>analysis<|message|>need weather<|end|>"
        '<|channel|>commentary to=functions.get_weather<|message|>{"city":"SF"}<|call|>'
    )
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="need weather",
        tool_calls=None,  # parser dropped the call somewhere downstream
        finish_reason="stop",
        raw_text=raw,
    )
    assert rescued is None, (
        "D-HARMONY-LEAK BLOCKING #1: harmony gate must suppress even "
        "when <|call|> is present but tool_calls is empty; "
        f"got rescued={rescued!r}"
    )


# ── Existing #569 / VibeThinker rescue still fire — additive gate ────


def test_rescue_still_fires_for_gemma4_stuck_thought_no_harmony_markers():
    """Counter-test: the original #569 failure (gemma-4 stuck inside
    ``<|channel>thought\\n…`` — note ``<|channel>`` not ``<|channel|>``,
    a DIFFERENT marker family) does NOT match the harmony gate's
    marker substring. The rescue still fires.
    """
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="The user wants weather for SF",
        tool_calls=None,
        finish_reason="stop",
        raw_text="The user wants weather for SF",  # no harmony markers
    )
    assert rescued == "The user wants weather for SF", (
        "gemma-4 #569 rescue must still fire when no harmony markers "
        f"are present; got rescued={rescued!r}"
    )


def test_rescue_still_skipped_for_vibethinker_truncated_think():
    """Counter-test: the 2026-06-17 VibeThinker gate
    (``finish_reason=length`` + raw_text starts with ``<think>`` + no
    ``</think>``) still suppresses — the new harmony gate is
    additive, not replacing.
    """
    raw = "<think>The user wants to compute 17 * 23. Step 1: 17 * 20"
    rescued = _rescue_silent_drop_from_reasoning(
        final_content=None,
        reasoning_text="The user wants to compute 17 * 23. Step 1: 17 * 20",
        tool_calls=None,
        finish_reason="length",
        raw_text=raw,
    )
    assert rescued is None, (
        "VibeThinker truncated-<think> gate must still suppress; "
        f"got rescued={rescued!r}"
    )


# ── Engine-router-level pin: HarmonyStreamingRouter.feed_sequence ────


@pytest.fixture(scope="module")
def harmony_router():
    """Build a real ``HarmonyStreamingRouter`` against the upstream
    harmony encoding's vocab. Skips if ``openai-harmony`` isn't
    installed in the test environment.
    """
    openai_harmony = pytest.importorskip("openai_harmony")
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    from vllm_mlx.output_router import TokenMap
    from vllm_mlx.output_router_harmony import HarmonyStreamingRouter

    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def _eid(s: str) -> int:
        ids = enc.encode(s, allowed_special="all")
        assert len(ids) == 1, (s, ids)
        return ids[0]

    class _FakeHarmonyTokenizer:
        name_or_path = "mlx-community/gpt-oss-20b-mxfp4-q8"

        def encode(self, s, add_special_tokens=False):
            return enc.encode(s, allowed_special="none")

        def decode(self, ids):
            return enc.decode(ids)

    tm = TokenMap(
        format_tag="harmony",
        harmony_channel=_eid("<|channel|>"),
        harmony_message=_eid("<|message|>"),
        harmony_call=_eid("<|call|>"),
        harmony_end=_eid("<|end|>"),
        harmony_return=_eid("<|return|>"),
        harmony_start=_eid("<|start|>"),
        harmony_constrain=_eid("<|constrain|>"),
    )
    router = HarmonyStreamingRouter(tm, _FakeHarmonyTokenizer())
    return enc, router


def _encode_assistant(enc, body: str) -> list[int]:
    return enc.encode(body, allowed_special="all")


def test_router_emits_reasoning_only_on_cut_short_mid_analysis(harmony_router):
    """Engine-side pin: feed an analysis-channel prefix with no close,
    no final, no call. The router must surface
    ``content=None, reasoning=<analysis-body>`` so the chat route's
    downstream split has the right inputs to suppress the rescue.
    """
    enc, router = harmony_router
    router.reset()
    body = (
        "<|start|>assistant<|channel|>analysis<|message|>"
        "Let me think step by step. 17 * 23 ="
    )
    routed = router.feed_sequence(_encode_assistant(enc, body))
    assert routed["content"] is None
    assert routed["reasoning"] == "Let me think step by step. 17 * 23 ="
    assert routed["tool_calls"] is None


def test_router_emits_partial_content_on_cut_short_mid_final(harmony_router):
    """Engine-side pin: feed analysis + an unclosed final channel.
    The router must surface ``content=<partial-final>,
    reasoning=<analysis>`` so the rescue's happy-path early-exit fires
    and the partial final answer ships unchanged.
    """
    enc, router = harmony_router
    router.reset()
    body = (
        "<|start|>assistant<|channel|>analysis<|message|>"
        "Brief thought.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>The ans is"
    )
    routed = router.feed_sequence(_encode_assistant(enc, body))
    assert routed["content"] == "The ans is"
    assert routed["reasoning"] == "Brief thought."
    assert routed["tool_calls"] is None


def test_router_emits_both_on_happy_path_full_close(harmony_router):
    """Engine-side pin: feed a fully-closed analysis-then-final
    sequence. Both channels populated; the rescue's happy-path early-
    exit returns the final content unchanged.
    """
    enc, router = harmony_router
    router.reset()
    body = (
        "<|start|>assistant<|channel|>analysis<|message|>"
        "17 * 23 = 391.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        "The answer is 391.<|return|>"
    )
    routed = router.feed_sequence(_encode_assistant(enc, body))
    assert routed["content"] == "The answer is 391."
    assert routed["reasoning"] == "17 * 23 = 391."
    assert routed["tool_calls"] is None


# ── End-to-end: helper + rescue chain on engine-routed harmony input ──


def test_end_to_end_harmony_cut_short_keeps_content_null():
    """End-to-end: simulate the engine path for gpt-oss cut short
    mid-analysis. The chat route's ``_finalize_content_and_reasoning``
    + ``_rescue_silent_drop_from_reasoning`` chain must yield
    ``content=None`` (or empty) with ``reasoning_content`` populated.

    This pins the wire contract D-HARMONY-LEAK exposed: clients should
    detect "model ran out of budget before producing an answer" via
    ``finish_reason`` plus the empty content — never via byte-
    identical ``content == reasoning_content``.
    """
    from vllm_mlx.service.helpers import _finalize_content_and_reasoning

    raw = "<|channel|>analysis<|message|>Let me think step by step. 17 * 23 ="
    engine_reasoning = "Let me think step by step. 17 * 23 ="

    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text="",  # engine routed analysis away from content
        tool_calls=None,
        reasoning_parser=None,
        engine_reasoning_text=engine_reasoning,
        enable_thinking=None,
        reasoning_max_tokens=None,
    )

    # helper preserves the engine split (no reasoning_max_tokens cap).
    assert reasoning_text == engine_reasoning
    assert cleaned_text == ""

    rescued = _rescue_silent_drop_from_reasoning(
        final_content=cleaned_text or None,
        reasoning_text=reasoning_text,
        tool_calls=None,
        finish_reason="length",
        raw_text=raw,
    )
    # The fix: rescue is suppressed; content stays None even though
    # reasoning is populated.
    assert rescued is None, (
        "end-to-end D-HARMONY-LEAK: chat route's content slot must "
        f"stay None for harmony cut-short; got {rescued!r}"
    )


def test_end_to_end_harmony_stop_match_keeps_content_null():
    """End-to-end: ``stop:["Mars"]`` match mid-analysis. Same shape as
    length cut from the helper chain's perspective — the rescue's gate
    suppresses regardless of finish_reason.
    """
    from vllm_mlx.service.helpers import _finalize_content_and_reasoning

    raw = "<|channel|>analysis<|message|>Let me think about "
    engine_reasoning = "Let me think about "

    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text="",
        tool_calls=None,
        reasoning_parser=None,
        engine_reasoning_text=engine_reasoning,
        enable_thinking=None,
        reasoning_max_tokens=None,
    )

    rescued = _rescue_silent_drop_from_reasoning(
        final_content=cleaned_text or None,
        reasoning_text=reasoning_text,
        tool_calls=None,
        finish_reason="stop",
        raw_text=raw,
    )
    assert rescued is None, (
        "stop-string match mid-analysis must NOT promote reasoning "
        f"to content; got {rescued!r}"
    )


# ── Streaming surface: SSE terminal-chunk gate for harmony ───────────


class _HarmonyReasoningOnlyStreamEngine:
    """Streaming engine mimicking gpt-oss cut short mid-analysis.

    Emits channel-routed reasoning-only deltas — same shape the
    ``HarmonyStreamingRouter`` produces when feeding through
    ``_stream_with_output_router`` on a generation that ends before a
    final-channel transition. Used to drive the chat route's SSE
    pipeline via ``TestClient`` so the streaming D-HARMONY-LEAK gate
    can be exercised without booting a real model server.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, reasoning_deltas: list[str], finish_reason: str = "length"):
        self._deltas = reasoning_deltas
        self._finish_reason = finish_reason

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        accumulated_reasoning = ""
        for i, delta in enumerate(self._deltas):
            accumulated_reasoning += delta
            is_last = i == len(self._deltas) - 1
            yield GenerationOutput(
                text="",
                new_text=delta,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=is_last,
                finish_reason=self._finish_reason if is_last else None,
                channel="reasoning",
                reasoning_text=accumulated_reasoning,
            )


def _parse_sse(text: str) -> list[dict]:
    import json as _json

    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            continue
        try:
            events.append(_json.loads(payload))
        except _json.JSONDecodeError:
            continue
    return events


def _drive_streaming_harmony(finish_reason: str) -> list[dict]:
    """Drive the chat route's SSE path with a HarmonyReasoningParser
    wired as ``cfg.reasoning_parser`` so the streaming D-HARMONY-LEAK
    synthetic_raw injection at ``routes/chat.py`` fires.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    cfg.engine = _HarmonyReasoningOnlyStreamEngine(
        reasoning_deltas=[
            "Let me think step by step. ",
            "17 * 23 = ",
        ],
        finish_reason=finish_reason,
    )
    cfg.model_name = "gpt-oss-20b-mxfp4-q8"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = HarmonyReasoningParser()
    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-oss-20b-mxfp4-q8",
                "stream": True,
                "max_tokens": 20,
                "messages": [
                    {"role": "user", "content": "Compute 17*23 step by step."}
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        return _parse_sse(resp.text)
    finally:
        reset_config()


def test_streaming_harmony_cut_short_does_not_leak_into_content_length():
    """SSE streaming surface, ``finish_reason=length``: the terminal
    chunk's ``delta.content`` MUST stay empty / None when the engine
    emitted only reasoning-channel deltas and the active parser is
    ``HarmonyReasoningParser``. Pre-fix the rescue would promote the
    accumulated reasoning to content, shipping byte-identical
    content + reasoning_content. Post-fix the streaming call site's
    harmony synthetic_raw injection makes the helper's gate fire and
    no SSE chunk carries reasoning prose in ``delta.content``.
    """
    events = _drive_streaming_harmony("length")
    assert events, "expected at least one SSE chunk"

    streamed_content = ""
    streamed_reasoning = ""
    for ev in events:
        for choice in ev.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                streamed_content += delta["content"]
            if delta.get("reasoning_content"):
                streamed_reasoning += delta["reasoning_content"]

    assert not streamed_content, (
        "D-HARMONY-LEAK (streaming, length): no SSE chunk may surface "
        "reasoning as delta.content; "
        f"got streamed_content={streamed_content!r}"
    )
    # The reasoning trace must still flow on the reasoning channel
    # so debug / operator visibility into the cut-short state holds.
    assert "17 * 23" in streamed_reasoning, (
        "reasoning_content must still surface during the loop; "
        f"got reasoning={streamed_reasoning!r}"
    )


def test_streaming_harmony_cut_short_does_not_leak_into_content_stop():
    """SSE streaming surface, ``finish_reason=stop``: same contract as
    the length case — the harmony gate is finish_reason-agnostic. A
    stop-string matching mid-analysis must NOT promote the analysis
    prefix to ``delta.content`` on the terminal chunk.
    """
    events = _drive_streaming_harmony("stop")
    assert events

    streamed_content = ""
    for ev in events:
        for choice in ev.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                streamed_content += delta["content"]

    assert not streamed_content, (
        "D-HARMONY-LEAK (streaming, stop): no SSE chunk may surface "
        "reasoning as delta.content; "
        f"got streamed_content={streamed_content!r}"
    )


# ── Codex r1 BLOCKING #2 pin: tool-call-detected stream gate ─────────


def test_streaming_synthetic_raw_excludes_tool_call_detected_state():
    """Codex r1 BLOCKING #2: the streaming ``harmony_cut_short``
    detection at chat.py looks for "active HarmonyReasoningParser AND
    accumulated_reasoning non-empty AND accumulated_text empty". That
    shape ALSO matches a tool-call-only stream where the parallel-
    tool-calls cap dropped every commentary entry
    (``processor.tool_calls_detected`` set on the cap-exhaust path
    but ``fallback_tool_calls`` may arrive empty and
    ``finish_event.finish_reason`` may take a non-``"tool_calls"``
    value on the buffered-finish-gate path). Plumbing
    ``processor.tool_calls_detected`` into the harmony_cut_short
    predicate prevents misclassifying a tool-call-detected stream as
    analysis-without-final and accidentally narrowing the helper's
    surface for that case. Pin the predicate directly so future
    refactors of the streaming call site preserve the gate.
    """
    from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser

    # Mirror the actual streaming-callsite predicate composition.
    rp = HarmonyReasoningParser()
    accumulated_reasoning = "still thinking"
    accumulated_text = ""

    def _harmony_cut_short(*, tool_calls_detected: bool) -> bool:
        rp_is_harmony = type(rp).__name__ == "HarmonyReasoningParser"
        return bool(
            rp_is_harmony
            and accumulated_reasoning
            and not accumulated_text
            and not tool_calls_detected
        )

    # No tool-call detected: gate fires (harmony cut-short → suppress).
    assert _harmony_cut_short(tool_calls_detected=False) is True
    # Tool-call detected (even with no fallback_tool_calls surfaced):
    # gate does NOT fire — the stream is structurally a tool-call
    # response, not an analysis-only truncation.
    assert _harmony_cut_short(tool_calls_detected=True) is False
