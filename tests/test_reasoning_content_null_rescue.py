# SPDX-License-Identifier: Apache-2.0
"""R-01 (was H-01) regression tests: reasoning-cutoff sentinel notice.

Background
----------
Every reasoning model (qwen3, deepseek_r1, phi-4-mini-reasoning, glm4,
gemma4, vibethinker, …) can be called with a low ``max_tokens`` budget
that cuts generation off BEFORE ``</think>`` (or the harmony
final-channel marker). The parser-wide rule pinned by D-STOP-THINK +
D-HARMONY-LEAK is "route everything to ``reasoning_content`` and leave
``content`` null/empty" — the in-progress thought trace is NOT the final
answer, so promoting it to ``content`` would ship byte-identical bytes
in both fields (the leak shape those PRs explicitly closed).

History
-------
* H-01 (PR #802, 0.8.3) introduced an opt-OUT sentinel that injected the
  literal string ``[truncated — reasoning incomplete; raise max_tokens]``
  into ``content`` by default so SDK consumers saw something instead of
  an empty bubble.
* R-01 (0.8.5 dogfood) flips the policy: synthesizing a placeholder text
  block the model never produced is harmful injection. Every transport
  already carries an unambiguous structured truncation signal
  (``finish_reason="length"`` / ``status="incomplete"`` /
  ``stop_reason="max_tokens"``) plus ``reasoning_content`` / ``thinking``
  populated. The sentinel is preserved as an opt-IN behaviour for callers
  (e.g. chat UIs that only render text blocks) who still want the legacy
  literal-text cue.

Default (R-01) is now OFF. ``RAPID_MLX_REASONING_CUTOFF_NOTICE=1``
re-enables the sentinel. The helper lives in
``vllm_mlx.service.helpers._apply_reasoning_cutoff_notice`` and is the
single source of truth for ``/v1/chat/completions``,
``/v1/responses``, and ``/v1/messages``.
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.service.helpers import (
    REASONING_CUTOFF_SENTINEL,
    _apply_reasoning_cutoff_notice,
)

# ──────────────────────────────────────────────────────────────────────
# Unit tests for the helper itself
# ──────────────────────────────────────────────────────────────────────


class TestApplyReasoningCutoffNotice:
    """Unit-level predicate tests on ``_apply_reasoning_cutoff_notice``.

    The helper owns every predicate (env opt-in, finish_reason, content
    emptiness, reasoning presence, tool-call gate). These tests pin the
    truth table so route call sites can stay trivial and any future
    drift between surfaces fails here first.
    """

    def test_default_is_disabled_when_env_unset(self, monkeypatch):
        """R-01 contract: unset env var → sentinel disabled. Strict-null
        contract is the default — the structured truncation signal on
        every transport (``finish_reason`` / ``status`` / ``stop_reason``)
        is the canonical cue, so the route ships ``content=None`` and
        clients render whatever they render for null content."""
        monkeypatch.delenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", raising=False)
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="<incomplete thought>",
            tool_calls=None,
            finish_reason="length",
        )
        assert result is None, (
            f"R-01: unset env var must keep the helper as a no-op; got {result!r}"
        )

    @pytest.mark.parametrize(
        "enabled_value",
        ["1", "true", "TRUE", "True", "on", "yes", "enabled"],
    )
    def test_env_opt_in_enables_sentinel(self, monkeypatch, enabled_value):
        """Opt-in: any of the documented enable values re-enables the
        legacy literal-text cue. Case-insensitive matching is locked so
        ``"True"`` and ``"TRUE"`` both work — env vars are commonly
        provided in mixed case by shell wrappers."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", enabled_value)
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="Let me think about 17*23... 17*20=340, 17*3=",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == REASONING_CUTOFF_SENTINEL, (
            f"env value {enabled_value!r} must enable the sentinel"
        )

    @pytest.mark.parametrize(
        "non_enable_value",
        # Documented disable spellings + arbitrary unknown strings.
        # R-01 closes the ENABLE set, so anything outside it stays off.
        [
            "0",
            "false",
            "FALSE",
            "no",
            "off",
            "disabled",
            "",
            "anything",
            "garbage",
            "maybe",
        ],
    )
    def test_env_non_enable_values_keep_sentinel_disabled(
        self, monkeypatch, non_enable_value
    ):
        """R-01 closes the enable set: anything outside
        ``{1, true, on, yes, enabled}`` (case-insensitive) leaves the
        sentinel disabled — including the empty string and any
        arbitrary unrecognised value."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", non_enable_value)
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="<truncated thought>",
            tool_calls=None,
            finish_reason="length",
        )
        assert result is None, (
            f"env value {non_enable_value!r} must keep the sentinel disabled"
        )

    # ----- opt-in branch: the legacy H-01 truth table still holds -----
    #
    # All gates below run with the opt-in env var set, so they exercise
    # the sentinel path. The same predicates govern when the sentinel
    # fires; only the default flipped.

    def test_optin_fires_on_length_cut_mid_think(self, monkeypatch):
        """Exact H-01 production failure shape under opt-in:
        ``finish_reason="length"`` + empty ``content`` + non-empty
        reasoning + no tool calls. The sentinel surfaces in ``content``;
        reasoning stays as-is."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="Let me think about 17*23... 17*20=340, 17*3=",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == REASONING_CUTOFF_SENTINEL

    def test_optin_fires_when_content_is_empty_string(self, monkeypatch):
        """Empty-string ``content`` (downstream sanitization collapsed
        the buffer to ``""``) is treated the same as ``None`` under
        opt-in — clients render an empty bubble either way."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content="",
            reasoning_text="<incomplete thought>",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == REASONING_CUTOFF_SENTINEL

    def test_optin_fires_when_content_is_whitespace_only(self, monkeypatch):
        """Whitespace-only ``content`` looks identical to clients under
        opt-in — they see an empty bubble. Match the same semantics as
        the silent-drop helper's whitespace-only check."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content="   \n\t",
            reasoning_text="<incomplete thought>",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == REASONING_CUTOFF_SENTINEL

    def test_optin_noop_when_content_is_populated(self, monkeypatch):
        """Happy path under opt-in: model produced a real answer. The
        sentinel must NEVER overwrite legitimate content. Closed
        ``<think>...</think>answer`` flows must come through unchanged
        even when the env knob is on."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content="The answer is 391.",
            reasoning_text="17*23 = 17*(20+3) = 340 + 51 = 391",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == "The answer is 391."

    def test_optin_noop_on_stop_finish_d_stop_think_regression_guard(self, monkeypatch):
        """D-STOP-THINK regression guard, even under opt-in: stop-string
        cut mid-think keeps strict-null behaviour. The sentinel ONLY
        fires on ``finish_reason="length"``."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="<incomplete thought before stop string>",
            tool_calls=None,
            finish_reason="stop",
        )
        assert result is None

    def test_optin_noop_on_tool_calls_finish(self, monkeypatch):
        """OpenAI spec: tool-call turns ship ``content=None``. Sentinel
        must not interfere even under opt-in."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="I should call get_weather",
            tool_calls=[{"id": "x", "type": "function"}],
            finish_reason="tool_calls",
        )
        assert result is None

    def test_optin_noop_when_tool_calls_present_even_on_length(self, monkeypatch):
        """Even on ``finish_reason="length"``, a tool-call turn ships
        ``content=None``. The tool-call gate is independent of
        finish_reason."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="planning the call...",
            tool_calls=[{"id": "x", "type": "function"}],
            finish_reason="length",
        )
        assert result is None

    def test_optin_noop_when_reasoning_is_none(self, monkeypatch):
        """If the model produced no reasoning either (truly empty
        response), there's nothing semantically rescue-worthy. We do
        NOT fabricate a "raise max_tokens" hint when the upstream
        bug is "model emitted zero tokens" — that's a different bug
        class."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text=None,
            tool_calls=None,
            finish_reason="length",
        )
        assert result is None

    def test_optin_noop_when_reasoning_is_whitespace_only(self, monkeypatch):
        """Whitespace-only reasoning — same as ``None`` for rescue
        purposes."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="   \n\t",
            tool_calls=None,
            finish_reason="length",
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────────
# Parser-wide assembly tests: drive _finalize_content_and_reasoning +
# _rescue_silent_drop_from_reasoning + _apply_reasoning_cutoff_notice
# in the same orchestration the chat route runs, against every
# ``<think>``-style parser family. Pins the parser-INDEPENDENT contract
# the helper guards — under R-01 default-off, the sentinel never fires;
# under opt-in, it fires uniformly across parsers.
# ──────────────────────────────────────────────────────────────────────


def _finalize_route_assembly(
    *,
    raw_text: str,
    reasoning_parser,
    finish_reason: str,
    enable_thinking: bool | None = None,
    engine_reasoning_text: str = "",
):
    """Mirror the chat route's final-content assembly sequence:

    1. ``_finalize_content_and_reasoning`` — extracts reasoning/content
    2. ``_rescue_silent_drop_from_reasoning`` — issue #569 rescue
    3. ``_apply_reasoning_cutoff_notice`` — R-01 sentinel (opt-in)

    Returns ``(final_content, reasoning_text)`` exactly as the route
    layer would set them on the AssistantMessage.
    """
    from vllm_mlx.api.utils import (
        clean_output_text,
        sanitize_output,
        strip_thinking_tags,
    )
    from vllm_mlx.service.helpers import (
        _finalize_content_and_reasoning,
        _rescue_silent_drop_from_reasoning,
    )

    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text=raw_text,
        cleaned_text=raw_text,
        tool_calls=[],
        reasoning_parser=reasoning_parser,
        engine_reasoning_text=engine_reasoning_text,
        enable_thinking=enable_thinking,
    )
    final_content = None
    if cleaned_text:
        final_content = strip_thinking_tags(clean_output_text(cleaned_text))
        final_content = sanitize_output(final_content)
    final_content = _rescue_silent_drop_from_reasoning(
        final_content,
        reasoning_text,
        tool_calls=[],
        finish_reason=finish_reason,
        raw_text=raw_text,
        reasoning_is_case4=False,
    )
    final_content = _apply_reasoning_cutoff_notice(
        final_content,
        reasoning_text,
        [],
        finish_reason,
    )
    return final_content, reasoning_text


def _parser_cases():
    """Every ``<think>``-tag reasoning-parser family alongside shared
    raw-text shapes that produce the four scenarios this helper cares
    about: open-only (mid-think), closed+truncated-answer, stop-cut
    mid-think, and the happy-path closed pair.

    Each entry: ``(name, parser_class, raw_open_only, raw_closed_trunc,
    raw_stop_cut_mid, raw_happy)``."""
    from vllm_mlx.reasoning.deepseek_r1_parser import (
        DeepSeekR1ReasoningParser,
        VibeThinkerReasoningParser,
    )
    from vllm_mlx.reasoning.glm4_parser import Glm4ReasoningParser
    from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    think_open_only = "<think>Let me think about 17*23. 17 * 20 = 340. 17 * 3 ="
    think_closed_trunc = "<think>17 * 23 = 391.</think>The answer is 39"
    think_stop_mid = "<think>The user wants weather. Let me check"
    think_happy = "<think>17*23 = 391</think>The answer is 391."

    return [
        (
            "qwen3",
            Qwen3ReasoningParser,
            think_open_only,
            think_closed_trunc,
            think_stop_mid,
            think_happy,
        ),
        (
            "deepseek_r1",
            DeepSeekR1ReasoningParser,
            think_open_only,
            think_closed_trunc,
            think_stop_mid,
            think_happy,
        ),
        (
            "vibethinker",
            VibeThinkerReasoningParser,
            think_open_only,
            think_closed_trunc,
            think_stop_mid,
            think_happy,
        ),
        (
            "glm4",
            Glm4ReasoningParser,
            think_open_only,
            think_closed_trunc,
            think_stop_mid,
            think_happy,
        ),
    ]


@pytest.fixture(params=_parser_cases(), ids=lambda p: p[0])
def parser_case(request):
    name, cls, raw_open_only, raw_closed_trunc, raw_stop_mid, raw_happy = request.param
    return {
        "name": name,
        "parser": cls(),
        "raw_open_only": raw_open_only,
        "raw_closed_trunc": raw_closed_trunc,
        "raw_stop_mid": raw_stop_mid,
        "raw_happy": raw_happy,
    }


class TestParserWideLengthCutMidThinkDefault:
    """R-01 default-off: length-cut mid-think must produce strict-null
    content (NO sentinel) for every reasoning parser family. The
    structured truncation signal (``finish_reason="length"`` +
    ``reasoning_content``) is the canonical cue.
    """

    def test_length_cut_mid_think_no_sentinel_by_default(self, parser_case):
        """Default-off contract: length-cut with an unclosed reasoning
        block must NOT inject the sentinel into ``content``. Reasoning
        stays populated so clients that read ``reasoning_content`` (or
        the equivalent ``thinking`` block on the Anthropic surface) see
        the trace."""
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_open_only"],
            reasoning_parser=parser_case["parser"],
            finish_reason="length",
        )
        # Strict-null contract: no synthetic text injection.
        assert "truncated" not in (content or "").lower(), (
            f"R-01 [{parser_case['name']}]: default must not inject the "
            f"truncated sentinel; got content={content!r}"
        )
        assert content != REASONING_CUTOFF_SENTINEL, (
            f"R-01 [{parser_case['name']}]: default must not surface the "
            f"sentinel literal; got content={content!r}"
        )
        # ``reasoning_content`` keeps the trace.
        assert reasoning is not None and "17" in reasoning, (
            f"reasoning_content must remain populated for "
            f"{parser_case['name']}; got {reasoning!r}"
        )

    def test_length_cut_after_close_preserves_partial_answer(self, parser_case):
        """Happy(ish) path: the reasoning block CLOSED before the
        truncation point. ``content`` carries the partial answer and
        must NOT be overwritten with anything — sentinel never gets a
        chance to fire (gate: ``content`` is non-empty)."""
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_closed_trunc"],
            reasoning_parser=parser_case["parser"],
            finish_reason="length",
        )
        assert content == "The answer is 39", (
            f"length-cut AFTER reasoning-close [{parser_case['name']}]: "
            f"must preserve partial answer; got content={content!r}"
        )
        assert reasoning is not None and "391" in reasoning

    def test_stop_cut_mid_think_strict_null(self, parser_case):
        """D-STOP-THINK contract still holds: stop-cut mid-think never
        carries the sentinel. Under R-01 default-off this is also
        strict-null, but the sentinel-absence assertion is what
        D-STOP-THINK pins."""
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_stop_mid"],
            reasoning_parser=parser_case["parser"],
            finish_reason="stop",
        )
        assert content != REASONING_CUTOFF_SENTINEL, (
            f"[{parser_case['name']}]: sentinel must not fire on "
            f"finish_reason=stop; got content={content!r}"
        )
        assert reasoning is not None and reasoning.strip()

    def test_happy_path_full_generation_preserves_answer(self, parser_case):
        """Full closed reasoning + answer flow: content has the answer,
        reasoning has the trace. Sentinel must not fire (gate: content
        non-empty)."""
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_happy"],
            reasoning_parser=parser_case["parser"],
            finish_reason="stop",
        )
        assert content == "The answer is 391.", (
            f"happy-path [{parser_case['name']}]: content must equal answer; "
            f"got {content!r}"
        )
        assert reasoning is not None and "391" in reasoning


class TestParserWideLengthCutMidThinkOptIn:
    """Opt-in path: ``RAPID_MLX_REASONING_CUTOFF_NOTICE=1`` restores the
    legacy H-01 sentinel for callers who want it. The same parser-
    independent contract still holds: every reasoning-parser family
    surfaces the sentinel uniformly on length-cut mid-think.
    """

    def test_optin_length_cut_mid_think_produces_sentinel(
        self, parser_case, monkeypatch
    ):
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_open_only"],
            reasoning_parser=parser_case["parser"],
            finish_reason="length",
        )
        assert content == REASONING_CUTOFF_SENTINEL, (
            f"opt-in [{parser_case['name']}]: length-cut mid-think must "
            f"surface sentinel; got content={content!r}"
        )
        assert reasoning is not None and "17" in reasoning


# ──────────────────────────────────────────────────────────────────────
# Gemma4 + Harmony: engine-routed reasoning shape
# ──────────────────────────────────────────────────────────────────────


class TestGemma4HarmonyEngineRouted:
    """Engine-routed reasoning families (gemma4, harmony) reach the
    helper via a different upstream path (the OutputRouter strips
    channel markers before the route's ``cleaned_text`` is computed).
    The helper-level contract is identical though: default-off →
    sentinel never fires; opt-in → sentinel surfaces.
    """

    def test_default_off_no_sentinel_on_empty_cleaned_text(self, monkeypatch):
        monkeypatch.delenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", raising=False)
        content = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text=("The user wants to know about the weather. Let me think"),
            tool_calls=None,
            finish_reason="length",
        )
        assert content is None

    def test_optin_surfaces_sentinel_on_empty_cleaned_text(self, monkeypatch):
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        content = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text=("The user wants to know about the weather. Let me think"),
            tool_calls=None,
            finish_reason="length",
        )
        assert content == REASONING_CUTOFF_SENTINEL

    def test_harmony_analysis_only_default_off(self, monkeypatch):
        monkeypatch.delenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", raising=False)
        content = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text=(
                "The user wants weather. Let me think about which tool to call"
            ),
            tool_calls=None,
            finish_reason="length",
        )
        assert content is None


# ──────────────────────────────────────────────────────────────────────
# Streaming SSE: default-off → no sentinel emitted; opt-in → one
# final-chunk event carrying the literal sentinel.
# ──────────────────────────────────────────────────────────────────────


class _StreamEngine:
    """Minimal streaming engine for the streaming-surface tests.

    Emits text-mode (``channel=None``) deltas so the route's reasoning
    parser is on the hot path — the test then configures a Qwen3 parser
    via ``cfg.reasoning_parser`` and the deltas start with a literal
    ``<think>`` opener so the parser's ``_saw_any_tag`` flag flips and
    the streaming rescue's truncated-think gate fires. That suppression
    is precisely the case the helper was filed for.

    Set ``include_open_tag=False`` to opt out of the ``<think>`` opener
    for tests that want to assert the plain silent-drop rescue path
    (no parser flag flip)."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(
        self,
        reasoning_deltas: list[str],
        finish_reason: str = "length",
        include_open_tag: bool = True,
    ):
        self._deltas = reasoning_deltas
        self._finish_reason = finish_reason
        self._include_open_tag = include_open_tag
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        accumulated = ""
        deltas = list(self._deltas)
        if self._include_open_tag and deltas:
            deltas[0] = "<think>" + deltas[0]
        for i, delta in enumerate(deltas):
            accumulated += delta
            is_last = i == len(deltas) - 1
            yield GenerationOutput(
                text=accumulated,
                new_text=delta,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=is_last,
                finish_reason=self._finish_reason if is_last else None,
                channel=None,
            )


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events


def _stream_post(
    reasoning_deltas,
    finish_reason="length",
    include_open_tag=True,
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    cfg.engine = _StreamEngine(
        reasoning_deltas,
        finish_reason=finish_reason,
        include_open_tag=include_open_tag,
    )
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser = Qwen3ReasoningParser()
    cfg.reasoning_parser_name = "qwen3"

    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": True,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "weather?"}],
            },
        )
        assert resp.status_code == 200, resp.text
        return _parse_sse(resp.text)
    finally:
        reset_config()


def test_streaming_default_off_no_sentinel_in_terminal_chunk(monkeypatch):
    """R-01 default contract on the streaming surface: when reasoning
    streamed but no content streamed AND ``finish_reason="length"``,
    NO sentinel must appear in any ``delta.content`` event. Per-delta
    ``reasoning_content`` chunks still flow during the loop."""
    monkeypatch.delenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", raising=False)
    events = _stream_post(
        ["Let me think about ", "the weather query. ", "I should call"],
        finish_reason="length",
    )
    assert events, "expected at least the terminal SSE chunk"

    for ev in events:
        for choice in ev.get("choices", []):
            d = choice.get("delta") or {}
            content = d.get("content")
            if content:
                assert "truncated" not in content.lower(), (
                    f"R-01 streaming default: no chunk may carry the "
                    f"truncated sentinel; got delta.content={content!r}"
                )
                assert content != REASONING_CUTOFF_SENTINEL

    # Reasoning content stream is independent and still flows.
    streamed_reasoning = ""
    for ev in events:
        for choice in ev.get("choices", []):
            d = choice.get("delta") or {}
            if d.get("reasoning_content"):
                streamed_reasoning += d["reasoning_content"]
    assert "weather query" in streamed_reasoning, (
        "per-delta reasoning_content chunks must still flow on default-off; "
        f"got streamed={streamed_reasoning!r}"
    )


def test_streaming_optin_emits_sentinel_in_terminal_chunk(monkeypatch):
    """Opt-in: ``RAPID_MLX_REASONING_CUTOFF_NOTICE=1`` restores the H-01
    streaming behaviour — the terminal chunk's ``delta.content`` carries
    the sentinel string. Per-delta reasoning_content unchanged."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
    events = _stream_post(
        ["Let me think about ", "the weather query. ", "I should call"],
        finish_reason="length",
    )
    assert events
    terminal_events = [
        e
        for e in events
        if any(ch.get("finish_reason") is not None for ch in e.get("choices", []))
    ]
    assert terminal_events
    terminal = terminal_events[-1]
    delta = terminal["choices"][0].get("delta", {})
    assert delta.get("content") == REASONING_CUTOFF_SENTINEL, (
        f"opt-in streaming: terminal chunk must carry sentinel; "
        f"got {delta.get('content')!r}"
    )


def test_streaming_optin_sentinel_is_single_event_not_per_token(monkeypatch):
    """When opt-in, the sentinel surfaces as ONE final-chunk event,
    not per-token. Counting ``content`` deltas across the whole stream
    MUST yield exactly one chunk carrying the sentinel — never split."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
    events = _stream_post(
        ["Buffer ", "more ", "thought"],
        finish_reason="length",
    )

    sentinel_chunks = []
    for ev in events:
        for choice in ev.get("choices", []):
            d = choice.get("delta") or {}
            if d.get("content"):
                sentinel_chunks.append(d["content"])

    assert sentinel_chunks == [REASONING_CUTOFF_SENTINEL], (
        f"sentinel must surface as exactly one final-chunk content "
        f"delta carrying the literal sentinel string; got {sentinel_chunks!r}"
    )


def test_streaming_stop_cut_mid_think_no_sentinel_d_stop_think_guard(monkeypatch):
    """SSE D-STOP-THINK regression guard: stop-string cut mid-think
    keeps ``delta.content=None`` on every chunk. The sentinel ONLY
    fires on ``finish_reason="length"`` even under opt-in."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
    events = _stream_post(
        ["Buffer ", "more ", "thought"],
        finish_reason="stop",
    )

    for ev in events:
        for choice in ev.get("choices", []):
            d = choice.get("delta") or {}
            assert not d.get("content"), (
                f"D-STOP-THINK: no content allowed on stop-cut mid-think; "
                f"got delta={d!r}"
            )


def test_streaming_happy_path_no_sentinel_when_content_streamed(monkeypatch):
    """Happy-path guard (default-off): when content WAS streamed during
    the loop (normal turn that closed ``</think>`` and produced an
    answer), the assembled content stream MUST equal the original
    output. No sentinel sneaks in via the length-finish path when
    content was actually emitted."""
    monkeypatch.delenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", raising=False)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.routes.chat import router as chat_router

    class _ContentEngine:
        preserve_native_tool_format = False
        is_mllm = False
        supports_guided_generation = False
        tokenizer = None

        def build_prompt(self, messages, tools=None, enable_thinking=None):
            return "PROMPT"

        async def stream_chat(self, messages, **kwargs):
            for i, txt in enumerate(["Hello ", "world."]):
                is_last = i == 1
                yield GenerationOutput(
                    text=txt,
                    new_text=txt,
                    prompt_tokens=4,
                    completion_tokens=i + 1,
                    finished=is_last,
                    finish_reason="length" if is_last else None,
                    channel=None,
                )

    cfg = reset_config()
    cfg.engine = _ContentEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True

    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": True,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200, resp.text
        events = _parse_sse(resp.text)
        streamed_content = ""
        for ev in events:
            for choice in ev.get("choices", []):
                d = choice.get("delta") or {}
                if d.get("content"):
                    streamed_content += d["content"]
        assert streamed_content == "Hello world.", (
            f"happy-path content stream must equal engine output; "
            f"got {streamed_content!r}"
        )
        assert REASONING_CUTOFF_SENTINEL not in streamed_content
    finally:
        reset_config()


# ──────────────────────────────────────────────────────────────────────
# Route-wiring tests: assert the helper is CALLED from every route
# (source-grep guard against accidental deletion of the call site) and
# the default-off contract holds end-to-end on every transport.
# ──────────────────────────────────────────────────────────────────────


def _route_source(module_name: str) -> str:
    import importlib
    import inspect

    mod = importlib.import_module(module_name)
    return inspect.getsource(mod)


def test_anthropic_route_helper_call_site_present():
    """Pins that the Anthropic ``/v1/messages`` route CALLS the helper,
    so the env-knob behaviour applies uniformly to Anthropic SDK
    consumers. Source-level grep guards against a future refactor that
    deletes the call site but leaves the import intact."""
    src = _route_source("vllm_mlx.routes.anthropic")
    assert "_apply_reasoning_cutoff_notice(" in src, (
        "Anthropic route must invoke the cutoff sentinel helper "
        "(not just import it) — single source of truth"
    )


def test_responses_route_helper_call_site_present():
    """Same call-site grep for ``/v1/responses``."""
    src = _route_source("vllm_mlx.routes.responses")
    assert "_apply_reasoning_cutoff_notice(" in src, (
        "Responses route must invoke the cutoff sentinel helper "
        "(not just import it) — single source of truth"
    )


def test_chat_route_helper_call_site_present():
    """Same call-site grep for ``/v1/chat/completions``. The chat
    module hosts BOTH the non-stream and stream paths, so the helper
    must be invoked twice."""
    src = _route_source("vllm_mlx.routes.chat")
    invocation_count = src.count("_apply_reasoning_cutoff_notice(")
    assert invocation_count >= 2, (
        "Chat route must invoke the cutoff sentinel helper from "
        "BOTH the non-stream and stream paths; "
        f"found {invocation_count} call site(s)"
    )


class _EngineLengthCutMidThink:
    """Shared mock engine for the route-wiring behavioral tests.

    Returns a single ``GenerationOutput`` with an unclosed ``<think>``
    opener and ``finish_reason="length"`` — the exact production
    failure shape the helper was filed against. Used across chat /
    responses / anthropic e2e wiring tests so the contract is identical
    on every surface.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None
    chat_template = ""

    def __init__(self):
        self.chat_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    def estimate_prompt_tokens(self, prompt):
        return 4

    async def chat(self, messages, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text="<think>Computing 17*23 step by step",
            new_text="<think>Computing 17*23 step by step",
            prompt_tokens=4,
            completion_tokens=12,
            finished=True,
            finish_reason="length",
            channel=None,
        )

    async def stream_chat(self, messages, **kwargs):
        yield await self.chat(messages, **kwargs)


def _seed_length_cut_engine(cfg):
    """Common cfg shape for the route-wiring behavioral tests:
    qwen3 reasoning parser + length-cut mock engine."""
    from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    cfg.engine = _EngineLengthCutMidThink()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser = Qwen3ReasoningParser()
    cfg.reasoning_parser_name = "qwen3"


def test_chat_route_default_off_no_sentinel_on_length_cut(monkeypatch):
    """R-01 e2e contract for ``/v1/chat/completions`` non-streaming:
    a length-cut mid-think envelope under the default env settings must
    NOT carry the sentinel. The structured truncation signal
    (``finish_reason="length"`` + ``reasoning_content``) is the cue."""
    monkeypatch.delenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", raising=False)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": False,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "compute 17*23"}],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        msg = payload["choices"][0]["message"]
        content = msg.get("content")
        assert content != REASONING_CUTOFF_SENTINEL, (
            f"R-01 e2e: chat non-stream must NOT inject sentinel by "
            f"default; got content={content!r}"
        )
        if content:
            assert "truncated" not in content.lower(), (
                f"R-01 e2e: chat non-stream content must not carry "
                f"'truncated' synthetic text; got {content!r}"
            )
        assert payload["choices"][0]["finish_reason"] == "length"
        assert msg.get("reasoning_content"), (
            "reasoning_content must remain populated as the canonical truncation cue"
        )
    finally:
        reset_config()


def test_chat_route_optin_surfaces_sentinel_on_length_cut(monkeypatch):
    """Opt-in: ``RAPID_MLX_REASONING_CUTOFF_NOTICE=1`` restores the
    legacy H-01 behaviour end-to-end on ``/v1/chat/completions``."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": False,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "compute 17*23"}],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        msg = payload["choices"][0]["message"]
        assert msg.get("content") == REASONING_CUTOFF_SENTINEL, (
            "opt-in: chat non-stream must surface sentinel on length-cut "
            f"mid-think; got content={msg.get('content')!r}"
        )
        assert payload["choices"][0]["finish_reason"] == "length"
    finally:
        reset_config()


def test_anthropic_route_default_off_no_sentinel_on_length_cut(monkeypatch):
    """R-01 e2e contract for ``/v1/messages``: default must NOT inject
    the sentinel into any content block. ``stop_reason="max_tokens"``
    + the ``thinking`` content block are the canonical truncation
    cues."""
    monkeypatch.delenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", raising=False)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.anthropic import router as anthropic_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

    try:
        app = FastAPI()
        app.include_router(anthropic_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "compute 17*23"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        body_str = json.dumps(payload)
        assert REASONING_CUTOFF_SENTINEL not in body_str, (
            f"R-01 e2e: /v1/messages must NOT carry the sentinel by "
            f"default; got payload={payload!r}"
        )
        assert "truncated" not in body_str.lower() or (
            # Allow harmless "truncated" inside non-content fields if
            # any future metadata mentions it — guard the content block
            # specifically.
            all(
                "truncated" not in (block.get("text") or "").lower()
                for block in payload.get("content") or []
                if block.get("type") == "text"
            )
        ), (
            f"R-01 e2e: /v1/messages text content blocks must not carry "
            f"'truncated' synthetic text; got payload={payload!r}"
        )
    finally:
        reset_config()


def test_anthropic_route_optin_surfaces_sentinel(monkeypatch):
    """Opt-in: legacy H-01 behaviour restored on ``/v1/messages``."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.anthropic import router as anthropic_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

    try:
        app = FastAPI()
        app.include_router(anthropic_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "compute 17*23"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        # Compare against the JSON-decoded text block payloads — the raw
        # ``json.dumps`` body escapes the em-dash to ``—`` and would
        # spuriously fail a substring check against the unicode literal.
        text_blocks = [
            (block.get("text") or "")
            for block in (payload.get("content") or [])
            if block.get("type") == "text"
        ]
        assert any(REASONING_CUTOFF_SENTINEL in t for t in text_blocks), (
            f"opt-in e2e: /v1/messages must surface sentinel in a text "
            f"content block on length-cut mid-think; got payload={payload!r}"
        )
    finally:
        reset_config()


def test_responses_route_default_off_no_sentinel_on_length_cut(monkeypatch):
    """R-01 e2e contract for ``/v1/responses``: default must NOT inject
    the sentinel into any output_text block. ``status="incomplete"`` +
    ``usage.output_tokens_details.reasoning_tokens`` are the canonical
    truncation cues."""
    monkeypatch.delenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", raising=False)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

    try:
        app = FastAPI()
        app.include_router(responses_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "max_output_tokens": 16,
                "input": "compute 17*23",
                "stream": False,
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        body_str = json.dumps(payload)
        assert REASONING_CUTOFF_SENTINEL not in body_str, (
            f"R-01 e2e: /v1/responses must NOT carry the sentinel by "
            f"default; got payload={payload!r}"
        )
        # Walk output[].content[] explicitly to catch any synthetic
        # output_text block.
        for item in payload.get("output") or []:
            for block in item.get("content") or []:
                text = block.get("text") or ""
                assert "truncated" not in text.lower(), (
                    f"R-01 e2e: /v1/responses output_text must not "
                    f"carry 'truncated' synthetic text; got block={block!r}"
                )
    finally:
        reset_config()


def test_responses_route_optin_surfaces_sentinel(monkeypatch):
    """Opt-in: legacy H-01 behaviour restored on ``/v1/responses``."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

    try:
        app = FastAPI()
        app.include_router(responses_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "max_output_tokens": 16,
                "input": "compute 17*23",
                "stream": False,
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        # Walk ``output[].content[]`` for a text block carrying the
        # sentinel literal — the JSON-dumped body escapes the em-dash,
        # so a substring check against the unicode literal there is
        # brittle. The decoded text block is the authoritative payload.
        sentinel_texts: list[str] = []
        for item in payload.get("output") or []:
            for block in item.get("content") or []:
                if block.get("type") == "output_text":
                    sentinel_texts.append(block.get("text") or "")
        assert any(REASONING_CUTOFF_SENTINEL in t for t in sentinel_texts), (
            f"opt-in e2e: /v1/responses must surface sentinel in an "
            f"output_text block on length-cut mid-think; "
            f"got payload={payload!r}"
        )
    finally:
        reset_config()
