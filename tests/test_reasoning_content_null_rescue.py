# SPDX-License-Identifier: Apache-2.0
"""H-01 regression tests: reasoning-cutoff sentinel notice.

Background
----------
Every OpenAI SDK consumer reads ``choices[0].message.content`` and
renders empty bubbles when:

* A reasoning model (qwen3, deepseek_r1, phi-4-mini-reasoning, glm4,
  gemma4, vibethinker, …) is called with low ``max_tokens`` (256 / 512
  is a common SDK default)
* ``</think>`` is never reached → ``content=null``, every byte goes to
  ``reasoning_content``
* ``finish_reason="length"``

The just-merged D-STOP-THINK (PR #799) and D-HARMONY-LEAK (PR #794)
reinforced "cut-short routes to ``reasoning_content`` only" as the
parser-wide rule. H-01 is the complementary UX gap: the strict routing
is correct for stop-string mid-think (where re-asking can drive past
the stop string), but on ``finish_reason="length"`` with no closed
``</think>`` AND no content streamed, SDK consumers see broken empty
messages with no signal beyond ``finish_reason``.

Policy A (parser-independent, route-boundary): when

* ``finish_reason="length"`` AND
* ``content`` is empty/None AND
* ``reasoning_content`` is non-empty AND
* no tool calls were extracted

surface a clearly-marked literal sentinel string in ``content`` so SDK
consumers see something. The sentinel is parser-independent text —
NEVER the parser-internal reasoning trace — so it cannot re-introduce
the leak D-STOP-THINK / D-HARMONY-LEAK closed.

Opt-out via ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled`` for callers
that already handle ``content is None`` natively.
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

    The helper owns every predicate (env var, finish_reason, content
    emptiness, reasoning presence, tool-call gate). These tests pin the
    truth table so route call sites can stay trivial and any future
    drift between surfaces fails here first.
    """

    def test_fires_on_length_cut_mid_think(self):
        """Exact H-01 production failure: ``finish_reason="length"``
        + empty ``content`` + non-empty reasoning + no tool calls.
        The sentinel surfaces in ``content``; reasoning stays as-is.
        """
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="Let me think about 17*23... 17*20=340, 17*3=",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == REASONING_CUTOFF_SENTINEL

    def test_fires_when_content_is_empty_string(self):
        """Empty-string ``content`` (downstream sanitization collapsed
        the buffer to ``""``) is treated the same as ``None`` — clients
        render an empty bubble either way."""
        result = _apply_reasoning_cutoff_notice(
            final_content="",
            reasoning_text="<incomplete thought>",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == REASONING_CUTOFF_SENTINEL

    def test_fires_when_content_is_whitespace_only(self):
        """Whitespace-only ``content`` looks identical to clients —
        they see an empty bubble. Match the same semantics as the
        silent-drop helper's whitespace-only check (#676 codex r3
        NIT)."""
        result = _apply_reasoning_cutoff_notice(
            final_content="   \n\t",
            reasoning_text="<incomplete thought>",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == REASONING_CUTOFF_SENTINEL

    def test_noop_when_content_is_populated(self):
        """Happy path: model produced a real answer. The sentinel must
        NEVER overwrite legitimate content. Closed
        ``<think>...</think>answer`` flows must come through
        unchanged."""
        result = _apply_reasoning_cutoff_notice(
            final_content="The answer is 391.",
            reasoning_text="17*23 = 17*(20+3) = 340 + 51 = 391",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == "The answer is 391."

    def test_noop_on_stop_finish_d_stop_think_regression_guard(self):
        """D-STOP-THINK regression guard: stop-string cut mid-think
        keeps strict-null behaviour. The sentinel ONLY fires on
        ``finish_reason="length"``. Stop-string is a documented
        contract (the caller asked for ``stop:["X"]`` and the model
        emitted X mid-thought — re-requesting with a different stop
        list is the right recovery path, not a sentinel)."""
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="<incomplete thought before stop string>",
            tool_calls=None,
            finish_reason="stop",
        )
        assert result is None

    def test_noop_on_tool_calls_finish(self):
        """OpenAI spec: tool-call turns ship ``content=None``. Sentinel
        must not interfere."""
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="I should call get_weather",
            tool_calls=[{"id": "x", "type": "function"}],
            finish_reason="tool_calls",
        )
        assert result is None

    def test_noop_when_tool_calls_present_even_on_length(self):
        """Even on ``finish_reason="length"``, a tool-call turn ships
        ``content=None``. The tool-call gate is independent of
        finish_reason — codex r1 BLOCKING on #676 documented the same
        precaution for the silent-drop rescue."""
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="planning the call...",
            tool_calls=[{"id": "x", "type": "function"}],
            finish_reason="length",
        )
        assert result is None

    def test_noop_when_reasoning_is_none(self):
        """If the model produced no reasoning either (truly empty
        response), there's nothing semantically rescue-worthy. We do
        NOT fabricate a "raise max_tokens" hint when the upstream
        bug is "model emitted zero tokens" — that's a different bug
        class."""
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text=None,
            tool_calls=None,
            finish_reason="length",
        )
        assert result is None

    def test_noop_when_reasoning_is_whitespace_only(self):
        """Whitespace-only reasoning — same as ``None`` for rescue
        purposes."""
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="   \n\t",
            tool_calls=None,
            finish_reason="length",
        )
        assert result is None

    @pytest.mark.parametrize(
        "disabled_value",
        [
            "disabled",
            "0",
            "false",
            "False",
            "FALSE",
            "no",
            "off",
            "",
        ],
    )
    def test_env_opt_out_disables_sentinel(self, monkeypatch, disabled_value):
        """``RAPID_MLX_REASONING_CUTOFF_NOTICE`` with any of the
        documented disable values reverts to strict null behaviour
        — for callers (internal agents, advanced clients) that
        already detect ``content is None`` natively and don't want
        the literal sentinel polluting their downstream parsing."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", disabled_value)
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="<truncated thought>",
            tool_calls=None,
            finish_reason="length",
        )
        assert result is None, f"env value {disabled_value!r} must disable the sentinel"

    @pytest.mark.parametrize(
        "enabled_value",
        [
            "enabled",
            "1",
            "true",
            "on",
            "yes",
            # Any other arbitrary string also stays enabled — the disable
            # set is closed and unknown values fall through to "default on".
            "anything",
        ],
    )
    def test_env_other_values_keep_sentinel_enabled(self, monkeypatch, enabled_value):
        """Anything outside the documented disable set leaves the
        sentinel enabled. This includes truthy strings AND unknown
        values — the disable set is closed, the default is on."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", enabled_value)
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="<truncated thought>",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == REASONING_CUTOFF_SENTINEL

    def test_default_is_enabled_when_env_unset(self, monkeypatch):
        """Unset env var → sentinel is enabled. Matches the H-01 fix
        intent: the broken empty-bubble UX is the default thing every
        rapid-mlx install used to suffer; the sentinel is the new
        default and only opt-out reverts."""
        monkeypatch.delenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", raising=False)
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="<truncated thought>",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == REASONING_CUTOFF_SENTINEL


# ──────────────────────────────────────────────────────────────────────
# Parser-wide assembly tests: drive _finalize_content_and_reasoning +
# _rescue_silent_drop_from_reasoning + _apply_reasoning_cutoff_notice
# in the same orchestration the chat route runs, against every
# ``<think>``-style parser family. Pins the parser-INDEPENDENT contract
# H-01 was filed against — the rescue lives at the route boundary, so
# it must produce the same sentinel for qwen3 / deepseek_r1 / glm4 /
# gemma4 / vibethinker / phi-4-mini-reasoning.
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
    3. ``_apply_reasoning_cutoff_notice`` — H-01 sentinel

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
        cleaned_text=raw_text,  # route hands the raw output through
        # clean_output_text earlier; mirror that
        # at the call site.
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
    raw-text shapes that produce the four scenarios H-01 cares about:
    open-only (mid-think), closed+answer, closed+truncated-answer, and
    the happy-path closed pair.

    Gemma4 (channel tokens) and Harmony (analysis/final channels) are
    structurally different — their parser path doesn't produce the
    ``(reasoning, None)`` silent-drop shape on unclosed input (Gemma4's
    parser routes orphan ``<|channel>thought\\n…`` to ``content``;
    harmony's analysis channel is engine-routed before reaching the
    parser). Those families exercise the H-01 rescue via the
    engine-routed branch in ``_finalize_content_and_reasoning`` and
    have their own dedicated tests below (``TestGemma4EngineRouted``,
    ``TestHarmonyEngineRouted``).

    Each entry: ``(name, parser_class, raw_open_only, raw_closed_trunc,
    raw_stop_cut_mid, raw_happy)``."""
    from vllm_mlx.reasoning.deepseek_r1_parser import (
        DeepSeekR1ReasoningParser,
        VibeThinkerReasoningParser,
    )
    from vllm_mlx.reasoning.glm4_parser import Glm4ReasoningParser
    from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    # ``<think>`` family — same input shapes for all four parsers
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
    """Each test using this fixture runs across every reasoning-parser
    family with the parser-specific raw-text shapes that produce the
    four H-01 scenarios. Confirms H-01 is fixed parser-wide at the
    route boundary."""
    name, cls, raw_open_only, raw_closed_trunc, raw_stop_mid, raw_happy = request.param
    return {
        "name": name,
        "parser": cls(),
        "raw_open_only": raw_open_only,
        "raw_closed_trunc": raw_closed_trunc,
        "raw_stop_mid": raw_stop_mid,
        "raw_happy": raw_happy,
    }


class TestParserWideLengthCutMidThink:
    """The core H-01 contract: length-cut mid-think MUST produce the
    sentinel for every reasoning parser family. The route boundary is
    where the fix lives, so the assertion is on the assembled
    ``(content, reasoning)`` pair — not on any individual parser's
    extract method."""

    def test_length_cut_mid_think_produces_sentinel(self, parser_case):
        """Length-cut with an unclosed reasoning block: sentinel fires,
        reasoning is preserved (parser-wide — same contract whether
        the protocol is ``<think>`` or ``<|channel>thought\\n…``)."""
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_open_only"],
            reasoning_parser=parser_case["parser"],
            finish_reason="length",
        )
        assert content == REASONING_CUTOFF_SENTINEL, (
            f"H-01 [{parser_case['name']}]: length-cut mid-think must "
            f"surface sentinel; got content={content!r}"
        )
        # ``reasoning_content`` keeps the trace; clients that DO read
        # it see the same bytes they saw before the fix.
        assert reasoning is not None and "17" in reasoning, (
            f"reasoning_content must remain populated for "
            f"{parser_case['name']}; got {reasoning!r}"
        )

    def test_length_cut_after_close_no_sentinel(self, parser_case):
        """Happy path: the reasoning block CLOSED before the truncation
        point. The sentinel must NOT fire — ``content`` carries the
        partial answer, and overwriting it with the sentinel would
        destroy real bytes the model produced."""
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_closed_trunc"],
            reasoning_parser=parser_case["parser"],
            finish_reason="length",
        )
        # ``content`` carries the partial answer — preserved exactly.
        assert content == "The answer is 39", (
            f"length-cut AFTER reasoning-close [{parser_case['name']}]: "
            f"must preserve partial answer; got content={content!r}"
        )
        # Sanity: reasoning is still populated.
        assert reasoning is not None and "391" in reasoning

    def test_stop_cut_mid_think_no_sentinel(self, parser_case):
        """H-01 contract: stop-cut mid-think NEVER carries the
        sentinel — the sentinel ONLY fires on
        ``finish_reason="length"``. This is the regression guard
        that the H-01 fix doesn't widen the rescue to stop-string
        cuts (D-STOP-THINK's job to police what the silent-drop
        rescue does on ``stop``; H-01's job is only the length-cut
        UX). Whatever the silent-drop rescue settled on for
        stop-cut, the sentinel string is NOT present.
        """
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_stop_mid"],
            reasoning_parser=parser_case["parser"],
            finish_reason="stop",
        )
        # Strict contract: H-01 sentinel is NEVER produced on stop-cut.
        # Whatever the downstream pipeline chose (None, the trace, an
        # empty string) — it MUST NOT be the sentinel literal that
        # H-01 introduced.
        assert content != REASONING_CUTOFF_SENTINEL, (
            f"[{parser_case['name']}]: H-01 sentinel must not fire on "
            f"finish_reason=stop; got content={content!r}"
        )
        # Reasoning still populated.
        assert reasoning is not None and reasoning.strip()

    def test_happy_path_full_generation_no_sentinel(self, parser_case):
        """Full closed reasoning + answer flow: content has the answer,
        reasoning has the trace. Sentinel must not fire."""
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


# ──────────────────────────────────────────────────────────────────────
# Gemma4 + Harmony: engine-routed reasoning shape
#
# These parsers don't share the ``<think>``-tag protocol. Their
# silent-drop shape arrives via the engine's token-level OutputRouter,
# which sets ``GenerationOutput.reasoning_text`` directly. The
# downstream ``clean_output_text`` strips channel markers and that
# determines whether content surfaces non-empty — those parser-
# specific details belong to the gemma4/harmony parser tests, not
# H-01. The contract H-01 owns is "when content lands empty AND
# reasoning is populated AND finish=length, emit sentinel" — pinned by
# the unit tests at the top of this file. The two tests below confirm
# the H-01 helper composes cleanly with the engine-routed branch of
# ``_finalize_content_and_reasoning`` when ``cleaned_text`` arrives
# empty (the post-clean shape that triggers the silent-drop UX).
# ──────────────────────────────────────────────────────────────────────


class TestGemma4EngineRouted:
    """H-01 contract on the engine-routed (token-level OutputRouter)
    branch — the production failure shape for gemma-4 family models.

    Gemma4's engine path strips channel markers before they ever land
    in the route's ``raw_text``; the route then sees ``cleaned_text``
    empty (or whitespace-only) and the rescue helper above already
    decided to leave content empty. H-01 surfaces the literal sentinel
    in that envelope.
    """

    def test_length_cut_with_empty_cleaned_text_surfaces_sentinel(self):
        """When the engine produced reasoning tokens but the route's
        ``cleaned_text`` is empty (engine consumed every channel
        marker, no final-channel emitted), H-01 must fire.

        Drives the helper directly because gemma4's ``clean_output_text``
        regex stripping pre-empts the empty-content shape — the
        engine-route path that produces it isn't easily reproducible at
        the unit level without spinning up the full pipeline. The
        contract being pinned: regardless of parser family, when the
        upstream pipeline lands at (empty content, populated reasoning,
        length finish, no tool calls), the sentinel fires.
        """
        content = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text=("The user wants to know about the weather. Let me think"),
            tool_calls=None,
            finish_reason="length",
        )
        assert content == REASONING_CUTOFF_SENTINEL


class TestHarmonyEngineRouted:
    """H-01 contract on the harmony analysis-without-final shape pinned
    by D-HARMONY-LEAK. Engine-routed analysis bytes hit
    ``reasoning_text`` and the silent-drop rescue suppresses promoting
    them to content — H-01 instead surfaces the literal sentinel.
    """

    def test_analysis_only_with_empty_cleaned_text_surfaces_sentinel(self):
        """Drives the helper directly because the full assembly path
        depends on ``clean_output_text``'s marker-stripping behaviour
        which is harmony-specific. The contract H-01 owns is helper-
        level: empty content + populated reasoning + length finish +
        no tool calls = sentinel. The route-boundary wiring tests
        below confirm this helper IS called from every relevant route
        module.
        """
        content = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text=(
                "The user wants weather. Let me think about which tool to call"
            ),
            tool_calls=None,
            finish_reason="length",
        )
        assert content == REASONING_CUTOFF_SENTINEL


# ──────────────────────────────────────────────────────────────────────
# Env opt-out integration test against the route-assembly pipeline:
# ensures the env-var gate composes correctly with the rescue helper —
# the env var unwinds H-01 behaviour cleanly without disturbing the
# pre-H-01 strict-null contract.
# ──────────────────────────────────────────────────────────────────────


def test_env_opt_out_reverts_to_pre_h01_strict_null(monkeypatch):
    """Setting ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled`` must
    restore the exact pre-H-01 envelope shape: ``content=None`` on
    length-cut mid-think. This is the regression contract for callers
    who already detect ``content is None`` and don't want the literal
    sentinel polluting their downstream parsing.
    """
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "disabled")
    from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    content, reasoning = _finalize_route_assembly(
        raw_text="<think>Some incomplete thought",
        reasoning_parser=Qwen3ReasoningParser(),
        finish_reason="length",
    )
    assert content is None, (
        "env opt-out: content must stay None on length-cut mid-think"
    )
    assert reasoning is not None and reasoning.strip(), (
        "reasoning must still be populated"
    )


# ──────────────────────────────────────────────────────────────────────
# Streaming SSE: sentinel appears as ONE final-chunk delta, not per-token
# ──────────────────────────────────────────────────────────────────────


class _StreamEngine:
    """Minimal streaming engine for the H-01 streaming surface tests.

    The engine emits text-mode (``channel=None``) deltas so the route's
    reasoning parser is on the hot path — the test then configures a
    Qwen3 parser via ``cfg.reasoning_parser`` and the deltas start with
    a literal ``<think>`` opener so the parser's ``_saw_any_tag`` flag
    flips and the streaming rescue's truncated-think gate fires. That
    suppression is precisely the case H-01 was filed for: rescue says
    "don't promote the trace into content", and H-01 surfaces the
    sentinel instead.

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
        # Prepend ``<think>`` to the first delta so the route's
        # streaming reasoning parser sees the opener and flips
        # ``_saw_any_tag=True`` — required for the streaming rescue's
        # truncated-think suppression gate to fire (the case H-01
        # rescues with the sentinel).
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
    # Qwen3 parser so streaming postprocessor flips ``_saw_any_tag=True``
    # on the ``<think>`` opener prepended by the engine. The streaming
    # rescue's truncated-think gate then suppresses promoting the trace
    # to content — exactly the case H-01 surfaces the sentinel for.
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


def test_streaming_length_cut_emits_sentinel_in_terminal_chunk():
    """SSE streaming surface: when reasoning streamed but no content
    streamed AND ``finish_reason="length"``, the terminal chunk's
    ``delta.content`` MUST carry the sentinel string (not the
    accumulated reasoning trace — that would re-introduce the leak
    D-STOP-THINK / D-HARMONY-LEAK closed). The per-delta
    ``reasoning_content`` chunks have already gone out during the
    loop."""
    events = _stream_post(
        ["Let me think about ", "the weather query. ", "I should call"],
        finish_reason="length",
    )
    assert events, "expected at least the terminal SSE chunk"

    terminal_events = [
        e
        for e in events
        if any(ch.get("finish_reason") is not None for ch in e.get("choices", []))
    ]
    assert terminal_events, "expected SSE chunk with finish_reason"
    terminal = terminal_events[-1]
    delta = terminal["choices"][0].get("delta", {})

    assert delta.get("content") == REASONING_CUTOFF_SENTINEL, (
        f"H-01 streaming: terminal chunk must carry sentinel; "
        f"got {delta.get('content')!r}"
    )

    # Reasoning content stream is independent — the per-delta
    # reasoning_content chunks are preserved, not collapsed into the
    # sentinel.
    streamed_reasoning = ""
    for ev in events:
        for choice in ev.get("choices", []):
            d = choice.get("delta") or {}
            if d.get("reasoning_content"):
                streamed_reasoning += d["reasoning_content"]
    assert "weather query" in streamed_reasoning, (
        "per-delta reasoning_content chunks must NOT be collapsed; "
        f"got streamed={streamed_reasoning!r}"
    )


def test_streaming_sentinel_is_single_event_not_per_token():
    """The sentinel surfaces as ONE final-chunk event, not per-token.
    Counting ``content`` deltas across the whole stream MUST yield
    exactly one chunk carrying the sentinel — never split. This
    guards against a future regression that would emit the sentinel
    character-by-character as if it were a model token."""
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


def test_streaming_stop_cut_mid_think_no_sentinel_d_stop_think_guard():
    """SSE D-STOP-THINK regression guard: stop-string cut mid-think
    keeps ``delta.content=None`` on every chunk (including the
    terminal). The sentinel ONLY fires on ``finish_reason="length"``."""
    events = _stream_post(
        ["Buffer ", "more ", "thought"],
        finish_reason="stop",
    )

    # No chunk should carry content at all.
    for ev in events:
        for choice in ev.get("choices", []):
            d = choice.get("delta") or {}
            assert not d.get("content"), (
                f"D-STOP-THINK: no content allowed on stop-cut mid-think; "
                f"got delta={d!r}"
            )


def test_streaming_env_opt_out_reverts_to_pre_h01_null(monkeypatch):
    """Env opt-out integration on the streaming surface: when
    ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled``, the terminal
    chunk's ``delta.content`` stays ``None`` on length-cut. Pins
    streaming/non-streaming parity under the env knob."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "disabled")
    events = _stream_post(
        ["Buffer ", "more"],
        finish_reason="length",
    )
    for ev in events:
        for choice in ev.get("choices", []):
            d = choice.get("delta") or {}
            assert not d.get("content"), (
                f"env opt-out: streaming must not emit sentinel; got delta={d!r}"
            )


def test_streaming_happy_path_no_sentinel_when_content_streamed():
    """SSE happy-path guard: when content WAS streamed during the loop
    (normal turn that closed ``</think>`` and produced an answer), the
    sentinel must NOT fire and the assembled content stream MUST equal
    the original output. The pre-existing #569 streaming test pins this
    for ``finish_reason="stop"``; this re-pins it for ``"length"`` so
    the sentinel can't sneak in via the length-finish path when content
    was actually emitted (truncated answer, not truncated thought)."""
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
                    # Force length-finish to specifically guard the
                    # H-01 path against firing on real-content streams.
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
        # No sentinel snuck in.
        assert REASONING_CUTOFF_SENTINEL not in streamed_content
    finally:
        reset_config()


# ──────────────────────────────────────────────────────────────────────
# Anthropic /v1/messages surface — same UX rescue must apply
# ──────────────────────────────────────────────────────────────────────


def test_anthropic_route_helper_call_site_present():
    """Pins that the Anthropic ``/v1/messages`` route calls the H-01
    helper, so the sentinel behaviour applies uniformly to anthropic
    SDK consumers. Imports the route module and checks the helper
    appears in its module-level dependency set — defends against
    accidental removal in a future refactor.
    """
    from vllm_mlx.routes import anthropic as anthropic_route

    assert "_apply_reasoning_cutoff_notice" in dir(anthropic_route), (
        "Anthropic route must wire the H-01 cutoff sentinel helper"
    )


def test_responses_route_helper_call_site_present():
    """Same wiring check for ``/v1/responses``."""
    from vllm_mlx.routes import responses as responses_route

    assert "_apply_reasoning_cutoff_notice" in dir(responses_route), (
        "Responses route must wire the H-01 cutoff sentinel helper"
    )


def test_chat_route_helper_call_site_present():
    """Same wiring check for ``/v1/chat/completions`` — non-stream
    and stream paths both live in this module."""
    from vllm_mlx.routes import chat as chat_route

    assert "_apply_reasoning_cutoff_notice" in dir(chat_route), (
        "Chat route must wire the H-01 cutoff sentinel helper"
    )
