# SPDX-License-Identifier: Apache-2.0
"""D-STOP-THINK: cross-parser-family regression tests.

Bug shape (cross-cycle bundle, 6 parser families confirmed):
when ``stop`` matches inside an unterminated ``<think>`` block OR
``max_tokens`` cuts mid-thought before ``</think>``, BOTH ``content`` AND
``reasoning_content`` receive the SAME buffered thinking trace on the
non-streaming OpenAI envelope AND on the Anthropic / Responses streaming
envelopes.

Cross-cycle evidence:

* qwen3 reasoning — bug_report.md:128, cycle-3 F-3, cycle-7 F-1 (nemotron-30b)
* hermes — cycle-5 F-1 (qwen3.5-27b-8bit, hermes tool parser layered on qwen3)
* glm4 — cycle-8 F-801 (glm4.7-9b-4bit)
* deepseek_r1 — cycle-11 F-11-7 (phi-4-mini-reasoning)
* gemma4 — cycle-6 F-CORR-2 (gemma-4-26b/12b)
* VibeThinker (DeepSeekR1 subclass) — cycle-2 F-12-1

Root cause (cross-family): each parser's ``finalize_streaming`` path
(and Gemma4's ``extract_reasoning`` channel-grammar variant) emitted
the buffered text into the ``content`` channel when the close marker
was never crossed. The streaming loop had already shipped the same
bytes as ``reasoning_content`` (thinking_delta on Anthropic;
delta.reasoning_content on OpenAI; thinking event on Responses), so
the finalize emission re-sent the trace as a fresh text block / text
delta — duplicating the trace into both channels.

Fix (shared base-class invariant): ``BaseThinkingReasoningParser``
exposes ``_finalize_in_think_block(accumulated_text)`` — True when
``</think>`` is absent — and a default ``finalize_streaming`` that
returns ``None``. Subclasses MUST NOT emit ``content`` when this
invariant holds; the Qwen3 / DeepSeek-R1 finalize correction paths
now surface the rescue text via ``reasoning`` so the Anthropic /
Responses routes' ``final_msg.content`` gate stays silent (no wire
duplication) while parser-contract callers (this test, custom routes)
still see the rescue signal.

These tests exercise the SHARED invariant across every
``BaseThinkingReasoningParser`` subclass + the Gemma 4 channel-grammar
variant, plus a Hermes-with-reasoning composition smoke test. They
also cover the ``max_tokens``-cut variant (same accumulator state as
``stop``-mid-think; the engine truncates the suffix in both cases).
"""

import pytest

from vllm_mlx.reasoning.deepseek_r1_parser import (
    DeepSeekR1ReasoningParser,
    VibeThinkerReasoningParser,
)
from vllm_mlx.reasoning.gemma4_parser import Gemma4ReasoningParser
from vllm_mlx.reasoning.glm4_parser import Glm4ReasoningParser
from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser


def _simulate_anthropic_stream(
    parser,
    chunks,
    *,
    matched_stop=None,
    prompt_thinking_active=False,
    finish_reason=None,
):
    """Replay ``chunks`` through the parser the way the Anthropic /
    Responses streaming routes do, then apply the route's
    ``finalize_streaming`` consumer protocol.

    The Anthropic route emits each streaming delta's ``reasoning``
    as a ``thinking_delta`` and each ``content`` as a ``text_delta``.
    At end-of-stream it calls ``finalize_streaming(accumulated_raw,
    matched_stop=..., prompt_thinking_active=..., finish_reason=...)``
    and emits ``final_msg.content`` as a NEW text block (only acts
    on ``.content``, NOT on ``.reasoning``).

    Codex round-8 BLOCKING (PR #799): the simulator now threads the
    finalize-time signals so tests can exercise the
    natural-EOS-vs-truncation discriminator the same way the routes
    do. Previously every test called ``finalize_streaming(prev)``
    without keywords, which silently exercised only the natural-EOS
    branches; the D-STOP-THINK suppression branches under
    ``matched_stop`` / ``finish_reason="length"`` were never reached
    by these regression tests.

    Returns (thinking_bytes, text_bytes) — the byte streams the
    client would see across the two Anthropic channels.
    """
    parser.reset_state()
    prev = ""
    thinking_bytes = ""
    text_bytes = ""
    for ch in chunks:
        cur = prev + ch
        msg = parser.extract_reasoning_streaming(prev, cur, ch)
        if msg:
            if msg.reasoning:
                thinking_bytes += msg.reasoning
            if msg.content:
                text_bytes += msg.content
        prev = cur
    final = parser.finalize_streaming(
        prev,
        matched_stop=matched_stop,
        prompt_thinking_active=prompt_thinking_active,
        finish_reason=finish_reason,
    )
    # Route consumer: only acts on final_msg.content (anthropic.py:1715,
    # responses.py:907). final_msg.reasoning is silently dropped — which
    # is the desired outcome because the bytes already shipped as
    # reasoning during the stream loop.
    if final and final.content:
        text_bytes += final.content
    return thinking_bytes, text_bytes


THINK_PARSERS_WITH_BASE = [
    ("qwen3", Qwen3ReasoningParser),
    ("deepseek_r1", DeepSeekR1ReasoningParser),
    ("vibethinker", VibeThinkerReasoningParser),
    ("glm4", Glm4ReasoningParser),
]


class TestBaseInvariant:
    """The shared ``_finalize_in_think_block`` invariant pins the rule
    for every ``BaseThinkingReasoningParser`` subclass.
    """

    @pytest.mark.parametrize("name,parser_cls", THINK_PARSERS_WITH_BASE)
    def test_finalize_in_think_block_true_without_closer(self, name, parser_cls):
        parser = parser_cls()
        assert parser._finalize_in_think_block("<think>partial thought")
        assert parser._finalize_in_think_block("Let me think about it")
        # Empty text: NOT considered mid-think (no bytes to leak).
        assert not parser._finalize_in_think_block("")

    @pytest.mark.parametrize("name,parser_cls", THINK_PARSERS_WITH_BASE)
    def test_finalize_in_think_block_false_with_closer(self, name, parser_cls):
        parser = parser_cls()
        assert not parser._finalize_in_think_block("<think>r</think>answer")
        assert not parser._finalize_in_think_block("r</think>answer")


class TestStopMidThinkExplicitOpener:
    """``stop`` matches inside the ``<think>`` block AFTER the explicit
    opener but BEFORE ``</think>`` arrives.

    Engine truncates the model output to the prefix before the stop
    marker; the parser sees ``<think>…<partial thought>``.
    """

    @pytest.mark.parametrize("name,parser_cls", THINK_PARSERS_WITH_BASE)
    def test_no_duplicate_bytes_across_channels(self, name, parser_cls):
        """D-STOP-THINK shape under a real truncation signal: a user
        stop string fires inside the unterminated ``<think>`` block
        (``matched_stop`` set) — finalize MUST route to reasoning to
        suppress duplication.

        Codex round-8 BLOCKING refinement (PR #799): thread the
        ``matched_stop`` truncation signal so the test pins the
        actual D-STOP-THINK suppression branch. Natural EOS without
        the signal now correctly flips to content per the #569
        silent-drop rescue contract (covered separately in
        ``test_natural_eos_with_explicit_opener_flips_to_content``).
        """
        parser = parser_cls()
        chunks = ["<think>", "Let me think ", "about 5+7. The "]
        # ``matched_stop`` simulates a user stop string firing inside
        # the unclosed ``<think>`` block — the live-repro D-STOP-THINK
        # shape (qwen3-0.6b-4bit with ``stop=["STOP"]``).
        thinking, text = _simulate_anthropic_stream(parser, chunks, matched_stop="STOP")
        # Some reasoning must have streamed; what matters is that the
        # text channel does NOT also receive the trace.
        assert "Let me think" in thinking, (
            f"[{name}] streaming did not route reasoning: thinking={thinking!r}"
        )
        assert not text, (
            f"[{name}] D-STOP-THINK regression — bytes duplicated into the "
            f"text channel.\n  thinking={thinking!r}\n  text={text!r}"
        )

    @pytest.mark.parametrize(
        "name,parser_cls",
        [("qwen3", Qwen3ReasoningParser)],
    )
    def test_whitespace_prefix_before_think_opener_recognised(self, name, parser_cls):
        """Codex round-2 BLOCKING fix (PR #799): the qwen3 finalize
        path used ``startswith(self.start_token)`` to detect the
        explicit opener, which missed valid streams with leading
        whitespace before ``<think>`` (e.g. the template-injected
        ``<think>\\n`` with leading newline padding, or a model
        emission like ``  <think>``). The whitespace-padded stream
        fell through to the no-evidence content correction —
        leaking the thought trace into ``content``.

        Post-fix: ``lstrip().startswith`` recognises the opener
        regardless of leading whitespace; under a real truncation
        signal (``matched_stop`` set OR ``finish_reason="length"``)
        the rescue then surfaces via reasoning so the D-STOP-THINK
        invariant holds.

        Codex round-8 BLOCKING refinement (PR #799): pass
        ``matched_stop`` so the suppression path actually fires.
        Without a truncation signal, natural-EOS now correctly flips
        to content (#569 silent-drop rescue).
        """
        parser = parser_cls()
        # Leading whitespace before the opener — the template-
        # injected case AND a real truncation signal (matched_stop)
        # so the D-STOP-THINK suppression fires.
        accumulated = "\n  <think>Let me think about 5+7."
        result = parser.finalize_streaming(accumulated, matched_stop="STOP")
        assert result is not None
        assert result.content is None, (
            f"[{name}] codex r2 BLOCKING regression — whitespace-prefixed "
            f"explicit opener leaked into content under matched_stop: "
            f"{result.content!r}"
        )
        assert result.reasoning is not None
        assert "Let me think" in result.reasoning

    @pytest.mark.parametrize(
        "name,parser_cls",
        [("qwen3", Qwen3ReasoningParser)],
    )
    def test_whitespace_prefix_before_think_opener_preserved(self, name, parser_cls):
        """Codex round-7 BLOCKING fix (PR #799): the previous
        ``stripped_text[len(self.start_token):]`` extraction discarded
        the whitespace prefix verbatim, so a response that legitimately
        opens with `` \\n<think>...`` lost its leading whitespace bytes
        from BOTH channels at finalize time. The bytes were user-visible
        — the streaming loop had already shipped them — so the parser
        contract violated wire-equivalence on the finalize correction.

        Post-fix: the leading whitespace prefix is preserved inside the
        reasoning emission under D-STOP-THINK suppression (matched_stop
        set). The route consumer drops ``final_msg.reasoning`` from the
        wire anyway (only ``content`` flows downstream on Anthropic /
        Responses) so this is purely a parser-contract invariant — but
        it guarantees no byte is silently dropped in the saw-prefix
        branch.

        Codex round-8 BLOCKING refinement (PR #799): thread
        ``matched_stop`` so we exercise the D-STOP-THINK suppression
        path; the prefix-preservation invariant applies to that
        branch.
        """
        parser = parser_cls()
        # Mimic the codex repro: leading newline+space before the opener.
        prefix = " \n"
        body = "Let me think about 5+7."
        accumulated = f"{prefix}<think>{body}"
        result = parser.finalize_streaming(accumulated, matched_stop="STOP")
        assert result is not None
        assert result.content is None, (
            f"[{name}] D-STOP-THINK regression — whitespace-prefixed "
            f"explicit opener leaked into content: {result.content!r}"
        )
        assert result.reasoning is not None
        # The whitespace prefix MUST survive into the emitted reasoning
        # payload — otherwise the parser is silently dropping user-
        # visible bytes (codex r7 BLOCKING).
        assert result.reasoning.startswith(prefix), (
            f"[{name}] codex r7 BLOCKING regression — leading whitespace "
            f"prefix dropped from saw-prefix branch: "
            f"reasoning={result.reasoning!r}"
        )
        assert body in result.reasoning

    @pytest.mark.parametrize(
        "name,parser_cls",
        [("qwen3", Qwen3ReasoningParser)],
    )
    def test_natural_eos_with_explicit_opener_flips_to_content(self, name, parser_cls):
        """Codex round-8 BLOCKING (PR #799): natural-EOS stream with a
        literal ``<think>`` opener but NO ``</think>`` AND NO truncation
        signal means the model voluntarily ended mid-thought. The
        finalize correction MUST flip to ``content`` so the
        Anthropic/Responses route surfaces the trace as the assistant
        turn — otherwise the user sees an empty turn because the route
        consumer drops ``final_msg.reasoning``.

        This is the #569 silent-drop rescue contract: a turn that
        produced only orphaned reasoning must still reach
        ``message.content`` so the wire is never silently empty.
        """
        parser = parser_cls()
        accumulated = "<think>just a thought that the model gave up on"
        # Natural EOS: no matched_stop, no finish_reason="length".
        result = parser.finalize_streaming(
            accumulated, matched_stop=None, finish_reason=None
        )
        assert result is not None, (
            f"[{name}] natural-EOS with unclosed <think>: no rescue emitted "
            f"— assistant turn would be empty"
        )
        assert result.content is not None, (
            f"[{name}] codex r8 BLOCKING regression — natural-EOS with "
            f"unclosed <think> routed to reasoning (route drops it from "
            f"wire); assistant turn would be empty: {result!r}"
        )
        assert "just a thought" in result.content
        assert result.reasoning is None


class TestStopMidThinkNoOpener:
    """``stop`` matches mid-thought WITHOUT an explicit ``<think>``
    opener (Qwen3 / DeepSeek-R1 chat templates pre-inject ``<think>\\n``
    into the prompt, so it never appears in model output).

    Codex round-N BLOCKING scope: the no-opener no-bare-preamble path
    is the "casual non-thinking answer" contract (#570 / #572) — the
    finalize correction MUST flip the buffered reasoning bytes to
    ``content`` so the route's consumer surfaces them as a text block.
    Without this flip, casual answers would appear as an empty
    assistant turn on OpenAI envelopes. The Anthropic-stream
    duplication risk here is the documented trade-off: the route
    consumer ignores ``final_msg.reasoning``, so emitting reasoning
    here would never reach the wire — the only correction surface
    available is ``content``.

    Therefore for the no-opener no-bare-preamble path:
      - Bare-preamble label (``Here's a thinking process:``) → reasoning
        (preserves #570 — the label IS thinking evidence)
      - Casual answer (``Let me think about 5+7``) → content
        (preserves the casual-answer contract; the streaming
        Case-3-to-reasoning routing is the bug, finalize is the fix)
    """

    @pytest.mark.parametrize("name,parser_cls", THINK_PARSERS_WITH_BASE)
    def test_casual_answer_flips_to_content(self, name, parser_cls):
        parser = parser_cls()
        chunks = ["Let me think ", "about 5+7. "]
        thinking, text = _simulate_anthropic_stream(parser, chunks)
        # GLM-4 overrides Case-3 to content (its chat template does NOT
        # inject ``<think>``; see ``Glm4ReasoningParser`` module
        # docstring). For GLM-4, the trace goes straight to text via
        # streaming; finalize doesn't need to correct.
        if name == "glm4":
            assert thinking == "", (
                f"[{name}] glm4 routes no-tag streams to content; "
                f"thinking should be empty: {thinking!r}"
            )
            assert "Let me think" in text, (
                f"[{name}] glm4 should have routed casual answer to "
                f"content via streaming: text={text!r}"
            )
            return
        # All other ``<think>``-family parsers (qwen3 / deepseek_r1 /
        # vibethinker) route the no-tag stream to reasoning via the
        # base class Case-3 default. Finalize then flips the buffered
        # trace to content via the casual-answer correction contract
        # (#572). The Anthropic route emits both — streaming thinking
        # block AND finalize text block — which IS the documented
        # behaviour for the no-evidence path (codex round-N BLOCKING
        # scope on D-STOP-THINK).
        assert "Let me think" in thinking, (
            f"[{name}] expected base-class Case-3 reasoning routing; "
            f"thinking={thinking!r}"
        )
        assert "Let me think" in text, (
            f"[{name}] expected finalize content correction for casual "
            f"answer: text={text!r}"
        )

    @pytest.mark.parametrize(
        "name,parser_cls",
        [
            ("qwen3", Qwen3ReasoningParser),
            # deepseek_r1 / vibethinker / glm4 do NOT implement the
            # bare-preamble label detector — it's Qwen3-specific (#570).
        ],
    )
    def test_bare_preamble_label_routes_to_reasoning(self, name, parser_cls):
        """The bare-preamble scratchpad-label fallback (#570) IS a
        think-mode evidence signal — surface via reasoning so the
        Anthropic stream does NOT duplicate."""
        parser = parser_cls()
        chunks = [
            "Here's a thinking process: ",
            "first I sort the items, then ",
            "I pick the largest. ",
        ]
        thinking, text = _simulate_anthropic_stream(parser, chunks)
        assert "thinking process" in thinking, (
            f"[{name}] bare-preamble must surface as reasoning: thinking={thinking!r}"
        )
        assert not text, (
            f"[{name}] D-STOP-THINK regression — bare-preamble "
            f"duplicated into text: text={text!r}"
        )


class TestPromptInjectedMidThinkDiscriminator:
    """Codex round-4 BLOCKING fix (PR #799): ``matched_stop`` alone
    is NOT enough to identify prompt-injected mid-think — a casual
    answer like ``"The answer is STOP"`` under ``stop=["STOP"]``
    also has matched_stop set but is not chain-of-thought.

    The discriminator at finalize time is the AND of:
    - ``matched_stop`` (engine: a user stop string trimmed the output)
    - ``prompt_thinking_active`` (route: chat template injected
      ``<think>`` AND ``enable_thinking`` is non-False)

    Both signals together identify a prompt-injected mid-think shape
    where the streaming loop routed bytes to reasoning and finalize
    must NOT duplicate them into content.
    """

    @pytest.mark.parametrize(
        "name,parser_cls",
        [
            ("qwen3", Qwen3ReasoningParser),
            ("deepseek_r1", DeepSeekR1ReasoningParser),
            ("vibethinker", VibeThinkerReasoningParser),
        ],
    )
    def test_matched_stop_and_thinking_active_routes_to_reasoning(
        self, name, parser_cls
    ):
        """Prompt-injected mid-think shape: matched_stop set AND
        thinking active → route to reasoning (suppress D-STOP-THINK
        duplication)."""
        parser = parser_cls()
        trace = "5+7 equals 12"
        result = parser.finalize_streaming(
            trace, matched_stop="STOP", prompt_thinking_active=True
        )
        assert result is not None
        assert result.content is None, (
            f"[{name}] D-STOP-THINK regression — prompt-injected "
            f"mid-think duplicated into content: {result.content!r}"
        )
        assert result.reasoning == trace

    @pytest.mark.parametrize(
        "name,parser_cls",
        [
            ("qwen3", Qwen3ReasoningParser),
            ("deepseek_r1", DeepSeekR1ReasoningParser),
            ("vibethinker", VibeThinkerReasoningParser),
        ],
    )
    def test_matched_stop_without_thinking_flips_to_content(self, name, parser_cls):
        """Casual stop-terminated answer: matched_stop set but
        thinking NOT active → flip to content. The codex round-4
        counter-example: ``"The answer is STOP"`` under
        ``stop=["STOP"]`` is a legitimate answer that must reach
        ``message.content``. Routing to reasoning here would silently
        drop the answer."""
        parser = parser_cls()
        trace = "The answer is 12"
        result = parser.finalize_streaming(
            trace, matched_stop="STOP", prompt_thinking_active=False
        )
        assert result is not None
        assert result.content == trace, (
            f"[{name}] codex r4 regression — casual stop-terminated "
            f"answer suppressed: {result!r}"
        )
        assert result.reasoning is None

    @pytest.mark.parametrize(
        "name,parser_cls",
        [
            ("qwen3", Qwen3ReasoningParser),
            ("deepseek_r1", DeepSeekR1ReasoningParser),
            ("vibethinker", VibeThinkerReasoningParser),
        ],
    )
    def test_natural_eos_flips_to_content_regardless_of_thinking(
        self, name, parser_cls
    ):
        """Natural-EOS (matched_stop=None, finish_reason="stop" or
        missing) → flip to content per #570/#572 regardless of thinking
        signal. The D-STOP-THINK shape requires BOTH matched_stop AND
        active thinking — without matched_stop we don't have the
        truncation signal.

        Note this branch is the COMPLEMENT of the max_tokens-mid-think
        case (covered separately in
        ``test_length_finish_with_thinking_routes_to_reasoning``): we
        explicitly model natural EOS by passing
        ``finish_reason=None`` / ``"stop"`` with ``matched_stop=None``
        so the parser cannot mistake this for a budget cut.
        """
        parser = parser_cls()
        trace = "5+7 equals 12"
        # Natural EOS: model finished cleanly. Bytes that streamed
        # as reasoning were the actual answer (parser misclassified
        # them as reasoning under Case-1 / short-no-tag), so flip
        # to content. This holds REGARDLESS of thinking signal —
        # only matched_stop OR finish_reason="length" combined with
        # thinking active trigger the D-STOP-THINK reasoning route.
        for thinking in (False, True):
            for finish in (None, "stop"):
                result = parser.finalize_streaming(
                    trace,
                    matched_stop=None,
                    prompt_thinking_active=thinking,
                    finish_reason=finish,
                )
                assert result is not None, (
                    f"[{name}] thinking={thinking} finish={finish!r}: no rescue emitted"
                )
                assert result.content == trace, (
                    f"[{name}] thinking={thinking} finish={finish!r}: "
                    f"#569 regression — natural-EOS answer "
                    f"suppressed: {result!r}"
                )
                assert result.reasoning is None

    @pytest.mark.parametrize(
        "name,parser_cls",
        [
            ("qwen3", Qwen3ReasoningParser),
            ("deepseek_r1", DeepSeekR1ReasoningParser),
            ("vibethinker", VibeThinkerReasoningParser),
        ],
    )
    def test_length_finish_with_thinking_routes_to_reasoning(self, name, parser_cls):
        """D-STOP-THINK codex round-6 BLOCKING (PR #799):
        ``max_tokens`` cut mid-think with prompt-injected ``<think>``
        is the SAME accumulator shape as stop-mid-think — the parser
        routed every byte to reasoning (Case-1 ``start_in_prev`` /
        short-no-tag) and the streaming loop already shipped them as
        ``reasoning_content``. Finalize MUST route to reasoning to
        suppress the same duplication the matched_stop branch closes.

        Without this, the bare-text-no-evidence flip in the parser's
        no-prefix branch would emit ``DeltaMessage(content=trace)`` and
        the route consumer would append the trace as a NEW text block —
        the exact D-STOP-THINK leak this PR closes for the stop case
        would silently regress for the budget-cut case.
        """
        parser = parser_cls()
        trace = "5+7 equals 12"
        result = parser.finalize_streaming(
            trace,
            matched_stop=None,
            prompt_thinking_active=True,
            finish_reason="length",
        )
        assert result is not None, (
            f"[{name}] max_tokens-mid-think: no correction emitted"
        )
        assert result.content is None, (
            f"[{name}] D-STOP-THINK round-6 regression — "
            f"max_tokens-mid-think duplicated into content: "
            f"{result.content!r}"
        )
        assert result.reasoning == trace

    @pytest.mark.parametrize(
        "name,parser_cls",
        [
            ("qwen3", Qwen3ReasoningParser),
            ("deepseek_r1", DeepSeekR1ReasoningParser),
            ("vibethinker", VibeThinkerReasoningParser),
        ],
    )
    def test_length_finish_without_thinking_flips_to_content(self, name, parser_cls):
        """Counter-case: ``finish_reason="length"`` with thinking NOT
        active is just a non-thinking model truncated mid-answer.
        Flip to content per #569 — the bytes are an incomplete answer,
        not an interrupted thought trace."""
        parser = parser_cls()
        trace = "The answer is 12"
        result = parser.finalize_streaming(
            trace,
            matched_stop=None,
            prompt_thinking_active=False,
            finish_reason="length",
        )
        assert result is not None
        assert result.content == trace, (
            f"[{name}] non-thinking length-cut: regression — "
            f"answer suppressed: {result!r}"
        )
        assert result.reasoning is None


class TestMaxTokensMidThink:
    """Same accumulator state as ``stop`` mid-think: the engine
    truncates the suffix when ``max_tokens`` fires. Covered separately
    to lock down the cycle-2 F-12-1 max_tokens variant.
    """

    @pytest.mark.parametrize("name,parser_cls", THINK_PARSERS_WITH_BASE)
    def test_explicit_think_max_tokens_no_duplicate(self, name, parser_cls):
        """``max_tokens`` cut inside ``<think>``: finalize MUST route to
        reasoning to suppress duplication.

        Codex round-8 BLOCKING refinement (PR #799): thread
        ``finish_reason="length"`` through the simulator so the
        D-STOP-THINK ``max_tokens`` suppression branch actually fires.
        Without the truncation signal, natural-EOS now correctly flips
        to content (#569 silent-drop rescue) — that's the complementary
        path tested in
        ``test_natural_eos_with_explicit_opener_flips_to_content``.
        """
        parser = parser_cls()
        # Short prefix simulating ``max_tokens`` cut after a few tokens.
        chunks = ["<think>", "5+7"]
        # ``finish_reason="length"`` simulates the engine budget cut —
        # the real ``max_tokens`` truncation signal.
        thinking, text = _simulate_anthropic_stream(
            parser, chunks, finish_reason="length"
        )
        assert "5+7" in thinking, (
            f"[{name}] expected reasoning routing; thinking={thinking!r}"
        )
        assert not text, f"[{name}] D-STOP-THINK regression — text={text!r}"


class TestNormalResponseRegression:
    """Full ``<think>…</think>answer`` flow must continue to split
    cleanly post-fix.
    """

    @pytest.mark.parametrize("name,parser_cls", THINK_PARSERS_WITH_BASE)
    def test_complete_response_splits_cleanly(self, name, parser_cls):
        parser = parser_cls()
        chunks = ["<think>", "Let me think.", "</think>", "The answer is 12."]
        thinking, text = _simulate_anthropic_stream(parser, chunks)
        assert thinking.strip() == "Let me think."
        assert text.strip() == "The answer is 12."

    @pytest.mark.parametrize("name,parser_cls", THINK_PARSERS_WITH_BASE)
    def test_implicit_close_only(self, name, parser_cls):
        # Qwen3-style chat template injection: only ``</think>`` in
        # output. Bytes before close → reasoning; after close → content.
        #
        # GLM-4's chat template does NOT prompt-inject ``<think>`` (see
        # ``Glm4ReasoningParser`` module docstring); a stream that
        # opens with bare text routes to content via the explicit
        # ``has_tags`` override, so the implicit-think case doesn't
        # apply here.
        if name == "glm4":
            pytest.skip("glm4 has no prompt-injection contract")
        parser = parser_cls()
        chunks = ["Let me think.", "</think>", "The answer is 12."]
        thinking, text = _simulate_anthropic_stream(parser, chunks)
        assert thinking.strip() == "Let me think."
        assert text.strip() == "The answer is 12."


class TestFinalizeContractSurface:
    """Spot-check the parser-contract surface of ``finalize_streaming``
    when mid-think with explicit-opener evidence — the rescue text
    MUST surface via ``reasoning``, NEVER via ``content``. This locks
    the invariant against future refactors that might re-introduce
    the content-emission path.

    Codex round-N BLOCKING scope: only the explicit-opener and
    implicit-think-evidence branches are pinned here. The no-opener
    no-bare-preamble branch (casual answer) IS allowed to emit
    content — see ``TestStopMidThinkNoOpener.test_casual_answer_flips_to_content``.
    """

    @pytest.mark.parametrize("name,parser_cls", THINK_PARSERS_WITH_BASE)
    def test_finalize_never_emits_content_mid_think_with_explicit_opener(
        self, name, parser_cls
    ):
        """D-STOP-THINK invariant: explicit-opener mid-think UNDER A
        REAL TRUNCATION SIGNAL (matched_stop set OR
        finish_reason="length") MUST surface via reasoning, never
        content — otherwise the bytes shipped during streaming would
        be duplicated by the route's finalize content emission.

        Codex round-8 BLOCKING refinement (PR #799): scope the
        invariant to the truncation-signal branches. Natural-EOS with
        an explicit opener now legitimately emits content (the #569
        silent-drop rescue) — that branch is tested separately in
        ``test_natural_eos_with_explicit_opener_flips_to_content``.
        """
        parser = parser_cls()
        for accumulated in [
            "<think>still thinking",
            "<think>5+7",
        ]:
            # Cover both truncation signals — the suppression branch
            # must hold for stop AND length.
            for truncation in (
                {"matched_stop": "STOP"},
                {"finish_reason": "length"},
            ):
                parser.reset_state()
                # Drive the streaming state so the parser's internal flags
                # mirror what the live route would have set.
                prev = ""
                for ch in [accumulated]:
                    cur = prev + ch
                    parser.extract_reasoning_streaming(prev, cur, ch)
                    prev = cur
                result = parser.finalize_streaming(accumulated, **truncation)
                # Either None (no correction) OR reasoning-only — never
                # content under a real truncation signal.
                if result is not None:
                    assert result.content is None, (
                        f"[{name}] D-STOP-THINK invariant violation under "
                        f"truncation={truncation}: finalize emitted "
                        f"content={result.content!r} for explicit-opener "
                        f"input {accumulated!r}"
                    )


class TestGemma4ChannelGrammar:
    """Gemma 4 uses ``<|channel>thought\\n…<channel|>`` channel grammar
    instead of ``<think>…</think>``. The same shape leak applies when
    ``stop`` cuts before the closing ``<channel|>`` — pre-fix
    ``extract_reasoning`` routed the entire thought trace into
    ``content`` (the no-blocks regex branch); post-fix detects the
    unclosed opener and routes the body to ``reasoning``.

    Cycle-6 F-CORR-2 (gemma-4-26b/12b).
    """

    def test_mid_thought_routes_to_reasoning(self):
        parser = Gemma4ReasoningParser()
        text = "<|channel>thought\nLet me think about 5+7. "
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning is not None, "mid-thought trace must surface as reasoning"
        assert "Let me think" in reasoning
        # No channel-marker leak into content
        if content:
            assert "<|channel>" not in content, (
                f"channel marker leaked into content: {content!r}"
            )
            assert "thought" not in content or content.startswith(
                ("Sure", "Okay", "Let")
            ), f"thought-trace bytes leaked into content: {content!r}"

    def test_full_thought_plus_content_unchanged(self):
        parser = Gemma4ReasoningParser()
        text = (
            "<|channel>thought\nLet me think.<channel|>"
            "<|channel>content\nThe answer is 12.<channel|>"
        )
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning is not None and "Let me think" in reasoning
        assert content is not None and "answer is 12" in content


class TestHermesWithReasoningComposition:
    """Hermes is NOT a reasoning parser; the cycle-5 cross-confirmation
    came from qwen3.5-27b-8bit which layers the hermes tool parser on
    top of the Qwen3 reasoning parser. The reasoning leak shape is
    fundamentally in the Qwen3 parser's ``finalize_streaming``; the
    hermes tool parser is incidental and doesn't change the finalize
    semantics. This test confirms the Qwen3 parser fix carries through
    when hermes is the tool layer.
    """

    def test_qwen3_finalize_under_hermes_alias(self):
        """Codex round-8 BLOCKING refinement (PR #799): thread the
        ``matched_stop`` truncation signal so the test pins the
        D-STOP-THINK suppression branch under hermes composition.
        """
        # Hermes tool parser inspection is orthogonal — finalize for
        # the reasoning channel runs on the Qwen3 parser regardless of
        # tool parser choice.
        parser = Qwen3ReasoningParser()
        chunks = ["<think>", "Let me reason ", "step by step. "]
        # ``matched_stop`` simulates the cycle-5 live-repro shape:
        # qwen3.5-27b-8bit + hermes tool layer with a user stop string
        # firing inside the unclosed ``<think>`` block.
        thinking, text = _simulate_anthropic_stream(parser, chunks, matched_stop="STOP")
        assert "Let me reason" in thinking
        assert not text, (
            f"D-STOP-THINK regression under hermes composition: text={text!r}"
        )


class TestNonStreamingHelperSymmetry:
    """The non-streaming OpenAI envelope path goes through
    ``service.helpers._finalize_content_and_reasoning`` which delegates
    to ``parser.extract_reasoning``. The truncated-``<think>`` plug
    there suppresses the same leak on the non-streaming surface. Pin
    the symmetric outcome across parsers.
    """

    @pytest.mark.parametrize("name,parser_cls", THINK_PARSERS_WITH_BASE)
    def test_non_streaming_unclosed_think_routes_to_reasoning_only(
        self, name, parser_cls
    ):
        from vllm_mlx.service.helpers import _finalize_content_and_reasoning

        raw_text = "<think>Let me think about 5+7."
        cleaned_text = raw_text
        parser = parser_cls()
        content, reasoning = _finalize_content_and_reasoning(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            enable_thinking=True,
        )
        # Reasoning must carry the trace; content must NOT duplicate it.
        assert reasoning and "Let me think" in reasoning, (
            f"[{name}] reasoning missing: reasoning={reasoning!r}"
        )
        assert not (content and content.strip() == (reasoning or "").strip()), (
            f"[{name}] D-STOP-THINK regression — content duplicates "
            f"reasoning.\n  content={content!r}\n  reasoning={reasoning!r}"
        )

    def test_gemma4_non_streaming_mid_thought(self):
        from vllm_mlx.api.utils import clean_output_text, strip_thinking_tags
        from vllm_mlx.service.helpers import _finalize_content_and_reasoning

        raw_text = "<|channel>thought\nLet me think about 5+7. The answer is "
        cleaned_text = raw_text
        parser = Gemma4ReasoningParser()
        content, reasoning = _finalize_content_and_reasoning(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            enable_thinking=True,
        )
        final_content = (
            strip_thinking_tags(clean_output_text(content)) if content else None
        )
        # Channel-marker bytes must not leak into the final content surface.
        assert reasoning and "Let me think" in reasoning
        assert not final_content or "<|channel>" not in final_content
