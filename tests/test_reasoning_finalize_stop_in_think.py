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


def _simulate_anthropic_stream(parser, chunks):
    """Replay ``chunks`` through the parser the way the Anthropic /
    Responses streaming routes do, then apply the route's
    ``finalize_streaming`` consumer protocol.

    The Anthropic route emits each streaming delta's ``reasoning``
    as a ``thinking_delta`` and each ``content`` as a ``text_delta``.
    At end-of-stream it calls ``finalize_streaming(accumulated_raw)``
    and emits ``final_msg.content`` as a NEW text block (only acts
    on ``.content``, NOT on ``.reasoning``).

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
    final = parser.finalize_streaming(prev)
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
        parser = parser_cls()
        chunks = ["<think>", "Let me think ", "about 5+7. The "]
        thinking, text = _simulate_anthropic_stream(parser, chunks)
        # Some reasoning must have streamed; what matters is that the
        # text channel does NOT also receive the trace.
        assert "Let me think" in thinking, (
            f"[{name}] streaming did not route reasoning: thinking={thinking!r}"
        )
        assert not text, (
            f"[{name}] D-STOP-THINK regression — bytes duplicated into the "
            f"text channel.\n  thinking={thinking!r}\n  text={text!r}"
        )


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


class TestMaxTokensMidThink:
    """Same accumulator state as ``stop`` mid-think: the engine
    truncates the suffix when ``max_tokens`` fires. Covered separately
    to lock down the cycle-2 F-12-1 max_tokens variant.
    """

    @pytest.mark.parametrize("name,parser_cls", THINK_PARSERS_WITH_BASE)
    def test_explicit_think_max_tokens_no_duplicate(self, name, parser_cls):
        parser = parser_cls()
        # Short prefix simulating ``max_tokens`` cut after a few tokens.
        chunks = ["<think>", "5+7"]
        thinking, text = _simulate_anthropic_stream(parser, chunks)
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
        parser = parser_cls()
        for accumulated in [
            "<think>still thinking",
            "<think>5+7",
        ]:
            parser.reset_state()
            # Drive the streaming state so the parser's internal flags
            # mirror what the live route would have set.
            prev = ""
            for ch in [accumulated]:
                cur = prev + ch
                parser.extract_reasoning_streaming(prev, cur, ch)
                prev = cur
            result = parser.finalize_streaming(accumulated)
            # Either None (no correction) OR reasoning-only — never
            # content.
            if result is not None:
                assert result.content is None, (
                    f"[{name}] D-STOP-THINK invariant violation: "
                    f"finalize emitted content={result.content!r} for "
                    f"explicit-opener input {accumulated!r}"
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
        # Hermes tool parser inspection is orthogonal — finalize for
        # the reasoning channel runs on the Qwen3 parser regardless of
        # tool parser choice.
        parser = Qwen3ReasoningParser()
        chunks = ["<think>", "Let me reason ", "step by step. "]
        thinking, text = _simulate_anthropic_stream(parser, chunks)
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
