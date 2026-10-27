# SPDX-License-Identifier: Apache-2.0
"""R12-M1b regression tests — Anthropic /v1/messages content-block contract.

Pins three invariants on the OpenAI-to-Anthropic adapter and the
canonical sanitizer that together close Mira r12 R-3 SEVERE finding on
``/v1/messages``:

* **Bug #1 — ``<think>`` literal leak at ``max_tokens=1``**. The
  prompt-template-injected ``<think>`` opener was the only token the
  reasoning parser saw at ``max_tokens=1`` on a thinking model and it
  ended up verbatim in BOTH the ``thinking`` AND the ``text`` content
  blocks. Fixed by routing reasoning-channel bytes (Anthropic
  ``thinking`` block) through the new
  :func:`strip_reasoning_channel_markup`, which strips BOTH the
  ``<think>`` opener AND ``</think>`` closer. The canonical
  :func:`sanitize_output` intentionally STILL preserves the bare
  ``<think>`` opener on content-channel bytes — that opener is
  sometimes legit there (Nemotron prefix injection, literal-tag prose
  like ``"use the <think> tag in HTML"``), and stripping it would
  regress ``TestStreamingPostProcessorNemotron::test_thinking_prefix_injected``
  + the ``test_streaming_reasoning_split_r8c.py`` literal-tag pin.

* **Length stop keeps only the thinking block** (#5295). A length stop
  before the reasoning end marker carries the trace in the ``thinking``
  block only; no text is synthesized.

Mira r12 dogfood report path:
  ``/tmp/dogfood-0815/mira-r12.md`` (R-3 section)
"""

from __future__ import annotations

import pytest

from rapid_mlx.api.anthropic_adapter import (
    _sanitize_reasoning_channel,
    openai_to_anthropic,
)
from rapid_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    Usage,
)
from rapid_mlx.api.utils import sanitize_output, strip_reasoning_channel_markup

# ---------------------------------------------------------------------------
# Bug #1 — ``<think>`` literal does NOT appear in ANY content block at
# max_tokens=1 on a thinking model. Covers both the canonical sanitizer
# and the Anthropic adapter's per-block sanitization.
# ---------------------------------------------------------------------------


class TestThinkLiteralLeak:
    """Mira r12 R-3 bonus regression — ``<think>`` literal leak."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # bare opener — only token seen at max_tokens=1
            ("<think>", ""),
            # opener as first chars of a longer string
            ("<think>hello", "hello"),
            # opener mid-string
            ("hello <think> world", "hello  world"),
            # closer (already stripped by ``sanitize_output`` pre-fix —
            # this helper now strips it ALSO so the two routes share
            # the same channel-aware strip semantics)
            ("</think>", ""),
            # both
            ("<think>x</think>", "x"),
            # trailing whitespace / newlines after opener — kept as-is
            # (helper is byte-level; surrounding callers handle
            # whitespace-only collapse via their own predicates)
            ("   <think>   ", "      "),
            ("<think>\n", "\n"),
        ],
    )
    def test_strip_reasoning_channel_markup_strips_both_tags(self, raw: str, expected):
        """The reasoning-channel sanitizer strips BOTH ``<think>`` and
        ``</think>``. This is distinct from the canonical
        ``sanitize_output`` (which only strips the closer) because the
        opener is sometimes legit ``content``-channel text (Nemotron
        prefix injection, literal-tag prose) and we MUST NOT erase
        those — but on the reasoning channel the wrapping tags are
        always structural parser artifact.
        """
        assert strip_reasoning_channel_markup(raw) == expected

    def test_sanitize_output_does_not_strip_think_opener(self):
        """The canonical ``sanitize_output`` MUST NOT strip the bare
        ``<think>`` opener — that would erase legitimate Nemotron
        prefix injection in the standard streaming path
        (``TestStreamingPostProcessorNemotron::test_thinking_prefix_injected``)
        and silently swallow legitimate literal-tag prose like
        ``"use the <think> tag in HTML"`` (covered by
        ``test_streaming_reasoning_split_r8c.py``). The opener strip
        lives in the channel-aware helper instead — see
        :func:`strip_reasoning_channel_markup`.
        """
        # Bare opener passes through unchanged (no special-token chars
        # other than the angle brackets — fast-path also returns as-is).
        assert sanitize_output("<think>") == "<think>"
        assert sanitize_output("use the <think> tag") == "use the <think> tag"
        # Closer is still stripped by the canonical sanitizer (legacy
        # contract preserved — pre-R12-M1b behaviour).
        assert sanitize_output("</think>") is None

    def test_max_tokens_1_no_think_leak_in_any_block(self):
        """Top-of-stack repro: ``max_tokens=1`` on a thinking model
        leaves ``reasoning_text="<think>"`` (the prompt template's
        pre-injected opener is the only token the parser sees) and
        ``content=None``.

        Invariant: the literal ``<think>`` MUST NOT appear in either
        the ``thinking`` content block OR the ``text`` content block
        on the Anthropic envelope.
        """
        reasoning_text = "<think>"
        # Build the OpenAI-side response that the route hands to the
        # adapter (matches the call shape in routes/anthropic.py).
        resp = ChatCompletionResponse(
            model="qwen3-0.6b-bf16",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=None,
                        reasoning_content=reasoning_text,
                        tool_calls=None,
                    ),
                    finish_reason="length",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=1, total_tokens=11),
        )
        anth = openai_to_anthropic(resp, "qwen3-0.6b-bf16", reasoning_enabled=True)
        # Walk every content block — the literal must not survive on
        # either the thinking field or the text field.
        for block in anth.content:
            if block.type == "thinking":
                assert "<think>" not in (block.thinking or ""), (
                    f"<think> literal leaked into thinking block: {block.thinking!r}"
                )
            elif block.type == "text":
                assert "<think>" not in (block.text or ""), (
                    f"<think> literal leaked into text block: {block.text!r}"
                )

    def test_max_tokens_1_thinking_block_suppressed_when_all_markup(self):
        """When the entire ``reasoning_text`` collapses to nothing after
        sanitization (``<think>`` only), the adapter must NOT emit an
        empty/whitespace thinking block — symmetric with the existing
        whitespace-only guard in the pre-R12-M1b code path.
        """
        reasoning_text = "<think>"
        resp = ChatCompletionResponse(
            model="qwen3-0.6b-bf16",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=None,
                        reasoning_content=reasoning_text,
                        tool_calls=None,
                    ),
                    finish_reason="length",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=1, total_tokens=11),
        )
        anth = openai_to_anthropic(resp, "qwen3-0.6b-bf16", reasoning_enabled=True)
        thinking_blocks = [b for b in anth.content if b.type == "thinking"]
        assert thinking_blocks == [], (
            f"thinking block should be suppressed when all reasoning is markup; "
            f"got {[b.thinking for b in thinking_blocks]!r}"
        )


# ---------------------------------------------------------------------------
# Length stop mid-think: the trace stays in the thinking block only.
# ---------------------------------------------------------------------------


class TestLengthStopThinkingOnly:
    """#5295: no synthesized text block on a length stop mid-think."""

    def test_length_stop_adapter_emits_only_thinking_block(self):
        """With ``content=None`` the adapter emits no text block and a
        thinking block carrying the full sanitized reasoning trace."""
        reasoning_text = "Mid-think reasoning."
        resp = ChatCompletionResponse(
            model="qwen3-0.6b-bf16",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=None,
                        reasoning_content=reasoning_text,
                        tool_calls=None,
                    ),
                    finish_reason="length",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )
        anth = openai_to_anthropic(resp, "qwen3-0.6b-bf16", reasoning_enabled=True)
        # No text block, thinking block carries reasoning.
        thinking_blocks = [b for b in anth.content if b.type == "thinking"]
        text_blocks = [b for b in anth.content if b.type == "text"]
        assert len(thinking_blocks) == 1
        assert reasoning_text in (thinking_blocks[0].thinking or "")
        # No non-empty text block. (The adapter always adds
        # a zero-length placeholder text block when content_blocks is
        # otherwise empty, but the thinking block above prevents that.)
        non_empty_text_blocks = [b for b in text_blocks if (b.text or "").strip()]
        assert non_empty_text_blocks == []


# ---------------------------------------------------------------------------
# Unit tests on the per-block sanitizer helper directly. Keeps the
# adapter-level integration tests above focused on observable wire shape
# and the helper-level tests focused on the contract.
# ---------------------------------------------------------------------------


class TestSanitizeReasoningChannel:
    """Unit tests on :func:`_sanitize_reasoning_channel`."""

    @pytest.mark.parametrize(
        "reasoning,expected",
        [
            # No reasoning → no block
            (None, None),
            ("", None),
            ("   ", None),
            # Plain reasoning → sanitized passthrough
            ("normal reasoning trace.", "normal reasoning trace."),
            # Special-token leak in reasoning → sanitized
            ("<think>", None),
            ("<think>kept</think>", "kept"),
            (
                "real thought <|im_end|> trailing markup",
                "real thought  trailing markup",
            ),
        ],
    )
    def test_sanitizes_thinking_content(self, reasoning, expected):
        assert _sanitize_reasoning_channel(reasoning) == expected


class TestMarkupOnlyDelta:
    """Codex r1 P2 (R12-M1b): the markup-only-delta dupe case.

    When ``reasoning_text`` and ``text`` differ ONLY by reasoning-channel
    markup (e.g. ``reasoning_text="<think>done</think>"`` and
    ``text="done"``), the channel-aware sanitizer normalizes the
    thinking payload to the same bytes the text block already carries.
    The pre-fix gate ``reasoning_text != text`` was a True predicate
    (the strings DO differ), so both blocks emitted with the same
    visible bytes — exactly the duplicate-block failure mode this
    suppression was supposed to prevent.

    Fix: gate on ``thinking_body != text`` (post-sanitize).
    """

    def test_markup_only_delta_does_not_emit_duplicate_thinking_block(self):
        """``reasoning_text`` is just the wrapped form of ``text`` — the
        adapter must NOT emit a thinking block that, after sanitization,
        contains the same visible bytes as the text block.
        """
        reasoning_text = "<think>done</think>"
        text = "done"
        resp = ChatCompletionResponse(
            model="qwen3-0.6b-bf16",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=text,
                        reasoning_content=reasoning_text,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )
        anth = openai_to_anthropic(resp, "qwen3-0.6b-bf16", reasoning_enabled=True)
        # The text block must carry the answer.
        text_blocks = [b for b in anth.content if b.type == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "done"
        # The thinking block MUST be suppressed — emitting it would
        # render the same visible bytes twice (Codex r1 P2 dupe).
        thinking_blocks = [b for b in anth.content if b.type == "thinking"]
        assert thinking_blocks == [], (
            "thinking block must be suppressed when its post-sanitize "
            "payload equals the text block (markup-only delta dupe); "
            f"got {[b.thinking for b in thinking_blocks]!r}"
        )

    def test_markup_only_delta_with_distinct_reasoning_still_emits_thinking(self):
        """Pin the inverse: when the post-sanitize thinking payload is
        genuinely DIFFERENT from the text block (the normal happy path),
        the adapter MUST emit the thinking block. Guards against an
        over-aggressive fix that suppresses the thinking block for
        every case where reasoning + content share a non-empty suffix.
        """
        reasoning_text = "<think>let me think about this carefully</think>"
        text = "the answer"
        resp = ChatCompletionResponse(
            model="qwen3-0.6b-bf16",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=text,
                        reasoning_content=reasoning_text,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        anth = openai_to_anthropic(resp, "qwen3-0.6b-bf16", reasoning_enabled=True)
        thinking_blocks = [b for b in anth.content if b.type == "thinking"]
        text_blocks = [b for b in anth.content if b.type == "text"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].thinking == "let me think about this carefully"
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "the answer"

    def test_unsanitised_text_with_removable_markup_still_dedupes(self):
        """Codex r3 BLOCKING (R12-M1b): the dedupe gate must compare
        ``thinking_body`` against ``text`` POST-sanitize on BOTH sides,
        not just one. Hostile shape:

        * ``reasoning_text="<think>done</think>"`` — channel-aware
          sanitizer normalizes the thinking payload to ``"done"``.
        * ``text="done</think>"`` — the canonical content-channel
          ``sanitize_output`` collapses this to ``"done"`` too.

        The Anthropic route layer runs ``sanitize_output`` on
        ``final_content`` before handing it to ``openai_to_anthropic``,
        so the in-route path normally never exhibits this shape. BUT
        if a non-route caller (test helper, future SSE finalize path,
        a future internal route that forgets the route's sanitization
        step) hands the adapter a raw ``text`` carrying removable
        markup, the dedupe gate must STILL refuse to render the same
        visible bytes twice. Gate on post-sanitize equality on both
        sides closes the hole regardless of upstream sanitization
        state.
        """
        reasoning_text = "<think>done</think>"
        text = "done</think>"  # carries removable markup
        resp = ChatCompletionResponse(
            model="qwen3-0.6b-bf16",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=text,
                        reasoning_content=reasoning_text,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )
        anth = openai_to_anthropic(resp, "qwen3-0.6b-bf16", reasoning_enabled=True)
        thinking_blocks = [b for b in anth.content if b.type == "thinking"]
        text_blocks = [b for b in anth.content if b.type == "text"]
        # The thinking block MUST be suppressed: post-sanitize, both
        # sides reduce to ``"done"`` — emitting both would render the
        # answer twice on the wire (codex r3 BLOCKING dupe shape).
        assert thinking_blocks == [], (
            "thinking block must be suppressed when its post-sanitize "
            "payload equals the post-sanitize text payload (codex r3 "
            f"BLOCKING dupe shape); got {[b.thinking for b in thinking_blocks]!r}"
        )
        # Codex r4 medium (R12-M1b): the text block emission must
        # ALSO go through the adapter-level sanitizer — half the gate
        # only suppresses the duplicate; the other half must scrub the
        # removable markup so ``</think>`` never reaches the wire.
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "done", (
            "text block must carry the sanitized payload, not the raw "
            f"input with markup; got {text_blocks[0].text!r}"
        )
        assert "</think>" not in (text_blocks[0].text or ""), (
            f"</think> leaked into text block: {text_blocks[0].text!r}"
        )
