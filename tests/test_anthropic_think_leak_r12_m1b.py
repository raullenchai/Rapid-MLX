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
  ``thinking`` block + rescue tail) through the new
  :func:`strip_reasoning_channel_markup`, which strips BOTH the
  ``<think>`` opener AND ``</think>`` closer. The canonical
  :func:`sanitize_output` intentionally STILL preserves the bare
  ``<think>`` opener on content-channel bytes — that opener is
  sometimes legit there (Nemotron prefix injection, literal-tag prose
  like ``"use the <think> tag in HTML"``), and stripping it would
  regress ``TestStreamingPostProcessorNemotron::test_thinking_prefix_injected``
  + the ``test_streaming_reasoning_split_r8c.py`` literal-tag pin.

* **Bug #2 — rescue tail duplicated across content blocks**. With
  ``max_tokens=40`` on a thinking model, the H-01 (PR #802 / R12-8 PR
  #875) rescue payload (``sentinel + "\\n\\n" + tail-of-reasoning``)
  was emitted into the ``text`` block, AND the same tail bytes
  remained the suffix of the ``reasoning_text`` that became the
  ``thinking`` block — so the same partial sentence rendered twice on
  the Anthropic envelope. The adapter now trims the matching
  RESCUE_TAIL_LENGTH suffix from the ``thinking`` block when the
  ``text`` carries a rescue payload, so each reasoning byte surfaces
  on exactly one content block.

* **Design preservation — rescue surfaces to Anthropic**. PR #802 /
  issue #858 explicitly chose to surface the H-01 rescue to the
  Anthropic surface (GUI clients that render only text bubbles benefit
  from the literal cue). This test pins that the rescue ``text`` block
  still carries the sentinel + tail — only the duplication is fixed.

Mira r12 dogfood report path:
  ``/tmp/dogfood-0815/mira-r12.md`` (R-3 section)
"""

from __future__ import annotations

import pytest

from vllm_mlx.api.anthropic_adapter import _thinking_block_content, openai_to_anthropic
from vllm_mlx.api.constants import (
    REASONING_CUTOFF_SENTINEL,
    RESCUE_TAIL_LENGTH,
    is_rescue_payload,
)
from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    Usage,
)
from vllm_mlx.api.utils import sanitize_output, strip_reasoning_channel_markup

# ---------------------------------------------------------------------------
# Pure-API rescue-payload reconstruction.
#
# Codex r2 P1 (R12-M1b): the test file previously imported
# ``_apply_reasoning_cutoff_notice`` from ``vllm_mlx.service.helpers``,
# which transitively pulls the engine layer into pytest collection.
# Existing route-helper tests already follow that pattern (see
# ``tests/test_reasoning_content_null_rescue.py``) and the CI test
# matrices stay green, but keeping THIS file pure-API removes a class
# of headless-environment failure modes by construction.
#
# The rescue payload shape is the public contract documented in
# ``api.constants.is_rescue_payload`` and the
# ``service.helpers._build_reasoning_rescue_payload`` docstring; we
# reconstruct it here against those same public symbols so the test
# pins the wire shape rather than the helper's call signature.
# ---------------------------------------------------------------------------


def _make_rescue_payload(reasoning_text: str) -> str:
    """Reconstruct the canonical rescue payload from public symbols.

    Pins the wire shape ``sentinel + "\\n\\n" + sanitized_tail`` —
    same layout the helper writes — without importing the helper
    itself. ``sanitized_tail`` runs through both stages of the
    channel-aware sanitizer (matching
    ``_build_reasoning_rescue_payload``).

    Strip-before-slice order mirrors the helper (codex r3 P2 fix):
    sanitizing channel markup before choosing the tail boundary
    means a ``<think>`` tag straddling the slice can't leak a
    partial fragment into the rescue tail.

    When the tail collapses to empty after sanitization, the rescue
    is just the bare sentinel (matching the helper's all-markup
    fallback).
    """
    stripped = strip_reasoning_channel_markup(reasoning_text.rstrip())
    tail = stripped[-RESCUE_TAIL_LENGTH:]
    sanitized = sanitize_output(tail)
    if not sanitized:
        return REASONING_CUTOFF_SENTINEL
    return f"{REASONING_CUTOFF_SENTINEL}\n\n{sanitized}"


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

    def test_max_tokens_1_no_think_leak_in_any_block(self, monkeypatch):
        """Top-of-stack repro: ``max_tokens=1`` on a thinking model
        leaves ``reasoning_text="<think>"`` (the prompt template's
        pre-injected opener is the only token the parser sees) and
        ``content=None``. The rescue fires on ``finish_reason="length"``
        and produces a sentinel-prefixed rescue payload.

        Invariant: the literal ``<think>`` MUST NOT appear in either
        the ``thinking`` content block OR the ``text`` content block
        on the Anthropic envelope.
        """
        # Enable the rescue (default-on, but pin for hermetic test).
        monkeypatch.setenv("RAPID_MLX_REASONING_RESCUE", "on")
        reasoning_text = "<think>"
        final_content = _make_rescue_payload(reasoning_text)
        # Build the OpenAI-side response that the route hands to the
        # adapter (matches the call shape in routes/anthropic.py).
        resp = ChatCompletionResponse(
            model="qwen3-0.6b-bf16",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=final_content,
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

    def test_max_tokens_1_thinking_block_suppressed_when_all_markup(self, monkeypatch):
        """When the entire ``reasoning_text`` collapses to nothing after
        sanitization (``<think>`` only), the adapter must NOT emit an
        empty/whitespace thinking block — symmetric with the existing
        whitespace-only guard in the pre-R12-M1b code path.
        """
        monkeypatch.setenv("RAPID_MLX_REASONING_RESCUE", "on")
        reasoning_text = "<think>"
        final_content = _make_rescue_payload(reasoning_text)
        resp = ChatCompletionResponse(
            model="qwen3-0.6b-bf16",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=final_content,
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
# Bug #2 — Rescue tail must appear in EXACTLY ONE block (the text block).
# ---------------------------------------------------------------------------


class TestRescueTailNoDuplication:
    """Mira r12 R-3 dupe arm — rescue tail must not appear in BOTH blocks."""

    def test_rescue_tail_in_text_block_not_thinking_block(self, monkeypatch):
        """``max_tokens=40`` repro: a thinking model emits a long
        reasoning trace and gets cut by length. The rescue payload
        (``sentinel + "\\n\\n" + tail-of-reasoning``) goes into the
        text block. The same tail bytes WERE also the suffix of the
        ``reasoning_text`` that became the thinking block — Mira saw
        the same partial sentence twice in the Anthropic envelope.

        Pin: the rescue tail (last RESCUE_TAIL_LENGTH chars of
        ``reasoning_text``) appears in the ``text`` block but NOT
        anywhere in the ``thinking`` block.
        """
        monkeypatch.setenv("RAPID_MLX_REASONING_RESCUE", "on")
        # Long enough reasoning that the tail slice is a strict
        # suffix (longer than RESCUE_TAIL_LENGTH). Uses a unique
        # marker substring inside the tail slice so the assertion
        # has a precise anchor.
        prefix = "A" * (RESCUE_TAIL_LENGTH * 2)  # 400 chars of filler
        tail_marker = "UNIQUE_TAIL_MARKER_XYZ"
        reasoning_text = prefix + " ... " + tail_marker + " conclusion."
        final_content = _make_rescue_payload(reasoning_text)
        # Confirm the marker landed in the rescue payload (sanity).
        assert tail_marker in final_content, (
            "test setup invariant: marker must be inside the rescue tail slice"
        )
        resp = ChatCompletionResponse(
            model="qwen3-0.6b-bf16",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=final_content,
                        reasoning_content=reasoning_text,
                        tool_calls=None,
                    ),
                    finish_reason="length",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=40, total_tokens=50),
        )
        anth = openai_to_anthropic(resp, "qwen3-0.6b-bf16", reasoning_enabled=True)
        text_blocks = [b for b in anth.content if b.type == "text"]
        thinking_blocks = [b for b in anth.content if b.type == "thinking"]
        assert len(text_blocks) == 1, (
            f"expected exactly one text block: {anth.content!r}"
        )
        assert len(thinking_blocks) == 1, (
            f"expected exactly one thinking block: {anth.content!r}"
        )
        # The marker must be in the text block …
        assert tail_marker in (text_blocks[0].text or ""), (
            f"rescue tail marker missing from text block: {text_blocks[0].text!r}"
        )
        # … and MUST NOT appear in the thinking block (the dedupe).
        assert tail_marker not in (thinking_blocks[0].thinking or ""), (
            f"rescue tail leaked into thinking block (duplication): "
            f"{thinking_blocks[0].thinking!r}"
        )

    def test_rescue_tail_appears_in_exactly_one_block(self, monkeypatch):
        """Cross-block uniqueness: the rescue payload's tail
        substring (everything after the sentinel + ``\\n\\n``
        separator) must appear in EXACTLY one of the content blocks.
        Parametrized over the per-block role pin (which one) is the
        ``text`` block — matches the PR #802 / R12-8 design where the
        rescue surfaces to ``text``.
        """
        monkeypatch.setenv("RAPID_MLX_REASONING_RESCUE", "on")
        reasoning_text = "X" * 500 + " final partial conclusion."
        final_content = _make_rescue_payload(reasoning_text)
        # Extract the tail substring from the rescue payload.
        separator = "\n\n"
        rescue_tail = final_content.split(separator, 1)[1]
        assert rescue_tail, "test invariant: rescue tail must be non-empty"

        resp = ChatCompletionResponse(
            model="qwen3-0.6b-bf16",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=final_content,
                        reasoning_content=reasoning_text,
                        tool_calls=None,
                    ),
                    finish_reason="length",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=200, total_tokens=210),
        )
        anth = openai_to_anthropic(resp, "qwen3-0.6b-bf16", reasoning_enabled=True)
        # Count occurrences of the tail substring across blocks.
        text_hits = sum(
            1
            for b in anth.content
            if b.type == "text" and rescue_tail in (b.text or "")
        )
        thinking_hits = sum(
            1
            for b in anth.content
            if b.type == "thinking" and rescue_tail in (b.thinking or "")
        )
        # The text block has the tail (rescue) AND no thinking block carries it.
        assert text_hits == 1, (
            f"rescue tail must surface in exactly one text block; got {text_hits}"
        )
        assert thinking_hits == 0, (
            f"rescue tail must NOT appear in any thinking block; got {thinking_hits}"
        )

    def test_thinking_block_carries_prefix_when_reasoning_longer_than_tail(
        self, monkeypatch
    ):
        """When ``reasoning_text`` is longer than ``RESCUE_TAIL_LENGTH``,
        the thinking block should carry the PREFIX (everything except
        the last RESCUE_TAIL_LENGTH chars). Reading both blocks
        reconstructs the full reasoning trace contextually — the
        sentinel anchors the rescue tail, the thinking block has the
        leading thought process.
        """
        monkeypatch.setenv("RAPID_MLX_REASONING_RESCUE", "on")
        prefix = "BEGINNING THOUGHT PROCESS HERE. "
        reasoning_text = prefix + "X" * (RESCUE_TAIL_LENGTH + 50)
        final_content = _make_rescue_payload(reasoning_text)
        resp = ChatCompletionResponse(
            model="qwen3-0.6b-bf16",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=final_content,
                        reasoning_content=reasoning_text,
                        tool_calls=None,
                    ),
                    finish_reason="length",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=300, total_tokens=310),
        )
        anth = openai_to_anthropic(resp, "qwen3-0.6b-bf16", reasoning_enabled=True)
        thinking_blocks = [b for b in anth.content if b.type == "thinking"]
        assert len(thinking_blocks) == 1
        # Prefix marker must survive in the thinking block.
        assert "BEGINNING THOUGHT PROCESS HERE." in thinking_blocks[0].thinking, (
            f"thinking block lost the prefix: {thinking_blocks[0].thinking!r}"
        )

    def test_thinking_block_suppressed_when_reasoning_entirely_in_rescue_tail(
        self, monkeypatch
    ):
        """When ``reasoning_text`` is shorter than ``RESCUE_TAIL_LENGTH``,
        the entire trace lives inside the rescue payload's tail.
        Emitting the thinking block would be a byte-for-byte duplicate
        of what's already in the text block. Suppress it.
        """
        monkeypatch.setenv("RAPID_MLX_REASONING_RESCUE", "on")
        reasoning_text = "short reasoning trace under the tail length"
        assert len(reasoning_text) <= RESCUE_TAIL_LENGTH, (
            "test invariant: reasoning must be shorter than the tail length"
        )
        final_content = _make_rescue_payload(reasoning_text)
        resp = ChatCompletionResponse(
            model="qwen3-0.6b-bf16",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=final_content,
                        reasoning_content=reasoning_text,
                        tool_calls=None,
                    ),
                    finish_reason="length",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        anth = openai_to_anthropic(resp, "qwen3-0.6b-bf16", reasoning_enabled=True)
        thinking_blocks = [b for b in anth.content if b.type == "thinking"]
        text_blocks = [b for b in anth.content if b.type == "text"]
        assert thinking_blocks == [], (
            "thinking block must be suppressed when the whole reasoning trace "
            f"already lives in the rescue tail; got {[b.thinking for b in thinking_blocks]!r}"
        )
        # The text block still carries the full reasoning via the rescue.
        assert text_blocks
        assert reasoning_text in (text_blocks[0].text or "")


# ---------------------------------------------------------------------------
# Design preservation — PR #802 / issue #858 explicitly surfaces the
# rescue to /v1/messages. A future contributor should NOT regress this
# behaviour without an explicit operator-policy decision. The positive
# test below pins the design so the dedupe fix above can never grow
# into "remove the rescue from Anthropic entirely".
# ---------------------------------------------------------------------------


class TestRescueSurfacesToAnthropic:
    """Pin PR #802 / issue #858 — the H-01 rescue surfaces to Anthropic.

    Mira r12's spec said the rescue should NOT touch Anthropic content
    blocks; the operator-approved design (PR #802 / #875) is that it
    DOES, because GUI clients that only render text blocks need a
    user-visible cue when ``max_tokens`` was too low. This test exists
    so a future contributor reading the dedupe fix above doesn't
    interpret it as "remove rescue from Anthropic" and silently drop
    the operator-blessed behaviour.
    """

    def test_rescue_text_block_carries_sentinel_when_length_cut_mid_think(
        self, monkeypatch
    ):
        monkeypatch.setenv("RAPID_MLX_REASONING_RESCUE", "on")
        reasoning_text = "Mid-think reasoning that got cut by max_tokens."
        final_content = _make_rescue_payload(reasoning_text)
        resp = ChatCompletionResponse(
            model="qwen3-0.6b-bf16",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=final_content,
                        reasoning_content=reasoning_text,
                        tool_calls=None,
                    ),
                    finish_reason="length",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=30, total_tokens=40),
        )
        anth = openai_to_anthropic(resp, "qwen3-0.6b-bf16", reasoning_enabled=True)
        text_blocks = [b for b in anth.content if b.type == "text"]
        assert text_blocks, "rescue text block must be present (PR #802 design)"
        text_payload = text_blocks[0].text or ""
        assert text_payload.startswith(REASONING_CUTOFF_SENTINEL), (
            f"rescue text block must start with the sentinel "
            f"(PR #802 design — agentic auto-retry pattern-matches the prefix); "
            f"got {text_payload!r}"
        )
        # Verify it is in fact the rescue payload shape (sentinel +
        # \\n\\n + tail) and not just a model echo of the sentinel.
        assert is_rescue_payload(text_payload), (
            f"rescue text block payload failed the canonical shape gate: {text_payload!r}"
        )

    def test_opt_out_path_adapter_emits_only_thinking_block(self):
        """When the operator opts out of the rescue (the route helper
        returns ``content=None`` instead of the sentinel-prefixed
        payload), the adapter must:
          * NOT emit a text block (no rescue cue)
          * emit the thinking block carrying the full sanitized
            reasoning trace (no dedupe — ``content`` is None, so
            ``is_rescue_payload(content)`` is False and no suffix is
            trimmed)
        Mirrors the wire shape ``routes/anthropic.py`` produces when
        ``RAPID_MLX_REASONING_RESCUE=off`` /
        ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled``.
        """
        # No env var manipulation needed — we drive the adapter with
        # ``content=None`` directly, which is what the helper would
        # return on the opt-out branch.
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
        # No text block (no rescue), thinking block carries reasoning.
        thinking_blocks = [b for b in anth.content if b.type == "thinking"]
        text_blocks = [b for b in anth.content if b.type == "text"]
        assert len(thinking_blocks) == 1
        assert reasoning_text in (thinking_blocks[0].thinking or "")
        # No rescue → no non-empty text block. (The adapter always adds
        # a zero-length placeholder text block when content_blocks is
        # otherwise empty, but the thinking block above prevents that.)
        non_empty_text_blocks = [b for b in text_blocks if (b.text or "").strip()]
        assert non_empty_text_blocks == []


# ---------------------------------------------------------------------------
# Unit tests on the per-block sanitizer helper directly. Keeps the
# adapter-level integration tests above focused on observable wire shape
# and the helper-level tests focused on the contract.
# ---------------------------------------------------------------------------


class TestThinkingBlockContentHelper:
    """Unit tests on :func:`_thinking_block_content`."""

    @pytest.mark.parametrize(
        "reasoning,text,expected",
        [
            # No reasoning → no block
            (None, None, None),
            ("", None, None),
            ("   ", None, None),
            # Plain reasoning, no rescue → sanitized passthrough
            ("normal reasoning trace.", None, "normal reasoning trace."),
            # Special-token leak in reasoning → sanitized
            ("<think>", None, None),
            ("<think>kept</think>", None, "kept"),
            (
                "real thought <|im_end|> trailing markup",
                None,
                "real thought  trailing markup",
            ),
        ],
    )
    def test_sanitizes_thinking_content(self, reasoning, text, expected):
        assert _thinking_block_content(reasoning, text) == expected

    def test_dedupes_rescue_tail_from_thinking_block(self):
        """When ``text`` is a rescue payload, trim the matching suffix
        from the thinking block content so the tail surfaces on exactly
        one block.
        """
        reasoning = "PREFIX_KEEP_ME " + "Y" * (RESCUE_TAIL_LENGTH + 10)
        text = REASONING_CUTOFF_SENTINEL + "\n\n" + "Y" * RESCUE_TAIL_LENGTH
        out = _thinking_block_content(reasoning, text)
        assert out is not None
        assert "PREFIX_KEEP_ME" in out, (
            f"prefix lost when deduping rescue tail: {out!r}"
        )
        # The trimmed suffix should NOT appear in the thinking block.
        assert ("Y" * RESCUE_TAIL_LENGTH) not in out, (
            f"rescue tail still present in thinking block (dedupe failed): {out!r}"
        )

    def test_dedupes_to_none_when_reasoning_fits_in_tail(self):
        """If ``reasoning`` is shorter than RESCUE_TAIL_LENGTH and the
        text block carries the rescue, the entire trace is already in
        the text block — return None so the thinking block is
        suppressed.
        """
        reasoning = "all of it fits in the tail"
        text = REASONING_CUTOFF_SENTINEL + "\n\n" + reasoning
        assert _thinking_block_content(reasoning, text) is None

    def test_non_rescue_text_does_not_trigger_dedupe(self):
        """When ``text`` is a normal model answer (not a rescue
        payload), the thinking block must keep the full sanitized
        reasoning. The dedupe is gated on
        :func:`is_rescue_payload` so a legit response that happens to
        share a suffix with reasoning is not silently truncated.
        """
        reasoning = "my full thought process"
        text = "the answer"
        assert _thinking_block_content(reasoning, text) == reasoning


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


class TestSliceBisectsThinkTag:
    """Codex r3 P2 (R12-M1b): the slice-bisects-tag boundary case.

    When ``reasoning_text`` is just slightly longer than
    ``RESCUE_TAIL_LENGTH`` and contains a ``<think>`` tag near the
    boundary, a naive ``reasoning_text.rstrip()[-RESCUE_TAIL_LENGTH:]``
    slice can bisect the tag, leaving an orphan ``<th`` /  ``ink>``
    fragment that the regex stripper no longer matches.

    Fix: strip channel markup BEFORE choosing the tail boundary so
    the slice operates on the clean, in-channel byte stream — no
    tags left to bisect by construction.
    """

    @pytest.mark.parametrize(
        "prefix_len,total_len",
        [
            # Reasoning lengths that force ``<think>`` to straddle
            # the L - RESCUE_TAIL_LENGTH boundary. Each pinning shape
            # gets tested under both the rescue helper (text block)
            # and the adapter dedupe (thinking block) — symmetric
            # because both helpers now strip-before-slice.
            (1, 203),  # '<think>' at positions 1-7, boundary at 3
            (3, 205),  # '<think>' at positions 3-9, boundary at 5
            (5, 207),  # '<think>' at positions 5-11, boundary at 7
        ],
    )
    def test_no_partial_think_fragment_in_any_block(
        self, prefix_len: int, total_len: int
    ):
        """Slice-bisects-``<think>`` pinning: NO ``<th`` / ``ink>``
        fragment may appear in the rescue payload OR in the thinking
        block. Parametrized over multiple boundary alignments so the
        invariant is enforced uniformly across slice positions.
        """
        prefix = "A" * prefix_len
        # The total reasoning length is fixed; fill remaining with B's
        body_len = total_len - prefix_len - len("<think>")
        reasoning = prefix + "<think>" + "B" * body_len
        assert len(reasoning) == total_len, "test setup invariant"

        rescue_text = _make_rescue_payload(reasoning)

        resp = ChatCompletionResponse(
            model="qwen3-0.6b-bf16",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=rescue_text,
                        reasoning_content=reasoning,
                        tool_calls=None,
                    ),
                    finish_reason="length",
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=total_len,
                total_tokens=10 + total_len,
            ),
        )
        anth = openai_to_anthropic(resp, "qwen3-0.6b-bf16", reasoning_enabled=True)

        # No partial think fragment in either block on the wire.
        for block in anth.content:
            body = block.thinking if block.type == "thinking" else (block.text or "")
            assert "<think>" not in (body or ""), (
                f"<think> literal leaked into {block.type}: {body!r}"
            )
            assert "</think>" not in (body or ""), (
                f"</think> literal leaked into {block.type}: {body!r}"
            )
            # The partial-fragment failure mode: regex misses the
            # bisected tail / head bytes.
            assert "ink>" not in (body or ""), (
                f"orphan 'ink>' fragment leaked into {block.type} "
                f"(slice bisected '<think>' tag): {body!r}"
            )
            assert "<th" not in (body or ""), (
                f"orphan '<th' fragment leaked into {block.type} "
                f"(slice bisected '<think>' tag): {body!r}"
            )
