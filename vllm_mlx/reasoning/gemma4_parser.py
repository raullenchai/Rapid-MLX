# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 reasoning parser.

Gemma 4 uses channel tokens for thinking:
  <|channel>thought\n...reasoning...<channel|>
  <|channel>content\n...answer...<channel|>

The parser separates thinking from content by tracking the active channel.
"""

import re

from .base import DeltaMessage, ReasoningParser

# Match full thought blocks in complete text
_THOUGHT_BLOCK = re.compile(r"<\|channel>thought\n[\s\S]*?<channel\|>\s*", re.DOTALL)
# Match content channel markers
_CONTENT_START = re.compile(r"<\|channel>(?:content|final)\n?")
_CHANNEL_END = re.compile(r"<channel\|>")
_TURN_END = re.compile(r"<turn\|>")


class Gemma4ReasoningParser(ReasoningParser):
    """Parser for Gemma 4's channel-based thinking format."""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._in_thought = False
        self._in_content = False
        self._saw_any_channel = False

    def reset_state(self):
        super().reset_state()
        self._in_thought = False
        self._in_content = False
        self._saw_any_channel = False

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning from complete output.

        ``enable_thinking`` accepted for cross-parser signature parity
        (#575); Gemma 4 uses unambiguous ``<|channel|>`` tokens so the
        flag is informational only.
        """
        del enable_thinking  # noqa: F841 — channel parser ignores the flag
        if not model_output:
            return None, model_output

        # r5-D finalize-on-truncation (F-DGF-V080-B-7): when the model
        # is cut mid-thought by ``finish_reason="length"`` the closing
        # ``<channel|>`` sentinel never arrives, so ``_THOUGHT_BLOCK``
        # (which only matches CLOSED blocks) misses the in-progress
        # buffer entirely. Pre-fix the "no thinking tags — all content"
        # fall-through returned the raw scratchpad as ``content`` AND
        # — when the engine's token-level OutputRouter ALSO populated
        # ``engine_reasoning_text`` from the same bytes — the route
        # then duplicated those bytes into ``reasoning_content`` for
        # the desktop client (identical 132/128/512-char repro).
        #
        # Detect via ``is_open_in_think`` (shared finalize-on-truncation
        # contract — see ``base.finalize_truncation``) and route the
        # post-opener body as reasoning instead. Strip the opener
        # marker bytes so the reasoning surface is clean.
        if self.is_open_in_think(model_output):
            last_open = model_output.rfind("<|channel>thought")
            before = model_output[:last_open]
            after = model_output[last_open + len("<|channel>thought") :]
            # Strip the leading newline that follows the opener.
            after = after.lstrip("\n")
            # When ``before`` contains a prior CLOSED ``thought`` block
            # (multi-block truncation shape), surface that as reasoning
            # too so we don't lose the earlier reasoning round; then
            # concatenate the trailing unclosed buffer. When ``before``
            # is empty (the common single-block truncation shape) this
            # collapses to ``(after, None)``. ``content`` is None on
            # truncation — the buffer was inside the think tag at EOS,
            # never the user-visible answer.
            prior_reasoning: str | None = None
            if before:
                prior_thought_blocks = _THOUGHT_BLOCK.findall(before)
                if prior_thought_blocks:
                    parts = []
                    for block in prior_thought_blocks:
                        inner = (
                            block.replace("<|channel>thought\n", "")
                            .replace("<channel|>", "")
                            .strip()
                        )
                        if inner:
                            parts.append(inner)
                    if parts:
                        prior_reasoning = "".join(parts)
            trailing_reasoning = after.strip() or None
            merged_reasoning = (
                "\n".join(p for p in (prior_reasoning, trailing_reasoning) if p) or None
            )
            return merged_reasoning, None

        # Extract thought blocks as reasoning
        thought_blocks = _THOUGHT_BLOCK.findall(model_output)
        if not thought_blocks:
            # No thinking tags — all content
            cleaned = _CONTENT_START.sub("", model_output)
            cleaned = _CHANNEL_END.sub("", cleaned)
            cleaned = _TURN_END.sub("", cleaned).strip()
            return None, cleaned

        # Reasoning = thought block contents (strip markers)
        reasoning = ""
        for block in thought_blocks:
            inner = (
                block.replace("<|channel>thought\n", "")
                .replace("<channel|>", "")
                .strip()
            )
            reasoning += inner

        # Content = everything after thought blocks, strip markers
        content = _THOUGHT_BLOCK.sub("", model_output)
        content = _CONTENT_START.sub("", content)
        content = _CHANNEL_END.sub("", content)
        content = _TURN_END.sub("", content).strip()

        return reasoning or None, content or None

    def extract_reasoning_streaming(
        self, previous_text: str, current_text: str, delta_text: str
    ) -> DeltaMessage | None:
        """Extract reasoning from streaming delta."""
        if not delta_text:
            return None

        # Snapshot pre-update state so we can detect a thought-to-content
        # flip that happened DURING this delta (issue #219).
        was_in_thought = self._in_thought

        # Track channel state based on accumulated text
        # Check if we just entered thought channel
        if "<|channel>thought" in current_text and not self._in_content:
            self._in_thought = True
            self._saw_any_channel = True

        # Check if we just entered content channel
        if "<|channel>content" in current_text or "<|channel>final" in current_text:
            self._in_thought = False
            self._in_content = True

        # Check if thought ended (first <channel|> after thought start)
        if self._in_thought and "<channel|>" in current_text:
            thought_starts = current_text.count("<|channel>thought")
            channel_ends = current_text.count("<channel|>")
            if channel_ends >= thought_starts:
                self._in_thought = False
                # If no explicit content channel follows, switch to content mode
                if (
                    "<|channel>content" not in current_text
                    and "<|channel>final" not in current_text
                ):
                    self._in_content = True

        # If a thought-to-content flip happened DURING this delta and a
        # state-flipping marker is fully visible in delta_text, split the
        # delta so reasoning bytes that arrived before the marker stay in
        # delta.reasoning instead of being misrouted into delta.content.
        # Pre-fix (#219), the whole-delta classifier below tagged the entire
        # delta as content whenever should_send() flushed a buffered delta
        # straddling the channel transition.
        if was_in_thought and not self._in_thought:
            flip_pos = -1
            for marker in ("<channel|>", "<|channel>content", "<|channel>final"):
                idx = delta_text.find(marker)
                if idx >= 0 and (flip_pos < 0 or idx < flip_pos):
                    flip_pos = idx
            if flip_pos >= 0:
                pre = delta_text[:flip_pos]
                post = delta_text[flip_pos:]
                for m in (
                    "<|channel>",
                    "<channel|>",
                    "<|turn>",
                    "<turn|>",
                    "thought\n",
                    "content\n",
                    "final\n",
                ):
                    pre = pre.replace(m, "")
                    post = post.replace(m, "")
                if pre or post:
                    return DeltaMessage(
                        reasoning=pre if pre else None,
                        content=post if post else None,
                    )

        # Filter out channel markers from delta
        clean = delta_text
        for marker in [
            "<|channel>",
            "<channel|>",
            "<|turn>",
            "<turn|>",
            "thought\n",
            "content\n",
            "final\n",
        ]:
            clean = clean.replace(marker, "")

        if not clean:
            return None  # pure marker token, skip

        if self._in_thought:
            return DeltaMessage(reasoning=clean)
        elif self._in_content:
            return DeltaMessage(content=clean)
        elif not self._saw_any_channel:
            # No channel tokens seen — plain content (no thinking)
            return DeltaMessage(content=clean)
        else:
            # Between channels — treat as reasoning
            return DeltaMessage(reasoning=clean)

    def finalize_streaming(self, accumulated_text: str) -> DeltaMessage | None:
        """Handle end of stream — emit any remaining content."""
        return None

    def is_open_in_think(self, accumulated_text: str) -> bool:
        """r5-D — gemma4 unclosed-thought-channel detection.

        Gemma 4's channel grammar opens with ``<|channel>thought\\n``
        and closes the channel with ``<channel|>``. When
        ``finish_reason="length"`` truncates before the closer arrives
        (max_tokens cut mid-thought), the non-streaming first-pass
        ``extract_reasoning`` runs ``_THOUGHT_BLOCK.findall`` which
        only matches CLOSED thought blocks, finds nothing, and falls
        through to the "all content" branch — leaking the raw
        scratchpad bytes into ``content`` and (when the engine's
        token-level OutputRouter has ALSO populated
        ``engine_reasoning_text``) duplicating the same bytes into
        both ``content`` and ``reasoning_content`` for the desktop
        client (the F-DGF-V080-B-7 132/128/512-char identical-dup
        repro).

        Open-in-think signal:

        * Saw an opener (``<|channel>thought``) AND
        * No closer (``<channel|>``) appears AFTER the latest opener.

        Multi-block resilience: ``rfind`` on the opener and ``find``
        past that position handles both single-block and multi-block
        truncation shapes (closed thought → content channel → second
        thought opener with no closer still reads open-in-think).
        """
        if not accumulated_text:
            return False
        # Codex r1 BLOCKING on PR #825: the pre-fix ``rfind`` gate
        # mis-fired on a legitimate answer that mentions the literal
        # ``<|channel>thought`` substring inside an already-opened
        # content/final channel — ``rfind`` lands on the literal
        # substring inside the answer, the tail is short, and the
        # closer-not-found check spuriously returns True. The
        # extracted ``cleaned_text`` path also tripped the gate
        # because the upstream content opener had already been
        # stripped by ``extract_reasoning`` before the route probed
        # the buffer.
        #
        # Systematic fix — three conservative guards:
        # 1. Buffer must START (modulo whitespace) with
        #    ``<|channel>thought``. This is the only shape where the
        #    EOS-was-mid-think interpretation is unambiguous,
        #    matching the minimax ``lstrip().startswith("<think>")``
        #    guard and the ``_sweep_residual_think_tags`` scope from
        #    PR #722 codex r3.
        # 2. No ``<|channel>content`` / ``<|channel>final`` opener
        #    appears — once content opens, any further
        #    ``<|channel>thought`` bytes are literal answer text in
        #    Gemma 4's single-round grammar.
        # 3. No ``<channel|>`` close after the (only) opener.
        if not accumulated_text.lstrip().startswith("<|channel>thought"):
            return False
        if (
            "<|channel>content" in accumulated_text
            or "<|channel>final" in accumulated_text
        ):
            return False
        last_open = accumulated_text.rfind("<|channel>thought")
        if last_open < 0:
            return False
        return "<channel|>" not in accumulated_text[last_open:]
