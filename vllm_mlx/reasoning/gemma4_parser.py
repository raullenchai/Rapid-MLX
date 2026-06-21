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

        # Extract thought blocks as reasoning
        thought_blocks = _THOUGHT_BLOCK.findall(model_output)
        if not thought_blocks:
            # D-STOP-THINK (cycle-6 F-CORR-2, gemma-4-26b/12b):
            # when ``stop`` matches inside ``<|channel>thought\n...``
            # before the closing ``<channel|>``, the regex above
            # (which requires the closer to match) returns no
            # blocks — and the no-blocks branch then leaks the
            # entire thought trace into ``content`` (only stripping
            # the literal ``<|channel>thought\n`` opener token, not
            # the body). Same shape as the qwen3 / deepseek_r1
            # ``</think>``-never-crossed bug, just expressed in
            # Gemma 4's channel grammar.
            #
            # Detection: ``<|channel>thought`` opener present AND
            # no matching ``<channel|>`` closer downstream. The
            # body from the opener to end-of-text is an unterminated
            # thought trace — route to ``reasoning``, leave content
            # as the pre-opener prefix (typically empty) so the
            # client doesn't see the thought trace in the user-
            # visible answer channel.
            thought_open_idx = model_output.find("<|channel>thought")
            if thought_open_idx >= 0:
                after_opener_idx = thought_open_idx + len("<|channel>thought")
                # Skip the optional newline directly after the opener.
                if (
                    after_opener_idx < len(model_output)
                    and model_output[after_opener_idx] == "\n"
                ):
                    after_opener_idx += 1
                trailing = model_output[after_opener_idx:]
                if "<channel|>" not in trailing:
                    pre_opener = model_output[:thought_open_idx]
                    # Strip any leading content-channel markers from the
                    # pre-opener prefix so the user-visible content
                    # surface stays clean.
                    pre_cleaned = _CONTENT_START.sub("", pre_opener)
                    pre_cleaned = _CHANNEL_END.sub("", pre_cleaned)
                    pre_cleaned = _TURN_END.sub("", pre_cleaned).strip()
                    reasoning_body = trailing.strip()
                    return reasoning_body or None, pre_cleaned or None
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

    def finalize_streaming(
        self,
        accumulated_text: str,
        *,
        matched_stop: str | None = None,
        prompt_thinking_active: bool = False,
        finish_reason: str | None = None,
    ) -> DeltaMessage | None:
        """Handle end of stream — emit any remaining content.

        ``matched_stop``, ``prompt_thinking_active`` and
        ``finish_reason`` are accepted for API symmetry with the
        ``<think>``-family parsers (PR #799 D-STOP-THINK). Gemma4's
        channel-grammar variant has a separate plug in
        ``extract_reasoning`` so this finalize is a no-op.
        """
        return None
