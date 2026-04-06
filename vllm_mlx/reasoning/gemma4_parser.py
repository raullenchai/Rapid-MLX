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
_THOUGHT_BLOCK = re.compile(
    r"<\|channel>thought\n[\s\S]*?<channel\|>\s*", re.DOTALL
)
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

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        """Extract reasoning from complete output."""
        if not model_output:
            return None, model_output

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
            inner = block.replace("<|channel>thought\n", "").replace("<channel|>", "").strip()
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
                if "<|channel>content" not in current_text and "<|channel>final" not in current_text:
                    self._in_content = True

        # Filter out channel markers from delta
        clean = delta_text
        for marker in ["<|channel>", "<channel|>", "<|turn>", "<turn|>",
                        "thought\n", "content\n", "final\n"]:
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
