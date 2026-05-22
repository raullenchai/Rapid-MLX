# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for GPT-OSS models using Harmony format.

Harmony uses channels for reasoning vs final content:

    <|channel|>analysis
    <|message|>Let me think about this...
    <|end|>
    <|channel|>final
    <|message|>The answer is 42.
    <|return|>

The analysis channel contains reasoning, and the final channel
contains the user-facing response.
"""

import re

from .base import DeltaMessage, ReasoningParser

# Analysis channel blocks: <|channel|>analysis<|message|>...<|end|>
_ANALYSIS_PATTERN = re.compile(
    r"<\|channel\|>analysis\s*<\|message\|>(.*?)<\|end\|>",
    re.DOTALL,
)

# Final channel content. Harmony spec uses ``<|return|>`` to terminate the
# final channel, but gpt-oss-20b emits ``<|end|>`` in practice for a sizeable
# fraction of non-streaming responses (observed in v0.6.64 pr_validate runs:
# anthropic_sdk 0/5, langchain 2/6, pydantic_ai 1/6 on
# ``mlx-community/gpt-oss-20b-MXFP4-Q8`` — every non-streaming test landed
# here). Accept either terminator so the regex matches the same set of
# completions that ``HarmonyToolParser._FINAL_BLOCK_PATTERN`` already
# accepts and that the streaming parser already handles via the
# ``<|end|>``/``<|return|>`` end-of-message check. Without this, the
# non-streaming path returns ``content=None`` and the chat response
# emits an empty TextBlock for what was actually a fully-formed answer.
# Prefer ``<|return|>`` over ``<|end|>``: if both appear, the model has
# definitively terminated the message with ``<|return|>``. Trying that
# pattern first avoids truncating answer text that happens to contain
# a literal ``<|end|>`` (e.g. a transcript of a harmony exchange).
_FINAL_PATTERN_RETURN = re.compile(
    r"<\|channel\|>final\s*<\|message\|>(.*?)<\|return\|>",
    re.DOTALL,
)
_FINAL_PATTERN_END = re.compile(
    r"<\|channel\|>final\s*<\|message\|>(.*?)<\|end\|>",
    re.DOTALL,
)


class HarmonyReasoningParser(ReasoningParser):
    """
    Reasoning parser for GPT-OSS models using Harmony format.

    Extracts reasoning from the 'analysis' channel and content from
    the 'final' channel. Commentary channels (tool calls) are ignored
    since they are handled by the tool parser.

    Example:
        Input: "<|channel|>analysis<|message|>Thinking...<|end|>
                <|channel|>final<|message|>Result.<|return|>"
        Output: reasoning="Thinking...", content="Result."
    """

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._current_channel: str | None = None
        self._in_message: bool = False

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete Harmony output.

        Collects all analysis channel blocks as reasoning and the
        final channel block as content.

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        # Collect all analysis blocks
        analysis_blocks = _ANALYSIS_PATTERN.findall(model_output)
        reasoning = "\n".join(block.strip() for block in analysis_blocks) or None

        # Extract final channel content. Prefer ``<|return|>`` over
        # ``<|end|>`` so a literal ``<|end|>`` in answer text does not
        # truncate a ``<|return|>``-terminated message.
        final_match = _FINAL_PATTERN_RETURN.search(
            model_output
        ) or _FINAL_PATTERN_END.search(model_output)
        content = final_match.group(1).strip() if final_match else None

        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming Harmony output.

        Tracks the current channel and emits reasoning deltas for
        analysis channel content and content deltas for final channel.

        Args:
            previous_text: Accumulated text before this delta.
            current_text: Accumulated text including this delta.
            delta_text: The new text in this streaming chunk.

        Returns:
            DeltaMessage with reasoning and/or content, or None.
        """
        # Detect channel switches in the delta
        if "<|channel|>" in delta_text:
            if "analysis" in delta_text:
                self._current_channel = "analysis"
                self._in_message = False
                return None
            elif "final" in delta_text:
                self._current_channel = "final"
                self._in_message = False
                return None
            elif "commentary" in delta_text:
                self._current_channel = "commentary"
                self._in_message = False
                # Pass through so tool parser can see the channel marker
                return DeltaMessage(content=delta_text)

        # Detect channel from full context if not yet determined
        if self._current_channel is None and "<|channel|>" in current_text:
            last_channel = current_text.rfind("<|channel|>")
            after = current_text[last_channel + len("<|channel|>") :]
            if after.startswith("analysis"):
                self._current_channel = "analysis"
            elif after.startswith("final"):
                self._current_channel = "final"
            elif after.startswith("commentary"):
                self._current_channel = "commentary"

        # Commentary channel: pass everything through as content
        # so the tool parser can accumulate and detect tool calls.
        if self._current_channel == "commentary":
            return DeltaMessage(content=delta_text)

        # Handle message start (analysis/final channels only)
        if "<|message|>" in delta_text:
            self._in_message = True
            # Don't emit the token itself
            return None

        # Handle channel/message end tokens
        if any(
            token in delta_text
            for token in ("<|end|>", "<|return|>", "<|call|>", "<|start|>")
        ):
            self._in_message = False
            return None

        # Skip control tokens
        if delta_text.strip().startswith("<|") and delta_text.strip().endswith("|>"):
            return None

        # Emit content based on current channel
        if self._in_message and self._current_channel == "analysis":
            return DeltaMessage(reasoning=delta_text)

        if self._in_message and self._current_channel == "final":
            return DeltaMessage(content=delta_text)

        # Unknown channel, suppress
        return None

    def reset_state(self):
        """Reset streaming state for a new request."""
        self._current_channel = None
        self._in_message = False
