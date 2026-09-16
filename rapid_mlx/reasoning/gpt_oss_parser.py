# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for GPT-OSS models using channel-based format.

GPT-OSS models use a channel-based token format instead of <think>...</think> tags:
    <|channel|>analysis<|message|>[reasoning]<|start|>assistant<|channel|>final<|message|>[content]<|return|>

Some models also emit an extended format with a constrain token:
    <|channel|>final <|constrain|>JSON<|message|>[content]<|return|>

This parser extracts reasoning from the 'analysis' channel and content from
the 'final' channel, stripping all structural tokens from API responses.
"""

import re

from .base import DeltaMessage, ReasoningParser

# Structural tokens that should be stripped from output
_STRUCTURAL_TOKENS = re.compile(
    r"<\|start\|>|<\|end\|>|<\|channel\|>|<\|return\|>|<\|call\|>|<\|constrain\|>"
)

# Flexible channel marker regex — matches both standard and extended formats:
#   <|channel|>analysis<|message|>
#   <|channel|>final<|message|>
#   <|channel|>final <|constrain|>JSON<|message|>
_CHANNEL_RE = re.compile(
    r"<\|channel\|>(analysis|final)(?:[^<]*(?:<\|constrain\|>[^<]*)?)?<\|message\|>"
)


def _extract_channel(text: str, channel_name: str) -> str | None:
    """
    Extract content from a named channel.

    Finds <|channel|>{name}...<|message|> (with optional constrain token)
    and extracts text up to the next structural token or end of string.

    Args:
        text: Full model output text.
        channel_name: Channel name to extract (e.g., "analysis", "final").

    Returns:
        Extracted channel content, or None if channel not found.
    """
    for m in _CHANNEL_RE.finditer(text):
        if m.group(1) == channel_name:
            start = m.end()
            # Find next structural token after message content
            end_match = _STRUCTURAL_TOKENS.search(text, start)
            content = text[start : end_match.start()] if end_match else text[start:]
            content = content.strip()
            return content if content else None
    return None


class GptOssReasoningParser(ReasoningParser):
    """
    Reasoning parser for GPT-OSS models.

    GPT-OSS uses channel-based tokens:
        <|channel|>analysis<|message|>[reasoning]
        <|start|>assistant<|channel|>final<|message|>[content]<|return|>

    The 'analysis' channel maps to reasoning, 'final' to content.

    Also handles extended format with constrain token:
        <|channel|>final <|constrain|>JSON<|message|>[content]<|return|>
    """

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning and content from complete model output.

        Args:
            model_output: Complete text output from the model.
            enable_thinking: Accepted for cross-parser signature parity
                (#575). GPT-OSS uses unambiguous ``<|channel|>`` tokens,
                so the flag is informational only — no branching here.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        del enable_thinking  # noqa: F841 — channel parser ignores the flag
        if not model_output or "<|channel|>" not in model_output:
            return None, model_output if model_output else None

        reasoning = _extract_channel(model_output, "analysis")
        content = _extract_channel(model_output, "final")

        # Strip trailing <|return|>
        if content:
            content = content.replace("<|return|>", "").strip()
            content = _STRUCTURAL_TOKENS.sub("", content).strip()
            content = content if content else None

        # Strip any remaining structural tokens from reasoning
        if reasoning:
            reasoning = _STRUCTURAL_TOKENS.sub("", reasoning).strip()
            reasoning = reasoning if reasoning else None

        # If no channels found, return as plain content
        if reasoning is None and content is None:
            return None, model_output

        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta.

        Uses stateless phase detection from current_text on each call.

        Args:
            previous_text: Accumulated text before this delta.
            current_text: Accumulated text including this delta.
            delta_text: Just the new text in this streaming chunk.

        Returns:
            DeltaMessage with reasoning and/or content, or None to skip.
        """
        prev_phase = self._detect_phase(previous_text)
        curr_phase = self._detect_phase(current_text)

        # Phase changed — extract content after the new marker
        if curr_phase != prev_phase and curr_phase in ("analysis", "final"):
            after_marker = self._extract_content_after_marker_in_delta(
                current_text, curr_phase
            )
            if after_marker:
                after_marker = self._strip_return(after_marker)
                if curr_phase == "analysis":
                    return DeltaMessage(reasoning=after_marker)
                else:
                    return DeltaMessage(content=after_marker)
            return None

        # In a steady phase — emit delta directly
        if curr_phase == "analysis":
            cleaned = self._strip_return(delta_text)
            # Skip structural tokens in the delta
            if _STRUCTURAL_TOKENS.search(cleaned):
                cleaned = _STRUCTURAL_TOKENS.sub("", cleaned)
            if cleaned:
                return DeltaMessage(reasoning=cleaned)
            return None
        elif curr_phase == "final":
            cleaned = self._strip_return(delta_text)
            if _STRUCTURAL_TOKENS.search(cleaned):
                cleaned = _STRUCTURAL_TOKENS.sub("", cleaned)
            if cleaned:
                return DeltaMessage(content=cleaned)
            return None

        # init or transition phase — skip structural tokens
        return None

    @staticmethod
    def _detect_phase(text: str) -> str:
        """
        Detect current streaming phase from accumulated text.

        Returns:
            "final"      — final channel marker complete
            "analysis"   — analysis marker complete, no structural token after
            "transition" — analysis present but structural token follows
            "init"       — no channel marker yet
        """
        # Find all channel markers in text
        matches = list(_CHANNEL_RE.finditer(text))
        if not matches:
            return "init"

        last = matches[-1]
        if last.group(1) == "final":
            return "final"

        # analysis channel found — check if there's a structural token after
        after = text[last.end() :]
        if _STRUCTURAL_TOKENS.search(after):
            return "transition"
        return "analysis"

    @staticmethod
    def _extract_content_after_marker_in_delta(
        current_text: str, phase: str
    ) -> str | None:
        """
        When phase changes, extract only the content after the phase marker
        that falls within the current accumulated text's tail.

        Args:
            current_text: Full accumulated text.
            phase: Current phase ("analysis" or "final").

        Returns:
            Content after the marker, or None.
        """
        channel_name = "analysis" if phase == "analysis" else "final"
        matches = list(_CHANNEL_RE.finditer(current_text))
        for m in reversed(matches):
            if m.group(1) == channel_name:
                return current_text[m.end() :]
        return None

    @staticmethod
    def _strip_return(text: str) -> str:
        """Strip <|return|> from text."""
        return text.replace("<|return|>", "")
