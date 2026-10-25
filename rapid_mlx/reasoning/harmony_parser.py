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

from ..streaming_markers import StreamingMarkerBuffer
from .base import DeltaMessage, ReasoningParser

# Analysis channel blocks: <|channel|>analysis<|message|>...<|end|>
_ANALYSIS_PATTERN = re.compile(
    r"<\|channel\|>analysis\s*<\|message\|>(.*?)<\|end\|>",
    re.DOTALL,
)

# Final channel content. Harmony spec uses ``<|return|>`` to terminate the
# final channel, but gpt-oss-20b-mxfp4-q8 emits ``<|end|>`` in practice for a sizeable
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
# Greedy ``(.*)`` so a literal ``<|end|>`` inside answer text is
# consumed and we stop at the LAST ``<|end|>`` — the real
# end-of-message marker. Combined with the
# ``_FINAL_PATTERN_RETURN``-first preference above, this covers both
# terminator paths (``<|return|>``-terminated outputs use the
# preferred regex; ``<|end|>``-only outputs use this greedy
# fallback).
_FINAL_PATTERN_END = re.compile(
    r"<\|channel\|>final\s*<\|message\|>(.*)<\|end\|>",
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
        self._stream_buffer = ""
        self._markers = StreamingMarkerBuffer(
            (
                "<|start|>",
                "<|channel|>",
                "<|message|>",
                "<|end|>",
                "<|return|>",
                "<|call|>",
            )
        )

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete Harmony output.

        Collects all analysis channel blocks as reasoning and the
        final channel block as content.

        Args:
            model_output: Complete model output text.
            enable_thinking: Accepted for cross-parser signature parity
                (#575). Harmony uses unambiguous channel tokens, so the
                flag is informational only.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        del enable_thinking  # noqa: F841 — channel parser ignores the flag
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
        self._stream_buffer += self._markers.process(delta_text)
        reasoning: list[str] = []
        content: list[str] = []
        while self._stream_buffer:
            text = self._stream_buffer
            if text.startswith(("<|start|>", "<|channel|>")):
                end = text.find("<|message|>")
                if end < 0:
                    channel = re.search(
                        r"<\|channel\|>(analysis|final|commentary)(?:\s|$)", text
                    )
                    if channel:
                        self._current_channel = channel[1]
                    break
                header = text[:end]
                channel = header.split("<|channel|>")[-1].split()
                self._current_channel = channel[0] if channel else None
                self._in_message = True
                self._stream_buffer = text[end + len("<|message|>") :]
                if self._current_channel == "commentary" and re.search(
                    r"\bto=", header
                ):
                    # Tool recipients need the complete envelope downstream;
                    # plain commentary contributes only its body as content.
                    content.append(text[: end + len("<|message|>")])
                continue
            match = re.search(r"<\|(?:start|channel|message|end|return|call)\|>", text)
            end = match.start() if match else len(text)
            body = text[:end]
            if self._current_channel == "analysis" and self._in_message:
                reasoning.append(body)
            elif self._current_channel in {"final", "commentary"} and self._in_message:
                content.append(body)
            self._stream_buffer = text[end:]
            if not match:
                break
            token = match[0]
            if token in {"<|start|>", "<|channel|>"}:
                continue
            if self._current_channel == "commentary":
                content.append(token)
            self._in_message = token == "<|message|>"
            self._stream_buffer = self._stream_buffer[len(token) :]
        if any(reasoning) or any(content):
            return DeltaMessage(
                reasoning="".join(reasoning) or None, content="".join(content) or None
            )
        return None

    def finalize_streaming(
        self, accumulated_text: str, **kwargs
    ) -> DeltaMessage | None:
        """Drain an ordinary body suffix without publishing an unfinished header."""
        tail = self._markers.flush()
        if tail == "<" and not self._stream_buffer and self._in_message:
            if self._current_channel == "analysis":
                return DeltaMessage(reasoning=tail)
            if self._current_channel in {"final", "commentary"}:
                return DeltaMessage(content=tail)
        return None

    def reset_state(self):
        """Reset streaming state for a new request."""
        self._current_channel = None
        self._in_message = False
        self._stream_buffer = ""
        self._markers.flush()
