# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for DeepSeek-R1 models.

DeepSeek-R1 uses <think>...</think> tags for reasoning content.
The model may sometimes start outputting reasoning without the explicit
<think> tag, so this parser is more lenient than Qwen3.
"""

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser


class DeepSeekR1ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for DeepSeek-R1 model.

    DeepSeek-R1 uses <think>...</think> tokens to denote reasoning text.
    This parser is more lenient than Qwen3:
    - The <think> tag may not be explicitly generated (model assumes it)
    - If only </think> is found, everything before it is reasoning

    Example:
        Input: "<think>Step 1: analyze...\nStep 2: solve...</think>The answer is 42."
        Output: reasoning="Step 1: analyze...\nStep 2: solve...", content="The answer is 42."

        Input: "reasoning content</think>final answer"  # No opening tag
        Output: reasoning="reasoning content", content="final answer"
    """

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from DeepSeek-R1 output.

        More lenient than Qwen3 - handles cases where start tag is implicit.

        Args:
            model_output: Complete model output text.
            enable_thinking: Threaded through to ``BaseThinkingReasoningParser``
                Case 4 — when True, no-tag output is routed to reasoning
                (#575 symmetric-with-streaming fallback). DeepSeek-R1
                callers rarely set this explicitly; the no-tag branch
                below short-circuits before the base call, so the flag
                only matters if a future caller wires it on.

        Returns:
            (reasoning, content) tuple.
        """
        # If we have end token but no start token, treat beginning as reasoning
        if self.end_token in model_output and self.start_token not in model_output:
            reasoning, _, content = model_output.partition(self.end_token)
            reasoning = reasoning.strip() or None
            content = content.strip() or None
            return reasoning, content

        # If neither token, return as pure content — UNLESS the caller
        # explicitly set enable_thinking=True, in which case the chat
        # template injected ``<think>`` into the prompt and a truncated
        # response with no tags is the model's continued thought trace.
        # See ``BaseThinkingReasoningParser.extract_reasoning`` for the
        # full rationale (#575).
        if self.end_token not in model_output and self.start_token not in model_output:
            if enable_thinking is True:
                return model_output.strip() or None, None
            return None, model_output

        # Use base class for standard case
        return super().extract_reasoning(model_output, enable_thinking=enable_thinking)

    # Character threshold for no-tag content detection.
    # If no think tags are seen after this many characters, treat output as
    # content rather than reasoning. Real reasoning models emit <think> within
    # the first few tokens; 64 chars (~15-20 tokens) is a safe threshold.
    NO_TAG_CONTENT_THRESHOLD = 64

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta.

        Handles DeepSeek-R1's pattern where <think> may be implicit.
        If no think tags are seen after NO_TAG_CONTENT_THRESHOLD characters,
        treats output as content to avoid misclassifying non-reasoning output.

        Args:
            previous_text: Text accumulated before this delta.
            current_text: Text including this delta.
            delta_text: Just the new text.

        Returns:
            DeltaMessage with reasoning/content, or None to skip.
        """
        # Check if any tags are in the current text
        has_tags = self.start_token in current_text or self.end_token in current_text

        # No tags seen yet and past threshold → treat as content
        if not has_tags and not self._saw_any_tag:
            if len(current_text) >= self.NO_TAG_CONTENT_THRESHOLD:
                return DeltaMessage(content=delta_text)
            # Under threshold: delegate to base (defaults to reasoning
            # for early implicit mode, will be corrected by finalize)

        # First try base class logic
        result = super().extract_reasoning_streaming(
            previous_text, current_text, delta_text
        )

        # Handle DeepSeek-R1 special case: no start token seen but end token appears
        if result is not None:
            start_in_prev = self.start_token in previous_text
            start_in_delta = self.start_token in delta_text
            end_in_delta = self.end_token in delta_text

            # If end token in delta but we never saw start token
            if not start_in_prev and not start_in_delta and end_in_delta:
                # Everything before end token is reasoning
                idx = delta_text.find(self.end_token)
                reasoning_part = delta_text[:idx]
                content_part = delta_text[idx + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None,
                    content=content_part if content_part else None,
                )

        return result

    def finalize_streaming(self, accumulated_text: str) -> DeltaMessage | None:
        """
        Finalize streaming output.

        If no tags were ever seen and the output was short (under threshold),
        the base class would have classified it all as reasoning. Emit a
        correction to reclassify as content.

        Args:
            accumulated_text: Complete accumulated text from stream.

        Returns:
            DeltaMessage correction, or None if no correction needed.
        """
        if (
            not self._saw_any_tag
            and accumulated_text
            and len(accumulated_text) < self.NO_TAG_CONTENT_THRESHOLD
        ):
            # Short no-tag output was misclassified as reasoning.
            # Return correction: emit as content. The caller should
            # yield a chunk that moves reasoning → content.
            return DeltaMessage(content=accumulated_text)
        return None
