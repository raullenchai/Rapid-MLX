# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for Qwen3 models.

Qwen3 uses <think>...</think> tags for reasoning content and supports
a strict switch via 'enable_thinking=False' in chat template kwargs.

Supports implicit reasoning mode where <think> is injected in the prompt
by AI agents (e.g., OpenCode) and only </think> appears in the output.
"""

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Qwen3 models.

    Qwen3 uses <think>...</think> tokens to denote reasoning text.

    Supports three scenarios:
    1. Both tags in output: <think>reasoning</think>content
    2. Only closing tag (think in prompt): reasoning</think>content
    3. No tags: pure content

    Example (normal):
        Input: "<think>Let me analyze this...</think>The answer is 42."
        Output: reasoning="Let me analyze this...", content="The answer is 42."

    Example (think in prompt):
        Input: "Let me analyze this...</think>The answer is 42."
        Output: reasoning="Let me analyze this...", content="The answer is 42."
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
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from Qwen3 output.

        Handles both explicit <think>...</think> tags and implicit mode
        where <think> was in the prompt (only </think> in output).

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple.
        """
        # If no end token at all:
        if self.end_token not in model_output:
            # If start token is present, model started thinking but never finished
            # (truncated by max_tokens or garbled by high temperature).
            # Treat everything after <think> as reasoning, content is None.
            if self.start_token in model_output:
                _, _, reasoning = model_output.partition(self.start_token)
                return reasoning.strip() or None, None
            # No think tags at all — pure content
            return None, model_output

        # Use base class implementation (handles both explicit and implicit)
        return super().extract_reasoning(model_output)

    def finalize_streaming(self, accumulated_text: str) -> DeltaMessage | None:
        """
        Finalize streaming output.

        Three cases:

        1. No tags seen at all — base class classified everything as reasoning
           (to support implicit think). Emit correction with full text.

        2. <think> seen (template injected or model generated) but </think>
           never appeared — model never produced the closing tag. The base
           class classified everything as reasoning. Emit correction with
           full text (stripping the template-injected <think> prefix).

        3. </think> seen — reasoning was properly completed. Either the model
           produced content after </think> (already emitted as text_delta), or
           the stream ended right at </think>. No correction needed.

        Cases 1 and 2 fix a regression in the Anthropic streaming adapter
        (#185 follow-on): when the chat template injects <think> as a prefix,
        _saw_any_tag is set True from the first delta, preventing the original
        no-tags correction. Checking for </think> presence directly handles
        both the template-injected and genuinely-no-thought scenarios.
        """
        if self.end_token in accumulated_text:
            # Case 3: proper close tag seen — no correction
            return None
        if accumulated_text:
            # Cases 1 & 2: no close tag — emit full text as content,
            # stripping the template-injected ``<think>`` prefix if present.
            cleaned = accumulated_text
            if cleaned.startswith(self.start_token):
                cleaned = cleaned[len(self.start_token) :]
            if cleaned:
                return DeltaMessage(content=cleaned)
        return None
