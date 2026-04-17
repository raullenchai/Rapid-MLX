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

        If no tags were ever seen, the base class classified everything as
        reasoning (to support implicit think mode where <think> is in the
        prompt). But if </think> never appeared either, the model simply
        doesn't use think tags — all output should have been content.

        Emit a correction chunk with the full text as content. The server
        emits this as a final SSE chunk so the client receives the complete
        response.

        This is safe because:
        - Explicit think (<think>...<think>content): _saw_any_tag is True, no correction
        - Implicit think (reasoning</think>content): _saw_any_tag is True, no correction
        - No tags at all: _saw_any_tag is False, correction emitted
        """
        if not self._saw_any_tag and accumulated_text:
            return DeltaMessage(content=accumulated_text)
        return None
