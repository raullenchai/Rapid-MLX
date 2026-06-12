# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for Mistral / Ministral reasoning models.

Models such as Magistral and Ministral-3-*-Reasoning wrap their reasoning in
[THINK]...[/THINK] delimiters (registered as special tokens in the tokenizer)
and support a strict switch via 'enable_thinking=False' in chat template kwargs.

This mirrors the Qwen3 parser, which uses <think>...</think>, but with the
Mistral bracket-token delimiters.

Supports implicit reasoning mode where [THINK] is injected in the prompt and
only [/THINK] appears in the output.
"""

from .think_parser import BaseThinkingReasoningParser


class MistralReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Mistral/Ministral reasoning models.

    Uses [THINK]...[/THINK] tokens to denote reasoning text.

    Supports three scenarios:
    1. Both tags in output: [THINK]reasoning[/THINK]content
    2. Only closing tag (think in prompt): reasoning[/THINK]content
    3. No tags: pure content

    Example (normal):
        Input: "[THINK]Let me analyze this...[/THINK]The answer is 42."
        Output: reasoning="Let me analyze this...", content="The answer is 42."

    Example (think in prompt):
        Input: "Let me analyze this...[/THINK]The answer is 42."
        Output: reasoning="Let me analyze this...", content="The answer is 42."
    """

    @property
    def start_token(self) -> str:
        return "[THINK]"

    @property
    def end_token(self) -> str:
        return "[/THINK]"

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from Mistral/Ministral output.

        Handles both explicit [THINK]...[/THINK] tags and implicit mode
        where [THINK] was in the prompt (only [/THINK] in output).

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple.
        """
        # If no end token at all, treat as pure content
        if self.end_token not in model_output:
            return None, model_output

        # Use base class implementation (handles both explicit and implicit)
        return super().extract_reasoning(model_output)
