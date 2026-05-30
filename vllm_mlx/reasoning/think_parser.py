# SPDX-License-Identifier: Apache-2.0
"""
Base parser for models using <think>...</think> tags for reasoning.

This module provides BaseThinkingReasoningParser, a concrete implementation
for extracting reasoning content from models that use thinking tags.

Supports three scenarios:
1. Both tags in output: <think>reasoning</think>content
2. Only closing tag (think injected in prompt): reasoning</think>content
3. No tags: pure content
"""

from abc import abstractmethod

from .base import DeltaMessage, ReasoningParser


class BaseThinkingReasoningParser(ReasoningParser):
    """
    Base parser for models using <think>...</think> style tags.

    This parser handles the common pattern where reasoning content is wrapped
    in special tags. Subclasses define the specific start and end tokens.

    Supports "implicit reasoning mode" where <think> is injected in the prompt
    and only </think> appears in the model output. This is common with AI agents
    like OpenCode that force models to reason by injecting thinking tags.

    The parser tracks state during streaming to correctly separate reasoning
    from content as tokens arrive incrementally.
    """

    @property
    @abstractmethod
    def start_token(self) -> str:
        """The token/tag that starts reasoning content (e.g., '<think>')."""

    @property
    @abstractmethod
    def end_token(self) -> str:
        """The token/tag that ends reasoning content (e.g., '</think>')."""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._saw_any_tag = False

    def reset_state(self):
        """Reset state for a new streaming request."""
        super().reset_state()
        self._saw_any_tag = False

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete output.

        Handles three cases:
        1. Both tags present: <think>reasoning</think>content
        2. Only closing tag: reasoning</think>content (think in prompt)
        3. No tags: pure content

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        text = model_output

        # Case 1: Both tags present (normal case)
        if self.start_token in text and self.end_token in text:
            # Get everything after start token
            _, _, after_start = text.partition(self.start_token)
            # Split on end token
            reasoning, _, content = after_start.partition(self.end_token)
            return reasoning.strip() or None, content.strip() or None

        # Case 2: Only closing tag (think was injected in prompt)
        # Everything before </think> is reasoning
        if self.end_token in text:
            reasoning, _, content = text.partition(self.end_token)
            return reasoning.strip() or None, content.strip() or None

        # Case 3: Only start tag (incomplete reasoning, no end yet)
        if self.start_token in text:
            _, _, reasoning = text.partition(self.start_token)
            return reasoning.strip() or None, None

        # Case 4: No tags at all - pure content
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta using text-based detection.

        Handles implicit reasoning mode where <think> was in the prompt
        and only </think> appears in the output.

        Args:
            previous_text: Text accumulated before this delta.
            current_text: Text including this delta.
            delta_text: Just the new text.

        Returns:
            DeltaMessage with reasoning/content, or None to skip.
        """
        # Skip if delta is just the special tokens themselves
        stripped_delta = delta_text.strip()
        if stripped_delta == self.start_token:
            return None
        if stripped_delta == self.end_token:
            return None

        # Check token positions in text (stateless text-based detection)
        start_in_prev = self.start_token in previous_text
        start_in_current = self.start_token in current_text
        end_in_prev = self.end_token in previous_text
        end_in_delta = self.end_token in delta_text

        # Case 1: Explicit <think> found in text - standard behavior
        if start_in_current:
            self._saw_any_tag = True
            return self._handle_explicit_think(
                previous_text, delta_text, start_in_prev, end_in_prev, end_in_delta
            )

        # Case 2: No <think> but </think> found - implicit reasoning mode
        # This handles when <think> was injected in the prompt
        if self.end_token in current_text:
            self._saw_any_tag = True
            return self._handle_implicit_think(delta_text, end_in_prev, end_in_delta)

        # Case 3: No think tags seen yet
        # We can't know if <think> was in the prompt, so we must make a choice:
        # - Treat as content (safe, but loses reasoning if think was in prompt)
        # - Treat as reasoning (risky, wrong if no thinking at all)
        # We choose to treat as reasoning IF we haven't seen </think> yet,
        # because if think was in prompt, we want to capture the reasoning.
        # This will be corrected once </think> is seen.
        return DeltaMessage(reasoning=delta_text)

    def _handle_explicit_think(
        self,
        previous_text: str,
        delta_text: str,
        start_in_prev: bool,
        end_in_prev: bool,
        end_in_delta: bool,
    ) -> DeltaMessage | None:
        """Handle case where <think> tag is explicitly in the output."""
        start_in_delta = self.start_token in delta_text

        if start_in_prev:
            # We're after the start token
            if end_in_delta:
                # Transition: end token in this delta
                idx = delta_text.find(self.end_token)
                reasoning_part = delta_text[:idx]
                content_part = delta_text[idx + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None,
                    content=content_part if content_part else None,
                )
            elif end_in_prev:
                # Already past reasoning phase - pure content
                return DeltaMessage(content=delta_text)
            else:
                # Still in reasoning phase
                return DeltaMessage(reasoning=delta_text)

        elif start_in_delta:
            # Start token is in this delta
            start_idx = delta_text.find(self.start_token)

            if end_in_delta:
                # Both tokens in this delta
                end_idx = delta_text.find(self.end_token)
                reasoning_part = delta_text[start_idx + len(self.start_token) : end_idx]
                content_part = delta_text[end_idx + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None,
                    content=content_part if content_part else None,
                )
            else:
                # Only start token - beginning of reasoning
                reasoning_part = delta_text[start_idx + len(self.start_token) :]
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None
                )

        # Fallback - treat as content
        return DeltaMessage(content=delta_text)

    def _handle_implicit_think(
        self,
        delta_text: str,
        end_in_prev: bool,
        end_in_delta: bool,
    ) -> DeltaMessage | None:
        """Handle case where <think> was in prompt (only </think> in output)."""
        if end_in_delta:
            # Transition: end token in this delta
            idx = delta_text.find(self.end_token)
            reasoning_part = delta_text[:idx]
            content_part = delta_text[idx + len(self.end_token) :]
            return DeltaMessage(
                reasoning=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )
        elif end_in_prev:
            # Already past reasoning phase - pure content
            return DeltaMessage(content=delta_text)
        else:
            # Still in implicit reasoning phase
            return DeltaMessage(reasoning=delta_text)
