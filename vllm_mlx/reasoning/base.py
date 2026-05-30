# SPDX-License-Identifier: Apache-2.0
"""
Base classes for reasoning content extraction.

This module provides the abstract base class for reasoning parsers that extract
thinking/reasoning content from model outputs (e.g., <think>...</think> tags).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class DeltaMessage:
    """
    Delta message for streaming model output.

    Contains reasoning content, regular content, and/or tool calls. A single
    delta typically populates one field; the boundary chunk between phases
    can populate both reasoning and content, or reasoning and tool_calls.

    tool_calls follows the OpenAI ChoiceDeltaToolCall shape:
        [{"index": 0, "id": "...", "type": "function",
          "function": {"name": "...", "arguments": "..."}}, ...]
    """

    role: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    @property
    def reasoning_content(self) -> str | None:
        """Deprecated: use reasoning instead. Maintained for backward compatibility."""
        return self.reasoning


class ReasoningParser(ABC):
    """
    Abstract base class for reasoning content extraction.

    Reasoning parsers extract thinking/reasoning content from model outputs,
    separating it from the final response content. This is useful for models
    like DeepSeek-R1, Qwen3, etc. that use special tokens to denote reasoning.

    Example:
        Input: "<think>Let me solve this step by step...</think>The answer is 42."
        Output: reasoning="Let me solve this step by step...", content="The answer is 42."
    """

    def __init__(self, tokenizer: Any | None = None):
        """
        Initialize parser with optional tokenizer.

        Args:
            tokenizer: Optional tokenizer for token-based parsing. For vllm-mlx,
                      text-based parsing is sufficient, so this is optional.
        """
        self.tokenizer = tokenizer

    @abstractmethod
    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from complete model output.

        Args:
            model_output: Complete text output from the model.

        Returns:
            Tuple of (reasoning_content, final_content).
            Either may be None if not present.
        """
        pass

    @abstractmethod
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta.

        Uses the "previous + delta = current" model where:
        - previous_text: All text accumulated before this delta
        - current_text: All text including this delta (previous + delta)
        - delta_text: Just the new text in this chunk

        Args:
            previous_text: Accumulated text before this delta.
            current_text: Accumulated text including this delta.
            delta_text: The new text in this streaming chunk.

        Returns:
            DeltaMessage with reasoning and/or content populated,
            or None if this delta should be skipped (e.g., special tokens).
        """
        pass

    def is_reasoning_end_streaming(self, previous_text: str, current_text: str) -> bool:
        """
        Check whether the reasoning phase has ended after the latest delta.

        Used by the unified ``Parser.parse_delta`` orchestrator to decide
        when to hand off from reasoning extraction to tool-call extraction
        within the same stream. Default implementation reads a
        ``reasoning_ended`` attribute that most concrete parsers (qwen3,
        deepseek_r1, glm4, harmony) already set during streaming. Parsers
        without that flag can override this method directly.
        """
        return bool(getattr(self, "reasoning_ended", False))

    def reset_state(self):  # noqa: B027
        """
        Reset any internal state for a new request.

        Called before starting to process a new streaming request.
        Override in subclasses if stateful parsing is needed.
        This is intentionally a default no-op implementation.
        """
        pass

    def finalize_streaming(  # noqa: B027
        self, accumulated_text: str
    ) -> "DeltaMessage | None":
        """
        Finalize streaming and return optional correction chunk.

        Called after the stream loop completes. Subclasses can override
        to emit a correction (e.g., reclassifying short no-tag output
        that was initially treated as reasoning).

        Args:
            accumulated_text: Complete accumulated text from the stream.

        Returns:
            DeltaMessage correction chunk, or None if no correction needed.
        """
        pass
