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
    Delta message for streaming reasoning output.

    Contains either reasoning content, regular content, or both when
    transitioning from reasoning to content phase.

    Note: reasoning and content should typically not both be non-None
    except during the transition chunk.
    """

    role: str | None = None
    content: str | None = None
    reasoning: str | None = None

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
            tokenizer: Optional tokenizer for token-based parsing. For rapid-mlx,
                      text-based parsing is sufficient, so this is optional.
        """
        self.tokenizer = tokenizer

    @abstractmethod
    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from complete model output.

        Args:
            model_output: Complete text output from the model.
            enable_thinking: Whether the request set
                ``chat_template_kwargs.enable_thinking=True``. ``None``
                preserves pre-#575 behaviour — the load-bearing path is
                ``BaseThinkingReasoningParser`` Case 4 / Qwen3 fallback
                where ``True`` routes truncated bare-text to reasoning
                instead of leaking the whole thought trace to content.
                Channel-based parsers (Harmony / GPT-OSS / Gemma 4) can
                accept and ignore the flag — their tags are unambiguous.

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

    def reset_state(self):  # noqa: B027
        """
        Reset any internal state for a new request.

        Called before starting to process a new streaming request.
        Override in subclasses if stateful parsing is needed.
        This is intentionally a default no-op implementation.
        """
        pass

    def finalize_streaming(  # noqa: B027
        self,
        accumulated_text: str,
        *,
        matched_stop: str | None = None,
    ) -> "DeltaMessage | None":
        """
        Finalize streaming and return optional correction chunk.

        Called after the stream loop completes. Subclasses can override
        to emit a correction (e.g., reclassifying short no-tag output
        that was initially treated as reasoning).

        Args:
            accumulated_text: Complete accumulated text from the stream.
            matched_stop: When non-None, indicates the engine truncated
                the output because a user-supplied stop string matched
                (scheduler.py:3673). This is the D-STOP-THINK
                truncation signal: subclasses with prompt-injected
                ``<think>`` semantics (Qwen3 / DeepSeek-R1 families
                where the chat template wraps the prompt with
                ``<think>\\n``) MUST treat this as evidence that the
                model was in active thinking mode when stop fired —
                because a casual non-thinking answer that legitimately
                contained the user's stop string would NOT be a
                D-STOP-THINK shape, but rather a successful early
                termination of an actual answer. ``matched_stop=None``
                means natural EOS / max_tokens — fall back to the
                content-correction casual-answer contract.

        Returns:
            DeltaMessage correction chunk, or None if no correction needed.
        """
        pass
