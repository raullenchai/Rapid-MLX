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

    # ------------------------------------------------------------------
    # r5-D — finalize-on-truncation hook (shared across parser families)
    # ------------------------------------------------------------------
    #
    # When the non-streaming aggregator finishes with
    # ``finish_reason="length"`` and the parser was still mid-think (the
    # closing sentinel — ``</think>``, ``<channel|>``, harmony
    # ``<|end|>`` — never arrived), the default ``extract_reasoning``
    # routing on several parsers leaked the in-progress thought into
    # ``content`` (glm4 autonomous, minimax <think>-opener) or duplicated
    # it across both fields (gemma4 ``<|channel>thought``). Each parser
    # SHOULD return True from ``is_open_in_think`` when its accumulated
    # buffer indicates a not-yet-closed reasoning state so the route's
    # finalize layer can re-classify the buffer as ``reasoning_content``
    # and set ``content=None``. The default implementation returns False
    # so non-thinking parsers (ui_tars, models that never opened a think
    # span) keep current behaviour.
    #
    # See ``finalize_truncation`` below for the shared router and the
    # ``gemma4`` / ``glm4`` / ``minimax`` / ``think_parser`` overrides
    # for the family-specific marker checks.
    def is_open_in_think(self, accumulated_text: str) -> bool:  # noqa: B027
        """Return True iff ``accumulated_text`` ends inside an
        unclosed reasoning span this parser would route as reasoning.

        Default: ``False`` (no-think / safe fallback). Subclasses with
        explicit think markers (``<think>``, ``<|channel>thought``,
        Harmony analysis) override and inspect their own tags.
        """
        del accumulated_text  # noqa: F841 — default is no-think
        return False


def finalize_truncation(
    open_in_think: bool, buffer: str | None
) -> tuple[str | None, str | None]:
    """Route an unclosed reasoning buffer at ``finish_reason="length"``.

    Shared finalize-on-truncation helper invoked by the non-streaming
    aggregator (``vllm_mlx/service/helpers.py::_finalize_content_and_reasoning``)
    when a reasoning parser's first-pass ``extract_reasoning`` would
    otherwise emit ``(None, buffer)`` (content leak) or
    ``(buffer, buffer)`` (duplication). Each parser exposes the
    family-specific open-in-think check via ``is_open_in_think``; the
    router itself is parser-agnostic.

    The contract is symmetric:

    * ``open_in_think=True``  → ``(reasoning_content=buffer,
      content=None)`` — everything in the buffer was inside the
      think tag.
    * ``open_in_think=False`` → ``(reasoning_content=None,
      content=buffer)`` — think already closed (or never opened);
      the buffer is plain content.

    Empty / ``None`` buffer short-circuits to ``(None, None)`` so
    callers do not need to guard the empty case at the call site.

    Returns ``(reasoning_content, final_content)``. Either may be
    ``None``.
    """
    if not buffer:
        return None, None
    if open_in_think:
        return buffer, None
    return None, buffer
