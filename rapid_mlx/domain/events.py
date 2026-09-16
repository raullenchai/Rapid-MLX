# SPDX-License-Identifier: Apache-2.0
"""Stream events — the seam between post-processing and SSE formatting.

Each streaming path (OpenAI Chat, Anthropic, Completions) produces the same
StreamEvent objects. The formatting layer converts them to spec-specific SSE.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StreamEvent:
    """A single event produced by the streaming post-processor.

    The post-processor yields these; each API adapter formats them
    into its own SSE wire format (OpenAI JSON, Anthropic events, etc.).
    """

    type: str
    """Event type:
    - "content": text content delta
    - "reasoning": reasoning/thinking content delta
    - "tool_call": structured tool call detected
    - "finish": generation finished (may carry final content/correction)
    - "suppress": chunk should be suppressed (tool markup being accumulated)
    """

    content: str | None = None
    """Text content delta (for "content" and "finish" events)."""

    reasoning: str | None = None
    """Reasoning/thinking content delta (for "reasoning" events)."""

    tool_calls: list | None = None
    """Structured tool calls (for "tool_call" events).
    Format: list of dicts with index, id, type, function keys."""

    finish_reason: str | None = None
    """Finish reason (for "finish" events): "stop", "length", "tool_calls"."""

    tool_calls_detected: bool = False
    """Whether tool calls were detected during this stream."""

    metadata: dict = field(default_factory=dict)
    """Optional metadata (e.g. usage info, prompt_tokens, completion_tokens)."""
