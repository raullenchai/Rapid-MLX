# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for K2 Horizon's graded IFM think protocol."""

from __future__ import annotations

from .base import DeltaMessage, ReasoningParser


class K2HorizonReasoningParser(ReasoningParser):
    """Separate IFM reasoning while preserving the model's tool-call handoff.

    K2 can use one of three marker pairs depending on ``reasoning_effort``.
    Recognizing all documented pairs keeps per-request template choices from
    diverging from the server's long-lived parser configuration.
    """

    EFFORT_TOKENS = (
        ("<ifm|think>", "</ifm|think>"),
        ("<ifm|think_fast>", "</ifm|think_fast>"),
        ("<ifm|think_faster>", "</ifm|think_faster>"),
    )
    TOOL_CALLS_START = "<ifm|tool_calls>"
    SANITIZED_REASONING_PROTOCOLS = frozenset({"k2_ifm"})
    implicit_reasoning_until_close = True
    # K2's template has no true no-reasoning mode: an OpenAI-compatible
    # ``enable_thinking=false`` request is rendered with the model's lowest
    # native effort instead.  Keep the parser active so that prompt-primed
    # reasoning cannot leak into ``delta.content`` on that compatibility path.
    sanitize_when_thinking_disabled = True

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self.reset_state()

    @property
    def reasoning_start_str(self) -> str:
        return self.EFFORT_TOKENS[0][0]

    @property
    def reasoning_end_str(self) -> str:
        return self.EFFORT_TOKENS[0][1]

    @property
    def end_token(self) -> str:
        """Compatibility with the reasoning-budget boundary helper."""
        return self.reasoning_end_str

    def reset_state(self):
        self._buffer = ""
        self._finished = False
        self._at_start = True
        self._expected_end: str | None = None

    @classmethod
    def _split_generated_start(cls, text: str) -> tuple[str, str | None]:
        for start, end in cls.EFFORT_TOKENS:
            if text.startswith(start):
                return text[len(start) :], end
        return text, None

    @classmethod
    def _first_boundary(
        cls, text: str, expected_end: str | None = None
    ) -> tuple[int, str] | None:
        endings = (
            [expected_end]
            if expected_end is not None
            else [end for _start, end in cls.EFFORT_TOKENS]
        )
        candidates = [(text.find(end), end) for end in endings if text.find(end) >= 0]
        tool_at = text.find(cls.TOOL_CALLS_START)
        if tool_at >= 0:
            candidates.append((tool_at, cls.TOOL_CALLS_START))
        return min(candidates, default=None, key=lambda item: item[0])

    @classmethod
    def _partial_boundary_overlap(
        cls, text: str, expected_end: str | None = None
    ) -> int:
        overlap = 0
        markers = (
            [expected_end]
            if expected_end is not None
            else [end for _start, end in cls.EFFORT_TOKENS]
        )
        markers.append(cls.TOOL_CALLS_START)
        for marker in markers:
            for size in range(1, min(len(text), len(marker) - 1) + 1):
                if text.endswith(marker[:size]):
                    overlap = max(overlap, size)
        return overlap

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        text, expected_end = self._split_generated_start(model_output)
        boundary = self._first_boundary(text, expected_end)
        if boundary is not None:
            index, marker = boundary
            reasoning = text[:index] or None
            content_start = (
                index if marker == self.TOOL_CALLS_START else index + len(marker)
            )
            content = text[content_start:] or None
            return reasoning, content
        # K2's template always primes one of the three reasoning lanes. Its
        # compatibility handling for ``enable_thinking=False`` merely selects
        # the lowest effort; it does not turn reasoning off. If generation is
        # truncated before a boundary, fail closed as reasoning instead of
        # exposing an unfinished private trace as answer content.
        del enable_thinking
        return text or None, None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        del previous_text, current_text
        if self._finished:
            return DeltaMessage(content=delta_text) if delta_text else None

        self._buffer += delta_text
        if self._at_start:
            starts = [start for start, _end in self.EFFORT_TOKENS]
            if any(start.startswith(self._buffer) for start in starts):
                if self._buffer not in starts:
                    return None
                self._expected_end = next(
                    end for start, end in self.EFFORT_TOKENS if start == self._buffer
                )
                self._buffer = ""
            else:
                for start, end in self.EFFORT_TOKENS:
                    if self._buffer.startswith(start):
                        self._buffer = self._buffer[len(start) :]
                        self._expected_end = end
                        break
            self._at_start = False

        boundary = self._first_boundary(self._buffer, self._expected_end)
        if boundary is not None:
            index, marker = boundary
            reasoning = self._buffer[:index]
            content_start = (
                index if marker == self.TOOL_CALLS_START else index + len(marker)
            )
            content = self._buffer[content_start:]
            self._buffer = ""
            self._finished = True
            return DeltaMessage(reasoning=reasoning or None, content=content or None)

        held = self._partial_boundary_overlap(self._buffer, self._expected_end)
        sendable = len(self._buffer) - held
        reasoning = self._buffer[:sendable]
        self._buffer = self._buffer[sendable:]
        return DeltaMessage(reasoning=reasoning) if reasoning else None

    def finish_stream(self) -> DeltaMessage | None:
        if not self._buffer:
            return None
        held = self._buffer
        self._buffer = ""
        lane = "content" if self._finished else "reasoning"
        return DeltaMessage(**{lane: held})

    def is_open_in_think(self, accumulated_text: str) -> bool:
        for start, end in self.EFFORT_TOKENS:
            if accumulated_text.startswith(start):
                generated = accumulated_text[len(start) :]
                return self._first_boundary(generated, end) is None
        text, _expected_end = self._split_generated_start(accumulated_text)
        return self._first_boundary(text) is None and bool(text)
