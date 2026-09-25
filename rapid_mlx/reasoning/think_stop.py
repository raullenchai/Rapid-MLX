# SPDX-License-Identifier: Apache-2.0
"""
Answer-scoped stop-string matching for ``<think>``-style reasoning models.

User-supplied ``stop=[...]`` sequences truncate the text the client gets
back. For a reasoning model that text is the answer: the chain of thought
between ``<think>`` and ``</think>`` is returned separately as
``reasoning_content`` (or dropped). The schedulers used to match stops
against the raw decoded stream, so a stop string the model merely wrote
while reasoning ended the request inside the ``<think>`` block, and the
client received an empty ``content``. A model asked to count to 10 with
``stop=["10"]`` counts in its reasoning first; an agent whose stop markers
are ``</execute_bash>``-style tags names them while planning.

#1049 fixed this for harmony models by scoping stops to the ``final``
channel (:mod:`.harmony_stop`). This module is the ``<think>`` counterpart:
stops only match the text after the reasoning close marker.

The scheduler cannot tell by itself whether generation starts inside a
reasoning block, because Qwen3.5 / DeepSeek-R1-style templates open
``<think>`` in the prompt and only ``</think>`` appears in the output. The
route layer already answers that question for the streaming reasoning
router (``service.helpers._should_start_in_thinking``), so it attaches a
:class:`ReasoningStopScope` to the request's ``SamplingParams`` and the
schedulers consult it here. Requests without a scope keep the raw-stream
match unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReasoningStopScope:
    """Where user stops may match for one request.

    ``start_marker`` / ``end_marker`` are the reasoning parser's
    ``start_token`` / ``end_token`` (``<think>`` / ``</think>``).
    ``starts_in_reasoning`` is True when the chat template opened the
    reasoning block in the prompt, so the output begins inside it.
    """

    start_marker: str
    end_marker: str
    starts_in_reasoning: bool = False


def build_reasoning_stop_scope(
    reasoning_parser: Any, *, starts_in_reasoning: bool
) -> ReasoningStopScope | None:
    """Return the stop scope for a configured reasoning parser, or ``None``.

    Only ``<think>``-tag parsers (``BaseThinkingReasoningParser``
    subclasses: qwen3, deepseek_r1, glm4, ...) get a scope. Harmony models
    are scoped separately by the scheduler's model-family gate (#1049), and
    other parser families keep the raw-stream match.
    """
    if reasoning_parser is None:
        return None
    from .think_parser import BaseThinkingReasoningParser

    if not isinstance(reasoning_parser, BaseThinkingReasoningParser):
        return None
    start = reasoning_parser.start_token
    end = reasoning_parser.end_token
    if not (isinstance(start, str) and start and isinstance(end, str) and end):
        return None
    return ReasoningStopScope(
        start_marker=start,
        end_marker=end,
        starts_in_reasoning=bool(starts_in_reasoning),
    )


def answer_start(decoded_so_far: str, scope: ReasoningStopScope) -> int | None:
    """Offset in ``decoded_so_far`` where the answer begins, or ``None``.

    * The first ``end_marker`` closes reasoning; the answer starts after it
      and after the whitespace the template writes there (the reasoning
      parser strips that whitespace from ``content`` as well).
    * With no close yet, the output is still reasoning when the prompt
      opened the block or the output itself opened it, and cannot be
      classified yet while it is empty or a partial ``start_marker``.
      ``None`` means no user stop may match yet. The scheduler searches the
      whole decoded text on every step, so a stop in text that later turns
      out to be answer is still found at its original position.
    * Otherwise the model answered without reasoning: ``0``, i.e. the
      pre-scope raw-stream match.
    """
    close = decoded_so_far.find(scope.end_marker)
    if close != -1:
        start = close + len(scope.end_marker)
        while start < len(decoded_so_far) and decoded_so_far[start].isspace():
            start += 1
        return start
    if scope.starts_in_reasoning:
        return None
    head = decoded_so_far.lstrip()
    if head.startswith(scope.start_marker) or scope.start_marker.startswith(head):
        return None
    return 0


def find_stop_in_answer(
    decoded_so_far: str, stop_params: list[str], scope: ReasoningStopScope
) -> tuple[str, int] | None:
    """Match ``stop_params`` against the answer part of ``decoded_so_far``.

    Returns ``(stop_str, global_offset)`` or ``None``. Iteration order
    matches the scheduler's raw-stream path (the first stop in
    ``stop_params`` that occurs wins), so a request whose answer starts at
    offset 0 behaves exactly as before.
    """
    start = answer_start(decoded_so_far, scope)
    if start is None:
        return None
    for stop_str in stop_params:
        if not stop_str:
            continue
        idx = decoded_so_far.find(stop_str, start)
        if idx != -1:
            return (stop_str, idx)
    return None
