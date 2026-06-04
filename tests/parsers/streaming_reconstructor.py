# SPDX-License-Identifier: Apache-2.0
"""Streaming reconstructors for parser regression tests.

Direct port of vLLM's ``tests/reasoning/utils.py`` and
``tests/tool_parsers/utils.py``. The pattern: a reconstructor walks the
streaming deltas produced by a parser, accumulates them, and asserts
per-delta invariants. Each test then asserts the reconstructed final
state matches the non-streaming extraction.

This pattern is SOTA across vLLM and SGLang for parser regression. See
``docs/parser_testing_patterns.md`` (referenced from the cluster fix PR)
for the upstream rationale. Adopted unchanged in shape; minor adapters
for Rapid-MLX:

  * Reasoning parsers already return ``DeltaMessage`` (see
    ``vllm_mlx/reasoning/base.py``) — same field layout as vLLM, so
    ``StreamingReasoningReconstructor`` is a verbatim port.

  * Tool parsers return ``dict[str, Any] | None`` (see
    ``vllm_mlx/tool_parsers/abstract_tool_parser.py``) with keys
    ``"content"`` and ``"tool_calls"``. ``StreamingToolReconstructor``
    accepts that dict shape rather than vLLM's typed ``DeltaMessage``,
    but enforces the same invariants.

Invariants enforced per delta (any violation = parser regression):

  R1. For reasoning: ``delta.content is None or delta.reasoning is None``
      — both populated in the same delta means a marker leaked into the
      "other" channel. (vLLM ``reasoning/utils.py:18-19``.)

  R2. For tools: each ``index`` slot emits ``id`` exactly once and
      ``function.name`` exactly once on first appearance; subsequent
      deltas, if any, may only append to ``function.arguments`` —
      Rapid-MLX parsers typically emit a single finalization delta at
      ``<|call|>`` rather than incremental argument streaming, but the
      invariant covers both shapes. (vLLM ``tool_parsers/utils.py:
      49-65``.)

  R3. For tools: ``type`` is ``None`` or ``"function"`` — anything else
      is a wire-format leak. (vLLM ``tool_parsers/utils.py:39-42``.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm_mlx.reasoning.base import DeltaMessage


class StreamingReasoningReconstructor:
    """Accumulate streaming reasoning deltas + assert per-delta invariants.

    Port of vLLM ``tests/reasoning/utils.py:10-30``. The invariant on
    line 18-19 — ``delta.content is None or delta.reasoning is None`` —
    is the cluster fix's primary target: harmony / gemma4 / hermes leaks
    populate both fields in the same delta when a channel marker bleeds
    into content.
    """

    def __init__(self) -> None:
        self.reasoning: str | None = None
        self.other_content: str | None = None

    def append_delta(self, delta: DeltaMessage) -> None:
        # R1: both populated = channel marker leak.
        assert delta.content is None or delta.reasoning is None, (
            "Both content and reasoning populated in the same delta — a "
            "channel marker (e.g. '<|channel|>', '</thought>') leaked into "
            f"the wrong channel. delta={delta!r}"
        )
        if delta.content is not None:
            if self.other_content is None:
                self.other_content = delta.content
            else:
                self.other_content += delta.content
        elif delta.reasoning is not None:
            if self.reasoning is None:
                self.reasoning = delta.reasoning
            else:
                self.reasoning += delta.reasoning


@dataclass
class ReconstructedToolCall:
    """Final state of a single tool call after stream reassembly."""

    id: str
    name: str
    arguments: str = ""
    type: str = "function"


@dataclass
class StreamingToolReconstructor:
    """Accumulate streaming tool deltas + assert per-delta invariants.

    Port of vLLM ``tests/tool_parsers/utils.py:17-81``, adapted for
    Rapid-MLX's dict-shaped delta return value. Each tool call slot is
    keyed by ``index``; ``id`` and ``function.name`` must appear exactly
    once per slot, and ``function.arguments`` accumulates.

    Args:
        assert_one_tool_per_delta: If True (default), each delta may
            contain at most one tool call entry. Matches vLLM
            ``tool_parsers/utils.py:30-37``. Some parsers legitimately
            batch (e.g. parallel-tool finalization at end-of-stream);
            those tests should pass False.
    """

    assert_one_tool_per_delta: bool = True
    other_content: str = ""
    tool_calls: list[ReconstructedToolCall] = field(default_factory=list)

    def append_delta(self, delta: dict[str, Any]) -> None:
        # Rapid-MLX tool parsers return either {"content": "..."},
        # {"tool_calls": [...]}, or {"content": ..., "tool_calls": [...]}.
        # Either content or tool_calls (or both) must be non-empty per
        # vLLM tool_parsers/utils.py:24-29 — empty deltas should have
        # been ``None`` rather than a populated dict.
        content = delta.get("content")
        tool_calls_raw = delta.get("tool_calls")
        # Codex K2: explicitly reject ``"tool_calls": None`` — the
        # ``or []`` fallback would silently treat it as "no tool calls"
        # and skip the per-call invariant pass entirely.
        assert tool_calls_raw is None or isinstance(tool_calls_raw, list), (
            f"Streaming tool delta has malformed 'tool_calls' field "
            f"(expected list or omitted; got {type(tool_calls_raw).__name__}): "
            f"{delta!r}"
        )
        tool_calls = tool_calls_raw or []

        assert content is not None or tool_calls, (
            "Streaming tool delta must include content or tool_calls "
            f"(or both); got empty delta={delta!r}. If the parser meant "
            "to suppress emission, it should return ``None`` instead."
        )

        if content is not None:
            self.other_content += content

        if self.assert_one_tool_per_delta:
            assert len(tool_calls) < 2, (
                "Streaming should emit only one tool call per delta; got "
                f"{len(tool_calls)} in {delta!r}. (Set "
                "``assert_one_tool_per_delta=False`` on the reconstructor "
                "for parallel-tool finalization tests.)"
            )

        for call_delta in tool_calls:
            self._append_tool_call_delta(call_delta)

    def _append_tool_call_delta(self, call_delta: dict[str, Any]) -> None:
        # R3: type must be absent or "function".
        delta_type = call_delta.get("type")
        assert delta_type is None or delta_type == "function", (
            f"Streaming tool calls must emit type='function'; got type={delta_type!r}"
        )

        index = call_delta.get("index")
        assert isinstance(index, int) and index >= 0, (
            f"Streaming tool delta is missing a non-negative integer "
            f"``index``; got {index!r}. (vLLM tool_parsers/utils.py:57-60.)"
        )

        # Codex K2: ``"function": None`` would otherwise be silently
        # accepted as ``{}`` via ``or``, masking missing name validation.
        function_raw = call_delta.get("function", {})
        assert isinstance(function_raw, dict), (
            f"Streaming tool delta has malformed 'function' field "
            f"(expected dict; got {type(function_raw).__name__}): {call_delta!r}"
        )
        function = function_raw
        delta_name = function.get("name")
        # Strict typing: ``arguments`` is optional but, when present,
        # MUST be a string. Using ``or ""`` would silently coerce
        # ``None`` / ``0`` / ``False`` to ``""``, swallowing a malformed
        # delta that should have failed the assertion. (Codex re-review.)
        delta_args_raw = function.get("arguments")
        assert delta_args_raw is None or isinstance(delta_args_raw, str), (
            f"Streaming tool delta has malformed 'arguments' field "
            f"(expected str or absent; got "
            f"{type(delta_args_raw).__name__}={delta_args_raw!r}): "
            f"{call_delta!r}"
        )
        delta_args = delta_args_raw if delta_args_raw is not None else ""

        existing = self.tool_calls[index] if index < len(self.tool_calls) else None

        if existing is not None:
            # R2: id and name must NOT reappear on subsequent deltas
            # for the same index.
            assert not call_delta.get("id"), (
                "Streaming tool calls must emit id only once per index. "
                f"Got reappearance: index={index}, id={call_delta.get('id')!r}"
            )
            assert not delta_name, (
                "Streaming tool calls must emit function.name only once "
                f"per index. Got reappearance: index={index}, "
                f"name={delta_name!r}"
            )
            assert index == len(self.tool_calls) - 1, (
                f"Incorrect index for tool delta. Got {index}, "
                f"expected {len(self.tool_calls) - 1}"
            )
            existing.arguments += delta_args
        else:
            # R2: id and name must appear on first delta for this index.
            # Codex re-review BLOCKING: require non-empty strings — pre-fix
            # the ``is not None`` check accepted ``id=""`` / ``name=""``
            # as satisfying the invariant, letting a malformed first delta
            # with blank identifiers slip through silently.
            first_id = call_delta.get("id")
            assert isinstance(first_id, str) and first_id, (
                "Streaming tool calls must include a non-empty string id "
                f"on first appearance. Got id={first_id!r} in {call_delta!r}"
            )
            assert isinstance(delta_name, str) and delta_name, (
                "Streaming tool calls must include a non-empty string "
                f"function.name on first appearance. Got name={delta_name!r} "
                f"in {call_delta!r}"
            )
            assert index == len(self.tool_calls), (
                f"Incorrect index for first tool delta. Got {index}, "
                f"expected {len(self.tool_calls)}"
            )
            self.tool_calls.append(
                ReconstructedToolCall(
                    id=str(call_delta["id"]),
                    name=str(delta_name),
                    arguments=delta_args,
                    type=delta_type or "function",
                )
            )
