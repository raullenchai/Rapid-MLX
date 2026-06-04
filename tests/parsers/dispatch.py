# SPDX-License-Identifier: Apache-2.0
"""Stream/non-stream dispatch helpers for parser regression tests.

Direct port of vLLM ``tests/reasoning/utils.py:33-53`` (reasoning
dispatcher) and ``tests/tool_parsers/utils.py:84-103`` (tool
dispatcher). The pattern: a single ``run_*_extraction`` function takes
a ``streaming: bool`` flag and dispatches to the appropriate code path.
``pytest.parametrize`` then drives the same ``TEST_CASES`` table
through both branches — the SOTA way to lock stream/non-stream parity
without writing two parallel test suites.
"""

from __future__ import annotations

from typing import Any

from vllm_mlx.reasoning.base import ReasoningParser
from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

from .streaming_reconstructor import (
    ReconstructedToolCall,
    StreamingReasoningReconstructor,
    StreamingToolReconstructor,
)


def run_reasoning_extraction(
    reasoning_parser: ReasoningParser,
    model_deltas: list[str],
    streaming: bool = False,
) -> tuple[str | None, str | None]:
    """Drive a reasoning parser through stream or non-stream extraction.

    Port of vLLM ``tests/reasoning/utils.py:33-53``.

    Args:
        reasoning_parser: Parser instance under test.
        model_deltas: Either per-token fragments (streaming) or a list
            whose ``"".join(...)`` is the full model output (both modes
            accept the same input — non-stream just joins first).
        streaming: If True, feed each delta to
            ``extract_reasoning_streaming``; otherwise call
            ``extract_reasoning`` once on the joined text.

    Returns:
        ``(reasoning, content)`` — either may be None if absent.
    """
    if streaming:
        reconstructor = _run_reasoning_streaming(reasoning_parser, model_deltas)
        return (
            reconstructor.reasoning,
            reconstructor.other_content or None,
        )
    return reasoning_parser.extract_reasoning("".join(model_deltas))


def _run_reasoning_streaming(
    reasoning_parser: ReasoningParser,
    model_deltas: list[str],
) -> StreamingReasoningReconstructor:
    """Walk the per-delta loop of vLLM ``reasoning/utils.py:96-125``.

    Calls ``parser.reset_state()`` once at the start so stateful
    parsers (most of ours) begin clean. After the loop, calls
    ``parser.finalize_streaming(accumulated)`` and appends any
    correction chunk — this matches what the route layer does at end
    of stream and is the only way to surface late-emitted reasoning.
    """
    reasoning_parser.reset_state()
    reconstructor = StreamingReasoningReconstructor()
    previous_text = ""
    for delta in model_deltas:
        current_text = previous_text + delta
        delta_message = reasoning_parser.extract_reasoning_streaming(
            previous_text, current_text, delta
        )
        if delta_message is not None:
            reconstructor.append_delta(delta_message)
        previous_text = current_text
    final = reasoning_parser.finalize_streaming(previous_text)
    if final is not None:
        reconstructor.append_delta(final)
    return reconstructor


def run_tool_extraction(
    tool_parser: ToolParser,
    model_deltas: list[str],
    streaming: bool = False,
    assert_one_tool_per_delta: bool = True,
) -> tuple[str | None, list[ReconstructedToolCall]]:
    """Drive a tool parser through stream or non-stream extraction.

    Port of vLLM ``tests/tool_parsers/utils.py:84-103``, adapted for
    Rapid-MLX's ``ExtractedToolCallInformation`` return type and dict-
    shaped streaming deltas.

    Returns:
        ``(content, tool_calls)`` — content may be None; tool_calls is
        a list of ``ReconstructedToolCall`` (empty if no tools).
    """
    if streaming:
        reconstructor = _run_tool_streaming(
            tool_parser,
            model_deltas,
            assert_one_tool_per_delta=assert_one_tool_per_delta,
        )
        return reconstructor.other_content or None, reconstructor.tool_calls

    extracted = tool_parser.extract_tool_calls("".join(model_deltas))
    tool_calls = [
        ReconstructedToolCall(
            id=tc.get("id", ""),
            name=tc["name"],
            arguments=tc.get("arguments", ""),
            type="function",
        )
        for tc in extracted.tool_calls
    ]
    return extracted.content, tool_calls


def _run_tool_streaming(
    tool_parser: ToolParser,
    model_deltas: list[str],
    assert_one_tool_per_delta: bool = True,
) -> StreamingToolReconstructor:
    """Walk the per-delta loop of vLLM ``tool_parsers/utils.py:129-167``.

    Same shape as the reasoning helper: reset, loop, accumulate, no
    explicit finalize call (tool parsers don't expose one in our
    abstract base — late-emitted tool calls come through the last
    real delta containing a terminator like ``<|call|>``).
    """
    reconstructor = StreamingToolReconstructor(
        assert_one_tool_per_delta=assert_one_tool_per_delta
    )
    previous_text = ""
    request: dict[str, Any] | None = None
    for delta in model_deltas:
        current_text = previous_text + delta
        delta_message = tool_parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta,
            previous_token_ids=None,
            current_token_ids=None,
            delta_token_ids=None,
            request=request,
        )
        if delta_message is not None:
            reconstructor.append_delta(delta_message)
        previous_text = current_text
    return reconstructor
