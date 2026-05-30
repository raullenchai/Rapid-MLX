# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vLLM's vllm/parser/abstract_parser.py (Apache-2.0). The
# original ships an OpenAI Responses-API surface and CUDA-shaped tool
# infrastructure that we don't use; we keep the unified ``Parser`` /
# ``DelegatingParser`` / ``_WrappedParser`` abstraction and the
# ``parse_delta`` state machine that resolves the reasoning↔tool-call
# boundary in a single place — that's the piece worth borrowing. Routes
# in this codebase carry text, not token IDs, so the boundary check is
# text-based (``is_reasoning_end_streaming(previous_text, current_text)``)
# rather than token-ID based.
"""
Unified ``Parser`` abstraction for streaming chat completions.

Closes the bug class where stream and non-stream paths re-implement the
"reasoning first, then tool calls, but watch the boundary" sequence and
silently diverge. ``DelegatingParser.parse_delta`` is the single seam.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizerBase

from vllm_mlx.reasoning.base import DeltaMessage, ReasoningParser
from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser


@dataclass
class StreamState:
    """Mutable per-stream state for ``Parser.parse_delta``."""

    reasoning_ended: bool = False
    tool_call_text_started: bool = False
    previous_text: str = ""
    # ``tool_phase_text`` accumulates only the post-reasoning portion of
    # the stream — what the tool parser is allowed to see. Without this
    # the tool parser receives the full ``previous_text`` (including the
    # reasoning body), and any ``<tool_call>`` / ``</tool_call>``
    # mentions hidden inside reasoning (e.g. a model thinking aloud
    # about the syntax it's about to emit) skew the parser's
    # open/close counting and drop the first real tool call. See codex
    # R5 (PR #488). Reset alongside ``previous_text`` on every
    # ``reset_state``.
    tool_phase_text: str = ""
    history_tool_call_cnt: int = 0


def _delta_from_tool_parser_result(
    result: dict[str, Any] | DeltaMessage | None,
) -> DeltaMessage | None:
    """Adapt a tool parser's streaming return to ``DeltaMessage``.

    Existing rapid-mlx tool parsers return ``dict | None`` with OpenAI
    ChoiceDelta-shaped keys (``content``, ``tool_calls``). Wrap that into
    the same ``DeltaMessage`` the orchestrator emits everywhere else, so
    callers downstream only ever see one type.
    """
    if result is None or isinstance(result, DeltaMessage):
        return result
    return DeltaMessage(
        content=result.get("content"),
        tool_calls=result.get("tool_calls"),
    )


class Parser(ABC):
    """
    Abstract Parser unifying ``ReasoningParser`` and ``ToolParser`` behind
    a single ``parse_delta`` entry point.

    Concrete subclasses either:

    1. Override ``parse_delta`` directly with model-specific logic
       (channel-routed models like Harmony, Gemma 4 — see Phase 2).
    2. Set ``self._reasoning_parser`` and ``self._tool_parser`` in
       ``__init__`` and inherit ``DelegatingParser``'s orchestration.

    ``reasoning_parser_cls`` / ``tool_parser_cls`` mirror vLLM's pattern
    for code that needs the class (not the instance) — e.g.
    ``ParserManager.get_parser`` composing a ``_WrappedParser``.
    """

    reasoning_parser_cls: type[ReasoningParser] | None = None
    tool_parser_cls: type[ToolParser] | None = None

    def __init__(self, tokenizer: PreTrainedTokenizerBase | None = None) -> None:
        self.model_tokenizer = tokenizer
        self._reasoning_parser: ReasoningParser | None = None
        self._tool_parser: ToolParser | None = None
        self._stream_state = StreamState()

    @property
    def reasoning_parser(self) -> ReasoningParser | None:
        return self._reasoning_parser

    @reasoning_parser.setter
    def reasoning_parser(self, parser: ReasoningParser | None) -> None:
        self._reasoning_parser = parser

    @property
    def tool_parser(self) -> ToolParser | None:
        return self._tool_parser

    @tool_parser.setter
    def tool_parser(self, parser: ToolParser | None) -> None:
        self._tool_parser = parser

    def reset_state(self) -> None:
        """Reset stream state for a new request."""
        self._stream_state = StreamState()
        if self._reasoning_parser is not None:
            self._reasoning_parser.reset_state()
        if self._tool_parser is not None and hasattr(self._tool_parser, "reset"):
            self._tool_parser.reset()

    @abstractmethod
    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        """Extract ``(reasoning, content)`` from a complete model output."""

    @abstractmethod
    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> Any:
        """Extract tool calls from a complete model output.

        Returns ``ExtractedToolCallInformation`` (see
        ``vllm_mlx.tool_parsers.abstract_tool_parser``).
        """

    @abstractmethod
    def parse_delta(
        self,
        delta_text: str,
        request: dict[str, Any] | None = None,
    ) -> DeltaMessage | None:
        """Parse a single streaming delta, advancing internal stream state.

        The orchestration runs reasoning extraction first, then tool-call
        extraction on the same delta, merging boundary chunks where a
        single token flush spans both phases. Returns the merged
        ``DeltaMessage`` (or ``None`` if the delta should be suppressed).
        """


class DelegatingParser(Parser):
    """
    The default ``Parser`` composite: holds an optional reasoning parser
    and an optional tool parser and runs the orchestration state machine.

    Subclasses should populate ``self._reasoning_parser`` and
    ``self._tool_parser`` in ``__init__``. Either may be ``None`` —
    methods short-circuit.
    """

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        if self._reasoning_parser is None:
            return None, model_output
        return self._reasoning_parser.extract_reasoning(model_output)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> Any:
        from vllm_mlx.tool_parsers.abstract_tool_parser import (
            ExtractedToolCallInformation,
        )

        if self._tool_parser is None:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        return self._tool_parser.extract_tool_calls(model_output, request)

    def _in_reasoning_phase(self, state: StreamState) -> bool:
        if self._reasoning_parser is None:
            return False
        return not state.reasoning_ended

    def _in_tool_call_phase(self, state: StreamState) -> bool:
        if self._tool_parser is None:
            return False
        if self._reasoning_parser is None:
            # No reasoning to wait for — start in tool-call phase from
            # chunk 0. Without this branch a tool-only parser would
            # never be invoked (state.reasoning_ended starts False and
            # nothing flips it), so every chunk would be suppressed.
            return True
        return state.reasoning_ended

    def _extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        if self._reasoning_parser is None:
            return DeltaMessage(content=delta_text)
        return self._reasoning_parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
        )

    def _is_reasoning_end_streaming(
        self, previous_text: str, current_text: str
    ) -> bool:
        if self._reasoning_parser is None:
            return True  # no reasoning parser → always in "content" phase
        return self._reasoning_parser.is_reasoning_end_streaming(
            previous_text=previous_text, current_text=current_text
        )

    def _extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        request: dict[str, Any] | None,
    ) -> DeltaMessage | None:
        if self._tool_parser is None:
            return None
        # Phase 1 scope: only the default ``tool_choice="auto"`` path is
        # wired here. Named / "required" tool_choice streaming (vLLM PR
        # #41876) is a Phase 2 follow-up — until then the route layer
        # continues to enforce those modes upstream.
        raw = self._tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=None,
            current_token_ids=None,
            delta_token_ids=None,
            request=request,
        )
        return _delta_from_tool_parser_result(raw)

    def parse_delta(
        self,
        delta_text: str,
        request: dict[str, Any] | None = None,
    ) -> DeltaMessage | None:
        state = self._stream_state
        previous_text = state.previous_text
        current_text = previous_text + delta_text
        delta_message: DeltaMessage | None = None
        reasoning_parser_consulted = False

        # ---- Reasoning phase ----
        if self._in_reasoning_phase(state):
            reasoning_parser_consulted = True
            delta_message = self._extract_reasoning_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
            )
            if self._is_reasoning_end_streaming(
                previous_text=previous_text, current_text=current_text
            ) or (
                delta_message is not None
                and delta_message.content
                and not delta_message.reasoning
            ):
                state.reasoning_ended = True

        # ---- Tool call phase ----
        if self._in_tool_call_phase(state):
            # Boundary delta carries reasoning AND a tool-call prefix in
            # the same chunk; preserve the reasoning side before the tool
            # parser overwrites delta_message. This is the literal fix
            # for vLLM PR #42691 / rapid-mlx PR #436's bug class.
            reasoning_to_preserve: str | None = None
            content_to_preserve: str | None = None
            if delta_message is not None:
                reasoning_to_preserve = delta_message.reasoning
                content_to_preserve = delta_message.content

            # The tool parser must only see the post-reasoning portion
            # of the stream — feeding it the full text would let
            # ``<tool_call>`` / ``</tool_call>`` mentions hidden inside
            # reasoning skew its open/close counting and drop the first
            # real tool call (codex R5).
            #
            # - Boundary chunk (reasoning parser was consulted AND
            #   emitted content_to_preserve): seed tool_phase_text from
            #   the reasoning parser's stripped post-``</think>`` body.
            # - Past-boundary chunks (reasoning parser not consulted):
            #   append the full delta to tool_phase_text.
            # - Tool-only mode (no reasoning parser ever consulted):
            #   tool_phase_text accumulates from chunk 0.
            if reasoning_parser_consulted:
                tool_delta_text = content_to_preserve or ""
            else:
                tool_delta_text = delta_text
            tool_previous_text = state.tool_phase_text
            tool_current_text = tool_previous_text + tool_delta_text
            state.tool_phase_text = tool_current_text

            tool_delta = self._extract_tool_calls_streaming(
                previous_text=tool_previous_text,
                current_text=tool_current_text,
                delta_text=tool_delta_text,
                request=request,
            )

            if tool_delta is not None:
                delta_message = tool_delta
                # Boundary chunk preservation — tool parser produced a
                # delta (either tool_calls or pass-through content). Keep
                # the reasoning side so callers don't lose pre-boundary
                # reasoning.
                if reasoning_to_preserve:
                    delta_message.reasoning = reasoning_to_preserve
                if content_to_preserve and delta_message.tool_calls is None:
                    # Tool parser is just passing the raw delta through
                    # (no tool call detected). The reasoning parser is
                    # authoritative for the boundary's content/reasoning
                    # split — it has already stripped ``</think>`` while
                    # the tool parser got the unprocessed delta_text.
                    # Prefer the reasoning parser's stripped content.
                    delta_message.content = content_to_preserve
                # Note: when the tool parser emitted a structured
                # ``tool_calls`` payload we deliberately discard
                # ``content_to_preserve`` — on the boundary chunk that
                # value IS the raw ``<tool_call>{...}</tool_call>``
                # markup the tool parser just consumed. Attaching it
                # would leak the tool-call JSON as a content delta
                # alongside the structured emission.
            else:
                # Tool parser returned None → intentionally buffering an
                # incomplete tool-call body. Suppress the tool-call
                # markup entirely. ``content_to_preserve`` here is the
                # post-``</think>`` text the reasoning parser handed us,
                # which on the boundary chunk IS the tool-call prefix
                # (``<tool_call>`` etc) — emitting it would leak the raw
                # markup to the client. Only ``reasoning_to_preserve``
                # is safe to surface.
                #
                # CRITICAL: explicitly rebuild ``delta_message`` (or
                # clear to None). The reasoning parser may have set
                # ``delta_message`` above with ``content=<tool_call>``
                # — without the explicit reset that raw markup would
                # flow through as a content delta (codex R6). For
                # split chunks like ``</think>`` arriving in one chunk
                # then ``<tool_call>...`` in the next, the reasoning
                # parser's content side is the tool-call prefix and
                # MUST be suppressed.
                delta_message = (
                    DeltaMessage(reasoning=reasoning_to_preserve)
                    if reasoning_to_preserve
                    else None
                )

        # Final fallback — pass the raw delta through as content in
        # exactly two cases:
        #   (a) no parser is wired at all
        #   (b) reasoning has already ended AND no tool parser is wired
        #       (post-reasoning plain-text passthrough — the reasoning
        #       parser handed off and the tool parser isn't there to
        #       interpret the rest)
        # The case we DO NOT cover: reasoning parser was consulted and
        # returned ``None`` deliberately (e.g. standalone ``<think>``
        # special token suppression). Fabricating content there would
        # leak the marker to the SSE stream.
        if delta_message is None:
            if not self._has_any_parser():
                delta_message = DeltaMessage(content=delta_text)
            elif (
                not reasoning_parser_consulted
                and not self._in_tool_call_phase(state)
                and not self._has_tool_parser()
            ):
                # Post-reasoning passthrough with no tool parser wired.
                delta_message = DeltaMessage(content=delta_text)

        state.previous_text = current_text
        return delta_message

    def _has_tool_parser(self) -> bool:
        """Return True when a tool parser is wired and can buffer chunks."""
        return self._tool_parser is not None

    def _has_any_parser(self) -> bool:
        """Return True when at least one of reasoning / tool parsers is wired."""
        return self._reasoning_parser is not None or self._tool_parser is not None


class _WrappedParser(DelegatingParser):
    """
    ``DelegatingParser`` that instantiates its ReasoningParser / ToolParser
    from class attributes — the shape ``ParserManager.get_parser`` builds
    when no model-specific unified ``Parser`` is registered.

    The manager sets ``_WrappedParser.reasoning_parser_cls`` and
    ``_WrappedParser.tool_parser_cls`` before instantiation; ``__init__``
    materializes the underlying parsers.
    """

    reasoning_parser_cls: type[ReasoningParser] | None = None
    tool_parser_cls: type[ToolParser] | None = None

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase | None = None, **kwargs: Any
    ) -> None:
        super().__init__(tokenizer)
        if self.__class__.reasoning_parser_cls is not None:
            self._reasoning_parser = self.__class__.reasoning_parser_cls(tokenizer)
        if self.__class__.tool_parser_cls is not None:
            self._tool_parser = self.__class__.tool_parser_cls(tokenizer)


__all__ = [
    "DelegatingParser",
    "Parser",
    "StreamState",
    "_WrappedParser",
]
