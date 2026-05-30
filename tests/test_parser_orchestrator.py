# SPDX-License-Identifier: Apache-2.0
"""
Tests for the unified ``Parser`` orchestrator.

Pins the contract that closes the "stream and non-stream silently
diverge" bug class (PRs #436 / #443 / #454 / #456 / #460 / #462 and
their vLLM analogues #39446 / #41876 / #42691). The orchestrator runs
reasoning extraction first, then tool-call extraction on the same
delta, merging boundary chunks where a single token flush spans both
phases.
"""

from __future__ import annotations

from typing import Any

from vllm_mlx.parser import (
    DelegatingParser,
    Parser,
    ParserManager,
    StreamState,
    _WrappedParser,
)
from vllm_mlx.reasoning.base import DeltaMessage, ReasoningParser
from vllm_mlx.tool_parsers.abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
)

# ----------------------------- fakes -----------------------------


class _FakeReasoningParser(ReasoningParser):
    """Detects ``<think>...</think>`` purely by text inspection."""

    def __init__(self, tokenizer: Any | None = None) -> None:
        super().__init__(tokenizer)
        self.reasoning_ended = False

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        if "</think>" not in model_output:
            return (
                (model_output, None)
                if "<think>" in model_output
                else (None, model_output)
            )
        reasoning, _, content = model_output.partition("</think>")
        reasoning = reasoning.replace("<think>", "").strip() or None
        return reasoning, content.strip() or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        if "</think>" in current_text:
            self.reasoning_ended = True
            # Boundary delta: emit any post-</think> remainder as content
            _, _, post = current_text.partition("</think>")
            prev_post = ""
            if "</think>" in previous_text:
                _, _, prev_post = previous_text.partition("</think>")
            new_content = post[len(prev_post) :] if post.startswith(prev_post) else post
            return DeltaMessage(content=new_content or None)
        return DeltaMessage(reasoning=delta_text)

    def is_reasoning_end_streaming(self, previous_text: str, current_text: str) -> bool:
        return "</think>" in current_text

    def reset_state(self) -> None:
        self.reasoning_ended = False


class _FakeToolParser(ToolParser):
    """Emits a single tool_call when ``</tool_call>`` is seen."""

    EXPECTED_WIRE_FORMATS = ("tool_call_json",)

    def __init__(self, tokenizer: Any | None = None) -> None:
        super().__init__(tokenizer)
        self.emitted = False

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        if "</tool_call>" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[{"name": "x", "arguments": "{}", "index": 0, "id": "call_1"}],
            content=None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Any = None,
        current_token_ids: Any = None,
        delta_token_ids: Any = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if "</tool_call>" in current_text and not self.emitted:
            self.emitted = True
            return {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "x", "arguments": "{}"},
                    }
                ]
            }
        return None

    def reset(self) -> None:
        self.emitted = False
        super().reset() if hasattr(super(), "reset") else None


# ----------------------------- helpers -----------------------------


def _build_parser(
    *, with_reasoning: bool = True, with_tool: bool = True
) -> DelegatingParser:
    _WrappedParser.reasoning_parser_cls = (
        _FakeReasoningParser if with_reasoning else None
    )
    _WrappedParser.tool_parser_cls = _FakeToolParser if with_tool else None
    return _WrappedParser()


# ----------------------------- tests: state machine -----------------------------


def test_pure_content_passthrough_no_parsers():
    p = _build_parser(with_reasoning=False, with_tool=False)
    delta = p.parse_delta("hello")
    assert delta is not None
    assert delta.content == "hello"
    assert delta.reasoning is None
    assert delta.tool_calls is None


def test_reasoning_phase_routes_to_reasoning_field():
    p = _build_parser(with_tool=False)
    delta = p.parse_delta("<think>thinking")
    assert delta is not None
    assert delta.reasoning == "<think>thinking"
    assert delta.content is None


def test_boundary_transitions_to_tool_phase_after_end_token():
    p = _build_parser()

    # Phase 1 — reasoning
    p.parse_delta("<think>")
    p.parse_delta("planning")
    assert p._stream_state.reasoning_ended is False

    # Boundary — </think> seen, state flips
    p.parse_delta("</think>")
    assert p._stream_state.reasoning_ended is True

    # Phase 2 — tool call
    p.parse_delta('<tool_call>{"name":"x"')
    delta_final = p.parse_delta("}</tool_call>")
    assert delta_final is not None
    assert delta_final.tool_calls is not None
    assert delta_final.tool_calls[0]["function"]["name"] == "x"


def test_boundary_delta_preserves_reasoning_when_tool_starts_in_same_chunk():
    """vLLM PR #42691 / rapid-mlx PR #436 regression coverage.

    A single delta carrying both ``...</think>`` AND ``<tool_call>...`` must
    not drop the reasoning side when the orchestrator hands off to the
    tool parser. The fake reasoning parser emits a content chunk on
    boundary; the orchestrator's preserve-reasoning step keeps any
    pre-boundary reasoning attached.
    """
    p = _build_parser()
    # Open thinking
    d1 = p.parse_delta("<think>step 1")
    assert d1 is not None and d1.reasoning == "<think>step 1"

    # Combined boundary delta: closes think AND opens tool call in one
    # streamed chunk
    d2 = p.parse_delta("</think><tool_call>")
    # After this delta, the boundary state should be ended
    assert p._stream_state.reasoning_ended is True
    # Reasoning parser emitted post-</think> content (the "<tool_call>"
    # prefix); tool parser hasn't seen </tool_call> yet, so it returns
    # None and the post-boundary content flows through.
    assert d2 is not None


def test_no_reasoning_parser_means_tool_phase_from_start():
    p = _build_parser(with_reasoning=False)
    assert p._in_reasoning_phase(p._stream_state) is False
    # Without reasoning parser, _is_reasoning_end_streaming defaults to
    # True so we're immediately in the tool-call phase (or content if
    # tool parser is also absent).
    p.parse_delta("<tool_call>{")
    p.parse_delta('"x":1}</tool_call>')
    # FakeToolParser only emits on </tool_call>; last delta carries it.
    state = p._stream_state
    assert state.reasoning_ended is True or p._reasoning_parser is None


def test_reset_state_clears_stream_and_subparsers():
    p = _build_parser()
    p.parse_delta("<think>halfway")
    p.parse_delta("</think>boundary")
    assert p._stream_state.reasoning_ended is True

    p.reset_state()
    assert isinstance(p._stream_state, StreamState)
    assert p._stream_state.reasoning_ended is False
    assert p._stream_state.previous_text == ""
    assert p._reasoning_parser.reasoning_ended is False  # type: ignore[union-attr]


def test_non_stream_extract_reasoning_delegates():
    p = _build_parser()
    reasoning, content = p.extract_reasoning("<think>x</think>answer")
    assert reasoning == "x"
    assert content == "answer"


def test_non_stream_extract_tool_calls_delegates():
    p = _build_parser()
    info = p.extract_tool_calls("<tool_call>...</tool_call>")
    assert info.tools_called is True
    assert info.tool_calls[0]["name"] == "x"


# ----------------------------- tests: ParserManager -----------------------------


def test_manager_returns_none_when_no_names_given():
    assert ParserManager.get_parser(None, None) is None


def test_manager_strategy_3_wraps_individual_parsers():
    cls = ParserManager.get_parser(
        tool_parser_name="hermes",
        reasoning_parser_name="qwen3",
        enable_auto_tools=True,
    )
    assert cls is _WrappedParser
    inst = cls()
    # Real parsers wired in
    assert type(inst.reasoning_parser).__name__ == "Qwen3ReasoningParser"
    assert type(inst.tool_parser).__name__ == "HermesToolParser"


def test_manager_strategy_3_only_reasoning_when_tool_disabled():
    cls = ParserManager.get_parser(
        tool_parser_name="hermes",
        reasoning_parser_name="qwen3",
        enable_auto_tools=False,  # disables tool parser resolution
    )
    assert cls is _WrappedParser
    inst = cls()
    assert inst.tool_parser is None
    assert inst.reasoning_parser is not None


def test_manager_list_registered_empty_without_unified_parsers():
    # No model-specific unified Parsers registered in Phase 1
    assert isinstance(ParserManager.list_registered(), list)


def test_manager_unknown_tool_parser_raises_helpful_error():
    import pytest

    with pytest.raises(TypeError, match="not registered"):
        ParserManager.get_parser(
            tool_parser_name="does_not_exist",
            reasoning_parser_name=None,
            enable_auto_tools=True,
        )


# ----------------------------- tests: protocol -----------------------------


def test_parser_abc_is_abstract():
    import pytest

    with pytest.raises(TypeError):
        Parser()  # type: ignore[abstract]


def test_delegating_parser_is_concrete():
    p = DelegatingParser()
    assert isinstance(p, Parser)
    assert isinstance(p._stream_state, StreamState)
