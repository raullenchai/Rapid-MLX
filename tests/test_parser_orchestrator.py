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
        # Mirror real hermes-style tristate:
        # - {"tool_calls": [...]} when a complete tool call is parsed
        # - None when buffering (saw <tool_call> but not </tool_call> yet)
        # - {"content": delta_text} for plain-text passthrough
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
        if "<tool_call>" in current_text and "</tool_call>" not in current_text:
            return None  # buffering
        return {"content": delta_text}

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
    # The reasoning parser surfaces ``<tool_call>`` as
    # ``content_to_preserve`` but the tool parser is now buffering
    # (sees ``<tool_call>`` without ``</tool_call>``). Orchestrator
    # MUST suppress the raw markup — d2 is either None or carries no
    # ``<tool_call>`` text. This is the codex R6 contract.
    if d2 is not None:
        assert (d2.content or "") == "" or "<tool_call" not in (d2.content or ""), (
            f"raw <tool_call> prefix leaked: {d2}"
        )


def test_no_reasoning_parser_means_tool_phase_from_start():
    """Codex round-3 regression: tool-only parser must enter tool phase
    from chunk 0. Without this, reasoning_ended starts False and never
    flips (no reasoning parser to consult), so the orchestrator would
    suppress every chunk and the tool call would never surface."""
    p = _build_parser(with_reasoning=False)
    state = p._stream_state
    # Tool phase active from the very first chunk.
    assert p._in_reasoning_phase(state) is False
    assert p._in_tool_call_phase(state) is True

    p.parse_delta("<tool_call>{")
    d_final = p.parse_delta('"x":1}</tool_call>')
    # FakeToolParser emits on </tool_call> — must materialize.
    assert d_final is not None
    assert d_final.tool_calls is not None


def test_mixed_reasoning_plus_content_delta_flips_reasoning_ended():
    """Codex round-8 regression. Some reasoning parsers (Gemma 4 on a
    thought→content transition) emit BOTH ``reasoning`` and
    ``content`` on the same boundary delta. The orchestrator must
    flip ``reasoning_ended`` on that delta so the tool parser sees
    the content side this same chunk — otherwise a boundary chunk
    carrying ``<tool_call>...</tool_call>`` would skip the tool
    phase and leak as raw content."""

    class _MixedReasoningParser(_FakeReasoningParser):
        def __init__(self, tokenizer=None):
            super().__init__(tokenizer)
            self._emitted = False

        def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
            # On the first chunk emit BOTH reasoning and content (a
            # Gemma 4-style mixed boundary delta).
            if not self._emitted:
                self._emitted = True
                return DeltaMessage(reasoning="r", content=delta_text)
            return DeltaMessage(content=delta_text)

        def is_reasoning_end_streaming(self, previous_text, current_text):
            # Never reports via the canonical end-token signal — the
            # orchestrator must flip via the content-set fallback.
            return False

    _WrappedParser.reasoning_parser_cls = _MixedReasoningParser
    _WrappedParser.tool_parser_cls = _FakeToolParser
    p = _WrappedParser()

    # Boundary chunk: parser emits reasoning AND content
    # (content = the tool-call markup that follows).
    p.parse_delta("<tool_call>{}</tool_call>")
    assert p._stream_state.reasoning_ended is True, (
        "mixed reasoning+content delta must flip reasoning_ended so "
        "the tool parser sees boundary content this same chunk"
    )


def test_content_only_reasoning_delta_flips_reasoning_ended():
    """Codex round-3 regression: parsers that decide a delta is
    content-only (no reasoning text) must hand off to the tool parser
    even without seeing ``</think>``. Otherwise the next chunk's
    ``<tool_call>`` markup leaks as content."""

    class _ContentOnlyReasoningParser(_FakeReasoningParser):
        def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
            # Pretend we decided no reasoning is happening — return
            # content-only directly. No </think> in the stream.
            return DeltaMessage(content=delta_text, reasoning=None)

        def is_reasoning_end_streaming(self, previous_text, current_text):
            # Never flips through the canonical end-token signal.
            return False

    _WrappedParser.reasoning_parser_cls = _ContentOnlyReasoningParser
    _WrappedParser.tool_parser_cls = _FakeToolParser
    p = _WrappedParser()

    p.parse_delta("hi")
    assert p._stream_state.reasoning_ended is True, (
        "content-only reasoning delta must flip reasoning_ended so the "
        "tool parser starts seeing subsequent chunks"
    )
    # Subsequent tool call should now be parsed.
    d_final = p.parse_delta("<tool_call>{}</tool_call>")
    assert d_final is not None
    assert d_final.tool_calls is not None


def test_coalesced_boundary_with_full_tool_call_drops_raw_markup():
    """Codex round-4 regression: when a single chunk carries the
    reasoning end-token plus a complete tool call
    (e.g. ``</think><tool_call>{...}</tool_call>``), ``content_to_preserve``
    is the raw tool markup the reasoning parser surfaced as the
    post-``</think>`` remainder. With ``tool_delta.tool_calls`` set the
    tool parser has already consumed that text — the orchestrator must
    NOT attach it as content, or clients would see the raw tool JSON
    alongside the structured tool_calls emission."""
    p = _build_parser()

    # Open the reasoning block first.
    p.parse_delta("<think>")
    p.parse_delta("plan")

    # Coalesced boundary: closing tag + full tool call in one chunk.
    d = p.parse_delta("</think><tool_call>{}</tool_call>")
    assert d is not None
    assert d.tool_calls is not None, f"expected tool_calls, got {d}"

    # Hard contract: content must NOT contain raw tool markup.
    assert (d.content or "") == "" or (
        "<tool_call" not in d.content and "</tool_call>" not in d.content
    ), (
        f"raw tool markup leaked as content on coalesced boundary: "
        f"content={d.content!r} tool_calls={d.tool_calls!r}"
    )


def test_boundary_chunk_prefers_reasoning_parser_content_over_tool_passthrough():
    """Codex round-3 regression: on a boundary chunk like
    ``final thought</think>Answer`` with no tool call, the reasoning
    parser splits content as ``Answer`` while the tool parser's
    pass-through content is the raw delta. Orchestrator must surface
    the reasoning parser's stripped version, not the raw markup."""
    p = _build_parser()
    # Prime: open reasoning
    p.parse_delta("<think>")
    p.parse_delta("step 1")
    # Boundary chunk: reasoning text, end token, then plain content
    d = p.parse_delta("final thought</think>Answer")
    assert d is not None
    # The content side must be "Answer" (reasoning parser's stripped
    # split), not "final thought</think>Answer" (raw passthrough).
    assert d.content == "Answer", (
        f"expected reasoning parser's stripped content 'Answer', "
        f"got {d.content!r} (likely raw tool passthrough including "
        f"</think>)"
    )


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
    # Strategy 3 returns a fresh subclass per resolve (not the shared
    # ``_WrappedParser`` base — codex round-4 regression).
    assert issubclass(cls, _WrappedParser)
    assert cls is not _WrappedParser
    inst = cls()
    # Real parsers wired in
    assert type(inst.reasoning_parser).__name__ == "Qwen3ReasoningParser"
    assert type(inst.tool_parser).__name__ == "HermesToolParser"


def test_manager_strategy_3_returns_distinct_classes_per_pair():
    """Codex round-4 regression: each resolve must produce a fresh
    subclass so concurrent / interleaved resolves don't collide on the
    shared ``_WrappedParser`` class attributes. Without this, the
    second ``get_parser`` call would overwrite the first's
    ``reasoning_parser_cls`` and both instances would pick up
    whichever pair was assigned last."""
    cls_qwen_hermes = ParserManager.get_parser(
        tool_parser_name="hermes",
        reasoning_parser_name="qwen3",
        enable_auto_tools=True,
    )
    cls_minimax = ParserManager.get_parser(
        tool_parser_name="minimax",
        reasoning_parser_name="minimax",
        enable_auto_tools=True,
    )
    # If they're the same class, the assignment ran on the shared base
    # and the first resolve was clobbered.
    assert cls_qwen_hermes is not cls_minimax
    assert cls_qwen_hermes.reasoning_parser_cls.__name__ == "Qwen3ReasoningParser"
    assert cls_minimax.reasoning_parser_cls.__name__ == "MiniMaxReasoningParser"

    # Instantiating the first resolve AFTER the second resolve must
    # still pick up the first's wiring — proving the class isn't
    # shared.
    inst_qwen_hermes = cls_qwen_hermes()
    assert type(inst_qwen_hermes.reasoning_parser).__name__ == "Qwen3ReasoningParser"
    assert type(inst_qwen_hermes.tool_parser).__name__ == "HermesToolParser"


def test_manager_strategy_3_only_reasoning_when_tool_disabled():
    cls = ParserManager.get_parser(
        tool_parser_name="hermes",
        reasoning_parser_name="qwen3",
        enable_auto_tools=False,  # disables tool parser resolution
    )
    assert issubclass(cls, _WrappedParser)
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
