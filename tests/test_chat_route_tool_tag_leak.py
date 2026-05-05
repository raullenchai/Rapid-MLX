# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the chat-route tool-tag leak bug.

Bug: when the tool parser successfully extracted tool_calls from output that
contained an unclosed `<tool_call>` block (model omitted the closing tag), it
returned `content=None` (i.e. fully stripped). routes/chat.py then fell back
to the RAW output for the reasoning parser via `cleaned_text or output.text`,
and the reasoning parser strips `<think>` but not `<tool_call>` — so the
opening tag + JSON survived all the way to user-facing `content`.

Detected by `rapid-mlx agents hermes --test` (no_tool_leak + stress_no_leak)
even on a strong 27B model, ruling out model weakness. Fix: when tool_calls
were extracted, trust the tool parser's cleaned_text and only run the
reasoning parser to recover reasoning_text from the raw output.
"""

from types import SimpleNamespace

from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
from vllm_mlx.routes.chat import _finalize_content_and_reasoning
from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser


def _make_request_stub() -> SimpleNamespace:
    """Match the shape of ChatCompletionRequest the parser may inspect.

    HermesToolParser may walk request.tools / request.model on
    schema-driven type-conversion paths; passing None lets a future
    parser change pass `request=None` silently while users would crash
    in production. SimpleNamespace gives the parser the same attribute
    access surface it sees in the live route.
    """
    return SimpleNamespace(
        model="test-model",
        tools=None,
        tool_choice=None,
        messages=[],
    )


def _drive_chat_route_pipeline(
    raw_output: str,
) -> tuple[str | None, list, str | None]:
    """Drive the REAL post-parse helper from routes/chat.py.

    Wraps the production tool parser + reasoning parser around the
    extracted ``_finalize_content_and_reasoning`` helper so the
    regression suite tests the exact orchestration the route runs —
    no parallel reimplementation that can silently drift from prod.

    Returns (final_content, tool_calls, reasoning_text). final_content
    is what the user receives in `choices[0].message.content`.
    """
    parser = HermesToolParser(tokenizer=None)
    parser.reset()
    result = parser.extract_tool_calls(raw_output, request=_make_request_stub())
    cleaned_text = result.content or ""
    tool_calls = list(result.tool_calls) if result.tools_called else []

    reasoning_parser = Qwen3ReasoningParser(tokenizer=None)
    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text=raw_output,
        cleaned_text=cleaned_text,
        tool_calls=tool_calls,
        reasoning_parser=reasoning_parser,
    )

    final_content = cleaned_text if cleaned_text else None
    return final_content, tool_calls, reasoning_text


_LEAK_MARKERS = ("<tool_call>", "<function=", "<|im_end|>", "<|tool_call|>")


def _assert_no_leak(content: str | None) -> None:
    if not content:
        return
    leaks = [m for m in _LEAK_MARKERS if m in content]
    assert not leaks, (
        f"Tool tags leaked into user content: {leaks!r} (content={content!r})"
    )


class TestToolTagLeakRegression:
    """The specific cases that fired in the agent test suite."""

    def test_unclosed_tool_call_does_not_leak(self):
        # Real Qwopus 27B output: model omitted </tool_call> closing tag.
        raw = '<tool_call>\n{"name": "terminal", "arguments": {"command": "echo test"}}'
        content, tool_calls, reasoning = _drive_chat_route_pipeline(raw)
        assert tool_calls and tool_calls[0]["name"] == "terminal"
        _assert_no_leak(content)
        # Qwen3's implicit-think heuristic could otherwise reroute a bare
        # tool_call into reasoning_content where it would also be visible
        # to the user. Guard both sinks.
        _assert_no_leak(reasoning)

    def test_unclosed_tool_call_with_thinking_does_not_leak(self):
        # The exact shape that fires in stress_no_leak — reasoning + tool_call.
        raw = (
            "<think>The user wants me to use the terminal tool.</think>\n"
            '<tool_call>\n{"name": "terminal", "arguments": {"command": "echo test"}}'
        )
        content, tool_calls, reasoning = _drive_chat_route_pipeline(raw)
        assert tool_calls and tool_calls[0]["name"] == "terminal"
        assert reasoning and "terminal tool" in reasoning
        _assert_no_leak(content)
        _assert_no_leak(reasoning)

    def test_well_formed_tool_call_still_passes(self):
        # Control: properly closed tag. Should still extract cleanly.
        raw = (
            "<think>I should use the terminal.</think>\n"
            '<tool_call>\n{"name": "terminal", "arguments": {"command": "echo test"}}\n'
            "</tool_call>"
        )
        content, tool_calls, reasoning = _drive_chat_route_pipeline(raw)
        assert tool_calls and tool_calls[0]["name"] == "terminal"
        assert reasoning and "terminal" in reasoning
        _assert_no_leak(content)

    def test_no_tool_call_path_preserves_content(self):
        # When no tool_calls fire, plain text content should pass through
        # unchanged (regression guard for the else branch). Note: Hermes
        # parser strips <think> tags itself before the reasoning parser
        # would see them, so reasoning_text from the route's reasoning
        # parser call is expected to be None in this branch — that is
        # pre-existing behavior unrelated to this fix.
        raw = "<think>Just thinking.</think>The actual answer is 42."
        content, tool_calls, _ = _drive_chat_route_pipeline(raw)
        assert not tool_calls
        assert content and "answer is 42" in content
        _assert_no_leak(content)
