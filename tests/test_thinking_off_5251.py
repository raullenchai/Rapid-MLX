# SPDX-License-Identifier: Apache-2.0
"""Thinking-off output keeps model-generated reasoning off every text lane."""

import pytest

from tests.test_flint_parser_routing import CALL
from tests.test_flint_parser_routing import test_flint_routes as run_route


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("step", [1, 7, 10000])
@pytest.mark.parametrize(
    "raw",
    [
        CALL,
        "<think>Private thought.</think>Done.",
        "Private thought.</think>Done.",
        "Private thought.</think>" + CALL,
        "Private thought.</think>",
    ],
)
def test_thinking_off(surface, stream, step, raw):
    run_route(surface, stream, False, raw, step, enable_thinking=False)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("step", [1, 7, 10000])
@pytest.mark.parametrize(
    "raw",
    ["Hello there.", "Here is the answer: 42.", "Use <tool_call>", "Literal <arg_"],
)
def test_thinking_off_plain_answer(surface, stream, step, raw):
    run_route(
        surface, stream, False, raw, step, expected_literal=raw, enable_thinking=False
    )


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_thinking_off_bare_tool_call(surface, stream):
    run_route(surface, stream, False, CALL, 1, enable_thinking=False)


def test_parser_holds_ambiguous_text_and_resets():
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    parser = Qwen3ReasoningParser()
    for text in ["Hello.", "Here's a thinking process: a literal label."]:
        parser.configure_request(enable_thinking=False)
        for end in range(1, len(text) + 1):
            assert (
                parser.extract_reasoning_streaming(
                    text[: end - 1], text[:end], text[end - 1 : end]
                )
                is None
            )
        assert parser.finish_stream().content == text
        assert parser.finish_stream() is None
    parser.configure_request(enable_thinking=True)
    assert (
        parser.extract_reasoning_streaming("", "Thought", "Thought").reasoning
        == "Thought"
    )


@pytest.mark.parametrize(
    "text", ["Secret.</think>Answer", "<think>Secret.</think>Answer"]
)
def test_thinking_off_parser_every_split(text):
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    for split in range(1, len(text)):
        parser = Qwen3ReasoningParser()
        parser.configure_request(enable_thinking=False)
        messages = [
            parser.extract_reasoning_streaming("", text[:split], text[:split]),
            parser.extract_reasoning_streaming(text[:split], text, text[split:]),
            parser.finish_stream(),
            parser.finalize_streaming(text),
        ]
        assert "".join(m.content or "" for m in messages if m) == "Answer"
        assert "Secret." in "".join(m.reasoning or "" for m in messages if m)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_thinking_off_tool_result_turn(surface, stream):
    run_route(
        surface,
        stream,
        True,
        "Private thought.</think>" + CALL,
        1,
        enable_thinking=False,
    )


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("step", [1, 7, 10000])
def test_thinking_off_unclosed_stream_does_not_republish_reasoning(surface, step):
    run_route(
        surface, True, False, "<think>Private thought.", step, enable_thinking=False
    )
