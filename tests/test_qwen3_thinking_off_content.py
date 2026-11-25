"""Thinking-off output needs a leading opener to enter reasoning."""

import pytest

from rapid_mlx.api.utils import clean_output_text
from rapid_mlx.prompt_parse_context import PromptParseContext
from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
from tests.parser_migration.harness import route

REPRO = "I will list files.\n```bash\nls\n```\n</think>\nRunning it now."
LITERAL = "Answer mentions <think>private</think> literally."


def chunks(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]


@pytest.mark.parametrize("raw", [REPRO, LITERAL, "</think>Answer", "Hello", " \nHello"])
def test_buffered_thinking_off_is_content(raw):
    parser = Qwen3ReasoningParser(None)
    assert parser.extract_reasoning(
        raw, enable_thinking=False, prompt_thinking_active=False
    ) == (None, raw)


@pytest.mark.parametrize("size", [1, 3, 7, 1000])
@pytest.mark.parametrize("raw", [REPRO, LITERAL, "Hello", " \nHello"])
def test_streaming_thinking_off_emits_each_delta(raw, size):
    parser = Qwen3ReasoningParser(None)
    parser.configure_request(enable_thinking=False, prompt_thinking_active=False)
    previous = ""
    for delta in chunks(raw, size):
        current = previous + delta
        message = parser.extract_reasoning_streaming(previous, current, delta)
        assert message is not None
        assert message.content == delta
        assert message.reasoning is None
        previous = current
    assert parser.finalize_streaming(previous) is None


@pytest.mark.parametrize(
    "thinking,prompt_open", [(True, False), (None, False), (False, True)]
)
def test_buffered_implicit_controls(thinking, prompt_open):
    assert Qwen3ReasoningParser().extract_reasoning(
        REPRO, enable_thinking=thinking, prompt_thinking_active=prompt_open
    ) == ("I will list files.\n```bash\nls\n```", "Running it now.")


@pytest.mark.parametrize(
    "thinking,prompt_open", [(True, False), (None, False), (False, True)]
)
@pytest.mark.parametrize("size", [1, 7, 1000])
def test_streaming_controls_match_unconfigured_parser(thinking, prompt_open, size):
    configured = Qwen3ReasoningParser()
    configured.configure_request(
        enable_thinking=thinking, prompt_thinking_active=prompt_open
    )
    original = Qwen3ReasoningParser()
    previous = ""
    for delta in chunks(REPRO, size):
        current = previous + delta
        assert configured.extract_reasoning_streaming(
            previous, current, delta
        ) == original.extract_reasoning_streaming(previous, current, delta)
        previous = current
    assert configured.finalize_streaming(
        previous, prompt_thinking_active=prompt_open
    ) == original.finalize_streaming(previous, prompt_thinking_active=prompt_open)


@pytest.mark.parametrize("size", [1, 3, 7, 1000])
@pytest.mark.parametrize("prefix", ["", " \n"])
def test_explicit_opener_keeps_existing_behavior(prefix, size):
    raw = prefix + "<think>Secret.</think>Answer"
    parser = Qwen3ReasoningParser()
    assert parser.extract_reasoning(raw, enable_thinking=False) == ("Secret.", "Answer")
    parser.configure_request(enable_thinking=False)
    original = Qwen3ReasoningParser()
    previous = ""
    for delta in chunks(raw, size):
        current = previous + delta
        if not previous.strip() and not current.strip():
            previous = current
            continue
        assert parser.extract_reasoning_streaming(
            previous, current, delta
        ) == original.extract_reasoning_streaming(previous, current, delta)
        previous = current


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("size", [1, 7, 1000])
def test_routes_keep_prose_in_content(surface, stream, size):
    result = route(
        "qwen3",
        "hermes",
        REPRO,
        surface,
        stream,
        chunks(REPRO, size),
        thinking=False,
        tools_mode="absent",
    )
    assert "I will list files.\n```bash\nls\n```" in result["content"]
    assert result["content"].endswith("Running it now.")
    assert result["reasoning"] == ""
    assert result["calls"] == []
    if stream:
        assert result["first_content_index"] == 1


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "thinking,raw",
    [(True, "Secret.</think>Answer"), (False, "<think>Secret.</think>Answer")],
)
def test_route_reasoning_controls(surface, stream, thinking, raw):
    result = route(
        "qwen3",
        "hermes",
        raw,
        surface,
        stream,
        [raw],
        thinking=thinking,
        tools_mode="absent",
    )
    assert result["content"] == "Answer"
    # Messages demotes reasoning when the request disables it.
    if thinking or surface != "messages":
        assert result["reasoning"] == "Secret."


@pytest.mark.parametrize(
    "thinking,prompt_open", [(True, False), (None, False), (False, True)]
)
def test_cleaner_implicit_controls(thinking, prompt_open):
    assert (
        clean_output_text(
            REPRO, enable_thinking=thinking, prompt_thinking_active=prompt_open
        )
        == "<think>" + REPRO
    )


def test_cleaner_thinking_off_does_not_synthesize_opener():
    assert (
        clean_output_text(REPRO, enable_thinking=False, prompt_thinking_active=False)
        == REPRO
    )


@pytest.mark.parametrize("size", [1, 3, 7, 1000])
def test_streaming_leading_closer_is_content(size):
    raw = "</think>Answer"
    parser = Qwen3ReasoningParser()
    parser.configure_request(enable_thinking=False)
    previous = ""
    content = ""
    for delta in chunks(raw, size):
        current = previous + delta
        message = parser.extract_reasoning_streaming(previous, current, delta)
        if message:
            assert not message.reasoning
            content += message.content or ""
        previous = current
    assert content == raw


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("size", [1, 7, 1000])
def test_prompt_opened_routes_keep_main_channels(surface, stream, size):
    raw = "Secret.</think>Answer"
    result = route(
        "qwen3",
        "hermes",
        raw,
        surface,
        stream,
        chunks(raw, size),
        thinking=False,
        model="lfm2.5",
        tools_mode="absent",
        prompt_context=PromptParseContext(
            "<think>", None, "reasoning", False, "chatml"
        ),
    )
    # These values were replayed against origin/main, including chunk timing.
    assert result == {
        "content": "Answer",
        "reasoning": "Secret.",
        "calls": [],
        "first_content_index": {1: 16, 7: 3, 1000: 1}[size] if stream else None,
        "content_deltas": (6 if size == 1 else 1) if stream else 0,
        "input_chunks": len(chunks(raw, size)),
    }
