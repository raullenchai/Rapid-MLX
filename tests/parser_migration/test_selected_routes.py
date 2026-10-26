"""Real route lifecycle, prompt-state latency and failure contracts."""

import asyncio
import itertools

import pytest

from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.prompt_parse_context import prompt_parse_context

from .harness import Engine, route

SURFACES = ("chat", "responses", "messages")
ANSWER = "Hello world. This is a streamed answer."
CALL = (
    '<tool_call>{"name":"run_shell_command","arguments":{"command":"pwd"}}</tool_call>'
)


@pytest.mark.parametrize(
    "surface,thinking,flint,tools",
    list(
        itertools.product(SURFACES, (False, True), (False, True), ("absent", "enabled"))
    ),
)
def test_delayed_engine_prompt_state_and_first_delta(
    monkeypatch, surface, thinking, flint, tools
):
    original = Engine.stream_chat

    async def delayed(self, **kwargs):
        async for output in original(self, **kwargs):
            await asyncio.sleep(0.001)
            yield output

    monkeypatch.setattr(Engine, "stream_chat", delayed)
    suffix = ("" if flint else "<think>") if thinking else "<think>\n\n</think>\n\n"
    prompt = ("<|assistant|>" if flint else "<|im_start|>assistant\n") + suffix
    context = prompt_parse_context(
        prompt, thinking=thinking, assistant_start=0, flint_template=flint
    )
    chunks = (["</think>"] if thinking else []) + [
        "Hello",
        " world",
        ".",
        " This",
        " is",
        " a",
        " streamed",
        " answer",
        ".",
    ]
    raw = "".join(chunks)
    result = route(
        "qwen3",
        "hermes",
        raw,
        surface,
        True,
        chunks=chunks,
        thinking=thinking,
        tools_mode=tools,
        prompt=prompt,
        parse_context=context,
        capture=True,
    )
    assert result["content"] == ANSWER
    assert result["reasoning"] == ""
    assert result["calls"] == []
    assert result["first_content_index"] <= 8
    assert result["content_deltas"] > 1
    assert "<think>" not in result["content"] and "</think>" not in result["content"]


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("split", (1, 7, 10000))
def test_buffered_streamed_tool_parity_and_partition(surface, split):
    raw = "<think>Private." + CALL + "</think>" + ANSWER
    chunks = [raw[i : i + split] for i in range(0, len(raw), split)]
    streamed = route("qwen3", "hermes", raw, surface, True, chunks=chunks, capture=True)
    buffered = route("qwen3", "hermes", raw, surface, False)
    for key in ("content", "reasoning", "calls"):
        assert streamed[key] == buffered[key]
    assert streamed["content"] == ANSWER
    assert streamed["reasoning"] == "Private."
    assert streamed["calls"] == [["run_shell_command", {"command": "pwd"}]]
    if surface == "messages":
        opened = set()
        used = set()
        for packet in streamed["packets"]:
            kind = packet.get("type")
            if kind == "content_block_start":
                assert packet["index"] not in used
                opened.add(packet["index"])
                used.add(packet["index"])
            elif kind == "content_block_delta":
                assert packet["index"] in opened
            elif kind == "content_block_stop":
                assert packet["index"] in opened
                opened.remove(packet["index"])
        assert not opened


@pytest.mark.parametrize("surface", SURFACES)
def test_empty_admission_does_not_steal_native_calls(monkeypatch, surface):
    async def native(self, **kwargs):
        yield GenerationOutput(text="", new_text="", finished=False)
        yield GenerationOutput(
            text="",
            new_text="",
            channel="tool_call",
            tool_calls=[
                {
                    "id": "call_native",
                    "name": "run_shell_command",
                    "arguments": '{"command":"pwd"}',
                }
            ],
            finished=False,
        )
        yield GenerationOutput(
            text="", new_text="", finished=True, finish_reason="tool_calls"
        )

    monkeypatch.setattr(Engine, "stream_chat", native)
    result = route("qwen3", "hermes", "", surface, True, capture=True)
    assert result["calls"] == [["run_shell_command", {"command": "pwd"}]]


@pytest.mark.parametrize("surface", SURFACES)
def test_committed_truncation_is_failure_never_success(surface):
    raw = CALL.removesuffix("</tool_call>")
    buffered = route("qwen3", "hermes", raw, surface, False)
    assert buffered["error"] == 502
    result = route("qwen3", "hermes", raw, surface, True, capture=True)
    packets = result["packets"]
    assert any(
        "error" in packet or packet.get("type") == "response.failed"
        for packet in packets
    )
    assert not any(
        packet.get("type") in {"response.completed", "message_stop"}
        for packet in packets
    )
    assert not any(
        choice.get("finish_reason") == "tool_calls"
        for packet in packets
        for choice in packet.get("choices", [])
    )


def test_messages_capability_gate_survives_core_migration():
    raw = "<think>Private.</think>Answer"
    for stream in (False, True):
        result = route(None, "hermes", raw, "messages", stream)
        assert result["content"] == "Answer"
        assert result["reasoning"] == ""


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("tools", ["absent", "enabled"])
def test_literal_and_private_spans_survive_serialization(surface, thinking, tools):
    literal = "Example <tool_call> and <|im_end|> are literal markers."
    raw = "<think>Private.</think>" + literal if thinking else literal
    streamed = route(
        "qwen3", "hermes", raw, surface, True, thinking=thinking, tools_mode=tools
    )
    buffered = route(
        "qwen3", "hermes", raw, surface, False, thinking=thinking, tools_mode=tools
    )
    for result in (streamed, buffered):
        assert result["content"] == literal
        assert result["reasoning"] == ("Private." if thinking else "")
        assert result["calls"] == []
