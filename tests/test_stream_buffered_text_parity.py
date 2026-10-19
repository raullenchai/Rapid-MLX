# SPDX-License-Identifier: Apache-2.0
"""Streaming text matches buffered text on /v1/responses and /v1/messages.

Contract: streamed text deltas are verbatim, including surrounding
whitespace, for every chunking (as on /v1/chat/completions); the buffered
reply is that same text with surrounding whitespace trimmed. Text that rides
on a chunk carrying structured ``tool_calls`` is visible text unless the
chunk is on the ``tool_call`` / ``reasoning`` channel.
"""

import random

import pytest

from rapid_mlx.config import get_config, reset_config
from rapid_mlx.engine.base import GenerationOutput
from tests.test_no_tool_parse_sanitizer_parity import _payload, _visible_and_calls
from tests.test_tool_call_finish_reason import ReplayEngine, client_for, sse

WHITESPACE_TEXTS = [
    "  hi there  ",
    "\n\nHello\n",
    "Hello world",
    "a  b\n\n c \t",
    " \n ",
    "line one\n\nline two\n",
]
CALL = {"name": "get_weather", "arguments": '{"city":"Paris"}'}


@pytest.fixture(autouse=True)
def _restore_config():
    yield
    reset_config()


def _chunkings(text, seed_count=6):
    yield [text]
    yield list(text)
    for seed in range(seed_count):
        rng = random.Random(seed)
        cuts = sorted(rng.sample(range(1, len(text)), min(3, len(text) - 1)))
        bounds = [0, *cuts, len(text)]
        yield [text[a:b] for a, b in zip(bounds, bounds[1:])]


def _run(engine, route, mode):
    client = client_for(engine)
    get_config().tool_call_parser = "hermes"
    payload = _payload(route, mode)
    buffered = client.post(f"/v1/{route}", json=payload)
    assert buffered.status_code == 200, buffered.text
    streaming = client.post(f"/v1/{route}", json={**payload, "stream": True})
    assert streaming.status_code == 200, streaming.text
    return _visible_and_calls(route, buffered.json(), sse(streaming.text))


@pytest.mark.parametrize("route", ["responses", "messages"])
@pytest.mark.parametrize("mode", ["missing", "auto"])
@pytest.mark.parametrize("text", WHITESPACE_TEXTS)
def test_surrounding_whitespace_parity(route, mode, text):
    # Messages never opens a blank text block, so a whitespace-only reply
    # streams nothing there.
    expected = "" if route == "messages" and not text.strip() else text
    for deltas in _chunkings(text):
        buffered, streamed, _, _ = _run(ReplayEngine(deltas, "stop"), route, mode)
        assert buffered == text.strip()
        assert streamed == expected, deltas
        assert streamed.strip() == buffered


class ToolChunkEngine(ReplayEngine):
    """Last text delta arrives on the chunk that carries structured calls."""

    def __init__(self, deltas, channel):
        super().__init__(deltas, "tool_calls", separate_finish=False)
        self.channel = channel

    async def stream_chat(self, messages, **kwargs):
        accumulated = ""
        for i, delta in enumerate(self.deltas):
            accumulated += delta
            last = i == len(self.deltas) - 1
            yield GenerationOutput(
                text=accumulated,
                new_text=delta,
                channel=self.channel,
                finished=last,
                finish_reason="tool_calls" if last else None,
                tool_calls=[CALL] if last else None,
            )

    async def chat(self, messages, **kwargs):
        output = await super().chat(messages, **kwargs)
        output.tool_calls = [CALL]
        return output


@pytest.mark.parametrize("route", ["responses", "messages"])
@pytest.mark.parametrize("channel", [None, "content"])
def test_text_on_structured_call_chunk_is_streamed(route, channel):
    engine = ToolChunkEngine(["Hello ", "world"], channel)
    buffered, streamed, buffered_calls, streamed_calls = _run(engine, route, "auto")
    assert buffered == "Hello world"
    assert streamed == buffered
    assert [c["name"] for c in streamed_calls] == ["get_weather"]
    assert [c["name"] for c in buffered_calls] == ["get_weather"]


@pytest.mark.parametrize("route", ["responses", "messages"])
def test_tool_call_channel_summary_stays_hidden(route):
    # On the tool_call channel the chunk text is an argument summary.
    engine = ToolChunkEngine(["Hello", '{"city":"Paris"}'], "tool_call")
    _, streamed, _, streamed_calls = _run(engine, route, "auto")
    # Responses never shows tool_call-channel text; Messages renders the
    # channel's plain deltas as text but not the summary on the call chunk.
    assert streamed == ("Hello" if route == "messages" else "")
    assert [c["name"] for c in streamed_calls] == ["get_weather"]
