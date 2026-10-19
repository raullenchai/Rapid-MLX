# SPDX-License-Identifier: Apache-2.0
"""Content sanitization must be independent of engine delta boundaries (#4925)."""

import pytest

from rapid_mlx.api.utils import StreamingContentSanitizer, sanitize_content_for_stream
from rapid_mlx.config import get_config, reset_config
from rapid_mlx.engine.base import GenerationOutput
from tests.test_tool_call_finish_reason import TOOLS, ReplayEngine, client_for, sse

GEMMA = '<|tool_call>call:get_weather{"city":"Paris"}<tool_call|>'
CALLING = '[Calling tool: get_weather({"city":"Paris"})]'
PROSE = "Use <|tool_call>call:x{}<tool_call|> to call."
PLAIN = "plain text with lone <"
UNIT_TEXTS = [
    GEMMA,
    CALLING,
    PROSE,
    "a<|im_end|>b",
    "x</think>y",
    "[Calling tool: unterminated",
    "<|tool_call>unterminated",
    "Hello <|tool_call>a<tool_call|> and [Calling tool: b] end",
    "a < b [c] <d> e<",
    # Every token alternative, whitespace in Calling, and nested candidates.
    '<|abc><abc|><|abc|><|a"b><a"b|></think>',
    "[Calling  tool: <|tool_call>nested] end",
    "[Calling\ntool: nested [Calling tool: x] end",
    "<|tool_call>nested [Calling tool: x<tool_call|> tail",
    '[CaX <|abc! <abc! </thix <|abc|! <a"|! end',
]


@pytest.fixture(autouse=True)
def _restore_config():
    yield
    reset_config()


@pytest.mark.parametrize("text", UNIT_TEXTS)
def test_sanitizer_all_boundaries(text):
    for deltas in [[text[:i], text[i:]] for i in range(len(text) + 1)] + [list(text)]:
        sanitizer = StreamingContentSanitizer()
        actual = "".join(sanitizer.process(delta) for delta in deltas)
        assert actual + sanitizer.flush() == sanitize_content_for_stream(text), deltas
        assert sanitizer.flush() == ""


def test_sanitizer_promptness():
    sanitizer = StreamingContentSanitizer()
    assert sanitizer.flush() == ""
    assert sanitizer.process("a ") == "a "
    assert sanitizer.process("<") == ""
    assert sanitizer.process(" b") == "< b"
    assert sanitizer.process("x < y") == "x < y"
    assert sanitizer.process("x [y]") == "x [y]"
    assert sanitizer.process("[") == ""
    assert sanitizer.process("z]") == "[z]"
    assert sanitizer.flush() == ""


def _payload(mode):
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Weather?"}],
        "max_tokens": 128,
    }
    if mode != "missing":
        payload.update(tools=TOOLS, tool_choice=mode)
    return payload


ENDPOINT_CASES = [
    (text, chunking)
    for text in [GEMMA, CALLING, PROSE, PLAIN]
    for chunking in ["whole", "characters"]
]
ENDPOINT_CASES += [(GEMMA, i) for i in range(len(GEMMA) + 1)]


@pytest.mark.parametrize("mode", ["missing", "none", "auto"])
@pytest.mark.parametrize("text,chunking", ENDPOINT_CASES)
def test_endpoint_parity(mode, text, chunking):
    if chunking == "whole":
        deltas = [text]
    elif chunking == "characters":
        deltas = list(text)
    else:
        deltas = [text[:chunking], text[chunking:]]
    client = client_for(ReplayEngine(deltas, "stop"))
    get_config().tool_call_parser = "hermes"
    payload = _payload(mode)
    buffered = client.post("/v1/chat/completions", json=payload)
    assert buffered.status_code == 200, buffered.text
    message = buffered.json()["choices"][0]["message"]
    response = client.post("/v1/chat/completions", json={**payload, "stream": True})
    assert response.status_code == 200, response.text
    events = sse(response.text)
    assert not any("error" in event for event in events), response.text
    choices = [choice for event in events for choice in event.get("choices", [])]
    content = "".join(choice["delta"].get("content") or "" for choice in choices)
    assert content == (message.get("content") or "")
    names = [
        call["function"]["name"]
        for choice in choices
        for call in choice["delta"].get("tool_calls", [])
        if call.get("function", {}).get("name")
    ]
    assert names == [
        call["function"]["name"] for call in message.get("tool_calls") or []
    ]
    for marker in ["<|tool_call>", "[Calling tool"]:
        if marker not in (message.get("content") or ""):
            assert marker not in content


@pytest.mark.parametrize("text", [GEMMA, CALLING, PROSE, PLAIN])
def test_logprobs_content_parity(text, monkeypatch):
    from rapid_mlx.api.models import TokenLogProb
    from rapid_mlx.routes import chat

    # One logprob entry per engine delta, labelled with that delta's text.
    monkeypatch.setattr(
        chat,
        "_extract_streaming_token_logprobs",
        lambda output, *a, **k: [
            TokenLogProb(token=output.new_text, logprob=-0.5, top_logprobs=[])
        ],
    )
    client = client_for(ReplayEngine(list(text), "stop"))
    get_config().tool_call_parser = "hermes"
    response = client.post(
        "/v1/chat/completions",
        json={
            **_payload("missing"),
            "stream": True,
            "logprobs": True,
        },
    )
    assert response.status_code == 200, response.text
    choices = [
        choice for event in sse(response.text) for choice in event.get("choices", [])
    ]
    assert "".join(
        choice["delta"].get("content") or "" for choice in choices
    ) == sanitize_content_for_stream(text)
    tokens = []
    for choice in choices:
        if not choice.get("logprobs"):
            continue
        chunk_tokens = "".join(e["token"] for e in choice["logprobs"]["content"])
        # Held text is released together with the logprobs of its tokens.
        assert sanitize_content_for_stream(chunk_tokens) == (
            choice["delta"].get("content") or ""
        ), choice
        tokens.append(chunk_tokens)
    # Every generated token's logprob is reported exactly once, in order.
    assert "".join(tokens) == text


@pytest.mark.parametrize("inline_finish", [False, True])
def test_sanitizer_held_prefix_precedes_structured_tool_call(inline_finish):
    class OrderedEngine(ReplayEngine):
        async def stream_chat(self, messages, **kwargs):
            yield GenerationOutput(
                text="before [", new_text="before [", channel="content", finished=False
            )
            yield GenerationOutput(
                text="",
                new_text="",
                channel="tool_call",
                tool_calls=[{"name": "get_weather", "arguments": '{"city":"Paris"}'}],
                finished=inline_finish,
                finish_reason="stop" if inline_finish else None,
            )
            if not inline_finish:
                yield GenerationOutput(
                    text="", new_text="", finished=True, finish_reason="stop"
                )

    client = client_for(OrderedEngine([], "stop"))
    get_config().tool_call_parser = "hermes"
    response = client.post(
        "/v1/chat/completions", json={**_payload("auto"), "stream": True}
    )
    assert response.status_code == 200, response.text
    choices = [
        choice for event in sse(response.text) for choice in event.get("choices", [])
    ]
    call_index = next(
        i for i, choice in enumerate(choices) if choice["delta"].get("tool_calls")
    )
    assert (
        "".join(choice["delta"].get("content") or "" for choice in choices[:call_index])
        == "before ["
    )
    assert not any(choice["delta"].get("content") for choice in choices[call_index:])
