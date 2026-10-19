# SPDX-License-Identifier: Apache-2.0
"""Content sanitization must be independent of engine delta boundaries (#4925)."""

import random
import re

import pytest

from rapid_mlx.api import utils
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
    # The whole-text pattern ends a Calling span at its first ``]``.
    "pre [Calling tool: f([1, [2]])] post",
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


@pytest.mark.parametrize("text", UNIT_TEXTS)
def test_sanitizer_random_chunking(text):
    rng = random.Random(4925)
    for _ in range(50):
        cuts = sorted(rng.sample(range(len(text) + 1), min(4, len(text) + 1)))
        deltas = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)])]
        sanitizer = StreamingContentSanitizer()
        actual = "".join(sanitizer.process(delta) for delta in deltas)
        assert actual + sanitizer.flush() == sanitize_content_for_stream(text), deltas


@pytest.mark.parametrize(
    ("opener", "closer"),
    [("<|tool_call>call:x{", "}<tool_call|>"), ("[Calling tool: x(", ")]")],
)
def test_sanitizer_open_block_is_not_rescanned(monkeypatch, opener, closer):
    """A long unclosed block costs O(delta) per delta, not a full rescan."""
    calls = []
    real = utils._FINAL_SANITIZER

    class _Counting:
        def match(self, text, pos):
            calls.append(pos)
            return real.match(text, pos)

        def sub(self, repl, text):
            return real.sub(repl, text)

    monkeypatch.setattr(utils, "_FINAL_SANITIZER", _Counting())
    sanitizer = StreamingContentSanitizer()
    assert sanitizer.process("pre " + opener) == "pre "
    scans = len(calls)
    for _ in range(200):
        assert sanitizer.process("ab") == ""
    assert len(calls) == scans
    assert sanitizer.holding
    released = "".join(sanitizer.process(char) for char in closer + " post")
    text = "pre " + opener + "ab" * 200 + closer + " post"
    assert "pre " + released + sanitizer.flush() == sanitize_content_for_stream(text)
    assert not sanitizer.holding


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


@pytest.mark.parametrize("text", [GEMMA, CALLING, PROSE, PLAIN, "hello <tail"])
@pytest.mark.parametrize("chunking", ["whole", "characters", "words"])
def test_logprobs_content_parity(text, chunking, monkeypatch):
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
    if chunking == "whole":
        deltas = [text]
    elif chunking == "characters":
        deltas = list(text)
    else:
        deltas = [part for part in re.split(r"(?<=[ <])", text) if part]
    client = client_for(ReplayEngine(deltas, "stop"))
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
    tokens = ""
    content = ""
    for choice in choices:
        content += choice["delta"].get("content") or ""
        if not choice.get("logprobs"):
            continue
        tokens += "".join(e["token"] for e in choice["logprobs"]["content"])
        # A token's logprob never arrives before the chunk completing its text.
        assert content.startswith(sanitize_content_for_stream(tokens)), choice
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


def test_logprobs_reported_once_for_multi_content_output(monkeypatch):
    from rapid_mlx.api.models import TokenLogProb
    from rapid_mlx.domain.events import StreamEvent
    from rapid_mlx.routes import chat
    from rapid_mlx.service.postprocessor import StreamingPostProcessor

    monkeypatch.setattr(
        chat,
        "_extract_streaming_token_logprobs",
        lambda output, *a, **k: [
            TokenLogProb(token=output.new_text, logprob=-0.5, top_logprobs=[])
        ],
    )
    process_chunk = StreamingPostProcessor.process_chunk

    def two_content_events(self, output):
        if output.new_text == "ab":
            return [
                StreamEvent(type="content", content="a"),
                StreamEvent(type="content", content="b"),
            ]
        return process_chunk(self, output)

    monkeypatch.setattr(StreamingPostProcessor, "process_chunk", two_content_events)
    client = client_for(ReplayEngine(["ab"], "stop"))
    response = client.post(
        "/v1/chat/completions",
        json={**_payload("missing"), "stream": True, "logprobs": True},
    )
    assert response.status_code == 200, response.text
    tokens = [
        entry["token"]
        for event in sse(response.text)
        for choice in event.get("choices", [])
        for entry in (choice.get("logprobs") or {}).get("content", [])
    ]
    assert tokens == ["ab", ""]


@pytest.mark.parametrize("last", ["x<", "x [Calli", "x <|tool_call>abc"])
def test_logprobs_finish_chunk_carry_their_held_text(monkeypatch, last):
    """A finish-bearing delta's held text and its logprob share the last chunk."""
    from rapid_mlx.api.models import TokenLogProb
    from rapid_mlx.routes import chat

    monkeypatch.setattr(
        chat,
        "_extract_streaming_token_logprobs",
        lambda output, *a, **k: [
            TokenLogProb(token=output.new_text, logprob=-0.5, top_logprobs=[])
        ],
    )

    class FinishWithTextEngine(ReplayEngine):
        async def stream_chat(self, messages, **kwargs):
            yield GenerationOutput(text="a ", new_text="a ", finished=False)
            yield GenerationOutput(
                text="a " + last, new_text=last, finished=True, finish_reason="stop"
            )

    client = client_for(FinishWithTextEngine([], "stop"))
    response = client.post(
        "/v1/chat/completions",
        json={**_payload("missing"), "stream": True, "logprobs": True},
    )
    assert response.status_code == 200, response.text
    choices = [
        choice for event in sse(response.text) for choice in event.get("choices", [])
    ]
    final = next(choice for choice in choices if choice.get("finish_reason"))
    assert final["delta"].get("content") == sanitize_content_for_stream(last)
    assert [e["token"] for e in final["logprobs"]["content"]] == [last]


@pytest.mark.parametrize("held", ["[", "<|", "[Calli"])
def test_logprobs_released_held_text_excludes_tool_call_tokens(monkeypatch, held):
    """Text released ahead of a tool call carries only its own logprobs (#4957)."""
    from rapid_mlx.api.models import TokenLogProb
    from rapid_mlx.routes import chat

    monkeypatch.setattr(
        chat,
        "_extract_streaming_token_logprobs",
        lambda output, *a, **k: [
            TokenLogProb(token=output.new_text, logprob=-0.5, top_logprobs=[])
        ],
    )
    markup = '<tool_call>{"name":"get_weather"}</tool_call>'

    class HeldThenToolEngine(ReplayEngine):
        async def stream_chat(self, messages, **kwargs):
            yield GenerationOutput(
                text="ok " + held, new_text="ok " + held, finished=False
            )
            yield GenerationOutput(
                text="ok " + held + markup,
                new_text=markup,
                channel="tool_call",
                tool_calls=[{"name": "get_weather", "arguments": '{"city":"Paris"}'}],
                finished=False,
            )
            yield GenerationOutput(
                text="", new_text="", finished=True, finish_reason="stop"
            )

    client = client_for(HeldThenToolEngine([], "stop"))
    get_config().tool_call_parser = "hermes"
    response = client.post(
        "/v1/chat/completions",
        json={**_payload("auto"), "stream": True, "logprobs": True},
    )
    assert response.status_code == 200, response.text
    choices = [
        choice for event in sse(response.text) for choice in event.get("choices", [])
    ]
    call_index = next(
        i for i, choice in enumerate(choices) if choice["delta"].get("tool_calls")
    )
    contents = [choice["delta"].get("content") or "" for choice in choices]
    released = sanitize_content_for_stream(held)
    assert released
    assert "".join(contents[:call_index]) == "ok " + released
    assert contents[call_index - 1] == released
    logprobs = choices[call_index - 1]["logprobs"]["content"]
    assert [e["token"] for e in logprobs] == ["ok " + held]
    tokens = [
        entry["token"]
        for choice in choices
        if choice["delta"].get("content")
        for entry in (choice.get("logprobs") or {}).get("content", [])
    ]
    assert markup not in tokens


def test_logprobs_held_narration_with_tool_call_keeps_its_logprobs(monkeypatch):
    """An output whose own narration is held and released keeps its logprobs."""
    from rapid_mlx.api.models import TokenLogProb
    from rapid_mlx.routes import chat

    monkeypatch.setattr(
        chat,
        "_extract_streaming_token_logprobs",
        lambda output, *a, **k: [
            TokenLogProb(token=output.new_text, logprob=-0.5, top_logprobs=[])
        ],
    )

    class NarrationAndToolEngine(ReplayEngine):
        async def stream_chat(self, messages, **kwargs):
            yield GenerationOutput(
                text="[",
                new_text="[",
                channel="content",
                tool_calls=[{"name": "get_weather", "arguments": '{"city":"Paris"}'}],
                finished=False,
            )
            yield GenerationOutput(
                text="", new_text="", finished=True, finish_reason="stop"
            )

    client = client_for(NarrationAndToolEngine([], "stop"))
    get_config().tool_call_parser = "hermes"
    response = client.post(
        "/v1/chat/completions",
        json={**_payload("auto"), "stream": True, "logprobs": True},
    )
    assert response.status_code == 200, response.text
    choices = [
        choice for event in sse(response.text) for choice in event.get("choices", [])
    ]
    released = next(c for c in choices if c["delta"].get("content") == "[")
    assert "[" in [e["token"] for e in released["logprobs"]["content"]]
