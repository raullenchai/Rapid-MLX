# SPDX-License-Identifier: Apache-2.0
"""Token exhaustion must survive tool-call parsing on every serving surface."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.routes.anthropic import router as anthropic_router
from rapid_mlx.routes.chat import router as chat_router
from rapid_mlx.routes.responses import router as responses_router

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
# The captured SSE contains only the function anchor with arguments="{".
# This XML prefix is sufficient to produce exactly that delta. The capture
# does not contain raw model token IDs; replay the implied decoded sequence.
HEADER = ["<tool_call>", "\n", "<function=", "get_weather", ">", "\n"]
PREFIX = HEADER + ["<parameter=city>"]
COMPLETE = HEADER + [
    "<parameter=city>",
    "Paris",
    "</parameter>",
    "</function>",
    "</tool_call>",
]


class ReplayEngine:
    preserve_native_tool_format = False
    supports_guided_generation = False
    is_mllm = False
    tokenizer = None

    def __init__(self, deltas, finish_reason, separate_finish=True):
        self.deltas = deltas
        self.finish_reason = finish_reason
        self.separate_finish = separate_finish

    def build_prompt(self, messages, **kwargs):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        accumulated = ""
        deltas = self.deltas + ([""] if self.separate_finish else [])
        for i, delta in enumerate(deltas):
            accumulated += delta
            finished = i == len(deltas) - 1
            yield GenerationOutput(
                text=accumulated,
                new_text=delta,
                prompt_tokens=341,
                completion_tokens=16,
                finished=finished,
                finish_reason=self.finish_reason if finished else None,
            )

    async def chat(self, messages, **kwargs):
        text = "".join(self.deltas)
        return GenerationOutput(
            text=text,
            raw_text=text,
            prompt_tokens=341,
            completion_tokens=16,
            finished=True,
            finish_reason=self.finish_reason,
        )


class StructuredReplayEngine(ReplayEngine):
    calls = [{"name": "web_search", "arguments": '{"query":"x"}'}]

    async def stream_chat(self, messages, **kwargs):
        yield GenerationOutput(
            text="answer",
            new_text="answer",
            channel="content",
            finished=False,
        )
        yield GenerationOutput(
            text="",
            new_text="wire call",
            channel="tool_call",
            finished=False,
            tool_calls=self.calls,
        )
        yield GenerationOutput(
            text="answer",
            new_text="",
            channel="content",
            finished=True,
            finish_reason="stop",
        )

    async def chat(self, messages, **kwargs):
        return GenerationOutput(
            text="answer",
            raw_text="answer",
            tool_calls=self.calls,
            finished=True,
            finish_reason="stop",
        )


def client_for(engine):
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "qwen3_coder_xml"
    app = FastAPI()
    for router in (chat_router, responses_router, anthropic_router):
        app.include_router(router)
    return TestClient(app)


def sse(text):
    return [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.parametrize(
    "choice",
    ["auto", "required", {"type": "function", "function": {"name": "get_weather"}}],
)
@pytest.mark.parametrize("separate_finish", [True, False])
@pytest.mark.parametrize(
    "finish,complete",
    [("length", False), ("length", True), ("stop", True), ("tool_calls", True)],
)
def test_chat_stream_preserves_engine_length(choice, separate_finish, finish, complete):
    client = client_for(
        ReplayEngine(COMPLETE if complete else PREFIX, finish, separate_finish)
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "stream": True,
            "tools": TOOLS,
            "tool_choice": choice,
            "max_tokens": 16,
        },
    )
    assert response.status_code == 200, response.text
    events = sse(response.text)
    assert not any("error" in event for event in events), response.text
    choices = [choice for event in events for choice in event.get("choices", [])]
    calls = [
        call for choice in choices for call in choice["delta"].get("tool_calls", [])
    ]
    assert calls[0]["function"]["name"] == "get_weather"
    arguments = "".join(call["function"].get("arguments", "") for call in calls)
    if complete:
        assert json.loads(arguments) == {"city": "Paris"}
    else:
        assert arguments == "{"
    assert [
        choice["finish_reason"] for choice in choices if choice.get("finish_reason")
    ] == ["length" if finish == "length" else "tool_calls"]
    assert response.text.endswith("data: [DONE]\n\n")


# Only token exhaustion blocks the upgrade; other engine finishes (including a
# channel-routed engine's own "tool_calls") still report a completed call.
@pytest.mark.parametrize("finish", ["length", "stop", "tool_calls"])
@pytest.mark.parametrize("route", ["chat/completions", "responses", "messages"])
def test_nonstream_tool_finish_reason_parity(route, finish):
    client = client_for(ReplayEngine(COMPLETE, finish))
    payload = {"model": "test-model", "stream": False}
    if route == "responses":
        payload.update(
            input="Weather in Paris?",
            tools=[{"type": "function", **TOOLS[0]["function"]}],
            max_output_tokens=16,
        )
    elif route == "messages":
        payload.update(
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            max_tokens=16,
            tools=[
                {
                    "name": "get_weather",
                    "input_schema": TOOLS[0]["function"]["parameters"],
                }
            ],
        )
    else:
        payload.update(
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            tools=TOOLS,
            max_tokens=16,
        )
    response = client.post(f"/v1/{route}", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    if route == "responses":
        assert any(item["type"] == "function_call" for item in body["output"])
        assert body["status"] == ("incomplete" if finish == "length" else "completed")
    elif route == "messages":
        assert any(item["type"] == "tool_use" for item in body["content"])
        assert body["stop_reason"] == (
            "max_tokens" if finish == "length" else "tool_use"
        )
    else:
        assert body["choices"][0]["message"]["tool_calls"]
        assert body["choices"][0]["finish_reason"] == (
            "length" if finish == "length" else "tool_calls"
        )


@pytest.mark.parametrize("finish", ["length", "stop"])
@pytest.mark.parametrize("fallback", [False, True])
def test_terminal_call_and_finalize_recovery_preserve_length(
    monkeypatch, finish, fallback
):
    """Cover a call on the final engine chunk and a call recovered after it."""
    if fallback:
        from rapid_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

        # Simulate a streaming parser missing a format that its full-text parser
        # supports. The route must preserve length when finalize recovers it.
        monkeypatch.setattr(
            Qwen3CoderToolParser, "extract_tool_calls_streaming", lambda *a, **kw: None
        )
    client = client_for(
        ReplayEngine(["".join(COMPLETE)], finish, separate_finish=False)
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Weather?"}],
            "stream": True,
            "tools": TOOLS,
            "max_tokens": 16,
        },
    )
    assert response.status_code == 200, response.text
    choices = [
        choice for event in sse(response.text) for choice in event.get("choices", [])
    ]
    assert any(choice["delta"].get("tool_calls") for choice in choices)
    assert [
        choice["finish_reason"] for choice in choices if choice.get("finish_reason")
    ] == ["length" if finish == "length" else "tool_calls"]


def test_nonstream_partial_call_preserves_length():
    client = client_for(ReplayEngine(PREFIX, "length"))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": TOOLS,
            "max_tokens": 16,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize("tools", [None, []])
@pytest.mark.parametrize("finish", ["stop", "length"])
@pytest.mark.parametrize("route", ["chat/completions", "responses", "messages"])
def test_no_tools_preserves_buffered_markup_and_chat_stream_parity(
    route, finish, tools
):
    """All buffered routes preserve markup; chat SSE emits identical text."""
    from rapid_mlx.config import get_config

    client = client_for(ReplayEngine(COMPLETE, finish))
    get_config().tool_call_parser = "hermes"
    raw = "".join(COMPLETE)
    payload = {"model": "test-model", "tools": tools}
    if route == "responses":
        payload.update(input="Weather?", max_output_tokens=32)
    else:
        payload.update(
            messages=[{"role": "user", "content": "Weather?"}], max_tokens=32
        )

    response = client.post(f"/v1/{route}", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    if route == "chat/completions":
        choice = body["choices"][0]
        buffered = choice["message"]["content"]
        assert not choice["message"].get("tool_calls")
        assert choice["finish_reason"] == finish
    elif route == "responses":
        assert all(item["type"] != "function_call" for item in body["output"])
        buffered = "".join(
            part["text"] for item in body["output"] for part in item.get("content", [])
        )
        assert body["status"] == ("incomplete" if finish == "length" else "completed")
    else:
        assert all(item["type"] == "text" for item in body["content"])
        buffered = "".join(item["text"] for item in body["content"])
        assert body["stop_reason"] == (
            "max_tokens" if finish == "length" else "end_turn"
        )
    assert buffered == raw

    if route != "chat/completions":
        return
    response = client.post(f"/v1/{route}", json={**payload, "stream": True})
    assert response.status_code == 200, response.text
    events = sse(response.text)
    assert not any("error" in event for event in events), response.text
    choices = [choice for event in events for choice in event.get("choices", [])]
    assert not any(choice["delta"].get("tool_calls") for choice in choices)
    streamed = "".join(choice["delta"].get("content") or "" for choice in choices)
    assert [c["finish_reason"] for c in choices if c.get("finish_reason")] == [finish]
    assert streamed == buffered == raw


@pytest.mark.parametrize(
    "parser", ["hermes", "qwen3_coder_xml", None, "minimax-inferred"]
)
@pytest.mark.parametrize("request_kind", ["missing", "none", "empty", "unknown"])
def test_tool_helper_without_tools_bypasses_all_text_parsers(
    monkeypatch, parser, request_kind
):
    from rapid_mlx.api.models import ChatCompletionRequest
    from rapid_mlx.service import helpers

    cfg = reset_config()
    cfg.enable_auto_tool_choice = parser != "minimax-inferred"
    cfg.tool_call_parser = None if parser == "minimax-inferred" else parser
    cfg.reasoning_parser_name = "minimax" if parser == "minimax-inferred" else None
    kwargs = (
        {}
        if request_kind == "missing"
        else {"tools": [] if request_kind == "empty" else None}
    )
    request = (
        None
        if request_kind == "unknown"
        else ChatCompletionRequest(
            model="test", messages=[{"role": "user", "content": "Weather?"}], **kwargs
        )
    )
    monkeypatch.setattr(
        helpers, "parse_tool_calls", lambda *a, **kw: pytest.fail("fallback ran")
    )
    monkeypatch.setattr(
        helpers.ToolParserManager,
        "get_tool_parser",
        lambda *a, **kw: pytest.fail("text parser ran"),
    )
    helpers._undeclared_tool_dropped_this_parse.set(True)
    raw = "".join(COMPLETE)
    assert helpers._parse_tool_calls_with_parser(raw, request) == (raw, None)
    assert not helpers._was_undeclared_tool_dropped()
    assert helpers._parse_tool_calls_with_parser(None, request) == ("", None)


@pytest.mark.parametrize("parser", ["hermes", None])
@pytest.mark.parametrize("choice", ["auto", "none"])
def test_tool_helper_with_tools_keeps_parsing(choice, parser):
    from rapid_mlx.api.models import ChatCompletionRequest
    from rapid_mlx.service import helpers

    cfg = reset_config()
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = parser
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "Weather?"}],
        tools=TOOLS,
        tool_choice=choice,
    )
    content, calls = helpers._parse_tool_calls_with_parser("".join(COMPLETE), request)
    assert not content
    if choice == "none":
        assert calls is None
    else:
        assert calls and calls[0].function.name == "get_weather"


@pytest.mark.parametrize("parser", ["hermes", "qwen3_coder_xml", None])
@pytest.mark.parametrize(
    "tools_mode", ["missing", "null", "empty", "unknown", "declared"]
)
def test_no_tools_guard_drops_structured_engine_channel(parser, tools_mode):
    from rapid_mlx.api.models import ChatCompletionRequest
    from rapid_mlx.service import helpers

    cfg = reset_config()
    cfg.tool_call_parser = parser
    request = (
        None
        if tools_mode == "unknown"
        else ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "Weather?"}],
            **(
                {}
                if tools_mode == "missing"
                else {
                    "tools": TOOLS
                    if tools_mode == "declared"
                    else ([] if tools_mode == "empty" else None)
                }
            ),
        )
    )
    helpers._undeclared_tool_dropped_this_parse.set(True)
    content, calls = helpers._parse_tool_calls_with_parser(
        "answer",
        request,
        structured_tool_calls=[
            {"name": "get_weather", "arguments": '{"city":"Paris"}'}
        ],
    )
    assert content == "answer"
    assert not helpers._was_undeclared_tool_dropped()
    if tools_mode == "declared":
        assert calls and calls[0].function.name == "get_weather"
    else:
        assert calls is None


@pytest.mark.parametrize("tools", [None, []])
@pytest.mark.parametrize("terminal_channel", ["tool_call", "content"])
@pytest.mark.parametrize("separate_finish", [False, True])
def test_no_tools_drops_structured_stream_channel(
    tools, terminal_channel, separate_finish
):
    from rapid_mlx.api.models import ChatCompletionRequest
    from rapid_mlx.service.postprocessor import StreamingPostProcessor

    cfg = reset_config()
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "hermes"
    request = ChatCompletionRequest(
        model="test", messages=[{"role": "user", "content": "Weather?"}], tools=tools
    )
    processor = StreamingPostProcessor(cfg, request=request)
    assert processor.tool_parser is None
    events = []
    for output in [
        GenerationOutput(
            text="answer", new_text="answer", channel="content", finished=False
        ),
        GenerationOutput(
            text="partial", new_text="partial", channel="tool_call", finished=False
        ),
        GenerationOutput(
            text="wire call",
            new_text="wire call",
            channel="tool_call",
            tool_calls=[{"name": "get_weather", "arguments": "{}"}],
            finished=not separate_finish,
            finish_reason=None if separate_finish else "tool_calls",
        ),
    ]:
        events.extend(processor.process_chunk(output))
    if separate_finish:
        events.extend(
            processor.process_chunk(
                GenerationOutput(
                    text="",
                    new_text="",
                    channel=terminal_channel,
                    finished=True,
                    finish_reason="tool_calls",
                )
            )
        )
    events.extend(processor.finalize())
    assert not any(event.type == "tool_call" for event in events)
    assert "".join(event.content or "" for event in events) == "answer"
    assert [event.finish_reason for event in events if event.type == "finish"] == [
        "stop"
    ]
    assert not processor.undeclared_tool_dropped


@pytest.mark.parametrize("route", ["responses", "messages"])
@pytest.mark.parametrize("tools_mode", ["missing", "null", "empty", "auto", "none"])
@pytest.mark.parametrize("chunks", ["whole", "tags", "characters", "structured"])
def test_stream_tool_markup_matches_buffered_request_capability(
    route, tools_mode, chunks
):
    """Unsolicited XML is text; declared tools retain structured-call behavior."""
    from rapid_mlx.config import get_config

    raw = (
        "<tool_call>\n<function=web_search>\n<parameter=query>\nx\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    deltas = {
        "whole": [raw],
        "tags": raw.splitlines(keepends=True),
        "characters": list(raw),
    }
    engine = (
        StructuredReplayEngine([], "tool_calls")
        if chunks == "structured"
        else ReplayEngine(deltas[chunks], "stop")
    )
    client = client_for(engine)
    get_config().tool_call_parser = "hermes"
    declared = tools_mode in ("auto", "none")
    payload = {"model": "test-model"}
    function = {
        "name": "web_search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    if route == "responses":
        payload.update(input="Search", max_output_tokens=128)
        tool = {"type": "function", **function}
        choice = tools_mode
    else:
        payload.update(messages=[{"role": "user", "content": "Search"}], max_tokens=128)
        tool = {"name": function["name"], "input_schema": function["parameters"]}
        choice = {"type": tools_mode}
    if declared:
        payload.update(tools=[tool], tool_choice=choice)
    elif tools_mode != "missing":
        payload["tools"] = None if tools_mode == "null" else []

    buffered_response = client.post(f"/v1/{route}", json=payload)
    assert buffered_response.status_code == 200, buffered_response.text
    body = buffered_response.json()
    response = client.post(f"/v1/{route}", json={**payload, "stream": True})
    assert response.status_code == 200, response.text
    events = sse(response.text)
    assert not any("error" in event for event in events), response.text
    expect_call = tools_mode == "auto"
    if route == "responses":
        buffered = "".join(
            part["text"] for item in body["output"] for part in item.get("content", [])
        )
        streamed = "".join(
            event["delta"]
            for event in events
            if event["type"] == "response.output_text.delta"
        )
        terminal = events[-1]
        no_answer = tools_mode == "none" and chunks != "structured"
        assert body["status"] == ("failed" if no_answer else "completed")
        if no_answer:
            # Existing behavior: hidden calls with no final answer fail SSE.
            assert terminal["type"] == "response.failed", response.text
            assert terminal["response"]["error"]["code"] == "model_no_final_answer"
        else:
            assert terminal["type"] == "response.completed", response.text
            assert terminal["response"]["status"] == "completed"
        for items in (
            body["output"],
            terminal["response"]["output"],
            [event["item"] for event in events if "item" in event],
        ):
            calls = [item for item in items if item["type"] == "function_call"]
            assert bool(calls) == expect_call
            for call in calls:
                assert call["name"] == "web_search"
                # The item-added anchor precedes the argument deltas.
                if call["arguments"]:
                    assert json.loads(call["arguments"]) == {"query": "x"}
    else:
        buffered = "".join(
            item["text"] for item in body["content"] if item["type"] == "text"
        )
        streamed = "".join(
            event["delta"]["text"]
            for event in events
            if event["type"] == "content_block_delta"
            and event["delta"]["type"] == "text_delta"
        )
        assert events[-1]["type"] == "message_stop", response.text
        stop = next(
            event["delta"]["stop_reason"]
            for event in events
            if event["type"] == "message_delta"
        )
        assert (
            stop == body["stop_reason"] == ("tool_use" if expect_call else "end_turn")
        )
        blocks = [
            event["content_block"]
            for event in events
            if event["type"] == "content_block_start"
        ]
        for items in (body["content"], blocks):
            calls = [item for item in items if item["type"] == "tool_use"]
            assert bool(calls) == expect_call
            assert all(call["name"] == "web_search" for call in calls)
        if expect_call:
            arguments = "".join(
                event["delta"]["partial_json"]
                for event in events
                if event["type"] == "content_block_delta"
                and event["delta"]["type"] == "input_json_delta"
            )
            assert json.loads(arguments) == {"query": "x"}
    # Messages removes converted tools for none, so both paths preserve text.
    parsing_enabled = declared and not (route == "messages" and tools_mode == "none")
    expected = "answer" if chunks == "structured" else ("" if parsing_enabled else raw)
    assert streamed == buffered == expected
