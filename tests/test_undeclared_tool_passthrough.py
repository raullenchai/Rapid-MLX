# SPDX-License-Identifier: Apache-2.0
"""Unknown model tool names reach clients so they can return tool errors."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.routes.anthropic import router as messages_router
from rapid_mlx.routes.chat import router as chat_router
from tests.qwen3coder_stream_harness import deltas
from tests.test_empty_rejected_tool_reply import TOOLS, _Engine

NAME = "mcp__canopiner__run_sql"
XML = f"<tool_call><function={NAME}><parameter=query>select 1</parameter></function></tool_call>"
FORMATS = [
    ("qwen3_coder_xml", XML),
    ("nemotron", XML),
    (
        "hermes",
        f'<tool_call>{{"name":"{NAME}","arguments":{{"query":"select 1"}}}}</tool_call>',
    ),
    (
        "glm47",
        f"<tool_call>{NAME}<arg_key>query</arg_key><arg_value>select 1</arg_value></tool_call>",
    ),
    (
        "hy_v3",
        f'<tool_call:opensource>{NAME}<tool_sep:opensource>{{"query":"select 1"}}<end_of_tool_call:opensource>',
    ),
    (
        "k2_horizon",
        f"<ifm|tool_calls><ifm|tool_call>{NAME}<ifm|arg_key>query</ifm|arg_key><ifm|arg_value>select 1</ifm|arg_value></ifm|tool_call></ifm|tool_calls>",
    ),
]


@pytest.fixture(autouse=True)
def config():
    cfg = reset_config()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    yield cfg
    reset_config()


@pytest.mark.parametrize("parser,wire", FORMATS)
@pytest.mark.parametrize("api", ["chat", "messages"])
@pytest.mark.parametrize("size", [None, 0, 1, 7])
def test_unknown_tool_reaches_client(config, parser, wire, api, size):
    config.tool_call_parser = parser
    config.engine = _Engine(
        "I'll query it. " + wire, deltas=deltas("I'll query it. " + wire, size or 0)
    )
    app = FastAPI()
    app.include_router(chat_router if api == "chat" else messages_router)
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Query it"}],
        "stream": size is not None,
        "tools": TOOLS
        if api == "chat"
        else [
            {"name": "write_file", "input_schema": {"type": "object", "properties": {}}}
        ],
    }
    if api == "messages":
        body["max_tokens"] = 100
    response = TestClient(app).post(
        "/v1/chat/completions" if api == "chat" else "/v1/messages", json=body
    )
    assert response.status_code == 200, response.text
    if size is None:
        data = response.json()
        if api == "chat":
            choice = data["choices"][0]
            calls = choice["message"]["tool_calls"]
            assert choice["finish_reason"] == "tool_calls"
            content = choice["message"].get("content") or ""
            name = calls[0]["function"]["name"]
            args = json.loads(calls[0]["function"]["arguments"])
        else:
            calls = [b for b in data["content"] if b["type"] == "tool_use"]
            assert data["stop_reason"] == "tool_use"
            content = "".join(b["text"] for b in data["content"] if b["type"] == "text")
            name, args = calls[0]["name"], calls[0]["input"]
        assert len(calls) == 1
        assert calls[0]["id"]
    else:
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        if api == "chat":
            choices = [c for event in events for c in event["choices"]]
            calls = [tc for c in choices for tc in c["delta"].get("tool_calls", [])]
            anchors = [tc for tc in calls if tc.get("function", {}).get("name")]
            assert choices[-1]["finish_reason"] == "tool_calls"
            content = "".join(c["delta"].get("content") or "" for c in choices)
            assert len(anchors) == 1
            name = anchors[0]["function"]["name"]
            assert anchors[0]["id"]
            args = json.loads(
                "".join(tc["function"].get("arguments") or "" for tc in calls)
            )
        else:
            anchors = [
                e["content_block"]
                for e in events
                if e["type"] == "content_block_start"
                and e["content_block"]["type"] == "tool_use"
            ]
            assert len(anchors) == 1
            name = anchors[0]["name"]
            assert anchors[0]["id"]
            args = json.loads(
                "".join(
                    e["delta"]["partial_json"]
                    for e in events
                    if e["type"] == "content_block_delta"
                    and e["delta"]["type"] == "input_json_delta"
                )
            )
            content = "".join(
                e["delta"]["text"]
                for e in events
                if e["type"] == "content_block_delta"
                and e["delta"]["type"] == "text_delta"
            )
            assert [e for e in events if e["type"] == "message_delta"][-1]["delta"][
                "stop_reason"
            ] == "tool_use"
    assert name == NAME
    assert args == {"query": "select 1"}
    assert "<" not in content


@pytest.mark.parametrize("parser,wire", FORMATS)
@pytest.mark.parametrize("api", ["chat", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", ["none", "named"])
def test_explicit_tool_choice_still_constrains_output(
    config, parser, wire, api, stream, choice
):
    config.tool_call_parser = parser
    text = wire + wire.replace(NAME, "write_file")
    config.engine = _Engine(text, deltas=deltas(text, 1))
    app = FastAPI()
    app.include_router(chat_router if api == "chat" else messages_router)
    tool_choice = (
        (
            "none"
            if choice == "none"
            else {"type": "function", "function": {"name": "write_file"}}
        )
        if api == "chat"
        else (
            {"type": "none"}
            if choice == "none"
            else {"type": "tool", "name": "write_file"}
        )
    )
    response = TestClient(app).post(
        "/v1/chat/completions" if api == "chat" else "/v1/messages",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Use a tool"}],
            "stream": stream,
            "max_tokens": 100,
            "tools": TOOLS
            if api == "chat"
            else [
                {
                    "name": "write_file",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": tool_choice,
        },
    )
    if (
        choice == "named"
        and api == "chat"
        and not stream
        and response.status_code == 422
    ):
        assert "tool_choice pinned function" in response.json()["detail"]
        return
    if (
        choice == "named"
        and api == "messages"
        and "produced no tool_calls" in response.text
    ):
        # A forced Anthropic choice with no call to the pinned tool is an
        # error, never a leak: 422 before headers, an SSE error once streaming.
        if stream:
            assert response.status_code == 200
            assert "event: error" in response.text
        else:
            assert response.status_code == 422
        return
    assert response.status_code == 200, response.text
    if stream:
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        if api == "chat":
            names = [
                tc["function"]["name"]
                for event in events
                for c in event["choices"]
                for tc in c["delta"].get("tool_calls", [])
                if tc.get("function", {}).get("name")
            ]
        else:
            names = [
                e["content_block"]["name"]
                for e in events
                if e["type"] == "content_block_start"
                and e["content_block"]["type"] == "tool_use"
            ]
    elif api == "chat":
        names = [
            tc["function"]["name"]
            for tc in response.json()["choices"][0]["message"].get("tool_calls", [])
        ]
    else:
        names = [
            b["name"] for b in response.json()["content"] if b["type"] == "tool_use"
        ]
    assert names == ([] if choice == "none" else ["write_file"])


@pytest.mark.parametrize("size", [0, 1, 7])
@pytest.mark.parametrize("known_first", [False, True])
def test_qwen_mixed_names_preserve_call_order_and_arguments(size, known_first):
    from tests.qwen3coder_stream_harness import non_stream, stream

    known = XML.replace(NAME, "write_file")
    first, second = (known, XML) if known_first else (XML, known)
    request = {"tools": TOOLS}
    expected = ["write_file", NAME] if known_first else [NAME, "write_file"]
    for calls, content in (
        non_stream(first + second, request),
        stream(first + second, request, size),
    ):
        assert [name for name, _ in calls] == expected
        assert all(json.loads(args) == {"query": "select 1"} for _, args in calls)
        assert not content.strip()


@pytest.mark.parametrize("api", ["chat", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_structured_unknown_tool_reaches_client(config, api, stream):
    class StructuredEngine(_Engine):
        async def chat(self, messages, **kwargs):
            output = await super().chat(messages, **kwargs)
            output.tool_calls = [{"name": NAME, "arguments": '{"query":"select 1"}'}]
            return output

        async def stream_chat(self, messages, **kwargs):
            async for output in super().stream_chat(messages, **kwargs):
                output.channel = "tool_call"
                output.tool_calls = [
                    {"name": NAME, "arguments": '{"query":"select 1"}'}
                ]
                yield output

    config.tool_call_parser = "qwen3_coder_xml"
    config.engine = StructuredEngine("")
    app = FastAPI()
    app.include_router(chat_router if api == "chat" else messages_router)
    response = TestClient(app).post(
        "/v1/chat/completions" if api == "chat" else "/v1/messages",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Query"}],
            "stream": stream,
            "max_tokens": 100,
            "tools": TOOLS
            if api == "chat"
            else [{"name": "write_file", "input_schema": {"type": "object"}}],
        },
    )
    assert response.status_code == 200, response.text
    assert NAME in response.text
    assert ('"tool_calls"' if api == "chat" else '"tool_use"') in response.text


def test_qwen_stream_retains_named_choice_when_only_first_chunk_has_request():
    from rapid_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    parser = Qwen3CoderToolParser(None)
    request = {
        "tools": TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "write_file"}},
    }
    previous = ""
    calls = []
    for index, piece in enumerate(deltas(XML, 1)):
        current = previous + piece
        result = parser.extract_tool_calls_streaming(
            previous, current, piece, request=request if index == 0 else None
        )
        calls.extend((result or {}).get("tool_calls", []))
        previous = current
    assert calls == []
