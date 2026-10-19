# SPDX-License-Identifier: Apache-2.0
"""Responses preserves model tool markup as ordinary text under tool_choice=none."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.api.models import ChatCompletionRequest
from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.routes.responses import _tool_parse_request, router

RAW = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
TOOL = {
    "type": "function",
    "name": "get_weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


class _Engine:
    preserve_native_tool_format = False
    supports_guided_generation = False
    is_mllm = False
    tokenizer = None

    def __init__(self):
        self.text = RAW
        self.tool_calls = None
        self.seen_tools = []

    def build_prompt(self, messages, **kwargs):
        return "PROMPT"

    def _output(self, text, delta="", finished=True):
        return GenerationOutput(
            text=text,
            new_text=delta,
            tool_calls=self.tool_calls if finished else None,
            finished=finished,
            finish_reason="stop" if finished else None,
            prompt_tokens=4,
            completion_tokens=32,
        )

    async def chat(self, messages, **kwargs):
        self.seen_tools.append(kwargs.get("tools"))
        return self._output(self.text)

    async def stream_chat(self, messages, **kwargs):
        self.seen_tools.append(kwargs.get("tools"))
        for index, delta in enumerate(self.text):
            yield self._output(self.text[: index + 1], delta, finished=False)
        yield self._output(self.text)


@pytest.fixture
def client():
    cfg = reset_config()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "hermes"
    engine = _Engine()
    cfg.engine = engine
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as http:
        yield http, engine
    reset_config()


def _request(client, *, stream=False, choice="none", **extra):
    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "Weather?",
            "tools": [TOOL],
            "tool_choice": choice,
            "stream": stream,
            "max_output_tokens": 128,
            **extra,
        },
    )
    assert response.status_code == 200, response.text
    return response


def _text(body):
    return "".join(
        part["text"]
        for item in body["output"]
        if item["type"] == "message"
        for part in item["content"]
        if part["type"] == "output_text"
    )


@pytest.mark.parametrize("structured", [False, True])
def test_none_preserves_markup_in_buffered_and_streaming_responses(client, structured):
    # None bypasses parsing in both paths, including engine-surfaced calls.
    http, engine = client
    if structured:
        engine.tool_calls = [{"name": "get_weather", "arguments": '{"city":"Paris"}'}]
    buffered = _request(http).json()
    response = _request(http, stream=True)
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    streamed = "".join(
        event["delta"]
        for event in events
        if event["type"] == "response.output_text.delta"
    )
    assert buffered["status"] == "completed"
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["status"] == "completed"
    assert streamed == _text(buffered) == _text(events[-1]["response"]) == RAW
    for items in (
        buffered["output"],
        events[-1]["response"]["output"],
        [event["item"] for event in events if "item" in event],
    ):
        assert not any(item["type"] == "function_call" for item in items)
    assert all(engine.seen_tools)  # Prompt rendering still receives declared tools.


@pytest.mark.parametrize("stream", [False, True])
def test_auto_still_parses_tool_markup(client, stream):
    # Auto retains ordinary function-call parsing on both paths.
    response = _request(client[0], choice="auto", stream=stream)
    body = (
        response.json() if not stream else json.loads(response.text.split("data: ")[-1])
    )
    if stream:
        assert body["type"] == "response.completed"
        body = body["response"]
    calls = [item for item in body["output"] if item["type"] == "function_call"]
    assert body["status"] == "completed"
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert json.loads(calls[0]["arguments"]) == {"city": "Paris"}


def test_none_strict_schema_probe_keeps_valid_json(client):
    # None also bypasses the tool-turn probe without changing schema validation.
    http, engine = client
    engine.text = '{"city":"Paris"}'
    body = _request(
        http,
        text={
            "format": {
                "type": "json_schema",
                "name": "weather",
                "strict": True,
                "schema": TOOL["parameters"],
            }
        },
    ).json()
    assert body["status"] == "completed"
    assert _text(body) == engine.text
    assert not any(item["type"] == "function_call" for item in body["output"])


@pytest.mark.parametrize("choice", ["none", "auto", None])
def test_parse_request_without_tools_is_unchanged(choice):
    # Requests without tools already treat model markup as text.
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hi"}],
        tool_choice=choice,
    )
    assert _tool_parse_request(request) is request
