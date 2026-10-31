# SPDX-License-Identifier: Apache-2.0
"""Tool-name length boundaries on the Messages and OpenAI lanes."""

import json
import sys
from contextvars import ContextVar
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rapid_mlx.api.models import ToolDefinition
from rapid_mlx.config import reset_config
from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.output_router_harmony import _RECIPIENT_SHAPE
from rapid_mlx.routes.anthropic import router
from rapid_mlx.tool_parsers.lfm_tool_parser import LfmToolParser


class _Tokenizer:
    chat_template = ""

    def encode(self, text, **kwargs):
        return list(text.encode())


class _Engine:
    preserve_native_tool_format = False
    tokenizer = _Tokenizer()

    def __init__(self, text="ok"):
        self.text = text

    async def chat(self, messages, **kwargs):
        return SimpleNamespace(
            text=self.text,
            raw_text=self.text,
            tool_calls=None,
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
            reasoning_text="",
        )

    async def stream_chat(self, messages, **kwargs):
        yield SimpleNamespace(
            new_text=self.text, prompt_tokens=10, completion_tokens=5, tool_calls=None
        )


@pytest.fixture
def client_factory(monkeypatch):
    batched = ModuleType("rapid_mlx.engine.batched")
    batched._admission_engine_context = ContextVar("test_admission", default=None)
    batched._admission_token_context = ContextVar("test_admission_token", default=None)
    monkeypatch.setitem(sys.modules, "rapid_mlx.engine.batched", batched)

    def make(text="ok"):
        cfg = reset_config()
        cfg.engine = _Engine(text)
        cfg.model_name = "test-model"
        cfg.no_thinking = True
        cfg.reasoning_parser_name = None
        cfg.model_registry = None
        cfg.tool_call_parser = "hermes"
        app = FastAPI()
        install_exception_handlers(app)
        app.include_router(router)
        return TestClient(app)

    yield make
    reset_config()


def _body(name, stream=False):
    return {
        "model": "test-model",
        "max_tokens": 32,
        "stream": stream,
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"name": name, "input_schema": {"type": "object", "properties": {}}}],
    }


def _events(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]


@pytest.mark.parametrize(
    "name,status",
    [
        ("a" * 64, 200),
        ("b" * 65, 200),
        ("c" * 128, 200),
        ("d" * 129, 400),
        ("jina: jina_reader", 400),
    ],
)
def test_messages_name_boundaries(client_factory, name, status):
    response = client_factory().post("/v1/messages", json=_body(name))
    assert response.status_code == status, response.text
    if status == 400:
        assert name not in response.text
        assert "Anthropic Messages API" in response.text


@pytest.mark.parametrize("length,status", [(65, 200), (128, 200), (129, 400)])
def test_stream_name_boundaries(client_factory, length, status):
    response = client_factory().post("/v1/messages", json=_body("a" * length, True))
    assert response.status_code == status, response.text
    if status == 200:
        assert any(e["type"] == "message_stop" for e in _events(response))


@pytest.mark.parametrize("length,status", [(128, 200), (129, 400)])
def test_count_tokens_name_boundaries(client_factory, length, status):
    response = client_factory().post(
        "/v1/messages/count_tokens", json=_body("a" * length)
    )
    assert response.status_code == status, response.text


@pytest.mark.parametrize("stream", [False, True])
def test_long_name_round_trip(client_factory, stream):
    name = "tool_" + "x" * 95
    text = "<tool_call>" + json.dumps({"name": name, "arguments": {}}) + "</tool_call>"
    client = client_factory(text)
    body = _body(name, stream)
    response = client.post("/v1/messages", json=body)
    assert response.status_code == 200, response.text
    if stream:
        blocks = [
            e["content_block"]
            for e in _events(response)
            if e["type"] == "content_block_start"
        ]
    else:
        blocks = response.json()["content"]
    tool = next(b for b in blocks if b["type"] == "tool_use")
    assert tool["name"] == name
    body["messages"] += [
        {"role": "assistant", "content": [tool]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool["id"], "content": "ok"}
            ],
        },
    ]
    body["tool_choice"] = {"type": "tool", "name": name}
    response = client.post("/v1/messages", json=body)
    assert response.status_code == 200, response.text
    if stream:
        assert any(e["type"] == "message_stop" for e in _events(response))


def test_openai_limit():
    ToolDefinition(function={"name": "a" * 64})
    with pytest.raises(ValidationError, match="1-64"):
        ToolDefinition(function={"name": "a" * 65})


def test_harmony_recipient_limit():
    assert _RECIPIENT_SHAPE.fullmatch("functions." + "a" * 128)
    assert not _RECIPIENT_SHAPE.fullmatch("functions." + "a" * 129)


def test_lfm_long_name():
    name = "a" * 100
    result = LfmToolParser().extract_tool_calls(f'[{name}({{"x": 1}})]')
    assert result.tools_called
    assert result.tool_calls[0]["name"] == name
