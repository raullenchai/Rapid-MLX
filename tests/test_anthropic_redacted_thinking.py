# SPDX-License-Identifier: Apache-2.0
"""Opaque assistant history must be accepted without entering prompts (#5741)."""

import copy
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rapid_mlx.api.anthropic_adapter import anthropic_to_openai
from rapid_mlx.api.anthropic_models import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicRequest,
)
from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.routes.anthropic import router

OPAQUE_DATA = "ENCRYPTEDDATA"


def _tool_loop():
    return {
        "model": "test-model",
        "max_tokens": 300,
        "tools": [
            {
                "name": "shell",
                "description": "Run a shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "Use the shell tool to run pwd."},
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": OPAQUE_DATA},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "shell",
                        "input": {"command": "pwd"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "/tmp\n",
                    }
                ],
            },
        ],
    }


@pytest.mark.parametrize("data", [OPAQUE_DATA, ""])
def test_redacted_thinking_validation(data):
    block = AnthropicContentBlock(type="redacted_thinking", data=data)
    assert block.data == data
    message = AnthropicMessage(role="assistant", content=[block])
    assert message.model_dump()["content"][0]["data"] == data


@pytest.mark.parametrize("fields", [{}, {"data": None}, {"data": 42}, {"data": {}}])
def test_redacted_thinking_requires_string_data(fields):
    with pytest.raises(ValidationError, match="data"):
        AnthropicContentBlock(type="redacted_thinking", **fields)


@pytest.mark.parametrize("role", ["user", "system"])
def test_redacted_thinking_remains_assistant_only(role):
    with pytest.raises(ValidationError, match="not allowed"):
        AnthropicMessage(
            role=role, content=[{"type": "redacted_thinking", "data": OPAQUE_DATA}]
        )


def test_unknown_block_type_still_rejected():
    with pytest.raises(ValidationError, match="not a recognized"):
        AnthropicContentBlock(type="unknown_thinking", data=OPAQUE_DATA)


@pytest.mark.parametrize("with_tool", [False, True])
def test_adapter_drops_redacted_thinking_like_thinking(with_tool):
    body = _tool_loop()
    if not with_tool:
        body["messages"][1]["content"] = body["messages"][1]["content"][:1]
        body["messages"][2]["content"] = "Continue."
    converted = anthropic_to_openai(AnthropicRequest(**body)).model_dump()
    plain = copy.deepcopy(body)
    plain["messages"][1]["content"][0] = {
        "type": "thinking",
        "thinking": "prior reasoning",
    }
    assert converted == anthropic_to_openai(AnthropicRequest(**plain)).model_dump()
    assert OPAQUE_DATA not in json.dumps(converted)
    assistant = converted["messages"][1]
    assert assistant["content"] == ""
    if with_tool:
        assert assistant["tool_calls"][0]["id"] == "toolu_1"
        assert converted["messages"][2]["tool_call_id"] == "toolu_1"
        assert converted["messages"][2]["content"] == "/tmp\n"


class _RecordingEngine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))

    def __init__(self):
        self.calls = []
        self.prompts = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        prompt = json.dumps({"messages": messages, "tools": tools})
        self.prompts.append(prompt)
        return prompt

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        self.build_prompt(messages, tools=kwargs.get("tools"))
        return GenerationOutput(
            text="The working directory is /tmp.",
            new_text="The working directory is /tmp.",
            tokens=[1],
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )

    async def stream_chat(self, messages, **kwargs):
        yield await self.chat(messages, **kwargs)


@pytest.fixture
def client_engine():
    cfg = reset_config()
    engine = _RecordingEngine()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.reasoning_parser = None
    cfg.tool_call_parser = None
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    try:
        with TestClient(app) as client:
            yield client, engine
    finally:
        reset_config()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
def test_tool_loop_returns_200(client_engine, stream, thinking):
    client, engine = client_engine
    body = _tool_loop()
    body["stream"] = stream
    if thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": 128}
    response = client.post("/v1/messages", json=body)
    assert response.status_code == 200, response.text
    if stream:
        assert "event: message_stop" in response.text
        assert "event: error" not in response.text
    else:
        assert response.json()["content"] == [
            {"type": "text", "text": "The working directory is /tmp."}
        ]
    assert len(engine.calls) == 1
    assert "toolu_1" in json.dumps(engine.calls)
    assert engine.prompts
    assert all(OPAQUE_DATA not in prompt for prompt in engine.prompts)


def test_count_tokens_ignores_redacted_payload(client_engine):
    client, engine = client_engine
    body = _tool_loop()
    response = client.post("/v1/messages/count_tokens", json=body)
    assert response.status_code == 200, response.text
    assert engine.prompts
    assert response.json()["input_tokens"] == len(engine.prompts[-1].encode())
    assert all(OPAQUE_DATA not in prompt for prompt in engine.prompts)
    body["messages"][1]["content"].pop(0)
    without_redacted = client.post("/v1/messages/count_tokens", json=body)
    assert without_redacted.status_code == 200, without_redacted.text
    assert response.json() == without_redacted.json()
