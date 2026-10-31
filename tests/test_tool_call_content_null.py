# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible null content for tool-call-only assistant turns."""

import importlib.util
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.api.models import AssistantMessage, ChatCompletionChunkDelta, ToolCall
from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.routes.chat import router

CALL = ToolCall(
    id="call_1", type="function", function={"name": "weather", "arguments": "{}"}
)


@pytest.mark.parametrize("content", [None, ""])
def test_assistant_tool_call_content_null(content):
    data = json.loads(
        AssistantMessage(content=content, tool_calls=[CALL]).model_dump_json(
            exclude_none=True
        )
    )
    assert "content" in data and data["content"] is None
    assert data["tool_calls"]


def test_assistant_empty_without_tools():
    assert (
        json.loads(AssistantMessage(content=None).model_dump_json(exclude_none=True))[
            "content"
        ]
        == ""
    )


def test_delta_tool_call_content_null():
    delta = ChatCompletionChunkDelta(
        tool_calls=[
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "weather", "arguments": "{}"},
            }
        ]
    )
    data = json.loads(delta.model_dump_json(exclude_none=True))
    assert "content" in data and data["content"] is None
    assert data["tool_calls"]
    reasoning = json.loads(
        ChatCompletionChunkDelta(reasoning_content="thinking").model_dump_json(
            exclude_none=True
        )
    )
    assert reasoning["content"] == ""


class ToolEngine:
    preserve_native_tool_format = True
    supports_guided_generation = False
    is_mllm = False
    _is_mllm = False
    tokenizer = SimpleNamespace(chat_template="JSON")
    text = '<tool_call>\n{"name": "weather", "arguments": {}}\n</tool_call>'

    def build_prompt(self, messages, **kwargs):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        return GenerationOutput(
            text=self.text,
            raw_text=self.text,
            new_text=self.text,
            finished=True,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
        )

    async def stream_chat(self, messages, **kwargs):
        yield GenerationOutput(
            text=self.text,
            new_text=self.text,
            finished=False,
            prompt_tokens=10,
            completion_tokens=20,
        )
        yield GenerationOutput(
            text=self.text,
            raw_text=self.text,
            new_text="",
            finished=True,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_chat_route_tool_call_content_null(stream, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "0")

    # The fake engine does not need admission or disconnect monitoring.
    async def wait_for_fake_engine(coro, *args, **kwargs):
        return await coro

    monkeypatch.setattr(
        "rapid_mlx.routes.chat._wait_with_disconnect", wait_for_fake_engine
    )
    cfg = reset_config()
    cfg.engine = ToolEngine()
    cfg.model_name = "test"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = "hermes"
    cfg.enable_auto_tool_choice = True
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "auto",
                "stream": stream,
            },
        )
    assert response.status_code == 200, response.text
    if not stream:
        body = response.json()
        choice = body["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        message = choice["message"]
        assert message["tool_calls"]
        assert "content" in message and message["content"] is None
        if importlib.util.find_spec("openai"):
            from openai.types.chat import ChatCompletion

            assert (
                ChatCompletion.model_validate(body).choices[0].message.content is None
            )
    else:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert not any("error" in chunk for chunk in chunks)
        choices = [choice for chunk in chunks for choice in chunk.get("choices", [])]
        deltas = [choice["delta"] for choice in choices]
        tool_deltas = [delta for delta in deltas if delta.get("tool_calls")]
        assert tool_deltas
        assert any(choice.get("finish_reason") == "tool_calls" for choice in choices)
        assert all(
            "content" in delta and delta["content"] is None for delta in tool_deltas
        )
        content = None
        for delta in deltas:
            if delta.get("content") is not None:
                content = (content or "") + delta["content"]
        assert not content
