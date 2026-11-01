# SPDX-License-Identifier: Apache-2.0
"""Count tokens from the same rendered prompt used by text and vision chat."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.api.anthropic_adapter import anthropic_to_openai
from rapid_mlx.api.anthropic_models import AnthropicRequest
from rapid_mlx.api.tool_calling import convert_tools_for_template
from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.engine.batched import BatchedEngine
from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.routes.anthropic import _prepare_anthropic_engine_messages, router
from rapid_mlx.service.helpers import count_prompt_tokens


class _Tokenizer:
    chat_template = "deterministic role markers and tool definitions"
    bos_token = None

    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split())))

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=False, **kw
    ):
        parts = ["<start>"]
        if tools:
            parts.extend(["<tools>", json.dumps(tools, sort_keys=True), "</tools>"])
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    "<image>" if part.get("type") == "image" else part.get("text", "")
                    for part in content
                )
            parts.extend([f"<{message['role']}>", content, "<end>"])
        if add_generation_prompt:
            parts.append("<assistant>")
        return " ".join(parts)


@pytest.fixture(params=[True, False], ids=["mllm", "text"])
def engine(request, monkeypatch):
    engine = object.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = request.param
    engine._model_name = "test-model"
    engine._tokenizer = _Tokenizer()
    engine._processor = (
        SimpleNamespace(
            tokenizer=engine._tokenizer,
            chat_template=_Tokenizer.chat_template,
            apply_chat_template=engine._tokenizer.apply_chat_template,
        )
        if request.param
        else None
    )
    # Stop at generation; no model, scheduler, cache, or parser is needed.
    monkeypatch.setattr(engine, "_needs_prefix_boundary_snapshot", lambda: False)
    monkeypatch.setattr(engine, "_prompt_parse_context", lambda *a, **kw: None)
    monkeypatch.setattr(engine, "_align_owned_stop_scope", lambda *a: None)
    monkeypatch.setattr(
        engine,
        "generate",
        AsyncMock(return_value=GenerationOutput(text="ok", finished=True)),
    )
    return engine


def _client(engine):
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = None
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    return TestClient(app)


_TOOL = {
    "name": "get_weather",
    "description": "Get current weather",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


@pytest.mark.parametrize("kind", ["no-tools", "one-tool", "system-tool", "image-tool"])
def test_count_tokens_matches_generation_prompt(engine, monkeypatch, kind):
    client = _client(engine)
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Weather in Tokyo please"}],
    }
    if kind != "no-tools":
        body["tools"] = [_TOOL]
    if kind == "system-tool":
        body["system"] = "You are a helpful weather assistant"
    if kind == "image-tool":
        if not engine.is_mllm:
            pytest.skip("Image placeholder rendering belongs to the vision lane")
        body["messages"][0]["content"] = [
            {"type": "text", "text": "Describe this weather map"},
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/map.png"},
            },
        ]

    response = client.post("/v1/messages/count_tokens", json=body)
    assert response.status_code == 200, response.text
    openai_request = anthropic_to_openai(AnthropicRequest(**body, max_tokens=1))
    messages, _, _ = _prepare_anthropic_engine_messages(openai_request, engine)
    kwargs = {
        "tools": convert_tools_for_template(openai_request.tools)
        if openai_request.tools
        else None,
        "enable_thinking": False,
    }
    # Drive real chat independently of the count route and capture exactly
    # what it hands downstream, without loading a model or running generation.
    render_chat = Mock(wraps=engine._render_chat_prompt)
    monkeypatch.setattr(engine, "_render_chat_prompt", render_chat)
    asyncio.run(engine.chat(messages, **kwargs))
    render_chat.assert_called_once()
    prompt = engine.generate.call_args.kwargs["prompt"]
    assert response.json()["input_tokens"] == count_prompt_tokens(engine, prompt)
    assert engine.render_request_prompt(messages, **kwargs) == prompt
    assert "<user>" in prompt and "<assistant>" in prompt
    if kind != "no-tools":
        assert "get_weather" in prompt
    if kind == "system-tool":
        assert "<system>" in prompt
    if kind == "image-tool":
        assert "<image>" in prompt
        assert engine.generate.call_args.kwargs["images"] == [
            "https://example.com/map.png"
        ]


def test_render_request_prompt_requires_loaded_engine(engine):
    engine._loaded = False
    with pytest.raises(RuntimeError, match="Engine not loaded"):
        engine.render_request_prompt([{"role": "user", "content": "hello"}])


def test_build_prompt_still_refuses_mllm(engine):
    if engine.is_mllm:
        with pytest.raises(RuntimeError, match="not supported for MLLM"):
            engine.build_prompt([{"role": "user", "content": "hello"}])


def test_count_tokens_accepts_token_id_prompt(engine, monkeypatch):
    prepared = Mock(return_value=([11, 22, 33, 44], None))
    monkeypatch.setattr(engine, "_prepare_harmony_no_thinking_prompt", prepared)
    response = _client(engine).post(
        "/v1/messages/count_tokens",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"input_tokens": 4}
    assert prepared.call_args.kwargs == {
        "enable_thinking": False,
        "has_tools": False,
        "as_token_ids": True,
    }


def test_zero_rendered_count_uses_legacy_fallback(engine, monkeypatch):
    count = Mock(return_value=0)
    monkeypatch.setattr("rapid_mlx.routes.anthropic.count_prompt_tokens", count)
    response = _client(engine).post(
        "/v1/messages/count_tokens",
        json={"messages": [{"role": "user", "content": "hello world"}]},
    )
    assert response.status_code == 200, response.text
    count.assert_called_once()
    assert response.json() == {"input_tokens": 2}


def test_shared_system_prompt_fallback_is_preserved(engine, monkeypatch):
    messages = [{"role": "system", "content": "shared instructions"}]
    monkeypatch.setattr(
        engine,
        "_apply_chat_template",
        Mock(side_effect=ValueError("No user query found in messages")),
    )
    monkeypatch.setattr(
        engine, "_shared_system_tools_prompt", Mock(return_value=([11, 22, 33], 2))
    )
    prepare = Mock(side_effect=AssertionError("Warm-up must skip Harmony preparation"))
    monkeypatch.setattr(engine, "_prepare_harmony_no_thinking_prompt", prepare)
    assert engine.render_request_prompt(messages) == [11, 22, 33]
    asyncio.run(engine.chat(messages))
    assert engine.generate.call_args.kwargs["prompt"] == [11, 22, 33]
    prepare.assert_not_called()
