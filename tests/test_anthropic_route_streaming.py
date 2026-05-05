# SPDX-License-Identifier: Apache-2.0
"""Route-level Anthropic streaming regressions."""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.routes.anthropic import router


class _ThinkingTemplateTokenizer:
    chat_template = "{% if add_generation_prompt %}<think>{% endif %}"


class _StreamingEngine:
    preserve_native_tool_format = False
    tokenizer = _ThinkingTemplateTokenizer()

    def __init__(self, deltas: list[str]):
        self._deltas = deltas
        self.calls = []

    async def stream_chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        for i, text in enumerate(self._deltas, start=1):
            yield SimpleNamespace(
                new_text=text,
                prompt_tokens=5,
                completion_tokens=i,
            )


def _make_client(engine: _StreamingEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.no_thinking = True
    cfg.reasoning_parser_name = None
    cfg.model_registry = None

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _parse_sse_data(response_text: str) -> list[dict]:
    events = []
    for raw_event in response_text.split("\n\n"):
        data_line = next(
            (line for line in raw_event.splitlines() if line.startswith("data: ")),
            None,
        )
        if not data_line:
            continue
        data = data_line.removeprefix("data: ")
        if data == "[DONE]":
            continue
        events.append(json.loads(data))
    return events


@pytest.fixture(autouse=True)
def _reset_server_config():
    reset_config()
    yield
    reset_config()


def test_anthropic_stream_route_no_thinking_template_answers_as_text():
    """Server no-thinking mode should keep direct answers as text blocks."""
    engine = _StreamingEngine(["Direct ", "answer"])
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "answer directly"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert engine.calls[0]["kwargs"]["enable_thinking"] is False

    events = _parse_sse_data(response.text)
    block_starts = [e for e in events if e.get("type") == "content_block_start"]
    assert [e["content_block"]["type"] for e in block_starts] == ["text"]

    text_deltas = [
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "text_delta"
    ]
    thinking_deltas = [
        e
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "thinking_delta"
    ]

    assert "".join(text_deltas) == "Direct answer"
    assert thinking_deltas == []
    assert any(
        e.get("type") == "message_delta"
        and e.get("delta", {}).get("stop_reason") == "end_turn"
        for e in events
    )
