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
