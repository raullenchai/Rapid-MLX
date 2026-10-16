# SPDX-License-Identifier: Apache-2.0
"""A rejected lone tool call must produce a visible, safe chat response."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import get_config, reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
from rapid_mlx.routes.chat import router as chat_router
from rapid_mlx.service import helpers

NOTICE = "I couldn't complete the request because a needed tool isn't available."

REJECTED = (
    "<tool_call>\n<function=web_search>\n"
    "<parameter=query>today's price</parameter>\n"
    "</function>\n</tool_call>"
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


class _Request:
    tools = TOOLS

    def model_dump(self):
        return {"tools": TOOLS}


@pytest.mark.parametrize(
    "rejected",
    [REJECTED, "<tool_call>\n<function=delete_everything>\n</function>\n</tool_call>"],
)
def test_nonstream_rejected_only_call_has_safe_content(monkeypatch, rejected):
    cfg = get_config()
    monkeypatch.setattr(cfg, "enable_auto_tool_choice", True)
    monkeypatch.setattr(cfg, "tool_call_parser", "qwen3_coder_xml")
    content, calls = helpers._parse_tool_calls_with_parser(rejected, _Request())
    assert calls is None
    assert content == ""
    assert helpers._was_undeclared_tool_dropped()
    helpers._parse_tool_calls_with_parser("Hello", _Request())
    assert not helpers._was_undeclared_tool_dropped()


class _Engine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, text=REJECTED, deltas=None):
        self.text = text
        self.deltas = deltas or [text]

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        return GenerationOutput(
            text=self.text,
            prompt_tokens=4,
            completion_tokens=20,
            finished=True,
            finish_reason="stop",
        )

    async def stream_chat(self, messages, **kwargs):
        accumulated = ""
        for i, delta in enumerate(self.deltas):
            accumulated += delta
            last = i == len(self.deltas) - 1
            yield GenerationOutput(
                text=accumulated,
                new_text=delta,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=last,
                finish_reason="stop" if last else None,
                channel=None,
            )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "rejected",
    [REJECTED, "<tool_call>\n<function=delete_everything>\n</function>\n</tool_call>"],
)
@pytest.mark.parametrize("reasoning_parser", [None, "qwen3"])
def test_chat_reply_after_rejected_only_call(stream, rejected, reasoning_parser):
    cfg = reset_config()
    cfg.engine = _Engine(rejected)
    cfg.reasoning_parser_name = reasoning_parser
    cfg.reasoning_parser = (
        Qwen3ReasoningParser(tokenizer=None) if reasoning_parser else None
    )
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "qwen3_coder_xml"
    app = FastAPI()
    app.include_router(chat_router)
    try:
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": stream,
                "messages": [{"role": "user", "content": "What's the price?"}],
                "tools": TOOLS,
            },
        )
        assert response.status_code == 200, response.text
        if stream:
            events = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: {")
            ]
            choices = [choice for event in events for choice in event["choices"]]
            content = "".join((c["delta"].get("content") or "") for c in choices)
            calls = [c["delta"].get("tool_calls") for c in choices]
            assert choices[-1]["finish_reason"] == "stop"
        else:
            choice = response.json()["choices"][0]
            content = choice["message"]["content"]
            calls = choice["message"].get("tool_calls")
            assert choice["finish_reason"] == "stop"
        assert content == NOTICE
        assert not any(calls) if stream else calls is None
    finally:
        reset_config()


@pytest.mark.parametrize("stream", [False, True])
def test_rejected_call_after_reasoning_has_visible_notice(stream):
    cfg = reset_config()
    cfg.engine = _Engine("<think>Need to search.</think>" + REJECTED)
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser_name = "qwen3"
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "qwen3_coder_xml"
    app = FastAPI()
    app.include_router(chat_router)
    try:
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": stream,
                "enable_thinking": True,
                "messages": [{"role": "user", "content": "What's the price?"}],
                "tools": TOOLS,
            },
        )
        assert response.status_code == 200, response.text
        if stream:
            choices = [
                choice
                for line in response.text.splitlines()
                if line.startswith("data: {")
                for choice in json.loads(line[6:])["choices"]
            ]
            content = "".join((c["delta"].get("content") or "") for c in choices)
            assert choices[-1]["finish_reason"] == "stop"
        else:
            choice = response.json()["choices"][0]
            content = choice["message"]["content"]
            assert choice["finish_reason"] == "stop"
        assert content == NOTICE
    finally:
        reset_config()


def test_streamed_whitespace_before_rejected_call_still_gets_notice():
    cfg = reset_config()
    cfg.engine = _Engine(deltas=[" \n", REJECTED])
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "qwen3_coder_xml"
    app = FastAPI()
    app.include_router(chat_router)
    try:
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": True,
                "messages": [{"role": "user", "content": "What's the price?"}],
                "tools": TOOLS,
            },
        )
        assert response.status_code == 200, response.text
        choices = [
            choice
            for line in response.text.splitlines()
            if line.startswith("data: {")
            for choice in json.loads(line[6:])["choices"]
        ]
        content = "".join((c["delta"].get("content") or "") for c in choices)
        assert content.strip() == NOTICE
    finally:
        reset_config()


@pytest.mark.parametrize("stream", [False, True])
def test_required_tool_recovery_does_not_include_failure_notice(stream):
    cfg = reset_config()
    cfg.engine = _Engine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "qwen3_coder_xml"
    app = FastAPI()
    app.include_router(chat_router)
    try:
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": stream,
                "tool_choice": "required",
                "messages": [{"role": "user", "content": "Use a tool."}],
                "tools": TOOLS,
            },
        )
        assert response.status_code == 200, response.text
        if stream:
            choices = [
                choice
                for line in response.text.splitlines()
                if line.startswith("data: {")
                for choice in json.loads(line[6:])["choices"]
            ]
            content = "".join((c["delta"].get("content") or "") for c in choices)
            assert any(c["delta"].get("tool_calls") for c in choices)
        else:
            choice = response.json()["choices"][0]
            content = choice["message"].get("content") or ""
            assert choice["message"].get("tool_calls")
        assert NOTICE not in content
    finally:
        reset_config()


@pytest.mark.parametrize("stream", [False, True])
def test_structured_output_does_not_receive_prose_notice(stream):
    cfg = reset_config()
    cfg.engine = _Engine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "qwen3_coder_xml"
    app = FastAPI()
    app.include_router(chat_router)
    try:
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": stream,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": "Return JSON."}],
                "tools": TOOLS,
            },
        )
        assert response.status_code == 200, response.text
        if stream:
            choices = [
                choice
                for line in response.text.splitlines()
                if line.startswith("data: {")
                for choice in json.loads(line[6:])["choices"]
            ]
            content = "".join((c["delta"].get("content") or "") for c in choices)
        else:
            content = response.json()["choices"][0]["message"].get("content") or ""
        assert content == ""
    finally:
        reset_config()
