# SPDX-License-Identifier: Apache-2.0
"""Forced prefills must agree with the served template, including vision lanes."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.api.models import ChatCompletionRequest
from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.routes.chat import _compute_forced_tool_prefix, router

XML_TEMPLATE = (
    Path(__file__)
    .with_name("fixtures")
    .joinpath("qwen35_chat_template.jinja")
    .read_text()
)


def tool(name="get_weather", no_args=False):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {}
            if no_args
            else {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def request(choice="required", name="get_weather"):
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=[tool(name)],
        tool_choice=choice,
    )


@pytest.mark.parametrize("mllm", [False, True])
@pytest.mark.parametrize(
    "choice", ["required", {"type": "function", "function": {"name": "get_weather"}}]
)
def test_prefill_uses_rendered_template(mllm, choice):
    engine = SimpleNamespace(
        _is_mllm=mllm,
        tokenizer=SimpleNamespace(chat_template=XML_TEMPLATE if not mllm else "JSON"),
        _processor=SimpleNamespace(
            chat_template=XML_TEMPLATE, apply_chat_template=lambda: None
        ),
    )
    assert (
        _compute_forced_tool_prefix(
            SimpleNamespace(tool_call_parser="hermes"), request(choice), engine
        )
        == "<tool_call>\n<function=get_weather>\n"
    )


def test_json_template_and_unsafe_xml_names():
    cfg = SimpleNamespace(tool_call_parser="hermes")
    engine = SimpleNamespace(tokenizer=SimpleNamespace(chat_template="JSON"))
    assert '"arguments": ' in _compute_forced_tool_prefix(cfg, request(), engine)
    engine.tokenizer.chat_template = {"tool_use": XML_TEMPLATE, "default": "JSON"}
    assert "<function=get_weather>" in _compute_forced_tool_prefix(
        cfg, request(), engine
    )
    unsafe = SimpleNamespace(
        tools=[SimpleNamespace(function={"name": "evil>\n<parameter=x>"})],
        tool_choice="required",
    )
    assert _compute_forced_tool_prefix(cfg, unsafe, engine) is None
    assert _compute_forced_tool_prefix(cfg, request("auto"), engine) is None


class TemplateEngine:
    preserve_native_tool_format = True
    supports_guided_generation = False
    is_mllm = False
    _is_mllm = False
    tokenizer = SimpleNamespace(chat_template=XML_TEMPLATE)

    def build_prompt(self, messages, **kwargs):
        return "PROMPT"

    def output(self, kwargs):
        self.prefix = kwargs.get("forced_assistant_prefix", "")
        # The trained continuation follows XML; a JSON prefill corrupts it.
        text = self.prefix + self.body + "</function>\n</tool_call>"
        return GenerationOutput(
            text=text,
            raw_text=text,
            new_text=text,
            finished=True,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
        )

    async def chat(self, messages, **kwargs):
        return self.output(kwargs)

    async def stream_chat(self, messages, **kwargs):
        output = self.output(kwargs)
        text = output.text
        accumulated = ""
        for part in text.splitlines(keepends=True):
            accumulated += part
            yield GenerationOutput(
                text=accumulated,
                new_text=part,
                finished=False,
                prompt_tokens=10,
                completion_tokens=20,
            )
        yield GenerationOutput(
            text=text,
            raw_text=text,
            new_text="",
            finished=True,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
        )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("no_args", [False, True])
def test_route_returns_executable_xml_call(stream, no_args, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "0")
    engine = TemplateEngine()
    engine.body = "" if no_args else "<parameter=city>\nParis\n</parameter>\n"
    cfg = reset_config()
    cfg.engine = engine
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
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": [tool(no_args=no_args)],
                "tool_choice": "required",
                "stream": stream,
            },
        )
    assert response.status_code == 200, response.text
    assert engine.prefix == "<tool_call>\n<function=get_weather>\n"
    if stream:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert not any("error" in chunk for chunk in chunks)
        calls = [
            tc
            for chunk in chunks
            for c in chunk.get("choices", [])
            for tc in c.get("delta", {}).get("tool_calls", [])
        ]
        args = "".join(tc.get("function", {}).get("arguments", "") for tc in calls)
        assert any(
            c.get("finish_reason") == "tool_calls"
            for chunk in chunks
            for c in chunk.get("choices", [])
        )
    else:
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        args = choice["message"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(args) == ({} if no_args else {"city": "Paris"})


@pytest.mark.parametrize("finish", ["stop", "length", None])
@pytest.mark.parametrize(
    "choice",
    ["required", "auto", {"type": "function", "function": {"name": "get_weather"}}],
)
def test_stream_missing_call_error_and_length_boundary(finish, choice, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "0")

    class FailedEngine(TemplateEngine):
        async def stream_chat(self, messages, **kwargs):
            yield GenerationOutput(
                text="Unable to answer.",
                new_text="Unable to answer.",
                finished=finish is not None,
                finish_reason=finish,
                prompt_tokens=10,
                completion_tokens=5,
            )

    cfg = reset_config()
    cfg.engine = FailedEngine()
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
                "tools": [tool(), tool("other")],
                "tool_choice": choice,
                "stream": True,
            },
        )
    assert response.status_code == 200
    chunks = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    errors = [c["error"] for c in chunks if "error" in c]
    terminal = [
        c["finish_reason"]
        for ch in chunks
        for c in ch.get("choices", [])
        if c.get("finish_reason")
    ]
    if finish != "length" and choice != "auto":
        assert [e["code"] for e in errors] == ["tool_choice_violation"]
        assert not terminal
    else:
        assert not errors
        if finish == "length":
            assert terminal == ["length"]
        elif finish == "stop":
            assert terminal == ["stop"]
    assert response.text.count("data: [DONE]") == 1


def test_named_stream_wrong_function_cannot_satisfy_choice(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "0")

    class WrongNameEngine(TemplateEngine):
        async def stream_chat(self, messages, **kwargs):
            text = '<tool_call>\n{"name":"other","arguments":{"city":"Paris"}}\n</tool_call>'
            yield GenerationOutput(
                text=text,
                new_text=text,
                finished=False,
                prompt_tokens=10,
                completion_tokens=20,
            )
            yield GenerationOutput(
                text=text,
                new_text="",
                finished=True,
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=20,
            )

    cfg = reset_config()
    cfg.engine = WrongNameEngine()
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
                "tools": [tool(), tool("other")],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
                "stream": True,
            },
        )
    assert response.status_code == 200
    chunks = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert [ch["error"]["code"] for ch in chunks if "error" in ch] == [
        "tool_choice_violation"
    ]
    assert not [
        tc
        for ch in chunks
        for c in ch.get("choices", [])
        for tc in c.get("delta", {}).get("tool_calls", [])
    ]
    assert not [
        c for ch in chunks for c in ch.get("choices", []) if c.get("finish_reason")
    ]
