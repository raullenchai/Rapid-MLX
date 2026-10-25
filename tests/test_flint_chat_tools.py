# SPDX-License-Identifier: Apache-2.0
"""#5257: Flint's native tool prompt must survive both HTTP surfaces unchanged."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.routes import chat, responses
from rapid_mlx.utils.chat_template import apply_chat_template, is_flint_chat_template
from tests.test_system_prompt_with_tools import _Template

FUNCTION = {
    "name": "get_value",
    "description": "Look up a value; the first result supplies the next key.",
    "parameters": {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
}
TOOLS = [{"type": "function", "function": FUNCTION}]
USER = {
    "role": "user",
    "content": "Look up first, then the returned key. Report the result.",
}


class _Engine:
    preserve_native_tool_format = True
    is_mllm = False
    supports_guided_generation = False

    def __init__(self):
        self.tokenizer = _Template("flint_chat_template.jinja")
        self.calls = []

    def build_prompt(self, messages, tools=None, enable_thinking=None, **kwargs):
        return apply_chat_template(
            self.tokenizer,
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            chat_template_kwargs=kwargs.get("chat_template_kwargs"),
            add_generation_prompt=kwargs.get("add_generation_prompt", True),
        )

    def _output(self, messages, kwargs):
        self.calls.append(
            SimpleNamespace(
                messages=messages,
                kwargs=kwargs,
                prompt=self.build_prompt(messages, **kwargs),
            )
        )
        text = "<tool_call>get_value<arg_key>key</arg_key><arg_value>first</arg_value></tool_call>"
        return GenerationOutput(
            text=text,
            raw_text=text,
            new_text=text,
            prompt_tokens=100,
            completion_tokens=20,
            finished=True,
            finish_reason="stop",
        )

    async def chat(self, messages, **kwargs):
        return self._output(messages, kwargs)

    async def stream_chat(self, messages, **kwargs):
        yield self._output(messages, kwargs)


@pytest.fixture
def client_engine(monkeypatch):
    engine = _Engine()
    cfg = reset_config()
    cfg.engine = engine
    # Template identity, not an alias/path match, also handles local copies.
    cfg.model_name = "local-copy"
    cfg.model_registry = None
    cfg.tool_call_parser = "glm47"
    cfg.enable_auto_tool_choice = True

    async def wait(coro, *_args, **_kwargs):
        return await coro

    for route in (chat, responses):
        monkeypatch.setattr(route, "_check_admission_or_503", lambda _engine: None)
        monkeypatch.setattr(route, "_wait_with_disconnect", wait)
    app = FastAPI()
    app.include_router(chat.router)
    app.include_router(responses.router)
    with TestClient(app) as client:
        yield client, engine
    reset_config()


def _histories(turn, system):
    messages = (
        [{"role": "system", "content": "Use tools accurately."}] if system else []
    ) + [USER]
    inputs = list(messages)
    for n in range(turn):
        arguments = '{"key":"first"}' if n == 0 else '{"key":"second-secret"}'
        result = '{"next_key":"second-secret"}' if n == 0 else '{"value":42}'
        call_id = f"call_{n}"
        messages += [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "get_value", "arguments": arguments},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": result},
        ]
        inputs += [
            {
                "type": "function_call",
                "call_id": call_id,
                "name": "get_value",
                "arguments": arguments,
            },
            {"type": "function_call_output", "call_id": call_id, "output": result},
        ]
    return messages, inputs


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking", [None, False, True])
@pytest.mark.parametrize("turn", [0, 1, 2])
@pytest.mark.parametrize("system", [False, True])
@pytest.mark.parametrize("choice", [None, "auto"])
def test_native_rendered_prompt_parity(
    client_engine, stream, thinking, turn, system, choice
):
    client, engine = client_engine
    messages, inputs = _histories(turn, system)
    common = {"model": "local-copy", "stream": stream, "temperature": 0}
    if thinking is not None:
        common["chat_template_kwargs"] = {"enable_thinking": thinking}
    if choice is not None:
        common["tool_choice"] = choice
    for endpoint, payload in [
        ("chat/completions", {"messages": messages, "tools": TOOLS, "max_tokens": 128}),
        (
            "responses",
            {
                "input": inputs,
                "tools": [{"type": "function", **FUNCTION}],
                "max_output_tokens": 128,
            },
        ),
    ]:
        result = client.post("/v1/" + endpoint, json={**common, **payload})
        assert result.status_code == 200, result.text
    a, b = engine.calls
    assert a.prompt == b.prompt
    expected = apply_chat_template(
        engine.tokenizer,
        messages,
        tools=TOOLS,
        enable_thinking=thinking if thinking is not None else False,
    )
    assert a.prompt == expected
    assert "IMPORTANT:" not in a.prompt
    assert a.prompt.count("# Tools") == 1
    assert '"name": "get_value"' in a.prompt
    assert a.prompt.count("<|observation|><tool_response>") == turn
    assert a.kwargs["tools"] == b.kwargs["tools"] == TOOLS
    assert a.kwargs.get("enable_thinking") == b.kwargs.get("enable_thinking")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "choice",
    ["required", {"type": "function", "function": {"name": "get_value"}}, "none"],
)
def test_explicit_choice_is_preserved(client_engine, stream, choice):
    client, engine = client_engine
    result = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-copy",
            "messages": [USER],
            "tools": TOOLS,
            "tool_choice": choice,
            "stream": stream,
            "max_tokens": 128,
        },
    )
    assert result.status_code == 200, result.text
    call = engine.calls[0]
    if choice == "none":
        assert "# Tools" not in call.prompt
        assert "get_value" not in call.prompt
    else:
        assert "CRITICAL:" in call.prompt
        assert "# Tools" in call.prompt


def test_custom_template_keeps_existing_auto_instruction(client_engine):
    client, engine = client_engine
    # A custom/future template is outside the validated checkpoint exception.
    engine.tokenizer.chat_template += "\n{# custom #}"
    result = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-copy",
            "messages": [USER],
            "tools": TOOLS,
            "max_tokens": 128,
        },
    )
    assert result.status_code == 200
    assert "IMPORTANT:" in engine.calls[0].prompt


@pytest.mark.parametrize("shape", ["str", "dict"])
def test_template_identity_and_explicit_override(shape):
    tokenizer = _Template("flint_chat_template.jinja")
    native = tokenizer.chat_template
    if shape == "dict":
        tokenizer.chat_template = {"default": "custom", "tool_use": native}
    assert is_flint_chat_template(tokenizer, tools=TOOLS)
    assert not is_flint_chat_template(
        tokenizer, tools=TOOLS, chat_template_kwargs={"chat_template": "custom"}
    )
    assert is_flint_chat_template(
        None, tools=TOOLS, chat_template_kwargs={"chat_template": native}
    )
    assert not is_flint_chat_template(None, tools=TOOLS)
