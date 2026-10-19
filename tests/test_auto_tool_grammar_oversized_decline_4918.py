# SPDX-License-Identifier: Apache-2.0
"""Oversized auto tools stay free-form when the template's native wire wins."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from rapid_mlx.api.models import ChatCompletionRequest
from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.routes import chat

XML_TEMPLATE = (
    Path(__file__)
    .with_name("fixtures")
    .joinpath("qwen35_chat_template.jinja")
    .read_text()
)
JSON_TEMPLATE = """{# <function= <parameter= are inactive source markers #}
{% for message in messages %}
{{ message.role }}:{{ message.content or '' }}
{% for call in message.tool_calls or [] %}
<tool_call>{{ call.function | tojson }}</tool_call>
{% endfor %}
{% endfor %}"""


def engine(template=XML_TEMPLATE):
    return SimpleNamespace(tokenizer=SimpleNamespace(chat_template=template))


def tool(name, parameters=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }


def request(*, tools=None, choice="auto", thinking=None, history=True):
    messages = [{"role": "user", "content": "Find the weather."}]
    if history:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"query":"weather"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "Sunny.",
                },
            ]
        )
    kwargs = {}
    if thinking is not None:
        kwargs["chat_template_kwargs"] = {"enable_thinking": thinking}
    return ChatCompletionRequest(
        model="test",
        messages=messages,
        tools=tools or [tool("search")],
        tool_choice=choice,
        **kwargs,
    )


def cfg(parser="hermes"):
    return SimpleNamespace(tool_call_parser=parser)


def many_tools(count=300):
    return [tool(f"tool_{index}") for index in range(count)]


def oversized_bytes_tool():
    return tool(
        "large",
        {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "string",
                    "description": "x" * chat._TOOL_GRAMMAR_MAX_SCHEMA_BYTES,
                }
            },
        },
    )


@pytest.fixture(autouse=True)
def _restore_config():
    yield
    reset_config()


@pytest.fixture(autouse=True)
def constrain_tools(monkeypatch):
    from rapid_mlx.api import tool_grammar

    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "1")
    monkeypatch.setattr(tool_grammar, "are_single_special_tokens", lambda *args: True)


@pytest.mark.parametrize("tools", [many_tools(), [oversized_bytes_tool()]])
def test_gate_exempts_oversized_auto_native_xml(tools):
    req = request(tools=tools)
    assert chat._tool_grammar_eligible(cfg(), req) is False
    assert chat._enforce_tool_grammar_bounds_or_400(cfg(), req, engine()) is None


def test_json_wire_oversized_auto_still_400():
    with pytest.raises(HTTPException) as exc:
        chat._enforce_tool_grammar_bounds_or_400(
            cfg(), request(tools=many_tools()), engine(JSON_TEMPLATE)
        )
    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "choice",
    ["required", {"type": "function", "function": {"name": "tool_0"}}],
)
def test_forced_oversized_native_xml_still_400(choice):
    with pytest.raises(HTTPException) as exc:
        chat._enforce_tool_grammar_bounds_or_400(
            cfg(), request(tools=many_tools(), choice=choice), engine()
        )
    assert exc.value.status_code == 400


def test_opt_out_oversized_still_free_form(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_CONSTRAIN_TOOLS", "0")
    assert (
        chat._enforce_tool_grammar_bounds_or_400(
            cfg(), request(tools=many_tools(), choice="required"), engine()
        )
        is None
    )


@pytest.mark.parametrize("template", [XML_TEMPLATE, JSON_TEMPLATE])
@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("history", [True, False])
def test_bounded_probe_agrees_with_full_tools(template, thinking, history):
    req = request(
        tools=[tool("search"), tool("forecast")], thinking=thinking, history=history
    )
    minimal = [tool("search")]
    full = chat._template_renders_native_xml_tool_call(engine(template), req, "search")
    bounded = chat._template_renders_native_xml_tool_call(
        engine(template), req, "search", tools=minimal
    )
    assert bounded is full


def test_gate_probe_is_one_minimal_tool(monkeypatch):
    probe = Mock(wraps=chat._template_renders_native_xml_tool_call)
    monkeypatch.setattr(chat, "_template_renders_native_xml_tool_call", probe)
    req = request(tools=many_tools())

    chat._enforce_tool_grammar_bounds_or_400(cfg(), req, engine())

    sent_tools = probe.call_args.kwargs["tools"]
    assert sent_tools == [tool("tool_0")]
    assert sent_tools[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {},
    }


def test_helper_handles_dict_and_object_tools_without_reading_schemas():
    class Function:
        name = "object_tool"

        @property
        def parameters(self):
            raise AssertionError("parameter schema was touched")

    class Tool:
        function = Function()

    req = request()
    req.tools = [Tool()]
    assert chat._auto_tool_grammar_declined_for_native_wire(engine(), cfg(), req)
    req.tools = [tool("dict_tool")]
    assert chat._auto_tool_grammar_declined_for_native_wire(engine(), cfg(), req)


def test_helper_returns_false_for_non_auto_no_structure_and_unsafe_name():
    parser = SimpleNamespace(structure_info=lambda: None)
    assert not chat._auto_tool_grammar_declined_for_native_wire(
        engine(), cfg(), request(choice="required"), parser
    )
    assert not chat._auto_tool_grammar_declined_for_native_wire(
        engine(), cfg(), request(), parser
    )
    structured = SimpleNamespace(
        structure_info=lambda: lambda name: SimpleNamespace(begin="<tool_call>")
    )
    unsafe = request()
    unsafe.tools[0].function["name"] = "bad>name"
    assert not chat._auto_tool_grammar_declined_for_native_wire(
        engine(), cfg(), unsafe, structured
    )


def test_helper_scans_at_most_tool_count_cap(monkeypatch):
    structured = SimpleNamespace(
        structure_info=lambda: lambda name: SimpleNamespace(begin="<tool_call>")
    )
    req = request()
    req.tools = [tool("bad>name") for _ in range(chat._TOOL_GRAMMAR_MAX_TOOLS)]
    req.tools.append(tool("search"))
    probe = Mock(return_value=True)
    monkeypatch.setattr(chat, "_template_renders_native_xml_tool_call", probe)

    assert not chat._auto_tool_grammar_declined_for_native_wire(
        engine(), cfg(), req, structured
    )
    probe.assert_not_called()


def test_helper_returns_false_when_parser_construction_fails(monkeypatch):
    from rapid_mlx.tool_parsers import ToolParserManager

    class BrokenParser:
        @classmethod
        def supports_grammar(cls):
            return True

        def __init__(self, tokenizer):  # noqa: ARG002
            raise RuntimeError("broken parser constructor")

    monkeypatch.setattr(ToolParserManager, "get_tool_parser", lambda name: BrokenParser)
    assert not chat._auto_tool_grammar_declined_for_native_wire(
        engine(), cfg(), request()
    )
    with pytest.raises(HTTPException) as exc:
        chat._enforce_tool_grammar_bounds_or_400(
            cfg(), request(tools=many_tools()), engine()
        )
    assert exc.value.status_code == 400


def test_builder_uses_shared_decline_helper(monkeypatch):
    from rapid_mlx.api import tool_grammar

    shared = Mock(return_value=True)
    monkeypatch.setattr(chat, "_auto_tool_grammar_declined_for_native_wire", shared)
    monkeypatch.setattr(tool_grammar, "are_single_special_tokens", lambda *args: True)
    monkeypatch.setattr(tool_grammar, "get_lltokenizer", lambda tokenizer: object())
    build = Mock(return_value=None)
    monkeypatch.setattr(tool_grammar, "build_tool_grammar", build)

    assert chat._maybe_build_tool_grammar_processor(engine(), cfg(), request()) is None
    assert shared.call_args.kwargs["parser"] is not None
    build.assert_not_called()


def test_chat_route_oversized_auto_native_xml_returns_response(monkeypatch):
    async def await_direct(coro, raw_request, timeout):  # noqa: ARG001
        return await coro

    class Tokenizer:
        chat_template = XML_TEMPLATE
        model_max_length = 100_000
        bos_token = None

        def encode(self, text, add_special_tokens=True):  # noqa: ARG002
            return [0] * max(1, len(text) // 4)

    class Engine:
        is_mllm = False
        supports_guided_generation = False
        supports_tool_calls = True
        preserve_native_tool_format = False
        tokenizer = Tokenizer()

        def build_prompt(self, messages, tools=None, enable_thinking=None):  # noqa: ARG002
            return "prompt"

        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return GenerationOutput(
                text="ok",
                raw_text="ok",
                prompt_tokens=1,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                cached_tokens=0,
            )

    from rapid_mlx.middleware.auth import rate_limiter

    monkeypatch.setattr(rate_limiter, "enabled", False)
    monkeypatch.setattr(chat, "_wait_with_disconnect", await_direct)
    route_cfg = reset_config()
    route_cfg.engine = Engine()
    route_cfg.model_name = "test"
    route_cfg.model_registry = None
    route_cfg.tool_call_parser = "hermes"
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(chat.router)

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": many_tools(),
            "tool_choice": "auto",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "ok"


@pytest.mark.parametrize(
    "factory",
    [
        lambda name: None,
        lambda name: SimpleNamespace(begin=None),
        Mock(side_effect=ValueError("rejected name")),
    ],
)
def test_gate_keeps_400_when_parser_factory_fails(monkeypatch, factory):
    from rapid_mlx.tool_parsers import ToolParserManager

    class Parser:
        @classmethod
        def supports_grammar(cls):
            return True

        def __init__(self, tokenizer):  # noqa: ARG002
            pass

        def structure_info(self):
            return factory

    monkeypatch.setattr(ToolParserManager, "get_tool_parser", lambda name: Parser)
    with pytest.raises(HTTPException) as exc:
        chat._enforce_tool_grammar_bounds_or_400(
            cfg(), request(tools=many_tools()), engine()
        )
    assert exc.value.status_code == 400


def test_gate_keeps_400_when_probe_raises(monkeypatch):
    monkeypatch.setattr(
        chat,
        "_template_renders_native_xml_tool_call",
        Mock(side_effect=KeyError("template blew up")),
    )
    with pytest.raises(HTTPException) as exc:
        chat._enforce_tool_grammar_bounds_or_400(
            cfg(), request(tools=many_tools()), engine()
        )
    assert exc.value.status_code == 400
