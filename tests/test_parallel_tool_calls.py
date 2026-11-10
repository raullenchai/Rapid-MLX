# SPDX-License-Identifier: Apache-2.0
"""Regression test for ``parallel_tool_calls`` being silently dropped.

Bug: ChatCompletionRequest did not declare ``parallel_tool_calls``, so
Pydantic dropped it on parse. Clients setting ``parallel_tool_calls:false``
still received multiple tool_calls back from models that emitted more than
one. Same shape as #459 (max_completion_tokens) and #355 (logit_bias) —
undeclared OpenAI-spec fields are dropped without warning.

Fix: declare the field and cap the parsed list at length 1 in the chat
route when the client passes False. The default (None / True) preserves
multi-call output.
"""

import json

import pytest

from rapid_mlx.api.models import (
    ChatCompletionRequest,
    FunctionCall,
    Message,
    ToolCall,
)


class TestSchemaDeclaration:
    """Field must round-trip through the request model."""

    def test_parallel_tool_calls_round_trips(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[Message(role="user", content="hi")],
            parallel_tool_calls=False,
        )
        assert req.parallel_tool_calls is False

    def test_parallel_tool_calls_round_trips_true(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[Message(role="user", content="hi")],
            parallel_tool_calls=True,
        )
        assert req.parallel_tool_calls is True

    def test_parallel_tool_calls_defaults_to_none(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[Message(role="user", content="hi")],
        )
        assert req.parallel_tool_calls is None

    def test_parallel_tool_calls_string_false_coerced(self):
        """Pydantic v2 lax mode coerces "false" → False; document the
        behavior so callers know they get coercion, not rejection."""
        req = ChatCompletionRequest(
            model="m",
            messages=[Message(role="user", content="hi")],
            parallel_tool_calls="false",
        )
        assert req.parallel_tool_calls is False


def _make_tool_call(name: str, args: str) -> ToolCall:
    return ToolCall(
        id=f"call_{name}",
        function=FunctionCall(name=name, arguments=args),
    )


def _apply_parallel_tool_calls_cap(
    tool_calls: list[ToolCall],
    parallel_tool_calls: bool | None,
) -> list[ToolCall]:
    """Mirror the route's cap logic; isolates the gate for unit-testability.

    Production wiring lives in ``routes/chat.py`` directly after
    ``_parse_tool_calls_with_parser``. Keeping the gate trivial means the
    test can pin its exact semantics without spinning up an engine.
    """
    if tool_calls and len(tool_calls) > 1 and parallel_tool_calls is False:
        return tool_calls[:1]
    return tool_calls


class TestCapBehavior:
    """The post-parse cap that the route applies to the parsed tool_calls."""

    def test_false_caps_multi_call_to_one(self):
        tcs = [
            _make_tool_call("get_weather", '{"city":"Tokyo"}'),
            _make_tool_call("get_weather", '{"city":"Paris"}'),
        ]
        out = _apply_parallel_tool_calls_cap(tcs, False)
        assert len(out) == 1
        assert out[0].function.name == "get_weather"
        assert out[0].function.arguments == '{"city":"Tokyo"}'

    def test_true_preserves_multi_call(self):
        tcs = [
            _make_tool_call("get_weather", '{"city":"Tokyo"}'),
            _make_tool_call("get_time", '{"city":"NYC"}'),
        ]
        out = _apply_parallel_tool_calls_cap(tcs, True)
        assert len(out) == 2

    def test_none_preserves_multi_call(self):
        """Default (field omitted) must NOT cap — only an explicit False does."""
        tcs = [
            _make_tool_call("a", "{}"),
            _make_tool_call("b", "{}"),
        ]
        out = _apply_parallel_tool_calls_cap(tcs, None)
        assert len(out) == 2

    def test_single_call_unaffected_by_false(self):
        tcs = [_make_tool_call("a", "{}")]
        out = _apply_parallel_tool_calls_cap(tcs, False)
        assert len(out) == 1
        assert out is tcs  # returns input list unchanged when cap doesn't trip

    def test_empty_list_unaffected(self):
        out = _apply_parallel_tool_calls_cap([], False)
        assert out == []


class TestRouteIntegration:
    """End-to-end: cap fires inside the chat route, response has 1 tool_call."""

    def test_route_caps_when_parser_returns_two_tool_calls(self, monkeypatch):
        """If the configured tool parser returns two tool_calls and the
        request asks for parallel_tool_calls=false, the response must
        surface only the first."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        # Patch the tool-call parser the route uses so we control output.
        import rapid_mlx.routes.chat as chat_module
        from rapid_mlx.config import reset_config
        from rapid_mlx.engine.base import GenerationOutput
        from rapid_mlx.routes.chat import router as chat_router

        def _fake_parse(
            text,
            request,
            *,
            structured_tool_calls=None,
            finish_reason=None,
            channels_owned=False,
        ):
            return (
                "",
                [
                    _make_tool_call("get_weather", '{"city":"Tokyo"}'),
                    _make_tool_call("get_weather", '{"city":"Paris"}'),
                ],
            )

        monkeypatch.setattr(chat_module, "_parse_tool_calls_with_parser", _fake_parse)

        class _StubEngine:
            preserve_native_tool_format = False
            is_mllm = False
            supports_guided_generation = False
            tokenizer = None

            def build_prompt(self, messages, tools=None, enable_thinking=None):
                return "PROMPT"

            async def chat(self, messages, **kwargs):
                return GenerationOutput(
                    text="dummy",
                    new_text="dummy",
                    tokens=[1, 2, 3],
                    prompt_tokens=4,
                    completion_tokens=3,
                    finished=True,
                    finish_reason="stop",
                    channel=None,
                )

        cfg = reset_config()
        cfg.engine = _StubEngine()
        cfg.model_name = "test-model"
        cfg.model_registry = None
        cfg.no_thinking = True
        cfg.reasoning_parser = None
        cfg.tool_parser = "hermes"  # any non-None parser; we patched the call

        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "weather in two cities"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "parallel_tool_calls": False,
                "max_tokens": 16,
            },
        )
        try:
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["choices"][0]["finish_reason"] == "tool_calls"
            msg = body["choices"][0]["message"]
            assert msg.get("tool_calls") is not None
            assert len(msg["tool_calls"]) == 1, (
                f"parallel_tool_calls=False must cap to 1; got "
                f"{len(msg['tool_calls'])} tool_calls"
            )
            assert msg["tool_calls"][0]["function"]["arguments"] == (
                '{"city":"Tokyo"}'
            ), "the first parsed tool_call must be preserved"
        finally:
            reset_config()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("parallel", [False, True, None])
@pytest.mark.parametrize("chunk_size", [1, 17, None])
def test_auto_response_caps_parsed_calls(stream, parallel, chunk_size):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.config import reset_config
    from rapid_mlx.routes.chat import router
    from tests.test_empty_rejected_tool_reply import _Engine

    wires = [
        '<tool_call>{"name":"get_weather","arguments":{"city":"'
        + city
        + '"}}</tool_call>'
        for city in ("Paris", "Tokyo")
    ]
    text = "\n".join(wires)
    deltas = (
        [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        if chunk_size
        else [text]
    )
    cfg = reset_config()
    cfg.engine = _Engine(text, deltas=deltas)
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "hermes"
    app = FastAPI()
    app.include_router(router)
    try:
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": stream,
                "messages": [{"role": "user", "content": "Weather in two cities?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "tool_choice": "auto",
                "parallel_tool_calls": parallel,
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
            calls = {}
            for choice in choices:
                for delta in choice["delta"].get("tool_calls", []):
                    call = calls.setdefault(
                        delta["index"], {"name": "", "arguments": ""}
                    )
                    for key, value in delta.get("function", {}).items():
                        call[key] += value
            assert set(calls) == ({0} if parallel is False else {0, 1})
            functions = list(calls.values())
            finish = choices[-1]["finish_reason"]
        else:
            choice = response.json()["choices"][0]
            functions = [call["function"] for call in choice["message"]["tool_calls"]]
            finish = choice["finish_reason"]
        assert [call["name"] for call in functions] == ["get_weather"] * (
            1 if parallel is False else 2
        )
        assert [json.loads(call["arguments"]) for call in functions] == (
            [{"city": "Paris"}]
            if parallel is False
            else [{"city": "Paris"}, {"city": "Tokyo"}]
        )
        assert finish == "tool_calls"
    finally:
        reset_config()
