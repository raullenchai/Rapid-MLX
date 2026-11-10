# SPDX-License-Identifier: Apache-2.0
"""Forced Anthropic tools use the chat route's generation constraints."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from rapid_mlx.config import get_config
from tests.test_anthropic_tool_validation_scope import (
    _GET_WEATHER,
    _call,
    _make_client,
    _MultiCallEngine,
)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "choice", [{"type": "tool", "name": "get_weather"}, {"type": "any"}]
)
@pytest.mark.parametrize("grammar_active", [False, True])
def test_forced_tools_share_chat_generation_helpers(
    monkeypatch, stream, choice, grammar_active
):
    from rapid_mlx.routes import chat

    processor = SimpleNamespace(reasoning_gate_id=None) if grammar_active else None
    build = AsyncMock(return_value=processor)
    bounds = AsyncMock()
    prefix = Mock(return_value='<tool_call>{"name":"get_weather","arguments":')
    monkeypatch.setattr(chat, "_offload_tool_grammar_build", build)
    monkeypatch.setattr(chat, "_enforce_tool_grammar_bounds_or_400_async", bounds)
    monkeypatch.setattr(chat, "_compute_forced_tool_prefix", prefix)

    class Engine(_MultiCallEngine):
        def check(self, kwargs):
            if grammar_active:
                assert kwargs["grammar_logits_processor"] is processor
                assert "forced_assistant_prefix" not in kwargs
            else:
                assert kwargs["forced_assistant_prefix"] == prefix.return_value
            self.seen = kwargs

        async def chat(self, messages, **kwargs):
            self.check(kwargs)
            return await super().chat(messages, **kwargs)

        async def stream_chat(self, messages, **kwargs):
            self.check(kwargs)
            async for chunk in super().stream_chat(messages, **kwargs):
                yield chunk

    engine = Engine([_call("get_weather", {"location": "SF"})])
    client = _make_client(engine)
    get_config().tool_call_parser = "hermes"
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 150,
            "stream": stream,
            "messages": [{"role": "user", "content": "Tell me a joke."}],
            "tools": [_GET_WEATHER],
            "tool_choice": {**choice, "disable_parallel_tool_use": True},
        },
    )
    assert response.status_code == 200, response.text
    assert "tool_use" in response.text
    assert "SF" in response.text
    build.assert_awaited_once()
    bounds.assert_awaited_once()
    adapted = build.call_args.args[2]
    assert adapted.parallel_tool_calls is False
    assert adapted.tool_choice == (
        "required"
        if choice["type"] == "any"
        else {"type": "function", "function": {"name": "get_weather"}}
    )
    assert prefix.call_count == (0 if grammar_active else 1)


@pytest.mark.parametrize("stream", [False, True])
def test_any_with_multiple_tools_uses_shared_grammar(monkeypatch, stream):
    from rapid_mlx.routes import chat
    from tests.test_anthropic_tool_validation_scope import _LOOKUP_ZIP

    processor = SimpleNamespace(reasoning_gate_id=None)
    build = AsyncMock(return_value=processor)
    monkeypatch.setattr(chat, "_offload_tool_grammar_build", build)
    engine = _MultiCallEngine([_call("get_weather", {"location": "SF"})])
    client = _make_client(engine)
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 150,
            "stream": stream,
            "messages": [{"role": "user", "content": "Tell me a joke."}],
            "tools": [_GET_WEATHER, _LOOKUP_ZIP],
            "tool_choice": {"type": "any"},
        },
    )
    assert response.status_code == 200, response.text
    build.assert_awaited_once()
    assert len(build.call_args.args[2].tools) == 2
    assert build.call_args.args[2].tool_choice == "required"


@pytest.mark.parametrize("stream", [False, True])
def test_tool_grammar_bounds_failure_is_http_error_before_generation(
    monkeypatch, stream
):
    from fastapi import HTTPException

    from rapid_mlx.routes import chat

    bounds = AsyncMock(side_effect=HTTPException(400, "tool schema exceeds bounds"))
    build = AsyncMock()
    monkeypatch.setattr(chat, "_enforce_tool_grammar_bounds_or_400_async", bounds)
    monkeypatch.setattr(chat, "_offload_tool_grammar_build", build)
    engine = _MultiCallEngine([])
    engine.chat = AsyncMock(side_effect=AssertionError("generation must not start"))
    client = _make_client(engine)
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 150,
            "stream": stream,
            "messages": [{"role": "user", "content": "Tell me a joke."}],
            "tools": [_GET_WEATHER],
            "tool_choice": {"type": "any"},
        },
    )
    assert response.status_code == 400, response.text
    assert "event:" not in response.text
    build.assert_not_awaited()
    engine.chat.assert_not_awaited()


@pytest.mark.parametrize(
    "disabled,expected", [(True, False), (False, None), (None, None)]
)
def test_adapter_preserves_parallel_tool_preference(disabled, expected):
    from rapid_mlx.api.anthropic_adapter import anthropic_to_openai
    from rapid_mlx.api.anthropic_models import AnthropicRequest

    choice = {"type": "any"}
    if disabled is not None:
        choice["disable_parallel_tool_use"] = disabled
    request = AnthropicRequest(
        model="test-model",
        max_tokens=150,
        messages=[{"role": "user", "content": "Weather?"}],
        tools=[_GET_WEATHER],
        tool_choice=choice,
    )
    assert anthropic_to_openai(request).parallel_tool_calls is expected


def test_forced_stream_preflight_client_disconnect_aborts(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    from rapid_mlx.routes import anthropic as route

    async def gen():
        yield "packet"

    async def disconnected(coro, request, timeout):
        coro.close()
        return None

    aborted = Mock()
    monkeypatch.setattr(route, "_wait_with_disconnect", disconnected)
    monkeypatch.setattr(route, "_force_abort_request", aborted)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            route._preflight_forced_tool_stream(
                gen(), Mock(), engine="engine", request_id_holder=["rid"]
            )
        )
    assert exc.value.status_code == 499
    aborted.assert_called_once_with("engine", ["rid"])


class _ConnectedRequest:
    async def is_disconnected(self):
        return False


def _drain(stream):
    async def run():
        return [packet async for packet in stream]

    return run()


def test_forced_stream_preflight_timeout_is_idle_not_total(monkeypatch):
    import asyncio

    from rapid_mlx.routes import anthropic as route

    async def steady():
        for index in range(5):
            await asyncio.sleep(0.03)
            yield index

    monkeypatch.setattr(get_config(), "default_timeout", 0.1)

    async def run():
        replay = await route._preflight_forced_tool_stream(
            steady(), _ConnectedRequest(), engine="engine", request_id_holder=["rid"]
        )
        return await _drain(replay)

    # 5 x 30 ms exceeds the 100 ms deadline in total, never between packets.
    assert asyncio.run(run()) == [0, 1, 2, 3, 4]


def test_forced_stream_preflight_stall_times_out_and_aborts(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    from rapid_mlx.routes import anthropic as route

    async def stalled():
        yield 0
        await asyncio.sleep(10)
        yield 1

    aborted = Mock()
    monkeypatch.setattr(route, "_force_abort_request", aborted)
    monkeypatch.setattr(get_config(), "default_timeout", 0.05)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            route._preflight_forced_tool_stream(
                stalled(),
                _ConnectedRequest(),
                engine="engine",
                request_id_holder=["rid"],
            )
        )
    assert exc.value.status_code == 504
    aborted.assert_called_once_with("engine", ["rid"])


@pytest.mark.parametrize(
    "choice", [{"type": "tool", "name": "get_weather"}, {"type": "any"}]
)
def test_forced_stream_schema_invalid_call_is_http_error(monkeypatch, choice):
    from rapid_mlx.routes import chat

    monkeypatch.setattr(
        chat, "_offload_tool_grammar_build", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(chat, "_enforce_tool_grammar_bounds_or_400_async", AsyncMock())
    monkeypatch.setattr(chat, "_compute_forced_tool_prefix", Mock(return_value=None))
    client = _make_client(_MultiCallEngine([_call("get_weather", {"location": 941})]))
    get_config().tool_call_parser = "hermes"
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 150,
            "stream": True,
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [_GET_WEATHER],
            "tool_choice": choice,
        },
    )
    assert response.status_code == 400, response.text
    assert "event: error" not in response.text
