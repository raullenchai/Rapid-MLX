"""Context settings and commands share owner policy with inference startup."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from rapid_mlx.chat_client import COMMAND_BY_NAME, ChatClient
from rapid_mlx.chat_runtime.api import PREFIX, create_app
from rapid_mlx.chat_runtime.context_limits import context_limit, startup_context_limit


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"max_position_embeddings": 262144}, 32768),
        ({"max_position_embeddings": 4096}, 4096),
        ({"text_config": {"max_position_embeddings": 8192}}, 8192),
        ({}, 32768),
    ],
)
def test_spawn_default_uses_offline_checkpoint_limit(monkeypatch, config, expected):
    monkeypatch.setattr("rapid_mlx.model_aliases.resolve_profile", lambda _: None)
    monkeypatch.setattr(
        "rapid_mlx.model_metadata.read_model_metadata",
        lambda _: SimpleNamespace(config=config),
    )
    assert startup_context_limit("fixture") == expected


@pytest.mark.parametrize(
    "config,requested,expected",
    [
        ({"max_position_embeddings": 262144}, 65536, 65536),
        ({"max_position_embeddings": 4096}, 65536, 4096),
        ({"text_config": {"max_position_embeddings": 8192}}, 65536, 8192),
        ({"max_position_embeddings": 262144}, 262144, 131072),
        ({}, 262144, 131072),
        ({}, 65536, 65536),
        ({"max_position_embeddings": 4096}, 2048, 2048),
    ],
)
def test_spawn_explicit_context_respects_native_and_runtime_limits(
    monkeypatch, config, requested, expected
):
    monkeypatch.setattr("rapid_mlx.model_aliases.resolve_profile", lambda _: None)
    monkeypatch.setattr(
        "rapid_mlx.model_metadata.read_model_metadata",
        lambda _: SimpleNamespace(config=config),
    )
    assert startup_context_limit("fixture", requested) == expected


def test_context_default_and_explicit_server_cap():
    assert context_limit() == 32768
    assert context_limit(65536, model={"max_model_len": 16384}) == 16384
    assert context_limit(model={"context_window": True}) == 32768
    assert context_limit(previous=8192) == 8192


@pytest.mark.parametrize("command", ["/context", "/compact keep the task"])
def test_context_commands_use_owner_breakdown(command, monkeypatch, capsys):
    client = ChatClient(SimpleNamespace(), "http://127.0.0.1:18800", "fixture")
    client.conversation_id = "conversation"
    calls = []
    body = {"model": "fixture", "branch_id": "branch"}
    monkeypatch.setattr(client, "context_request", lambda: body)
    monkeypatch.setattr(client, "refresh", lambda: None)
    context = {
        "percent": 92,
        "notice": "Context almost full — /compact or /new",
        "used_tokens": 7536,
        "limit_tokens": 8192,
        "system_tools_tokens": 1100,
        "conversation_tokens": 3936,
        "tool_output_tokens": 2500,
        "free_tokens": 656,
    }

    def request(method, path, **kwargs):
        assert kwargs["retry"] is (not command.startswith("/compact"))
        calls.append((method, path, kwargs["json"]))
        return {"context": context, "notice": "Context compacted."}

    monkeypatch.setattr(client, "request", request)
    try:
        assert client.command(command)
        output = capsys.readouterr().out
        assert "Context almost full — /compact or /new" in output
        assert "7,536 / 8,192 tokens" in output
        assert "System + tools" in output and "Tool output" in output
        assert "/context" in COMMAND_BY_NAME and "/compact" in COMMAND_BY_NAME
        if command.startswith("/compact"):
            assert calls[0][2] == {"request": body, "instructions": "keep the task"}
        else:
            assert calls[0][2] == body
    finally:
        client.client.close()


@pytest.mark.asyncio
async def test_context_endpoints_validate_conversation_and_delegate():
    store = SimpleNamespace(get_conversation=AsyncMock(return_value={}))
    runtime = SimpleNamespace(
        context_info=AsyncMock(return_value={"used_tokens": 120, "limit_tokens": 8192}),
        compact=AsyncMock(return_value={"notice": "Context compacted."}),
    )
    app = create_app(store, runtime, "test-token")
    request = {
        "turn_id": "context-1",
        "conversation_id": "c",
        "branch_id": "b",
        "expected_revision": 3,
        "inference_url": "http://127.0.0.1:18800",
        "model": "fixture",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://owner",
        headers={"Authorization": "Bearer test-token"},
    ) as client:
        response = await client.post(PREFIX + "/conversations/c/context", json=request)
        assert response.status_code == 200 and response.json()["used_tokens"] == 120
        response = await client.post(
            PREFIX + "/conversations/c/compact",
            json={"request": request, "instructions": "Keep decisions"},
        )
        assert response.status_code == 200
        assert runtime.compact.call_args.kwargs == {"instructions": "Keep decisions"}
        response = await client.post(
            PREFIX + "/conversations/other/context", json=request
        )
        assert response.status_code == 400
        response = await client.post(
            PREFIX + "/conversations/c/compact",
            json={"request": request},
            headers={"Authorization": ""},
        )
        assert response.status_code == 401
