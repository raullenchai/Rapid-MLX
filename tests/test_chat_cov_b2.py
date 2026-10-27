from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic_ai.messages import ModelRequest, ToolCallPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from rapid_mlx.chat_runtime import api, context


async def test_api_remaining_routes(tmp_path, monkeypatch):
    from rapid_mlx.chat_runtime.runtime import Runtime
    from rapid_mlx.chat_runtime.store import Store

    store = await Store.open(tmp_path)
    runtime = Runtime(store)
    with pytest.raises(ValueError, match="bearer"):
        api.create_app(store, runtime, "")
    app = api.create_app(store, runtime, "token")
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer token"},
        ) as client:
            assert (await client.get(api.PREFIX + "/schema")).status_code == 200
            assert (
                await client.post(api.PREFIX + "/clients", json={"client_id": "c"})
            ).json()["client_id"] == "c"
            assert (await client.delete(api.PREFIX + "/clients/c")).status_code == 204
            assert (await client.get(api.PREFIX + "/conversations")).json() == []
            conv = (await client.post(api.PREFIX + "/conversations", json={})).json()
            cid = conv["id"]
            body = {
                "turn_id": "t",
                "conversation_id": "wrong",
                "branch_id": conv["branch_id"],
                "expected_revision": 0,
                "text": "hello",
                "model": "m",
                "inference_url": "http://127.0.0.1:1",
            }
            assert (
                await client.post(
                    api.PREFIX + f"/conversations/{cid}/compact", json={"request": body}
                )
            ).status_code == 400
            body["conversation_id"] = cid
            monkeypatch.setattr(
                runtime, "context_info", AsyncMock(return_value={"used": 1})
            )
            monkeypatch.setattr(
                runtime, "compact", AsyncMock(return_value={"summary": "ok"})
            )
            assert (
                await client.post(
                    api.PREFIX + f"/conversations/{cid}/context", json=body
                )
            ).json() == {"used": 1}
            assert (
                await client.post(
                    api.PREFIX + f"/conversations/{cid}/compact",
                    json={"request": body, "instructions": "focus"},
                )
            ).json() == {"summary": "ok"}
            assert runtime.compact.await_args.kwargs["instructions"] == "focus"
            monkeypatch.setattr(api, "MAX_ATTACHMENT_BYTES", 2)
            url = api.PREFIX + f"/conversations/{cid}/attachments"
            assert (await client.post(url, content=b"abc")).status_code == 413
            assert (await client.post(url + "?name=", content=b"a")).status_code == 400
            assert (await client.get(api.PREFIX + "/turns/missing")).status_code == 404
            assert (
                await client.get(api.PREFIX + "/operations/missing")
            ).status_code == 404
            monkeypatch.setattr(store, "get_turn", AsyncMock(return_value={"id": "t"}))
            monkeypatch.setattr(store, "replay_events", AsyncMock(return_value=[]))

            async def events(*args):
                yield None

            monkeypatch.setattr(runtime, "events", events)
            response = await client.get(api.PREFIX + "/turns/t/events")
            assert response.text == ": keepalive\n\n"
    finally:
        await runtime.close()
        await store.close()


async def test_middleware_non_http_and_changes():
    called = AsyncMock()
    middleware = api.ChatBodyLimitMiddleware(called, "token")
    await middleware({"type": "websocket"}, AsyncMock(), AsyncMock())
    assert called.await_count == 1
    store = SimpleNamespace(catalog_revision=AsyncMock(side_effect=[3, 3]))
    app = api.create_app(store, SimpleNamespace(), "token")
    endpoint = next(r.endpoint for r in app.routes if r.path == api.PREFIX + "/changes")
    request = SimpleNamespace(
        is_disconnected=AsyncMock(side_effect=[False, False, True])
    )
    response = await endpoint(request, after=0)
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    assert '"revision": 3' in chunks[0]
    assert chunks[1] == ": keepalive\n\n"


async def test_context_edges(monkeypatch):
    assert context.context_event({"used_tokens": 95}, 100)["notice"]

    class Counter:
        async def measure(self, messages, parameters):
            return {"used_tokens": 10000, "system_tools_tokens": 0}

    with pytest.raises(ValueError, match="incomplete tool round"):
        await context.fit_synthesis(
            [ModelRequest(parts=[ToolCallPart("tool", {}, tool_call_id="id")])],
            ModelRequestParameters(),
            Counter(),
            1,
        )

    row = {
        "id": "a",
        "role": "user",
        "created_at": "2026-01-01T00:00:00+00:00",
        "parts": [{"type": "text", "text": "hello"}],
    }
    runtime = SimpleNamespace(
        model_override=True, store=SimpleNamespace(save_summary=AsyncMock())
    )
    request = SimpleNamespace(
        context_tokens=256,
        max_tokens=64,
        branch_id="b",
        expected_revision=1,
        turn_id="t",
    )
    deps = SimpleNamespace(
        child_usage={}, emit=AsyncMock(), result_budget_exhausted=False
    )
    manager = context.ContextManager(runtime, request, deps, [row], None)
    assert await manager.summarize([]) == ""
    monkeypatch.setattr(context, "TokenCounter", lambda *args: Counter())
    with pytest.raises(ValueError, match="instructions exceed"):
        await manager.summarize([row])
    manager.summarize = AsyncMock(return_value="")
    assert not await manager.checkpoint_older(manual=True)
    manager.checkpoint = ModelRequest(parts=[UserPromptPart("checkpoint")])
    manager.covered_ids = {"a"}
    manager.counter = SimpleNamespace(
        measure=AsyncMock(return_value={"used_tokens": 1, "system_tools_tokens": 0})
    )
    rc = SimpleNamespace(
        model_request_parameters=ModelRequestParameters(),
        messages=[
            ModelRequest(parts=[UserPromptPart("old")], metadata={"message_id": "a"})
        ],
        model_settings={},
    )
    await manager.prepare(None, rc)
    assert rc.messages == [manager.checkpoint]


@pytest.mark.parametrize("mode", ["cannot_fit", "empty_output"])
async def test_summary_rejects_unusable_requests(monkeypatch, mode):
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    row = {
        "id": "a",
        "role": "user",
        "created_at": "2026-01-01T00:00:00+00:00",
        "parts": [{"type": "text", "text": "long enough text"}],
    }

    class Counter:
        def __init__(self, *args):
            self.calls = 0

        async def measure(self, messages, parameters):
            self.calls += 1
            return {
                "used_tokens": 1 if self.calls == 1 or mode == "empty_output" else 1000,
                "system_tools_tokens": 0,
            }

    monkeypatch.setattr(context, "TokenCounter", Counter)
    manager = context.ContextManager(
        SimpleNamespace(model_override=True),
        SimpleNamespace(context_tokens=256),
        SimpleNamespace(child_usage={}),
        [row],
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(" ")])),
    )
    with pytest.raises(
        ValueError, match="cannot fit" if mode == "cannot_fit" else "no checkpoint"
    ):
        await manager.summarize([row])


async def test_prepare_uses_new_checkpoint():
    row = {
        "id": "a",
        "role": "user",
        "created_at": "2026-01-01T00:00:00+00:00",
        "parts": [{"type": "text", "text": "hello"}],
    }
    runtime = SimpleNamespace(
        model_override=True, store=SimpleNamespace(save_summary=AsyncMock())
    )
    request = SimpleNamespace(
        context_tokens=512,
        max_tokens=64,
        branch_id="b",
        expected_revision=1,
        turn_id="t",
    )
    deps = SimpleNamespace(
        child_usage={}, emit=AsyncMock(), result_budget_exhausted=False
    )
    manager = context.ContextManager(runtime, request, deps, [row], None)
    manager.summarize = AsyncMock(return_value="checkpoint")

    async def checkpoint():
        return await context.ContextManager.checkpoint_older(manager, manual=True)

    manager.checkpoint_older = AsyncMock(side_effect=checkpoint)

    manager.counter = SimpleNamespace(
        measure=AsyncMock(
            side_effect=[
                {"used_tokens": 1000, "system_tools_tokens": 0},
                {"used_tokens": 1000, "system_tools_tokens": 0},
                {"used_tokens": 1, "system_tools_tokens": 0},
            ]
        )
    )
    rc = SimpleNamespace(
        model_request_parameters=ModelRequestParameters(),
        messages=[
            ModelRequest(parts=[UserPromptPart("old")], metadata={"message_id": "a"})
        ],
        model_settings={},
    )
    await manager.prepare(None, rc)
    assert rc.messages == [manager.checkpoint]
    assert deps.context_state["compacted"] is True
