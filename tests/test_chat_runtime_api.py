# SPDX-License-Identifier: Apache-2.0
import asyncio
import json

import httpx
import pytest
from pydantic_ai.models.function import FunctionModel

from rapid_mlx.chat_runtime.api import PREFIX, create_app
from rapid_mlx.chat_runtime.runtime import Runtime, TurnRequest
from rapid_mlx.chat_runtime.store import Store


@pytest.fixture
async def api(tmp_path):
    async def answer(messages, info):
        yield "Hello "
        yield "world."

    store = await Store.open(tmp_path)
    runtime = Runtime(store, model=FunctionModel(stream_function=answer))
    app = create_app(store, runtime, "test-token")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://owner",
        headers={"Authorization": "Bearer test-token"},
    ) as client:
        yield client, store, runtime
    await runtime.close()
    await store.close()


def body(conversation):
    return {
        "turn_id": "turn-1",
        "conversation_id": conversation["id"],
        "branch_id": conversation["branch_id"],
        "expected_revision": 0,
        "text": "Hello",
        "model": "fake",
        "inference_url": "http://127.0.0.1:18800",
    }


def events(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/capabilities"),
        ("GET", "/schema"),
        ("GET", "/conversations"),
        ("POST", "/conversations"),
        ("GET", "/conversations/missing"),
        ("DELETE", "/conversations/missing"),
        ("POST", "/turns"),
        ("GET", "/turns/missing/events"),
        ("POST", "/turns/missing/cancel"),
        ("POST", "/conversations/missing/attachments"),
        ("POST", "/clients"),
        ("GET", "/changes"),
        ("POST", "/permission.response"),
        ("GET", "/operations/missing"),
        ("GET", "/conversations/missing/operations"),
        ("POST", "/conversations/missing/undo"),
    ],
)
async def test_all_routes_require_auth(api, method, path):
    client, _, _ = api
    response = await client.request(
        method, PREFIX + path, headers={"Authorization": ""}
    )
    assert response.status_code == 401


async def test_submit_replay_idempotency_and_conflict(api):
    client, store, _ = api
    conversation = (await client.post(PREFIX + "/conversations", json={})).json()
    request = body(conversation)
    response = await client.post(PREFIX + "/turns", json=request)
    assert response.status_code == 200
    trace = events(response)
    assert trace[0]["event"]["type"] == "turn.started"
    assert trace[-1]["event"]["outcome"] == "completed"
    repeated = await client.post(PREFIX + "/turns", json=request)
    assert events(repeated) == trace
    replay = await client.get(
        PREFIX + "/turns/turn-1/events", headers={"Last-Event-ID": "turn-1:1"}
    )
    assert events(replay) == trace[1:]
    bad = await client.post(PREFIX + "/turns", json={**request, "text": "different"})
    assert bad.status_code == 409
    assert len((await store.get_conversation(conversation["id"]))["messages"]) == 2
    assert (
        await client.get(
            PREFIX + "/turns/turn-1/events", headers={"Last-Event-ID": "other:1"}
        )
    ).status_code == 400
    assert (
        await client.get(
            PREFIX + "/turns/turn-1/events", headers={"Last-Event-ID": "turn-1:999"}
        )
    ).status_code == 409


async def test_api_cancel_waits_for_cleanup(api):
    client, store, runtime = api
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def slow(messages, info):
        try:
            entered.set()
            yield "Partial"
            await asyncio.sleep(30)
        finally:
            cleaned.set()

    runtime.model_override = FunctionModel(stream_function=slow)
    conversation = await store.create_conversation()
    await runtime.submit(TurnRequest(**body(conversation)))
    await entered.wait()
    response = await client.post(PREFIX + "/turns/turn-1/cancel")
    assert response.json()["outcome"] == "cancelled"
    assert cleaned.is_set()
    trace = events(await client.get(PREFIX + "/turns/turn-1/events"))
    assert trace[-1]["event"]["outcome"] == "cancelled"
    assert sum(e["event"]["type"] == "turn.terminal" for e in trace) == 1


async def test_attachment_branches_delete_and_secret_errors(api):
    client, _, _ = api
    conversation = (await client.post(PREFIX + "/conversations", json={})).json()
    cid = conversation["id"]
    uploaded = await client.post(
        PREFIX + f"/conversations/{cid}/attachments",
        params={"name": "notes.txt"},
        content=b"Some notes",
    )
    assert uploaded.status_code == 201
    assert (
        await client.post(PREFIX + f"/conversations/{cid}/attachments", content=b"\xff")
    ).status_code == 415
    request = body(conversation)
    await client.post(PREFIX + "/turns", json=request)
    snapshot = (await client.get(PREFIX + f"/conversations/{cid}")).json()
    edited = await client.post(
        PREFIX + f"/conversations/{cid}/edit",
        json={
            "branch_id": request["branch_id"],
            "operation_id": "edit-1",
            "message_id": snapshot["messages"][0]["id"],
            "text": "new prompt",
        },
    )
    assert edited.status_code == 200
    assert edited.json()["branch_id"] != request["branch_id"]
    regenerated = await client.post(
        PREFIX + f"/conversations/{cid}/regenerate",
        json={
            "branch_id": request["branch_id"],
            "operation_id": "regen-1",
        },
    )
    assert regenerated.status_code == 200
    invalid = await client.post(
        PREFIX + "/turns", json={"inference_api_key": "DO-NOT-ECHO"}
    )
    assert invalid.status_code == 422 and "DO-NOT-ECHO" not in invalid.text
    assert (await client.delete(PREFIX + f"/conversations/{cid}")).status_code == 204
    assert (await client.get(PREFIX + f"/conversations/{cid}")).status_code == 404
