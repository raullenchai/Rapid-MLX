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


@pytest.mark.parametrize("declared_length", [None, "1"])
@pytest.mark.parametrize("path", ["/conversations", "/clients", "/turns"])
async def test_streamed_body_cap_precedes_json_parsing(
    api, monkeypatch, declared_length, path
):
    client, store, _ = api
    monkeypatch.setattr("rapid_mlx.chat_runtime.api.MAX_REQUEST_BYTES", 64)
    consumed = []

    async def chunks():
        for chunk in (b" " * 32, b" " * 32, b"x"):
            consumed.append(len(chunk))
            yield chunk
        pytest.fail("oversized request must not consume more upload bytes")

    headers = {"Content-Type": "application/json"}
    if declared_length is not None:
        headers["Content-Length"] = declared_length
    response = await client.post(PREFIX + path, content=chunks(), headers=headers)
    assert response.status_code == 413
    assert response.json() == {"detail": "Request too large"}
    assert response.headers["cache-control"] == "no-store"
    assert consumed == [32, 32, 1]
    assert await store.list_conversations() == []


async def test_streamed_body_at_cap_reaches_handler(api, monkeypatch):
    client, _, _ = api
    monkeypatch.setattr("rapid_mlx.chat_runtime.api.MAX_REQUEST_BYTES", 64)
    payload = b'{"title":"chunked"}'.ljust(64)

    async def chunks():
        yield payload[:32]
        yield payload[32:]

    response = await client.post(
        PREFIX + "/conversations",
        content=chunks(),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 201
    assert response.json()["title"] == "chunked"


@pytest.mark.parametrize("authorized", [False, True])
async def test_auth_and_declared_body_cap_reject_without_reading(
    api, monkeypatch, authorized
):
    client, _, _ = api
    monkeypatch.setattr("rapid_mlx.chat_runtime.api.MAX_REQUEST_BYTES", 64)

    async def chunks():
        pytest.fail("request must be rejected before reading its body")
        yield b"unused"

    response = await client.post(
        PREFIX + "/conversations",
        content=chunks(),
        headers={
            "Content-Length": "65",
            "Authorization": "Bearer test-token" if authorized else "",
        },
    )
    assert response.status_code == (413 if authorized else 401)


async def test_body_cap_preserves_partial_upload_disconnect():
    from rapid_mlx.chat_runtime.api import ChatBodyLimitMiddleware

    incoming = iter(
        [
            {"type": "http.request", "body": b'{"title":', "more_body": True},
            {"type": "http.disconnect"},
        ]
    )
    observed = []

    async def receive():
        return next(incoming)

    async def downstream(scope, receive, send):
        observed.append(await receive())
        observed.append(await receive())

    async def send(message):
        pytest.fail("disconnected partial body cannot create a success response")

    await ChatBodyLimitMiddleware(downstream, "test-token")(
        {"type": "http", "headers": [(b"authorization", b"Bearer test-token")]},
        receive,
        send,
    )
    assert observed == [
        {"type": "http.request", "body": b'{"title":', "more_body": True},
        {"type": "http.disconnect"},
    ]


@pytest.mark.parametrize(
    "code,status",
    [
        ("binary_unsupported", 415),
        ("file_too_large", 413),
        ("timeout", 504),
        ("worker_failed", 500),
        ("document_invalid", 422),
        ("other_worker_failure", 422),
    ],
)
async def test_attachment_worker_error_status_and_safe_message(
    api, monkeypatch, code, status
):
    client, _, _ = api
    conversation = (await client.post(PREFIX + "/conversations", json={})).json()

    async def worker(*args, **kwargs):
        return {"error": {"code": code, "message": "Safe extraction message."}}

    monkeypatch.setattr("rapid_mlx.chat_runtime.api.run_worker", worker)
    response = await client.post(
        PREFIX + f"/conversations/{conversation['id']}/attachments", content=b"x"
    )
    assert response.status_code == status
    assert response.json()["detail"] == {
        "code": code,
        "message": "Safe extraction message.",
    }


async def test_body_buffer_authenticates_and_times_out(monkeypatch):
    import asyncio

    from rapid_mlx.chat_runtime.api import ChatBodyLimitMiddleware

    monkeypatch.setattr("rapid_mlx.chat_runtime.api.BODY_READ_TIMEOUT", 0.01)
    received = 0
    sent = []

    async def receive():
        nonlocal received
        received += 1
        if received == 1:
            return {"type": "http.request", "body": b"x", "more_body": True}
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)

    async def downstream(scope, receive, send):
        pytest.fail("incomplete body must not reach downstream")

    middleware = ChatBodyLimitMiddleware(downstream, "test-token")
    scope = {"type": "http", "headers": []}
    await middleware(scope, receive, send)
    assert sent[0]["status"] == 401
    assert received == 0
    sent.clear()
    scope["headers"] = [(b"authorization", b"Bearer test-token")]
    await middleware(scope, receive, send)
    assert sent[0]["status"] == 408
    assert (b"connection", b"close") in sent[0]["headers"]
    assert received == 2
