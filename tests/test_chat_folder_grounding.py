"""Folder chat owns immutable evidence, never implicit tool or network authority."""

import asyncio
import json

import httpx
import pytest
from pydantic_ai.models.function import FunctionModel

from rapid_mlx.chat_runtime.api import create_app
from rapid_mlx.chat_runtime.model_client import ModelClients
from rapid_mlx.chat_runtime.prompts import LANGUAGE_INSTRUCTIONS
from rapid_mlx.chat_runtime.runtime import Runtime, TurnRequest
from rapid_mlx.chat_runtime.store import Store
from rapid_mlx.local_search.grounding import ABSTENTION
from tests.test_local_search_grounding import hit


class FolderServer:
    def __init__(self, results=None):
        self.results = [hit()] if results is None else results
        self.calls = []
        self.enabled = True
        self.state = "idle"
        self.status_code = 200

    def __call__(self, request):
        self.calls.append(request)
        assert request.headers["authorization"] == "Bearer local-secret"
        if request.url.path.endswith("/status"):
            return httpx.Response(
                self.status_code,
                json={
                    "enabled": self.enabled,
                    "folders": [{"id": "selected", "state": self.state}],
                },
            )
        if request.url.path.endswith("/search"):
            assert json.loads(request.content) == {
                "query": "能退款吗？",
                "limit": 3,
                "folder_ids": ["selected"],
                "kinds": ["text"],
                "include_text": True,
            }
            return httpx.Response(
                200, json={"results": self.results, "mode": "keyword"}
            )
        assert request.url.path.endswith("/messages/count_tokens")
        return httpx.Response(200, json={"input_tokens": 100})


async def fixture(path, server, stream):
    store = await Store.open(path)
    conversation = await store.create_conversation()
    runtime = Runtime(store, model=FunctionModel(stream_function=stream))
    await runtime.clients.close()
    runtime.clients = ModelClients(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(server))
    )
    request = TurnRequest(
        turn_id="folder-1",
        conversation_id=conversation["id"],
        branch_id=conversation["branch_id"],
        expected_revision=0,
        text="能退款吗？",
        inference_url="http://localhost:18800",
        model="local",
        inference_api_key="local-secret",
        local_search={"folder_ids": ["selected"]},
    )
    return store, runtime, request


async def collect(runtime, request):
    await runtime.submit(request)
    return [e async for e in runtime.events(request.turn_id) if e]


def test_evidence_is_frozen_before_answer_and_authenticated_preview(tmp_path):
    calls = []

    async def stream(messages, info):
        assert not info.function_tools
        text = repr(messages)
        assert LANGUAGE_INSTRUCTIONS in text
        assert "能退款吗？" in text
        assert "30 days" in text
        assert "/private/books/" not in text
        calls.append(messages)
        yield "可以，30 天内 [1]。"

    async def scenario():
        server = FolderServer()
        store, runtime, request = await fixture(tmp_path, server, stream)
        try:
            events = await collect(runtime, request)
            assert events[-1]["event"]["outcome"] == "completed"
            source = next(e for e in events if e["event"]["type"] == "source")
            answer = next(e for e in events if e["event"]["type"] == "text.delta")
            assert source["seq"] < answer["seq"]
            source = source["event"]
            assert source["source_id"] == "local-search:folder-1:1"
            assert source["url"] is None
            aid = source["attachment_id"]
            assert (await store.preview_attachment(request.conversation_id, aid))[
                "text"
            ] == hit()["text"]
            server.results = [hit(text="Changed index content")]
            _, created = await runtime.submit(request)
            assert not created
            assert len(calls) == 1
            assert len([r for r in server.calls if r.url.path.endswith("/search")]) == 1
            assert (await store.preview_attachment(request.conversation_id, aid))[
                "text"
            ] == hit()["text"]
            other = await store.create_conversation()
            app = create_app(store, runtime, "owner-token")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://owner"
            ) as client:
                route = f"/v1/rapid/chat/conversations/{request.conversation_id}/attachments/{aid}"
                assert (await client.get(route)).status_code == 401
                auth = {"Authorization": "Bearer owner-token"}
                response = await client.get(route, headers=auth)
                assert response.status_code == 200
                assert response.json()["text"] == hit()["text"]
                assert (
                    await client.get(
                        f"/v1/rapid/chat/conversations/{other['id']}/attachments/{aid}",
                        headers=auth,
                    )
                ).status_code == 404
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "results,code",
    [
        ([], None),
        ([hit(kind="image", matched=["name"], text=None)], None),
        ([hit(matched=["name"], text=None)], None),
        ([hit(text=None)], "local_search_full_text_unsupported"),
    ],
)
def test_no_evidence_and_old_server_never_invoke_model(tmp_path, results, code):
    async def stream(messages, info):
        raise AssertionError("No model may answer without usable evidence")
        yield ""

    async def scenario():
        server = FolderServer(results)
        store, runtime, request = await fixture(tmp_path, server, stream)
        try:
            events = await collect(runtime, request)
            terminal = events[-1]["event"]
            assert terminal["outcome"] == ("error" if code else "completed")
            assert terminal.get("error_code") == code
            assert not any(e["event"]["type"] == "source" for e in events)
            assert not any(r.url.path.endswith("count_tokens") for r in server.calls)
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


def test_completed_folder_chat_regenerates_and_followup_ignores_source_receipts(
    tmp_path,
):
    async def stream(messages, info):
        assert not info.function_tools
        yield "答案 [1]"

    async def scenario():
        server = FolderServer()
        store, runtime, request = await fixture(tmp_path, server, stream)
        try:
            await collect(runtime, request)
            conversation = await store.get_conversation(request.conversation_id)
            branch = conversation["branches"][0]
            second = request.model_copy(
                update={"turn_id": "folder-2", "expected_revision": branch["revision"]}
            )
            events = await collect(runtime, second)
            assert events[-1]["event"]["outcome"] == "completed"
            fork = await store.regenerate(request.branch_id, "regenerate")
            regenerated = request.model_copy(
                update={
                    "turn_id": "folder-3",
                    "branch_id": fork["id"],
                    "expected_revision": fork["revision"],
                    "mode": "regenerate",
                    "text": "",
                }
            )
            events = await collect(runtime, regenerated)
            assert events[-1]["event"]["outcome"] == "completed"
            assert len([r for r in server.calls if r.url.path.endswith("/search")]) == 3
            source_ids = [
                e["event"]["source_id"]
                for tid in ("folder-1", "folder-2", "folder-3")
                for e in await store.replay_events(tid)
                if e["event"]["type"] == "source"
            ]
            assert len(set(source_ids)) == 3
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "enabled,state,status,code",
    [
        (False, "idle", 200, "local_search_disabled"),
        (True, "missing", 200, "folder_not_found"),
        (True, "error", 200, "folder_not_found"),
        (True, "idle", 401, "local_search_unauthorized"),
        (True, "idle", 500, "local_search_unavailable"),
    ],
)
def test_unavailable_index_never_falls_back(tmp_path, enabled, state, status, code):
    async def stream(messages, info):
        raise AssertionError("Must not infer without the selected index")
        yield ""

    async def scenario():
        server = FolderServer()
        server.enabled, server.state, server.status_code = enabled, state, status
        store, runtime, request = await fixture(tmp_path, server, stream)
        try:
            events = await collect(runtime, request)
            assert events[-1]["event"]["error_code"] == code
            assert len(server.calls) == 1
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["retrieval", "generation"])
def test_cancellation_keeps_only_admitted_receipts_before_terminal(tmp_path, phase):
    entered = asyncio.Event()
    blocked = asyncio.Event()

    async def stream(messages, info):
        entered.set()
        await blocked.wait()
        yield "Must never publish"

    async def scenario():
        server = FolderServer()

        async def transport(request):
            if phase == "retrieval" and request.url.path.endswith("/search"):
                entered.set()
                await blocked.wait()
            return server(request)

        store, runtime, request = await fixture(tmp_path, transport, stream)
        try:
            await runtime.submit(request)
            await asyncio.wait_for(entered.wait(), 5)
            await runtime.cancel(request.turn_id)
            events = [e async for e in runtime.events(request.turn_id) if e]
            assert events[-1]["event"]["outcome"] == "cancelled"
            sources = [e for e in events if e["event"]["type"] == "source"]
            assert len(sources) == (1 if phase == "generation" else 0)
            assert not any(
                e["event"].get("text") == "Must never publish" for e in events
            )
            assert (
                len([e for e in events if e["event"]["type"] == "turn.terminal"]) == 1
            )
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


def test_scope_and_preview_follow_conversation_lifecycle(tmp_path):
    async def scenario():
        store = await Store.open(tmp_path)
        try:
            conversation = await store.create_conversation()
            identifier = conversation["id"]
            assert "local_search" not in conversation
            await store.set_folder_scope(
                identifier, {"folder_ids": ["selected"], "limit": 3}
            )
            assert (await store.get_conversation(identifier))["local_search"][
                "folder_ids"
            ] == ["selected"]
            await store.set_folder_scope(identifier, None)
            assert "local_search" not in await store.get_conversation(identifier)
            attachment = await store.put_attachment(
                identifier, "evidence.txt", b"Frozen evidence"
            )
            await store.delete_conversation(identifier)
            with pytest.raises(KeyError):
                await store.preview_attachment(identifier, attachment["id"])
            with pytest.raises(KeyError):
                await store.set_folder_scope(identifier, {"folder_ids": ["selected"]})
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "count,code",
    [
        (None, "local_search_token_count_unavailable"),
        (1000000, "local_search_context_too_large"),
    ],
)
def test_immutable_context_requires_exact_token_admission(tmp_path, count, code):
    async def stream(messages, info):
        raise AssertionError("Token admission must precede inference")
        yield ""

    async def scenario():
        server = FolderServer()

        def handler(request):
            if request.url.path.endswith("count_tokens"):
                return httpx.Response(
                    404 if count is None else 200, json={"input_tokens": count}
                )
            return server(request)

        store, runtime, request = await fixture(tmp_path, handler, stream)
        try:
            events = await collect(runtime, request)
            assert events[-1]["event"]["error_code"] == code
            assert len([e for e in events if e["event"]["type"] == "source"]) == 1
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "question,enabled,expected",
    [
        (
            "能退款吗？",
            True,
            "在所选文件夹的已索引内容中没有找到匹配的片段，无法根据此文件夹回答这个问题。",
        ),
        (
            "Can I return it?",
            True,
            ABSTENTION,
        ),
        ("能退款吗？", False, "请启用本地搜索并选择已登记的文件夹。"),
        (
            "Can I return it?",
            False,
            "Enable Local Search and choose a registered folder.",
        ),
    ],
)
def test_deterministic_replies_use_human_language_without_inference(
    tmp_path, question, enabled, expected
):
    async def stream(messages, info):
        raise AssertionError("Deterministic reply must not invoke model")
        yield ""

    async def scenario():
        server = FolderServer([])
        server.enabled = enabled

        def handler(request):
            if request.url.path.endswith("/search"):
                assert json.loads(request.content)["query"] == question
                server.calls.append(request)
                return httpx.Response(200, json={"results": [], "mode": "keyword"})
            return server(request)

        store, runtime, request = await fixture(tmp_path, handler, stream)
        request = request.model_copy(update={"text": question})
        try:
            events = await collect(runtime, request)
            text = "".join(
                e["event"]["text"] for e in events if e["event"]["type"] == "text.delta"
            )
            assert text == expected
            assert (await store.get_turn(request.turn_id))["usage"] == {
                "input_tokens": 0,
                "output_tokens": 0,
                "requests": 0,
            }
            rows = await store.branch_messages(request.branch_id)
            assert rows[-1]["parts"] == [{"type": "text", "text": text}]
            assert not any(r.url.path.endswith("count_tokens") for r in server.calls)
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())
