"""Model-free lifecycle contracts for the built-in chat runtime."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai.models.function import DeltaThinkingPart, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel as PydanticTestModel

from rapid_mlx.chat_runtime.events import Envelope, json_schema
from rapid_mlx.chat_runtime.model_client import ModelClients, inference_url
from rapid_mlx.chat_runtime.runtime import Runtime, TurnRequest
from rapid_mlx.chat_runtime.store import Store


def test_wire_golden_schema():
    root = Path(__file__).parent / "fixtures" / "chat_runtime"
    assert json_schema() == json.loads((root / "wire-v1.schema.json").read_text())
    for trace in root.glob("*.jsonl"):
        events = [
            Envelope.model_validate_json(line)
            for line in trace.read_text().splitlines()
        ]
        assert [event.seq for event in events] == list(range(1, len(events) + 1))
        assert sum(event.event.type == "turn.terminal" for event in events) == 1
        assert events[-1].event.type == "turn.terminal"
        for event in events:
            assert f"id: {event.turn_id}:{event.seq}\n".encode() in event.sse()


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:80",
        "http://example.com:80",
        "http://127.0.0.1:80/other",
        "http://user:secret@127.0.0.1:80",
        "http://localhost:80/?secret=x",
    ],
)
def test_reject_nonlocal_or_credential_urls(url):
    with pytest.raises(ValueError):
        inference_url(url)


def test_invalid_request_and_secret_excluded():
    data = dict(
        turn_id="id",
        conversation_id="c",
        branch_id="b",
        expected_revision=0,
        inference_url="http://localhost:18800",
        model="local",
        inference_api_key="SECRET",
    )
    request = TurnRequest(**data)
    assert "SECRET" not in request.model_dump_json()
    assert request.inference_url == "http://localhost:18800/v1"
    with pytest.raises(ValidationError):
        TurnRequest(**{**data, "turn_id": "bad:id"})


def test_client_settings():
    async def scenario():
        clients = ModelClients()
        model = clients.model("http://localhost:18800", "local")
        assert clients.model("http://localhost:18800/v1", "local") is model
        assert model.client.max_retries == 0
        assert model.profile["openai_chat_send_back_thinking_parts"] == "field"
        assert model.profile["openai_chat_thinking_field"] == "reasoning_content"
        await clients.close()

    asyncio.run(scenario())


async def setup_runtime(path, model):
    store = await Store.open(path)
    conversation = await store.create_conversation()
    runtime = Runtime(store, model=model, web_enabled=False)
    request = TurnRequest(
        turn_id="turn",
        conversation_id=conversation["id"],
        branch_id=conversation["branch_id"],
        expected_revision=0,
        text="hello",
        inference_url="http://localhost:18800",
        model="local",
    )
    return store, runtime, request


async def collect(runtime, request):
    await runtime.submit(request)
    return [e async for e in runtime.events(request.turn_id) if e]


def test_stream_idempotent_terminal_and_persisted_history(tmp_path):
    async def stream(messages, info):
        assert info.model_settings["temperature"] == 0.3
        yield "Hello "
        yield "there."

    async def scenario():
        store, runtime, request = await setup_runtime(
            tmp_path, FunctionModel(stream_function=stream)
        )
        try:
            request.temperature = 0.3
            events = await collect(runtime, request)
            assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
            terminal = events[-1]["event"]
            assert terminal["outcome"] == "completed"
            assert terminal["final_message_id"]
            assert (
                len([e for e in events if e["event"]["type"] == "turn.terminal"]) == 1
            )
            _, created = await runtime.submit(request)
            assert not created
            assert (await store.branch_messages(request.branch_id))[-1]["parts"] == [
                {"type": "text", "text": "Hello there."}
            ]
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("recover", [True, False])
def test_empty_final_one_tools_disabled_recovery(tmp_path, recover):
    calls = []

    async def stream(messages, info):
        calls.append(len(info.function_tools))
        if len(calls) == 2 and recover:
            yield "Recovered answer"
        else:
            yield {0: DeltaThinkingPart(content="private reasoning")}

    async def scenario():
        store, runtime, request = await setup_runtime(
            tmp_path, FunctionModel(stream_function=stream)
        )
        attachment = await store.put_attachment(
            request.conversation_id, "fact.txt", b"A fact for synthesis."
        )
        request.attachment_ids = [attachment["id"]]
        try:
            events = await collect(runtime, request)
            assert calls == [2, 0]
            terminal = events[-1]["event"]
            assert terminal["outcome"] == ("completed" if recover else "error")
            durable = await store.replay_events(request.turn_id)
            assert "private reasoning" not in json.dumps(durable)
            for file in tmp_path.glob("*.sqlite*"):
                assert b"private reasoning" not in file.read_bytes()
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


def test_tool_preamble_does_not_complete_turn(tmp_path):
    calls = 0
    attachment_id = ""

    async def stream(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield "I will read the file."
            yield {
                0: DeltaToolCall(
                    name="read_attachment",
                    json_args=json.dumps({"attachment_id": attachment_id}),
                    tool_call_id="read-1",
                )
            }
        else:
            assert any(
                p.part_kind == "tool-return" and "blue" in p.content
                for m in messages
                for p in m.parts
            )
            yield "The project color is blue."

    async def scenario():
        nonlocal attachment_id
        store, runtime, request = await setup_runtime(
            tmp_path, FunctionModel(stream_function=stream)
        )
        attachment = await store.put_attachment(
            request.conversation_id, "fact.txt", b"The project color is blue."
        )
        attachment_id = attachment["id"]
        request.attachment_ids = [attachment_id]
        try:
            events = await collect(runtime, request)
            kinds = [e["event"]["type"] for e in events]
            assert kinds.count("turn.terminal") == 1
            assert (
                kinds.index("tool.started")
                < kinds.index("tool.result")
                < kinds.index("turn.terminal")
            )
            assert events[-1]["event"]["outcome"] == "completed"
            rows = await store.branch_messages(request.branch_id)
            assert [row["role"] for row in rows] == [
                "user",
                "assistant",
                "tool",
                "assistant",
            ]
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


def test_cancel_awaits_model_cleanup_and_disconnect_does_not_cancel(tmp_path):
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def stream(messages, info):
        try:
            yield "Starting"
            started.set()
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    async def scenario():
        store, runtime, request = await setup_runtime(
            tmp_path, FunctionModel(stream_function=stream)
        )
        try:
            await runtime.submit(request)
            iterator = runtime.events(request.turn_id)
            await anext(iterator)
            await iterator.aclose()
            await asyncio.wait_for(started.wait(), 2)
            assert runtime.active_turns == 1
            await runtime.cancel(request.turn_id)
            assert cleaned.is_set()
            assert runtime.active_turns == 0
            events = await store.replay_events(request.turn_id)
            assert events[-1]["event"]["outcome"] == "cancelled"
            assert (
                len([e for e in events if e["event"]["type"] == "turn.terminal"]) == 1
            )
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


def test_cancel_before_task_starts(tmp_path):
    async def scenario():
        store, runtime, request = await setup_runtime(
            tmp_path, PydanticTestModel(custom_output_text="Answer", call_tools=[])
        )
        try:
            await runtime.submit(request)
            await runtime.cancel(request.turn_id)
            assert (await store.get_turn(request.turn_id))["status"] == "cancelled"
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


def test_openai_transport_maps_reasoning_and_explicit_thinking():
    import httpx
    from pydantic_ai import Agent
    from pydantic_ai.messages import ThinkingPart

    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        chunks = [
            {
                "id": "chat-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "local",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "private deliberation",
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chat-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "local",
                "choices": [
                    {"index": 0, "delta": {"content": "42"}, "finish_reason": None}
                ],
            },
            {
                "id": "chat-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "local",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        content = (
            "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=content
        )

    async def scenario():
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients = ModelClients(http_client=http_client)
        model = clients.model("http://localhost:18800", "local")
        agent = Agent(model)
        try:
            for thinking in (False, True):
                async with agent.iter(
                    "answer",
                    model_settings={
                        "extra_body": {
                            "chat_template_kwargs": {"enable_thinking": thinking}
                        }
                    },
                ) as run:
                    async for node in run:
                        if Agent.is_model_request_node(node):
                            async with node.stream(run.ctx) as stream:
                                async for _ in stream:
                                    pass
                assert run.result.output == "42"
                assert any(
                    isinstance(p, ThinkingPart)
                    for m in run.result.new_messages()
                    for p in m.parts
                )
            assert [r["chat_template_kwargs"]["enable_thinking"] for r in requests] == [
                False,
                True,
            ]
        finally:
            await clients.close()

    asyncio.run(scenario())


def test_cancelled_submit_still_owns_committed_turn(tmp_path):
    async def stream(messages, info):
        yield "Done"

    async def scenario():
        store, runtime, request = await setup_runtime(
            tmp_path, FunctionModel(stream_function=stream)
        )
        original = store.begin_turn
        committed = asyncio.Event()
        release = asyncio.Event()

        async def delayed_begin(body):
            result = await original(body)
            committed.set()
            await release.wait()
            return result

        store.begin_turn = delayed_begin
        try:
            submission = asyncio.create_task(runtime.submit(request))
            await committed.wait()
            submission.cancel()
            with pytest.raises(asyncio.CancelledError):
                await submission
            release.set()
            _, created = await runtime.submit(request)
            assert not created
            events = [e async for e in runtime.events(request.turn_id) if e]
            assert events[-1]["event"]["outcome"] == "completed"
        finally:
            release.set()
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


def test_current_prompt_respects_context_budget(tmp_path):
    calls = 0

    async def stream(messages, info):
        nonlocal calls
        calls += 1
        yield "answer"

    async def scenario():
        store, runtime, request = await setup_runtime(
            tmp_path, FunctionModel(stream_function=stream)
        )
        request.context_tokens = 2048
        request.max_tokens = 512
        request.text = "x" * 1024
        try:
            events = await collect(runtime, request)
            assert events[-1]["event"]["outcome"] == "completed"
            assert calls == 1
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


def test_tool_loop_growth_respects_context_budget(tmp_path):
    calls = 0

    async def stream(messages, info):
        nonlocal calls
        calls += 1
        if calls > 1:
            yield "The attachment was unavailable; no answer could be verified."
            return
        yield "preamble " * 128
        yield {
            0: DeltaToolCall(
                name="read_attachment",
                json_args='{"attachment_id":"missing"}',
                tool_call_id="read-1",
            )
        }

    async def scenario():
        store, runtime, request = await setup_runtime(
            tmp_path, FunctionModel(stream_function=stream)
        )
        attachment = await store.put_attachment(
            request.conversation_id, "fact.txt", b"A fact."
        )
        request.attachment_ids = [attachment["id"]]
        request.context_tokens = 3072
        request.max_tokens = 512
        try:
            events = await collect(runtime, request)
            assert events[-1]["event"]["outcome"] == "completed"
            assert calls == 2
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


def test_disconnected_cancel_request_does_not_interrupt_cleanup(tmp_path):
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleaned = asyncio.Event()

    async def stream(messages, info):
        try:
            yield "Starting"
            started.set()
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()
            cleaned.set()

    async def scenario():
        store, runtime, request = await setup_runtime(
            tmp_path, FunctionModel(stream_function=stream)
        )
        try:
            await runtime.submit(request)
            await started.wait()
            stop_request = asyncio.create_task(runtime.cancel(request.turn_id))
            await cleanup_started.wait()
            stop_request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await stop_request
            release_cleanup.set()
            result = await runtime.cancel(request.turn_id)
            assert cleaned.is_set()
            assert result["status"] == "cancelled"
            events = await store.replay_events(request.turn_id)
            assert (
                len([e for e in events if e["event"]["type"] == "turn.terminal"]) == 1
            )
        finally:
            release_cleanup.set()
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


def test_slow_live_observer_receives_reasoning_after_terminal_commit(tmp_path):
    release = asyncio.Event()

    async def stream(messages, info):
        await release.wait()
        yield {0: DeltaThinkingPart(content="live-only thought")}
        yield "Answer"

    async def scenario():
        store, runtime, request = await setup_runtime(
            tmp_path, FunctionModel(stream_function=stream)
        )
        try:
            await runtime.submit(request)
            task = runtime._tasks[request.turn_id]
            observer = runtime.events(request.turn_id)
            first = await anext(observer)
            assert first["event"]["type"] == "turn.started"
            release.set()
            await task
            replay = [e async for e in runtime.events(request.turn_id) if e]
            assert "live-only thought" not in json.dumps(replay)
            rest = [e async for e in observer if e]
            assert "live-only thought" in json.dumps(rest)
            assert runtime._tasks == {}
            assert runtime._signals == {}
            assert runtime._volatile == {}
        finally:
            release.set()
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


def test_tool_cannot_read_unattached_id_or_model_supplied_path(tmp_path):
    calls = 0
    uploaded_id = ""

    async def stream(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield {
                0: DeltaToolCall(
                    name="read_attachment",
                    json_args=json.dumps({"attachment_id": uploaded_id}),
                    tool_call_id="hidden-file",
                )
            }
        else:
            returns = [
                p.content
                for m in messages
                for p in m.parts
                if p.part_kind == "tool-return"
            ]
            assert returns == ["Attachment is not authorized on this branch."]
            yield "Please attach the document first."

    async def scenario():
        nonlocal uploaded_id
        store, runtime, request = await setup_runtime(
            tmp_path, FunctionModel(stream_function=stream)
        )
        uploaded = await store.put_attachment(
            request.conversation_id, "private.txt", b"hidden content"
        )
        uploaded_id = uploaded["id"]
        allowed = await store.put_attachment(
            request.conversation_id, "public.txt", b"authorized content"
        )
        request.attachment_ids = [allowed["id"]]
        # Upload alone does not authorize access from every branch.
        try:
            events = await collect(runtime, request)
            assert events[-1]["event"]["outcome"] == "completed"
            assert "hidden content" not in json.dumps(events)
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


def test_attachment_tools_and_instructions_follow_branch_on_persistent_agent(tmp_path):
    from rapid_mlx.chat_runtime.runtime import BASE_INSTRUCTIONS

    seen = []

    async def stream(messages, info):
        seen.append(([tool.name for tool in info.function_tools], info.instructions))
        yield "How can I help?"

    async def scenario():
        store, runtime, request = await setup_runtime(
            tmp_path, FunctionModel(stream_function=stream)
        )
        persistent_agent = runtime.agent
        try:
            await collect(runtime, request)
            first_user = (await store.branch_messages(request.branch_id))[0]["id"]
            uploaded = await store.put_attachment(
                request.conversation_id, "notes.txt", b"A selected reference."
            )
            # An upload without a message attachment must not change the persona.
            revision = (await store.get_branch(request.branch_id))["revision"]
            await collect(
                runtime,
                request.model_copy(
                    update={
                        "turn_id": "uploaded-only",
                        "expected_revision": revision,
                        "text": "Introduce yourself",
                    }
                ),
            )
            revision = (await store.get_branch(request.branch_id))["revision"]
            await collect(
                runtime,
                request.model_copy(
                    update={
                        "turn_id": "attached",
                        "expected_revision": revision,
                        "text": "Read my notes",
                        "attachment_ids": [uploaded["id"]],
                    }
                ),
            )
            # Editing an earlier, unattached user prompt creates a branch that
            # must not inherit the later branch's tools or instructions.
            branch = await store.edit(
                request.branch_id, first_user, "Introduce yourself", "edit-first"
            )
            await collect(
                runtime,
                request.model_copy(
                    update={
                        "turn_id": "sibling",
                        "branch_id": branch["id"],
                        "expected_revision": branch["revision"],
                        "mode": "regenerate",
                        "text": "",
                    }
                ),
            )
            assert runtime.agent is persistent_agent
            assert [set(tools) for tools, _ in seen] == [
                {"delegate_task"},
                {"delegate_task"},
                {"delegate_task", "read_attachment"},
                {"delegate_task"},
            ]
            plain = [
                instructions
                for tools, instructions in seen
                if "read_attachment" not in tools
            ]
            assert len(plain) == 3 and len(set(plain)) == 1
            assert plain[0].startswith(BASE_INSTRUCTIONS)
            assert "read_attachment" not in plain[0]
            assert "read_attachment" in seen[2][1]
            assert "untrusted" in seen[2][1]
            assert "being genuinely useful over being verbose" in BASE_INSTRUCTIONS
        finally:
            await runtime.close()
            await store.close()

    asyncio.run(scenario())


def test_localhost_preserves_dual_stack_resolution():
    assert inference_url("http://localhost:18800") == "http://localhost:18800/v1"
    assert inference_url("http://[::1]:18800") == "http://[::1]:18800/v1"


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
@pytest.mark.parametrize("fail", [False, True])
def test_loopback_transports_pin_and_fallback(monkeypatch, host, fail):
    import httpx

    from rapid_mlx.chat_runtime.loopback_transport import (
        AsyncLoopbackTransport,
        LoopbackTransport,
    )

    sync_seen = []
    async_seen = []

    def sync_send(self, request):
        sync_seen.append(request.url.host)
        if fail and request.url.host == "127.0.0.1":
            raise httpx.ConnectError("unavailable")
        return httpx.Response(200)

    async def async_send(self, request):
        async_seen.append(request.url.host)
        if fail and request.url.host == "127.0.0.1":
            raise httpx.ConnectError("unavailable")
        return httpx.Response(200)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", sync_send)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_send)
    request = httpx.Request("GET", f"http://{host}:1234")
    expected = ["127.0.0.1", "::1"] if host == "localhost" and fail else ["127.0.0.1"]
    transport = LoopbackTransport()
    if host != "localhost" and fail:
        with pytest.raises(httpx.ConnectError):
            transport.handle_request(request)
    else:
        transport.handle_request(request)
    assert sync_seen == expected and request.url.host == host

    async def run():
        async_transport = AsyncLoopbackTransport()
        if host != "localhost" and fail:
            with pytest.raises(httpx.ConnectError):
                await async_transport.handle_async_request(request)
        else:
            await async_transport.handle_async_request(request)
        await async_transport.aclose()

    asyncio.run(run())
    transport.close()
    assert async_seen == expected and request.url.host == host


def test_loopback_transports_restore_url_when_both_connections_fail(monkeypatch):
    import httpx

    from rapid_mlx.chat_runtime.loopback_transport import (
        AsyncLoopbackTransport,
        LoopbackTransport,
    )

    def fail(*args):
        raise httpx.ConnectError("unavailable")

    async def async_fail(*args):
        raise httpx.ConnectError("unavailable")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fail)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_fail)
    request = httpx.Request("GET", "http://localhost:1234")
    transport = LoopbackTransport()
    with pytest.raises(httpx.ConnectError):
        transport.handle_request(request)
    transport.close()

    async def run():
        transport = AsyncLoopbackTransport()
        with pytest.raises(httpx.ConnectError):
            await transport.handle_async_request(request)
        await transport.aclose()

    asyncio.run(run())
    assert request.url.host == "localhost"
