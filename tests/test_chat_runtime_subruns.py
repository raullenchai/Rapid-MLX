"""Hidden child persistence, recovery, approval scope and API boundaries."""

import json

import httpx
import pytest

from rapid_mlx.chat_runtime.api import PREFIX, create_app
from rapid_mlx.chat_runtime.events import Envelope
from rapid_mlx.chat_runtime.runtime import Runtime
from rapid_mlx.chat_runtime.store import Conflict, Store


async def parent(store):
    conversation = await store.create_conversation()
    await store.begin_turn(
        {
            "turn_id": "parent",
            "conversation_id": conversation["id"],
            "branch_id": conversation["branch_id"],
            "expected_revision": 0,
            "text": "Original parent question",
            "model": "fake",
            "inference_url": "http://127.0.0.1:18800",
        }
    )
    return conversation


async def child(store, id="helper", call_id="delegate"):
    return await store.create_subrun(
        id=id,
        turn_id="parent",
        call_id=call_id,
        title="Researching docs",
        task="Read docs",
        context="Only supplied context",
        tools=["read_file"],
    )


async def test_subruns_hidden_idempotent_redacted_and_authenticated(tmp_path):
    store = await Store.open(tmp_path)
    runtime = Runtime(store, web_enabled=False)
    try:
        conversation = await parent(store)
        first = await child(store)
        assert (await child(store, id="other"))["id"] == first["id"]
        with pytest.raises(Conflict):
            await store.create_subrun(
                id="other",
                turn_id="parent",
                call_id="delegate",
                title="Other",
                task="Different task",
                context=None,
                tools=[],
            )
        await store.update_subrun(
            "helper",
            status="running",
            messages=[
                {"role": "user", "parts": [{"type": "text", "text": "Read docs"}]},
                {
                    "role": "assistant",
                    "parts": [
                        {"type": "reasoning", "text": "NEVER_PERSIST_CHILD_REASONING"},
                        {"type": "text", "text": "HIDDEN_CHILD_ANSWER"},
                    ],
                },
            ],
        )
        await store.append_subrun_event(
            "helper",
            {
                "type": "reasoning.delta",
                "part_id": "thought",
                "text": "NEVER_PERSIST_CHILD_REASONING",
            },
        )
        await store.append_subrun_event(
            "helper",
            {
                "type": "tool.started",
                "call_id": "helper:read",
                "name": "read_file",
                "args": {"path": "docs/a.md"},
            },
        )
        await store.update_subrun(
            "helper", status="completed", summary="Docs summary", usage={"requests": 1}
        )
        await store.finish_turn(
            "parent", "completed", parts=[{"type": "text", "text": "Parent answer"}]
        )
        # The finalizer repairs a committed child completion not yet published.
        replay = await store.replay_events("parent")
        assert [e["event"]["type"] for e in replay] == [
            "turn.started",
            "subagent.finished",
            "turn.terminal",
        ]
        assert replay[1]["event"]["status"] == "completed"
        for event in replay:
            Envelope.model_validate(event)
        snapshot = await store.get_conversation(conversation["id"])
        assert "HIDDEN_CHILD_ANSWER" not in json.dumps(snapshot)
        assert len(snapshot["messages"]) == 2
        app = create_app(store, runtime, "credential")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://owner"
        ) as client:
            assert (await client.get(PREFIX + "/subruns/helper")).status_code == 401
            client.headers["Authorization"] = "Bearer credential"
            listing = (await client.get(PREFIX + "/turns/parent/subruns")).json()
            assert len(listing) == 1 and "messages" not in listing[0]
            detail = (await client.get(PREFIX + "/subruns/helper")).json()
            assert detail["status"] == "completed"
            assert detail["tool_count"] == 1
            assert detail["started_at"] <= detail["finished_at"]
            assert detail["events"][0]["event"]["text"] == ""
            assert detail["events"][0]["event"]["omitted"] is True
            assert detail["messages"][-1]["parts"] == [
                {"type": "text", "text": "HIDDEN_CHILD_ANSWER"}
            ]
            await store.delete_conversation(conversation["id"])
            assert (await client.get(PREFIX + "/subruns/helper")).status_code == 404
            assert (
                await client.get(PREFIX + "/turns/parent/subruns")
            ).status_code == 404
    finally:
        await runtime.close()
        await store.close()
    assert all(
        b"NEVER_PERSIST_CHILD_REASONING" not in path.read_bytes()
        for path in tmp_path.glob("*.sqlite*")
    )


@pytest.mark.parametrize("status", ["queued", "running"])
async def test_restart_finalizes_child_before_parent_and_retains_partial_transcript(
    tmp_path, status
):
    store = await Store.open(tmp_path)
    await parent(store)
    await child(store)
    await store.update_subrun(
        "helper",
        status=status,
        messages=[
            {
                "role": "assistant",
                "parts": [{"type": "text", "text": "Partial findings"}],
            },
        ],
    )
    await store.close()
    store = await Store.open(tmp_path)
    try:
        await store.recover_interrupted()
        record = await store.get_subrun("helper")
        assert record["status"] == "error"
        assert record["messages"][0]["parts"][0]["text"] == "Partial findings"
        events = await store.replay_events("parent")
        assert events[-2]["event"]["type"] == "subagent.finished"
        assert events[-1]["event"]["error_code"] == "owner_restarted"
        await store.recover_interrupted()
        assert await store.replay_events("parent") == events
        with pytest.raises(Conflict):
            await store.append_subrun_event(
                "helper", {"type": "text.delta", "part_id": "late", "text": "late"}
            )
    finally:
        await store.close()


async def test_child_timeout_only_settles_own_approval_operations(tmp_path):
    store = await Store.open(tmp_path)
    try:
        await parent(store)
        await child(store)
        operations = {}
        for call_id in (
            "helper:waiting",
            "helper:dispatched",
            "sibling:waiting",
            "parent-call",
        ):
            op, _ = await store.create_operation(
                turn_id="parent",
                call_id=call_id,
                tool="terminal",
                schema_revision=1,
                args={"command": "test " + call_id},
                risk="execution",
                reason="Test",
                summary="Test operation",
                preview="",
                prompt=True,
                allow_conversation=False,
            )
            if call_id == "helper:dispatched":
                op = await store.decide_operation(op["id"], op["revision"], "allow")
                op = await store.dispatch_operation(op["id"], op["revision"], 1)
            operations[call_id] = op["id"]
        await store.update_subrun(
            "helper", status="timeout", summary="Timed out; partial result"
        )
        states = {
            key: (await store.get_operation(id))["state"]
            for key, id in operations.items()
        }
        assert states == {
            "helper:waiting": "cancelled_before_effect",
            "helper:dispatched": "outcome_unknown",
            "sibling:waiting": "intent",
            "parent-call": "intent",
        }
        await store.finish_turn("parent", "cancelled")
        assert (await store.get_subrun("helper"))["status"] == "timeout"
    finally:
        await store.close()
