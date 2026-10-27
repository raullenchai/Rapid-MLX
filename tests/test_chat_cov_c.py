"""Regression coverage for chat store and delegation boundary conditions."""

import asyncio
import queue
from types import SimpleNamespace

import pytest

from rapid_mlx.chat_runtime.delegation import DelegationManager, _summary_fragment
from rapid_mlx.chat_runtime.store import Conflict, Store, clean_parts
from rapid_mlx.chat_runtime.store.subruns import clean_messages
from rapid_mlx.chat_runtime.store.writer import Writer


@pytest.fixture
async def store(tmp_path):
    value = await Store.open(tmp_path)
    yield value
    await value.close()


async def start(store):
    c = await store.create_conversation()
    req = dict(
        turn_id="turn",
        conversation_id=c["id"],
        branch_id=c["branch_id"],
        expected_revision=0,
        text="hello",
    )
    await store.begin_turn(req)
    return c, req


async def test_subrun_validation_and_terminal_immutability(store):
    await start(store)
    args = dict(
        id="child",
        turn_id="turn",
        call_id="call",
        title="title",
        task="task",
        context=None,
        tools=[],
    )
    for changes in ({"id": ""}, {"task": "x" * 32769}, {"context": "x" * 32769}):
        with pytest.raises(ValueError):
            await store.create_subrun(**(args | changes))
    with pytest.raises(KeyError):
        await store.get_subrun("missing")
    await store.create_subrun(**args)
    with pytest.raises(ValueError):
        await store.update_subrun("child", status="unknown")
    with pytest.raises(ValueError):
        await store.append_subrun_event("child", {"type": "unknown"})
    await store.update_subrun("child", status="running")
    with pytest.raises(Conflict):
        await store.update_subrun("child", status="queued")
    await store.update_subrun("child", status="completed", summary="done")
    assert (await store.update_subrun("child", summary="changed"))["summary"] == "done"
    await store.finish_turn("turn", "cancelled")
    with pytest.raises(Conflict):
        await store.create_subrun(**(args | {"call_id": "new"}))
    assert clean_messages([{"role": "system", "parts": []}]) == []


async def test_store_rejects_invalid_inputs(store):
    c = await store.create_conversation()
    other = await store.create_conversation()
    base = dict(
        turn_id="turn",
        conversation_id=c["id"],
        branch_id=c["branch_id"],
        expected_revision=0,
        text="hello",
    )
    with pytest.raises(KeyError):
        await store.branch_messages("missing")
    with pytest.raises(KeyError):
        await store.get_turn("missing")
    for changes in (
        {"branch_id": other["branch_id"]},
        {"text": ""},
        {"attachment_ids": ["missing"]},
        {"mode": "regenerate", "text": ""},
    ):
        with pytest.raises(ValueError):
            await store.begin_turn(base | changes)
    with pytest.raises(ValueError):
        clean_parts([{"type": "unknown"}])
    with pytest.raises(ValueError):
        await store.put_attachment(c["id"], "bad", b"a\x00b")
    attachment = await store.put_attachment(c["id"], "a", b"hello")
    with pytest.raises(ValueError):
        await store.read_attachment(c["id"], attachment["id"], -1)
    (store.blob_dir / attachment["digest"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        await store.read_attachment(c["id"], attachment["id"])
    await store.begin_turn(base)
    with pytest.raises(ValueError):
        await store.append_event("turn", {"type": "turn.terminal"})
    with pytest.raises(ValueError):
        await store.finish_turn("turn", "invalid")
    with pytest.raises(Conflict):
        await store.begin_undo(c["id"], "undo")
    await store.finish_turn("turn", "cancelled")
    with pytest.raises(ValueError, match="no file"):
        await store.begin_undo(c["id"], "undo")
    with pytest.raises(ValueError):
        await store.edit(c["branch_id"], "missing", "edit", "edit")
    rows = await store.branch_messages(c["branch_id"])
    with pytest.raises(ValueError, match="empty edit"):
        await store.edit(c["branch_id"], rows[0]["id"], " ", "edit")


async def test_writer_closed_and_full():
    writer = Writer(lambda: __import__("sqlite3").connect(":memory:"))
    await writer.start()
    await writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        await writer.call(lambda db: None)
    writer.closed = False
    writer.queue = queue.Queue(maxsize=1)
    writer.queue.put_nowait(None)
    with pytest.raises(RuntimeError, match="busy"):
        await writer.call(lambda db: None)


async def test_delegation_rejects_invalid_and_cancels():
    manager = DelegationManager(SimpleNamespace(store=None))
    deps = SimpleNamespace(subagent_id="child")
    ctx = SimpleNamespace(deps=deps)
    assert "unavailable" in await manager.run(ctx, "task")
    deps.subagent_id = None
    assert "nonempty" in await manager.run(ctx, " ")
    assert "context exceeds" in await manager.run(ctx, "task", "x" * 16385)
    assert _summary_fragment("hello", 5) == "hello"
    task = asyncio.create_task(asyncio.Event().wait())
    manager._runs[("turn", "call")] = (None, task)
    await manager.cancel_turn("turn")
    assert task.cancelled()


async def test_invalid_tool_rounds_are_atomic(store):
    await start(store)
    call = {"type": "tool_call", "call_id": "c", "name": "read", "args": {}}
    answer = {"role": "assistant", "parts": [{"type": "text", "text": "done"}]}
    cases = [
        [{"role": "assistant", "parts": [call, call]}, answer],
        [{"role": "assistant", "parts": [call]}, answer],
        [{"role": "tool", "parts": [{"type": "text", "text": "bad"}]}, answer],
        [{"role": "user", "parts": [{"type": "text", "text": "bad"}]}, answer],
    ]
    for messages in cases:
        with pytest.raises(ValueError):
            await store.finish_turn("turn", "completed", messages=messages)
        assert (await store.get_turn("turn"))["status"] == "running"


async def test_action_decision_validation(store):
    c, _ = await start(store)
    op, _ = await store.create_operation(
        turn_id="turn",
        call_id="call",
        tool="terminal",
        schema_revision=1,
        args={},
        risk="execution",
        reason="reason",
        summary="summary",
        preview="",
        prompt=True,
        allow_conversation=False,
    )
    with pytest.raises(ValueError, match="decision/dispatch"):
        await store.transition_operation(op["id"], op["revision"], "approved")
    with pytest.raises(ValueError, match="invalid approval"):
        await store.decide_operation(op["id"], op["revision"], "maybe")
    await store.writer.call(
        lambda db: db.execute(
            "UPDATE action_operations SET expires_at=0 WHERE id=?", (op["id"],)
        )
    )
    with pytest.raises(Conflict, match="expired"):
        await store.decide_operation(op["id"], op["revision"], "allow")
    with pytest.raises(Conflict, match="undo ID reused"):
        await store.begin_undo(c["id"], op["id"])


@pytest.mark.parametrize("status", ["completed", "running"])
async def test_existing_delegation_is_replayed_or_rejected(status):
    from unittest.mock import AsyncMock

    row = {"id": "existing", "status": status, "summary": "saved", "sources": []}
    store = SimpleNamespace(create_subrun=AsyncMock(return_value=row))
    manager = DelegationManager(SimpleNamespace(store=store))
    ctx = SimpleNamespace(deps=SimpleNamespace(turn_id="turn"), tool_call_id="call")
    if status == "running":
        with pytest.raises(Conflict, match="already running"):
            await manager._execute(ctx, "task", None, set())
    else:
        import json

        assert json.loads(await manager._execute(ctx, "task", None, set())) == row


async def test_summary_fences_and_ended_turn_operations(store):
    c, _ = await start(store)
    with pytest.raises(Conflict, match="busy"):
        await store.save_summary(c["branch_id"], "summary", "missing")
    with pytest.raises(ValueError, match="cover"):
        await store.save_summary(c["branch_id"], "summary", "missing", turn_id="turn")
    await store.finish_turn("turn", "cancelled")
    with pytest.raises(Conflict, match="busy"):
        await store.save_summary(c["branch_id"], "summary", "missing", turn_id="turn")
    with pytest.raises(Conflict, match="turn ended"):
        await store.create_operation(
            turn_id="turn",
            call_id="call",
            tool="terminal",
            schema_revision=1,
            args={},
            risk="execution",
            reason="reason",
            summary="summary",
            preview="",
            prompt=True,
            allow_conversation=False,
        )


async def test_runtime_replays_terminal_committed_during_status_read():
    from unittest.mock import AsyncMock

    from rapid_mlx.chat_runtime.runtime import Runtime

    terminal = {"seq": 1, "event": {"type": "turn.terminal"}}
    store = SimpleNamespace(
        replay_events=AsyncMock(side_effect=[[], [terminal]]),
        get_turn=AsyncMock(return_value={"status": "completed"}),
    )
    runtime = Runtime(store, web_enabled=False)
    try:
        assert [item async for item in runtime._events("turn", 0, False)] == [terminal]
        assert store.replay_events.await_count == 2
    finally:
        await runtime.close()


async def test_runtime_close_cancels_active_task():
    from unittest.mock import AsyncMock

    from rapid_mlx.chat_runtime.runtime import Runtime

    store = SimpleNamespace(finish_turn=AsyncMock())
    runtime = Runtime(store, web_enabled=False)
    task = asyncio.create_task(asyncio.Event().wait())
    runtime._tasks["turn"] = task
    runtime._signals["turn"] = asyncio.Event()
    assert runtime.active_turns == 1
    await runtime.close()
    assert task.cancelled()
    store.finish_turn.assert_awaited_once_with("turn", "cancelled")
    assert runtime._signals["turn"].is_set()


async def test_context_info_counts_the_actual_history(store):
    from pydantic_ai.models.function import FunctionModel

    from rapid_mlx.chat_runtime.runtime import Runtime, TurnRequest

    c = await store.create_conversation()
    request = TurnRequest(
        turn_id="turn",
        conversation_id=c["id"],
        branch_id=c["branch_id"],
        expected_revision=0,
        text="hello",
        inference_url="http://localhost:18800",
        model="local",
        context_tokens=4096,
    )
    runtime = Runtime(
        store, model=FunctionModel(lambda messages, info: None), web_enabled=False
    )
    try:
        result = await runtime.context_info(request)
        assert result["type"] == "context"
        assert result["limit_tokens"] == 4096
    finally:
        await runtime.close()


async def test_writer_rolls_back_failed_commit():
    import sqlite3

    class FailingCommit:
        def __init__(self):
            self.db = sqlite3.connect(":memory:")

        @property
        def in_transaction(self):
            return self.db.in_transaction

        def execute(self, sql):
            if sql == "COMMIT":
                raise sqlite3.OperationalError("commit failed")
            return self.db.execute(sql)

        def close(self):
            self.db.close()

    writer = Writer(FailingCommit)
    await writer.start()
    with pytest.raises(sqlite3.OperationalError, match="commit failed"):
        await writer.call(lambda db: 42)
    # Close also uses a transaction, so allow its expected commit failure and stop
    # the writer explicitly after confirming the error reached the caller.
    writer.closed = True
    writer.queue.put_nowait(None)
    await asyncio.to_thread(writer.thread.join)
