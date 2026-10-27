"""Storage contracts: durable publication, branch isolation and crash recovery."""

import asyncio
import json
import os
import signal
import sqlite3
import subprocess
import sys
import uuid

import pytest

from rapid_mlx.chat_runtime.history import _project_messages
from rapid_mlx.chat_runtime.store import Conflict, Store


@pytest.fixture
async def store(tmp_path):
    value = await Store.open(tmp_path)
    yield value
    await value.close()


def request(conversation, **changes):
    return {
        "v": 1,
        "turn_id": str(uuid.uuid4()),
        "conversation_id": conversation["id"],
        "branch_id": conversation["branch_id"],
        "expected_revision": 0,
        "text": "Hello",
        "model": "test",
        "inference_url": "http://127.0.0.1:18800",
        **changes,
    }


async def completed(store, req, text="Answer"):
    turn, _ = await store.begin_turn(req)
    await store.finish_turn(
        turn["id"], "completed", parts=[{"type": "text", "text": text}]
    )
    return turn


async def test_idempotency_cas_and_terminal(store):
    c = await store.create_conversation()
    req = request(c)
    first, created = await store.begin_turn(req)
    again, duplicate = await store.begin_turn(req)
    assert created and not duplicate and first == again
    with pytest.raises(Conflict):
        await store.begin_turn({**req, "text": "different"})
    with pytest.raises(Conflict):
        await store.begin_turn(request(c))
    with pytest.raises(ValueError, match="nonempty"):
        await store.finish_turn(first["id"], "completed")
    terminal = await store.finish_turn(
        first["id"], "completed", parts=[{"type": "text", "text": "answer"}]
    )
    assert await store.finish_turn(first["id"], "error") == terminal
    assert [e["event"]["type"] for e in await store.replay_events(first["id"])] == [
        "turn.started",
        "turn.terminal",
    ]
    with pytest.raises(Conflict):
        await store.append_event(
            first["id"], {"type": "text.delta", "part_id": "x", "text": "late"}
        )


async def test_branch_edit_regenerate_and_operation_id(store):
    c = await store.create_conversation()
    await completed(store, request(c))
    original = await store.branch_messages(c["branch_id"])
    fork = await store.edit(c["branch_id"], original[0]["id"], "Edited", "edit-1")
    assert (
        await store.edit(c["branch_id"], original[0]["id"], "Edited", "edit-1") == fork
    )
    with pytest.raises(Conflict):
        await store.edit(c["branch_id"], original[0]["id"], "Other", "edit-1")
    edited = await store.branch_messages(fork["id"])
    assert edited[0]["parent_id"] is None
    assert edited[0]["id"] != original[0]["id"]
    regenerated = await store.regenerate(c["branch_id"], "regen-1")
    await completed(
        store,
        request(c, branch_id=regenerated["id"], mode="regenerate", text=""),
        "Second answer",
    )
    new = await store.branch_messages(regenerated["id"])
    assert new[0]["id"] == original[0]["id"]
    assert new[-1]["parent_id"] == original[-1]["parent_id"]
    assert await store.branch_messages(c["branch_id"]) == original


async def test_attachments_authorization_cas_and_preserved_edit(store):
    c, other = await store.create_conversation(), await store.create_conversation()
    a = await store.put_attachment(c["id"], "../../notes.txt", "héllo".encode())
    b = await store.put_attachment(c["id"], "copy.txt", "héllo".encode())
    assert a["digest"] == b["digest"] and a["id"] != b["id"]
    assert a["name"] == "notes.txt"
    assert await store.read_attachment(c["id"], a["id"], 1, 3) == "éll"
    with pytest.raises(KeyError):
        await store.read_attachment(other["id"], a["id"])
    with pytest.raises(ValueError):
        await store.put_attachment(c["id"], "big.txt", b"x" * (1024 * 1024 + 1))
    with pytest.raises(UnicodeDecodeError):
        await store.put_attachment(c["id"], "binary", b"\xff")
    await completed(store, request(c, attachment_ids=[a["id"]]))
    messages = await store.branch_messages(c["branch_id"])
    edit = await store.edit(c["branch_id"], messages[0]["id"], "new", "edit")
    assert (await store.branch_messages(edit["id"]))[0]["parts"][1][
        "attachment_id"
    ] == a["id"]


async def test_reasoning_never_reaches_database_or_wal(store):
    c = await store.create_conversation()
    turn, _ = await store.begin_turn(request(c, inference_api_key="secret-key-marker"))
    event = await store.append_event(
        turn["id"],
        {
            "type": "reasoning.delta",
            "part_id": "r",
            "text": "hidden-reasoning-marker",
            "provider_metadata": "secret-key-marker",
        },
    )
    assert event["event"]["text"] == "hidden-reasoning-marker"
    assert "provider_metadata" not in event["event"]
    await store.finish_turn(
        turn["id"],
        "completed",
        parts=[
            {
                "type": "text",
                "text": "safe",
                "reasoning_content": "hidden-reasoning-marker",
            },
            {"type": "thinking", "text": "hidden-reasoning-marker"},
        ],
    )
    events = await store.replay_events(turn["id"])
    assert events[1]["event"] == {
        "type": "reasoning.delta",
        "part_id": "r",
        "text": "",
        "omitted": True,
    }
    for path in store.directory.glob("history.sqlite*"):
        assert b"hidden-reasoning-marker" not in path.read_bytes()
        assert b"secret-key-marker" not in path.read_bytes()


@pytest.mark.parametrize("version", [1, 2, 3, Store.schema_version + 1])
async def test_unsupported_schema_refused_without_writes(tmp_path, version):
    db = sqlite3.connect(tmp_path / "history.sqlite")
    db.execute(f"PRAGMA user_version={version}")
    db.close()
    before = (tmp_path / "history.sqlite").read_bytes()
    with pytest.raises(ValueError, match="newer|experimental"):
        await Store.open(tmp_path)
    assert (tmp_path / "history.sqlite").read_bytes() == before


async def test_grouped_events_commit_before_publish_and_isolate_errors(store):
    c = await store.create_conversation()
    turn, _ = await store.begin_turn(request(c))
    jobs = [
        store.append_event(
            turn["id"], {"type": "text.delta", "part_id": "p", "text": str(i)}
        )
        for i in range(32)
    ]
    jobs.insert(3, store.append_event(turn["id"], {"type": "unknown"}))
    result = await asyncio.gather(*jobs, return_exceptions=True)
    assert isinstance(result[3], ValueError)
    db = sqlite3.connect(store.path)
    assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 33
    db.close()
    assert [e["seq"] for e in await store.replay_events(turn["id"])] == list(
        range(1, 34)
    )


async def test_tombstone_and_recovery_once(store):
    c = await store.create_conversation()
    turn, _ = await store.begin_turn(request(c))
    with pytest.raises(Conflict):
        await store.delete_conversation(c["id"])
    result = await store.recover_interrupted()
    assert result[0]["event"]["error_code"] == "owner_restarted"
    assert await store.recover_interrupted() == []
    assert (await store.get_turn(turn["id"]))["status"] == "error"
    await store.delete_conversation(c["id"])
    assert await store.list_conversations() == []
    with pytest.raises(KeyError):
        await store.get_conversation(c["id"])


@pytest.mark.parametrize("after_terminal", [False, True])
async def test_sigkill_before_and_after_terminal(tmp_path, after_terminal):
    marker = tmp_path / "ready.json"
    script = """
import asyncio,json,sys
from pathlib import Path
from rapid_mlx.chat_runtime.store import Store
async def main():
 s=await Store.open(sys.argv[1]); c=await s.create_conversation()
 t,_=await s.begin_turn(dict(turn_id='crash',conversation_id=c['id'],branch_id=c['branch_id'],expected_revision=0,text='prompt'))
 await s.append_event('crash',dict(type='text.delta',part_id='p',text='partial'))
 if sys.argv[2]=='True': await s.finish_turn('crash','completed',parts=[dict(type='text',text='final')])
 marker=Path(sys.argv[1],'ready.tmp')
 marker.write_text(json.dumps(c))
 marker.replace(Path(sys.argv[1],'ready.json'))
 await asyncio.Event().wait()
asyncio.run(main())
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), str(after_terminal)]
    )
    try:
        for _ in range(200):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        assert marker.exists()
        os.kill(proc.pid, signal.SIGKILL)
        await asyncio.to_thread(proc.wait)
        store = await Store.open(tmp_path)
        try:
            await store.recover_interrupted()
            events = await store.replay_events("crash")
            terminals = [e for e in events if e["event"]["type"] == "turn.terminal"]
            assert len(terminals) == 1
            assert terminals[0]["event"]["outcome"] == (
                "completed" if after_terminal else "error"
            )
            c = json.loads(marker.read_text())
            assert len(await store.branch_messages(c["branch_id"])) == (
                2 if after_terminal else 1
            )
        finally:
            await store.close()
    finally:
        if proc.poll() is None:
            proc.kill()
            await asyncio.to_thread(proc.wait)


async def test_projection_stable_ids_and_tool_round(store):
    c = await store.create_conversation()
    turn, _ = await store.begin_turn(request(c))
    await store.finish_turn(
        turn["id"],
        "completed",
        messages=[
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "text": "Reading."},
                    {
                        "type": "tool_call",
                        "call_id": "call",
                        "name": "read_attachment",
                        "args": {"attachment_id": "id"},
                    },
                ],
            },
            {
                "role": "tool",
                "parts": [
                    {
                        "type": "tool_result",
                        "call_id": "call",
                        "name": "read_attachment",
                        "result": "Document",
                    }
                ],
            },
            {"role": "assistant", "parts": [{"type": "text", "text": "Answer"}]},
        ],
    )
    rows = await store.branch_messages(c["branch_id"])
    first, second = _project_messages(rows), _project_messages(rows)
    assert first == second
    assert first[0].metadata["message_id"] == rows[0]["id"]
    assert first[1].parts[1].tool_call_id == first[2].parts[0].tool_call_id
    with pytest.raises(ValueError, match="incomplete"):
        _project_messages(rows[:2])
    with pytest.raises(ValueError, match="orphaned"):
        _project_messages([rows[0], *rows[2:]])


async def test_concurrent_admission_has_one_prompt_and_no_lost_update(store):
    c = await store.create_conversation()
    results = await asyncio.gather(
        *(store.begin_turn(request(c)) for _ in range(20)), return_exceptions=True
    )
    assert sum(isinstance(result, tuple) for result in results) == 1
    assert sum(isinstance(result, Conflict) for result in results) == 19
    assert len(await store.branch_messages(c["branch_id"])) == 1


async def test_invalid_tool_round_rolls_back_every_message(store):
    c = await store.create_conversation()
    turn, _ = await store.begin_turn(request(c))
    with pytest.raises(ValueError, match="orphaned"):
        await store.finish_turn(
            turn["id"],
            "completed",
            messages=[
                {
                    "role": "tool",
                    "parts": [
                        {
                            "type": "tool_result",
                            "call_id": "missing",
                            "name": "read_attachment",
                            "result": "bad",
                        }
                    ],
                },
                {"role": "assistant", "parts": [{"type": "text", "text": "answer"}]},
            ],
        )
    assert len(await store.branch_messages(c["branch_id"])) == 1
    assert (await store.get_turn(turn["id"]))["status"] == "running"


async def test_writer_io_never_blocks_event_loop(store):
    import threading

    started, release = threading.Event(), threading.Event()

    def slow_io(db):
        started.set()
        return release.wait(timeout=2)

    pending = asyncio.create_task(store.writer.call(slow_io))
    assert await asyncio.to_thread(started.wait, 1)
    release.set()
    assert await pending


@pytest.mark.parametrize("edit", [False, True])
async def test_regenerated_started_event_uses_persisted_user_parent(store, edit):
    c = await store.create_conversation()
    await completed(store, request(c))
    await completed(store, request(c, expected_revision=2, text="Follow-up"))
    messages = await store.branch_messages(c["branch_id"])
    if edit:
        branch = await store.edit(
            c["branch_id"], messages[-2]["id"], "Edited follow-up", "edit-parent"
        )
    else:
        branch = await store.regenerate(c["branch_id"], "regen-parent")
    user = (await store.branch_messages(branch["id"]))[-1]
    turn, _ = await store.begin_turn(
        request(c, branch_id=branch["id"], mode="regenerate", text="")
    )
    started = (await store.replay_events(turn["id"]))[0]["event"]
    assert started["message_id"] == user["id"]
    assert started["parent_message_id"] == user["parent_id"] == messages[1]["id"]
    assert started["message_id"] != started["parent_message_id"]


async def test_temperature_is_persisted_and_part_of_idempotency(store):
    c = await store.create_conversation()
    req = request(c, temperature=0.3)
    turn, _ = await store.begin_turn(req)
    assert turn["settings"]["temperature"] == 0.3
    with pytest.raises(Conflict):
        await store.begin_turn({**req, "temperature": 0.9})


async def test_long_history_trim_still_allows_model_request(store):
    from pydantic_ai.models.function import FunctionModel

    from rapid_mlx.chat_runtime.runtime import Runtime, TurnRequest

    c = await store.create_conversation()
    for i in range(30):
        await completed(store, request(c, expected_revision=i * 2, text="Hi"), "Hello")
    called = []

    async def answer(messages, info):
        called.append(messages)
        yield "Still here."

    runtime = Runtime(store, model=FunctionModel(stream_function=answer))
    try:
        req = TurnRequest(**request(c, expected_revision=60, text="Continue"))
        await runtime.submit(req)
        events = [event async for event in runtime.events(req.turn_id) if event]
        assert called
        assert events[-1]["event"]["outcome"] == "completed"
        assert (
            len(called[0]) <= 61
        )  # Token accounting no longer discards history that fits.
    finally:
        await runtime.close()


async def test_history_summaries_derive_title_and_selected_branch_message_count(store):
    c = await store.create_conversation()
    title = "  介绍一下你自己\n " + "海洋" * 30
    first = request(c, text=title)
    await completed(store, first)
    summary = (await store.list_conversations())[0]
    normalized = " ".join(title.split())
    assert summary["display_title"] == normalized[:39] + "…"
    assert len(summary["display_title"]) == 40
    assert summary["message_count"] == 2
    assert summary["title"] == "New chat"  # Display-only, no generated title mutation.
    branch = await store.regenerate(c["branch_id"], "summary-regenerate")
    summary = (await store.list_conversations())[0]
    assert summary["branch_id"] == branch["id"]
    assert summary["message_count"] == 1
    await completed(
        store, request(c, branch_id=branch["id"], mode="regenerate", text="")
    )
    assert (await store.list_conversations())[0]["message_count"] == 2
    empty = await store.create_conversation()
    summary = next(
        item for item in await store.list_conversations() if item["id"] == empty["id"]
    )
    assert summary["display_title"] == "New chat"
    assert summary["message_count"] == 0
