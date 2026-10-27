"""Helpers keep private history while sharing the owner's policy and lifecycle."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from rapid_mlx.chat_runtime.api import PREFIX, create_app
from rapid_mlx.chat_runtime.delegation import ChildLimits
from rapid_mlx.chat_runtime.runtime import Runtime, TurnRequest
from rapid_mlx.chat_runtime.store import Store
from rapid_mlx.chat_runtime.tools import TurnDeps
from rapid_mlx.chat_runtime.tools.delegate import delegate_task
from rapid_mlx.chat_runtime.tools.web import SearchResult, WebResult


def child_prompt(messages):
    return next(
        (
            p.content
            for m in messages
            for p in m.parts
            if p.part_kind == "user-prompt"
            and isinstance(p.content, str)
            and "\nTask: " in p.content
        ),
        None,
    )


def returns(messages):
    return [p for m in messages for p in m.parts if p.part_kind == "tool-return"]


def delegate_calls(tasks, tools=None):
    return {
        index: DeltaToolCall(
            name="delegate_task",
            json_args=json.dumps(
                {"task": task, "context": "Explicit shared context", "tools": tools}
            ),
            tool_call_id=f"helper-{index}",
        )
        for index, task in enumerate(tasks)
    }


async def setup(tmp_path, stream, *, web=None):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("Verified workspace fact")
    store = await Store.open(tmp_path / "profile")
    conversation = await store.create_conversation(workspace=str(workspace))
    runtime = Runtime(
        store,
        model=FunctionModel(stream_function=stream),
        web_enabled=web is not None,
        web_tools=web,
    )
    request = TurnRequest(
        turn_id="parent",
        conversation_id=conversation["id"],
        branch_id=conversation["branch_id"],
        expected_revision=0,
        text="PARENT PRIVATE HISTORY: ask helpers",
        inference_url="http://localhost:18800",
        model="fake",
        max_tokens=1024,
        context_tokens=16384,
    )
    return store, runtime, request, workspace


async def collect(runtime, request):
    await runtime.submit(request)
    return [e async for e in runtime.events(request.turn_id) if e]


async def test_delegation_isolated_history_subset_persistence_and_api(tmp_path):
    async def stream(messages, info):
        prompt = child_prompt(messages)
        if prompt:
            assert "PARENT PRIVATE HISTORY" not in str(messages)
            assert "Explicit shared context" in prompt
            assert {t.name for t in info.function_tools} == {"read_file"}
            assert "delegate_task" not in info.instructions
            assert "web_search" not in info.instructions
            if returns(messages):
                yield "Private helper summary from verified workspace fact"
            else:
                yield {
                    0: DeltaToolCall(
                        name="read_file",
                        json_args='{"path":"note.txt"}',
                        tool_call_id="read",
                    )
                }
        elif returns(messages):
            result = json.loads(returns(messages)[-1].content)
            assert result["status"] == "completed"
            assert "Private helper summary" in result["summary"]
            yield "Combined parent answer"
        else:
            assert "delegate_task" in {t.name for t in info.function_tools}
            yield delegate_calls(["Summarize note"], ["read_file"])

    store, runtime, request, _ = await setup(tmp_path, stream)
    try:
        events = await collect(runtime, request)
        assert events[-1]["event"]["outcome"] == "completed"
        assert [
            e["event"]["text"] for e in events if e["event"]["type"] == "text.delta"
        ] == ["Combined parent answer"]
        assert not any(e["event"].get("name") == "read_file" for e in events)
        activity = [
            e["event"] for e in events if e["event"]["type"].startswith("subagent.")
        ]
        assert all(
            "Private helper summary" not in json.dumps(event) for event in activity
        )
        assert activity[-1]["summary"] == "Completed 1 tools"
        children = await store.list_subruns(request.turn_id)
        assert len(children) == 1 and children[0]["status"] == "completed"
        child = await store.get_subrun(children[0]["id"])
        assert child["messages"][0]["role"] == "user"
        assert any(e["event"]["type"] == "tool.result" for e in child["events"])
        assert child["usage"]["requests"] == 2
        # Parent makes two requests and rolls up the child's two exactly once.
        assert events[-1]["event"]["usage"]["requests"] == 4
        rows = await store.branch_messages(request.branch_id)
        assert not any(
            p.get("name") == "read_file" for row in rows for p in row["parts"]
        )
        app = create_app(store, runtime, "test-token")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://owner",
            headers={"Authorization": "Bearer test-token"},
        ) as client:
            listing = await client.get(PREFIX + f"/turns/{request.turn_id}/subruns")
            assert listing.status_code == 200
            detail = await client.get(PREFIX + f"/subruns/{child['id']}")
            assert detail.status_code == 200
            assert detail.json()["messages"] == child["messages"]
        assert await store.replay_events(request.turn_id) == events
    finally:
        await runtime.close()
        await store.close()
    reopened = await Store.open(tmp_path / "profile")
    try:
        assert (await reopened.get_subrun(child["id"]))["messages"] == child["messages"]
    finally:
        await reopened.close()


async def test_concurrency_is_capped_at_three(tmp_path):
    active = peak = finished = 0

    async def stream(messages, info):
        nonlocal active, peak, finished
        if child_prompt(messages):
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.04)
                yield "Helper done"
                finished += 1
            finally:
                active -= 1
        elif returns(messages):
            yield "Five helpers combined"
        else:
            yield delegate_calls([f"Task {index}" for index in range(5)], [])

    store, runtime, request, _ = await setup(tmp_path, stream)
    try:
        events = await collect(runtime, request)
        assert events[-1]["event"]["outcome"] == "completed"
        assert peak == 3 and finished == 5 and active == 0
        rows = await store.list_subruns(request.turn_id)
        assert len(rows) == 5 and all(
            r["started_at"] and r["finished_at"] for r in rows
        )
    finally:
        await runtime.close()
        await store.close()


async def test_recursion_and_unavailable_subset_rejected(tmp_path):
    calls = 0

    async def stream(messages, info):
        nonlocal calls
        calls += 1
        if returns(messages):
            assert "Unavailable helper tools" in returns(messages)[-1].content
            yield "Cannot recurse"
        else:
            yield delegate_calls(["Recurse"], ["delegate_task"])

    store, runtime, request, _ = await setup(tmp_path, stream)
    try:
        events = await collect(runtime, request)
        assert events[-1]["event"]["outcome"] == "completed"
        assert await store.list_subruns(request.turn_id) == []
        assert calls == 2
        deps = TurnDeps(
            store,
            request.conversation_id,
            frozenset(),
            None,
            subagent_id="child",
            delegation=runtime.delegation,
        )
        assert "cannot create" in await delegate_task(
            SimpleNamespace(deps=deps), "Recurse"
        )
    finally:
        await runtime.close()
        await store.close()


async def test_child_approvals_share_ledger_with_namespaced_ids(tmp_path):
    async def stream(messages, info):
        prompt = child_prompt(messages)
        if prompt:
            if returns(messages):
                yield "Helper action handled"
            else:
                name = "one.txt" if "Write one" in prompt else "two.txt"
                yield {
                    0: DeltaToolCall(
                        name="write_file",
                        json_args=json.dumps(
                            {"path": name, "content": "approved content"}
                        ),
                        tool_call_id="same-local-call",
                    )
                }
        elif returns(messages):
            yield "Both helpers reported"
        else:
            yield delegate_calls(["Write one", "Write two"], ["write_file"])

    store, runtime, request, workspace = await setup(tmp_path, stream)
    request = request.model_copy(update={"approvals": "always"})
    try:
        await runtime.submit(request)
        events = []
        permissions = []
        async for event in runtime.events(request.turn_id):
            if not event:
                continue
            events.append(event)
            data = event["event"]
            if data["type"] == "permission.request":
                permissions.append(data)
                decision = "allow" if "one.txt" in data["summary"] else "deny"
                await runtime.actions.respond(data["op_id"], data["revision"], decision)
        assert events[-1]["event"]["outcome"] == "completed"
        assert len(permissions) == 2
        assert len({p["subagent_id"] for p in permissions}) == 2
        for permission in permissions:
            operation = await store.get_operation(permission["op_id"])
            assert operation["turn_id"] == request.turn_id
            assert (
                operation["call_id"] == permission["subagent_id"] + ":same-local-call"
            )
        assert (workspace / "one.txt").read_text() == "approved content"
        assert not (workspace / "two.txt").exists()
        assert not runtime.actions.prepared
    finally:
        await runtime.close()
        await store.close()


async def test_stop_cancels_running_and_queued_children(tmp_path):
    entered = asyncio.Event()
    active = finished = 0

    async def stream(messages, info):
        nonlocal active, finished
        if child_prompt(messages):
            active += 1
            if active == 3:
                entered.set()
            try:
                await asyncio.Event().wait()
                yield "Unreachable"
            finally:
                finished += 1
                active -= 1
        else:
            yield delegate_calls([f"Task {i}" for i in range(4)], [])

    store, runtime, request, _ = await setup(tmp_path, stream)
    try:
        await runtime.submit(request)
        await asyncio.wait_for(entered.wait(), 3)
        await runtime.cancel(request.turn_id)
        events = await store.replay_events(request.turn_id)
        assert events[-1]["event"]["outcome"] == "cancelled"
        rows = await store.list_subruns(request.turn_id)
        assert len(rows) == 4 and all(r["status"] == "cancelled" for r in rows)
        assert active == 0 and finished == 3
        assert not runtime.delegation._runs
    finally:
        await runtime.close()
        await store.close()


@pytest.mark.parametrize(
    "limits,status",
    [
        (ChildLimits(requests=1), "budget_exhausted"),
        (ChildLimits(tool_calls=0), "budget_exhausted"),
        (ChildLimits(tokens=1), "budget_exhausted"),
        (ChildLimits(timeout=0.04), "timeout"),
    ],
)
async def test_child_limit_returns_nonempty_summary(tmp_path, limits, status):
    async def stream(messages, info):
        if child_prompt(messages):
            if status == "timeout":
                await asyncio.sleep(5)
                yield "Too late"
            else:
                yield {
                    0: DeltaToolCall(
                        name="read_file",
                        json_args='{"path":"note.txt"}',
                        tool_call_id=f"read-{len(messages)}",
                    )
                }
        elif returns(messages):
            result = json.loads(returns(messages)[-1].content)
            assert result["status"] == status
            assert result["summary"].startswith("Partial result:")
            yield "Partial helper result reported"
        else:
            yield delegate_calls(["Read fact"], ["read_file"])

    store, runtime, request, _ = await setup(tmp_path, stream)
    runtime.delegation.limits = limits
    try:
        events = await collect(runtime, request)
        assert events[-1]["event"]["outcome"] == "completed"
        child = (await store.list_subruns(request.turn_id))[0]
        assert child["status"] == status and child["summary"]
    finally:
        await runtime.close()
        await store.close()


async def test_child_sources_are_promoted_without_number_collisions(tmp_path):
    class Web:
        async def search(self, query):
            return WebResult(
                content="",
                sources=(
                    SearchResult(
                        title=query,
                        url="https://example.com/" + query,
                        snippet="Verified " + query,
                    ),
                ),
            )

    async def stream(messages, info):
        prompt = child_prompt(messages)
        if prompt:
            if returns(messages):
                yield "Verified evidence [1]"
            else:
                query = "alpha" if "Alpha" in prompt else "beta"
                yield {
                    0: DeltaToolCall(
                        name="web_search",
                        json_args=json.dumps({"query": query}),
                        tool_call_id="search",
                    )
                }
        elif returns(messages):
            results = [json.loads(part.content) for part in returns(messages)]
            assert {item["sources"][0]["source_id"] for item in results} == {"1", "2"}
            assert {item["summary"] for item in results} == {
                "Verified evidence [1]",
                "Verified evidence [2]",
            }
            yield "Combined [1] [2]"
        else:
            yield delegate_calls(["Alpha", "Beta"], ["web_search"])

    store, runtime, request, _ = await setup(tmp_path, stream, web=Web())
    try:
        events = await collect(runtime, request)
        assert events[-1]["event"]["outcome"] == "completed"
        sources = [e["event"] for e in events if e["event"]["type"] == "source"]
        assert len(sources) == 2 and {s["source_id"] for s in sources} == {"1", "2"}
        assert all(s["call_id"].startswith("helper-") for s in sources)
    finally:
        await runtime.close()
        await store.close()


async def test_duplicate_call_id_shares_one_child_and_rejects_changed_task(tmp_path):
    calls = 0

    async def stream(messages, info):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        yield "One helper result"

    store, runtime, request, workspace = await setup(tmp_path, stream)
    await store.begin_turn(request.model_dump())
    deps = TurnDeps(
        store,
        request.conversation_id,
        frozenset(),
        lambda event: runtime._emit(request.turn_id, event),
        workspace=str(workspace),
        turn_id=request.turn_id,
        actions=runtime.actions,
        delegation=runtime.delegation,
    )
    ctx = SimpleNamespace(
        deps=deps,
        tool_call_id="same-call",
        model=runtime.model_override,
        model_settings={"max_tokens": 256},
    )
    try:
        results = await asyncio.gather(
            *(runtime.delegation.run(ctx, "One task", tools=[]) for _ in range(2))
        )
        assert results[0] == results[1]
        assert calls == 1 and len(await store.list_subruns(request.turn_id)) == 1
        assert await runtime.delegation.run(ctx, "One task", tools=[]) == results[0]
        assert "different arguments" in await runtime.delegation.run(
            ctx, "Other task", tools=[]
        )
        assert calls == 1
        assert deps.child_usage["requests"] == 1
    finally:
        await runtime.close()
        await store.finish_turn(request.turn_id, "cancelled")
        await store.close()


async def test_timeout_settles_child_approval_without_side_effect(
    tmp_path, monkeypatch
):
    import rapid_mlx.chat_runtime.delegation as delegation

    timeouts = []

    class AsyncioProxy:
        def __getattr__(self, name):
            return getattr(asyncio, name)

        def timeout(self, delay):
            scope = asyncio.timeout(delay)
            timeouts.append(scope)
            return scope

    monkeypatch.setattr(delegation, "asyncio", AsyncioProxy())

    async def stream(messages, info):
        if child_prompt(messages):
            yield {
                0: DeltaToolCall(
                    name="write_file",
                    json_args='{"path":"unapproved.txt","content":"no"}',
                    tool_call_id="edit",
                )
            }
        elif returns(messages):
            assert json.loads(returns(messages)[-1].content)["status"] == "timeout"
            yield "Approval timed out"
        else:
            yield delegate_calls(["Write file"], ["write_file"])

    store, runtime, request, workspace = await setup(tmp_path, stream)
    runtime.delegation.limits = ChildLimits(timeout=30)
    request = request.model_copy(update={"approvals": "always"})
    try:
        await runtime.submit(request)
        events = []
        async with asyncio.timeout(30):
            async for envelope in runtime.events(request.turn_id):
                if not envelope:
                    continue
                events.append(envelope)
                if envelope["event"]["type"] == "permission.request":
                    permission = envelope["event"]
                    timeouts[-1].reschedule(asyncio.get_running_loop().time())
        assert events[-1]["event"]["outcome"] == "completed"
        assert (await store.get_operation(permission["op_id"]))[
            "state"
        ] == "cancelled_before_effect"
        assert not (workspace / "unapproved.txt").exists()
        assert not runtime.actions.prepared and not runtime.actions.signals
    finally:
        await runtime.close()
        await store.close()


async def test_stop_kills_child_owned_process(tmp_path):
    import os

    code = "import os,time; print(os.getpid(),flush=True); time.sleep(30)"

    async def stream(messages, info):
        if child_prompt(messages):
            yield {
                0: DeltaToolCall(
                    name="execute_code",
                    json_args=json.dumps({"language": "python", "code": code}),
                    tool_call_id="process",
                )
            }
        else:
            yield delegate_calls(["Run a process"], ["execute_code"])

    store, runtime, request, _ = await setup(tmp_path, stream)
    request = request.model_copy(update={"approvals": "always"})
    pid = None
    try:
        await runtime.submit(request)
        async for event in runtime.events(request.turn_id):
            if event and event["event"]["type"] == "permission.request":
                permission = event["event"]
                await runtime.actions.respond(
                    permission["op_id"], permission["revision"], "allow"
                )
                break
        async with asyncio.timeout(5):
            while pid is None:
                child = await store.get_subrun(permission["subagent_id"])
                outputs = [
                    e["event"]["text"]
                    for e in child["events"]
                    if e["event"]["type"] == "tool.output"
                ]
                if outputs:
                    pid = int(outputs[0].strip())
                else:
                    await asyncio.sleep(0.01)
        await runtime.cancel(request.turn_id)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert (await store.get_operation(permission["op_id"]))[
            "state"
        ] == "outcome_unknown"
        assert (await store.get_subrun(permission["subagent_id"]))[
            "status"
        ] == "cancelled"
    finally:
        await runtime.close()
        await store.close()


def test_summary_envelope_including_sources_is_bounded():
    from rapid_mlx.chat_runtime.delegation import DelegationManager

    sources = [
        {
            "source_id": str(i),
            "title": "Source" * 80,
            "url": "https://example.com/" + "a" * 600,
        }
        for i in range(20)
    ]
    result = DelegationManager._result(
        "child", "completed", ('Quote "\\\n é [1] ' * 1000), sources, 4000
    )
    assert len(result.encode()) <= 4000
    data = json.loads(result)
    assert data["summary"] and data["sources"] and data["source_note"]


async def test_failed_child_preserves_latest_durable_snapshot(tmp_path, monkeypatch):
    async def stream(messages, info):
        if returns(messages):
            yield "done"
        else:
            yield delegate_calls(["Read fact"], ["read_file"])

    store, runtime, request, _ = await setup(tmp_path, stream)
    original = runtime._attempt
    saved = [{"role": "assistant", "parts": [{"type": "text", "text": "incremental"}]}]

    async def attempt(agent, model, history, deps, settings, number):
        if deps.subagent_id:
            await store.update_subrun(deps.subagent_id, messages=saved)
            raise RuntimeError("interrupted")
        return await original(agent, model, history, deps, settings, number)

    monkeypatch.setattr(runtime, "_attempt", attempt)
    try:
        await collect(runtime, request)
        child = (await store.list_subruns(request.turn_id))[0]
        assert child["status"] == "error"
        assert (await store.get_subrun(child["id"]))["messages"] == saved
    finally:
        await runtime.close()
        await store.close()
