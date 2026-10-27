"""Additional coverage of chat lifecycle and persistence boundaries."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rapid_mlx.chat_runtime.runtime import Runtime
from rapid_mlx.chat_runtime.store import Conflict, Store


async def test_store_boundaries(tmp_path):
    store = await Store.open(tmp_path)
    try:
        c = await store.create_conversation()
        with pytest.raises(KeyError, match="branch"):
            await store.get_conversation(c["id"], branch_id="missing")
        with pytest.raises(ValueError, match="attachment"):
            await store.writer.call(
                lambda db: Store._insert(
                    db,
                    c["id"],
                    None,
                    "user",
                    [{"type": "attachment", "attachment_id": "missing"}],
                )
            )
        with pytest.raises(Conflict, match="stale"):
            await store.writer.call(
                lambda db: Store._advance(db, c["branch_id"], -1, None)
            )
        request = dict(
            turn_id="turn",
            conversation_id=c["id"],
            branch_id=c["branch_id"],
            expected_revision=0,
            text="hello",
        )
        await store.begin_turn(request)
        with pytest.raises(Conflict, match="active turn"):
            await store.begin_turn(
                request | {"turn_id": "other", "expected_revision": 1}
            )
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
        await store.writer.call(
            lambda db: db.execute("UPDATE turns SET status='cancelled' WHERE id='turn'")
        )
        with pytest.raises(Conflict, match="turn ended"):
            await store.decide_operation(op["id"], op["revision"], "allow")
    finally:
        await store.close()


async def test_runtime_closed_compact_and_heartbeat(monkeypatch):
    runtime = Runtime(
        SimpleNamespace(
            replay_events=AsyncMock(return_value=[]),
            get_turn=AsyncMock(return_value={"status": "running"}),
        ),
        web_enabled=False,
    )
    manager = SimpleNamespace(checkpoint_older=AsyncMock(return_value=True))
    runtime._context_manager = AsyncMock(return_value=manager)
    runtime.context_info = AsyncMock(return_value={"tokens": 1})
    assert (await runtime.compact(object(), "short"))["notice"].startswith(
        "Context compacted"
    )
    manager.checkpoint_older.return_value = False
    assert (await runtime.compact(object()))["notice"] == "Nothing to compact yet."

    async def timeout(awaitable, timeout):
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", timeout)
    events = runtime._events("turn", 0, False)
    assert await anext(events) is None
    await events.aclose()
    await runtime.close()
    with pytest.raises(RuntimeError, match="runtime_closed"):
        await runtime.submit(None)
    with pytest.raises(RuntimeError, match="runtime_closed"):
        await runtime._admit(None)


async def test_volatile_reasoning_is_bounded():
    event = {"type": "reasoning.delta", "part_id": "p", "text": "new"}
    store = SimpleNamespace(
        append_event=AsyncMock(
            return_value={"turn_id": "turn", "seq": 4097, "event": event}
        )
    )
    runtime = Runtime(store, web_enabled=False)
    runtime._volatile["turn"] = {i: {} for i in range(4096)}
    await runtime._emit("turn", event)
    assert 0 not in runtime._volatile["turn"]
    assert runtime._volatile["turn"][4097]["event"] == event
    await runtime.close()


@pytest.mark.parametrize(
    "child_result, expected", [("", "error"), ("answer", "completed")]
)
async def test_child_empty_output_synthesis(tmp_path, child_result, expected):
    from tests.test_chat_runtime_delegation import (
        child_prompt,
        collect,
        delegate_calls,
        returns,
        setup,
    )

    async def stream(messages, info):
        if child_prompt(messages):
            yield "child"
        elif returns(messages):
            yield "parent"
        else:
            yield delegate_calls(["task"], [])

    store, runtime, request, _ = await setup(tmp_path, stream)
    original = runtime._attempt
    calls = []

    async def attempt(agent, model, history, deps, settings, attempt_no):
        if deps.subagent_id:
            calls.append(attempt_no)
            return ("" if attempt_no == 0 else child_result), [], deps.run_usage
        return await original(agent, model, history, deps, settings, attempt_no)

    runtime._attempt = attempt
    try:
        await collect(runtime, request)
        children = await store.list_subruns(request.turn_id)
        assert children[0]["status"] == expected
        assert calls == [0, 1]
    finally:
        await runtime.close()
        await store.close()


async def test_attempt_deferred_failure():
    from pydantic_ai import DeferredToolRequests

    from rapid_mlx.chat_runtime.runtime import _AttemptError

    runtime = Runtime(SimpleNamespace(), web_enabled=False)
    runtime._segment = AsyncMock(return_value=(DeferredToolRequests(), [], None))
    runtime.actions.deferred_results = AsyncMock(side_effect=ValueError("failure"))
    deps = SimpleNamespace(run_usage=None, snapshot=None, subagent_id=None)
    with pytest.raises(_AttemptError):
        await runtime._attempt(None, None, [], deps, {}, 0)
    deps.subagent_id = "child"
    with pytest.raises(ValueError, match="failure"):
        await runtime._attempt(None, None, [], deps, {}, 0)
    await runtime.close()


def test_history_with_missing_ancestor_is_rejected():
    db = SimpleNamespace(
        execute=lambda *args: SimpleNamespace(
            fetchall=lambda: [{"parent_id": "missing"}]
        )
    )
    with pytest.raises(ValueError, match="depth"):
        Store._path(db, "conversation", "head")


def test_writer_batch_deadline_and_shutdown(monkeypatch):
    import concurrent.futures
    import queue
    from unittest.mock import Mock

    from rapid_mlx.chat_runtime.store import writer as module

    for shutdown in (False, True):
        writer = module.Writer.__new__(module.Writer)
        writer.ready = concurrent.futures.Future()
        writer.queue = queue.Queue()
        result = concurrent.futures.Future()
        writer.queue.put((lambda db: 42, result, False))
        writer.queue.put(None)
        db = Mock(in_transaction=False)
        if not shutdown:
            clock = iter([0, 1])
            monkeypatch.setattr(
                module.time, "monotonic", lambda clock=clock: next(clock)
            )
            writer._run(lambda db=db: db)
            assert result.result() == 42
        else:
            monkeypatch.setattr(module.time, "monotonic", lambda: 0)
            with pytest.raises(RuntimeError, match="closed while batching"):
                writer._run(lambda db=db: db)
        db.close.assert_called_once()


@pytest.mark.parametrize("failure", ["empty", "truncated", "http"])
async def test_segment_terminal_failures(failure):
    from contextlib import asynccontextmanager

    from pydantic_ai.exceptions import IncompleteToolCall, ModelHTTPError
    from pydantic_ai.usage import RunUsage

    from rapid_mlx.chat_runtime.runtime import _TRUNCATED_CALL

    usage = RunUsage()

    class Run:
        result = None

        def __init__(self):
            self.usage = usage

        def new_messages(self):
            return []

        def __aiter__(self):
            async def iterate():
                if failure == "truncated":
                    raise IncompleteToolCall("incomplete")
                if failure == "http":
                    raise ModelHTTPError(503, "model")
                if False:
                    yield None

            return iterate()

    @asynccontextmanager
    async def iterate(**kwargs):
        yield Run()

    runtime = Runtime(SimpleNamespace(), web_enabled=False)
    agent = SimpleNamespace(iter=iterate)
    deps = SimpleNamespace(subagent_id=None, usage_limits=None, snapshot=None)
    if failure == "http":
        with pytest.raises(ModelHTTPError) as exc:
            await runtime._segment(agent, None, [], deps, {}, 0, usage, None)
        assert exc.value.status_code == 503
    else:
        result, messages, _ = await runtime._segment(
            agent, None, [], deps, {}, 0, usage, None
        )
        assert result is _TRUNCATED_CALL if failure == "truncated" else result == ""
        assert messages == []
    await runtime.close()


async def test_segment_calibrates_provider_usage(monkeypatch):
    from contextlib import asynccontextmanager
    from unittest.mock import Mock

    from pydantic_ai.usage import RunUsage

    from rapid_mlx.chat_runtime.runtime import Agent

    usage = RunUsage()
    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=123),
        parts=[],
        tool_calls=[],
        finish_reason=None,
    )

    class Stream:
        def __init__(self):
            self.response = response

        def __aiter__(self):
            async def events():
                if False:
                    yield None

            return events()

    class Node:
        @asynccontextmanager
        async def stream(self, ctx):
            yield Stream()

    class Run:
        ctx = None

        def __init__(self):
            self.usage = usage

        def new_messages(self):
            return []

        def __aiter__(self):
            async def nodes():
                yield Node()

            return nodes()

    @asynccontextmanager
    async def iterate(**kwargs):
        yield Run()

    monkeypatch.setattr(Agent, "is_model_request_node", lambda node: True)
    counter = Mock()
    deps = SimpleNamespace(
        subagent_id=None,
        usage_limits=None,
        snapshot=None,
        result_budget_exhausted=False,
        result_bytes=0,
        web=None,
        read_budget_bytes=16000,
        token_counter=counter,
    )
    runtime = Runtime(SimpleNamespace(), web_enabled=False)
    result, _, _ = await runtime._segment(
        SimpleNamespace(iter=iterate), None, [], deps, {}, 0, usage, None
    )
    assert result == ""
    counter.calibrate.assert_called_once_with(123)
    await runtime.close()


async def test_workspace_and_running_summary(tmp_path):
    store = await Store.open(tmp_path)
    try:
        c = await store.create_conversation()
        request = dict(
            turn_id="turn",
            conversation_id=c["id"],
            branch_id=c["branch_id"],
            expected_revision=0,
            text="hello",
            workspace=str(tmp_path),
        )
        await store.begin_turn(request)
        messages = await store.branch_messages(c["branch_id"])
        with pytest.raises(Conflict, match="stale"):
            await store.save_summary(
                c["branch_id"],
                "summary",
                messages[0]["id"],
                expected_revision=0,
                turn_id="turn",
            )
        branch = await store.save_summary(
            c["branch_id"],
            "summary",
            messages[0]["id"],
            expected_revision=1,
            turn_id="turn",
        )
        assert branch["revision"] == 2
        assert (await store.get_turn("turn"))["revision"] == 2
    finally:
        await store.close()


async def test_regeneration_skips_non_user_history(monkeypatch, tmp_path):
    from pydantic_ai.messages import ModelResponse, TextPart

    from rapid_mlx.chat_runtime import runtime as module
    from tests.test_chat_runtime_delegation import collect, setup

    async def stream(messages, info):
        yield "answer"

    store, runtime, request, _ = await setup(tmp_path, stream)
    try:
        await collect(runtime, request)
        branch = await store.regenerate(request.branch_id, "regen")
        original = module._dated_history
        monkeypatch.setattr(
            module,
            "_dated_history",
            lambda history: (
                original(history) + [ModelResponse(parts=[TextPart("prior")])]
            ),
        )
        runtime.web_tools = SimpleNamespace()
        regenerated = request.model_copy(
            update={
                "turn_id": "again",
                "branch_id": branch["id"],
                "expected_revision": branch["revision"],
                "mode": "regenerate",
                "text": "",
            }
        )
        events = await collect(runtime, regenerated)
        assert events[-1]["event"]["outcome"] == "completed"
    finally:
        await runtime.close()
        await store.close()
