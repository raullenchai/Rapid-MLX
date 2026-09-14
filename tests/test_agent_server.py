# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from vllm_mlx.agent_runtime import (
    AgentModelTurn,
    AgentRunStatus,
    AgentToolCall,
    AgentToolResult,
    ToolRisk,
    ToolSpec,
    resolve_agent_profile,
)
from vllm_mlx.agent_runtime.server import (
    AgentApprovalRequest,
    AgentRunCapacityError,
    AgentRunConflictError,
    AgentRunCreateRequest,
    AgentRunNotFoundError,
    AgentServerService,
    AgentToolExecutionError,
    AgentToolResultRequest,
    AgentToolSelectionError,
    MCPToolRegistry,
    _approval_argument_summary,
    classify_mcp_tool,
)

READ = ToolSpec(
    name="files__read_file",
    description="Read a file",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    risk=ToolRisk.READ_ONLY,
)
SEND = ToolSpec(
    name="mail__send_message",
    description="Send a message",
    parameters={
        "type": "object",
        "properties": {"body": {"type": "string"}},
        "required": ["body"],
        "additionalProperties": False,
    },
    risk=ToolRisk.EXTERNAL_SIDE_EFFECT,
)


class FakeRegistry:
    def __init__(self, tools: Sequence[ToolSpec] = (READ, SEND)) -> None:
        self.tools = list(tools)
        self.calls: list[AgentToolCall] = []

    def list_tools(self) -> Sequence[ToolSpec]:
        return list(self.tools)

    async def execute(self, call: AgentToolCall) -> AgentToolResult:
        self.calls.append(call)
        return AgentToolResult(
            call_id=call.id,
            content=f"result for {call.name}",
            safe_summary="Tool completed.",
        )


class ScriptedDriver:
    def __init__(self, *turns: AgentModelTurn) -> None:
        self.turns = list(turns)
        self.requests: list[tuple[str, list[dict], list[ToolSpec], object]] = []

    async def __call__(self, model, messages, tools, settings):
        self.requests.append((model, messages, list(tools), settings))
        if not self.turns:
            raise AssertionError("unexpected model turn")
        return self.turns.pop(0)


async def wait_for_status(
    service: AgentServerService,
    run_id: str,
    *statuses: AgentRunStatus,
):
    for _ in range(100):
        view = await service.get(run_id)
        if view.status in statuses:
            return view
        await asyncio.sleep(0)
    raise AssertionError(f"run did not reach {statuses!r}")


@pytest.mark.asyncio
async def test_direct_answer_completes_without_tools_and_keeps_output_out_of_events():
    driver = ScriptedDriver(AgentModelTurn(content="Done."))
    service = AgentServerService(registry=FakeRegistry(()), chat_driver=driver)

    created = await service.create(
        AgentRunCreateRequest(goal="private goal"), model="minicpm5-2b-4bit"
    )
    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    assert done.profile == "minicpm5-2b"
    assert done.output == "Done."
    assert done.pending_action is None
    wire = (await service.events(done.id)).model_dump_json()
    assert "private goal" not in wire
    assert "Done." not in wire


@pytest.mark.asyncio
async def test_public_run_model_never_exposes_profile_filesystem_path():
    service = AgentServerService(
        registry=FakeRegistry(()),
        chat_driver=ScriptedDriver(AgentModelTurn(content="Done.")),
    )

    created = await service.create(
        AgentRunCreateRequest(goal="x", model="friendly-name"),
        model="/Users/private/models/minicpm",
        request_model="friendly-name",
    )
    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    assert done.model == "friendly-name"
    assert "/Users/private" not in done.model_dump_json()


@pytest.mark.asyncio
async def test_server_mode_executes_read_only_tool_and_attaches_transient_ledger():
    call = AgentToolCall(
        id="call-read", name=READ.name, arguments={"path": "private.txt"}
    )
    driver = ScriptedDriver(
        AgentModelTurn(tool_calls=[call]),
        AgentModelTurn(content="The file was inspected."),
    )
    registry = FakeRegistry((READ,))
    service = AgentServerService(registry=registry, chat_driver=driver)

    created = await service.create(
        AgentRunCreateRequest(goal="Inspect private.txt"), model="minicpm5-2b-4bit"
    )
    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    assert len(registry.calls) == 1
    assert registry.calls[0].name == call.name
    assert registry.calls[0].arguments == call.arguments
    assert registry.calls[0].id != call.id
    assert done.tool_rounds == 1
    second_history = driver.requests[1][1]
    assert second_history[-1]["role"] == "tool"
    assert "[Rapid task state]" in second_history[-1]["content"]
    assert "Inspect private.txt" in second_history[-1]["content"]
    events_wire = (await service.events(done.id)).model_dump_json()
    assert "private.txt" not in events_wire


@pytest.mark.asyncio
async def test_side_effect_waits_for_exact_approval_before_server_execution():
    call = AgentToolCall(
        id="call-send", name=SEND.name, arguments={"body": "private message"}
    )
    driver = ScriptedDriver(
        AgentModelTurn(tool_calls=[call]), AgentModelTurn(content="Sent.")
    )
    registry = FakeRegistry((SEND,))
    service = AgentServerService(registry=registry, chat_driver=driver)

    created = await service.create(
        AgentRunCreateRequest(goal="Send it"), model="minicpm5-2b-4bit"
    )
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_APPROVAL
    )

    assert registry.calls == []
    assert waiting.pending_action is not None
    assert waiting.pending_action.arguments == {}
    assert waiting.pending_action.approval_summary == {"body": "private message"}
    assert waiting.pending_action.approval_required is True
    assert "private message" not in (await service.events(created.id)).model_dump_json()

    with pytest.raises(AgentRunConflictError, match="does not match"):
        await service.approve(
            created.id, AgentApprovalRequest(call_id="wrong", approved=True)
        )
    opaque_id = waiting.pending_action.call_id
    approved = await service.approve(
        created.id, AgentApprovalRequest(call_id=opaque_id, approved=True)
    )
    assert approved.pending_action is not None
    assert approved.pending_action.arguments == {}
    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    assert len(registry.calls) == 1
    assert registry.calls[0].id == opaque_id
    assert registry.calls[0].arguments == call.arguments
    assert done.output == "Sent."


@pytest.mark.asyncio
async def test_denial_becomes_tool_observation_and_does_not_execute():
    call = AgentToolCall(id="call-send", name=SEND.name, arguments={"body": "no"})
    driver = ScriptedDriver(
        AgentModelTurn(tool_calls=[call]), AgentModelTurn(content="Not sent.")
    )
    registry = FakeRegistry((SEND,))
    service = AgentServerService(registry=registry, chat_driver=driver)
    created = await service.create(
        AgentRunCreateRequest(goal="Maybe send"), model="minicpm5-2b-4bit"
    )
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_APPROVAL
    )
    assert service._entry(created.id).task is None
    assert waiting.pending_action is not None
    assert waiting.pending_action.arguments == {}
    assert waiting.pending_action.approval_summary == {"body": "no"}

    await service.approve(
        created.id,
        AgentApprovalRequest(call_id=waiting.pending_action.call_id, approved=False),
    )
    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    assert registry.calls == []
    assert done.output == "Not sent."
    assert "user denied" in driver.requests[1][1][-1]["content"].casefold()


@pytest.mark.asyncio
async def test_client_mode_releases_call_then_accepts_one_matching_result():
    call = AgentToolCall(id="call-read", name=READ.name, arguments={"path": "x"})
    driver = ScriptedDriver(
        AgentModelTurn(tool_calls=[call]), AgentModelTurn(content="Client result used.")
    )
    registry = FakeRegistry((READ,))
    service = AgentServerService(registry=registry, chat_driver=driver)
    created = await service.create(
        AgentRunCreateRequest(goal="Read x", execution="client"),
        model="minicpm5-2b-4bit",
    )
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_TOOL_RESULT
    )
    assert service._entry(created.id).task is None

    assert waiting.pending_action is not None
    assert waiting.pending_action.approval_required is False
    assert registry.calls == []
    with pytest.raises(AgentRunConflictError, match="does not match"):
        await service.submit_result(
            created.id,
            AgentToolResultRequest(call_id="wrong", content="result", executed=False),
        )

    await service.submit_result(
        created.id,
        AgentToolResultRequest(
            call_id=waiting.pending_action.call_id,
            content="client secret result",
            executed=False,
        ),
    )
    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    assert done.output == "Client result used."
    tool_message = next(
        message
        for message in service._entry(done.id).messages
        if message["role"] == "tool"
    )
    assert tool_message["content"].startswith("Client tool was not executed.")
    assert "client secret result" not in tool_message["content"]
    event_json = (await service.events(done.id)).model_dump_json()
    assert "client secret result" not in event_json
    assert "Client tool was not executed." in event_json


@pytest.mark.asyncio
async def test_client_side_effect_requires_approval_before_result():
    call = AgentToolCall(id="call-send", name=SEND.name, arguments={"body": "x"})
    driver = ScriptedDriver(
        AgentModelTurn(tool_calls=[call]), AgentModelTurn(content="Client sent it.")
    )
    service = AgentServerService(registry=FakeRegistry((SEND,)), chat_driver=driver)
    created = await service.create(
        AgentRunCreateRequest(goal="Send", execution="client"),
        model="minicpm5-2b-4bit",
    )
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_APPROVAL
    )
    assert waiting.pending_action is not None
    assert waiting.pending_action.arguments == {}
    assert waiting.pending_action.approval_summary == {"body": "x"}
    opaque_id = waiting.pending_action.call_id

    with pytest.raises(AgentRunConflictError, match="awaiting_tool_result"):
        await service.submit_result(
            created.id,
            AgentToolResultRequest(call_id=opaque_id, content="forged", executed=False),
        )
    approved = await service.approve(
        created.id, AgentApprovalRequest(call_id=opaque_id, approved=True)
    )
    assert approved.status is AgentRunStatus.AWAITING_TOOL_RESULT
    assert approved.pending_action is not None
    assert approved.pending_action.arguments == {"body": "x"}
    assert approved.pending_action.risk is ToolRisk.EXTERNAL_SIDE_EFFECT
    await service.submit_result(
        created.id,
        AgentToolResultRequest(call_id=opaque_id, content="sent", executed=True),
    )

    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)
    assert done.output == "Client sent it."


@pytest.mark.asyncio
async def test_cancel_after_client_action_release_records_unknown_outcome():
    call = AgentToolCall(id="call-send", name=SEND.name, arguments={"body": "x"})
    service = AgentServerService(
        registry=FakeRegistry((SEND,)),
        chat_driver=ScriptedDriver(AgentModelTurn(tool_calls=[call])),
    )
    created = await service.create(
        AgentRunCreateRequest(goal="Send", execution="client"), model="model"
    )
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_APPROVAL
    )
    approved = await service.approve(
        created.id,
        AgentApprovalRequest(call_id=waiting.pending_action.call_id, approved=True),
    )
    assert approved.pending_action is not None
    assert approved.pending_action.arguments == {"body": "x"}

    cancelled = await service.cancel(created.id)

    assert cancelled.status is AgentRunStatus.CANCELLED
    assert cancelled.pending_action is None
    events = (await service.events(created.id)).events
    assert [event.type for event in events[-2:]] == [
        "tool.completed",
        "run.cancelled",
    ]
    assert events[-2].data["result"]["executed"] is None
    assert "unknown" in events[-2].data["result"]["safe_summary"].casefold()


@pytest.mark.asyncio
async def test_close_after_client_action_release_records_unknown_outcome():
    call = AgentToolCall(id="call-read", name=READ.name, arguments={"path": "x"})
    service = AgentServerService(
        registry=FakeRegistry((READ,)),
        chat_driver=ScriptedDriver(AgentModelTurn(tool_calls=[call])),
    )
    created = await service.create(
        AgentRunCreateRequest(goal="Read", execution="client"), model="model"
    )
    await wait_for_status(service, created.id, AgentRunStatus.AWAITING_TOOL_RESULT)

    await service.close()

    closed = await service.get(created.id)
    assert closed.status is AgentRunStatus.CANCELLED
    events = (await service.events(created.id)).events
    assert [event.type for event in events[-2:]] == [
        "tool.completed",
        "run.cancelled",
    ]
    assert events[-2].data["result"]["executed"] is None


@pytest.mark.asyncio
async def test_server_mode_rejects_client_result_injection():
    blocker = asyncio.Event()

    async def blocked_driver(*_args):
        await blocker.wait()
        return AgentModelTurn(content="done")

    service = AgentServerService(registry=FakeRegistry(()), chat_driver=blocked_driver)
    created = await service.create(AgentRunCreateRequest(goal="Wait"), model="model")

    with pytest.raises(AgentRunConflictError, match="do not accept"):
        await service.submit_result(
            created.id,
            AgentToolResultRequest(
                call_id="invented", content="inject", executed=False
            ),
        )
    await service.cancel(created.id)


@pytest.mark.asyncio
async def test_capacity_never_evicts_an_active_run():
    blocker = asyncio.Event()

    async def blocked_driver(*_args):
        await blocker.wait()
        return AgentModelTurn(content="done")

    service = AgentServerService(
        registry=FakeRegistry(()), chat_driver=blocked_driver, max_runs=1
    )
    first = await service.create(AgentRunCreateRequest(goal="First"), model="model")

    with pytest.raises(AgentRunCapacityError, match="all agent run slots are active"):
        await service.create(AgentRunCreateRequest(goal="Second"), model="model")
    assert (await service.get(first.id)).id == first.id
    await service.cancel(first.id)


@pytest.mark.asyncio
async def test_full_or_closed_store_rejects_before_registry_snapshot():
    blocker = asyncio.Event()

    async def blocked_driver(*_args):
        await blocker.wait()
        return AgentModelTurn(content="done")

    class SnapshotRegistry(FakeRegistry):
        snapshots = 0

        def snapshot(self):
            self.snapshots += 1
            return self

    registry = SnapshotRegistry(())
    service = AgentServerService(
        registry=registry, chat_driver=blocked_driver, max_runs=1
    )
    first = await service.create(AgentRunCreateRequest(goal="First"), model="model")
    assert registry.snapshots == 1

    with pytest.raises(AgentRunCapacityError, match="slots are active"):
        await service.create(AgentRunCreateRequest(goal="Second"), model="model")
    assert registry.snapshots == 1

    await service.cancel(first.id)
    await service.close()
    with pytest.raises(AgentRunCapacityError, match="shutting down"):
        await service.create(AgentRunCreateRequest(goal="Third"), model="model")
    assert registry.snapshots == 1


@pytest.mark.asyncio
async def test_old_terminal_run_is_evicted_to_make_room():
    driver = ScriptedDriver(
        AgentModelTurn(content="one"), AgentModelTurn(content="two")
    )
    service = AgentServerService(
        registry=FakeRegistry(()), chat_driver=driver, max_runs=1
    )
    first = await service.create(AgentRunCreateRequest(goal="First"), model="model")
    await wait_for_status(service, first.id, AgentRunStatus.COMPLETED)

    second = await service.create(AgentRunCreateRequest(goal="Second"), model="model")

    with pytest.raises(AgentRunNotFoundError):
        await service.get(first.id)
    assert (await service.get(second.id)).id == second.id
    await service.close()


@pytest.mark.asyncio
async def test_terminal_ttl_expires_without_a_background_reaper():
    now = [10.0]
    service = AgentServerService(
        registry=FakeRegistry(()),
        chat_driver=ScriptedDriver(AgentModelTurn(content="done")),
        terminal_ttl_seconds=5,
        monotonic=lambda: now[0],
    )
    created = await service.create(AgentRunCreateRequest(goal="Task"), model="model")
    await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)
    now[0] = 15.0

    with pytest.raises(AgentRunNotFoundError, match="expired"):
        await service.get(created.id)


@pytest.mark.asyncio
async def test_adapter_exception_fails_closed_without_leaking_message():
    async def broken_driver(*_args):
        raise RuntimeError("secret prompt and /private/path")

    service = AgentServerService(registry=FakeRegistry(()), chat_driver=broken_driver)
    created = await service.create(
        AgentRunCreateRequest(goal="secret goal"), model="model"
    )
    failed = await wait_for_status(service, created.id, AgentRunStatus.FAILED)

    assert failed.failure_code == "agent_adapter_failure"
    wire = (await service.events(created.id)).model_dump_json()
    assert "secret" not in wire
    assert "/private/path" not in wire


@pytest.mark.asyncio
async def test_cancel_aborts_inflight_generation_and_close_is_idempotent():
    started = asyncio.Event()

    async def blocked_driver(*_args):
        started.set()
        await asyncio.Future()

    service = AgentServerService(registry=FakeRegistry(()), chat_driver=blocked_driver)
    created = await service.create(AgentRunCreateRequest(goal="Wait"), model="model")
    await started.wait()

    cancelled = await service.cancel(created.id)
    cancelled_again = await service.cancel(created.id)
    await service.close()
    await service.close()

    assert cancelled.status is AgentRunStatus.CANCELLED
    assert cancelled_again.status is AgentRunStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_returns_existing_terminal_outcome_without_500():
    service = AgentServerService(
        registry=FakeRegistry(()),
        chat_driver=ScriptedDriver(AgentModelTurn(content="done")),
    )
    created = await service.create(AgentRunCreateRequest(goal="finish"), model="model")
    completed = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    after_cancel = await service.cancel(created.id)

    assert after_cancel.status is AgentRunStatus.COMPLETED
    assert after_cancel.output == completed.output == "done"

    async def broken_driver(*_args):
        raise RuntimeError("failure")

    failed_service = AgentServerService(
        registry=FakeRegistry(()), chat_driver=broken_driver
    )
    failed_created = await failed_service.create(
        AgentRunCreateRequest(goal="fail"), model="model"
    )
    failed = await wait_for_status(
        failed_service, failed_created.id, AgentRunStatus.FAILED
    )
    failed_after_cancel = await failed_service.cancel(failed.id)
    assert failed_after_cancel.status is AgentRunStatus.FAILED

    cancelled_task = asyncio.create_task(asyncio.sleep(60))
    failed_entry = failed_service._entry(failed.id)
    failed_entry.task = cancelled_task
    failed_entry.tool_in_flight = True
    cancelled_task.cancel()
    still_failed = await failed_service.cancel(failed.id)
    assert still_failed.status is AgentRunStatus.FAILED


@pytest.mark.asyncio
async def test_cancel_racing_approval_never_schedules_side_effect():
    call = AgentToolCall(id="call-send", name=SEND.name, arguments={"body": "x"})
    driver = ScriptedDriver(AgentModelTurn(tool_calls=[call]))
    registry = FakeRegistry((SEND,))
    service = AgentServerService(registry=registry, chat_driver=driver)
    created = await service.create(
        AgentRunCreateRequest(goal="Send"), model="minicpm5-2b-4bit"
    )
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_APPROVAL
    )
    assert waiting.pending_action is not None
    entry = service._entry(created.id)

    await entry.lock.acquire()
    approval = asyncio.create_task(
        service.approve(
            created.id,
            AgentApprovalRequest(call_id=waiting.pending_action.call_id, approved=True),
        )
    )
    await asyncio.sleep(0)
    cancellation = asyncio.create_task(service.cancel(created.id))
    await asyncio.sleep(0)
    entry.lock.release()

    with pytest.raises(AgentRunConflictError, match="cancellation"):
        await approval
    cancelled = await cancellation
    await asyncio.sleep(0)

    assert cancelled.status is AgentRunStatus.CANCELLED
    assert registry.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_approval_maps_missing_runtime_payload(monkeypatch, approved):
    call = AgentToolCall(id="call-send", name=SEND.name, arguments={"body": "x"})
    service = AgentServerService(
        registry=FakeRegistry((SEND,)),
        chat_driver=ScriptedDriver(AgentModelTurn(tool_calls=[call])),
    )
    created = await service.create(AgentRunCreateRequest(goal="send"), model="model")
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_APPROVAL
    )
    monkeypatch.setattr(
        service._runtime, "resolve_approval", lambda *_args, **_kw: None
    )

    expected = "approved action payload" if approved else "denial observation"
    with pytest.raises(AgentRunConflictError, match=expected):
        await service.approve(
            created.id,
            AgentApprovalRequest(
                call_id=waiting.pending_action.call_id, approved=approved
            ),
        )


@pytest.mark.asyncio
async def test_client_result_rejects_cancellation_in_progress():
    call = AgentToolCall(id="call-read", name=READ.name, arguments={"path": "x"})
    service = AgentServerService(
        registry=FakeRegistry((READ,)),
        chat_driver=ScriptedDriver(AgentModelTurn(tool_calls=[call])),
    )
    created = await service.create(
        AgentRunCreateRequest(goal="read", execution="client"), model="model"
    )
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_TOOL_RESULT
    )
    service._entry(created.id).cancel_requested = True
    with pytest.raises(AgentRunConflictError, match="cancellation"):
        await service.submit_result(
            created.id,
            AgentToolResultRequest(
                call_id=waiting.pending_action.call_id,
                content="result",
                executed=True,
            ),
        )


@pytest.mark.asyncio
async def test_cancel_wins_when_model_driver_swallows_task_cancellation():
    started = asyncio.Event()

    async def stubborn_driver(*_args):
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            return AgentModelTurn(content="must not complete")

    service = AgentServerService(registry=FakeRegistry(()), chat_driver=stubborn_driver)
    created = await service.create(AgentRunCreateRequest(goal="wait"), model="model")
    await started.wait()

    cancelled = await service.cancel(created.id)

    assert cancelled.status is AgentRunStatus.CANCELLED
    assert cancelled.output is None
    assert "run.completed" not in [
        event.type for event in (await service.events(created.id)).events
    ]


@pytest.mark.asyncio
async def test_cancel_keeps_noncooperative_generation_non_terminal(monkeypatch):
    import vllm_mlx.agent_runtime.server as agent_server

    started = asyncio.Event()
    swallowed = asyncio.Event()
    release = asyncio.Event()

    async def stubborn_driver(*_args):
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            swallowed.set()
            await release.wait()
            return AgentModelTurn(content="must not complete")

    monkeypatch.setattr(agent_server, "_CANCEL_JOIN_SECONDS", 0.01)
    service = AgentServerService(registry=FakeRegistry(()), chat_driver=stubborn_driver)
    created = await service.create(AgentRunCreateRequest(goal="wait"), model="model")
    await started.wait()

    with pytest.raises(AgentRunConflictError, match="cancellation deadline"):
        await asyncio.wait_for(service.cancel(created.id), timeout=0.5)

    await swallowed.wait()
    entry = service._entry(created.id)
    assert entry.run.status is AgentRunStatus.AWAITING_MODEL
    assert entry.terminal_mono is None
    assert entry.task is not None and not entry.task.done()

    release.set()
    await entry.task
    cancelled = await service.cancel(created.id)
    assert cancelled.status is AgentRunStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_after_side_effect_dispatch_preserves_outcome_before_cancel():
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowRegistry(FakeRegistry):
        async def execute(self, call):
            self.calls.append(call)
            started.set()
            await release.wait()
            return AgentToolResult(
                call_id=call.id,
                content="remote action completed",
                safe_summary="Tool completed.",
            )

    call = AgentToolCall(id="call-send", name=SEND.name, arguments={"body": "x"})
    registry = SlowRegistry((SEND,))
    service = AgentServerService(
        registry=registry,
        chat_driver=ScriptedDriver(AgentModelTurn(tool_calls=[call])),
    )
    created = await service.create(
        AgentRunCreateRequest(goal="Send"), model="minicpm5-2b-4bit"
    )
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_APPROVAL
    )
    assert waiting.pending_action is not None
    await service.approve(
        created.id,
        AgentApprovalRequest(call_id=waiting.pending_action.call_id, approved=True),
    )
    await started.wait()

    cancellation = asyncio.create_task(service.cancel(created.id))
    await asyncio.sleep(0)
    assert not cancellation.done()
    release.set()
    cancelled = await cancellation

    event_types = [event.type for event in (await service.events(created.id)).events]
    assert cancelled.status is AgentRunStatus.CANCELLED
    assert len(registry.calls) == 1
    assert registry.calls[0].arguments == call.arguments
    assert event_types[-2:] == ["tool.completed", "run.cancelled"]


@pytest.mark.asyncio
async def test_cancel_request_disconnect_does_not_cancel_dispatched_tool():
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowRegistry(FakeRegistry):
        async def execute(self, call):
            self.calls.append(call)
            started.set()
            await release.wait()
            return AgentToolResult(
                call_id=call.id,
                content="committed",
                safe_summary="Tool completed.",
            )

    call = AgentToolCall(id="call-send", name=SEND.name, arguments={"body": "x"})
    registry = SlowRegistry((SEND,))
    service = AgentServerService(
        registry=registry,
        chat_driver=ScriptedDriver(AgentModelTurn(tool_calls=[call])),
    )
    created = await service.create(AgentRunCreateRequest(goal="Send"), model="model")
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_APPROVAL
    )
    await service.approve(
        created.id,
        AgentApprovalRequest(call_id=waiting.pending_action.call_id, approved=True),
    )
    await started.wait()

    disconnected_request = asyncio.create_task(service.cancel(created.id))
    await asyncio.sleep(0)
    disconnected_request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await disconnected_request

    entry = service._entry(created.id)
    assert entry.task is not None and not entry.task.done()
    release.set()
    await entry.task
    cancelled = await service.cancel(created.id)
    assert cancelled.status is AgentRunStatus.CANCELLED
    assert [event.type for event in (await service.events(created.id)).events[-2:]] == [
        "tool.completed",
        "run.cancelled",
    ]


@pytest.mark.asyncio
async def test_cancel_records_unknown_if_dispatched_tool_task_is_cancelled():
    started = asyncio.Event()

    class CancelledRegistry(FakeRegistry):
        async def execute(self, call):
            self.calls.append(call)
            started.set()
            await asyncio.Future()

    call = AgentToolCall(id="call-send", name=SEND.name, arguments={"body": "x"})
    service = AgentServerService(
        registry=CancelledRegistry((SEND,)),
        chat_driver=ScriptedDriver(AgentModelTurn(tool_calls=[call])),
    )
    created = await service.create(AgentRunCreateRequest(goal="Send"), model="model")
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_APPROVAL
    )
    await service.approve(
        created.id,
        AgentApprovalRequest(call_id=waiting.pending_action.call_id, approved=True),
    )
    await started.wait()

    cancellation = asyncio.create_task(service.cancel(created.id))
    await asyncio.sleep(0)
    entry = service._entry(created.id)
    entry.task.cancel()
    cancelled = await cancellation

    assert cancelled.status is AgentRunStatus.CANCELLED
    completed = [
        event
        for event in (await service.events(created.id)).events
        if event.type == "tool.completed"
    ]
    assert completed[-1].data["result"]["executed"] is None


@pytest.mark.asyncio
async def test_cancel_after_dispatched_tool_exception_records_outcome_then_cancel():
    started = asyncio.Event()
    release = asyncio.Event()

    class RaisingRegistry(FakeRegistry):
        async def execute(self, call):
            self.calls.append(call)
            started.set()
            await release.wait()
            raise AgentToolExecutionError(executed=True)

    call = AgentToolCall(id="model-id", name=SEND.name, arguments={"body": "x"})
    registry = RaisingRegistry((SEND,))
    service = AgentServerService(
        registry=registry,
        chat_driver=ScriptedDriver(AgentModelTurn(tool_calls=[call])),
    )
    created = await service.create(AgentRunCreateRequest(goal="Send"), model="model")
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_APPROVAL
    )
    assert waiting.pending_action is not None
    await service.approve(
        created.id,
        AgentApprovalRequest(call_id=waiting.pending_action.call_id, approved=True),
    )
    await started.wait()

    cancellation = asyncio.create_task(service.cancel(created.id))
    await asyncio.sleep(0)
    release.set()
    cancelled = await cancellation

    events = (await service.events(created.id)).events
    assert cancelled.status is AgentRunStatus.CANCELLED
    assert [event.type for event in events[-2:]] == [
        "tool.completed",
        "run.cancelled",
    ]
    assert events[-2].data["result"]["executed"] is True
    assert (
        "private transport detail"
        not in (await service.events(created.id)).model_dump_json()
    )


@pytest.mark.asyncio
async def test_untyped_registry_failure_preserves_unknown_execution_state():
    class FailingRegistry(FakeRegistry):
        async def execute(self, _call):
            raise RuntimeError("setup failed")

    service = AgentServerService(
        registry=FailingRegistry((READ,)),
        chat_driver=ScriptedDriver(
            AgentModelTurn(
                tool_calls=[
                    AgentToolCall(
                        id="model-id", name=READ.name, arguments={"path": "x"}
                    )
                ]
            ),
            AgentModelTurn(content="Handled."),
        ),
    )

    created = await service.create(AgentRunCreateRequest(goal="Read"), model="model")
    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)
    completed = [
        event
        for event in (await service.events(done.id)).events
        if event.type == "tool.completed"
    ]

    assert len(completed) == 1
    assert completed[0].data["result"]["executed"] is None
    assert "do not retry automatically" in completed[0].data["result"]["safe_summary"]


def test_tool_selection_is_exact_bounded_and_read_first_by_default():
    tools = [
        ToolSpec(name=f"mutate_{index}", risk=ToolRisk.EXTERNAL_SIDE_EFFECT)
        for index in range(6)
    ] + [READ]
    registry = FakeRegistry(tools)
    service = AgentServerService(registry=registry)
    profile = resolve_agent_profile("minicpm5-2b-4bit")

    selected = service._select_tools(None, profile, registry)
    assert len(selected) == 6
    assert READ in selected

    with pytest.raises(AgentToolSelectionError, match="unknown"):
        service._select_tools(["missing"], profile, registry)
    with pytest.raises(AgentToolSelectionError, match="at most 6"):
        service._select_tools([tool.name for tool in tools], profile, registry)


@pytest.mark.parametrize(
    ("name", "declared", "risk"),
    [
        ("files__read_file", ["files__read_file"], ToolRisk.READ_ONLY),
        ("web__search", [], ToolRisk.EXTERNAL_SIDE_EFFECT),
        ("get_and_delete", [], ToolRisk.EXTERNAL_SIDE_EFFECT),
        ("mail__send_message", ["other__send_message"], ToolRisk.EXTERNAL_SIDE_EFFECT),
    ],
)
def test_mcp_risk_classifier_requires_exact_operator_declaration(name, declared, risk):
    assert classify_mcp_tool(name, declared_read_only=declared) is risk


def test_request_contract_rejects_duplicate_tools_and_non_boolean_controls():
    assert AgentRunCreateRequest(goal="x", tool_names=None).tool_names is None
    assert AgentRunCreateRequest(goal="x", tool_names=["a"]).tool_names == ["a"]
    with pytest.raises(ValidationError, match="1-128"):
        AgentRunCreateRequest(goal="x", tool_names=[""])
    with pytest.raises(ValidationError, match="unique"):
        AgentRunCreateRequest(goal="x", tool_names=["a", "a"])
    with pytest.raises(ValidationError):
        AgentRunCreateRequest(goal="x", enable_thinking=1)
    with pytest.raises(ValidationError):
        AgentApprovalRequest(call_id="call", approved="yes")
    with pytest.raises(ValidationError, match="safe_summary"):
        AgentToolResultRequest(
            call_id="call",
            content="result",
            safe_summary="client-controlled event payload",
        )


def test_approval_summary_preserves_decision_fields_and_redacts_credentials():
    summary = _approval_argument_summary(
        {
            "recipient": "ops@example.com",
            "amount": 42,
            "api_token": "secret-token",
            "apiKey": "secret-api-key",
            "auth": "secret-auth",
            "authentication": "secret-authentication",
            "signing_key": "secret-signing-key",
            "bearer": "secret-bearer",
            "jwt": "secret-jwt",
            "access_key_id": "secret-access-key",
            "headers": {"Bearer secret-token": "present"},
            "keyboard_shortcut": "cmd-k",
            "monkey_patch": True,
            "keynote_id": 7,
            "nested": {
                "password": "secret-password",
                "accessToken": "secret-access-token",
                "clientSecret": "secret-client",
            },
        }
    )

    assert summary == {
        "recipient": "ops@example.com",
        "amount": 42,
        "[redacted-key-3]": "[redacted]",
        "[redacted-key-4]": "[redacted]",
        "[redacted-key-5]": "[redacted]",
        "[redacted-key-6]": "[redacted]",
        "[redacted-key-7]": "[redacted]",
        "[redacted-key-8]": "[redacted]",
        "[redacted-key-9]": "[redacted]",
        "[redacted-key-10]": "[redacted]",
        "headers": {"[redacted-key-1]": "[redacted]"},
        "keyboard_shortcut": "cmd-k",
        "monkey_patch": True,
        "keynote_id": 7,
        "nested": {
            "[redacted-key-1]": "[redacted]",
            "[redacted-key-2]": "[redacted]",
            "[redacted-key-3]": "[redacted]",
        },
    }


def test_approval_summary_bounds_depth_items_keys_and_text():
    nested = {"value": "bottom"}
    for _ in range(10):
        nested = {"next": nested}
    summary = _approval_argument_summary(
        {
            "nested": nested,
            "long": "x" * 1_000,
            "k" * 1_000: "value",
            "items": list(range(100)),
        }
    )

    encoded = json.dumps(summary)
    assert "[truncated]" in encoded
    assert "x" * 257 not in encoded
    assert "k" * 129 not in encoded
    assert len(encoded) < 12_000

    assert _approval_argument_summary("secret", key="api_token") == "[redacted]"
    many_fields = _approval_argument_summary({str(index): index for index in range(40)})
    assert many_fields["[truncated]"] == "additional fields omitted"


def test_minicpm_shape_rejects_missing_metadata():
    from vllm_mlx.agent_runtime.profiles import _is_minicpm5_2b_config

    assert _is_minicpm5_2b_config(None) is False


@pytest.mark.asyncio
async def test_minicpm_profile_enforces_qualified_output_budget():
    driver = ScriptedDriver(AgentModelTurn(content="Done."))
    service = AgentServerService(registry=FakeRegistry(()), chat_driver=driver)

    created = await service.create(
        AgentRunCreateRequest(goal="x", max_tokens=4096),
        model="minicpm5-2b-4bit",
    )
    await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    assert driver.requests[0][3].max_tokens == 900


def test_store_configuration_must_be_bounded():
    with pytest.raises(ValueError, match="max_runs"):
        AgentServerService(max_runs=0)
    with pytest.raises(ValueError, match="terminal_ttl"):
        AgentServerService(terminal_ttl_seconds=-1)


def test_event_cursor_returns_only_new_events():
    async def scenario():
        service = AgentServerService(
            registry=FakeRegistry(()),
            chat_driver=ScriptedDriver(AgentModelTurn(content="done")),
        )
        created = await service.create(AgentRunCreateRequest(goal="x"), model="model")
        done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)
        all_events = await service.events(done.id)
        tail = await service.events(done.id, after=all_events.events[-2].sequence)
        ahead = await service.events(done.id, after=999)
        return all_events, tail, ahead

    all_events, tail, ahead = asyncio.run(scenario())
    assert len(tail.events) == 1
    assert tail.events[0].type == "run.completed"
    assert tail.next_after == all_events.next_after
    assert ahead.events == []
    assert ahead.next_after == all_events.next_after


@pytest.mark.asyncio
async def test_model_authored_call_id_is_replaced_before_history_and_events():
    secret = "copied-private-goal-and-credential"
    driver = ScriptedDriver(
        AgentModelTurn(
            tool_calls=[
                AgentToolCall(id=secret, name=READ.name, arguments={"path": "private"})
            ]
        )
    )
    service = AgentServerService(
        registry=FakeRegistry((READ,)),
        chat_driver=driver,
        call_id_factory=lambda: "call_opaque",
    )

    created = await service.create(
        AgentRunCreateRequest(goal="private", execution="client"), model="model"
    )
    pending = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_TOOL_RESULT
    )

    assert pending.pending_action is not None
    assert pending.pending_action.call_id == "call_opaque"
    assert service._entry(created.id).messages[-1]["tool_calls"][0]["id"] == (
        "call_opaque"
    )
    assert secret not in (await service.events(created.id)).model_dump_json()


@pytest.mark.asyncio
async def test_repeated_model_authored_call_id_fails_before_opaque_replacement():
    repeated = "model-reused-id"
    service = AgentServerService(
        registry=FakeRegistry((READ,)),
        chat_driver=ScriptedDriver(
            AgentModelTurn(
                tool_calls=[
                    AgentToolCall(
                        id=repeated, name=READ.name, arguments={"path": "one"}
                    )
                ]
            ),
            AgentModelTurn(
                tool_calls=[
                    AgentToolCall(
                        id=repeated, name=READ.name, arguments={"path": "two"}
                    )
                ]
            ),
        ),
    )

    created = await service.create(AgentRunCreateRequest(goal="Read"), model="model")
    failed = await wait_for_status(service, created.id, AgentRunStatus.FAILED)

    assert failed.failure_code == "agent_adapter_failure"
    assert repeated not in (await service.events(created.id)).model_dump_json()


@pytest.mark.asyncio
async def test_opaque_call_id_must_be_unique_across_the_run():
    service = AgentServerService(
        registry=FakeRegistry((READ,)),
        chat_driver=ScriptedDriver(
            AgentModelTurn(
                tool_calls=[
                    AgentToolCall(id="first", name=READ.name, arguments={"path": "one"})
                ]
            ),
            AgentModelTurn(
                tool_calls=[
                    AgentToolCall(
                        id="second", name=READ.name, arguments={"path": "two"}
                    )
                ]
            ),
        ),
        call_id_factory=lambda: "call_same",
    )

    created = await service.create(AgentRunCreateRequest(goal="Read"), model="model")
    failed = await wait_for_status(service, created.id, AgentRunStatus.FAILED)

    assert failed.failure_code == "agent_adapter_failure"


@pytest.mark.asyncio
async def test_assistant_history_uses_json_arguments_not_python_repr():
    call = AgentToolCall(id="c", name=READ.name, arguments={"path": "你好"})
    driver = ScriptedDriver(AgentModelTurn(tool_calls=[call]))
    service = AgentServerService(registry=FakeRegistry((READ,)), chat_driver=driver)
    run = await service.create(
        AgentRunCreateRequest(goal="x", execution="client"), model="model"
    )
    await wait_for_status(service, run.id, AgentRunStatus.AWAITING_TOOL_RESULT)
    message = service._entry(run.id).messages[-1]

    encoded = message["tool_calls"][0]["function"]["arguments"]
    assert json.loads(encoded) == {"path": "你好"}


def test_mcp_projection_uses_declared_risk_and_skips_unsupported_schemas():
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config
    from vllm_mlx.mcp.types import MCPTool

    cfg = reset_config()
    cfg.mcp_manager = SimpleNamespace(
        config=SimpleNamespace(agent_read_only_tools=["files__read_file"]),
        get_all_tools=lambda: [
            MCPTool("files", "read_file", "read", {"type": "object"}),
            MCPTool("files", "write_file", "write", {"type": "object"}),
            MCPTool("refs", "search", "bad", {"$ref": "#/$defs/x"}),
            MCPTool("bad.server", "tool", "bad name", {"type": "object"}),
        ],
    )

    tools = list(MCPToolRegistry().list_tools())

    assert [(tool.name, tool.risk) for tool in tools] == [
        ("files__read_file", ToolRisk.READ_ONLY),
        ("files__write_file", ToolRisk.EXTERNAL_SIDE_EFFECT),
    ]
    reset_config()


@pytest.mark.asyncio
async def test_mcp_execution_preserves_sandbox_and_audit():
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config
    from vllm_mlx.mcp.types import MCPToolResult

    audited = []

    class Sandbox:
        def validate_tool_execution(self, *args):
            assert args == ("read_file", "files", {"path": "x"})

        def record_execution(self, *args, **kwargs):
            audited.append((args, kwargs))

    class Manager:
        def resolve_tool_target(self, name):
            assert name == "files__read_file"
            return "files", "read_file"

        async def execute_tool(self, name, arguments):
            return MCPToolResult(name, {"text": "ok"})

    cfg = reset_config()
    cfg.mcp_manager = Manager()
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())
    call = AgentToolCall(id="call", name="files__read_file", arguments={"path": "x"})

    result = await MCPToolRegistry().execute(call)

    assert result.content == '{"text": "ok"}'
    assert result.executed is True
    assert audited[0][1]["success"] is True
    reset_config()


@pytest.mark.asyncio
async def test_mcp_audit_failure_never_rewrites_committed_tool_outcome(caplog):
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config
    from vllm_mlx.mcp.types import MCPToolResult

    class Sandbox:
        def validate_tool_execution(self, *_args):
            return None

        def record_execution(self, *_args, **_kwargs):
            raise OSError("audit sink echoed secret-token-123")

    class Manager:
        def resolve_tool_target(self, _name):
            return "files", "write_file"

        async def execute_tool(self, _name, _arguments):
            return MCPToolResult("files__write_file", "committed")

    cfg = reset_config()
    cfg.mcp_manager = Manager()
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())

    result = await MCPToolRegistry().execute(
        AgentToolCall(
            id="side-effect",
            name="files__write_file",
            arguments={"path": "x"},
        )
    )

    assert result.content == "committed"
    assert result.executed is True
    assert result.is_error is False
    assert result.safe_summary == (
        "Tool execution completed, but its MCP audit record could not be written."
    )
    assert "AUDIT_FALLBACK mcp_execution" in caplog.text
    assert "identity_sha256=" in caplog.text
    assert "secret-token-123" not in caplog.text
    reset_config()


@pytest.mark.asyncio
async def test_mcp_snapshot_never_executes_against_reloaded_registry():
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config
    from vllm_mlx.mcp.types import MCPTool, MCPToolResult

    calls = []

    class Sandbox:
        def validate_tool_execution(self, *_args):
            return None

        def record_execution(self, *_args, **_kwargs):
            return None

    class Manager:
        def __init__(self, generation):
            self.generation = generation
            self.config = SimpleNamespace(
                agent_read_only_tools=[], default_timeout=30.0
            )
            self.tool = MCPTool("same", "tool", "tool", {"type": "object"})
            manager = self

            class Client:
                is_connected = True
                tools = [manager.tool]

                async def call_tool(self, *_args, **_kwargs):
                    calls.append(manager.generation)
                    return MCPToolResult("same__tool", "ok")

            self.client = Client()

        def get_all_tools(self):
            return [self.tool]

        def get_client(self, _name):
            return self.client

        @asynccontextmanager
        async def tool_generation_lease(self):
            yield

    cfg = reset_config()
    advertised = Manager("advertised")
    cfg.mcp_manager = advertised
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())
    snapshot = MCPToolRegistry().snapshot()
    assert [tool.name for tool in snapshot.list_tools()] == ["same__tool"]

    cfg.mcp_manager = Manager("reloaded")
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())
    await snapshot.execute(AgentToolCall(id="call", name="same__tool", arguments={}))

    assert calls == ["advertised"]

    # An in-place refresh replaces the advertised tool identity. The old run
    # must fail closed rather than dispatch by the same mutable name.
    advertised.client.tools = [
        MCPTool("same", "tool", "replacement", {"type": "object"})
    ]
    stale = await snapshot.execute(
        AgentToolCall(id="stale", name="same__tool", arguments={})
    )
    assert stale.executed is False
    assert calls == ["advertised"]
    reset_config()


@pytest.mark.asyncio
async def test_pinned_mcp_dispatch_leases_generation_until_call_finishes():
    from types import SimpleNamespace

    from vllm_mlx.agent_runtime.server import _PinnedMCPManager
    from vllm_mlx.mcp.manager import MCPClientManager
    from vllm_mlx.mcp.types import MCPTool, MCPToolResult

    both_calls_started = asyncio.Event()
    release_call = asyncio.Event()
    events = []
    started_count = 0
    tool = MCPTool("same", "tool", "tool", {"type": "object"})

    class Client:
        is_connected = True
        tools = [tool]

        async def call_tool(self, *_args, **_kwargs):
            nonlocal started_count
            events.append("call-started")
            started_count += 1
            if started_count == 2:
                both_calls_started.set()
            await release_call.wait()
            events.append("call-finished")
            return MCPToolResult("same__tool", "ok")

        async def disconnect(self):
            events.append("disconnect")

        async def connect(self):
            events.append("connect")

    class Manager:
        tool_generation_lease = MCPClientManager.tool_generation_lease
        _generation_condition = MCPClientManager._generation_condition
        _generation_mutation = MCPClientManager._generation_mutation
        reconnect = MCPClientManager.reconnect

        def __init__(self):
            self._lock = asyncio.Lock()
            self._lease_condition = asyncio.Condition(self._lock)
            self._active_tool_leases = 0
            self._generation_mutation_waiters = 0
            self.config = SimpleNamespace(
                agent_read_only_tools=[], default_timeout=30.0
            )
            self._clients = {"same": Client()}

        def get_all_tools(self):
            return [tool]

        def get_client(self, name):
            return self._clients.get(name)

    manager = Manager()
    pinned = _PinnedMCPManager(manager)
    executions = [
        asyncio.create_task(pinned.execute_tool("same__tool", {})) for _ in range(2)
    ]
    await asyncio.wait_for(both_calls_started.wait(), timeout=0.5)

    reconnection = asyncio.create_task(manager.reconnect("same"))
    await asyncio.sleep(0)
    assert events == ["call-started", "call-started"]

    release_call.set()
    results = await asyncio.gather(*executions)
    await reconnection

    assert [result.content for result in results] == ["ok", "ok"]
    assert events == [
        "call-started",
        "call-started",
        "call-finished",
        "call-finished",
        "disconnect",
        "connect",
    ]


@pytest.mark.asyncio
async def test_pinned_mcp_targets_fail_closed_on_lookup_errors():
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from vllm_mlx.agent_runtime.server import _PinnedMCPManager
    from vllm_mlx.mcp.types import MCPTool

    tool = MCPTool("same", "tool", "tool", {"type": "object"})

    class Client:
        is_connected = True
        tools = [tool]

    class Manager:
        config = SimpleNamespace(agent_read_only_tools=[], default_timeout=30.0)
        client = Client()

        def get_all_tools(self):
            return [tool]

        def get_client(self, _name):
            return self.client

        @asynccontextmanager
        async def tool_generation_lease(self):
            yield

    manager = Manager()
    pinned = _PinnedMCPManager(manager)
    assert pinned.resolve_tool_target("missing") == (None, "missing")
    assert pinned.get_client("missing") is None
    with pytest.raises(AgentToolExecutionError):
        await pinned.execute_tool("missing", {})

    def broken_lookup(_name):
        raise RuntimeError("lookup failed")

    manager.get_client = broken_lookup
    assert pinned.resolve_tool_target("same__tool") == (None, "same__tool")


@pytest.mark.asyncio
async def test_pinned_mcp_untyped_transport_failure_stays_unknown():
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from vllm_mlx.agent_runtime.server import _PinnedMCPManager
    from vllm_mlx.mcp.types import MCPTool

    tool = MCPTool("same", "tool", "tool", {"type": "object"})

    class Client:
        is_connected = True
        tools = [tool]
        error: BaseException = RuntimeError("private post-dispatch failure")

        async def call_tool(self, *_args, **_kwargs):
            raise self.error

    class Manager:
        config = SimpleNamespace(agent_read_only_tools=[], default_timeout=30.0)
        client = Client()

        def get_all_tools(self):
            return [tool]

        def get_client(self, _name):
            return self.client

        @asynccontextmanager
        async def tool_generation_lease(self):
            yield

    manager = Manager()
    pinned = _PinnedMCPManager(manager)
    with pytest.raises(RuntimeError, match="private post-dispatch failure"):
        await pinned.execute_tool("same__tool", {})

    manager.client.error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await pinned.execute_tool("same__tool", {})

    manager.client.error = AgentToolExecutionError(executed=False)
    with pytest.raises(AgentToolExecutionError) as typed:
        await pinned.execute_tool("same__tool", {})
    assert typed.value.executed is False


def test_mcp_snapshot_and_listing_map_manager_failures():
    from types import SimpleNamespace

    from vllm_mlx.agent_runtime.server import AgentToolRegistryUnavailableError
    from vllm_mlx.config import reset_config

    class BrokenManager:
        config = SimpleNamespace(agent_read_only_tools=[], default_timeout=30.0)

        def get_all_tools(self):
            raise RuntimeError("registry failed")

    cfg = reset_config()
    cfg.mcp_manager = BrokenManager()
    with pytest.raises(AgentToolRegistryUnavailableError):
        MCPToolRegistry().snapshot()

    registry = MCPToolRegistry(manager=BrokenManager(), executor=None, pinned=True)
    with pytest.raises(AgentToolRegistryUnavailableError):
        registry.list_tools()
    reset_config()


@pytest.mark.asyncio
async def test_mcp_bare_and_dispatch_failure_paths_are_audited():
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config

    audited = []

    class Sandbox:
        def validate_tool_execution(self, *_args):
            return None

        def record_execution(self, *args, **kwargs):
            audited.append((args, kwargs))

    class MissingBareManager:
        def resolve_tool_target(self, _name):
            return None, "bare"

    cfg = reset_config()
    cfg.mcp_manager = MissingBareManager()
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())
    bare = await MCPToolRegistry().execute(
        AgentToolCall(id="bare", name="bare", arguments={})
    )
    assert bare.executed is False
    assert audited[-1][0][:2] == ("bare", "unknown")

    class BrokenClientManager:
        def resolve_tool_target(self, _name):
            return "server", "tool"

        def get_client(self, _name):
            raise RuntimeError("client lookup failed")

    cfg.mcp_manager = BrokenClientManager()
    unavailable = await MCPToolRegistry().execute(
        AgentToolCall(id="client", name="server__tool", arguments={})
    )
    assert unavailable.executed is False

    class RejectedDispatchManager:
        def resolve_tool_target(self, _name):
            return "server", "tool"

        async def execute_tool(self, *_args):
            raise AgentToolExecutionError(executed=True)

    cfg.mcp_manager = RejectedDispatchManager()
    rejected = await MCPToolRegistry().execute(
        AgentToolCall(id="dispatch", name="server__tool", arguments={})
    )
    assert rejected.executed is True
    assert rejected.is_error is True
    assert audited[-1][1]["error_message"] == "MCP dispatch rejected"
    reset_config()


@pytest.mark.asyncio
async def test_run_never_switches_to_replacement_model_generation():
    from vllm_mlx.config import reset_config
    from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry
    from vllm_mlx.service.helpers import get_engine

    first_engine = object()
    replacement_engine = object()
    first = ModelEntry(
        engine=first_engine,
        model_name="canonical",
        model_path="/models/first",
        aliases={"served"},
    )
    registry = ModelRegistry()
    registry.add(first, is_default=True)
    cfg = reset_config()
    cfg.model_registry = registry
    seen_engines = []

    async def driver(model, _messages, _tools, _settings):
        seen_engines.append(get_engine(model))
        if len(seen_engines) == 1:
            return AgentModelTurn(
                tool_calls=[
                    AgentToolCall(id="call", name=READ.name, arguments={"path": "x"})
                ]
            )
        return AgentModelTurn(content="done")

    service = AgentServerService(registry=FakeRegistry((READ,)), chat_driver=driver)
    created = await service.create(
        AgentRunCreateRequest(goal="read", execution="client"),
        model="canonical",
        request_model="served",
        model_generation=first,
    )
    waiting = await wait_for_status(
        service, created.id, AgentRunStatus.AWAITING_TOOL_RESULT
    )
    assert waiting.pending_action is not None

    registry.remove("canonical")
    registry.add(
        ModelEntry(
            engine=replacement_engine,
            model_name="canonical",
            model_path="/models/replacement",
            aliases={"served"},
        ),
        is_default=True,
    )
    await service.submit_result(
        created.id,
        AgentToolResultRequest(
            call_id=waiting.pending_action.call_id,
            content="result",
            executed=True,
        ),
    )
    failed = await wait_for_status(service, created.id, AgentRunStatus.FAILED)

    assert failed.failure_code == "agent_adapter_failure"
    assert seen_engines == [first_engine]
    reset_config()


def test_exact_model_generation_access_and_single_engine_binding():
    from fastapi import HTTPException

    from vllm_mlx.config import reset_config
    from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry
    from vllm_mlx.service.helpers import bind_model_generation, get_engine

    accessed = []
    engine = object()
    entry = ModelEntry(engine=engine, model_name="model", model_path="/model")
    registry = ModelRegistry()
    registry.on_engine_access = accessed.append
    registry.add(entry, is_default=True)
    assert registry.get_engine_if_entry(entry) is engine
    assert accessed == ["model"]

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_registry = None
    with bind_model_generation(engine):
        assert get_engine("model") is engine
    with bind_model_generation(object()), pytest.raises(HTTPException) as raised:
        get_engine("model")
    assert raised.value.status_code == 503
    reset_config()


@pytest.mark.asyncio
async def test_mcp_sandbox_rejection_is_reported_as_unexecuted():
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config
    from vllm_mlx.mcp.security import MCPSecurityError

    executed = []
    audited = []

    class Sandbox:
        def validate_tool_execution(self, *_args):
            raise MCPSecurityError("secret policy detail")

        def record_execution(self, *args, **kwargs):
            audited.append((args, kwargs))

    class Manager:
        def resolve_tool_target(self, _name):
            return "shell", "execute"

        async def execute_tool(self, *_args):
            executed.append(True)

    cfg = reset_config()
    cfg.mcp_manager = Manager()
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())

    result = await MCPToolRegistry().execute(
        AgentToolCall(id="call", name="shell__execute", arguments={"cmd": "x"})
    )

    assert result.executed is False
    assert result.is_error is True
    assert "secret policy detail" not in result.content
    assert executed == []
    assert audited[0][1]["success"] is False
    assert audited[0][1]["error_message"] == "blocked by server security policy"
    reset_config()


@pytest.mark.asyncio
async def test_mcp_unavailable_and_disappeared_calls_are_unexecuted():
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config

    call = AgentToolCall(id="call", name="files__read_file", arguments={})
    cfg = reset_config()
    assert (await MCPToolRegistry().execute(call)).executed is False

    cfg.mcp_manager = SimpleNamespace(
        resolve_tool_target=lambda _name: (None, "read_file")
    )
    disappeared_audit = []

    class DisappearedSandbox:
        def record_execution(self, *args, **kwargs):
            disappeared_audit.append((args, kwargs))

    cfg.mcp_executor = SimpleNamespace(sandbox=DisappearedSandbox())
    assert (await MCPToolRegistry().execute(call)).executed is False
    assert disappeared_audit[0][1]["error_message"] == "MCP tool unavailable"

    audited = []

    class Sandbox:
        def record_execution(self, *args, **kwargs):
            audited.append((args, kwargs))

    cfg.mcp_manager = SimpleNamespace(
        resolve_tool_target=lambda _name: ("files", "read_file"),
        get_client=lambda _name: SimpleNamespace(is_connected=False),
    )
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())
    unavailable = await MCPToolRegistry().execute(call)
    assert unavailable.executed is False
    assert unavailable.is_error is True
    assert audited[0][1]["error_message"] == "MCP server unavailable"
    reset_config()


@pytest.mark.asyncio
async def test_mcp_registry_lookup_failure_is_audited_and_unexecuted():
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config

    audited = []

    class Sandbox:
        def record_execution(self, *args, **kwargs):
            audited.append((args, kwargs))

    class Manager:
        def resolve_tool_target(self, _name):
            raise ConnectionError("private disconnect detail")

    cfg = reset_config()
    cfg.mcp_manager = Manager()
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())

    result = await MCPToolRegistry().execute(
        AgentToolCall(id="call", name="files__read_file", arguments={})
    )

    assert result.executed is False
    assert result.is_error is True
    assert "private disconnect detail" not in result.content
    assert audited[0][0][:2] == ("read_file", "files")
    assert audited[0][1]["error_message"] == "MCP registry unavailable"
    reset_config()


@pytest.mark.asyncio
async def test_mcp_sandbox_internal_failure_is_unexecuted_and_audited():
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config

    audited = []
    dispatched = []

    class Sandbox:
        def validate_tool_execution(self, *_args):
            raise RuntimeError("private sandbox detail")

        def record_execution(self, *args, **kwargs):
            audited.append((args, kwargs))

    class Manager:
        def resolve_tool_target(self, _name):
            return "files", "read_file"

        async def execute_tool(self, *_args):
            dispatched.append(True)

    cfg = reset_config()
    cfg.mcp_manager = Manager()
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())

    result = await MCPToolRegistry().execute(
        AgentToolCall(id="call", name="files__read_file", arguments={})
    )

    assert result.executed is False
    assert result.is_error is True
    assert "private sandbox detail" not in result.content
    assert dispatched == []
    assert audited[0][1]["error_message"] == "MCP sandbox unavailable"
    reset_config()


@pytest.mark.asyncio
async def test_mcp_result_shapes_and_execution_exception_are_audited():
    from types import SimpleNamespace

    from vllm_mlx.config import reset_config
    from vllm_mlx.mcp.types import MCPToolResult

    audited = []

    class Sandbox:
        def validate_tool_execution(self, *_args):
            return None

        def record_execution(self, *args, **kwargs):
            audited.append((args, kwargs))

    class Manager:
        result = MCPToolResult("tool", "plain")

        def resolve_tool_target(self, _name):
            return "server", "tool"

        async def execute_tool(self, *_args):
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    manager = Manager()
    cfg = reset_config()
    cfg.mcp_manager = manager
    cfg.mcp_executor = SimpleNamespace(sandbox=Sandbox())
    call = AgentToolCall(id="call", name="server__tool", arguments={})

    plain = await MCPToolRegistry().execute(call)
    assert plain.content == "plain"

    manager.result = MCPToolResult(
        "tool", None, is_error=True, error_message="expected failure"
    )
    failed = await MCPToolRegistry().execute(call)
    assert failed.content == "expected failure"
    assert failed.executed is True

    manager.result = MCPToolResult(
        "tool", None, is_error=True, error_message="x" * 300_000
    )
    long_error = await MCPToolRegistry().execute(call)
    assert long_error.executed is True
    assert long_error.is_error is True
    assert long_error.content.endswith("[tool result truncated by Rapid]")

    manager.result = MCPToolResult("tool", "x" * 250_000)
    truncated = await MCPToolRegistry().execute(call)
    assert truncated.content.endswith("[tool result truncated by Rapid]")

    class Unserializable:
        def __str__(self):
            raise RuntimeError("private serialization detail")

    manager.result = MCPToolResult("tool", Unserializable())
    serialization_failure = await MCPToolRegistry().execute(call)
    assert serialization_failure.executed is True
    assert serialization_failure.is_error is True
    assert serialization_failure.safe_summary == (
        "Tool executed, but its result could not be serialized."
    )
    assert "private serialization detail" not in serialization_failure.content

    manager.result = RuntimeError("private exception")
    uncertain = await MCPToolRegistry().execute(call)
    assert uncertain.executed is None
    assert uncertain.is_error is True
    assert "private exception" not in uncertain.content
    assert audited[-1][1]["error_message"] == "RuntimeError"
    reset_config()


@pytest.mark.asyncio
async def test_registry_without_mcp_has_no_tools_and_internal_request_stays_live():
    from types import SimpleNamespace

    from vllm_mlx.agent_runtime.server import _InternalRequest
    from vllm_mlx.config import reset_config

    reset_config()
    registry = MCPToolRegistry()
    assert registry.list_tools() == []
    assert registry.execution_timeout_seconds == 30.0
    configured = MCPToolRegistry(
        manager=SimpleNamespace(config=SimpleNamespace(default_timeout=12.5)),
        pinned=True,
    )
    assert configured.execution_timeout_seconds == 12.5
    assert await _InternalRequest().is_disconnected() is False


@pytest.mark.asyncio
async def test_close_cancels_active_work_and_rejects_new_runs():
    started = asyncio.Event()

    async def blocked_driver(*_args):
        started.set()
        await asyncio.Future()

    service = AgentServerService(registry=FakeRegistry(()), chat_driver=blocked_driver)
    created = await service.create(AgentRunCreateRequest(goal="wait"), model="model")
    await started.wait()

    await service.close()

    assert (await service.get(created.id)).status is AgentRunStatus.CANCELLED
    with pytest.raises(AgentRunCapacityError, match="shutting down"):
        await service.create(AgentRunCreateRequest(goal="new"), model="model")


@pytest.mark.asyncio
async def test_close_fails_boundedly_if_generation_does_not_stop(monkeypatch):
    import vllm_mlx.agent_runtime.server as agent_server

    started = asyncio.Event()
    release = asyncio.Event()
    swallowed = asyncio.Event()

    async def cancellation_delaying_driver(*_args):
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                swallowed.set()
        return AgentModelTurn(content="must not complete")

    monkeypatch.setattr(agent_server, "_SHUTDOWN_JOIN_SECONDS", 0.01)
    service = AgentServerService(
        registry=FakeRegistry(()), chat_driver=cancellation_delaying_driver
    )
    created = await service.create(AgentRunCreateRequest(goal="wait"), model="model")
    await started.wait()

    with pytest.raises(AgentRunCapacityError, match="shutdown deadline"):
        await service.close()

    task = service._entry(created.id).task
    assert task is not None
    await asyncio.sleep(0)
    assert swallowed.is_set()
    assert not task.done()
    release.set()
    await task


@pytest.mark.asyncio
async def test_close_fails_boundedly_if_dispatched_tool_exceeds_deadline(monkeypatch):
    import vllm_mlx.agent_runtime.server as agent_server

    started = asyncio.Event()
    release = asyncio.Event()

    class SlowRegistry(FakeRegistry):
        async def execute(self, call):
            self.calls.append(call)
            started.set()
            await release.wait()
            return AgentToolResult(
                call_id=call.id,
                content="committed",
                safe_summary="Tool completed.",
            )

    monkeypatch.setattr(agent_server, "_SHUTDOWN_JOIN_SECONDS", 0.01)
    registry = SlowRegistry((READ,))
    service = AgentServerService(
        registry=registry,
        chat_driver=ScriptedDriver(
            AgentModelTurn(
                tool_calls=[
                    AgentToolCall(
                        id="model-call", name=READ.name, arguments={"path": "x"}
                    )
                ]
            )
        ),
    )
    created = await service.create(AgentRunCreateRequest(goal="read"), model="model")
    await started.wait()

    shutdown = asyncio.create_task(service.close())
    with pytest.raises(AgentRunCapacityError, match="tool work.*shutdown deadline"):
        await asyncio.wait_for(shutdown, timeout=0.5)
    entry = service._entry(created.id)
    assert entry.task is not None and not entry.task.done()
    release.set()
    await entry.task
    await service.close()

    events = (await service.events(created.id)).events
    assert [event.type for event in events[-2:]] == [
        "tool.completed",
        "run.cancelled",
    ]
    assert events[-2].data["result"]["executed"] is True


@pytest.mark.asyncio
async def test_schedule_rejects_parallel_driver_for_same_run():
    blocker = asyncio.Event()

    async def blocked_driver(*_args):
        await blocker.wait()
        return AgentModelTurn(content="done")

    service = AgentServerService(registry=FakeRegistry(()), chat_driver=blocked_driver)
    created = await service.create(AgentRunCreateRequest(goal="wait"), model="model")
    entry = service._entry(created.id)

    with pytest.raises(AgentRunConflictError, match="already has work"):
        service._schedule(entry)
    await service.cancel(created.id)

    entry.cancel_requested = True
    with pytest.raises(AgentRunConflictError, match="cancellation"):
        service._schedule(entry)


@pytest.mark.asyncio
async def test_get_waits_for_atomic_run_snapshot():
    service = AgentServerService(
        registry=FakeRegistry(()),
        chat_driver=ScriptedDriver(AgentModelTurn(content="done")),
    )
    created = await service.create(AgentRunCreateRequest(goal="answer"), model="model")
    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)
    entry = service._entry(done.id)

    async with entry.lock:
        read = asyncio.create_task(service.get(done.id))
        await asyncio.sleep(0)
        assert not read.done()
        entry.pending_action = AgentToolCall(id="stale", name=READ.name, arguments={})
        entry.pending_risk = ToolRisk.READ_ONLY
        entry.pending_action = None
        entry.pending_risk = None

    view = await read
    assert view.status is AgentRunStatus.COMPLETED
    assert view.pending_action is None


@pytest.mark.asyncio
async def test_drive_cancellation_and_stale_state_guards(monkeypatch):
    class CancelledRegistry(FakeRegistry):
        async def execute(self, _call):
            raise asyncio.CancelledError

    call = AgentToolCall(id="call", name=READ.name, arguments={"path": "x"})
    cancelled_service = AgentServerService(
        registry=CancelledRegistry((READ,)),
        chat_driver=ScriptedDriver(AgentModelTurn(tool_calls=[call])),
    )
    created = await cancelled_service.create(
        AgentRunCreateRequest(goal="read"), model="model"
    )
    entry = cancelled_service._entry(created.id)
    await entry.task
    assert entry.tool_in_flight is False
    failed = await cancelled_service.get(created.id)
    assert failed.status is AgentRunStatus.FAILED
    assert failed.failure_code == "agent_adapter_cancelled"
    completed = [
        event
        for event in (await cancelled_service.events(created.id)).events
        if event.type == "tool.completed"
    ]
    assert completed[-1].data["result"]["executed"] is None
    await cancelled_service._drive(entry, call=call)
    entry.cancel_requested = True
    await cancelled_service._drive(entry)

    approval_service = AgentServerService(
        registry=FakeRegistry((SEND,)),
        chat_driver=ScriptedDriver(
            AgentModelTurn(
                tool_calls=[
                    AgentToolCall(
                        id="approval", name=SEND.name, arguments={"body": "x"}
                    )
                ]
            )
        ),
    )
    approval_run = await approval_service.create(
        AgentRunCreateRequest(goal="send"), model="model"
    )
    await wait_for_status(
        approval_service, approval_run.id, AgentRunStatus.AWAITING_APPROVAL
    )
    await approval_service._drive(approval_service._entry(approval_run.id))

    invalid_service = AgentServerService(
        registry=FakeRegistry(()),
        chat_driver=ScriptedDriver(AgentModelTurn(content="answer")),
    )
    monkeypatch.setattr(
        invalid_service._runtime, "accept_model_turn", lambda *_args: None
    )
    invalid = await invalid_service.create(
        AgentRunCreateRequest(goal="answer"), model="model"
    )
    failed = await wait_for_status(invalid_service, invalid.id, AgentRunStatus.FAILED)
    assert failed.failure_code == "agent_adapter_failure"


@pytest.mark.asyncio
async def test_drive_discards_result_if_run_became_terminal():
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowRegistry(FakeRegistry):
        async def execute(self, call):
            started.set()
            await release.wait()
            return AgentToolResult(
                call_id=call.id,
                content="late",
                safe_summary="late",
            )

    call = AgentToolCall(id="call", name=READ.name, arguments={"path": "x"})
    service = AgentServerService(
        registry=SlowRegistry((READ,)),
        chat_driver=ScriptedDriver(AgentModelTurn(tool_calls=[call])),
    )
    created = await service.create(AgentRunCreateRequest(goal="read"), model="model")
    await started.wait()
    entry = service._entry(created.id)
    service._runtime.cancel(entry.run)
    release.set()
    await entry.task
    assert entry.run.status is AgentRunStatus.CANCELLED


@pytest.mark.asyncio
async def test_runtime_rejection_is_marked_terminal_by_adapter():
    driver = ScriptedDriver(
        AgentModelTurn(
            tool_calls=[AgentToolCall(id="bad", name="not_advertised", arguments={})]
        )
    )
    service = AgentServerService(registry=FakeRegistry((READ,)), chat_driver=driver)
    created = await service.create(AgentRunCreateRequest(goal="x"), model="model")

    failed = await wait_for_status(service, created.id, AgentRunStatus.FAILED)

    assert failed.failure_code == "unadvertised_tool_call"


@pytest.mark.asyncio
async def test_repeated_call_guard_observation_reaches_final_synthesis():
    calls = [
        AgentToolCall(id=f"call-{index}", name=READ.name, arguments={"path": "x"})
        for index in range(3)
    ]
    driver = ScriptedDriver(
        *(AgentModelTurn(tool_calls=[call]) for call in calls),
        AgentModelTurn(content="Stopped repeating."),
    )
    registry = FakeRegistry((READ,))
    service = AgentServerService(registry=registry, chat_driver=driver)
    created = await service.create(
        AgentRunCreateRequest(goal="read x"), model="minicpm5-2b-4bit"
    )

    done = await wait_for_status(service, created.id, AgentRunStatus.COMPLETED)

    assert done.output == "Stopped repeating."
    assert len(registry.calls) == 2
    assert driver.requests[-1][2] == []
    assert "blocked because it repeated" in driver.requests[-1][1][-1]["content"]
