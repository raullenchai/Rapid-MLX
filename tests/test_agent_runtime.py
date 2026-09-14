# SPDX-License-Identifier: Apache-2.0

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from pydantic import ValidationError

from vllm_mlx.agent_runtime import (
    AgentEvent,
    AgentModelTurn,
    AgentProfile,
    AgentRun,
    AgentRunStatus,
    AgentRuntime,
    AgentRuntimeError,
    AgentToolCall,
    AgentToolResult,
    ToolRisk,
    ToolSpec,
    resolve_agent_profile,
)


def _clock():
    value = 100.0
    while True:
        yield value
        value += 1


def _runtime() -> AgentRuntime:
    ticks = _clock()
    return AgentRuntime(clock=lambda: next(ticks))


READ = ToolSpec(name="read_file", risk=ToolRisk.READ_ONLY)
WRITE = ToolSpec(name="write_file", risk=ToolRisk.LOCAL_CHANGE)
SEND = ToolSpec(name="send_message", risk=ToolRisk.EXTERNAL_SIDE_EFFECT)


def _call(call_id: str = "call-1", **arguments) -> AgentToolCall:
    return AgentToolCall(id=call_id, name="read_file", arguments=arguments)


def test_minicpm_profile_is_alias_and_repo_aware():
    for model in (
        "minicpm5-2b-4bit",
        "openbmb/MiniCPM5-2B-MLX",
        "mlx-community/MiniCPM5_2B_8bit",
        "mlx-community/MiniCPM5-2B-4bit",
    ):
        profile = resolve_agent_profile(model)
        assert profile.name == "minicpm5-2b"
        assert profile.max_visible_tools == 8
        assert profile.max_tool_rounds == 12

    assert resolve_agent_profile("qwen3.5-4b-4bit").name == "default"


def test_minicpm_profile_uses_exact_loaded_metadata_for_custom_local_paths():
    config = {
        "model_type": "llama",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 42,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "vocab_size": 130560,
    }

    assert (
        resolve_agent_profile(
            "/models/my-local-copy",
            model_config=config,
            tool_call_parser="minicpm",
        ).name
        == "minicpm5-2b"
    )
    assert (
        resolve_agent_profile(
            "/models/my-local-copy",
            model_config=config,
            tool_call_parser="hermes",
        ).name
        == "default"
    )
    unverified = resolve_agent_profile("/models/minicpm5-2b-copy")
    assert unverified.name == "default-conservative"
    assert unverified.max_visible_tools == 8
    assert unverified.max_tool_rounds == 8


def test_run_identity_and_profile_are_immutable_after_creation():
    run = _runtime().create_run(model="minicpm5-2b-4bit", goal="Do the task")

    with pytest.raises(ValidationError, match="frozen"):
        run.profile = resolve_agent_profile("qwen3.5-4b-4bit")
    with pytest.raises(ValidationError, match="frozen"):
        run.model = "different-model"


def test_only_read_only_tools_bypass_approval():
    assert READ.risk.requires_approval is False
    assert WRITE.risk.requires_approval is True
    assert SEND.risk.requires_approval is True


def test_tool_arguments_are_json_only_at_the_wire_boundary():
    with pytest.raises(ValidationError):
        AgentToolCall(id="call-1", name="read_file", arguments={"bad": object()})
    with pytest.raises(ValidationError):
        AgentToolCall(id="call-1", name="read_file", arguments={"bad": float("nan")})


def test_tool_risk_is_required_at_registry_boundary():
    with pytest.raises(ValidationError, match="risk"):
        ToolSpec(name="unclassified")


def test_adapter_can_fail_a_live_run_with_a_stable_code():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Do the task")

    runtime.fail(run, "model_request_failed")

    assert run.status is AgentRunStatus.FAILED
    assert run.failure_code == "model_request_failed"
    assert run.events[-1].type == "run.failed"


@pytest.mark.parametrize("code", ["", "Has Caps", "contains-secret/path", "a" * 129])
def test_adapter_failure_codes_are_safe_for_events(code):
    runtime = _runtime()
    run = runtime.create_run(model="model", goal="Do the task")

    with pytest.raises(AgentRuntimeError, match="failure code"):
        runtime.fail(run, code)


def test_adapter_cannot_replace_a_terminal_outcome():
    runtime = _runtime()
    run = runtime.create_run(model="model", goal="Do the task")
    runtime.cancel(run)

    with pytest.raises(AgentRuntimeError, match="terminal run"):
        runtime.fail(run, "late_failure")


def test_tool_parameters_must_be_a_valid_json_schema():
    with pytest.raises(ValidationError, match="valid JSON Schema"):
        ToolSpec(name="broken", risk=ToolRisk.READ_ONLY, parameters={"type": 7})


def test_tool_parameters_reject_references_in_p0():
    with pytest.raises(ValidationError, match="inline JSON Schema"):
        ToolSpec(
            name="referenced",
            risk=ToolRisk.READ_ONLY,
            parameters={"$ref": "#/$defs/missing"},
        )


def test_non_json_tool_parameters_raise_a_validation_error():
    with pytest.raises(ValidationError, match="JSON serializable"):
        ToolSpec(name="broken", risk=ToolRisk.READ_ONLY, parameters={"x": object()})


@pytest.mark.parametrize("parameters_json", ["not-json", "[]"])
def test_tool_parameters_json_requires_an_object(parameters_json):
    with pytest.raises(ValidationError):
        ToolSpec(
            name="broken",
            risk=ToolRisk.READ_ONLY,
            parameters_json=parameters_json,
        )


def test_tool_parameters_rejects_both_wire_shapes():
    with pytest.raises(ValidationError, match="must not contain both"):
        ToolSpec(
            name="broken",
            risk=ToolRisk.READ_ONLY,
            parameters={},
            parameters_json="{}",
        )


def test_non_json_event_data_raise_a_validation_error():
    with pytest.raises(ValidationError, match="JSON serializable"):
        AgentEvent(
            sequence=1,
            type="run.created",
            created_at=1,
            data={"x": object()},
        )


@pytest.mark.parametrize("payload_json", ["not-json", "[]"])
def test_event_payload_json_requires_an_object(payload_json):
    with pytest.raises(ValidationError):
        AgentEvent(
            sequence=1,
            type="run.created",
            created_at=1,
            payload_json=payload_json,
        )


def test_event_rejects_both_wire_shapes():
    with pytest.raises(ValidationError, match="must not contain both"):
        AgentEvent(
            sequence=1,
            type="run.created",
            created_at=1,
            data={},
            payload_json="{}",
        )


def test_successful_tool_round_has_stable_events_and_roundtrips():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read the report")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(
        run,
        AgentModelTurn(tool_calls=[_call(path="report.md")]),
    )
    assert run.status is AgentRunStatus.AWAITING_TOOL_RESULT

    runtime.accept_tool_result(
        run,
        AgentToolResult(call_id="call-1", content="Revenue fell 12%."),
    )
    runtime.request_model(run, [READ])
    completed = runtime.accept_model_turn(
        run,
        AgentModelTurn(content="Revenue fell 12%."),
    )

    assert run.status is AgentRunStatus.COMPLETED
    assert completed is not None
    assert completed.final_content == "Revenue fell 12%."
    assert [event.sequence for event in run.events] == list(
        range(1, len(run.events) + 1)
    )
    assert [event.type for event in run.events] == [
        "run.created",
        "model.requested",
        "tool.requested",
        "tool.completed",
        "model.requested",
        "run.completed",
    ]
    event = run.events[-1]
    restored = AgentEvent.model_validate_json(event.model_dump_json())
    assert restored == event
    assert event.data == {"content_bytes": len(b"Revenue fell 12%.")}
    assert "ledger" not in run.events[3].data
    assert "Choose only the next necessary action" in runtime.ledger_context(run)


def test_goal_bearing_ledger_never_enters_wire_events():
    runtime = _runtime()
    secret = "clipboard-secret-do-not-persist"
    run = runtime.create_run(model="minicpm5-2b-4bit", goal=f"Use {secret}")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call(path="a.md")]))
    runtime.accept_tool_result(
        run,
        AgentToolResult(call_id="call-1", content="sensitive result"),
    )

    assert secret in runtime.ledger_context(run)
    assert secret not in "".join(event.model_dump_json() for event in run.events)


def test_final_content_is_transient_and_never_enters_wire_events():
    runtime = _runtime()
    secret = "tool-derived-secret-do-not-persist"
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Answer")
    runtime.request_model(run, [])

    output = runtime.accept_model_turn(run, AgentModelTurn(content=secret))

    assert output is not None
    assert output.final_content == secret
    assert run.status is AgentRunStatus.COMPLETED
    assert run.events[-1].data == {"content_bytes": len(secret.encode())}
    assert secret not in "".join(event.model_dump_json() for event in run.events)


def test_minicpm_rejects_an_oversized_tool_surface():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Do the task")
    tools = [
        ToolSpec(name=f"tool_{index}", risk=ToolRisk.READ_ONLY) for index in range(9)
    ]

    with pytest.raises(AgentRuntimeError, match="at most 8 visible tools"):
        runtime.request_model(run, tools)

    assert run.status is AgentRunStatus.READY
    assert run.model_turns == 0


def test_visible_tool_names_must_be_unique():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read")

    with pytest.raises(AgentRuntimeError, match="must be unique"):
        runtime.request_model(run, [READ, READ])

    assert run.status is AgentRunStatus.READY


def test_profile_override_cannot_weaken_model_limits():
    runtime = _runtime()
    weakened = AgentProfile(
        name="unsafe",
        max_visible_tools=9,
        max_tool_rounds=8,
        repeated_call_limit=2,
    )

    with pytest.raises(AgentRuntimeError, match="weakens required limits"):
        runtime.create_run(
            model="minicpm5-2b-4bit",
            goal="Do the task",
            profile=weakened,
        )


def test_unadvertised_tool_call_fails_closed():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read the report")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(
        run,
        AgentModelTurn(
            tool_calls=[AgentToolCall(id="call-1", name="exec", arguments={})]
        ),
    )

    assert run.status is AgentRunStatus.FAILED
    assert run.failure_code == "unadvertised_tool_call"
    assert run.events[-1].type == "run.failed"


def test_schema_incompatible_tool_arguments_fail_closed():
    runtime = _runtime()
    read = ToolSpec(
        name="read_file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        risk=ToolRisk.READ_ONLY,
    )
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read")
    runtime.request_model(run, [read])

    output = runtime.accept_model_turn(
        run,
        AgentModelTurn(tool_calls=[_call(path=42)]),
    )

    assert output is None
    assert run.status is AgentRunStatus.FAILED
    assert run.failure_code == "invalid_tool_arguments"
    assert not any(event.type == "tool.requested" for event in run.events)


def test_tool_policy_is_snapshotted_before_the_model_turn():
    runtime = _runtime()
    source = ToolSpec(
        name="send_message",
        parameters={"properties": {"text": {"type": "string"}}},
        risk=ToolRisk.EXTERNAL_SIDE_EFFECT,
    )
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Send the update")

    offered = runtime.request_model(run, [source])
    offered[0].parameters["properties"].clear()
    source.parameters["properties"]["text"]["type"] = "integer"
    runtime.accept_model_turn(
        run,
        AgentModelTurn(
            tool_calls=[
                AgentToolCall(
                    id="send-1",
                    name="send_message",
                    arguments={"text": "Done"},
                )
            ]
        ),
    )

    assert run.status is AgentRunStatus.AWAITING_APPROVAL
    assert run.pending_risk is ToolRisk.EXTERNAL_SIDE_EFFECT
    assert run.visible_tools[0].parameters == {
        "properties": {"text": {"type": "string"}}
    }

    detached = run.visible_tools[0].parameters
    detached["properties"].clear()
    assert run.visible_tools[0].parameters == {
        "properties": {"text": {"type": "string"}}
    }


def test_external_side_effect_denial_finishes_with_deterministic_copy():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Send the update")
    runtime.request_model(run, [SEND])
    runtime.accept_model_turn(
        run,
        AgentModelTurn(
            tool_calls=[
                AgentToolCall(
                    id="send-1",
                    name="send_message",
                    arguments={"to": "Mina", "text": "Done"},
                )
            ]
        ),
    )

    assert run.status is AgentRunStatus.AWAITING_APPROVAL
    denied = runtime.resolve_approval(run, call_id="send-1", approved=False)
    assert run.status is AgentRunStatus.COMPLETED
    assert run.pending_call is None
    assert denied is not None
    assert denied.final_content == "That action wasn’t approved, so it wasn’t run."
    assert run.events[-3].type == "approval.resolved"
    assert run.events[-2].data["result"]["is_error"] is True
    assert run.events[-2].data["result"]["safe_summary"] == (
        "User denied the tool call."
    )
    assert run.events[-2].data["result"]["executed"] is False
    assert run.events[-1].type == "run.completed"


def test_external_call_is_released_only_after_exact_approval():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Send")
    runtime.request_model(run, [SEND])
    initial = runtime.accept_model_turn(
        run,
        AgentModelTurn(
            tool_calls=[
                AgentToolCall(
                    id="send-1",
                    name="send_message",
                    arguments={"text": "Ready"},
                )
            ]
        ),
    )

    assert initial is None
    assert "Ready" not in run.model_dump_json()

    approved = runtime.resolve_approval(run, call_id="send-1", approved=True)
    assert approved is not None
    assert approved.call is not None
    assert approved.call.arguments == {"text": "Ready"}


def test_approval_fails_closed_if_transient_payload_is_unavailable():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Send")
    runtime.request_model(run, [SEND])
    runtime.accept_model_turn(
        run,
        AgentModelTurn(tool_calls=[AgentToolCall(id="send-1", name="send_message")]),
    )
    runtime._call_counts_by_run[id(run)][1].pending_side_effect_call = None

    with pytest.raises(AgentRuntimeError, match="payload is unavailable"):
        runtime.resolve_approval(run, call_id="send-1", approved=True)

    assert run.status is AgentRunStatus.AWAITING_APPROVAL


def test_concurrent_approvals_release_an_external_call_only_once():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Send")
    runtime.request_model(run, [SEND])
    runtime.accept_model_turn(
        run,
        AgentModelTurn(
            tool_calls=[
                AgentToolCall(
                    id="send-1",
                    name="send_message",
                    arguments={"text": "Done"},
                )
            ]
        ),
    )
    barrier = Barrier(2)

    def approve():
        barrier.wait()
        try:
            return runtime.resolve_approval(run, call_id="send-1", approved=True)
        except AgentRuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda _: approve(), range(2)))

    released = [output for output in outputs if output is not None]
    assert len(released) == 1
    assert released[0].call is not None
    assert released[0].call.id == "send-1"
    assert [event.type for event in run.events].count("approval.resolved") == 1


def test_stale_approval_does_not_authorize_the_pending_call():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Send the update")
    runtime.request_model(run, [SEND])
    runtime.accept_model_turn(
        run,
        AgentModelTurn(tool_calls=[AgentToolCall(id="current", name="send_message")]),
    )

    with pytest.raises(AgentRuntimeError, match="does not match pending call"):
        runtime.resolve_approval(run, call_id="stale", approved=True)

    assert run.status is AgentRunStatus.AWAITING_APPROVAL
    assert run.pending_call is not None
    assert run.pending_call.id == "current"


@pytest.mark.parametrize("not_bool", [1, 0, "true", "false", None])
def test_approval_rejects_truthy_and_falsy_non_booleans(not_bool):
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Send")
    runtime.request_model(run, [SEND])
    runtime.accept_model_turn(
        run,
        AgentModelTurn(tool_calls=[AgentToolCall(id="send-1", name="send_message")]),
    )

    with pytest.raises(AgentRuntimeError, match="must be a boolean"):
        runtime.resolve_approval(run, call_id="send-1", approved=not_bool)

    assert run.status is AgentRunStatus.AWAITING_APPROVAL


def test_mismatched_tool_result_does_not_advance_the_run():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read the report")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))

    with pytest.raises(AgentRuntimeError, match="does not match pending call"):
        runtime.accept_tool_result(
            run, AgentToolResult(call_id="different", content="wrong")
        )

    assert run.status is AgentRunStatus.AWAITING_TOOL_RESULT
    assert run.pending_call is not None


def test_concurrent_tool_results_complete_a_call_only_once():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))
    barrier = Barrier(2)

    def submit_result():
        barrier.wait()
        try:
            runtime.accept_tool_result(
                run,
                AgentToolResult(call_id="call-1", content="done"),
            )
            return "accepted"
        except AgentRuntimeError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: submit_result(), range(2)))

    assert sorted(outcomes) == ["accepted", "rejected"]
    assert [event.type for event in run.events].count("tool.completed") == 1


def test_tool_result_content_is_transient_not_persisted_in_the_event_log():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read the clipboard")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))

    secret = "clipboard-secret-123"
    runtime.accept_tool_result(
        run,
        AgentToolResult(
            call_id="call-1",
            content=secret,
            safe_summary="Clipboard read completed.",
        ),
    )

    serialized = run.model_dump_json()
    assert secret not in serialized
    assert "Clipboard read completed." in serialized
    assert run.events[-1].data["result"]["content_bytes"] == len(secret)


def test_tool_argument_values_are_transient_not_persisted():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Use a credential ref")
    runtime.request_model(run, [READ])
    secret = "credential-secret-456"
    turn = AgentModelTurn(tool_calls=[_call(path="report.md", token=secret)])

    output = runtime.accept_model_turn(run, turn)

    serialized = run.model_dump_json()
    assert secret not in serialized
    assert run.pending_call is not None
    assert "arguments" not in run.pending_call.model_dump(mode="json")
    assert output is not None
    assert output.call is not None
    assert output.call.arguments["token"] == secret
    assert run.events[-1].data["call"]["argument_names"] == ["path", "token"]
    # The adapter still owns the transient input needed for immediate execution.
    assert turn.tool_calls[0].arguments["token"] == secret


def test_reused_tool_call_id_fails_closed():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read two files")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))
    runtime.accept_tool_result(run, AgentToolResult(call_id="call-1", content="one"))
    runtime.request_model(run, [READ])

    runtime.accept_model_turn(
        run,
        AgentModelTurn(tool_calls=[_call(call_id="call-1", path="other.md")]),
    )

    assert run.status is AgentRunStatus.FAILED
    assert run.failure_code == "reused_tool_call_id"


def test_repeated_identical_call_forces_tools_off_instead_of_looping():
    profile = AgentProfile(
        name="test",
        max_visible_tools=2,
        max_tool_rounds=8,
        repeated_call_limit=1,
    )
    runtime = _runtime()
    run = runtime.create_run(model="test-model", goal="Read once", profile=profile)

    runtime.request_model(run, [READ])
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))
    runtime.accept_tool_result(
        run, AgentToolResult(call_id="call-1", content="contents")
    )
    runtime.request_model(run, [READ])
    blocked = runtime.accept_model_turn(
        run,
        AgentModelTurn(tool_calls=[_call(call_id="call-2")]),
    )

    assert run.status is AgentRunStatus.READY
    assert run.final_synthesis is True
    assert run.pending_call is None
    assert run.events[-1].data["reason"] == "repeated_tool_call"
    assert [event.type for event in run.events[-3:]] == [
        "tool.requested",
        "tool.completed",
        "synthesis.required",
    ]
    assert run.events[-2].data["result"]["call_id"] == "call-2"
    assert run.events[-2].data["result"]["is_error"] is True
    assert blocked is not None
    assert blocked.observation is not None
    assert blocked.observation.call_id == "call-2"
    assert "blocked" in blocked.observation.content

    visible = runtime.request_model(run, [READ])
    assert visible == []
    assert run.events[-1].type == "model.requested"
    runtime.accept_model_turn(run, AgentModelTurn(content="Here is the result."))
    assert run.status is AgentRunStatus.COMPLETED


def test_repeat_counters_are_isolated_for_duplicate_public_run_ids():
    profile = AgentProfile(
        name="strict-repeat",
        max_visible_tools=1,
        max_tool_rounds=4,
        repeated_call_limit=1,
    )
    runtime = _runtime()
    first = runtime.create_run(
        model="test", goal="First", run_id="same", profile=profile
    )
    second = runtime.create_run(
        model="test", goal="Second", run_id="same", profile=profile
    )

    for run in (first, second):
        runtime.request_model(run, [READ])
        output = runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))
        assert output is not None
        assert output.call is not None
        assert output.observation is None


def test_abandoned_run_does_not_leak_through_repeat_tracking():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Abandon")
    identity = id(run)
    reference = weakref.ref(run)

    del run
    gc.collect()

    assert reference() is None
    assert identity not in runtime._call_counts_by_run


def test_tool_budget_reserves_a_tools_disabled_final_synthesis():
    profile = AgentProfile(
        name="one-round",
        max_visible_tools=1,
        max_tool_rounds=1,
        repeated_call_limit=2,
    )
    runtime = _runtime()
    run = runtime.create_run(model="test", goal="Read", profile=profile)
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))
    runtime.accept_tool_result(run, AgentToolResult(call_id="call-1", content="ok"))

    assert runtime.request_model(run, [READ]) == []
    assert run.final_synthesis is True
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call(call_id="again")]))
    assert run.status is AgentRunStatus.FAILED
    assert run.failure_code == "tool_call_during_final_synthesis"


def test_empty_model_turn_fails_closed():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Answer")
    runtime.request_model(run, [])

    output = runtime.accept_model_turn(run, AgentModelTurn(content="   "))

    assert output is None
    assert run.status is AgentRunStatus.FAILED
    assert run.failure_code == "empty_model_turn"


def test_parallel_calls_fail_instead_of_being_partially_executed():
    runtime = _runtime()
    run = runtime.create_run(model="qwen3.5-4b-4bit", goal="Read two files")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(
        run,
        AgentModelTurn(tool_calls=[_call(call_id="one"), _call(call_id="two")]),
    )

    assert run.status is AgentRunStatus.FAILED
    assert run.failure_code == "parallel_tool_call_limit_exceeded"
    assert not any(event.type == "tool.requested" for event in run.events)


def test_cancel_is_idempotent_and_clears_pending_state():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read")
    runtime.request_model(run, [READ])
    runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))

    runtime.cancel(run)
    runtime.cancel(run)

    assert run.status is AgentRunStatus.CANCELLED
    assert run.pending_call is None
    assert [event.type for event in run.events].count("run.cancelled") == 1


def test_run_has_no_public_event_append_escape_hatch():
    run = _runtime().create_run(model="minicpm5-2b-4bit", goal="Read")
    assert not hasattr(run, "append_event")


def test_transition_rejects_a_run_not_created_by_runtime():
    profile = resolve_agent_profile("minicpm5-2b-4bit")
    run = AgentRun(model="minicpm5-2b-4bit", goal="Read", profile=profile)

    with pytest.raises(AgentRuntimeError, match="not owned by this AgentRuntime"):
        _runtime().request_model(run, [READ])

    with pytest.raises(AgentRuntimeError, match="not owned by this AgentRuntime"):
        _runtime().cancel(run)


def test_defensive_pending_call_guards_reject_inconsistent_state():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read")

    with pytest.raises(AgentRuntimeError, match="no pending tool call"):
        runtime._pending_call(run)

    object.__setattr__(
        run,
        "pending_call",
        AgentToolCall(id="unexpected", name="read_file"),
    )
    with pytest.raises(AgentRuntimeError, match="still has a pending tool call"):
        runtime.request_model(run, [READ])


def test_event_history_and_payload_are_immutable_to_consumers():
    run = _runtime().create_run(model="minicpm5-2b-4bit", goal="Read")
    event = run.events[0]

    with pytest.raises(ValidationError, match="frozen"):
        run.events = ()
    with pytest.raises(ValidationError, match="Instance is frozen"):
        event.sequence = 9

    detached = event.data
    detached["model"] = "rewritten"
    assert event.data["model"] == "minicpm5-2b-4bit"


def test_reducer_safety_collections_are_immutable_to_consumers():
    runtime = _runtime()
    run = runtime.create_run(model="minicpm5-2b-4bit", goal="Read")
    runtime.request_model(run, [READ])

    with pytest.raises(ValidationError, match="frozen"):
        run.visible_tools = (*run.visible_tools, SEND)
    with pytest.raises(ValidationError, match="frozen"):
        run.used_call_ids = ()

    output = runtime.accept_model_turn(run, AgentModelTurn(tool_calls=[_call()]))
    assert run.pending_risk is ToolRisk.READ_ONLY
    assert output is not None
    assert output.call is not None
    assert output.call.arguments == {}

    with pytest.raises(ValidationError, match="frozen"):
        run.pending_call = AgentToolCall(id="rewritten", name="send_message")
    with pytest.raises(ValidationError, match="Instance is frozen"):
        run.pending_call.id = "rewritten"
