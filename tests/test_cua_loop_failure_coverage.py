"""Failure injection at CUA step boundaries must stop without executing stale input."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from rapid_mlx.computer_use.errors import ComputerUseError
from rapid_mlx.cua import loop as loop_mod
from rapid_mlx.cua.config import CUAConfig, PlannerConfig
from rapid_mlx.cua.gates import ConsentError


def _snapshot() -> dict:
    return {
        "snapshot_id": "fresh",
        "app": {"name": "TextEdit", "bundleId": "com.apple.TextEdit", "pid": 42},
        "window_id": "cg:1",
        "window_index": 0,
        "window": {"window_id": "cg:1", "index": 0, "title": "Note"},
        "elements": [
            {
                "index": 1,
                "role": "AXButton",
                "label": "Apply",
                "x": 10,
                "y": 10,
                "width": 40,
                "height": 20,
                "center": [30, 20],
                "source_window_id": "cg:1",
            }
        ],
        "tree_text": "[1] AXButton Apply",
    }


def _plan(action="click", **updates) -> dict:
    plan = {
        "action": action,
        "step_instruction": "apply change",
        "element_index": 1,
        "final_summary": "work is partial",
        "text": "",
        "key": "",
        "direction": "",
    }
    plan.update(updates)
    return plan


def _runner(tmp_path, monkeypatch, *, window_id="cg:1", plan=None):
    config = CUAConfig(
        planner=PlannerConfig(
            preset="test", url="http://127.0.0.1:1/v1/chat/completions", model="fake"
        ),
        fast_ranker_url="",
    )
    events = []
    run = loop_mod.CUARun(
        config,
        "TextEdit",
        "edit note",
        tmp_path,
        event_sink=events.append,
        window_id=window_id,
    )
    monkeypatch.setattr(run, "_get_app_state", lambda **kwargs: _snapshot())
    monkeypatch.setattr(run, "_read_url", lambda snapshot: "")
    monkeypatch.setattr(loop_mod.gates, "check_plan_consents", lambda *args: None)
    monkeypatch.setattr(loop_mod.gates, "consequential_action", lambda *args: None)
    monkeypatch.setattr(loop_mod.backend, "read_url", lambda *args, **kwargs: "")
    monkeypatch.setattr(loop_mod.backend, "click", lambda *args, **kwargs: {"ok": True})

    async def request_plan(*args, **kwargs):
        return dict(plan or _plan()), "", 0.01, []

    monkeypatch.setattr(run, "_request_plan", request_plan)

    async def no_delay(_seconds):
        return None

    monkeypatch.setattr(loop_mod.asyncio, "sleep", no_delay)
    return run, events


def _fail(message="selected window disappeared", code="window_stale"):
    raise ComputerUseError(code, message)


def test_selected_window_observation_loss_stops_before_planning(tmp_path, monkeypatch):
    run, events = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(run, "_get_app_state", lambda **kwargs: _fail())
    result = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert result == {
        "status": "stopped",
        "reason": "selected window unavailable: selected window disappeared",
        "error": "window_stale",
    }
    assert run.trace["status"] == "stopped"
    assert events[-1]["kind"] == "terminal"


@pytest.mark.parametrize("terminal", ["partial", "blocked"])
def test_failed_prior_action_accepts_honest_noncompletion(
    tmp_path, monkeypatch, terminal
):
    run, _ = _runner(tmp_path, monkeypatch, plan=_plan(terminal))
    run._last_execution_failed = True
    result = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert result == {
        "status": "stalled",
        "reason": "work is partial",
        "completion_disposition": terminal,
    }
    assert run.trace["completion_disposition"] == terminal


@pytest.mark.parametrize("terminal", ["partial", "blocked"])
def test_unfailed_noncompletion_is_explicitly_recorded(tmp_path, monkeypatch, terminal):
    run, _ = _runner(tmp_path, monkeypatch, plan=_plan(terminal))
    result = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert result["status"] == "stalled"
    assert result["completion_disposition"] == terminal
    assert run.trace["final_summary"] == "work is partial"


def test_save_requires_prebound_identity_before_consent_or_action(
    tmp_path, monkeypatch
):
    run, _ = _runner(tmp_path, monkeypatch, plan=_plan("save"))
    monkeypatch.setattr(
        loop_mod.backend,
        "inspect_save_document",
        lambda *args: _fail("selected document changed", "target_drift"),
    )
    result = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert result["status"] == "stopped"
    assert result["error"] == "target_drift"
    assert "selected document changed" in result["reason"]


def test_plan_consent_failure_stops_before_execution(tmp_path, monkeypatch):
    run, _ = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        loop_mod.gates,
        "check_plan_consents",
        lambda *args: (_ for _ in ()).throw(ConsentError("credential input blocked")),
    )
    result = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert result["status"] == "stopped"
    assert result["reason"] == "credential input blocked"
    assert run.trace["consent_stop"] == "credential input blocked"


def test_approved_target_reobserve_failure_blocks_input(tmp_path, monkeypatch):
    run, _ = _runner(tmp_path, monkeypatch)
    calls = 0

    def observe(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            _fail("window closed during approval")
        return _snapshot()

    monkeypatch.setattr(run, "_get_app_state", observe)
    monkeypatch.setattr(
        loop_mod.gates,
        "consequential_action",
        lambda *args: SimpleNamespace(
            reason="external commit", action="click", target="Apply", kind="commit"
        ),
    )

    async def approved(*args, **kwargs):
        return True

    monkeypatch.setattr(run, "_request_approval", approved)
    result = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert result["status"] == "stopped"
    assert result["error"] == "window_stale"
    assert "after approval" in result["reason"]


def test_approved_target_rechecks_consent_after_pause(tmp_path, monkeypatch):
    run, _ = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        loop_mod.gates,
        "consequential_action",
        lambda *args: SimpleNamespace(
            reason="external commit", action="click", target="Apply", kind="commit"
        ),
    )

    async def approved(*args, **kwargs):
        return True

    monkeypatch.setattr(run, "_request_approval", approved)
    checks = 0

    def consent(*args):
        nonlocal checks
        checks += 1
        if checks == 2:
            raise ConsentError("target became credential field")

    monkeypatch.setattr(loop_mod.gates, "check_plan_consents", consent)
    result = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert result == {
        "status": "stopped",
        "reason": "target became credential field",
    }
    assert checks == 2


def test_selected_window_loss_between_plan_and_input_blocks_action(
    tmp_path, monkeypatch
):
    run, _ = _runner(tmp_path, monkeypatch)
    calls = 0

    def observe(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            _fail("window replaced before input")
        return _snapshot()

    monkeypatch.setattr(run, "_get_app_state", observe)
    result = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert result["status"] == "stopped"
    assert result["error"] == "window_stale"
    assert "before action" in result["reason"]
    assert calls == 2


def test_occlusion_raise_error_stops_without_retrying_click(tmp_path, monkeypatch):
    run, _ = _runner(tmp_path, monkeypatch)
    attempts = []

    async def occluded(*args):
        attempts.append("click")
        return {"ok": False, "error_code": "target_occluded"}

    monkeypatch.setattr(run, "_execute", occluded)
    monkeypatch.setattr(
        loop_mod.backend,
        "raise_selected_window",
        lambda *args: _fail("another window blocks it", "target_occluded"),
    )
    result = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert result["status"] == "stopped"
    assert result["error"] == "target_occluded"
    assert "remains occluded" in result["reason"]
    assert attempts == ["click"]


def test_occlusion_recovery_rechecks_consent_before_retry(tmp_path, monkeypatch):
    run, _ = _runner(tmp_path, monkeypatch)
    attempts = []

    async def occluded(*args):
        attempts.append("click")
        return {"ok": False, "error_code": "target_occluded"}

    monkeypatch.setattr(run, "_execute", occluded)
    monkeypatch.setattr(
        loop_mod.backend, "raise_selected_window", lambda *args: {"window_id": "cg:1"}
    )
    checks = 0

    def consent(*args):
        nonlocal checks
        checks += 1
        if checks == 2:
            raise ConsentError("target became payment control")

    monkeypatch.setattr(loop_mod.gates, "check_plan_consents", consent)
    result = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert result == {
        "status": "stopped",
        "reason": "target became payment control",
    }
    assert attempts == ["click"]


def test_persistent_occlusion_stops_after_one_focused_retry(tmp_path, monkeypatch):
    run, _ = _runner(tmp_path, monkeypatch)
    attempts = []

    async def occluded(*args):
        attempts.append("click")
        return {"ok": False, "error_code": "target_occluded"}

    monkeypatch.setattr(run, "_execute", occluded)
    monkeypatch.setattr(
        loop_mod.backend, "raise_selected_window", lambda *args: {"window_id": "cg:1"}
    )
    result = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert result["status"] == "stopped"
    assert result["error"] == "target_occluded"
    assert len(attempts) == 2


def test_selected_window_loss_after_input_stops_without_claiming_completion(
    tmp_path, monkeypatch
):
    run, _ = _runner(tmp_path, monkeypatch)
    calls = 0

    def observe(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            _fail("window closed after input")
        return _snapshot()

    monkeypatch.setattr(run, "_get_app_state", observe)
    result = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert result["status"] == "stopped"
    assert result["error"] == "window_stale"
    assert "after action" in result["reason"]
    assert calls == 3


def test_signin_file_gate_uses_dedicated_marker_without_gui_callback(
    tmp_path, monkeypatch
):
    run, _ = _runner(tmp_path, monkeypatch)
    seen = []

    async def wait(path, marker, timeout, **kwargs):
        seen.append((path, marker, timeout, kwargs))
        return True

    monkeypatch.setattr(loop_mod.gates, "wait_for_human", wait)
    assert asyncio.run(run._request_signin_approval()) is True
    assert seen == [(tmp_path, "APPROVE_SIGNIN", run.config.pause_timeout, {})]


def test_unknown_frozen_target_cannot_activate(tmp_path, monkeypatch):
    run, _ = _runner(tmp_path, monkeypatch)
    with pytest.raises(ComputerUseError, match="unknown target_id"):
        run._activate_target("unreviewed")


def test_final_assessment_records_honest_partial_disposition(tmp_path, monkeypatch):
    run, _ = _runner(tmp_path, monkeypatch, plan=_plan("partial"))
    result = asyncio.run(run.final_assessment(SimpleNamespace(text_only=True), 2))
    assert result == {
        "status": "stalled",
        "reason": "work is partial",
        "completion_disposition": "partial",
    }
    assert run.trace["completion_disposition"] == "partial"


def test_save_execution_refuses_missing_bound_identity(tmp_path, monkeypatch):
    run, _ = _runner(tmp_path, monkeypatch)
    result = asyncio.run(run._execute(_plan("save"), _snapshot()))
    assert result["ok"] is False
    assert result["error_code"] == "target_drift"
    assert result["executed"] is False


def test_switch_to_already_active_target_is_no_effect(tmp_path, monkeypatch):
    run, events = _runner(
        tmp_path,
        monkeypatch,
        plan=_plan("switch_target", target_id="one"),
    )
    run.targets = {
        "one": {
            "target_id": "one",
            "app": "pid:42",
            "window_id": "cg:1",
            "expected_app": {"pid": 42},
        }
    }
    run.active_target_id = "one"
    result = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert result is None
    assert run.history[-1]["outcome"] == "no_effect"
    assert not any(event["kind"] == "target_switched" for event in events)


def test_human_signin_gate_timeout_stops_before_input(tmp_path, monkeypatch):
    run, _ = _runner(tmp_path, monkeypatch)
    run.config.human_login = True
    monkeypatch.setattr(loop_mod.gates, "looks_like_sign_in", lambda snapshot: True)

    async def denied():
        return False

    monkeypatch.setattr(run, "_request_signin_approval", denied)
    result = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert result == {
        "status": "stopped",
        "reason": "sign-in gate not approved",
    }
    assert run.trace["human_gate"] == "sign-in gate timed out"


def test_save_binding_changed_after_approval_is_window_stale(tmp_path, monkeypatch):
    run, _ = _runner(tmp_path, monkeypatch, plan=_plan("save"))
    monkeypatch.setattr(
        loop_mod.gates,
        "consequential_action",
        lambda *args: SimpleNamespace(
            reason="save file", action="save", target="Note", kind="commit"
        ),
    )

    async def approved(*args, **kwargs):
        return True

    monkeypatch.setattr(run, "_request_approval", approved)
    calls = 0

    def inspect(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            _fail("document identity changed", "target_drift")
        return {"save_identity": ("pid:42", "cg:1", "Note")}

    monkeypatch.setattr(loop_mod.backend, "inspect_save_document", inspect)
    result = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert result["status"] == "stopped"
    assert result["error"] == "window_stale"
    assert calls == 2


def test_unselected_app_post_action_observation_error_bubbles(tmp_path, monkeypatch):
    run, _ = _runner(tmp_path, monkeypatch, window_id=None)
    calls = 0

    def observe(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            _fail("AX service unavailable", "ax_unavailable")
        return _snapshot()

    monkeypatch.setattr(run, "_get_app_state", observe)
    with pytest.raises(ComputerUseError, match="AX service unavailable"):
        asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    assert calls == 2


def test_non_signin_file_gate_carries_exact_action_reason(tmp_path, monkeypatch):
    run, _ = _runner(tmp_path, monkeypatch)
    seen = []

    async def wait(path, marker, timeout, **kwargs):
        seen.append((path, marker, timeout, kwargs))
        return False

    monkeypatch.setattr(loop_mod.gates, "wait_for_human", wait)
    approved = asyncio.run(run._request_approval("delete external file"))
    assert approved is False
    assert seen[0][0] == tmp_path
    assert seen[0][1].startswith("APPROVE_ACTION_")
    assert seen[0][3] == {"reason": "delete external file"}


def test_repeated_false_done_claim_stops_after_two_rejections(tmp_path, monkeypatch):
    run, _ = _runner(tmp_path, monkeypatch, plan=_plan("done"))
    run._last_commit_unverified = True
    first = asyncio.run(run.step(SimpleNamespace(text_only=True), 1))
    second = asyncio.run(run.step(SimpleNamespace(text_only=True), 2))
    assert first is None
    assert second["status"] == "stopped"
    assert "could not be verified" in second["reason"]
    assert run._failed_completion_rejections == 2
    assert [item["outcome"] for item in run.history] == ["no_effect", "no_effect"]
