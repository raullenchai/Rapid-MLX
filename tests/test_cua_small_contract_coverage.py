"""Boundary cases for CUA run identity and planner response contracts."""

from __future__ import annotations

import asyncio

import pytest

from rapid_mlx.cua import service
from rapid_mlx.cua.planner import Planner, validate_plan


@pytest.mark.parametrize(
    "domain",
    ["bad domain", "-bad.example", "x" * 201, "example..com"],
)
def test_allowed_domain_rejects_non_dns_authority(domain):
    with pytest.raises(ValueError, match="invalid allowed_domain"):
        service._normalize_allowed_domain(domain)


@pytest.mark.parametrize(
    ("targets", "initial", "app", "message"),
    [
        (
            [{"target_id": "one", "app": "pid:42", "pid": 42}],
            "one",
            "pid:42",
            "two or three",
        ),
        (
            [
                {"target_id": "one", "app": "pid:42", "pid": 42},
                {"target_id": "one", "app": "pid:43", "pid": 43},
            ],
            "one",
            "pid:42",
            "unique target_id",
        ),
        (
            [
                {"target_id": "one", "app": "pid:42", "pid": 42},
                {"target_id": "two", "app": "pid:43", "pid": 43},
            ],
            "missing",
            "pid:42",
            "initial_target_id",
        ),
        (
            [
                {"target_id": "one", "app": "pid:99", "pid": 42},
                {"target_id": "two", "app": "pid:43", "pid": 43},
            ],
            "one",
            "pid:99",
            "pid:<pid>",
        ),
        (
            [
                {"target_id": "one", "app": "pid:42", "pid": 42},
                {"target_id": "two", "app": "pid:43", "pid": 43},
            ],
            "one",
            "pid:43",
            "initial target app",
        ),
    ],
)
def test_service_rejects_ambiguous_multi_target_authority(
    targets, initial, app, message
):
    registry = service.CUAService()
    with pytest.raises(ValueError, match=message):
        asyncio.run(
            registry.create(
                app=app,
                goal="read",
                targets=targets,
                initial_target_id=initial,
            )
        )
    assert registry.list_runs() == []


def test_request_id_lookup_prunes_stale_retention_index():
    registry = service.CUAService()
    registry._request_runs["stale-id"] = (object(), "absent-run")
    with pytest.raises(service.CUARequestIdentityNotFoundError):
        registry.get_by_request_id("stale-id")
    assert "stale-id" not in registry._request_runs


def test_approval_fallback_allocates_gate_identity_and_target_event():
    run = service.CUAServiceRun(
        run_id="r",
        app="pid:42",
        goal="read",
        config=None,
        targets=[{"target_id": "one", "app": "pid:42", "window_id": "cg:1"}],
        active_target_id="one",
    )
    run._awaiting = True
    assert run.resolve_gate(True) is True
    gate = run.view()["pending_gate"]
    assert gate["gate_id"]
    assert gate["approved"] is True
    run.emit({"kind": "gate_detail", "reason": "review"})
    assert run.events[-1]["target_id"] == "one"


def test_switch_target_requires_nonempty_target_id():
    with pytest.raises(ValueError, match="requires target_id"):
        validate_plan(
            {"action": "switch_target", "step_instruction": "switch"},
            valid_target_ids={"one", "two"},
        )


@pytest.mark.parametrize(
    "body",
    [{}, {"choices": []}, {"choices": [None]}, {"choices": [{"message": None}]}],
)
def test_planner_rejects_missing_or_unstructured_choice(monkeypatch, body):
    class Response:
        is_error = False

        def json(self):
            return body

    planner = Planner(url="http://127.0.0.1:9/v1", model="m")

    async def post(*args, **kwargs):
        return Response()

    monkeypatch.setattr(planner.client, "post", post)
    with pytest.raises(RuntimeError, match=r"choices\[0\].message"):
        asyncio.run(planner._ask([], 1, {}, "test"))
    asyncio.run(planner.close())


def test_multi_target_service_rejects_legacy_window_authority_mix():
    registry = service.CUAService()
    targets = [
        {"target_id": "one", "app": "pid:42", "pid": 42, "window_id": "cg:1"},
        {"target_id": "two", "app": "pid:43", "pid": 43, "window_id": "cg:2"},
    ]
    with pytest.raises(ValueError, match="targets cannot be combined"):
        asyncio.run(
            registry.create(
                app="pid:42",
                goal="read",
                targets=targets,
                initial_target_id="one",
                window_id="cg:1",
            )
        )
    assert registry.list_runs() == []


def test_gate_detail_retains_pending_target_identity_without_event_target():
    run = service.CUAServiceRun(run_id="r", app="pid:42", goal="read", config=None)
    run._pending_gate = {"gate_id": "g", "target_id": "one"}
    run.emit({"kind": "gate_detail", "reason": "review"})
    assert run.events[-1]["target_id"] == "one"
    assert run.events[-1]["gate_id"] == "g"
