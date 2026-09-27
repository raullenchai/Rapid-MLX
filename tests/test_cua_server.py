"""Tests for the server-side CUA surface (/v1/cua/*).

The router is mounted on a bare FastAPI app; the backend (AX) and the run
loop are stubbed so no computer access or HTTP planner calls happen.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import get_config, reset_config
from rapid_mlx.cua import service as cua_service
from rapid_mlx.routes import cua as cua_routes


@pytest.fixture()
def authorized():
    cfg = reset_config()
    cfg.api_key = "secret"
    yield cfg
    reset_config()


@pytest.fixture()
def client(monkeypatch, tmp_path, authorized):
    from rapid_mlx.computer_use import backend as backend_mod
    from rapid_mlx.cua import config as config_mod

    # isolated run dir + config
    monkeypatch.setattr(config_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "cua-config.json")

    # fake AX backend: one element, never changes
    def fake_get_app_state(app, screenshot=True, use_cache=True):
        return {
            "app": {"name": app},
            "elements": [{"index": 1, "label": "Search", "role": "AXTextField"}],
            "tree_text": "[1] AXTextField Search",
        }

    monkeypatch.setattr(backend_mod, "get_app_state", fake_get_app_state)
    monkeypatch.setattr(
        backend_mod, "read_url", lambda app: "https://www.wikipedia.org/"
    )
    monkeypatch.setattr(backend_mod, "click", lambda app, index, **k: {"ok": True})
    monkeypatch.setattr(
        backend_mod,
        "set_value",
        lambda app, index, value: {"ok": True, "verified": True, "actual": value},
    )

    # fake loop: emits plan/executed then completes (no live planner/AX)
    async def fake_run_loop(config, app, goal, **kwargs):
        sink = kwargs.get("event_sink")
        if sink is not None:
            sink(
                {
                    "kind": "plan",
                    "step": 1,
                    "action": "click",
                    "step_instruction": "do it",
                }
            )
            sink({"kind": "executed", "step": 1, "outcome": "success"})
        return {"status": "done", "final_summary": "opened the article"}

    monkeypatch.setattr(cua_service, "run_loop", fake_run_loop)

    # fresh service per test
    fresh = cua_service.CUAService()
    monkeypatch.setattr(cua_service, "_SERVICE", fresh)
    monkeypatch.setattr(cua_routes, "get_config", get_config, raising=False)
    app = FastAPI()
    app.include_router(cua_routes.router)
    with TestClient(app) as test_client:
        test_client.fresh_service = fresh  # type: ignore[attr-defined]
        yield test_client


AUTH = {"Authorization": "Bearer secret"}


def _post_run(client, **overrides):
    body = {
        "app": "Google Chrome",
        "goal": "open the article",
        "planner": "local-9b",
        "max_steps": 6,
        **overrides,
    }
    return client.post("/v1/cua/runs", headers=AUTH, json=body)


def test_requires_auth(client):
    test_client = client
    assert (
        test_client.post("/v1/cua/runs", json={"app": "A", "goal": "g"}).status_code
        == 401
    )
    assert test_client.get("/v1/cua/runs").status_code == 401


def test_list_planners(client):
    test_client = client
    response = test_client.get("/v1/cua/planners", headers=AUTH)
    assert response.status_code == 200
    names = {p["name"] for p in response.json()}
    assert {"cloud-glm", "local-27b", "local-9b"} <= names


def test_run_lifecycle_done(client):
    test_client = client
    created = _post_run(client)
    assert created.status_code == 202
    run_id = created.json()["run_id"]

    import time

    deadline = time.time() + 10
    view = None
    while time.time() < deadline:
        view = test_client.get(f"/v1/cua/runs/{run_id}", headers=AUTH).json()
        if view["status"] not in {"running", "awaiting_approval"}:
            break
        time.sleep(0.1)
    assert view is not None and view["status"] == "completed"
    assert view["final_summary"] == "opened the article"
    kinds = [e["kind"] for e in view["events"]]
    assert kinds[0] == "started" and "plan" in kinds and "executed" in kinds
    assert kinds[-1] == "terminal"

    # pagination: events after the last seq is empty
    tail = test_client.get(
        f"/v1/cua/runs/{run_id}/events",
        headers=AUTH,
        params={"after": len(view["events"])},
    ).json()["events"]
    assert tail == []


def test_create_rejects_bad_planner_and_concurrency(client):
    test_client = client
    fresh = client.fresh_service
    bad = _post_run(client, planner="nope")
    assert bad.status_code == 400

    # simulate an active run -> conflict
    active = cua_service.CUAServiceRun(
        run_id="active1",
        app="A",
        goal="g",
        config=None,  # type: ignore[arg-type]
    )
    active.status = "running"
    fresh._runs["active1"] = active
    conflict = _post_run(client)
    assert conflict.status_code == 409


def test_approval_gate_flow(client):
    test_client = client
    fresh = client.fresh_service
    # seed a run that is awaiting approval
    active = cua_service.CUAServiceRun(
        run_id="gate1",
        app="A",
        goal="g",
        config=None,  # type: ignore[arg-type]
    )
    active.status = "awaiting_approval"
    active._awaiting = True
    fresh._runs["gate1"] = active

    not_waiting = test_client.post("/v1/cua/runs/none/approval", headers=AUTH)
    assert not_waiting.status_code == 404

    async def scenario():
        task = asyncio.create_task(active.wait_for_approval("sign-in", timeout=5))
        await asyncio.sleep(0.05)
        assert active.approve() is True
        return await task

    assert asyncio.run(scenario()) is True
    # loop-level rule: approve only resolves a waiting gate
    assert active.approve() is False


def test_cancel_requests_stop(client):
    test_client = client
    fresh = client.fresh_service
    active = cua_service.CUAServiceRun(
        run_id="stop1",
        app="A",
        goal="g",
        config=None,  # type: ignore[arg-type]
    )
    active.status = "running"
    fresh._runs["stop1"] = active
    response = test_client.post("/v1/cua/runs/stop1/cancel", headers=AUTH)
    assert response.status_code == 200
    assert active._stop_event.is_set()
    missing = test_client.post("/v1/cua/runs/zzz/cancel", headers=AUTH)
    assert missing.status_code == 404


def test_events_after_seq_pagination(client):
    test_client = client
    fresh = client.fresh_service
    active = cua_service.CUAServiceRun(
        run_id="ev1",
        app="A",
        goal="g",
        config=None,  # type: ignore[arg-type]
    )
    fresh._runs["ev1"] = active
    active.emit({"kind": "started"})
    active.emit({"kind": "plan", "step": 1})
    active.emit({"kind": "executed", "step": 1})
    view = test_client.get(
        "/v1/cua/runs/ev1/events", headers=AUTH, params={"after": 2}
    ).json()
    assert [e["seq"] for e in view["events"]] == [3]
    assert view["events"][0]["kind"] == "executed"
