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
            sink({"kind": "started", "app": app, "run_dir": "/tmp/fake-cua-run"})
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
    assert {"local-27b", "local-9b"} <= names


def test_planner_crud_roundtrip(client):
    """Users add a cloud brain from settings: create, masked listing, delete,
    built-in presets protected."""
    test_client = client
    created = test_client.post(
        "/v1/cua/planners",
        headers=AUTH,
        json={
            "name": "My Cloud",
            "url": "https://api.example.com/v1/chat/completions",
            "model": "deepseek-reasoner",
            "api_key": "sk-secret",
            "text_only": True,
            "allow_remote": True,
        },
    )
    assert created.status_code == 201
    assert created.json()["name"] == "my-cloud"
    assert created.json()["has_api_key"] is True
    assert created.json()["user_created"] is True
    assert created.json()["allow_remote"] is True

    listed = test_client.get("/v1/cua/planners", headers=AUTH).json()
    entry = next(p for p in listed if p["name"] == "my-cloud")
    assert entry["has_api_key"] is True
    assert "api_key" not in entry  # never echoed back

    assert (
        test_client.delete("/v1/cua/planners/local-27b", headers=AUTH).status_code
        == 409
    )
    assert (
        test_client.delete("/v1/cua/planners/my-cloud", headers=AUTH).status_code == 200
    )
    assert (
        test_client.delete("/v1/cua/planners/my-cloud", headers=AUTH).status_code == 409
    )


def test_planner_created_from_settings_base_url_is_runnable(client):
    created = client.post(
        "/v1/cua/planners",
        headers=AUTH,
        json={
            "name": "My Cloud",
            "url": "https://api.example.com/v1",
            "model": "m",
            "api_key": "sk-secret",
            "allow_remote": True,
        },
    )
    assert created.status_code == 201
    assert created.json()["url"] == "https://api.example.com/v1/chat/completions"
    assert "api_key" not in created.json()

    run = _post_run(client, planner="my-cloud")
    assert run.status_code == 202


def test_planner_create_rejects_plaintext_remote(client):
    test_client = client
    response = test_client.post(
        "/v1/cua/planners",
        headers=AUTH,
        json={
            "name": "insecure",
            "url": "http://api.example.com/v1/chat/completions",
            "model": "m",
            "api_key": "sk-x",
            "allow_remote": True,
        },
    )
    assert response.status_code == 422
    assert "HTTPS" in response.json()["detail"]


def test_keyless_https_planner_requires_and_preserves_remote_consent(client):
    denied = client.post(
        "/v1/cua/planners",
        headers=AUTH,
        json={"name": "keyless", "url": "https://planner.example/v1", "model": "m"},
    )
    assert denied.status_code == 422
    assert "explicit consent" in denied.json()["detail"]

    created = client.post(
        "/v1/cua/planners",
        headers=AUTH,
        json={
            "name": "keyless",
            "url": "https://planner.example/v1",
            "model": "m",
            "allow_remote": True,
            "text_only": True,
        },
    )
    assert created.status_code == 201
    assert created.json()["has_api_key"] is False
    assert created.json()["allow_remote"] is True
    assert _post_run(client, planner="keyless").status_code == 202


def test_loopback_planner_needs_neither_key_nor_remote_consent(client):
    created = client.post(
        "/v1/cua/planners",
        headers=AUTH,
        json={
            "name": "local-custom",
            "url": "http://localhost:1234/v1",
            "model": "m",
        },
    )
    assert created.status_code == 201
    assert created.json()["has_api_key"] is False
    assert created.json()["allow_remote"] is False
    assert _post_run(client, planner="local-custom").status_code == 202


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
    assert kinds.count("started") == 1
    assert kinds.count("terminal") == 1
    assert view["run_dir"] == "/tmp/fake-cua-run"

    # pagination: events after the last seq is empty
    tail = test_client.get(
        f"/v1/cua/runs/{run_id}/events",
        headers=AUTH,
        params={"after": len(view["events"])},
    ).json()["events"]
    assert tail == []

    listed = test_client.get("/v1/cua/runs", headers=AUTH)
    assert listed.status_code == 200
    assert listed.json()["runs"][0]["run_id"] == run_id


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


def test_create_rejects_invalid_urls_and_blank_text(client):
    assert _post_run(client, goal="   ").status_code == 400
    assert _post_run(client, open_url="file:///tmp/private").status_code == 400
    remote = _post_run(client, planner_url="https://example.com/v1/chat")
    assert remote.status_code == 400


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
    conflict = test_client.post("/v1/cua/runs/gate1/approval", headers=AUTH)
    assert conflict.status_code == 409

    active._awaiting = True
    approved = test_client.post("/v1/cua/runs/gate1/approval", headers=AUTH)
    assert approved.status_code == 200
    assert approved.json() == {"run_id": "gate1", "approved": True}


def test_approval_timeout(client):
    active = cua_service.CUAServiceRun(
        run_id="timeout1",
        app="A",
        goal="g",
        config=None,  # type: ignore[arg-type]
    )
    assert asyncio.run(active.wait_for_approval("sign-in", timeout=0.001)) is False


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

    assert client.get("/v1/cua/runs/missing", headers=AUTH).status_code == 404
    assert client.get("/v1/cua/runs/missing/events", headers=AUTH).status_code == 404


def test_event_cursor_metadata_cannot_be_forged(client):
    active = cua_service.CUAServiceRun(
        run_id="cursor1",
        app="A",
        goal="g",
        config=None,  # type: ignore[arg-type]
    )
    active.emit(
        {
            "kind": "started",
            "seq": 999,
            "ts": 0,
            "run_dir": "/tmp/cursor1",
        }
    )
    event = active.events[0]
    assert event["seq"] == 1
    assert event["ts"] > 0
    assert active.run_dir == "/tmp/cursor1"


def test_cancel_interrupts_active_background_task(client, monkeypatch):
    service = client.fresh_service

    async def blocked_run(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(cua_service, "run_loop", blocked_run)

    async def scenario():
        run = await service.create(app="Chrome", goal="wait", planner="local-9b")
        await asyncio.sleep(0)
        service.cancel(run.run_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return run

    run = asyncio.run(scenario())
    assert run.status == "stopped"
    assert run.final_summary == "cancelled by client"
    assert run.run_id not in service._tasks
    assert [event["kind"] for event in run.events].count("terminal") == 1


def test_service_failure_pruning_and_shutdown(client, monkeypatch):
    service = client.fresh_service

    async def failed_run(*args, **kwargs):
        raise RuntimeError("planner exploded")

    monkeypatch.setattr(cua_service, "run_loop", failed_run)

    async def failure_scenario():
        run = await service.create(app="Chrome", goal="fail", planner="local-9b")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return run

    failed = asyncio.run(failure_scenario())
    assert failed.status == "failed"
    assert failed.error == "planner exploded"
    assert failed.events[-1]["status"] == "failed"

    old_limit = cua_service.MAX_RETAINED_RUNS
    monkeypatch.setattr(cua_service, "MAX_RETAINED_RUNS", 1)
    service._prune_runs()
    assert service._runs == {}
    monkeypatch.setattr(cua_service, "MAX_RETAINED_RUNS", old_limit)

    service._closing = True
    with pytest.raises(cua_service.CUARunConflictError, match="shutting down"):
        asyncio.run(service.create(app="Chrome", goal="g", planner="local-9b"))

    async def blocked_run(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(cua_service, "run_loop", blocked_run)

    async def close_active_service():
        closing = cua_service.CUAService()
        run = await closing.create(app="Chrome", goal="wait", planner="local-9b")
        await asyncio.sleep(0)
        await closing.close()
        return closing, run

    closing, stopped = asyncio.run(close_active_service())
    assert stopped.status == "stopped"
    assert closing._tasks == {}

    fresh = cua_service.CUAService()
    monkeypatch.setattr(cua_service, "_SERVICE", fresh)
    asyncio.run(cua_service.close_cua_service())
    assert cua_service.get_cua_service() is not fresh


def test_unknown_http_error_maps_to_500():
    error = cua_routes._http_error(RuntimeError("unexpected"))
    assert error.status_code == 500


def test_real_server_mounts_cua_router():
    from rapid_mlx.server import app

    paths = set(app.openapi()["paths"])
    assert "/v1/cua/runs" in paths
    assert "/v1/cua/runs/{run_id}/approval" in paths


def test_real_server_lifespan_closes_cua_service(monkeypatch):
    import rapid_mlx.server as server
    from rapid_mlx.routes import agents, audio, video

    closed = []

    async def fake_close():
        closed.append(True)

    async def no_op_async(*args, **kwargs):
        return None

    class FakeResidencyManager:
        async def start(self):
            return None

        async def shutdown(self):
            return None

    cfg = get_config()
    previous_ready = cfg.ready
    previous_draining = cfg.draining
    monkeypatch.setattr(cua_service, "close_cua_service", fake_close)
    monkeypatch.setattr(agents, "start_agent_service_lifecycle", lambda: None)
    monkeypatch.setattr(agents, "close_agent_service", no_op_async)
    monkeypatch.setattr(video, "start_video_jobs", lambda: None)
    monkeypatch.setattr(video, "shutdown_video_jobs", no_op_async)
    monkeypatch.setattr(audio, "shutdown_audio_lanes", no_op_async)
    monkeypatch.setattr(server, "_residency_manager", FakeResidencyManager())
    monkeypatch.setattr(server, "_drain_deferred_prefix_cache_load", no_op_async)
    monkeypatch.setattr(server, "_shutdown_save_prefix_cache", no_op_async)
    try:
        with TestClient(server.app):
            pass
    finally:
        cfg.ready = previous_ready
        cfg.draining = previous_draining

    assert closed == [True]


def test_loop_gate_callback_wiring(client, tmp_path):
    """Regression: CUARun must honor the injected gate callback.

    Before the fix the loop always used the file sentinel, so a GUI/server
    approval POST resolved nothing and the run stalled to a timeout stop.
    """
    from rapid_mlx.cua import loop as cua_loop
    from rapid_mlx.cua.config import resolve_planner

    config = cua_loop.CUAConfig(planner=resolve_planner("local-9b"), human_login=True)
    service_run = cua_service.CUAServiceRun(
        run_id="gatetest", app="Google Chrome", goal="g", config=config
    )

    async def scenario():
        run = cua_loop.CUARun(
            config,
            app="Google Chrome",
            goal="sign in and continue",
            run_dir=tmp_path / "gatetest",
            event_sink=service_run.emit,
            gate=lambda reason: service_run.wait_for_approval(reason, timeout=5.0),
        )

        async def approver():
            await asyncio.sleep(0.05)
            assert service_run.approve() is True

        task = asyncio.create_task(approver())
        approved = await run._request_signin_approval()
        await task
        return approved

    assert asyncio.run(scenario()) is True
    kinds = [e["kind"] for e in service_run.events]
    assert "gate" in kinds and "gate_resolved" in kinds
    assert service_run.status == "running"  # reset after resolution
