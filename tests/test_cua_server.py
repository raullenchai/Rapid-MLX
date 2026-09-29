"""Tests for the server-side CUA surface (/v1/cua/*).

The router is mounted on a bare FastAPI app; the backend (AX) and the run
loop are stubbed so no computer access or HTTP planner calls happen.
"""

from __future__ import annotations

import asyncio
import threading

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
        backend_mod,
        "permissions",
        lambda: {
            "accessibility": True,
            "screen_recording": False,
            "hints": ["Grant Screen Recording."],
        },
    )
    monkeypatch.setattr(
        backend_mod,
        "list_apps",
        lambda: [{"name": "Finder", "bundleId": "com.apple.finder", "pid": 42}],
    )
    monkeypatch.setattr(
        backend_mod,
        "list_windows",
        lambda app: [
            {
                "window_id": "cg:123",
                "index": 0,
                "title": f"{app} window",
                "x": 1,
                "y": 2,
                "width": 3,
                "height": 4,
            }
        ],
    )
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
    assert (
        test_client.get(
            "/v1/cua/runs", headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )


@pytest.mark.parametrize("api_key", [None, ""])
def test_cua_routes_fail_closed_without_server_api_key(client, api_key):
    cfg = get_config()
    cfg.api_key = api_key
    requests = (
        ("get", "/v1/cua/planners"),
        ("get", "/v1/cua/capabilities"),
        ("get", "/v1/cua/permissions"),
        ("get", "/v1/cua/apps"),
        ("get", "/v1/cua/apps/Finder/windows"),
        ("post", "/v1/cua/observations"),
        ("post", "/v1/cua/planners"),
        ("delete", "/v1/cua/planners/custom"),
        ("get", "/v1/cua/runs"),
        ("get", "/v1/cua/runs/by-request/unknown"),
        ("post", "/v1/cua/runs"),
        ("get", "/v1/cua/runs/unknown"),
        ("get", "/v1/cua/runs/unknown/events"),
        ("post", "/v1/cua/runs/unknown/approval"),
        ("post", "/v1/cua/runs/unknown/cancel"),
    )
    for method, path in requests:
        for headers in ({}, AUTH):
            response = getattr(client, method)(path, headers=headers)
            assert response.status_code == 503, (method, path, headers)
    assert client.fresh_service.list_runs() == []


def test_list_planners(client):
    test_client = client
    response = test_client.get("/v1/cua/planners", headers=AUTH)
    assert response.status_code == 200
    names = {p["name"] for p in response.json()}
    assert {"local-27b", "local-9b"} <= names


def test_discovery_contract(client):
    capabilities = client.get("/v1/cua/capabilities", headers=AUTH)
    assert capabilities.status_code == 200
    assert capabilities.json()["run_operations"] == [
        "create",
        "poll",
        "approve",
        "deny",
        "cancel",
    ]
    assert capabilities.json()["features"] == {
        "app_discovery": True,
        "window_discovery": True,
        "window_selection": True,
        "visual_observation": cua_routes.sys.platform == "darwin",
        "screenshot_observation": False,
        "observation_without_activation": cua_routes.sys.platform == "darwin",
        "approval_gate_id": True,
        "idempotent_run_create": True,
        "multi_target_runs": True,
        "switch_target": True,
    }
    assert capabilities.json()["protocol_version"] == 2
    assert capabilities.json()["max_run_targets"] == 3

    permissions = client.get("/v1/cua/permissions", headers=AUTH)
    assert permissions.json()["accessibility"] is True
    assert permissions.json()["screen_recording"] is False

    apps = client.get("/v1/cua/apps", headers=AUTH)
    assert apps.json() == [
        {"name": "Finder", "bundle_id": "com.apple.finder", "pid": 42}
    ]
    windows = client.get("/v1/cua/apps/Finder/windows", headers=AUTH)
    assert windows.json()[0]["window_id"] == "cg:123"
    assert windows.json()[0]["title"] == "Finder window"

    event_schema = client.app.openapi()["components"]["schemas"]["CUAEvent"]
    assert "target" in event_schema["properties"]


def test_create_run_freezes_selected_window_and_rejects_open_url(client, monkeypatch):
    from rapid_mlx.computer_use import backend as backend_mod

    observed: list[tuple[str, str | None]] = []

    def validate_window(app, window_id):
        observed.append((app, window_id))
        return {
            "app": {"name": app, "pid": 42},
            "window_id": "cg:123",
            "window": {"window_id": "cg:123", "index": 0},
        }

    monkeypatch.setattr(backend_mod, "validate_window", validate_window)
    run_kwargs: dict = {}

    async def capture_run(config, app, goal, **kwargs):
        run_kwargs.update(kwargs)
        return {"status": "done", "final_summary": "done"}

    monkeypatch.setattr(cua_service, "run_loop", capture_run)
    response = _post_run(client, window_id="opaque-client-id")
    assert response.status_code == 202
    assert response.json()["window_id"] == "cg:123"
    run_id = response.json()["run_id"]
    view = client.get(f"/v1/cua/runs/{run_id}", headers=AUTH).json()
    assert view["window_id"] == "cg:123"
    assert observed == [("Google Chrome", "opaque-client-id")]
    assert run_kwargs["backend_app"] == "pid:42"
    assert run_kwargs["expected_app"]["pid"] == 42

    rejected = _post_run(client, window_id="cg:123", open_url="https://example.com")
    assert rejected.status_code == 400
    assert "cannot be used" in rejected.json()["detail"]


def test_multi_target_create_freezes_and_echoes_full_authority(client, monkeypatch):
    from rapid_mlx.computer_use import backend as backend_mod

    observed = []

    def validate_window(app, window_id):
        observed.append((app, window_id))
        pid = int(app.removeprefix("pid:"))
        bundle = "com.apple.Safari" if pid == 42 else "com.apple.TextEdit"
        return {
            "app": {
                "name": "Safari" if pid == 42 else "TextEdit",
                "bundleId": bundle,
                "pid": pid,
            },
            "window_id": window_id,
            "window": {"window_id": window_id, "index": 0},
        }

    monkeypatch.setattr(backend_mod, "validate_window", validate_window)
    run_kwargs = {}

    async def capture_run(config, app, goal, **kwargs):
        run_kwargs.update(kwargs)
        return {"status": "done", "final_summary": "done"}

    monkeypatch.setattr(cua_service, "run_loop", capture_run)
    payload = {
        "app": "pid:42",
        "goal": "read then edit",
        "client_request_id": "multi-1",
        "initial_target_id": "web",
        "targets": [
            {
                "target_id": "web",
                "app": "pid:42",
                "pid": 42,
                "window_id": "cg:1",
                "allowed_domain": "example.com",
            },
            {
                "target_id": "notes",
                "app": "pid:43",
                "pid": 43,
                "window_id": "cg:2",
                "allowed_domain": "",
            },
        ],
    }
    created = client.post("/v1/cua/runs", headers=AUTH, json=payload)
    assert created.status_code == 202, created.text
    body = created.json()
    assert body["active_target_id"] == "web"
    assert body["targets"] == payload["targets"]
    assert observed == [("pid:42", "cg:1"), ("pid:43", "cg:2")]
    assert client.fresh_service.get(body["run_id"]).active_target_id == "web"

    replay = client.post("/v1/cua/runs", headers=AUTH, json=payload)
    recovered = client.get("/v1/cua/runs/by-request/multi-1", headers=AUTH)
    assert replay.json()["run_id"] == body["run_id"]
    assert replay.json()["targets"] == body["targets"]
    assert replay.json()["active_target_id"] == body["active_target_id"]
    assert recovered.json()["targets"] == payload["targets"]

    reordered = dict(payload)
    reordered["targets"] = list(reversed(payload["targets"]))
    conflict = client.post("/v1/cua/runs", headers=AUTH, json=reordered)
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "request_identity_conflict"


def test_multi_target_request_rejects_ambiguous_or_unscoped_authority(client):
    base = {
        "app": "pid:42",
        "goal": "g",
        "initial_target_id": "web",
        "targets": [
            {
                "target_id": "web",
                "app": "pid:42",
                "pid": 42,
                "window_id": "cg:1",
                "allowed_domain": "example.com",
            },
            {
                "target_id": "notes",
                "app": "pid:43",
                "pid": 43,
                "window_id": "cg:2",
                "allowed_domain": "",
            },
        ],
    }
    mixed = {**base, "window_id": "cg:1"}
    assert client.post("/v1/cua/runs", headers=AUTH, json=mixed).status_code == 422
    mismatched = {
        **base,
        "targets": [dict(base["targets"][0], app="pid:99"), base["targets"][1]],
    }
    assert client.post("/v1/cua/runs", headers=AUTH, json=mismatched).status_code == 422


def test_multi_target_gate_rejects_decision_after_target_change(client):
    run = cua_service.CUAServiceRun(
        run_id="r",
        app="pid:42",
        goal="g",
        config=None,
        window_id="cg:1",
        targets=[
            {
                "target_id": "web",
                "app": "pid:42",
                "pid": 42,
                "window_id": "cg:1",
                "allowed_domain": "example.com",
            },
            {
                "target_id": "notes",
                "app": "pid:43",
                "pid": 43,
                "window_id": "cg:2",
                "allowed_domain": "",
            },
        ],
        active_target_id="web",
    )
    run.emit({"kind": "gate", "reason": "external_commit", "target_id": "web"})
    gate_id = run.view()["pending_gate"]["gate_id"]
    run.emit(
        {
            "kind": "target_switched",
            "from_target_id": "web",
            "target_id": "notes",
        }
    )
    with pytest.raises(cua_service.CUAGateMismatchError):
        run.resolve_gate(True, gate_id=gate_id)
    assert not run._approve_event.is_set()
    run.emit({"kind": "terminal", "status": "stopped"})
    assert run.events[-1]["target_id"] == "notes"


def test_multi_target_browser_requires_reviewed_domain(client, monkeypatch):
    from rapid_mlx.computer_use import backend as backend_mod

    def validate_window(app, window_id):
        pid = int(app.removeprefix("pid:"))
        return {
            "app": {
                "name": "Safari" if pid == 42 else "TextEdit",
                "bundleId": "com.apple.Safari" if pid == 42 else "com.apple.TextEdit",
                "pid": pid,
            },
            "window_id": window_id,
        }

    monkeypatch.setattr(backend_mod, "validate_window", validate_window)
    response = client.post(
        "/v1/cua/runs",
        headers=AUTH,
        json={
            "app": "pid:42",
            "goal": "g",
            "initial_target_id": "web",
            "targets": [
                {
                    "target_id": "web",
                    "app": "pid:42",
                    "pid": 42,
                    "window_id": "cg:1",
                    "allowed_domain": "",
                },
                {
                    "target_id": "notes",
                    "app": "pid:43",
                    "pid": 43,
                    "window_id": "cg:2",
                    "allowed_domain": "",
                },
            ],
        },
    )
    assert response.status_code == 400
    assert "requires allowed_domain" in response.json()["detail"]


def test_idempotent_create_replays_one_run_and_lookup_is_authenticated(client):
    request_id = "desktop%launch:42"
    first = _post_run(client, client_request_id=request_id)
    replay = _post_run(
        client,
        app="  Google Chrome  ",
        goal="  open the article  ",
        client_request_id=request_id,
    )

    assert first.status_code == replay.status_code == 202
    assert first.json()["run_id"] == replay.json()["run_id"]
    assert replay.json()["status"] in {"running", "completed"}
    assert first.json()["client_request_id"] == request_id
    assert len(client.fresh_service._runs) == 1

    path = "/v1/cua/runs/by-request/desktop%25launch%3A42"
    assert client.get(path).status_code == 401
    recovered = client.get(path, headers=AUTH)
    assert recovered.status_code == 200
    assert recovered.json()["run_id"] == first.json()["run_id"]
    assert recovered.json()["client_request_id"] == request_id


def test_request_identity_conflict_and_typed_lookup_miss(client):
    created = _post_run(client, client_request_id="same-id")
    assert created.status_code == 202

    conflict = _post_run(client, client_request_id="same-id", goal="a different task")
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "request_identity_conflict"
    assert len(client.fresh_service._runs) == 1

    missing = client.get("/v1/cua/runs/by-request/missing", headers=AUTH)
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "request_identity_not_found"

    too_long = _post_run(client, client_request_id="x" * 129)
    assert too_long.status_code == 422


def test_request_identity_is_one_url_path_segment(client):
    rejected = _post_run(client, client_request_id="desktop/run-1")
    assert rejected.status_code == 422
    assert client.fresh_service.list_runs() == []

    # Starlette decodes %2F before matching route segments. Neither an encoded
    # nor a literal slash may enter the by-request handler.
    encoded = client.get("/v1/cua/runs/by-request/desktop%2Frun-1", headers=AUTH)
    literal = client.get("/v1/cua/runs/by-request/desktop/run-1", headers=AUTH)
    assert encoded.status_code == 404
    assert literal.status_code == 404


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
    assert "run_dir" not in view
    assert all("run_dir" not in event for event in view["events"])

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


def test_stalled_completion_disposition_never_counts_as_completed(client, monkeypatch):
    from rapid_mlx.cua import service as cua_service

    summary = "The edit succeeded, but persistence could not be verified."

    async def partial_run(*args, **kwargs):
        sink = kwargs.get("event_sink")
        if sink is not None:
            sink(
                {
                    "kind": "terminal",
                    "status": "stalled",
                    "reason": summary,
                    "final_summary": summary,
                }
            )
        return {
            "status": "stalled",
            "final_summary": summary,
            "completion_disposition": "partial",
        }

    monkeypatch.setattr(cua_service, "run_loop", partial_run)
    created = _post_run(client)
    run_id = created.json()["run_id"]

    import time

    deadline = time.time() + 5
    while time.time() < deadline:
        view = client.get(f"/v1/cua/runs/{run_id}", headers=AUTH).json()
        if view["status"] != "running":
            break
        time.sleep(0.05)

    assert view["status"] == "stalled"
    assert view["final_summary"] == summary
    assert view["events"][-1]["status"] == "stalled"


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

    not_waiting = test_client.post(
        "/v1/cua/runs/none/approval",
        headers=AUTH,
        json={"gate_id": "missing", "approved": True},
    )
    assert not_waiting.status_code == 404

    async def scenario():
        task = asyncio.create_task(active.wait_for_approval("sign-in", timeout=5))
        await asyncio.sleep(0.05)
        assert active.approve() is True
        return await task

    assert asyncio.run(scenario()) is True
    # loop-level rule: approve only resolves a waiting gate
    assert active.approve() is False
    conflict = test_client.post(
        "/v1/cua/runs/gate1/approval",
        headers=AUTH,
        json={"gate_id": "expired", "approved": True},
    )
    assert conflict.status_code == 409

    active.emit({"kind": "gate", "reason": "sign-in"})
    gate_id = active.view()["pending_gate"]["gate_id"]
    approved = test_client.post(
        "/v1/cua/runs/gate1/approval",
        headers=AUTH,
        json={"gate_id": gate_id, "approved": True},
    )
    assert approved.status_code == 200
    assert approved.json() == {"run_id": "gate1", "approved": True}


def test_gate_is_visible_and_can_be_denied(client):
    active = cua_service.CUAServiceRun(
        run_id="gate-deny",
        app="A",
        goal="g",
        config=None,  # type: ignore[arg-type]
    )
    client.fresh_service._runs[active.run_id] = active

    async def scenario():
        waiter = asyncio.create_task(active.wait_for_approval("sign in", timeout=5))
        await asyncio.sleep(0.01)
        view = client.get(f"/v1/cua/runs/{active.run_id}", headers=AUTH).json()
        assert view["pending_gate"]["reason"] == "sign in"
        gate_id = view["pending_gate"]["gate_id"]
        denied = client.post(
            f"/v1/cua/runs/{active.run_id}/approval",
            headers=AUTH,
            json={"gate_id": gate_id, "approved": False},
        )
        assert denied.json() == {"run_id": active.run_id, "approved": False}
        return await waiter

    assert asyncio.run(scenario()) is False


def test_stale_gate_decision_cannot_resolve_current_gate(client):
    active = cua_service.CUAServiceRun(
        run_id="gate-stale",
        app="A",
        goal="g",
        config=None,  # type: ignore[arg-type]
    )
    client.fresh_service._runs[active.run_id] = active

    async def scenario():
        waiter = asyncio.create_task(active.wait_for_approval("sign in", timeout=5))
        await asyncio.sleep(0.01)
        current = active.view()["pending_gate"]
        stale = client.post(
            f"/v1/cua/runs/{active.run_id}/approval",
            headers=AUTH,
            json={"gate_id": "previous-gate", "approved": True},
        )
        assert stale.status_code == 409
        assert active.view()["pending_gate"]["gate_id"] == current["gate_id"]
        assert not active._approve_event.is_set()
        resolved = client.post(
            f"/v1/cua/runs/{active.run_id}/approval",
            headers=AUTH,
            json={"gate_id": current["gate_id"], "approved": False},
        )
        assert resolved.status_code == 200
        return await waiter

    assert asyncio.run(scenario()) is False


def test_gate_decision_is_idempotent_and_first_decision_wins(client):
    active = cua_service.CUAServiceRun(
        run_id="gate-once",
        app="A",
        goal="g",
        config=None,  # type: ignore[arg-type]
    )
    client.fresh_service._runs[active.run_id] = active

    async def scenario():
        waiter = asyncio.create_task(active.wait_for_approval("sign in", timeout=5))
        await asyncio.sleep(0.01)
        gate_id = active.view()["pending_gate"]["gate_id"]
        path = f"/v1/cua/runs/{active.run_id}/approval"
        decision = {"gate_id": gate_id, "approved": True}
        assert client.post(path, headers=AUTH, json=decision).status_code == 200
        assert client.post(path, headers=AUTH, json=decision).status_code == 200
        conflict = client.post(
            path,
            headers=AUTH,
            json={"gate_id": gate_id, "approved": False},
        )
        assert conflict.status_code == 409
        assert active.view()["pending_gate"]["approved"] is True
        return await waiter

    assert asyncio.run(scenario()) is True


def test_bodyless_approval_cannot_replay_across_gates(client):
    active = cua_service.CUAServiceRun(
        run_id="gate-replay",
        app="A",
        goal="g",
        config=None,  # type: ignore[arg-type]
    )
    client.fresh_service._runs[active.run_id] = active
    active.emit({"kind": "gate", "reason": "gate-a"})
    gate_a = active.view()["pending_gate"]["gate_id"]
    assert active.resolve_gate(True, gate_id=gate_a) is True

    active.emit({"kind": "gate", "reason": "gate-b"})
    gate_b = active.view()["pending_gate"]["gate_id"]
    assert gate_b != gate_a

    replay = client.post(f"/v1/cua/runs/{active.run_id}/approval", headers=AUTH)
    assert replay.status_code == 422
    assert active.view()["pending_gate"]["gate_id"] == gate_b
    assert not active._approve_event.is_set()


def test_gate_events_share_id_and_fast_decision_is_not_lost(client):
    active = cua_service.CUAServiceRun(
        run_id="gate-fast",
        app="A",
        goal="g",
        config=None,  # type: ignore[arg-type]
    )
    client.fresh_service._runs[active.run_id] = active
    active.emit(
        {
            "kind": "gate",
            "reason": "external_commit",
            "action": "click",
            "target": "Send",
        }
    )
    gate_event = active.events[-1]
    gate_id = gate_event["gate_id"]
    assert active.view()["pending_gate"]["target"] == "Send"
    http_gate = client.get(f"/v1/cua/runs/{active.run_id}", headers=AUTH).json()[
        "pending_gate"
    ]
    assert http_gate["gate_id"] == gate_id
    assert http_gate["action"] == "click"
    assert http_gate["target"] == "Send"

    # A custom GUI can decide as soon as the gate event is visible, before the
    # loop coroutine enters wait_for_approval.
    assert active.resolve_gate(True, gate_id=gate_id) is True

    async def scenario():
        assert await active.wait_for_approval("external_commit", timeout=5) is True
        active.emit({"kind": "gate_resolved", "approved": True})

    asyncio.run(scenario())
    relevant = [
        event
        for event in active.events
        if event["kind"] in {"gate", "gate_detail", "gate_resolved"}
    ]
    assert [event["gate_id"] for event in relevant] == [gate_id, gate_id, gate_id]


def test_approval_timeout(client):
    active = cua_service.CUAServiceRun(
        run_id="timeout1",
        app="A",
        goal="g",
        config=None,  # type: ignore[arg-type]
    )
    assert asyncio.run(active.wait_for_approval("sign-in", timeout=0.001)) is False


def test_late_approval_event_cannot_approve_the_next_gate(client):
    active = cua_service.CUAServiceRun(
        run_id="timeout-replay",
        app="A",
        goal="g",
        config=None,  # type: ignore[arg-type]
    )

    async def scenario():
        active.emit({"kind": "gate", "reason": "first"})
        assert await active.wait_for_approval("first", timeout=0.001) is False
        expired_event = active._approve_event
        expired_event.set()
        active.emit({"kind": "gate", "reason": "second"})
        return await active.wait_for_approval("second", timeout=0.001)

    assert asyncio.run(scenario()) is False


def test_fast_approval_after_gate_emit_is_not_lost(client):
    active = cua_service.CUAServiceRun(
        run_id="fast-approval",
        app="A",
        goal="g",
        config=None,  # type: ignore[arg-type]
    )
    active.emit({"kind": "gate", "reason": "external_commit"})
    assert active.approve() is True
    assert (
        asyncio.run(active.wait_for_approval("external_commit", timeout=0.01)) is True
    )


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
    assert view["events_after_seq"] == 3

    # Feeding the returned cursor back never redelivers an event.
    next_view = test_client.get(
        "/v1/cua/runs/ev1/events",
        headers=AUTH,
        params={"after": view["events_after_seq"]},
    ).json()
    assert next_view["events"] == []
    assert next_view["events_after_seq"] == 3
    active.emit({"kind": "plan", "step": 2})
    fresh_view = test_client.get(
        "/v1/cua/runs/ev1/events",
        headers=AUTH,
        params={"after": next_view["events_after_seq"]},
    ).json()
    assert [event["seq"] for event in fresh_view["events"]] == [4]
    assert fresh_view["events_after_seq"] == 4

    # A persisted or corrupt cursor beyond the service's retained tail clamps
    # to the current tail so future events remain pollable.
    excessive = test_client.get(
        "/v1/cua/runs/ev1/events", headers=AUTH, params={"after": 999}
    ).json()
    assert excessive["events"] == []
    assert excessive["events_after_seq"] == 4
    active.emit({"kind": "executed", "step": 2})
    recovered = test_client.get(
        "/v1/cua/runs/ev1/events",
        headers=AUTH,
        params={"after": excessive["events_after_seq"]},
    ).json()
    assert [event["seq"] for event in recovered["events"]] == [5]
    assert recovered["events_after_seq"] == 5

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


def test_create_lock_serializes_validation_and_starts_one_task(client, monkeypatch):
    from rapid_mlx.computer_use import backend as backend_mod

    service = client.fresh_service
    validation_started = threading.Event()
    release_validation = threading.Event()
    loop_started = 0

    def delayed_validate(app, window_id):
        validation_started.set()
        assert release_validation.wait(timeout=5)
        return {
            "app": {"name": app, "pid": 42},
            "window_id": window_id,
            "window": {"window_id": window_id, "index": 0},
        }

    async def blocked_run(*args, **kwargs):
        nonlocal loop_started
        loop_started += 1
        await asyncio.Event().wait()

    monkeypatch.setattr(backend_mod, "validate_window", delayed_validate)
    monkeypatch.setattr(cua_service, "run_loop", blocked_run)

    async def scenario():
        first = asyncio.create_task(
            service.create(
                app="pid:42",
                goal="first",
                planner="local-9b",
                window_id="cg:1",
                client_request_id="request-a",
            )
        )
        assert await asyncio.to_thread(validation_started.wait, 5)
        second = asyncio.create_task(
            service.create(
                app="pid:42",
                goal="second",
                planner="local-9b",
                window_id="cg:2",
                client_request_id="request-b",
            )
        )
        await asyncio.sleep(0.01)
        release_validation.set()
        run = await first
        with pytest.raises(cua_service.CUARunConflictError):
            await second
        await asyncio.sleep(0)
        assert len(service._tasks) == 1
        assert loop_started == 1
        service.cancel(run.run_id)
        await asyncio.sleep(0)

    asyncio.run(scenario())


def test_concurrent_same_identity_validates_and_starts_once(client, monkeypatch):
    from rapid_mlx.computer_use import backend as backend_mod

    service = client.fresh_service
    validation_started = threading.Event()
    release_validation = threading.Event()
    validation_calls = 0
    loop_started = 0

    def delayed_validate(app, window_id):
        nonlocal validation_calls
        validation_calls += 1
        validation_started.set()
        assert release_validation.wait(timeout=5)
        return {
            "app": {"name": app, "pid": 42},
            "window_id": window_id,
            "window": {"window_id": window_id, "index": 0},
        }

    async def blocked_run(*args, **kwargs):
        nonlocal loop_started
        loop_started += 1
        await asyncio.Event().wait()

    monkeypatch.setattr(backend_mod, "validate_window", delayed_validate)
    monkeypatch.setattr(cua_service, "run_loop", blocked_run)

    async def scenario():
        kwargs = {
            "app": "pid:42",
            "goal": "same task",
            "planner": "local-9b",
            "window_id": "cg:1",
            "client_request_id": "same-concurrent-request",
        }
        first = asyncio.create_task(service.create(**kwargs))
        assert await asyncio.to_thread(validation_started.wait, 5)
        replay = asyncio.create_task(service.create(**kwargs))
        await asyncio.sleep(0.01)
        release_validation.set()
        first_run, replayed_run = await asyncio.gather(first, replay)
        await asyncio.sleep(0)
        assert replayed_run is first_run
        assert validation_calls == 1
        assert loop_started == 1
        service.cancel(first_run.run_id)
        await asyncio.sleep(0)

    asyncio.run(scenario())


def test_cancelled_create_before_commit_leaves_no_identity_or_task(client, monkeypatch):
    from rapid_mlx.computer_use import backend as backend_mod

    service = client.fresh_service
    validation_started = threading.Event()
    release_validation = threading.Event()

    def delayed_validate(app, window_id):
        validation_started.set()
        assert release_validation.wait(timeout=5)
        return {
            "app": {"name": app, "pid": 42},
            "window_id": window_id,
            "window": {"window_id": window_id, "index": 0},
        }

    monkeypatch.setattr(backend_mod, "validate_window", delayed_validate)

    async def scenario():
        create = asyncio.create_task(
            service.create(
                app="pid:42",
                goal="cancel before commit",
                planner="local-9b",
                window_id="cg:1",
                client_request_id="cancelled-before-commit",
            )
        )
        assert await asyncio.to_thread(validation_started.wait, 5)
        create.cancel()
        with pytest.raises(asyncio.CancelledError):
            await create
        release_validation.set()
        await asyncio.sleep(0.01)

    asyncio.run(scenario())
    assert service._runs == {}
    assert service._tasks == {}
    with pytest.raises(cua_service.CUARequestIdentityNotFoundError):
        service.get_by_request_id("cancelled-before-commit")


def test_response_loss_after_commit_is_recoverable_without_second_task(
    client, monkeypatch
):
    service = client.fresh_service
    loop_started = 0

    async def blocked_run(*args, **kwargs):
        nonlocal loop_started
        loop_started += 1
        await asyncio.Event().wait()

    monkeypatch.setattr(cua_service, "run_loop", blocked_run)

    async def scenario():
        committed = asyncio.Event()

        async def handler_that_loses_response():
            run = await service.create(
                app="Chrome",
                goal="recover me",
                planner="local-9b",
                client_request_id="response-lost",
            )
            committed.set()
            await asyncio.Event().wait()
            return run

        handler = asyncio.create_task(handler_that_loses_response())
        await committed.wait()
        handler.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handler
        recovered = service.get_by_request_id("response-lost")
        replay = await service.create(
            app="Chrome",
            goal="recover me",
            planner="local-9b",
            client_request_id="response-lost",
        )
        assert replay is recovered
        assert loop_started == 1
        service.cancel(recovered.run_id)
        await asyncio.sleep(0)

    asyncio.run(scenario())


def test_request_identity_retention_matches_run_retention(client, monkeypatch):
    service = client.fresh_service
    monkeypatch.setattr(cua_service, "MAX_RETAINED_RUNS", 1)

    async def completed_run(*args, **kwargs):
        return {"status": "done", "final_summary": "done"}

    monkeypatch.setattr(cua_service, "run_loop", completed_run)

    async def scenario():
        first = await service.create(
            app="Chrome",
            goal="first",
            planner="local-9b",
            client_request_id="retained-first",
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert service.get_by_request_id("retained-first") is first

        second = await service.create(
            app="Chrome",
            goal="second",
            planner="local-9b",
            client_request_id="retained-second",
        )
        assert service.get_by_request_id("retained-second") is second
        with pytest.raises(cua_service.CUARequestIdentityNotFoundError):
            service.get_by_request_id("retained-first")
        await asyncio.sleep(0)

    asyncio.run(scenario())


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
