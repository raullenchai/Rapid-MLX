from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.agent_runtime import AgentRunStatus, AgentRuntime
from rapid_mlx.agent_runtime.server import (
    AgentRunCreateRequest,
    AgentServerService,
    _ServerRun,
)
from rapid_mlx.cua import service as cua
from rapid_mlx.middleware import activity
from rapid_mlx.middleware.auth import verify_api_key
from rapid_mlx.routes import agents, residency, video
from rapid_mlx.runtime.audio_worker import audio_worker


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(agents, "_service", None)
    monkeypatch.setattr(agents, "_service_shutting_down", False)
    monkeypatch.setattr(activity, "activity_tracker", activity.ActivityTracker())
    monkeypatch.setattr(
        residency, "_manager", lambda: SimpleNamespace(snapshot=lambda: {})
    )
    monkeypatch.setattr(audio_worker, "snapshot", lambda: [])
    monkeypatch.setattr(video, "active_video_jobs", lambda: 0)
    monkeypatch.setattr(cua, "_SERVICE", cua.CUAService())
    app = FastAPI()
    app.include_router(residency.router)
    app.dependency_overrides[verify_api_key] = lambda: None
    with TestClient(app) as test_client:
        yield test_client


def test_no_agent_service_is_created(client):
    assert agents.active_agent_runs() == 0
    assert agents._service is None


@pytest.mark.parametrize(
    "status", [AgentRunStatus.AWAITING_APPROVAL, AgentRunStatus.AWAITING_TOOL_RESULT]
)
def test_residency_counts_parked_agent_run(client, status):
    service = AgentServerService()
    run = AgentRuntime().create_run(model="minicpm5-2b-4bit", goal="Read")
    entry = _ServerRun(
        run=run.model_copy(update={"status": status}),
        request_model="test",
        settings=AgentRunCreateRequest(goal="Read"),
        tools=(),
        registry=service._registry,
        model_generation=None,
        messages=[],
    )
    service._runs[run.id] = entry
    agents._service = service
    result = client.get("/v1/models/residency").json()["activity"]
    assert result["in_flight"] >= 1
    assert result["idle_seconds"] == 0
    entry.run = entry.run.model_copy(update={"status": AgentRunStatus.COMPLETED})
    assert client.get("/v1/models/residency").json()["activity"]["in_flight"] == 0


def test_residency_counts_active_cua_run(client):
    cua._SERVICE._runs["run"] = SimpleNamespace(status="awaiting_approval")
    assert client.get("/v1/models/residency").json()["activity"]["in_flight"] == 1
    cua._SERVICE._runs["run"].status = "stopped"
    assert client.get("/v1/models/residency").json()["activity"]["in_flight"] == 0
