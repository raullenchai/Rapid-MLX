import json
import sys

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx import cli
from rapid_mlx.agent_runtime.server import (
    AgentRunCreateRequest,
    AgentServerService,
    _ServerRun,
)
from rapid_mlx.config import get_config
from rapid_mlx.control import RUNNER_ENV, SECRET_ENV, SOCKET_ENV, protocol
from rapid_mlx.routes import agents
from rapid_mlx.routes.control_status import agent_jobs, mount_status


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setattr(get_config(), "api_key", "key")
    monkeypatch.setattr(agents, "_service", None)
    monkeypatch.setattr(agents, "_service_shutting_down", False)
    monkeypatch.delenv(RUNNER_ENV, raising=False)
    mount_status(bare := FastAPI())
    assert TestClient(bare).get("/rapid/v1/jobs").status_code == 404
    monkeypatch.setenv(RUNNER_ENV, "1")
    monkeypatch.setenv(SOCKET_ENV, "/unused")
    monkeypatch.setenv(SECRET_ENV, "unused")
    mount_status(app := FastAPI())
    client = TestClient(app, client=("127.0.0.1", 1))
    client.headers["Authorization"] = "Bearer key"
    assert client.get("/rapid/v1/jobs").json() == {"jobs": []}
    assert agents._service is None
    service = AgentServerService()
    monkeypatch.setattr(agents, "_service", service)
    run = service._runtime.create_run(model="model", goal="Summarize\n  this")
    service._runs[run.id] = _ServerRun(
        run=run,
        request_model="model",
        settings=AgentRunCreateRequest(goal=run.goal),
        tools=(),
        registry=service._registry,
        model_generation=None,
        messages=[],
    )
    return client, service, run


def test_routes(local, monkeypatch):
    client, service, run = local
    row = client.get("/rapid/v1/jobs").json()["jobs"][0]
    assert row == dict(
        id=run.id,
        kind="agent",
        status="ready",
        created=run.events[0].created_at,
        updated=run.events[-1].created_at,
        summary="Summarize this",
    )
    for method, path in (
        ("get", "/rapid/v1/jobs"),
        ("post", f"/rapid/v1/jobs/{run.id}/cancel"),
    ):
        for peer, key, code in (("192.0.2.1", "key", 403), ("127.0.0.1", "", 401)):
            other = TestClient(client.app, client=(peer, 1))
            assert (
                getattr(other, method)(
                    path, headers={"Authorization": f"Bearer {key}"}
                ).status_code
                == code
            )
    with monkeypatch.context() as patch:
        patch.setattr(service, "list_runs", lambda: pytest.fail("cancel listed runs"))
        response = client.post("/rapid/v1/jobs/unknown/cancel")
        assert response.status_code == 404
        assert response.json()["detail"] == "agent run was not found or has expired"
    response = client.post(f"/rapid/v1/jobs/{run.id}/cancel")
    assert response.json()["status"] == "cancelled"
    row = client.get("/rapid/v1/jobs").json()["jobs"][0]
    assert row["status"] == "cancelled" and row["updated"] >= row["created"]
    monkeypatch.setattr(service, "_monotonic", lambda: 1000)
    service._runs[run.id].terminal_mono = 101
    assert service.list_runs()[0]["goal"] == run.goal
    service._runs[run.id].terminal_mono = 100
    assert agent_jobs() == [] and run.id in service._runs
    assert client.post(f"/rapid/v1/jobs/{run.id}/cancel").status_code == 404


def test_cli(local, monkeypatch, capsys):
    client, service, run = local
    monkeypatch.setenv("RAPID_MLX_API_KEY", "key")
    monkeypatch.setattr(protocol, "discover", lambda port: [{"port": port or 8000}])
    monkeypatch.setattr(protocol.time, "time", lambda: run.events[0].created_at + 12)

    def transport(method):
        def send(url, **kw):
            assert kw == dict(
                headers={"Authorization": "Bearer key"}, trust_env=False, timeout=2
            )
            assert url.startswith("http://127.0.0.1:8123/")
            return getattr(client, method)(url)

        return send

    monkeypatch.setattr(httpx, "get", transport("get"))
    monkeypatch.setattr(httpx, "post", transport("post"))
    for flags in (
        ["--port", "8123"],
        ["--json", "--port", "8123"],
        ["cancel", run.id, "--port", "8123"],
    ):
        monkeypatch.setattr(sys, "argv", ["rapid-mlx", "jobs", *flags])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        output = capsys.readouterr().out
        if flags[0] == "--port":
            header, row = output.splitlines()
            assert header.split() == ["ID", "KIND", "STATUS", "AGE", "SUMMARY"]
            assert row.index("agent") == header.index("KIND")
            assert row.endswith("12s  Summarize this")
        else:
            assert json.loads(output)
    service._runs[run.id].run = run.model_copy(update={"events": ()})
    row = client.get("/rapid/v1/jobs").json()["jobs"][0]
    assert row["created"] is None and row["updated"] is None
    args = cli.build_parser().parse_args(["jobs", "--port", "8123"])
    assert protocol.jobs_command(args) == 0
    assert capsys.readouterr().out.splitlines()[1].endswith("-  Summarize this")
    monkeypatch.setattr(protocol, "discover", lambda port: [])
    args = cli.build_parser().parse_args(["jobs"])
    assert protocol.jobs_command(args) == 1
    assert "No running serve" in capsys.readouterr().err
