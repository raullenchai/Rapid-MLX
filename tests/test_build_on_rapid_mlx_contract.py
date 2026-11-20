"""Executable stability promises in the tool-author guide (#5886)."""

import ast
import asyncio
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from rapid_mlx.cli import build_parser
from rapid_mlx.routes import health, residency
from tests.test_metrics_route import metrics_client as _metrics_client
from tests.test_resident_models import (
    FakeImageEngine,
    entry,
    manager_fixture,
)
from tests.test_resident_models import (
    residency_device_memory as _residency_device_memory,
)

metrics_client = _metrics_client
residency_device_memory = _residency_device_memory

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/guides/build-on-rapid-mlx.md"
ENDPOINTS = {
    "GET /v1/models/residency",
    "POST /v1/models/load",
    "DELETE /v1/models/{id}",
    "GET /v1/status",
    "GET /healthz",
    "GET /health/ready",
    "GET /metrics",
}


def sections():
    text = GUIDE.read_text()
    return dict(
        re.findall(r"### Stable: `([^`]+)`\n(.*?)(?=\n### |\n## |\Z)", text, re.S)
    )


def assert_shape(actual, example):
    """Examples promise a subset of fields; additive fields remain compatible."""
    assert type(actual) is type(example)
    if isinstance(example, dict):
        for key, value in example.items():
            assert key in actual
            assert_shape(actual[key], value)
    elif isinstance(example, list):
        assert actual  # Never let an empty model list hide a changed row shape.
        for item in actual:
            assert_shape(item, example[0])


def test_stable_inventory_and_readme_link():
    assert set(sections()) == ENDPOINTS
    assert (
        "[Build on Rapid-MLX](docs/guides/build-on-rapid-mlx.md)"
        in (ROOT / "README.md").read_text()
    )
    text = GUIDE.read_text()
    assert "`ps --json`" in text
    assert "coming in the next release" in text
    for section in sections().values():
        assert "```bash\ncurl " in section
        assert "```python\n" in section
    for target in re.findall(r"\]\(([^)]+)\)", text):
        assert (GUIDE.parent / target.split("#")[0]).is_file(), target


@pytest.mark.parametrize(
    "scenario",
    [
        "existing_ready",
        "existing_starting",
        "existing_timeout",
        "foreign_listener",
        "probe_timeout",
        "child_ready",
        "child_exited",
        "child_timeout",
        "child_kill",
        "interrupted",
    ],
)
def test_documented_child_lifecycle(scenario, monkeypatch):
    text = GUIDE.read_text()
    lifecycle = text.split("## Run a background server", 1)[1].split("\n## ", 1)[0]
    block = re.search(r"```python\n(.*?)\n```", lifecycle, re.S)[1]
    assert "subprocess.Popen" in block
    assert text.index("## Always-on Mac (system daemon)") > text.index(
        "## Observe the server"
    )
    request = httpx.Request("GET", "http://127.0.0.1:8000/health/ready")
    ready = httpx.Response(200, json={"ready": True}, request=request)
    starting = httpx.Response(503, json={"detail": "model loading"}, request=request)
    refused = httpx.ConnectError("refused", request=request)
    scenarios = {
        "existing_ready": ([ready, ready], None),
        "existing_starting": ([starting, starting, ready], None),
        "existing_timeout": ([starting], TimeoutError),
        "foreign_listener": (
            [httpx.Response(404, request=request)],
            httpx.HTTPStatusError,
        ),
        "probe_timeout": (
            [httpx.ReadTimeout("timed out", request=request)],
            httpx.ReadTimeout,
        ),
        "child_ready": ([refused, refused, starting, ready], None),
        "child_exited": ([refused], RuntimeError),
        "child_timeout": ([refused, starting], TimeoutError),
        "child_kill": ([refused, starting], TimeoutError),
        "interrupted": ([refused, KeyboardInterrupt()], KeyboardInterrupt),
    }
    responses, error = scenarios[scenario]

    def get(path):
        assert path == "/health/ready"
        result = responses.pop(0) if len(responses) > 1 else responses[0]
        if isinstance(result, BaseException):
            raise result
        return result

    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.get.side_effect = get
    monkeypatch.setattr(httpx, "Client", Mock(return_value=client))
    child = Mock()
    child.poll.return_value = 1 if scenario == "child_exited" else None
    if scenario == "child_kill":
        child.wait.side_effect = [subprocess.TimeoutExpired("rapid-mlx", 30), 0]
    popen = Mock(return_value=child)
    monkeypatch.setattr(subprocess, "Popen", popen)
    now = 0.0

    def sleep(seconds):
        nonlocal now
        now += seconds

    monkeypatch.setattr(time, "monotonic", lambda: now)
    monkeypatch.setattr(time, "sleep", sleep)
    if error:
        with pytest.raises(error):
            exec(compile(block, str(GUIDE), "exec"), {})
    else:
        exec(compile(block, str(GUIDE), "exec"), {})
    owns_child = scenario.startswith("child_") or scenario == "interrupted"
    assert popen.call_count == int(owns_child)
    if owns_child:
        args = build_parser().parse_args(popen.call_args.args[0][1:])
        assert (args.command, args.model, args.host, args.port) == (
            "serve",
            "qwen3.5-4b-4bit",
            "127.0.0.1",
            8000,
        )
        child.terminate.assert_called_once_with()
        assert child.wait.call_count == (2 if scenario == "child_kill" else 1)
        assert child.kill.call_count == int(scenario == "child_kill")
    else:
        child.terminate.assert_not_called()
    client.__exit__.assert_called_once()


def test_stable_command_examples_parse():
    parser = build_parser()
    commands = re.findall(r"^(?:sudo )?rapid-mlx (.+)$", GUIDE.read_text(), re.M)
    assert commands
    # Check literal subprocess argv in the Python examples as well.
    for block in re.findall(r"```python\n(.*?)\n```", GUIDE.read_text(), re.S):
        for node in ast.walk(ast.parse(block)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if ast.unparse(node.func) in {"subprocess.run", "subprocess.Popen"}:
                    argv = ast.literal_eval(node.args[0])
                    assert "rapid-mlx" in argv[:2]
                    commands.append(shlex.join(argv[argv.index("rapid-mlx") + 1 :]))
    for command in commands:
        args = parser.parse_args(shlex.split(command))
        if args.command == "serve":
            assert (args.model, args.host, args.port) == (
                "qwen3.5-4b-4bit",
                "127.0.0.1",
                8000,
            )
        else:
            assert args.command == "service"
            assert args.service_command in {"install", "status", "restart", "uninstall"}


def test_stable_route_examples(metrics_client, residency_device_memory, monkeypatch):
    client, cfg = metrics_client.client, metrics_client.cfg
    manager, registry, _, _ = manager_fixture(limit_gib=12)
    cfg.residency_manager = manager
    cfg.engine = registry.get_engine("chat")
    cfg.model_name = "chat"
    cfg.ready = True
    client.app.include_router(residency.router)
    client.app.include_router(health.router)
    client.app.include_router(health.probe_router)
    for path in ("/v1/models/residency", "/v1/status"):
        assert client.get(path).status_code == 401
    for path in ("/healthz", "/health/ready", "/metrics"):
        assert client.get(path).status_code == 200
    client.headers["Authorization"] = "Bearer test-secret"
    monkeypatch.setattr(residency, "get_config", lambda: cfg)

    monkeypatch.setattr(
        "rapid_mlx.runtime.resident_models.estimate_model_bytes", lambda _: 1024**3
    )
    documented = sections()
    loaded_id = ""
    # Execute in lifecycle order, including a slash-containing model ID.
    for endpoint in (
        "POST /v1/models/load",
        "GET /v1/models/residency",
        "GET /v1/status",
        "GET /healthz",
        "GET /health/ready",
        "GET /metrics",
        "DELETE /v1/models/{id}",
    ):
        section = documented[endpoint]
        method, path = endpoint.split()
        path = path.replace("{id}", "org/model")
        request_match = re.search(r"-d '([^']+)'", section)
        body = json.loads(request_match[1]) if request_match else None
        if method == "DELETE":
            # Separate secondary, since replacement deliberately pins the primary.
            assert registry.get_entry("chat").model_name == loaded_id
            assert client.delete("/v1/models/" + loaded_id).status_code == 409
            manager.loader = AsyncMock(
                return_value=entry("org/model", FakeImageEngine())
            )
            asyncio.run(manager.load("org/model", estimated_bytes=1024**3))
            secondary_engine = registry.get_engine("org/model")
            secondary_engine.running = 1
            assert client.delete(path).status_code == 409
            secondary_engine.running = 0
        response = client.request(method, path, json=body)
        if method == "POST":
            loaded_id = response.json()["id"]
            assert loaded_id == body["model"]
            assert response.json()["primary"] is True
            assert response.json()["pinned"] is True
            cfg.engine = registry.get_engine(loaded_id)
            cfg.model_name = loaded_id
        if method == "DELETE":
            assert response.status_code == 204
            assert response.content == b""
            assert "org/model" not in registry
        else:
            assert response.status_code == 200
            if path == "/metrics":
                assert response.headers["content-type"].startswith("text/plain")
                assert "version=0.0.4" in response.headers["content-type"]
                assert "rapid_mlx_model_loaded 1" in response.text
            else:
                example = json.loads(
                    re.search(r"```json\n(.*?)\n```", section, re.S)[1]
                )
                assert_shape(response.json(), example)

    assert client.delete("/v1/models/org/model").status_code == 404
    assert (
        client.post(
            "/v1/models/load", json={"model": "too-large", "estimated_size_gb": 1024}
        ).status_code
        == 507
    )
    cfg.draining = True
    draining = client.get("/healthz")
    assert draining.status_code == 503
    assert draining.json()["status"] == "draining"
    assert draining.json()["ready"] is False
    cfg.draining = False
    cfg.ready = False
    assert client.get("/health/ready").status_code == 503
    cfg.engine = None
    assert client.get("/v1/status").json() == {
        "status": "not_loaded",
        "model": None,
        "requests": [],
    }
