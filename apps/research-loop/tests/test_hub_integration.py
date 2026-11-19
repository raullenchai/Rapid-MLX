"""Opt-in routing, token propagation, and cooperative job cancellation."""

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import hub_client
import literature
import protocol
import pytest


@pytest.fixture
def hub_env(monkeypatch):
    monkeypatch.setenv("RAPID_HUB_URL", "http://hub")
    monkeypatch.setenv("RAPID_HUB_TOKEN", "scoped-secret")


def test_model_discovery(hub_env, monkeypatch):
    monkeypatch.setattr(
        hub_client,
        "call",
        lambda path: {
            "data": [
                dict(id="chat-model", engine="q35", capabilities=["chat"]),
                dict(id="embed-model", engine="embed", capabilities=["embeddings"]),
            ]
        },
    )
    models, endpoint, model = hub_client.discover()
    assert models == {"chat-model": "http://hub/hub/v1/engines/q35/v1"}
    assert endpoint == "http://hub/hub/v1/engines/embed/v1" and model == "embed-model"


def test_subprocess_inherits_scoped_token_project_and_priority(hub_env, tmp_path):
    project = tmp_path / "grokking"
    env = hub_client.subprocess_env(project)
    code = "import json,hub_client;print(json.dumps(hub_client.headers()))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(hub_client.__file__).parent,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    headers = json.loads(result.stdout)
    assert headers["Authorization"] == "Bearer scoped-secret"
    assert headers["X-Rapid-Project"] == "grokking"
    assert headers["X-Rapid-Priority"] == "background"
    assert "scoped-secret" not in str(result.args)


def test_protocol_and_literature_http_headers(hub_env, monkeypatch):
    requests = []

    def upstream(request, **kwargs):
        requests.append(request)
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": '{"ok":"yes"}'},
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", upstream)
    with hub_client.project_scope("grokking"):
        model = protocol.Model(
            "http://hub/hub/v1/engines/q35/v1", "q35", lambda *a, **k: None
        )
        assert model.message(
            "custom",
            "test",
            schema={
                "type": "object",
                "properties": {"ok": {"type": "string"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        ) == {"ok": "yes"}
    assert requests[0].get_header("Authorization") == "Bearer scoped-secret"
    assert requests[0].get_header("X-rapid-project") == "grokking"
    assert requests[0].get_header("X-rapid-priority") == "background"

    class Opener:
        def open(self, request, **kwargs):
            requests.append(request)
            return io.BytesIO(b'{"data":[{"index":0,"embedding":[1,0]}]}')

    monkeypatch.setattr(literature, "build_opener", lambda *a: Opener())
    with hub_client.project_scope("grokking"):
        assert literature.HTTPEmbedder("http://hub/v1")(["passage"]) == [[1, 0]]
    assert requests[-1].get_header("X-rapid-project") == "grokking"
    assert requests[-1].get_header("Authorization") == "Bearer scoped-secret"


def test_job_cancel_uses_stop_button_files(hub_env, monkeypatch, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "PAUSE").touch()
    updates = []

    def call(path, body=None, method=None):
        updates.append((path, body, method))
        return {"state": "cancel_requested"}

    monkeypatch.setattr(hub_client, "call", call)

    class Process:
        def poll(self):
            return 0 if (run / "STOP").exists() else None

    hub_client.watch_run(tmp_path, run, Process())
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        if any(body and body.get("state") == "done" for _, body, _ in updates):
            break
        time.sleep(0.02)
    assert (run / "STOP").exists() and not (run / "PAUSE").exists()
    assert updates[0][1]["state"] == "running"
    assert any(body and body.get("state") == "done" for _, body, _ in updates)


def test_standalone_headers_unchanged(monkeypatch):
    monkeypatch.delenv("RAPID_HUB_URL", raising=False)
    assert hub_client.headers() == {"Content-Type": "application/json"}
    assert hub_client.subprocess_env(Path("grokking")) == dict(os.environ)
