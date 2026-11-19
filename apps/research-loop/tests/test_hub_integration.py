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
    assert models == {"q35/chat-model": "http://hub/hub/v1"}
    assert endpoint == "http://hub/hub/v1" and model == "embed/embed-model"


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
            return io.BytesIO(
                b'{"model":"org/actual-embed","data":[{"index":0,"embedding":[1,0]}]}'
            )

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
        def terminate(self):
            pass

        def wait(self, timeout):
            return 0

        def poll(self):
            return 0 if (run / "STOP").exists() else None

    hub_client.watch_run(tmp_path, run, Process())
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        if any(body and body.get("state") == "cancelled" for _, body, _ in updates):
            break
        time.sleep(0.02)
    assert (run / "STOP").exists() and not (run / "PAUSE").exists()
    assert updates[0][1]["state"] == "running"
    assert any(body and body.get("state") == "cancelled" for _, body, _ in updates)


def test_standalone_headers_unchanged(monkeypatch):
    monkeypatch.delenv("RAPID_HUB_URL", raising=False)
    assert hub_client.headers() == {"Content-Type": "application/json"}
    assert hub_client.subprocess_env(Path("grokking")) == dict(os.environ)


def test_hub_ui_session_required_pages_and_api(hub_env, monkeypatch, tmp_path):
    import threading

    import httpx
    import server

    monkeypatch.setenv("RAPID_PLUGIN_ID", "research-loop")
    redeemed = []

    def call(path, body=None, method=None):
        assert path == "ui-codes/redeem" and method == "POST"
        if body["code"] != "valid" or redeemed:
            raise ValueError("consumed")
        redeemed.append(body["code"])
        return {"authenticated": True}

    monkeypatch.setattr(hub_client, "call", call)
    app = server.App(
        ("127.0.0.1", 0), "http://unused/v1", "m", 1, projects=tmp_path / "projects"
    )
    app.hub_mode, app.hub_sessions = True, set()
    worker = threading.Thread(target=app.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{app.server_port}"
    try:
        with httpx.Client(
            base_url=url,
            trust_env=False,
            headers={"Host": f"research-loop.localhost:{app.server_port}"},
        ) as c:
            assert c.get("/healthz").status_code == 200
            assert c.get("/").status_code == 401
            assert c.get("/api/projects").status_code == 401
            assert c.post("/api/projects", json={}).status_code == 401
            response = c.get("/hub-auth?code=valid")
            assert response.status_code == 303 and response.headers["location"] == "/"
            assert response.headers["referrer-policy"] == "no-referrer"
            cookie = response.headers["set-cookie"]
            assert (
                "Domain=" not in cookie
                and "HttpOnly" in cookie
                and "SameSite=Strict" in cookie
            )
            assert (
                c.get("/", headers={"Cookie": cookie.split(";")[0]}).status_code == 200
            )
            assert (
                c.get(
                    "/api/projects", headers={"Cookie": cookie.split(";")[0]}
                ).status_code
                == 200
            )
            assert c.get("/hub-auth?code=valid").status_code == 401
            assert (
                c.get(
                    "/",
                    headers={"Host": "hub.localhost", "Cookie": cookie.split(";")[0]},
                ).status_code
                == 403
            )
    finally:
        app.shutdown()
        app.server_close()
        worker.join(3)


def test_cancel_ack_terminates_real_long_experiment_group(
    hub_env, monkeypatch, tmp_path
):
    """Run the actual loop SIGTERM handler while a seed and grandchild sleep."""
    import signal

    project = tmp_path / "long"
    project.mkdir()
    (project / "task.json").write_text(
        json.dumps(
            {
                "title": "Long",
                "editable": "long.py",
                "metric": "score",
                "direction": "max",
                "noise": 0,
                "seeds": [0],
                "fixed": [],
                "timeout_s": 120,
            }
        )
    )
    (project / "program.md").write_text("Long running cancellation fixture")
    pid_file = tmp_path / "experiment-pids"
    (project / "long.py").write_text(
        "import subprocess,sys,time,os\nfrom pathlib import Path\n"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])\n"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())+' '+str(p.pid))\n"
        "time.sleep(120)\n"
    )
    run = project / "runs/test"
    run.mkdir(parents=True)
    root = Path(hub_client.__file__).parent
    updates = []

    def call(path, body=None, method=None):
        updates.append((path, body, method))
        return {"state": "cancel_requested"}

    monkeypatch.setattr(hub_client, "call", call)
    with (run / "loop.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(root / "loop.py"),
                str(project),
                str(run),
                "--rounds",
                "1",
            ],
            stdout=log,
            stderr=log,
            env=hub_client.subprocess_env(project),
        )
    pids = []
    try:
        deadline = time.monotonic() + 8
        while (
            not pid_file.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert pid_file.exists(), (run / "loop.log").read_text()
        pids = [int(p) for p in pid_file.read_text().split()]
        hub_client.watch_run(project, run, process)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not any(
            b and b.get("state") == "cancelled" for _, b, _ in updates
        ):
            time.sleep(0.02)
        states = [b["state"] for _, b, _ in updates if b and "state" in b]
        assert states == ["running", "cancelling", "cancelled"]
        assert process.poll() is not None
        for pid in pids:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.02)
            else:
                pytest.fail(f"Experiment descendant {pid} survived cancellation")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("failure", ["conflict", "transient"])
def test_terminal_job_update_reconciles_and_retries(
    hub_env, monkeypatch, tmp_path, failure
):
    import threading
    from urllib.error import HTTPError

    run = tmp_path / "run"
    run.mkdir()
    (run / "PAUSE").touch()
    state = "running"
    attempts, updates = [], []
    finished = threading.Event()

    def call(path, body=None, method=None):
        nonlocal state
        if method == "PATCH":
            target = body.get("state")
            updates.append(target)
            if target == "done" and not attempts:
                attempts.append(target)
                if failure == "conflict":
                    state = "cancel_requested"
                raise HTTPError(
                    "http://hub/jobs",
                    409 if failure == "conflict" else 503,
                    "retry",
                    {},
                    None,
                )
            if target:
                state = target
            if target in ("done", "cancelled"):
                finished.set()
        return {"state": state}

    class Process:
        def poll(self):
            return 0

        def terminate(self):
            pass

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(hub_client, "call", call)
    hub_client.watch_run(tmp_path, run, Process())
    assert finished.wait(4), updates
    assert state == ("cancelled" if failure == "conflict" else "done")
    if failure == "conflict":
        assert updates == ["done", "cancelling", "cancelled"]
        assert (run / "STOP").exists() and not (run / "PAUSE").exists()


def test_printed_public_plugin_url():
    import server

    assert (
        server.public_url("127.0.0.1", 54321, True)
        == "http://research-loop.localhost:54321"
    )
    assert server.public_url("127.0.0.1", 54321, False) == "http://127.0.0.1:54321"
