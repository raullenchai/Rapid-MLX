import ast
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from rapid_hub.app import Hub, create_app
from rapid_hub.manifest import discover, load
from rapid_hub.supervisor import Supervisor

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (ROOT / "apps/research-loop/rapid-plugin.toml").read_text()


def test_import_boundaries():
    for directory, forbidden in (
        (ROOT / "hub/rapid_hub", {"mlx", "rapid_mlx"}),
        (ROOT / "rapid_mlx", {"rapid_hub"}),
    ):
        for path in directory.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
                assert not forbidden.intersection(n.split(".")[0] for n in names), path


@pytest.mark.parametrize(
    "before,after",
    [
        ("^1", "^2"),
        ('id = "research-loop"', 'id = "../bad"'),
        ('files = "plugin"', 'files = "all"'),
        ("network = true", 'network = "yes"'),
        ('["{python}", "{plugin_dir}/server.py", "--hub"]', '"python"'),
        ("models.chat", "models.unknown"),
    ],
)
def test_invalid_manifest(tmp_path, before, after):
    path = tmp_path / "rapid-plugin.toml"
    path.write_text(MANIFEST.replace(before, after))
    with pytest.raises(ValueError, match=str(path)):
        load(path)


def test_discovery(tmp_path):
    for directory in (
        tmp_path / "one",
        tmp_path / "two",
        tmp_path / "nested" / "three",
    ):
        directory.mkdir(parents=True)
        (directory / "rapid-plugin.toml").write_text(MANIFEST)
    plugins, errors = discover(tmp_path)
    assert list(plugins) == ["research-loop"]
    assert len(errors) == 1 and "duplicate" in errors[0]


class SSE(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        await asyncio.sleep(0)
        yield b'data: {"usage":{"total_tokens":7}}\n\ndata: [DONE]\n\n'


def make_hub(tmp_path, handler):
    hub = Hub(
        {"chat": "http://engine/v1"},
        tmp_path,
        tmp_path / "empty",
        transport=httpx.MockTransport(handler),
    )
    hub.tokens.update(one="one", two="two")
    return hub


def client(hub, token="one"):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(hub, supervise=False)),
        base_url="http://127.0.0.1:4800",
        headers={"Authorization": "Bearer " + token},
    )


def test_gateway_auth_json_sse_journal_jobs(tmp_path):
    seen = []

    async def upstream(request):
        seen.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": "m", "capabilities": ["text", "embedding"]}]}
            )
        if request.url.path == "/health":
            return httpx.Response(200, json={"ready": True})
        payload = json.loads(request.content or b"{}")
        if payload.get("stream"):
            return httpx.Response(
                200, stream=SSE(), headers={"Content-Type": "text/event-stream"}
            )
        return httpx.Response(
            429,
            json={"error": "busy", "usage": {"total_tokens": 3}},
            headers={"Retry-After": "2"},
        )

    async def run():
        hub = make_hub(tmp_path, upstream)
        async with client(hub) as c:
            assert (
                await c.get("/hub/v1/models", headers={"Authorization": ""})
            ).status_code == 401
            catalog = (await c.get("/hub/v1/models")).json()
            assert catalog["default"]["id"] == "m"
            assert catalog["data"][0]["capabilities"] == ["chat", "embeddings"]
            assert catalog["engines"][0]["ready"]
            path = "/hub/v1/engines/chat/v1/chat/completions"
            response = await c.post(
                path,
                json={"model": "m", "messages": ["secret text"]},
                headers={"X-Rapid-Project": "p"},
            )
            assert (
                response.status_code == 429 and response.headers["retry-after"] == "2"
            )
            response = await c.post(
                path,
                json={"model": "m", "stream": True},
                headers={"X-Rapid-Project": "p"},
            )
            assert response.content.endswith(b"data: [DONE]\n\n")
            entries = (await c.get("/hub/v1/journal?project=p")).json()["entries"]
            assert len(entries) == 2 and entries[0]["usage"] == {"total_tokens": 7}
            row = entries[1]
            assert (
                row["plugin"] == "one"
                and row["engine"] == "chat"
                and row["priority"] == "interactive"
            )
            assert (
                row["request"]["messages"] == ["secret text"] and row["status"] == 429
            )
            assert row["latency_ms"] >= row["queue_wait_ms"] >= 0 and row["ts"]
            assert (
                seen[-1].headers["x-rapid-plugin"] == "one"
                and seen[-1].headers["x-rapid-project"] == "p"
            )
            assert "authorization" not in seen[-1].headers
            assert (await c.get("/hub/v1/journal?plugin=two")).status_code == 403
            assert (
                await c.get("/hub/v1/journal", headers={"Authorization": "Bearer two"})
            ).json() == {"entries": []}
            job = {"id": "j", "title": "Test", "state": "running", "project": "p"}
            assert (await c.post("/hub/v1/jobs", json=job)).status_code == 200
            assert (await c.post("/hub/v1/jobs/j/cancel")).status_code == 403
            assert (
                await c.get("/hub/v1/jobs/j", headers={"Authorization": "Bearer two"})
            ).status_code == 403
            assert (
                await c.post(
                    "/hub/v1/jobs/j/cancel",
                    headers={"Authorization": "Bearer " + hub.admin},
                )
            ).json()["state"] == "cancel_requested"
            assert (
                await c.patch("/hub/v1/jobs", json={"id": "j", "progress": "step"})
            ).json()["state"] == "cancel_requested"
            assert (await c.patch("/hub/v1/jobs/j", json={"state": "done"})).json()[
                "state"
            ] == "done"
            assert (
                json.loads((tmp_path / "jobs.json").read_text())["j"]["state"] == "done"
            )
            assert (await c.get("/", headers={"Authorization": ""})).status_code == 401
            nonce = hub.login
            response = await c.get(
                "/login?nonce=" + nonce, headers={"Authorization": ""}
            )
            assert (
                response.status_code == 303
                and "HttpOnly" in response.headers["set-cookie"]
                and "SameSite=strict" in response.headers["set-cookie"]
            )
            assert (await c.get("/", headers={"Authorization": ""})).status_code == 200
            assert (await c.get("/login?nonce=" + nonce)).status_code == 401
            assert (
                await c.get("/hub/v1/models", headers={"Host": "evil.example"})
            ).status_code == 403
            assert (
                await c.post(path, json={}, headers={"Origin": "http://127.0.0.1:9999"})
            ).status_code == 403
        await hub.client.aclose()
        for path in [
            tmp_path / "admin-token",
            *tmp_path.glob("journal/*/*.jsonl"),
            tmp_path / "jobs.json",
        ]:
            assert path.stat().st_mode & 0o777 == 0o600

    asyncio.run(run())


def test_background_fifo_and_interactive(tmp_path):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        order = []

        async def upstream(request):
            value = json.loads(request.content)["model"]
            order.append(value)
            if value == "first":
                entered.set()
                await release.wait()
            return httpx.Response(200, json={"model": value})

        hub = make_hub(tmp_path, upstream)
        async with client(hub) as c:
            path = "/hub/v1/engines/chat/v1/chat/completions"

            async def post(value, priority="background"):
                return await c.post(
                    path, json={"model": value}, headers={"X-Rapid-Priority": priority}
                )

            first = asyncio.create_task(post("first"))
            await entered.wait()
            second = asyncio.create_task(post("second"))
            await asyncio.sleep(0.02)
            third = asyncio.create_task(post("third"))
            await asyncio.sleep(0.02)
            assert (
                await asyncio.wait_for(post("interactive", "interactive"), 1)
            ).status_code == 200
            assert order == ["first", "interactive"]
            release.set()
            await asyncio.gather(first, second, third)
            assert order == ["first", "interactive", "second", "third"]
        assert hub.journal.read()[0]["queue_wait_ms"] > 0
        await hub.client.aclose()

    asyncio.run(run())


def test_supervisor_ready_restart_failure_and_process_group(tmp_path):
    script = tmp_path / "plugin.py"
    script.write_text("""import os, signal, subprocess, sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
root = Path(os.environ["RAPID_PLUGIN_DATA_DIR"])
if (root / "crash").exists(): sys.exit(1)
child = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(100)"])
(root / "child").write_text(str(child.pid))
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
HTTPServer((os.environ["RAPID_PLUGIN_HOST"], int(os.environ["RAPID_PLUGIN_PORT"])), H).serve_forever()
""")
    manifest = load(ROOT / "apps/research-loop/rapid-plugin.toml")
    manifest["run"]["command"] = [sys.executable, str(script)]
    manifest["requires"] = []

    async def run():
        hub = make_hub(
            tmp_path / "state", lambda r: httpx.Response(200, json={"data": []})
        )
        # Health requests must reach the real fake plugin, not MockTransport.
        await hub.client.aclose()
        hub.client = httpx.AsyncClient(trust_env=False)
        supervisor = Supervisor(
            hub, {"research-loop": manifest}, retries=2, delay=0.02, ready_timeout=2
        )
        item = supervisor.items["research-loop"]
        task = asyncio.create_task(supervisor.run("research-loop"))
        supervisor.tasks["research-loop"] = task

        async def ready():
            while item["state"] != "ready":
                await asyncio.sleep(0.02)

        try:
            await asyncio.wait_for(ready(), 5)
            child = int((hub.data_dir / "plugins/research-loop/data/child").read_text())
            (hub.data_dir / "plugins/research-loop/data/crash").touch()
            os.kill(item["process"].pid, signal.SIGKILL)
            await asyncio.wait_for(task, 5)
            assert item["state"] == "failed" and item["restarts"] == 2
            await asyncio.sleep(0.1)
            with pytest.raises(ProcessLookupError):
                os.kill(child, 0)
            (hub.data_dir / "plugins/research-loop/data/crash").unlink()
            await supervisor.restart("research-loop")
            await asyncio.wait_for(ready(), 5)
            pid = item["process"].pid
            child = int((hub.data_dir / "plugins/research-loop/data/child").read_text())
            await supervisor.stop()
            await asyncio.sleep(0.1)
            for ident in (pid, child):
                with pytest.raises(ProcessLookupError):
                    os.kill(ident, 0)
        finally:
            await supervisor.stop()
            await hub.client.aclose()

    asyncio.run(run())


def test_sse_delivered_before_upstream_finishes_and_disconnect_releases_slot(tmp_path):
    async def run():
        release = asyncio.Event()
        closed = asyncio.Event()

        class BlockingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[]}\n\n'
                await release.wait()
                yield b"data: [DONE]\n\n"

            async def aclose(self):
                closed.set()

        hub = make_hub(
            tmp_path,
            lambda request: httpx.Response(
                200,
                stream=BlockingStream(),
                headers={"Content-Type": "text/event-stream"},
            ),
        )
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(hub, supervise=False), log_level="critical", access_log=False
            )
        )
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started:
                await asyncio.sleep(0.01)
            async with (
                httpx.AsyncClient(trust_env=False) as c,
                c.stream(
                    "POST",
                    f"http://127.0.0.1:{port}/hub/v1/engines/chat/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer one",
                        "X-Rapid-Priority": "background",
                    },
                    json={"stream": True},
                ) as response,
            ):
                line = await asyncio.wait_for(anext(response.aiter_lines()), 1)
                assert line == 'data: {"choices":[]}' and not release.is_set()
            await asyncio.wait_for(closed.wait(), 1)
            for _ in range(100):
                if not hub.gates["chat"].locked():
                    break
                await asyncio.sleep(0.01)
            assert not hub.gates["chat"].locked()
            assert hub.journal.read()[0]["phase"] == "finished"
        finally:
            release.set()
            server.should_exit = True
            await task
            sock.close()

    asyncio.run(run())


def test_catalog_does_not_guess_and_unavailable_engine_is_journaled(tmp_path):
    async def run():
        async def upstream(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"ready": False})
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "unknown"}]})
            raise httpx.ConnectError("offline", request=request)

        hub = make_hub(tmp_path, upstream)
        async with client(hub) as c:
            catalog = (await c.get("/hub/v1/models")).json()
            assert catalog["default"] is None
            assert catalog["data"][0]["capabilities"] == []
            assert not catalog["data"][0]["ready"]
            assert (
                await c.post(
                    "/hub/v1/engines/chat/v1/chat/completions",
                    json={"model": "unknown"},
                )
            ).status_code == 502
            assert hub.journal.read()[0]["status"] == 502
        manifest = load(ROOT / "apps/research-loop/rapid-plugin.toml")
        hub.supervisor = Supervisor(hub, {"research-loop": manifest})
        await hub.supervisor.start()
        assert hub.supervisor.items["research-loop"]["state"] == "blocked"
        assert not hub.supervisor.tasks
        await hub.client.aclose()

    asyncio.run(run())


def test_guardian_cleans_group_when_parent_is_killed(tmp_path):
    runner = ROOT / "hub/rapid_hub/runner.py"
    child_file, guardian_file = tmp_path / "child", tmp_path / "guardian"
    child_code = f"import os,time;from pathlib import Path;Path({str(child_file)!r}).write_text(str(os.getpid()));time.sleep(100)"
    code = f"import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,{str(runner)!r},sys.executable,'-c',{child_code!r}],start_new_session=True);Path({str(guardian_file)!r}).write_text(str(p.pid));p.wait()"
    parent = subprocess.Popen([sys.executable, "-c", code])
    try:
        deadline = time.monotonic() + 4
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        child = int(child_file.read_text())
        parent.kill()
        parent.wait()
        while time.monotonic() < deadline:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            pytest.fail("Guardian left plugin running after parent death")
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if guardian_file.exists():
            try:
                os.killpg(int(guardian_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("returncode", [0, None])
def test_cleanup_permission_error_for_reaped_group(tmp_path, monkeypatch, returncode):
    class Process:
        pid = 12345

    process = Process()
    process.returncode = returncode

    def gone_group(pid, sig):
        raise PermissionError("sandbox denied missing process group")

    monkeypatch.setattr(os, "killpg", gone_group)

    async def run():
        hub = make_hub(tmp_path, lambda r: httpx.Response(200))
        item = {"process": process}
        if returncode is None:
            with pytest.raises(PermissionError):
                await hub.supervisor.terminate(item)
        else:
            await hub.supervisor.terminate(item)
            assert item["process"] is None
        await hub.client.aclose()

    asyncio.run(run())
