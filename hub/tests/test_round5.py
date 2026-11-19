"""Fixed plugin port CLI and supervised lifecycle regressions."""

import asyncio
import socket
import sys
from types import SimpleNamespace

import httpx
import pytest
from rapid_hub import __main__ as cli
from rapid_hub.manifest import load
from rapid_hub.supervisor import Supervisor
from test_hub import ROOT, make_hub


def test_cli_repeatable_plugin_ports(monkeypatch, tmp_path):
    seen = {}

    def hub(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(dashboard_url="http://hub.localhost:4800", login="nonce")

    monkeypatch.setattr(cli, "Hub", hub)
    monkeypatch.setattr(cli, "create_app", lambda h: h)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **kw: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rapid-hub",
            "up",
            "--plugin-port",
            "research-loop=4910",
            "--plugin-port",
            "other=4911",
            "--data-dir",
            str(tmp_path),
        ],
    )
    cli.main()
    assert seen["plugin_ports"] == {"research-loop": 4910, "other": 4911}


@pytest.mark.parametrize(
    "values",
    [
        ["research-loop=0"],
        ["research-loop=65536"],
        ["research-loop=no"],
        ["research-loop"],
        ["research-loop=4910", "research-loop=4911"],
    ],
)
def test_cli_invalid_plugin_ports(monkeypatch, capsys, values):
    monkeypatch.setattr(
        sys,
        "argv",
        ["rapid-hub", "up", *[s for value in values for s in ("--plugin-port", value)]],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "--plugin-port requires" in capsys.readouterr().err


def test_plugin_port_taken_and_invalid_assignments():
    hub = SimpleNamespace(port=4800, plugin_host="127.0.0.1", tokens={})
    manifests = {"research-loop": {"name": "Research Loop"}, "other": {"name": "Other"}}
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.listen(1)
        with pytest.raises(
            ValueError, match=f"research-loop port {port} is unavailable"
        ):
            Supervisor(hub, manifests, ports={"research-loop": port})
        with pytest.raises(ValueError, match="Unknown plugin"):
            Supervisor(hub, manifests, ports={"missing": port})
    with pytest.raises(ValueError, match="duplicate"):
        Supervisor(hub, manifests, ports={"research-loop": 4800})
    with pytest.raises(ValueError, match="duplicate"):
        Supervisor(hub, manifests, ports={"research-loop": port, "other": port})


def test_fixed_plugin_port_survives_crash_restart_and_hub_restart(tmp_path):
    script = tmp_path / "plugin.py"
    script.write_text("""import os
from http.server import HTTPServer, BaseHTTPRequestHandler
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
HTTPServer.allow_reuse_address = True
HTTPServer((os.environ["RAPID_PLUGIN_HOST"], int(os.environ["RAPID_PLUGIN_PORT"])), H).serve_forever()
""")
    manifest = load(ROOT / "apps/research-loop/rapid-plugin.toml")
    manifest["run"]["command"] = [sys.executable, str(script)]
    manifest["requires"] = []
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    async def run():
        for cycle in range(2):
            hub = make_hub(
                tmp_path / str(cycle), lambda r: httpx.Response(200, json={"data": []})
            )
            await hub.client.aclose()
            hub.client = httpx.AsyncClient(trust_env=False)
            supervisor = Supervisor(
                hub,
                {"research-loop": manifest},
                ports={"research-loop": port},
                delay=0.01,
            )

            async def ready(supervisor=supervisor):
                async with asyncio.timeout(8):
                    while supervisor.items["research-loop"]["state"] != "ready":
                        await asyncio.sleep(0.01)
                item = supervisor.items["research-loop"]
                assert item["port"] == port
                assert supervisor.public()[0]["url"].endswith(f":{port}")
                return item

            try:
                await supervisor.start()
                item = await ready()
                old = item["process"]
                old.kill()
                await old.wait()
                async with asyncio.timeout(8):
                    while item["process"] is old or item["state"] != "ready":
                        await asyncio.sleep(0.01)
                await ready()
                await supervisor.restart("research-loop")
                await ready()
            finally:
                await supervisor.stop()
                await hub.client.aclose()
                await hub.journal.close()

    asyncio.run(run())


def test_fixed_plugin_port_taken_after_preflight(tmp_path):
    hub = SimpleNamespace(
        port=4800,
        plugin_host="127.0.0.1",
        tokens={},
        data_dir=tmp_path,
        emit=lambda *args: None,
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    supervisor = Supervisor(
        hub, {"research-loop": {"name": "Research Loop"}}, ports={"research-loop": port}
    )

    async def run():
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", port))
            occupied.listen(1)
            await supervisor.run("research-loop")
            item = supervisor.items["research-loop"]
            assert item["state"] == "failed"
            assert f"port {port} is unavailable" in item["error"]
            assert item["process"] is None
            assert not hub.tokens

    asyncio.run(run())
