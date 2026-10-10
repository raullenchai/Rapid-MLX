# SPDX-License-Identifier: Apache-2.0
"""Cold MCP imports must work inside the real serving lifespan (#4452)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from uvicorn.main import STARTUP_FAILURE

from rapid_mlx._uvicorn import AcceptingConnectionsServer, run_uvicorn


@pytest.mark.parametrize("listener", ["tcp", "uds", "fd"])
def test_cold_mcp_connect_and_execute_during_lifespan(tmp_path, listener):
    tool_server = tmp_path / "tools.py"
    tool_server.write_text(
        textwrap.dedent("""\
        import json
        import sys

        # Keep the peer independent of SDK server API renames. The serving
        # process below still cold-imports and exercises the real MCP client.
        for line in sys.stdin:
            message = json.loads(line)
            if "id" not in message:
                continue
            method = message.get("method")
            if method == "initialize":
                result = {
                    "protocolVersion": message["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "startup-regression", "version": "1"},
                }
            elif method == "tools/list":
                result = {"tools": [{
                    "name": "echo", "description": "Echo text",
                    "inputSchema": {"type": "object", "properties": {
                        "text": {"type": "string"}}, "required": ["text"]},
                }]}
            elif method == "tools/call":
                result = {"content": [{"type": "text", "text":
                    message["params"]["arguments"]["text"]}], "isError": False}
            else:
                result = {}
            print(json.dumps({"jsonrpc": "2.0", "id": message["id"],
                              "result": result}), flush=True)
    """)
    )
    # A subprocess is essential: the main test process may already have imported
    # MCP/SSE during collection, hiding an import-time Server compatibility bug.
    program = textwrap.dedent("""\
        import importlib
        import os
        import signal
        import socket
        import sys
        from rapid_mlx._uvicorn import run_uvicorn

        main = importlib.import_module("uvicorn.main")
        original = main.Server
        assert "mcp" not in sys.modules

        async def app(scope, receive, send):
            assert scope["type"] == "lifespan"
            assert (await receive())["type"] == "lifespan.startup"
            assert main.Server is original, "Server replaced during serving"
            assert callable(main.Server.handle_exit)
            from rapid_mlx.mcp.client import MCPClient
            from rapid_mlx.mcp.types import MCPServerConfig
            client = MCPClient(MCPServerConfig(
                name="regression", command=sys.executable, args=[sys.argv[1]]
            ))
            assert await client.connect(), client.get_status()
            try:
                assert [tool.name for tool in client.tools] == ["echo"]
                result = await client.call_tool("echo", {"text": "cold-start-ok"})
                assert not result.is_error, result
                assert "cold-start-ok" in str(result.content), result
                await send({"type": "lifespan.startup.complete"})
                assert (await receive())["type"] == "lifespan.shutdown"
            finally:
                await client.disconnect()
            await send({"type": "lifespan.shutdown.complete"})

        def ready():
            assert main.Server is original, "Server replaced during serving"
            print("MCP_READY", flush=True)
            os.kill(os.getpid(), signal.SIGINT)

        options = {"host": "127.0.0.1", "port": 0}
        inherited = None
        uds = sys.argv[1] + ".sock"
        if sys.argv[2] == "uds":
            os.chdir(os.path.dirname(sys.argv[1]))
            uds = "listener.sock"
            options = {"uds": uds}
        elif sys.argv[2] == "fd":
            inherited = socket.socket()
            inherited.bind(("127.0.0.1", 0))
            inherited.listen()
            options = {"fd": inherited.fileno()}
        try:
            run_uvicorn(app, **options, lifespan="on",
                        log_level="error", on_server_accepting=ready)
        finally:
            assert main.Server is original, "Server replaced during serving"
            assert not os.path.exists(uds)
            if inherited is not None:
                inherited.close()
    """)
    result = subprocess.run(
        [sys.executable, "-c", program, str(tool_server), listener],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "listener.sock").exists()
    assert "MCP_READY" in result.stdout, result.stdout + result.stderr
    assert "Traceback" not in result.stderr, result.stderr


@pytest.mark.parametrize("failure", [None, KeyboardInterrupt(), RuntimeError("run")])
def test_runner_cleans_uds_and_preserves_errors(monkeypatch, tmp_path, failure):
    uds = tmp_path / "listener.sock"

    def run(instance):
        assert instance.config.uds == str(uds)
        uds.touch()
        instance.started = True
        if failure is not None:
            raise failure

    monkeypatch.setattr(AcceptingConnectionsServer, "run", run)
    if isinstance(failure, RuntimeError):
        with pytest.raises(RuntimeError, match="run"):
            run_uvicorn("unused:app", uds=str(uds), log_level="error")
    else:
        run_uvicorn("unused:app", uds=str(uds), log_level="error")
    assert not uds.exists()


def test_runner_returns_startup_failure_when_server_never_started(monkeypatch):
    monkeypatch.setattr(AcceptingConnectionsServer, "run", lambda self: None)
    with pytest.raises(SystemExit) as caught:
        run_uvicorn("unused:app", log_level="error")
    assert caught.value.code == STARTUP_FAILURE


def test_runner_passes_inherited_fd_to_server(monkeypatch):
    observed = []

    def run(instance):
        observed.append(instance.config.fd)
        instance.started = True

    monkeypatch.setattr(AcceptingConnectionsServer, "run", run)
    run_uvicorn("unused:app", fd=17, log_level="error")
    assert observed == [17]


@pytest.mark.parametrize("options", [{"reload": True}, {"workers": 2}])
def test_runner_retains_import_string_requirement(options):
    with pytest.raises(SystemExit) as caught:
        run_uvicorn(object(), log_level="error", **options)
    assert caught.value.code == 1


@pytest.mark.parametrize(
    ("options", "supervisor_name"),
    [({"reload": True}, "ChangeReload"), ({"workers": 2}, "Multiprocess")],
)
def test_runner_delegates_import_string_app_to_supervisor(
    monkeypatch, options, supervisor_name
):
    import rapid_mlx._uvicorn as runner_module

    bound_socket = object()
    calls = []

    def bind_socket(config):
        calls.append(("bind", config))
        return bound_socket

    class Supervisor:
        def __init__(self, config, *, target, sockets):
            assert target.__self__.config is config
            assert target.__self__._on_server_accepting is ready
            assert sockets == [bound_socket]
            self.config = config

        def run(self):
            calls.append(("supervise", self.config))

    def ready():
        pytest.fail("the supervisor must own server execution")

    monkeypatch.setattr(runner_module.uvicorn.Config, "bind_socket", bind_socket)
    monkeypatch.setattr(runner_module, supervisor_name, Supervisor)
    run_uvicorn("unused:app", on_server_accepting=ready, log_level="error", **options)
    assert [name for name, _ in calls] == ["bind", "supervise"]
    assert calls[0][1] is calls[1][1]


def test_runner_loads_application_from_app_dir(monkeypatch, tmp_path):
    module_name = "startup_app_dir_fixture"
    (tmp_path / f"{module_name}.py").write_text(
        "async def app(scope, receive, send): pass\n"
    )
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    def run(instance):
        instance.config.load()
        assert instance.config.loaded_app is sys.modules[module_name].app
        instance.started = True

    monkeypatch.setattr(AcceptingConnectionsServer, "run", run)
    try:
        run_uvicorn(
            f"{module_name}:app",
            app_dir=str(tmp_path),
            interface="asgi3",
            proxy_headers=False,
            log_level="error",
        )
    finally:
        sys.modules.pop(module_name, None)


def test_custom_runner_retains_uvicorn_keyword_contract():
    calls = []

    def runner(app, *, host, port):
        calls.append((app, host, port))

    run_uvicorn(
        "unused:app",
        uvicorn_runner=runner,
        on_server_accepting=lambda: None,
        host="127.0.0.1",
        port=12345,
    )
    assert calls == [("unused:app", "127.0.0.1", 12345)]
