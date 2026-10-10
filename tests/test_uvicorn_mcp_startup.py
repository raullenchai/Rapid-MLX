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
        from mcp.server.fastmcp import FastMCP
        mcp = FastMCP("startup-regression")
        @mcp.tool()
        def echo(text: str) -> str:
            return text
        mcp.run(transport="stdio")
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
