# SPDX-License-Identifier: Apache-2.0
"""U1 regression coverage for the post-bind Ready banner."""

from __future__ import annotations

import importlib
import inspect
import socket

import pytest
import uvicorn
from uvicorn.main import STARTUP_FAILURE

from rapid_mlx import cli, server
from rapid_mlx._uvicorn import AcceptingConnectionsServer, run_uvicorn


async def _asgi_app(scope, receive, send):
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return


@pytest.mark.asyncio
async def test_occupied_port_never_prints_ready_and_keeps_exit_code(capsys):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        config = uvicorn.Config(
            _asgi_app,
            host="127.0.0.1",
            port=port,
            log_level="error",
        )
        instance = AcceptingConnectionsServer(
            config,
            on_server_accepting=lambda: print("Ready: must-not-print"),
        )

        with pytest.raises(SystemExit) as raised:
            await instance.serve()

    captured = capsys.readouterr()
    assert raised.value.code == STARTUP_FAILURE
    assert "Ready:" not in captured.out + captured.err
    assert str(port) in captured.err
    assert "--port <n>" in captured.err


@pytest.mark.asyncio
async def test_success_banner_once_and_only_after_listener_exists(capsys):
    observations: list[tuple[bool, int]] = []
    instance: AcceptingConnectionsServer

    def on_accepting() -> None:
        listeners = instance.servers
        listener = listeners[0].sockets[0]
        host, port = listener.getsockname()[:2]
        with socket.socket() as client:
            client.settimeout(1)
            connect_result = client.connect_ex((host, port))
        observations.append((instance.started, connect_result))
        print(f"Ready: http://{host}:{port}")
        instance.should_exit = True

    config = uvicorn.Config(
        _asgi_app,
        host="127.0.0.1",
        port=0,
        log_level="error",
    )
    instance = AcceptingConnectionsServer(
        config,
        on_server_accepting=on_accepting,
    )

    await instance.serve()

    assert observations == [(True, 0)]
    assert capsys.readouterr().out.count("Ready:") == 1


def test_cli_audio_and_inherited_fd_register_shared_callback(monkeypatch):
    calls: list[dict] = []

    def fake_run(_app, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("rapid_mlx._uvicorn.run_uvicorn", fake_run)
    args = type("Args", (), {"host": "127.0.0.1", "port": 8000, "listen_fd": 7})()
    cli._run_uvicorn(object(), args, "error")

    assert calls[0]["fd"] == 7
    assert calls[0]["on_server_accepting"] is server.print_ready_banner
    assert "_run_uvicorn(app, args" in inspect.getsource(cli._serve_audio_mode)


def test_shared_runner_injects_server_subclass_and_restores_uvicorn(monkeypatch):
    uvicorn_main = importlib.import_module("uvicorn.main")
    original_server = uvicorn_main.Server
    observed: list[AcceptingConnectionsServer] = []

    def fake_uvicorn_run(app, **kwargs):
        config = uvicorn.Config(app, **kwargs)
        observed.append(uvicorn_main.Server(config))

    callback = lambda: None
    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
    run_uvicorn(_asgi_app, host="127.0.0.1", port=0, on_server_accepting=callback)

    assert isinstance(observed[0], AcceptingConnectionsServer)
    assert observed[0]._on_server_accepting is callback
    assert uvicorn_main.Server is original_server


def test_standalone_entrypoint_uses_shared_post_bind_seam():
    source = inspect.getsource(server.main)
    assert "run_uvicorn(" in source
    assert "on_server_accepting=print_ready_banner" in source
