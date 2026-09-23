# SPDX-License-Identifier: Apache-2.0
"""U1 regression coverage for the post-bind Ready banner."""

from __future__ import annotations

import importlib
import inspect
import socket
import sys
from concurrent.futures import Future
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import uvicorn
from uvicorn.main import STARTUP_FAILURE

from rapid_mlx import cli, server
from rapid_mlx._uvicorn import (
    AcceptingConnectionsServer,
    _port_is_in_use,
    run_uvicorn,
)


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


@pytest.mark.asyncio
async def test_inherited_listener_runs_callback_only_after_socket_accepts():
    observations: list[int] = []
    instance: AcceptingConnectionsServer

    with socket.socket() as inherited:
        inherited.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        inherited.bind(("127.0.0.1", 0))
        inherited.listen()
        address = inherited.getsockname()

        def on_accepting() -> None:
            with socket.socket() as client:
                client.settimeout(1)
                observations.append(client.connect_ex(address))
            instance.should_exit = True

        config = uvicorn.Config(
            _asgi_app,
            fd=inherited.fileno(),
            log_level="error",
        )
        instance = AcceptingConnectionsServer(
            config,
            on_server_accepting=on_accepting,
        )
        await instance.serve()

    assert observations == [0]


@pytest.mark.asyncio
async def test_empty_listener_collection_never_announces_ready(monkeypatch):
    callbacks: list[str] = []

    async def startup_without_listener(_instance, sockets=None):
        _instance.started = True
        _instance.servers = []

    monkeypatch.setattr(uvicorn.Server, "startup", startup_without_listener)
    instance = AcceptingConnectionsServer(
        uvicorn.Config(_asgi_app),
        on_server_accepting=lambda: callbacks.append("ready"),
    )

    await instance.startup()

    assert callbacks == []
    assert instance._accepting_callback_ran is False


def test_unoccupied_port_discriminator_returns_false():
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]

    assert _port_is_in_use("127.0.0.1", port) is False


@pytest.mark.parametrize("listen_fd", [None, 7])
def test_cli_host_port_and_inherited_fd_register_shared_callback(
    monkeypatch, listen_fd
):
    calls: list[dict] = []

    def fake_run(_app, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("rapid_mlx._uvicorn.run_uvicorn", fake_run)
    args = SimpleNamespace(host="127.0.0.1", port=8000, listen_fd=listen_fd)
    cli._run_uvicorn(object(), args, "error")

    if listen_fd is None:
        assert calls[0]["host"] == "127.0.0.1"
        assert calls[0]["port"] == 8000
        assert "fd" not in calls[0]
    else:
        assert calls[0]["fd"] == 7
        assert "host" not in calls[0]
    assert calls[0]["on_server_accepting"] is server.print_ready_banner
    assert "_run_uvicorn(app, args" in inspect.getsource(cli._serve_audio_mode)


def test_cli_preserves_non_bind_oserror(monkeypatch):
    error = OSError(13, "permission denied")

    def fail_before_bind(_app, **_kwargs):
        raise error

    monkeypatch.setattr("rapid_mlx._uvicorn.run_uvicorn", fail_before_bind)
    args = SimpleNamespace(host="127.0.0.1", port=80, listen_fd=None)

    with pytest.raises(OSError) as raised:
        cli._run_uvicorn(object(), args, "error")

    assert raised.value is error


def test_cli_preserves_already_reported_uvicorn_exit(monkeypatch):
    error = SystemExit(STARTUP_FAILURE)
    error.rapid_mlx_bind_reported = True

    def fail_after_report(_app, **_kwargs):
        raise error

    monkeypatch.setattr("rapid_mlx._uvicorn.run_uvicorn", fail_after_report)
    monkeypatch.setattr(
        cli,
        "_port_is_busy",
        lambda *_args: pytest.fail("reported bind failures must not be probed twice"),
    )
    args = SimpleNamespace(host="127.0.0.1", port=8000, listen_fd=None)

    with pytest.raises(SystemExit) as raised:
        cli._run_uvicorn(object(), args, "error")

    assert raised.value is error


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


def test_standalone_entrypoint_stashes_endpoint_and_registers_callback(monkeypatch):
    from rapid_mlx import _uvicorn
    from rapid_mlx.config import get_config

    calls: list[tuple[object, dict]] = []
    cfg = get_config()
    monkeypatch.setattr(cfg, "bind_host", cfg.bind_host)
    monkeypatch.setattr(cfg, "bind_port", cfg.bind_port)
    monkeypatch.setattr(cfg, "bind_listen_fd", cfg.bind_listen_fd)
    monkeypatch.setattr(cli, "_port_preflight_or_die", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_hard_exit_after_serve", lambda: None)
    monkeypatch.setattr(server, "_preflight_vision_runtime", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "load_model", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "rapid_mlx._version_check.prompt_upgrade_if_available", lambda: False
    )
    monkeypatch.setattr(
        "rapid_mlx._version_check.print_staleness_warning_if_any",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        _uvicorn,
        "run_uvicorn",
        lambda app, **kwargs: calls.append((app, kwargs)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rapid_mlx.server",
            "--model",
            "offline/test-model",
            "--no-mllm",
            "--host",
            "0.0.0.0",
            "--port",
            "8100",
        ],
    )

    server.main()

    assert cfg.bind_host == "localhost"
    assert cfg.bind_port == 8100
    assert cfg.bind_listen_fd is None
    assert calls == [
        (
            server.app,
            {
                "host": "0.0.0.0",
                "port": 8100,
                "log_level": "info",
                "on_server_accepting": server.print_ready_banner,
            },
        )
    ]


class _ImmediateExecutor:
    def submit(self, function, *args, **kwargs):
        future: Future = Future()
        future.set_result(function(*args, **kwargs))
        return future


def test_ddtree_runner_uses_shared_seam_without_false_ready(monkeypatch, capsys):
    from rapid_mlx.speculative.ddtree import server as ddtree_server

    calls: list[tuple[object, dict]] = []
    pending: Future = Future()
    monkeypatch.setattr(ddtree_server, "have_runtime", lambda: True)
    monkeypatch.setattr(
        ddtree_server,
        "_ddtree_loader_executor",
        SimpleNamespace(submit=lambda *_args, **_kwargs: pending),
    )
    monkeypatch.setattr(ddtree_server, "_build_app", lambda **_kwargs: "ddtree-app")
    monkeypatch.setattr(
        "rapid_mlx._uvicorn.run_uvicorn",
        lambda app, **kwargs: calls.append((app, kwargs)),
    )

    ddtree_server.run_ddtree_server(
        main_model_repo="target",
        drafter_repo="drafter",
        speculative_tokens=4,
        tree_budget=8,
        host="0.0.0.0",
        port=8101,
        served_model_name="model",
        default_max_tokens=32,
        cors_origins=[],
        uvicorn_log_level="warning",
    )

    output = capsys.readouterr().out
    assert "Starting: http://localhost:8101/v1" in output
    assert "Ready:" not in output
    assert calls == [
        (
            "ddtree-app",
            {
                "host": "0.0.0.0",
                "port": 8101,
                "log_level": "warning",
                "timeout_keep_alive": 30,
            },
        )
    ]


def test_dflash_runner_defers_its_existing_banner_to_callback(monkeypatch, capsys):
    from rapid_mlx.speculative.dflash import server as dflash_server

    mlx_vlm = ModuleType("mlx_vlm")
    mlx_vlm.load = lambda *_args, **_kwargs: (object(), object())
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setattr(dflash_server, "have_runtime", lambda: True)
    monkeypatch.setattr(dflash_server, "load_runtime", lambda *_a, **_k: object())
    monkeypatch.setattr(dflash_server, "_dflash_executor", _ImmediateExecutor())
    monkeypatch.setattr(dflash_server, "_build_app", lambda **_kwargs: "dflash-app")
    calls: list[tuple[object, dict]] = []
    monkeypatch.setattr(
        "rapid_mlx._uvicorn.run_uvicorn",
        lambda app, **kwargs: calls.append((app, kwargs)),
    )

    dflash_server.run_dflash_server(
        main_model_repo="target",
        drafter_repo="drafter",
        host="0.0.0.0",
        port=8102,
        served_model_name="model",
        default_max_tokens=32,
        cors_origins=[],
        uvicorn_log_level="warning",
    )

    assert "Ready:" not in capsys.readouterr().out
    app, kwargs = calls[0]
    assert app == "dflash-app"
    callback = kwargs.pop("on_server_accepting")
    assert kwargs == {
        "host": "0.0.0.0",
        "port": 8102,
        "log_level": "warning",
        "timeout_keep_alive": 30,
    }
    callback()
    assert capsys.readouterr().out == (
        "  Ready: http://localhost:8102/v1  (DFlash mode)\n"
        "  Docs:  http://localhost:8102/docs\n\n"
    )


def test_native_mtp_runner_defers_its_existing_banner_to_callback(monkeypatch, capsys):
    from rapid_mlx.speculative.dflash import server as dflash_server
    from rapid_mlx.speculative.native_mtp import server as native_server
    from rapid_mlx.speculative.native_mtp.eligibility import QWEN36_35B_4BIT

    mlx_vlm = ModuleType("mlx_vlm")
    mlx_vlm.load = lambda *_args, **_kwargs: (object(), object())
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    runtime = SimpleNamespace(
        drafter=SimpleNamespace(bind=lambda _model: None),
        kind="mtp",
        block_size=3,
    )
    monkeypatch.setattr(native_server, "load_runtime", lambda *_a, **_k: runtime)
    monkeypatch.setattr(dflash_server, "_dflash_executor", _ImmediateExecutor())
    monkeypatch.setattr(dflash_server, "_build_app", lambda **_kwargs: "mtp-app")
    calls: list[tuple[object, dict]] = []
    monkeypatch.setattr(
        native_server,
        "run_uvicorn",
        lambda app, **kwargs: calls.append((app, kwargs)),
    )

    native_server.run_native_mtp_server(
        pair=QWEN36_35B_4BIT,
        host="0.0.0.0",
        port=8103,
        served_model_name="model",
        default_max_tokens=32,
        cors_origins=[],
        uvicorn_log_level="warning",
    )

    assert "Ready:" not in capsys.readouterr().out
    app, kwargs = calls[0]
    assert app == "mtp-app"
    callback = kwargs.pop("on_server_accepting")
    assert kwargs["uvicorn_runner"] is uvicorn.run
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8103
    callback()
    assert capsys.readouterr().out == (
        "  Ready: http://localhost:8103/v1  (Native MTP mode)\n  Model: model\n"
    )


def test_dspark_runner_defers_its_existing_banner_to_callback(monkeypatch, capsys):
    from rapid_mlx.models.deepseek_v41_native import server as dspark_server
    from rapid_mlx.speculative.dflash import server as dflash_server

    monkeypatch.setattr(dspark_server, "require_product_memory", lambda: None)
    monkeypatch.setattr(
        dspark_server, "download_target_snapshot", lambda: Path("/target")
    )
    monkeypatch.setattr(dspark_server, "download_mtp_snapshot", lambda: Path("/mtp"))
    monkeypatch.setattr(
        dspark_server,
        "load_product_runtime",
        lambda *_args, **_kwargs: (object(), object(), object()),
    )
    monkeypatch.setattr(dflash_server, "_dflash_executor", _ImmediateExecutor())
    monkeypatch.setattr(dflash_server, "_build_app", lambda **_kwargs: "dspark-app")
    calls: list[tuple[object, dict]] = []
    monkeypatch.setattr(
        "rapid_mlx._uvicorn.run_uvicorn",
        lambda app, **kwargs: calls.append((app, kwargs)),
    )

    dspark_server.run_server(
        host="0.0.0.0",
        port=8104,
        served_model_name="model",
        default_max_tokens=32,
        cors_origins=[],
        uvicorn_log_level="warning",
    )

    assert "Ready:" not in capsys.readouterr().out
    app, kwargs = calls[0]
    assert app == "dspark-app"
    callback = kwargs.pop("on_server_accepting")
    assert kwargs == {
        "host": "0.0.0.0",
        "port": 8104,
        "log_level": "warning",
        "timeout_keep_alive": 30,
    }
    callback()
    assert capsys.readouterr().out == (
        "  Ready: http://localhost:8104/v1  (DSpark K4 serial mode)\n"
        "  Docs:  http://localhost:8104/docs\n\n"
    )
