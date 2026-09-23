# SPDX-License-Identifier: Apache-2.0
"""U1 regression coverage for the post-bind Ready banner."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import socket
import sys
import tempfile
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
    elif scope["type"] == "http":
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


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
async def test_uds_listener_runs_callback_only_after_socket_accepts():
    observations: list[int] = []
    instance: AcceptingConnectionsServer
    with tempfile.TemporaryDirectory(prefix="u1-", dir="/tmp") as root:
        uds_path = Path(root) / "s"

        def on_accepting() -> None:
            with socket.socket(socket.AF_UNIX) as client:
                client.settimeout(1)
                observations.append(client.connect_ex(str(uds_path)))
            instance.should_exit = True

        config = uvicorn.Config(
            _asgi_app,
            uds=str(uds_path),
            log_level="error",
        )
        instance = AcceptingConnectionsServer(
            config,
            on_server_accepting=on_accepting,
        )

        await instance.serve()

    assert observations == [0]


@pytest.mark.asyncio
async def test_supplied_socket_runs_callback_only_after_socket_accepts():
    observations: list[int] = []
    instance: AcceptingConnectionsServer

    with socket.socket() as supplied:
        supplied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        supplied.bind(("127.0.0.1", 0))
        supplied.listen()
        address = supplied.getsockname()

        def on_accepting() -> None:
            with socket.socket() as client:
                client.settimeout(1)
                observations.append(client.connect_ex(address))
            instance.should_exit = True

        config = uvicorn.Config(_asgi_app, log_level="error")
        instance = AcceptingConnectionsServer(
            config,
            on_server_accepting=on_accepting,
        )
        await instance.serve(sockets=[supplied])

    assert observations == [0]


@pytest.mark.asyncio
@pytest.mark.parametrize("log_failure", [False, True])
async def test_throwing_callback_does_not_stop_bound_server(
    caplog, monkeypatch, log_failure
):
    callback_ran = asyncio.Event()

    def fail_after_bind() -> None:
        callback_ran.set()
        raise RuntimeError("banner output failed")

    if log_failure:

        def fail_to_log(*_args, **_kwargs) -> None:
            raise RuntimeError("logging failed")

        monkeypatch.setattr("rapid_mlx._uvicorn.logger.exception", fail_to_log)

    config = uvicorn.Config(
        _asgi_app,
        host="127.0.0.1",
        port=0,
        log_level="error",
    )
    instance = AcceptingConnectionsServer(
        config,
        on_server_accepting=fail_after_bind,
    )
    serve_task = asyncio.create_task(instance.serve())

    try:
        await asyncio.wait_for(callback_ran.wait(), timeout=1)
        assert not serve_task.done()
        listener = instance.servers[0].sockets[0]
        host, port = listener.getsockname()[:2]
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), timeout=1)
        writer.close()
        await writer.wait_closed()
        assert b"HTTP/1.1 200 OK" in response
        assert b"ok" in response
        assert instance._accepting_callback_ran is True
    finally:
        instance.should_exit = True
        if serve_task.done():
            for listener_server in getattr(instance, "servers", ()):
                listener_server.close()
                await listener_server.wait_closed()
        else:
            await serve_task

    if not log_failure:
        assert "banner output failed" in caplog.text


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


def _real_listener_runner(observations: list[int]):
    def run(app, **kwargs):
        uvicorn_main = importlib.import_module("uvicorn.main")
        instance = uvicorn_main.Server(uvicorn.Config(app, **kwargs))
        assert isinstance(instance, AcceptingConnectionsServer)

        async def serve_once() -> None:
            serve_task = asyncio.create_task(instance.serve())
            try:
                for _ in range(100):
                    if serve_task.done():
                        await serve_task
                    listeners = getattr(instance, "servers", ())
                    if instance.started and listeners and listeners[0].sockets:
                        break
                    await asyncio.sleep(0.01)
                else:
                    pytest.fail("real Uvicorn listener did not start")

                listener = instance.servers[0].sockets[0]
                port = listener.getsockname()[1]
                with socket.socket() as client:
                    client.settimeout(1)
                    observations.append(client.connect_ex(("127.0.0.1", port)))
            finally:
                instance.should_exit = True
                await serve_task

        asyncio.run(serve_once())

    return run


def test_standalone_entrypoint_uses_shared_post_bind_seam():
    source = inspect.getsource(server.main)
    assert "run_uvicorn(" in source
    assert "on_server_accepting=print_ready_banner" in source


def test_standalone_entrypoint_stashes_endpoint_and_runs_real_seam(
    monkeypatch, capsys, unused_tcp_port
):
    from rapid_mlx.config import get_config

    scheduler = ModuleType("rapid_mlx.scheduler")

    class SchedulerConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    scheduler.SchedulerConfig = SchedulerConfig
    monkeypatch.setitem(sys.modules, "rapid_mlx.scheduler", scheduler)
    turboquant = ModuleType("rapid_mlx.turboquant")
    turboquant.resolve_turboquant_mode_default = lambda *_args, **_kwargs: None
    turboquant.turboquant_scheduler_kwargs = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "rapid_mlx.turboquant", turboquant)
    observations: list[int] = []
    cfg = get_config()
    monkeypatch.setattr(cfg, "bind_host", cfg.bind_host)
    monkeypatch.setattr(cfg, "bind_port", cfg.bind_port)
    monkeypatch.setattr(cfg, "bind_listen_fd", cfg.bind_listen_fd)
    monkeypatch.setattr(cli, "_port_preflight_or_die", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_hard_exit_after_serve", lambda: None)
    monkeypatch.setattr(server, "_preflight_vision_runtime", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "load_model", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "app", _asgi_app)
    monkeypatch.setattr(uvicorn, "run", _real_listener_runner(observations))
    monkeypatch.setattr(
        "rapid_mlx._version_check.prompt_upgrade_if_available", lambda: False
    )
    monkeypatch.setattr(
        "rapid_mlx._version_check.print_staleness_warning_if_any",
        lambda **_kwargs: None,
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
            str(unused_tcp_port),
        ],
    )

    server.main()

    assert cfg.bind_host == "localhost"
    assert cfg.bind_port == unused_tcp_port
    assert cfg.bind_listen_fd is None
    assert observations == [0]
    assert capsys.readouterr().out.count("Ready:") == 1


class _ImmediateExecutor:
    def submit(self, function, *args, **kwargs):
        future: Future = Future()
        future.set_result(function(*args, **kwargs))
        return future


def test_ddtree_runner_uses_shared_seam_without_false_ready(monkeypatch, capsys):
    from rapid_mlx.speculative.ddtree import server as ddtree_server

    observations: list[int] = []
    pending: Future = Future()
    monkeypatch.setattr(ddtree_server, "have_runtime", lambda: True)
    monkeypatch.setattr(
        ddtree_server,
        "_ddtree_loader_executor",
        SimpleNamespace(submit=lambda *_args, **_kwargs: pending),
    )
    monkeypatch.setattr(ddtree_server, "_build_app", lambda **_kwargs: _asgi_app)
    monkeypatch.setattr(uvicorn, "run", _real_listener_runner(observations))

    ddtree_server.run_ddtree_server(
        main_model_repo="target",
        drafter_repo="drafter",
        speculative_tokens=4,
        tree_budget=8,
        host="127.0.0.1",
        port=0,
        served_model_name="model",
        default_max_tokens=32,
        cors_origins=[],
        uvicorn_log_level="warning",
    )

    output = capsys.readouterr().out
    assert "Starting: http://127.0.0.1:0/v1" in output
    assert "Ready:" not in output
    assert observations == [0]


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
    from rapid_mlx.models import deepseek_v41_native
    from rapid_mlx.speculative.dflash import server as dflash_server

    serving_name = "rapid_mlx.models.deepseek_v41_native.serving"
    fake_serving = ModuleType(serving_name)
    for name in (
        "generate",
        "generation_kwargs",
        "load_product_runtime",
        "render_prompt",
        "stream_generate",
        "validate_request",
    ):
        setattr(fake_serving, name, lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, serving_name, fake_serving)
    server_name = "rapid_mlx.models.deepseek_v41_native.server"
    monkeypatch.delitem(sys.modules, server_name, raising=False)
    monkeypatch.delattr(deepseek_v41_native, "server", raising=False)
    dspark_server = importlib.import_module(server_name)

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
