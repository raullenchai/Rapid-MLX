# SPDX-License-Identifier: Apache-2.0
"""Real-socket regressions for startup progress and listener handoff."""

import asyncio
import http.client
import json
import socket
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest
import uvicorn

from rapid_mlx import startup_progress as progress_module
from rapid_mlx._startup_probe import StartupProbe
from rapid_mlx._uvicorn import AcceptingConnectionsServer


def _get(address, path="/health/ready"):
    connection = http.client.HTTPConnection(*address[:2], timeout=2)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


@pytest.fixture
def progress(monkeypatch):
    state = progress_module.StartupProgress()
    state.begin("tiny-fixture")
    assert state.claim_weights()
    state.weights_total(10)
    state.weights_done(5)
    monkeypatch.setattr(progress_module, "startup_progress", state)
    return state


def test_probe_only_exposes_progress_on_readiness_and_retains_socket(progress):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        address = listener.getsockname()
        probe = StartupProbe(listener)
        try:
            status, body = _get(address)
            assert status == 503
            assert body == {
                "ready": False,
                "detail": "model loading",
                **progress.snapshot(),
            }
            assert _get(address, "/v1/models") == (503, {"detail": "model loading"})
        finally:
            probe.close()
        assert listener.fileno() >= 0
        assert not probe._thread.is_alive()
        with socket.socket() as contender, pytest.raises(OSError):
            contender.bind(address)


def test_incomplete_http_request_cannot_hold_handoff(progress):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        probe = StartupProbe(listener)
        with socket.create_connection(listener.getsockname(), timeout=1) as client:
            client.sendall(b"GET /health/ready HTTP/1.1\r\nHost:")
            stop = threading.Event()

            def trickle():
                deadline = time.monotonic() + 2
                while not stop.wait(0.05) and time.monotonic() < deadline:
                    try:
                        client.sendall(b"x")
                    except OSError:
                        return

            sender = threading.Thread(target=trickle)
            sender.start()
            time.sleep(0.02)
            start = time.monotonic()
            try:
                probe.close()
            finally:
                stop.set()
                sender.join(timeout=1)
            assert time.monotonic() - start < 0.5
            assert not probe._thread.is_alive()


def test_trickling_client_cannot_monopolize_responder(progress):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        address = listener.getsockname()
        probe = StartupProbe(listener)
        stop = threading.Event()
        try:
            with socket.create_connection(address, timeout=1) as client:
                client.sendall(b"GET /health/ready HTTP/1.1\r\nHost:")

                def trickle():
                    while not stop.wait(0.05):
                        try:
                            client.sendall(b"x")
                        except OSError:
                            return

                sender = threading.Thread(target=trickle)
                sender.start()
                try:
                    start = time.monotonic()
                    status, _ = _get(address)
                    assert status == 503
                    assert time.monotonic() - start < 1.8
                finally:
                    stop.set()
                    sender.join(timeout=1)
        finally:
            probe.close()


@pytest.mark.asyncio
async def test_blocking_loader_exposes_intermediate_progress_before_ready(
    monkeypatch, progress
):
    """A loader blocking the ASGI loop must still expose real intermediate work.

    The loader cannot finish until an external HTTP client has observed 50%,
    making this fail on the original post-lifespan-only listener implementation.
    """
    accepting = threading.Event()
    observed = threading.Event()
    loading = threading.Event()
    client_results = []
    errors = []
    callbacks = []
    materialized = []
    chunk_bytes = progress_module._EVAL_BATCH_BYTES

    def evaluate(weights):
        materialized.extend(weights)
        if len(materialized) == 2:
            # The first chunk has completed and been counted. Keep the second
            # materialization outstanding until the HTTP client observes it.
            loading.set()
            assert observed.wait(4), "progress was not reachable during loading"

    def loader(*, lazy):
        assert lazy is True
        return (
            SimpleNamespace(
                parameters=lambda: {
                    "first": SimpleNamespace(nbytes=chunk_bytes),
                    "second": SimpleNamespace(nbytes=chunk_bytes),
                }
            ),
            None,
        )

    mlx = ModuleType("mlx")
    mlx.__path__ = []
    core = ModuleType("mlx.core")
    core.eval = evaluate
    utils = ModuleType("mlx.utils")
    utils.tree_flatten = lambda params: list(params.items())
    for name, module in (("mlx", mlx), ("mlx.core", core), ("mlx.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        address = reservation.getsockname()

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    progress.begin("tiny-fixture")
                    progress_module.load_with_progress(loader)
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        else:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b'{"ready":true}'})

    def on_accepting():
        callbacks.append("ready")
        accepting.set()

    instance = AcceptingConnectionsServer(
        uvicorn.Config(app, host=address[0], port=address[1], log_level="error"),
        on_server_accepting=on_accepting,
    )

    def client():
        try:
            assert loading.wait(3)
            status, body = _get(address)
            assert status == 503
            assert body["progress"] == {
                "done": chunk_bytes,
                "total": 2 * chunk_bytes,
                "unit": "bytes",
            }
            assert not callbacks
            client_results.append((status, body))
            observed.set()
            # Handoff intentionally interrupts active probe requests. Wait for
            # the application listener before opening the final connection.
            assert accepting.wait(3), "server never completed listener handoff"
            status, body = _get(address)
            assert status == 200
            client_results.append((status, body))
        except BaseException as exc:
            errors.append(exc)
        finally:
            instance.should_exit = True

    worker = threading.Thread(target=client)
    worker.start()
    try:
        await asyncio.wait_for(instance.serve(), timeout=10)
    finally:
        observed.set()
        worker.join(timeout=5)
    assert not errors
    assert [item[0] for item in client_results] == [503, 200]
    assert client_results[-1][1] == {"ready": True}
    assert callbacks == ["ready"]


@pytest.mark.asyncio
async def test_startup_failure_releases_probe_and_port(progress):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        address = reservation.getsockname()

    async def app(scope, receive, send):
        await receive()
        raise RuntimeError("fixture loader failed")

    instance = AcceptingConnectionsServer(
        uvicorn.Config(
            app, host=address[0], port=address[1], lifespan="on", log_level="error"
        )
    )
    with pytest.raises(SystemExit):
        await instance.serve()
    with socket.socket() as replacement:
        replacement.bind(address)
    assert not any(t.name == "startup-readiness" for t in threading.enumerate())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"host": "localhost"},
        {"host": None},
        {"host": ""},
        {"fd": 17},
        {"uds": "unused.sock"},
        {"ssl_certfile": "unused.pem"},
        {"workers": 2},
        {"reload": True},
    ],
)
async def test_specialized_transports_keep_original_startup(monkeypatch, options):
    calls = []

    async def original_startup(self, sockets=None):
        calls.append(sockets)

    monkeypatch.setattr(uvicorn.Server, "startup", original_startup)
    instance = AcceptingConnectionsServer(uvicorn.Config("unused:app", **options))
    instance.lifespan = SimpleNamespace(should_exit=False)
    await instance.startup()
    assert calls == [None]


def test_malformed_target_still_gets_loading_response(progress):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        probe = StartupProbe(listener)
        try:
            with socket.create_connection(listener.getsockname(), timeout=2) as client:
                client.sendall(b"GET http://[ HTTP/1.1\r\nHost: x\r\n\r\n")
                reply = b""
                while chunk := client.recv(4096):  # headers and body may split
                    reply += chunk
            assert reply.startswith(b"HTTP/1.0 503") or reply.startswith(
                b"HTTP/1.1 503"
            )
            assert b"model loading" in reply
        finally:
            probe.close()


@pytest.mark.asyncio
async def test_listener_option_failure_closes_reserved_socket(monkeypatch):
    created = []

    class FailingSocket(socket.socket):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def setsockopt(self, *args):
            raise OSError("unsupported option")

    monkeypatch.setattr(socket, "socket", FailingSocket)
    instance = AcceptingConnectionsServer(
        uvicorn.Config("unused:app", host="127.0.0.1", port=0, log_level="error")
    )
    instance.lifespan = SimpleNamespace(should_exit=False, startup=None)
    with pytest.raises(SystemExit):
        await instance.startup()
    assert created
    assert all(sock.fileno() == -1 for sock in created)


@pytest.mark.asyncio
@pytest.mark.skipif(not socket.has_ipv6, reason="IPv6 unavailable")
async def test_ipv6_listener_preserves_ipv6_only_binding():
    observed = []

    async def app(scope, receive, send):
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    def accepting():
        listener = instance.servers[0].sockets[0]
        observed.append(listener.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY))
        instance.should_exit = True

    instance = AcceptingConnectionsServer(
        uvicorn.Config(app, host="::1", port=0, log_level="error"),
        on_server_accepting=accepting,
    )
    await instance.serve()
    assert observed == [1]


def test_shutdown_ignores_closed_connection(monkeypatch):
    from rapid_mlx._startup_probe import _ProbeHandler, _ProbeServer

    class ClosedConnection:
        def shutdown(self, how):
            assert how == socket.SHUT_RDWR
            raise OSError("already closed")

    handler = object.__new__(_ProbeHandler)
    handler.connection = ClosedConnection()
    handler._abort()
    server = _ProbeServer(("127.0.0.1", 0), _ProbeHandler, bind_and_activate=False)
    try:
        server._connection = ClosedConnection()
        server.stop_requests()
        assert server._stopping
    finally:
        server.server_close()


def test_accept_during_shutdown_closes_connection(monkeypatch):
    from http.server import HTTPServer

    from rapid_mlx._startup_probe import _ProbeHandler, _ProbeServer

    connection = SimpleNamespace(close=lambda: closed.append(True))
    closed = []
    monkeypatch.setattr(HTTPServer, "get_request", lambda self: (connection, "address"))
    server = _ProbeServer(("127.0.0.1", 0), _ProbeHandler, bind_and_activate=False)
    try:
        server.stop_requests()
        with pytest.raises(OSError, match="startup responder stopped"):
            server.get_request()
        assert closed == [True]
        assert server._connection is None
    finally:
        server.server_close()
