# SPDX-License-Identifier: Apache-2.0
"""Headless probes must pass the real default HTTP Host boundary (#4512)."""

import asyncio
import http.client
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from starlette.responses import JSONResponse

from rapid_mlx.headless_service.install import _probe_host, _readyz_ready
from rapid_mlx.headless_service.status import _endpoint_health
from rapid_mlx.middleware.browser_security import (
    DEFAULT_TRUSTED_HOSTS,
    BrowserSecurityMiddleware,
)


@contextmanager
def secured_endpoint(bind_host, *, hosts=DEFAULT_TRUSTED_HOSTS):
    requests = []

    async def healthy(scope, receive, send):
        requests.append(dict(scope["headers"]))
        await JSONResponse({"ready": True, "model_loaded": True})(scope, receive, send)

    app = BrowserSecurityMiddleware(healthy, hosts=list(hosts), deny_cross_origin=True)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            messages = []

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                messages.append(message)

            asyncio.run(
                app(
                    {
                        "type": "http",
                        "scheme": "http",
                        "method": "GET",
                        "path": self.path,
                        "headers": [
                            (key.lower().encode("ascii"), value.encode("latin-1"))
                            for key, value in self.headers.items()
                        ],
                    },
                    receive,
                    send,
                )
            )
            start = messages[0]
            body = b"".join(message.get("body", b"") for message in messages[1:])
            headers = b"".join(
                key + b": " + value + b"\r\n" for key, value in start["headers"]
            )
            # Keep framing out of this Host-policy regression: the existing
            # qualification receiver reads one chunk, unlike the status reader.
            self.connection.sendall(
                f"HTTP/1.1 {start['status']} Response\r\n".encode("ascii")
                + headers
                + b"\r\n"
                + body
            )

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
        daemon_threads = True

    try:
        server = Server((bind_host, 0), Handler)
    except OSError:
        if Server.address_family == socket.AF_INET6:
            pytest.skip("IPv6 loopback is unavailable")
        raise
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive(), "probe test server did not stop"


@pytest.mark.parametrize(
    "bind_host,configured_host",
    [
        ("127.0.0.1", "127.0.0.1"),
        ("127.0.0.1", "localhost"),
        ("0.0.0.0", "0.0.0.0"),
        ("::1", "::1"),
        ("::", "::"),
    ],
)
def test_service_probes_pass_default_host_boundary(bind_host, configured_host):
    with secured_endpoint(bind_host) as (port, requests):
        destination = _probe_host(configured_host)
        # Do not fix qualification by relaxing the browser security boundary.
        connection = http.client.HTTPConnection(destination, port, timeout=2)
        try:
            connection.request("GET", "/readyz", headers={"Host": "x"})
            assert connection.getresponse().status == 400
        finally:
            connection.close()

        assert _readyz_ready(configured_host, port, require_model_loaded=True)
        assert _endpoint_health(configured_host, port) == (True, True)
        assert len(requests) == 3
        authority = f"[{destination}]" if ":" in destination else destination
        assert all(
            request[b"host"] == f"{authority}:{port}".encode("ascii")
            for request in requests
        )
        assert all(b"authorization" not in request for request in requests)


def test_legacy_dns_probe_authority_uses_idna(monkeypatch):
    original_connect = socket.create_connection
    connected_hosts = []

    def resolve_local(address, **kwargs):
        host, port = address
        connected_hosts.append(host)
        assert host == "café.localhost"
        return original_connect(("127.0.0.1", port), **kwargs)

    # Resolve the legacy hostname locally without depending on external DNS.
    monkeypatch.setattr(socket, "create_connection", resolve_local)
    with secured_endpoint("127.0.0.1", hosts=["xn--caf-dma.localhost"]) as (
        port,
        requests,
    ):
        assert _readyz_ready("café.localhost", port, require_model_loaded=True)
        assert _endpoint_health("café.localhost", port) == (True, True)
        assert connected_hosts == ["café.localhost"] * 3
        assert len(requests) == 3
        assert all(
            request[b"host"] == f"xn--caf-dma.localhost:{port}".encode("ascii")
            for request in requests
        )
        assert all(b"authorization" not in request for request in requests)
