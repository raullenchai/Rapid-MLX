# SPDX-License-Identifier: Apache-2.0
"""Serve startup-only HTTP probes while the model-owning thread is busy.

The listener remains bound when handed to Uvicorn. This responder never runs
application routes or reports readiness; those belong to the normal ASGI app.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit


class _ProbeHandler(BaseHTTPRequestHandler):
    def handle_one_request(self) -> None:
        # A client which sends only part of a request must not hold startup
        # shutdown indefinitely. The responder handles no request bodies.
        self.connection.settimeout(0.2)
        super().handle_one_request()

    def _respond(self) -> None:
        body = {"detail": "model loading"}
        if (
            self.command in {"GET", "HEAD"}
            and urlsplit(self.path).path == "/health/ready"
        ):
            from .startup_progress import startup_progress

            body.update(ready=False, **startup_progress.snapshot())
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(encoded)
        self.close_connection = True

    do_GET = _respond  # noqa: N815 — stdlib HTTP handler protocol
    do_HEAD = _respond  # noqa: N815
    do_POST = _respond  # noqa: N815
    do_PUT = _respond  # noqa: N815
    do_PATCH = _respond  # noqa: N815
    do_DELETE = _respond  # noqa: N815
    do_OPTIONS = _respond  # noqa: N815

    def log_message(self, format, *args) -> None:
        pass


class _ProbeServer(HTTPServer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._connection_lock = threading.Lock()
        self._connection = None
        self._stopping = False

    def get_request(self):
        connection, address = super().get_request()
        with self._connection_lock:
            if self._stopping:
                connection.close()
                raise OSError("startup responder stopped")
            self._connection = connection
        return connection, address

    def shutdown_request(self, request) -> None:
        with self._connection_lock:
            if self._connection is request:
                self._connection = None
        super().shutdown_request(request)

    def stop_requests(self) -> None:
        # A per-read timeout alone does not stop clients sending partial
        # headers continuously. Interrupt the active connection before joining
        # the responder; the lock also covers an accept racing with shutdown.
        with self._connection_lock:
            self._stopping = True
            if self._connection is not None:
                try:
                    self._connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def handle_error(self, request, client_address) -> None:
        if not self._stopping:
            super().handle_error(request, client_address)


class StartupProbe:
    """Own a temporary accept loop, but never own the reserved listener."""

    def __init__(self, listener: socket.socket) -> None:
        self._server = _ProbeServer(
            listener.getsockname(), _ProbeHandler, bind_and_activate=False
        )
        self._server.socket.close()
        # A duplicate lets server_close release its descriptor without
        # unbinding the listener that Uvicorn will adopt next.
        self._server.socket = listener.dup()
        self._server.server_address = listener.getsockname()
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.02},
            name="startup-readiness",
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._server.stop_requests()
        self._server.shutdown()
        self._thread.join()
        self._server.server_close()
