# SPDX-License-Identifier: Apache-2.0
"""Probe access noise must not obscure unhealthy probes or ordinary requests."""

import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import uvicorn

from rapid_mlx._uvicorn import AcceptingConnectionsServer, _SuccessfulProbeFilter


@pytest.fixture
def access_logger():
    logger = logging.getLogger("uvicorn.access")
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        yield logger
    finally:
        logger.setLevel(previous)


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("target", ["/healthz", "/livez", "/healthz?ts=123"])
@pytest.mark.parametrize("status", [200, 204, 299])
def test_successful_probes_are_quiet(access_logger, method, target, status):
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:123", method, target, "1.1", status),
        None,
    )
    assert not _SuccessfulProbeFilter().filter(record)
    access_logger.setLevel(logging.DEBUG)
    assert _SuccessfulProbeFilter().filter(record)


@pytest.mark.parametrize(
    ("method", "target", "status"),
    [
        ("GET", "/healthz", status)
        for status in (199, 300, 301, 401, 404, 405, 429, 500, 503)
    ]
    + [
        ("POST", "/healthz", 200),
        ("GET", "/healthz/", 200),
        ("GET", "/healthz/detail", 200),
        ("GET", "/v1/models", 200),
        ("GET", "/v1/models/residency", 200),
        ("GET", "/readyz", 200),
        ("GET", "/healthz", "200"),
        ("GET", None, 200),
    ],
)
def test_other_access_records_remain_visible(access_logger, method, target, status):
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:123", method, target, "1.1", status),
        None,
    )
    assert _SuccessfulProbeFilter().filter(record)


@pytest.mark.parametrize(
    ("name", "level", "message", "args"),
    [
        ("uvicorn.access", logging.INFO, "application status", ()),
        ("uvicorn.access", logging.INFO, '%s - "%s %s HTTP/%s" %d', ("short",)),
        (
            "uvicorn.error",
            logging.INFO,
            '%s - "%s %s HTTP/%s" %d',
            ("client", "GET", "/healthz", "1.1", 200),
        ),
        (
            "uvicorn.access",
            logging.WARNING,
            '%s - "%s %s HTTP/%s" %d',
            ("client", "GET", "/healthz", "1.1", 200),
        ),
    ],
)
def test_unrecognized_or_non_access_records_fail_open(
    access_logger, name, level, message, args
):
    record = logging.LogRecord(name, level, "", 0, message, args, None)
    assert _SuccessfulProbeFilter().filter(record)


@pytest.mark.parametrize("failure", [None, RuntimeError("serve failed")])
def test_filter_removed_after_server_run(monkeypatch, access_logger, failure):
    before = access_logger.filters[:]

    def run(self, sockets=None):
        assert len(access_logger.filters) == len(before) + 1
        assert isinstance(access_logger.filters[-1], _SuccessfulProbeFilter)
        if failure:
            raise failure

    monkeypatch.setattr(uvicorn.Server, "run", run)
    server = AcceptingConnectionsServer(uvicorn.Config("unused:app", log_config=None))
    if failure:
        with pytest.raises(RuntimeError, match="serve failed"):
            server.run()
    else:
        server.run()
    assert access_logger.filters == before


@pytest.mark.parametrize("level", ["info", "debug"])
def test_real_uvicorn_probe_logs(level):
    program = textwrap.dedent("""\
        import http.client
        import logging
        import os
        import signal
        import socket
        import sys
        import threading
        from rapid_mlx._uvicorn import run_uvicorn

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        errors = []

        async def app(scope, receive, send):
            status = 503 if scope["query_string"] == b"draining=1" else 200
            await send({"type": "http.response.start", "status": status, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        def probe():
            try:
                for method, path, expected in [
                    ("GET", "/healthz?tick=1", 200),
                    ("HEAD", "/livez", 200),
                    ("GET", "/healthz?draining=1", 503),
                    ("GET", "/v1/models", 200),
                    ("POST", "/healthz", 200),
                ]:
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    try:
                        conn.request(method, path)
                        response = conn.getresponse()
                        assert response.status == expected
                        response.read()
                    finally:
                        conn.close()
                logging.getLogger("uvicorn.error").info("APPLICATION_INFO_VISIBLE")
            except BaseException as exc:
                errors.append(exc)
            finally:
                os.kill(os.getpid(), signal.SIGINT)

        worker = threading.Thread(target=probe)
        run_uvicorn(app, fd=listener.fileno(), lifespan="off", log_level=sys.argv[1],
                    on_server_accepting=worker.start)
        worker.join(timeout=5)
        listener.close()
        assert not worker.is_alive()
        assert not errors, errors
    """)
    result = subprocess.run(
        [sys.executable, "-c", program, level],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert ('GET /healthz?tick=1 HTTP/1.1" 200' in output) == (level == "debug"), output
    assert ('HEAD /livez HTTP/1.1" 200' in output) == (level == "debug"), output
    for visible in [
        'GET /healthz?draining=1 HTTP/1.1" 503',
        'GET /v1/models HTTP/1.1" 200',
        'POST /healthz HTTP/1.1" 200',
        "APPLICATION_INFO_VISIBLE",
    ]:
        assert visible in output, output
