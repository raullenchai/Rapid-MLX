# SPDX-License-Identifier: Apache-2.0
"""Service qualification probes on dedicated speculative servers."""

import concurrent.futures
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import uvicorn

from rapid_mlx.config import reset_config
from rapid_mlx.headless_service.install import _readyz_ready, _wait_qualified
from rapid_mlx.speculative.ddtree.runtime import DDTreeRuntime
from rapid_mlx.speculative.ddtree.server import _build_app as build_ddtree_app
from rapid_mlx.speculative.dflash.runtime import DFlashRuntime
from rapid_mlx.speculative.dflash.server import _build_app as build_dflash_app


def _fake_runtime():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "user: 2+2?\nassistant:"
    tokenizer.encode.return_value = [10, 11, 12]
    target = MagicMock()
    target.tokenizer = tokenizer
    generator = MagicMock()
    generator.target = target
    return DDTreeRuntime(
        generator=generator,
        main_model_repo="mlx-community/Qwen3.5-9B-8bit",
        drafter_repo="z-lab/Qwen3.5-9B-DFlash",
        speculative_tokens=16,
        tree_budget=24,
    )


@pytest.fixture(autouse=True)
def clean_config():
    reset_config()
    yield
    reset_config()


@pytest.mark.parametrize(
    "content_length,send_body,expected",
    [
        (b"37", True, True),
        (None, True, True),
        (b"37", False, False),
        (b"abc", True, False),
        (b"-1", True, False),
    ],
    ids=[
        "content-length",
        "read-until-close",
        "missing-body",
        "malformed-content-length",
        "negative-content-length",
    ],
)
def test_readyz_reads_segmented_response(content_length, send_body, expected):
    body = b'{"ready": true, "model_loaded": true}'
    headers = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
    if content_length is not None:
        headers += b"Content-Length: " + content_length + b"\r\n"
    headers += b"\r\n"

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(2.0)
        port = listener.getsockname()[1]

        def respond():
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(2.0)
                request = b""
                while b"\r\n\r\n" not in request:
                    chunk = connection.recv(4096)
                    if not chunk:
                        break
                    request += chunk
                connection.sendall(headers)
                time.sleep(0.05)
                if send_body:
                    connection.sendall(body)
            return request

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            response = executor.submit(respond)
            ready = _readyz_ready("127.0.0.1", port, require_model_loaded=True)
            request = response.result(timeout=3)

    assert ready is expected
    assert b"Connection: close\r\n" in request


@contextmanager
def running_server(app):
    # Hand uvicorn the already-bound socket so no other process can take
    # the port between choosing it and serving on it.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started, "server did not start"
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        assert not thread.is_alive(), "server did not stop"


@pytest.mark.parametrize("api_key", [None, "secret"])
@pytest.mark.parametrize("server_type", ["dflash", "ddtree"])
def test_service_qualification(server_type, api_key):
    common = dict(
        served_model_name="test-model",
        default_max_tokens=64,
        cors_origins=[],
        api_key=api_key,
    )
    if server_type == "dflash":
        runtime = DFlashRuntime(
            drafter=MagicMock(),
            kind="dflash",
            drafter_repo="drafter",
            target_revision="a" * 40,
            drafter_revision="b" * 40,
        )
        app = build_dflash_app(
            model=MagicMock(), processor=MagicMock(), runtime=runtime, **common
        )
    else:
        app = build_ddtree_app(runtime=_fake_runtime(), **common)
    with running_server(app) as port:
        assert _wait_qualified(
            SimpleNamespace(host="127.0.0.1", port=port), ready_timeout_s=10
        )
        response = httpx.get(f"http://127.0.0.1:{port}/livez")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}


def test_ddtree_loading_service_probes():
    future = concurrent.futures.Future()
    app = build_ddtree_app(
        runtime_future=future,
        served_model_name="test-model",
        default_max_tokens=64,
        cors_origins=[],
        api_key="secret",
    )
    with running_server(app) as port:
        assert not _readyz_ready("127.0.0.1", port, require_model_loaded=True)
        assert httpx.get(f"http://127.0.0.1:{port}/readyz").status_code == 503
        assert httpx.get(f"http://127.0.0.1:{port}/livez").status_code == 200
        future.set_result(_fake_runtime())
        deadline = time.monotonic() + 10
        while not _readyz_ready("127.0.0.1", port, require_model_loaded=True):
            assert time.monotonic() < deadline
            time.sleep(0.01)
