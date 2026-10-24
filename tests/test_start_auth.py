# SPDX-License-Identifier: Apache-2.0
"""CPU-only authentication regressions for reuse of a running server."""

import io
import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from rapid_mlx import http_auth
from rapid_mlx.config import get_config
from rapid_mlx.run import cli


@pytest.fixture(autouse=True)
def isolated_key(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    monkeypatch.setattr(get_config(), "api_key", None)


@contextmanager
def server(reply):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status, headers, body = reply(self)
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("source", ["missing", "wrong", "env", "config", "precedence"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_start_reuses_authenticated_server(
    monkeypatch, capsys, status, source, dry_run
):
    if source in ("env", "precedence"):
        monkeypatch.setenv("RAPID_MLX_API_KEY", "test-secret")
    if source == "wrong":
        monkeypatch.setenv("RAPID_MLX_API_KEY", "wrong-secret")
    if source in ("config", "precedence"):
        monkeypatch.setattr(
            get_config(),
            "api_key",
            "test-secret" if source == "config" else "wrong-secret",
        )
    seen = []

    def reply(request):
        seen.append((request.path, request.headers.get("Authorization")))
        if seen[-1][1] != "Bearer test-secret":
            return status, {}, b"auth denied"
        return (
            200,
            {},
            json.dumps({"data": [{"id": "other"}, {"id": "wanted"}]}).encode(),
        )

    attached = []
    monkeypatch.setattr(cli, "_hf_id", lambda value: value)
    monkeypatch.setattr(
        cli, "_attach_and_configure", lambda *args: attached.append(args) or 0
    )
    with server(reply) as base:
        result = cli._reuse_or_refuse(
            base,
            "wanted",
            "wanted",
            None,
            SimpleNamespace(port=int(base.rsplit(":", 1)[1]), dry_run=dry_run),
        )
    output = capsys.readouterr().out
    assert seen[0][0] == "/v1/models"
    if source in ("missing", "wrong"):
        assert result == 1
        assert not attached
        assert "server requires an API key" in output
        assert "RAPID_MLX_API_KEY" in output
        assert "not a healthy" not in output
    else:
        assert result == 0
        assert len(attached) == 1
        assert "reuse" in output.lower() or "reusing" in output.lower()
    assert "test-secret" not in output
    assert "wrong-secret" not in output


@pytest.mark.parametrize(
    "base,allowed",
    [
        ("http://127.0.0.1:1234/v1", True),
        ("http://[::1]:1234/v1", True),
        ("http://localhost:1234/v1", True),
        ("https://example.test/v1", True),
        ("http://example.test/v1", False),
        ("http://192.0.2.1/v1", False),
    ],
)
def test_credential_transport_policy(monkeypatch, base, allowed):
    monkeypatch.setenv("RAPID_MLX_API_KEY", "test-secret")

    def open_request(request, timeout):
        assert timeout == 2
        assert request.get_header("Authorization") == (
            "Bearer test-secret" if allowed else None
        )
        assert "Authorization" not in request.headers
        return io.BytesIO(b'{"data": [{"id": "m"}]}')

    monkeypatch.setattr(urllib.request, "urlopen", open_request)
    assert http_auth.probe_models(base) == [{"id": "m"}]


def test_redirect_does_not_receive_key(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_API_KEY", "test-secret")
    received = []

    def target(request):
        received.append(request.headers.get("Authorization"))
        return 401, {}, b"denied"

    with server(target) as destination:

        def redirect(request):
            assert request.headers.get("Authorization") == "Bearer test-secret"
            return 302, {"Location": destination + "/v1/models"}, b""

        with server(redirect) as source, pytest.raises(PermissionError):
            http_auth.probe_models(source + "/v1")
    assert received == [None]


@pytest.mark.parametrize(
    "body,expected",
    [
        (b'{"data": [{"id":"public"}, null, 1]}', [{"id": "public"}]),
        (b'{"data": []}', []),
        (b'{"data": {}}', []),
        (b"[]", []),
        (b"invalid", []),
    ],
)
def test_public_and_malformed_responses(monkeypatch, body, expected):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body))
    assert http_auth.probe_models("http://127.0.0.1/v1") == expected


@pytest.mark.parametrize("status", [401, 403, 500])
def test_http_error_stream_closed(monkeypatch, status):
    stream = io.BytesIO(b"error")

    def fail(*args, **kwargs):
        raise urllib.error.HTTPError(
            "http://127.0.0.1/v1/models", status, "error", {}, stream
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    if status in (401, 403):
        with pytest.raises(PermissionError):
            http_auth.probe_models("http://127.0.0.1/v1")
    else:
        assert http_auth.probe_models("http://127.0.0.1/v1") == []
    assert stream.closed


@pytest.mark.parametrize(
    "error", [urllib.error.URLError("offline"), TimeoutError(), OSError()]
)
def test_unavailable_server(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    assert http_auth.probe_models("http://127.0.0.1/v1") == []
