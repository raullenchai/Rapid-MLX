import json
import sys
import time
from collections import deque

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx import cli
from rapid_mlx.config import get_config
from rapid_mlx.control import RUNNER_ENV, SECRET_ENV, SOCKET_ENV, call_log, protocol
from rapid_mlx.routes.control_status import mount_status


async def test_recording():
    calls, sent = deque(maxlen=1000), []

    async def app(s, receive, send):
        if s.get("fail"):
            raise ValueError("boom")
        await send({"type": "http.response.start", "status": 201})
        await send({"type": "http.response.start", "status": 202})
        await send({"type": "http.response.body", "body": b"secret"})

    async def send(message):
        sent.append(message)

    log = call_log.CallLog(app, calls)
    s = dict(type="http", method="POST", path="/v1/messages", query_string=b"secret")
    s["headers"] = [(b"x-rapid-client", "a\x1b\n\u202e".encode() + b"b" * 50)]
    await log(s, None, send)
    assert " ".join(sorted(calls[0])) == "client duration_ms method path status ts"
    assert calls[0]["client"] == "a" + "b" * 39
    assert calls[0]["status"] == 201 and calls[0]["path"] == "/v1/messages"
    assert isinstance(calls[0]["duration_ms"], int) and calls[0]["duration_ms"] >= 0
    for path in ("/v1/models", "/health", "/rapid/v1/log", "/anthropic/messages"):
        await log(dict(s, path=path), None, send)
    assert len(calls) == 1 and sent[-1]["body"] == b"secret"
    with pytest.raises(ValueError):
        await log(dict(s, fail=True), None, send)
    assert calls[-1]["status"] == 500
    s.update(path="/v1/messages", headers=[(b"user-agent", b"curl/1.2 other")])
    for _ in range(1001):
        await log(s, None, send)
    assert len(calls) == 1000 and calls[-1]["client"] == "curl"


def test_endpoint_cli(monkeypatch, capsys):
    monkeypatch.setattr(get_config(), "api_key", "key")
    monkeypatch.setenv(RUNNER_ENV, "1")
    monkeypatch.setenv(SOCKET_ENV, "/unused")
    monkeypatch.setenv(SECRET_ENV, "unused")
    mount_status(app := FastAPI())
    local = TestClient(app, client=("127.0.0.1", 1))
    local.headers["Authorization"] = "Bearer key"
    local.post("/v1/messages?secret=hidden")
    calls = local.get("/rapid/v1/log?limit=1").json()["calls"]
    assert calls[0]["path"] == "/v1/messages"
    assert local.get("/rapid/v1/log?limit=1001").status_code == 422
    for peer, code in (("127.0.0.1", 401), ("192.0.2.1", 403)):
        client = TestClient(app, client=(peer, 1))
        assert client.get("/rapid/v1/log").status_code == code
    monkeypatch.setattr(
        protocol, "discover", lambda port: [{"port": port or 8000}, {"port": 9000}]
    )
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: local.get(*a))
    for flags in ([], ["--json"], ["--port", "8001"]):
        monkeypatch.setattr(sys, "argv", ["rapid-mlx", "log", *flags])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        output = capsys.readouterr().out
        assert exc.value.code == 0
        if "--json" in flags:
            assert json.loads(output)["calls"]
        else:
            header, row = output.splitlines()
            assert f"serve :{8001 if flags else 8000}" in header
            assert row[:8] == time.strftime("%H:%M:%S", time.localtime(calls[0]["ts"]))
            assert header.index("PATH") == row.index("/v1/messages")
            assert len(row) == 114
    monkeypatch.setattr(protocol, "discover", lambda port: [])
    assert protocol.log_command(None) == 1
    assert "No running serve" in capsys.readouterr().err


@pytest.mark.parametrize(
    "failure",
    [
        302,
        401,
        403,
        500,
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
    ],
)
def test_cli_errors(monkeypatch, capsys, failure):
    monkeypatch.setattr(protocol, "discover", lambda port: [{"port": 8123}])

    def get(*a, **kw):
        assert kw["headers"] == {"Authorization": "Bearer key"}
        assert kw["trust_env"] is False and kw["timeout"] == 2
        if not isinstance(failure, int):
            raise failure("broken")
        return httpx.Response(failure)

    monkeypatch.setenv("RAPID_MLX_API_KEY", "key")
    monkeypatch.setattr(httpx, "get", get)
    assert protocol.log_command(type("Args", (), dict(limit=50))()) == 1
    expected = (
        "serve requires an API key: set RAPID_MLX_API_KEY"
        if failure in (401, 403)
        else f"serve returned HTTP {failure}"
        if isinstance(failure, int)
        else "could not reach serve on port 8123"
    )
    assert capsys.readouterr().err == expected + "\n"
