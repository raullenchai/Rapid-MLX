# SPDX-License-Identifier: Apache-2.0
"""Attack and compatibility contracts for local browser transport defaults."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from rapid_mlx.middleware import browser_security as security


@pytest.fixture
def local_app(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_TRUSTED_HOSTS", raising=False)
    monkeypatch.setattr(security, "_configured_hosts", None)
    app = FastAPI()
    calls = []

    @app.api_route("/probe", methods=["GET", "POST", "DELETE"])
    async def probe():
        calls.append(True)
        return {"ok": True}

    security.install_browser_security(app, origins=[])
    return app, calls


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "localhost:8898",
        "127.0.0.1:8898",
        "[::1]",
        "[::1]:8898",
        "app.localhost:8898",
        "LOCALHOST:8898",
    ],
)
def test_default_local_hosts_allowed(local_app, host):
    app, calls = local_app
    r = TestClient(app).get("/probe", headers={"Host": host})
    assert r.status_code == 200
    assert calls == [True]


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        "localhost.evil.example",
        "evil-localhost",
        "localhost@evil.example",
        "localhost/evil",
        "localhost,evil.example",
        "localhost:bad",
        "[::1]:bad",
        "",
        "[::1",
        "localhost:65536",
        "[::1]evil",
        "[::1]:",
        "localhost:",
        ":8898",
        "local\x00host",
    ],
)
def test_rebinding_and_malformed_authorities_do_not_dispatch(local_app, host):
    app, calls = local_app
    r = TestClient(app).get("/probe", headers={"Host": host})
    assert r.status_code == 400
    assert not calls


def test_duplicate_host_is_rejected(local_app):
    app, calls = local_app
    r = TestClient(app).get(
        "/probe", headers=[("Host", "localhost"), ("Host", "evil.example")]
    )
    assert r.status_code == 400
    assert not calls


@pytest.mark.parametrize(
    "headers",
    [
        [("Host", "evil.example")],
        [("Host", "localhost:8898"), ("Origin", "https://evil.example")],
        [("Host", "localhost:8898"), ("Origin", "null")],
        [
            ("Host", "localhost:8898"),
            ("Origin", "http://localhost:8898"),
            ("Origin", "http://localhost:8898"),
        ],
    ],
)
def test_websocket_policy_violation_closes_before_dispatch(local_app, headers):
    app, calls = local_app

    @app.websocket("/socket")
    async def socket(websocket: WebSocket):
        calls.append(True)
        await websocket.accept()

    client = TestClient(app)
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect(
            "ws://localhost:8898/socket", headers=httpx.Headers(headers)
        ),
    ):
        pytest.fail("Rejected WebSocket reached the application")
    assert exc.value.code == 1008
    assert not calls


@pytest.mark.parametrize("scheme,origin_scheme", [("ws", "http"), ("wss", "https")])
@pytest.mark.parametrize("native", [True, False])
def test_native_and_same_origin_websockets_dispatch(
    local_app, scheme, origin_scheme, native
):
    app, calls = local_app

    @app.websocket("/socket")
    async def socket(websocket: WebSocket):
        calls.append(True)
        await websocket.accept()
        await websocket.send_json({"ok": True})
        await websocket.close()

    headers = {} if native else {"Origin": f"{origin_scheme}://localhost:8898"}
    with (
        TestClient(app) as client,
        client.websocket_connect(
            f"{scheme}://localhost:8898/socket", headers=headers
        ) as ws,
    ):
        assert ws.receive_json() == {"ok": True}
    assert calls == [True]


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "null",
        "http://localhost:9999",
        "http://localhost.evil.example:8898",
        "https://localhost:8898",
        "http://localhost:8898/",
        "http://localhost:8898@evil.example",
    ],
)
@pytest.mark.parametrize("method", ["GET", "POST", "DELETE", "OPTIONS"])
def test_foreign_origins_cannot_read_or_mutate(local_app, origin, method):
    app, calls = local_app
    client = TestClient(app, base_url="http://localhost:8898")
    r = client.request(
        method,
        "/probe",
        content='{"messages":[]}',
        headers={"Origin": origin, "Content-Type": "text/plain"},
    )
    assert r.status_code == 403
    assert "access-control-allow-origin" not in r.headers
    assert not calls


@pytest.mark.parametrize(
    "base,origin",
    [
        ("http://localhost:8898", "http://localhost:8898"),
        ("http://localhost", "http://localhost:80"),
        ("http://[::1]:8898", "http://[::1]:8898"),
    ],
)
def test_same_origin_browser_and_native_clients_work(local_app, base, origin):
    app, calls = local_app
    client = TestClient(
        app, base_url="http://localhost:8898", headers={"Host": base.split("//", 1)[1]}
    )
    assert client.post("/probe", headers={"Origin": origin}).status_code == 200
    assert client.post("/probe").status_code == 200
    assert len(calls) == 2


LOOPBACK_SPELLINGS = ("localhost", "127.0.0.1", "[::1]")


def _check_browser_origin(app, calls, scheme, host, origin, *, allowed):
    """Exercise the shared boundary through both public browser transports."""
    headers = {"Host": host, "Origin": origin}
    client = TestClient(app)
    if scheme in ("ws", "wss"):

        @app.websocket("/socket")
        async def socket(websocket: WebSocket):
            calls.append(True)
            await websocket.accept()
            await websocket.send_json({"ok": True})
            await websocket.close()

        if allowed:
            with client.websocket_connect(
                f"{scheme}://localhost/socket", headers=headers
            ) as ws:
                assert ws.receive_json() == {"ok": True}
        else:
            with (
                pytest.raises(WebSocketDisconnect) as exc,
                client.websocket_connect(
                    f"{scheme}://localhost/socket", headers=headers
                ),
            ):
                pytest.fail("Rejected WebSocket reached the application")
            assert exc.value.code == 1008
    else:
        response = client.post(f"{scheme}://localhost/probe", headers=headers)
        assert response.status_code == (200 if allowed else 403)
        if not allowed:
            assert response.text == "Cross-origin requests disabled"
            assert "access-control-allow-origin" not in response.headers
    assert calls == ([True] if allowed else [])


@pytest.mark.parametrize("scheme", ["http", "https", "ws", "wss"])
@pytest.mark.parametrize(
    "port_style", ["explicit", "implicit", "origin-default", "host-default"]
)
@pytest.mark.parametrize(
    "host_name,origin_name",
    [
        (host, origin)
        for host in LOOPBACK_SPELLINGS
        for origin in LOOPBACK_SPELLINGS
        if host != origin
    ],
)
def test_loopback_origin_aliases_work(
    local_app, scheme, port_style, host_name, origin_name
):
    app, calls = local_app
    origin_scheme = {"ws": "http", "wss": "https"}.get(scheme, scheme)
    default_port = 443 if origin_scheme == "https" else 80
    host_port, origin_port = {
        "explicit": (":8898", ":8898"),
        "implicit": ("", ""),
        "origin-default": ("", f":{default_port}"),
        "host-default": (f":{default_port}", ""),
    }[port_style]
    _check_browser_origin(
        app,
        calls,
        scheme,
        f"{host_name}{host_port}",
        f"{origin_scheme}://{origin_name}{origin_port}",
        allowed=True,
    )


@pytest.mark.parametrize("scheme", ["http", "ws"])
@pytest.mark.parametrize("deny_cross_origin", [True, False])
@pytest.mark.parametrize(
    "host,origin",
    [
        ("127.0.0.1:8898", origin)
        for origin in [
            "http://localhost.evil.com:8898",
            "http://localhost.evil.example:8898",
            "http://evil-localhost:8898",
            "http://127.0.0.1.evil.com:8898",
            "http://app.localhost:8898",
            "http://127.0.0.2:8898",
            "http://0.0.0.0:8898",
            "http://[::ffff:127.0.0.1]:8898",
            "http://localhost:9999",
            "https://127.0.0.1:8898",
            "http://localhost:8898@evil.example",
            "http://evil.example:8898",
            "http://localhost:8898/",
            "http://localhost:8898,evil.example",
            "http://local host:8898",
            "http://localhost:bad",
            "http://localhost:65536",
            "http://[::1]:bad",
            "http://[::1]:65536",
            "null",
        ]
    ]
    + [
        ("myhost.lan:8898", "http://localhost:8898"),
        ("localhost:8898", "http://myhost.lan:8898"),
        ("app.localhost:8898", "http://localhost:8898"),
        ("127.0.0.2:8898", "http://localhost:8898"),
        ("0.0.0.0:8898", "http://localhost:8898"),
        ("[::ffff:127.0.0.1]:8898", "http://localhost:8898"),
    ],
)
def test_loopback_origin_boundary(local_app, scheme, deny_cross_origin, host, origin):
    app, calls = local_app
    # Explicitly trust these Hosts so Origin policy must enforce the boundary.
    app.user_middleware.clear()
    app.add_middleware(
        security.BrowserSecurityMiddleware,
        hosts=[
            *security.DEFAULT_TRUSTED_HOSTS,
            "myhost.lan",
            "127.0.0.2",
            "0.0.0.0",
            "[::ffff:127.0.0.1]",
        ],
        deny_cross_origin=deny_cross_origin,
    )
    _check_browser_origin(
        app, calls, scheme, host, origin, allowed=not deny_cross_origin
    )


def test_explicit_cors_does_not_disable_host_protection(monkeypatch):
    import rapid_mlx.server as server

    monkeypatch.setattr(security, "_configured_hosts", None)
    monkeypatch.delenv("RAPID_MLX_TRUSTED_HOSTS", raising=False)
    app = FastAPI()
    app.get("/probe")(lambda: {"ok": True})
    monkeypatch.setattr(server, "app", app)
    server.configure_cors_from_env(["https://frontend.example"])
    server.configure_trusted_hosts()
    client = TestClient(app, base_url="http://localhost")
    r = client.get("/probe", headers={"Origin": "https://frontend.example"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "https://frontend.example"
    assert client.get("/probe", headers={"Host": "evil.example"}).status_code == 400


def test_cli_host_policy_is_shared_by_dedicated_servers(monkeypatch):
    import rapid_mlx.server as server

    monkeypatch.setattr(security, "_configured_hosts", None)
    monkeypatch.setattr(server, "app", FastAPI())
    server.configure_trusted_hosts(["lan.example"])
    app = FastAPI()
    app.get("/probe")(lambda: {"ok": True})
    security.install_browser_security(app, origins=[])
    assert (
        TestClient(app, base_url="http://lan.example").get("/probe").status_code == 200
    )
    assert (
        TestClient(app, base_url="http://evil.example").get("/probe").status_code == 400
    )


def test_explicit_empty_cli_cors_overrides_wildcard_env(monkeypatch):
    import rapid_mlx.server as server

    app = FastAPI()
    app.post("/probe")(lambda: {"ok": True})
    monkeypatch.setattr(server, "app", app)
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", "*")
    assert server.configure_cors_from_env([]) == []
    assert (
        TestClient(app, base_url="http://localhost")
        .post("/probe", headers={"Origin": "https://evil.example"})
        .status_code
        == 403
    )


@pytest.mark.parametrize(
    "host,shown",
    [
        ("EVIL.example:8123", "evil.example"),
        ("[2001:db8::1]:8123", "2001:db8::1"),
        ("evil.example/path?secret", "<invalid>"),
        ("user:secret@evil.example", "<invalid>"),
        ("evil.example\r\nsecret", "<invalid>"),
        ("evil.example%secret", "<invalid>"),
        ("[::1%secret]:8123", "<invalid>"),
        ("--secret", "<invalid>"),
        ("[gg::1]:8123", "<invalid>"),
    ],
)
def test_host_rejection_is_actionable_and_sanitized(local_app, caplog, host, shown):
    app, calls = local_app
    response = TestClient(app).get("/probe", headers={"Host": host})
    expected = f"Invalid host header: {shown}. Configure --trusted-hosts <name> to allow this hostname."
    assert response.status_code == 400
    assert response.text == expected
    assert [
        (r.levelname, r.message) for r in caplog.records if r.name == security.__name__
    ] == [("WARNING", expected)]
    assert not calls
