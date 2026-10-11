# SPDX-License-Identifier: Apache-2.0
"""Attack and compatibility contracts for local browser transport defaults."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
