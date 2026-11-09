# SPDX-License-Identifier: Apache-2.0
"""Phone bootstrap, socket boundaries, discovery and secret handling."""

import argparse
import json
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlsplit

import httpx
import psutil
import pytest
from fastapi import Depends, FastAPI

from rapid_mlx import phone_connect as phone
from rapid_mlx.config import get_config
from rapid_mlx.middleware.auth import verify_api_key
from rapid_mlx.middleware.browser_security import (
    DEFAULT_TRUSTED_HOSTS,
    BrowserSecurityMiddleware,
)
from rapid_mlx.middleware.lan_inference import LANInferenceMiddleware
from rapid_mlx.routes.phone import router


@pytest.mark.asyncio
@pytest.mark.parametrize("lan", [False, True])
async def test_page_and_api_boundary(monkeypatch, lan):
    monkeypatch.setattr(get_config(), "api_key", "server-secret")
    monkeypatch.setenv("RAPID_MLX_LAN_INFERENCE_ONLY", "1" if lan else "0")
    app = FastAPI()
    app.include_router(router)

    @app.get("/v1/models", dependencies=[Depends(verify_api_key)])
    async def models():
        return {"data": []}

    @app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
    async def chat():
        return {"ok": True}

    app.add_middleware(LANInferenceMiddleware, enabled=lan, router=app.router)
    app.add_middleware(
        BrowserSecurityMiddleware,
        hosts=list(DEFAULT_TRUSTED_HOSTS),
        deny_cross_origin=True,
    )
    # ASGITransport populates server from base_url, as Uvicorn does from the socket.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app, client=("192.168.1.9", 123)),
        base_url="http://192.168.1.5:8000",
    ) as client:
        r = await client.get("/connect")
        assert r.status_code == 200
        assert "server-secret" not in r.text
        assert "sessionStorage" in r.text or "localStorage" in r.text
        assert "innerHTML" not in r.text
        assert "apple-mobile-web-app-capable" in r.text
        assert r.headers["cache-control"] == "no-store"
        assert "connect-src 'self'" in r.headers["content-security-policy"]
        assert (await client.post("/connect")).status_code == (403 if lan else 400)
        assert (
            await client.get("/connect", headers={"Origin": "http://evil.test"})
        ).status_code == 403
        assert (
            await client.get(
                "/v1/models",
                headers={
                    "Origin": "http://192.168.1.5:8000",
                    "Authorization": "Bearer server-secret",
                },
            )
        ).status_code == 200
        assert (await client.get("/v1/models")).status_code == 401
        for origin, expected in [
            ("http://192.168.1.5:8000", 200),
            ("http://evil.test", 403),
        ]:
            assert (
                await client.post(
                    "/v1/chat/completions",
                    headers={"Origin": origin, "Authorization": "Bearer server-secret"},
                )
            ).status_code == expected
        assert (
            await client.get(
                "/v1/models",
                headers={
                    "Origin": "http://evil.test",
                    "Authorization": "Bearer server-secret",
                },
            )
        ).status_code == 403
        assert (await client.get("/admin")).status_code == (403 if lan else 400)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host,server,policy",
    [
        ("evil.test", ("192.168.1.5", 8000), None),
        ("192.168.1.6", ("192.168.1.5", 8000), None),
        ("192.168.1.5", ("192.168.1.5", 8000), ["localhost"]),
    ],
)
async def test_socket_host_exception_is_narrow(monkeypatch, host, server, policy):
    monkeypatch.delenv("RAPID_MLX_LAN_INFERENCE_ONLY", raising=False)
    sent = []

    async def downstream(scope, receive, send):
        pytest.fail("Untrusted host reached route")

    async def send(message):
        sent.append(message)

    app = BrowserSecurityMiddleware(
        downstream, hosts=policy or list(DEFAULT_TRUSTED_HOSTS)
    )
    await app(
        {
            "type": "http",
            "method": "GET",
            "path": "/connect",
            "scheme": "http",
            "server": server,
            "headers": [(b"host", f"{host}:8000".encode())],
        },
        None,
        send,
    )
    assert sent[0]["status"] == 400


@pytest.mark.parametrize(
    "value,expected",
    [
        ("10.0.0.5", True),
        ("192.168.1.2", True),
        ("127.1.2.3", False),
        ("169.254.2.3", False),
        ("0.0.0.0", False),
        ("::1", False),
        ("bad", False),
        ("224.0.0.1", False),
        ("255.255.255.255", False),
    ],
)
def test_usable_ip(value, expected):
    assert phone.usable_ip(value) is expected


def test_addresses(monkeypatch):
    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {
            "en0": [SimpleNamespace(family=socket.AF_INET, address="192.168.1.5")],
            "en1": [SimpleNamespace(family=socket.AF_INET, address="10.0.0.5")],
            "lo": [SimpleNamespace(family=socket.AF_INET, address="127.0.0.1")],
            "down": [SimpleNamespace(family=socket.AF_INET, address="10.0.0.6")],
        },
    )
    monkeypatch.setattr(
        psutil,
        "net_if_stats",
        lambda: {
            "en0": SimpleNamespace(isup=True),
            "en1": SimpleNamespace(isup=True),
            "lo": SimpleNamespace(isup=True),
            "down": SimpleNamespace(isup=False),
        },
    )
    probe = MagicMock()
    probe.__enter__.return_value = probe
    probe.getsockname.return_value = ("192.168.1.5", 0)
    monkeypatch.setattr(socket, "socket", lambda *a: probe)
    assert phone.lan_addresses() == ["192.168.1.5", "10.0.0.5"]
    probe.getsockname.return_value = ("10.1.1.1", 0)
    assert phone.lan_addresses() == ["10.0.0.5", "192.168.1.5"]
    probe.connect.side_effect = OSError()
    assert phone.lan_addresses() == ["10.0.0.5", "192.168.1.5"]


def test_process_info(monkeypatch):
    proc = MagicMock()
    monkeypatch.setattr(psutil, "Process", lambda pid: proc)
    proc.net_connections.return_value = [
        SimpleNamespace(
            status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip="0.0.0.0", port=8000)
        ),
        SimpleNamespace(
            status="CLOSE", laddr=SimpleNamespace(ip="127.0.0.1", port=8000)
        ),
    ]
    proc.environ.return_value = {"RAPID_MLX_API_KEY": "env-key"}
    for tokens, key in [
        (["serve", "--api-key=equals"], "equals"),
        (["serve", "--api-key", "space"], "space"),
        (["serve"], "env-key"),
    ]:
        proc.cmdline.return_value = tokens
        assert phone.connection_info(SimpleNamespace(pid=5, port=8000)) == (
            ["0.0.0.0"],
            key,
        )


@pytest.fixture
def setup_connect(monkeypatch):
    server = SimpleNamespace(pid=5, port=8000, model="model", listening=True)
    monkeypatch.setattr(phone, "scan_servers", lambda **kw: [server])
    monkeypatch.setattr(phone, "connection_info", lambda s: (["0.0.0.0"], "secret+&"))
    monkeypatch.setattr(phone, "lan_addresses", lambda: ["192.168.1.5", "10.0.0.5"])
    return argparse.Namespace(port=None, json=False, host_ip=None)


def test_happy(setup_connect, capsys):
    phone.connect_phone(setup_connect)
    out = capsys.readouterr().out
    assert "▀" in out and "▄" in out and "█" in out
    assert "Base URL: http://192.168.1.5:8000/v1" in out
    assert "API key: secret+&" in out and "same Wi-Fi" in out and "--host-ip" in out
    setup_connect.json = True
    phone.connect_phone(setup_connect)
    data = json.loads(capsys.readouterr().out)
    parsed = urlsplit(data["url"])
    assert parsed.path == "/connect" and not parsed.query
    assert parse_qs(parsed.fragment) == {"k": ["secret+&"]}
    assert data["api_key"] == "secret+&"
    setup_connect.host_ip = "10.0.0.5"
    phone.connect_phone(setup_connect)
    assert json.loads(capsys.readouterr().out)["base_url"] == "http://10.0.0.5:8000/v1"


@pytest.mark.parametrize(
    "listeners,key,message",
    [(["127.0.0.1"], "old", "loopback"), (["0.0.0.0"], "", "requires an API key")],
)
def test_restart(setup_connect, monkeypatch, capsys, listeners, key, message):
    monkeypatch.setattr(phone, "connection_info", lambda s: (listeners, key))
    monkeypatch.setattr(phone.secrets, "token_urlsafe", lambda n: "fresh-secret")
    with pytest.raises(SystemExit):
        phone.connect_phone(setup_connect)
    out = capsys.readouterr().out
    assert message in out and "--host 0.0.0.0 --api-key fresh-secret" in out
    assert out.count("fresh-secret") == 1 and "/connect#" not in out


@pytest.mark.parametrize(
    "scenario,message",
    [
        ("none", "No running"),
        ("multiple", "Multiple servers"),
        ("denied", "Cannot inspect"),
        ("invalid", "--host-ip must"),
        ("noip", "No LAN"),
        ("mismatch", "not a server listener"),
    ],
)
def test_failures(setup_connect, monkeypatch, capsys, scenario, message):
    if scenario == "none":
        monkeypatch.setattr(phone, "scan_servers", lambda **kw: [])
    if scenario == "multiple":
        monkeypatch.setattr(
            phone,
            "scan_servers",
            lambda **kw: [SimpleNamespace(port=8000, listening=True)] * 2,
        )
    if scenario == "denied":
        monkeypatch.setattr(
            phone, "connection_info", MagicMock(side_effect=psutil.AccessDenied())
        )
    if scenario == "invalid":
        setup_connect.host_ip = "127.0.0.1"
    if scenario == "noip":
        monkeypatch.setattr(phone, "lan_addresses", lambda: [])
    if scenario == "mismatch":
        monkeypatch.setattr(phone, "connection_info", lambda s: (["10.0.0.5"], "key"))
    setup_connect.json = True
    with pytest.raises(SystemExit):
        phone.connect_phone(setup_connect)
    assert message in json.loads(capsys.readouterr().out)["error"]


def test_parser():
    from rapid_mlx.cli_parser import build_parser

    args = build_parser().parse_args(["connect", "--host-ip", "192.168.1.5", "--json"])
    assert args.host_ip == "192.168.1.5" and args.json


def test_cli_dispatch(setup_connect, monkeypatch, capsys):
    from rapid_mlx.cli import connect_command

    setup_connect.target = None
    setup_connect.base_url = None
    setup_connect.host = None
    connect_command(setup_connect)
    assert "/connect#k=" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_production_router_wiring():
    from rapid_mlx.server import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://localhost"
    ) as client:
        response = await client.get("/connect")
    assert response.status_code == 200
    assert "Rapid · Phone" in response.text


@pytest.mark.asyncio
async def test_bootstrap_public_even_without_configured_key(monkeypatch):
    monkeypatch.setattr(get_config(), "api_key", None)
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(LANInferenceMiddleware, enabled=True, router=app.router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app, client=("10.0.0.3", 123)),
        base_url="http://10.0.0.2:8000",
    ) as client:
        assert (await client.get("/connect")).status_code == 200
        for method in ["HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]:
            assert (
                await client.request(
                    method, "/connect", headers={"Authorization": "Bearer key"}
                )
            ).status_code == 403
        assert (await client.get("/v1/models")).status_code == 401
