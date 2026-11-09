# SPDX-License-Identifier: Apache-2.0
"""Desktop LAN boundary, without starting an engine or downloading weights."""

import logging

import httpx
import pytest
from fastapi import FastAPI

from rapid_mlx.config import get_config
from rapid_mlx.middleware.lan_inference import LANInferenceMiddleware


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(get_config(), "api_key", "secret")
    app = FastAPI()

    @app.get("/v1/models/residency")
    async def residency():
        return {"local": True}

    @app.get("/v1/models/{model_id:path}")
    async def model(model_id: str):
        return {"id": model_id}

    @app.api_route("/{path:path}", methods=["GET", "POST", "DELETE"])
    async def echo(path: str):
        return {"path": path}

    return app


async def request(app, method, path, *, peer="192.168.1.42", key="secret", **kwargs):
    extra_headers = kwargs.pop("extra_headers", {})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=(peer, 1234)),
        base_url="http://server",
    ) as client:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        headers.update(extra_headers)
        return await client.request(method, path, headers=headers, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/v1/local-search/search",
        "/v1/cua/apps",
        "/v1/cache/clear",
        "/v1/status",
        "/v1/models/residency",
        "/docs",
        "/openapi.json",
        "/v1/unknown",
        "/v1/chat/completions/extra",
    ],
)
@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
async def test_local_only_forbidden_even_with_valid_key(app, path, method):
    app.add_middleware(LANInferenceMiddleware, enabled=True, router=app.router)
    response = await request(app, method, path)
    assert response.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/v1/models"),
        ("GET", "/v1/models/org/model"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/embeddings"),
        ("POST", "/v1/responses"),
        ("POST", "/v1/audio/speech"),
        ("POST", "/v1/rerank"),
        ("POST", "/v1/videos"),
        ("POST", "/v1/videos/extend"),
        ("GET", "/v1/videos"),
        ("GET", "/health"),
        ("GET", "/healthz"),
        ("GET", "/livez"),
    ],
)
async def test_inference_requires_key(app, method, path):
    app.add_middleware(LANInferenceMiddleware, enabled=True, router=app.router)
    assert (await request(app, method, path)).status_code == 200
    for key in (None, "wrong"):
        assert (await request(app, method, path, key=key)).status_code == 401


@pytest.mark.asyncio
async def test_missing_configured_key_fails_closed(app, monkeypatch):
    monkeypatch.setattr(get_config(), "api_key", None)
    app.add_middleware(LANInferenceMiddleware, enabled=True, router=app.router)
    assert (await request(app, "GET", "/v1/models")).status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("peer", ["127.0.0.1", "127.1.2.3", "::1", "::ffff:127.0.0.1"])
async def test_loopback_unaffected(app, peer):
    app.add_middleware(LANInferenceMiddleware, enabled=True, router=app.router)
    assert (
        await request(app, "POST", "/v1/cache/clear", peer=peer, key=None)
    ).status_code == 200


@pytest.mark.asyncio
async def test_flag_off_preserves_existing_behavior(app, monkeypatch):
    monkeypatch.delenv("RAPID_MLX_LAN_INFERENCE_ONLY", raising=False)
    app.add_middleware(LANInferenceMiddleware, router=app.router)
    assert (await request(app, "POST", "/v1/cache/clear", key=None)).status_code == 200


@pytest.mark.asyncio
async def test_environment_opt_in_and_untrusted_forwarded_header(app, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_LAN_INFERENCE_ONLY", "1")
    app.add_middleware(LANInferenceMiddleware, router=app.router)
    assert (
        await request(
            app, "GET", "/v1/cua/apps", extra_headers={"X-Forwarded-For": "127.0.0.1"}
        )
    ).status_code == 403
    assert (await request(app, "DELETE", "/v1/models/model")).status_code == 403


@pytest.mark.asyncio
async def test_boundary_wraps_health_fastpath(app):
    from rapid_mlx.middleware.probe_fastpath import install_probe_fastpath_middleware

    install_probe_fastpath_middleware(app)
    app.add_middleware(LANInferenceMiddleware, enabled=True, router=app.router)
    assert (await request(app, "GET", "/livez", key=None)).status_code == 401
    assert (await request(app, "GET", "/livez")).status_code == 200


@pytest.mark.parametrize("lan", [False, True])
def test_uvicorn_disables_proxy_trust_only_in_lan_mode(monkeypatch, lan):
    from rapid_mlx import _uvicorn

    class Server:
        started = True

        def __init__(self, config, **kwargs):
            self.config = config

        def run(self):
            assert self.config.proxy_headers is (not lan)

    monkeypatch.setenv("RAPID_MLX_LAN_INFERENCE_ONLY", "1" if lan else "0")
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")
    monkeypatch.setattr(_uvicorn, "AcceptingConnectionsServer", Server)
    _uvicorn._run_server(FastAPI())


@pytest.mark.asyncio
async def test_model_management_is_not_model_lookup(app):
    app.add_middleware(LANInferenceMiddleware, enabled=True, router=app.router)
    assert (await request(app, "POST", "/v1/models/activate")).status_code == 403
    assert (await request(app, "DELETE", "/v1/models/model")).status_code == 403


@pytest.mark.asyncio
async def test_unknown_peer_fails_closed(app):
    app.add_middleware(LANInferenceMiddleware, enabled=True, router=app.router)
    assert (
        await request(app, "GET", "/v1/cua/apps", peer="unknown")
    ).status_code == 403


@pytest.mark.asyncio
async def test_websocket_cannot_bypass_lan_boundary(app):
    guarded = LANInferenceMiddleware(app, enabled=True)
    messages = []

    async def send(message):
        messages.append(message)

    await guarded({"type": "websocket", "client": ("192.168.1.42", 1234)}, None, send)
    assert messages == [{"type": "websocket.close", "code": 1008}]


@pytest.mark.asyncio
async def test_future_specific_model_management_route_stays_local(monkeypatch):
    monkeypatch.setattr(get_config(), "api_key", "secret")
    app = FastAPI()

    @app.get("/v1/models/new-admin")
    async def admin():
        return {"local": True}

    @app.get("/v1/models/{model_id:path}")
    async def model(model_id: str):
        return {"id": model_id}

    app.add_middleware(LANInferenceMiddleware, enabled=True, router=app.router)
    assert (await request(app, "GET", "/v1/models/new-admin")).status_code == 403
    assert (await request(app, "GET", "/v1/models/org/model")).status_code == 200


@pytest.mark.asyncio
async def test_lan_startup_never_installs_unauthenticated_probe(monkeypatch):
    import uvicorn

    from rapid_mlx import _startup_probe
    from rapid_mlx._uvicorn import AcceptingConnectionsServer

    monkeypatch.setenv("RAPID_MLX_LAN_INFERENCE_ONLY", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("LAN startup must not bypass the authenticated ASGI boundary")

    monkeypatch.setattr(_startup_probe, "StartupProbe", forbidden)
    started = []
    banners = []

    def banner(listeners):
        assert started == [True]
        assert all(listener.sockets for listener in listeners)
        banners.append(True)

    monkeypatch.setattr("rapid_mlx._uvicorn._print_lan_banner", banner)

    async def app(scope, receive, send):
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    def accepting():
        started.append(True)
        instance.should_exit = True

    instance = AcceptingConnectionsServer(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"),
        on_server_accepting=accepting,
    )
    await instance.serve()
    assert started == [True]
    assert banners == [True]


@pytest.fixture
def lan_stack(app, monkeypatch):
    from rapid_mlx.middleware import browser_security
    from rapid_mlx.middleware.probe_fastpath import install_probe_fastpath_middleware

    monkeypatch.setenv("RAPID_MLX_LAN_INFERENCE_ONLY", "1")
    monkeypatch.delenv("RAPID_MLX_TRUSTED_HOSTS", raising=False)
    monkeypatch.setattr(browser_security, "_configured_hosts", None)
    # Production order: probe, LAN boundary, then browser/CORS policy.
    install_probe_fastpath_middleware(app)
    app.add_middleware(LANInferenceMiddleware, router=app.router)
    browser_security.install_browser_security(app, origins=[])
    return app


@pytest.mark.asyncio
async def test_real_stack_accepts_authenticated_lan_host(lan_stack):
    headers = {"Host": "192.168.1.147:8467"}
    assert (
        await request(lan_stack, "POST", "/v1/chat/completions", extra_headers=headers)
    ).status_code == 200
    assert (
        await request(
            lan_stack, "POST", "/v1/chat/completions", key=None, extra_headers=headers
        )
    ).status_code == 401
    for method, path in [
        ("GET", "/v1/status"),
        ("GET", "/metrics"),
        ("DELETE", "/v1/cache"),
    ]:
        assert (
            await request(lan_stack, method, path, extra_headers=headers)
        ).status_code == 403
        assert (
            await request(
                lan_stack,
                method,
                path,
                extra_headers={
                    **headers,
                    "X-Forwarded-For": "127.0.0.1",
                    "X-Forwarded-Host": "localhost",
                },
            )
        ).status_code == 403
    assert (
        await request(lan_stack, "GET", "/health", extra_headers=headers)
    ).status_code == 200
    assert (
        await request(lan_stack, "GET", "/livez", key=None, extra_headers=headers)
    ).status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("peer", ["127.0.0.1", "::1", "::ffff:127.0.0.1"])
async def test_real_stack_keeps_loopback_host_protection(lan_stack, peer):
    assert (
        await request(
            lan_stack,
            "POST",
            "/v1/chat/completions",
            peer=peer,
            extra_headers={"Host": "evil.example"},
        )
    ).status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["cli", "env"])
async def test_lan_loopback_host_overrides(app, monkeypatch, source):
    from rapid_mlx.middleware import browser_security

    monkeypatch.setenv("RAPID_MLX_LAN_INFERENCE_ONLY", "1")
    monkeypatch.setattr(browser_security, "_configured_hosts", None)
    monkeypatch.setenv("RAPID_MLX_TRUSTED_HOSTS", "custom.example")
    app.add_middleware(LANInferenceMiddleware, router=app.router)
    browser_security.install_browser_security(
        app, origins=[], hosts=["custom.example"] if source == "cli" else None
    )
    assert (
        await request(
            app,
            "GET",
            "/health",
            peer="127.0.0.1",
            extra_headers={"Host": "custom.example"},
        )
    ).status_code == 200
    assert (
        await request(
            app,
            "GET",
            "/health",
            peer="127.0.0.1",
            extra_headers={"Host": "evil.example"},
        )
    ).status_code == 400


@pytest.mark.asyncio
async def test_lan_host_exception_preserves_origin_checks(lan_stack):
    response = await request(
        lan_stack,
        "POST",
        "/v1/chat/completions",
        extra_headers={"Host": "192.168.1.147:8467", "Origin": "http://evil.example"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_non_lan_mode_keeps_remote_host_checks(app, monkeypatch):
    from rapid_mlx.middleware import browser_security

    monkeypatch.delenv("RAPID_MLX_LAN_INFERENCE_ONLY", raising=False)
    monkeypatch.setattr(browser_security, "_configured_hosts", None)
    monkeypatch.delenv("RAPID_MLX_TRUSTED_HOSTS", raising=False)
    app.add_middleware(LANInferenceMiddleware, router=app.router)
    browser_security.install_browser_security(app, origins=[])
    assert (
        await request(
            app,
            "POST",
            "/v1/chat/completions",
            extra_headers={"Host": "192.168.1.147:8467"},
        )
    ).status_code == 400


@pytest.mark.parametrize(
    "lan,host", [(True, "0.0.0.0"), (False, "0.0.0.0"), (True, "127.0.0.1")]
)
def test_lan_banner_uses_interface_ipv4_and_actual_port(monkeypatch, capsys, lan, host):
    import socket
    from types import SimpleNamespace

    import psutil

    from rapid_mlx._uvicorn import _print_lan_banner

    monkeypatch.setenv("RAPID_MLX_LAN_INFERENCE_ONLY", "1" if lan else "0")
    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {
            "en0": [
                SimpleNamespace(family=socket.AF_INET, address=address)
                for address in [
                    "127.0.0.1",
                    "127.2.3.4",
                    "0.0.0.0",
                    "192.168.1.147",
                    "10.0.0.2",
                    "192.168.1.147",
                ]
            ]
            + [SimpleNamespace(family=socket.AF_INET6, address="::1")]
        },
    )
    with socket.socket() as listener:
        listener.bind((host, 0))
        port = listener.getsockname()[1]
        _print_lan_banner([SimpleNamespace(sockets=[listener])])
    output = capsys.readouterr().out
    if lan and host == "0.0.0.0":
        assert output == (
            "  LAN inference: API key required (Authorization: Bearer <key>)\n"
            f"  LAN: http://10.0.0.2:{port}\n"
            f"  LAN: http://192.168.1.147:{port}\n"
        )
    else:
        assert output == ""


def test_lan_banner_reports_missing_interface_address(monkeypatch, capsys):
    import socket
    from types import SimpleNamespace

    import psutil

    from rapid_mlx._uvicorn import _print_lan_banner

    monkeypatch.setenv("RAPID_MLX_LAN_INFERENCE_ONLY", "1")
    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {"lo0": [SimpleNamespace(family=socket.AF_INET, address="127.0.0.1")]},
    )
    with socket.socket() as listener:
        listener.bind(("0.0.0.0", 0))
        _print_lan_banner([SimpleNamespace(sockets=[listener])])
    assert capsys.readouterr().out == (
        "  LAN inference: API key required (Authorization: Bearer <key>)\n"
        "  LAN: no non-loopback IPv4 address currently available\n"
    )


@pytest.mark.asyncio
async def test_lan_banner_failure_never_stops_server(monkeypatch, caplog):
    import uvicorn

    from rapid_mlx._uvicorn import AcceptingConnectionsServer

    def banner(listeners):
        raise RuntimeError("interface enumeration failed")

    monkeypatch.setattr("rapid_mlx._uvicorn._print_lan_banner", banner)

    async def app(scope, receive, send):
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    def accepting():
        instance.should_exit = True

    instance = AcceptingConnectionsServer(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"),
        on_server_accepting=accepting,
    )
    with caplog.at_level(logging.WARNING, logger="rapid_mlx._uvicorn"):
        await instance.serve()
    assert "Could not enumerate LAN URLs" in caplog.text
