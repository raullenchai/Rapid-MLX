# SPDX-License-Identifier: Apache-2.0
"""Local HTTP server boundaries against drive-by browser access and rebinding."""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlsplit

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .lan_inference import is_loopback_peer

_configured_hosts: list[str] | None = None


DEFAULT_TRUSTED_HOSTS = ("localhost", "127.0.0.1", "[::1]", "*.localhost")
# Closed set of URL hostname spellings; do not expand via DNS or IP ranges.
_LOOPBACK_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def resolve_trusted_hosts(cli_hosts: list[str] | None = None) -> list[str]:
    """Explicit CLI/env values replace the local-only default; empty fails closed."""
    entries = (
        cli_hosts
        if cli_hosts is not None
        else [os.environ.get("RAPID_MLX_TRUSTED_HOSTS", "")]
    )
    hosts = [
        part.strip().lower()
        for entry in entries
        for part in entry.split(",")
        if part.strip()
    ]
    return hosts or list(DEFAULT_TRUSTED_HOSTS)


def _authority(value: str, scheme: str) -> tuple[str, int] | None:
    """Parse an HTTP authority without confusing IPv6 colons with the port."""
    try:
        if (
            not value
            or any(c in value for c in "/\\?#@,")
            or any(ord(c) <= 32 or ord(c) == 127 for c in value)
        ):
            return None
        if value.startswith("["):
            closing = value.find("]")
            if closing < 0:
                return None
            suffix = value[closing + 1 :]
            if suffix and (
                not suffix.startswith(":")
                or not suffix[1:].isascii()
                or not suffix[1:].isdigit()
            ):
                return None
        elif ":" in value:
            if (
                value.count(":") != 1
                or not value.rsplit(":", 1)[1].isascii()
                or not value.rsplit(":", 1)[1].isdigit()
            ):
                return None
        parsed = urlsplit(f"{scheme}://{value}")
        if parsed.hostname is None:
            return None
        return parsed.hostname.lower(), (
            parsed.port
            if parsed.port is not None
            else (443 if scheme == "https" else 80)
        )
    except ValueError:
        return None


class BrowserSecurityMiddleware:
    """Reject untrusted Hosts and, with CORS disabled, foreign Origins before dispatch.

    Host policy applies even to probes and preflights; authenticated LAN mode
    exempts non-loopback peers from hostname matching. Origin checking additionally
    prevents simple cross-origin POSTs from mutating state without a preflight.
    Native clients omit Origin; same-origin browser clients remain supported.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        hosts: list[str] | None = None,
        deny_cross_origin: bool = False,
    ) -> None:
        self.app = app
        self.hosts = hosts
        self.deny_cross_origin = deny_cross_origin
        self.lan_inference_only = os.environ.get("RAPID_MLX_LAN_INFERENCE_ONLY") == "1"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers", [])
        host_values = [v.decode("latin-1") for k, v in headers if k.lower() == b"host"]
        raw_scheme = scope.get("scheme") or "http"
        scheme = {"ws": "http", "wss": "https"}.get(raw_scheme, raw_scheme)
        authority = (
            _authority(host_values[0], scheme) if len(host_values) == 1 else None
        )
        allowed = authority is not None
        # Remote LAN requests are guarded by mandatory auth and the inference
        # allowlist. Local peers still need Host checks against DNS rebinding,
        # including when the operator supplies an explicit host policy.
        remote_lan = self.lan_inference_only and not is_loopback_peer(
            scope.get("client")
        )
        if authority is not None and self.hosts is not None and not remote_lan:
            hostname = authority[0]
            allowed = any(
                pattern == "*"
                or hostname == pattern.strip("[]")
                or (pattern.startswith("*.") and hostname.endswith(pattern[1:]))
                for pattern in self.hosts
            )
        # Phone routes may use a numeric LAN Host matching the accepted socket.
        # No DNS resolution, wildcard trust, or expansion to management routes.
        if not allowed and authority and scope.get("server"):
            try:
                address = ipaddress.IPv4Address(authority[0])
                allowed = (
                    not address.is_loopback
                    and authority == tuple(scope["server"])
                    and (scope.get("method"), scope.get("path"))
                    in {
                        ("GET", "/connect"),
                        ("GET", "/v1/models"),
                        ("POST", "/v1/chat/completions"),
                    }
                    and (
                        self.hosts is None or self.hosts == list(DEFAULT_TRUSTED_HOSTS)
                    )
                )
            except ipaddress.AddressValueError:
                pass
        status = 400 if not allowed else None
        if allowed and self.deny_cross_origin:
            origins = [
                v.decode("latin-1") for k, v in headers if k.lower() == b"origin"
            ]
            if origins:
                origin = origins[0]
                prefix = f"{scheme}://"
                origin_authority = (
                    _authority(origin[len(prefix) :], scheme)
                    if origin.startswith(prefix)
                    else None
                )
                if (
                    len(origins) != 1
                    or not origin.startswith(prefix)
                    or (
                        origin_authority != authority
                        and not (
                            origin_authority is not None
                            and authority is not None
                            and authority[0] in _LOOPBACK_HOSTNAMES
                            and origin_authority[0] in _LOOPBACK_HOSTNAMES
                            and origin_authority[1] == authority[1]
                        )
                    )
                ):
                    status = 403
        if status is not None:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await PlainTextResponse(
                    "Invalid host header"
                    if status == 400
                    else "Cross-origin requests disabled",
                    status_code=status,
                )(scope, receive, send)
            return
        await self.app(scope, receive, send)


def install_browser_security(
    app, *, origins: list[str], hosts: list[str] | None = None
) -> None:
    app.add_middleware(
        BrowserSecurityMiddleware,
        hosts=(
            resolve_trusted_hosts(hosts)
            if hosts is not None
            else (_configured_hosts or resolve_trusted_hosts())
        ),
        deny_cross_origin=not origins,
    )
