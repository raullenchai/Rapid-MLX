# SPDX-License-Identifier: Apache-2.0
"""HTTP client for the rapidmlx.com control-plane session endpoint.

Session flow:

    POST {relay_url}/share/session
      Body: {"client_version": "0.6.71", "model": "qwen3.5-4b"}
      Resp: {"subdomain": "abc123", "token": "...",
             "frps_host": "tunnel.rapidmlx.com", "frps_port": 7000,
             "public_url": "https://abc123.rapidmlx.com",
             "expires_at": "2026-06-02T12:34:56Z"}

The control plane owns subdomain allocation + token signing; the client
just consumes the response and feeds it into the frpc config.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from .. import __version__
from ._constants import DEFAULT_RELAY_URL


@dataclass(frozen=True)
class Session:
    subdomain: str
    token: str
    frps_host: str
    frps_port: int
    public_url: str
    expires_at: str


def relay_base_url() -> str:
    return os.environ.get("RAPID_MLX_RELAY_URL", DEFAULT_RELAY_URL).rstrip("/")


def request(model: str, *, timeout: float = 10.0) -> Session:
    """Ask the control plane for a fresh share session.

    Raises ``RuntimeError`` with a user-readable message if the relay is
    unreachable or rejects the request — share is useless without a
    working control plane, so we fail loudly.
    """
    body = json.dumps({"client_version": __version__, "model": model}).encode()
    req = urllib.request.Request(
        f"{relay_base_url()}/share/session",
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"rapid-mlx/{__version__}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"relay rejected share request (HTTP {exc.code}): {exc.reason}. "
            f"If this is a dev environment, point RAPID_MLX_RELAY_URL at "
            f"your local control plane."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"could not reach rapidmlx.com relay at {relay_base_url()}: "
            f"{exc.reason}. Check your network or override with "
            f"RAPID_MLX_RELAY_URL=http://localhost:8080 for local dev."
        ) from exc

    return Session(
        subdomain=payload["subdomain"],
        token=payload["token"],
        frps_host=payload["frps_host"],
        frps_port=int(payload["frps_port"]),
        public_url=payload["public_url"],
        expires_at=payload["expires_at"],
    )
