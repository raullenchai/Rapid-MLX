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
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .. import __version__
from ._constants import DEFAULT_RELAY_URL

# Pulled out so the routing-shape audit
# (tests/test_no_out_of_band_routing.py::test_no_routing_shaped_rapid_mlx_env_vars)
# sees one clean RAPID_MLX_* literal — embedding it in an f-string error
# message yields ``RAPID_MLX_RELAY_URL `` (with trailing space) which is
# NOT in the allowlist and tripwires the audit.
_RELAY_URL_ENV_VAR = "RAPID_MLX_RELAY_URL"


@dataclass(frozen=True)
class Session:
    subdomain: str
    token: str
    frps_host: str
    frps_port: int
    public_url: str
    expires_at: str


def relay_base_url() -> str:
    """Resolve the control-plane URL.

    The default is ``https://api.rapidmlx.com``. The ``RAPID_MLX_RELAY_URL``
    override exists for docker-compose dev (``http://localhost:8080``) but
    must NEVER silently send a freshly-minted bearer key to an arbitrary
    plaintext URL — that would be a leak vector if an attacker can poison
    the user's environment. So we require either HTTPS, or a loopback
    target (``localhost``/``127.0.0.0/8``) for the override to apply.
    """
    override = os.environ.get(_RELAY_URL_ENV_VAR)
    if override is None:
        return DEFAULT_RELAY_URL.rstrip("/")
    parsed = urllib.parse.urlparse(override)
    if not parsed.hostname:
        # ``RAPID_MLX_RELAY_URL=https://`` parses but has hostname=None;
        # otherwise we'd build ``https:///share/session`` and fail with
        # a confusing low-level error. (DeepSeek round-4 NIT #4.)
        raise RuntimeError(
            f"{_RELAY_URL_ENV_VAR}={override!r} has no hostname; "
            f"expected e.g. https://api.example.com or http://localhost:8080."
        )
    host = parsed.hostname.lower()
    is_loopback = host == "localhost" or host.startswith("127.")
    if parsed.scheme == "https":
        return override.rstrip("/")
    if parsed.scheme == "http" and is_loopback:
        # Loud warning: the operator chose plaintext; it's their call but
        # we don't want this to feel safe.
        print(
            f"rapid-mlx share: {_RELAY_URL_ENV_VAR} points at a plaintext "
            f"loopback target — fine for local dev, NEVER for production.",
            file=sys.stderr,
        )
        return override.rstrip("/")
    raise RuntimeError(
        f"{_RELAY_URL_ENV_VAR}={override!r} is unsafe: must be HTTPS, or "
        f"http://localhost / http://127.0.0.0/8 for local dev."
    )


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
            f"If this is a dev environment, point {_RELAY_URL_ENV_VAR} at "
            f"your local control plane."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"could not reach rapidmlx.com relay at {relay_base_url()}: "
            f"{exc.reason}. Check your network or override with "
            f"{_RELAY_URL_ENV_VAR}=http://localhost:8080 for local dev."
        ) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        # The 2xx-body-is-malformed-JSON path — common if a reverse proxy
        # serves an HTML error page in front of the relay. Without this
        # branch ValueError leaks as a raw traceback. (DeepSeek round-4
        # BLOCKER #1.)
        raise RuntimeError(
            f"relay returned a non-JSON body (rapidmlx.com edge "
            f"intermediary may be down): {exc}. Try again, or check "
            f"{_RELAY_URL_ENV_VAR}."
        ) from exc

    # The relay is owned by us but we want a clear user-readable error if
    # a future schema change drops a field — better than a bare KeyError
    # traceback that looks like a client bug.
    try:
        return Session(
            subdomain=payload["subdomain"],
            token=payload["token"],
            frps_host=payload["frps_host"],
            frps_port=int(payload["frps_port"]),
            public_url=payload["public_url"],
            expires_at=payload["expires_at"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"relay returned an unexpected response shape "
            f"(missing/invalid field: {exc}). Upgrade rapid-mlx or check "
            f"{_RELAY_URL_ENV_VAR}."
        ) from exc
