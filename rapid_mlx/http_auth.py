# SPDX-License-Identifier: Apache-2.0
"""Header helpers for Rapid-owned HTTP clients of a local Rapid server."""

from __future__ import annotations

import os


def rapid_mlx_auth_headers() -> dict[str, str]:
    """Return the bearer header configured for the local Rapid-MLX server.

    Keeping the key in ``RAPID_MLX_API_KEY`` avoids exposing it in argv or
    command output. An unset or empty value means the server is unsecured.
    """
    api_key = os.environ.get("RAPID_MLX_API_KEY")
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def rapid_mlx_client_headers(
    label: str, extra: dict[str, str] | None = None
) -> dict[str, str]:
    """Auth header plus ``X-Rapid-Client: <label>`` for a Rapid-owned client.

    Every HTTP client we ship that talks to a Rapid server goes through
    here, so server-side caller attribution does not depend on whichever
    HTTP library set the ``User-Agent`` this release. ``label`` must be one
    of ``rapid_mlx.client_header.RAPID_CLIENT_LABELS``.
    """
    from .client_header import rapid_client_headers

    headers = rapid_mlx_auth_headers()
    headers.update(extra or {})
    return rapid_client_headers(label, headers)


def model_probe_request(base_url: str, api_key: str | None = None):
    """Build a model request with explicit, env, then config credentials.

    Credentials are sent only over loopback HTTP or HTTPS and are never
    forwarded on redirects. Callers must not log this request or its headers.
    """
    import ipaddress
    import urllib.parse
    import urllib.request

    from rapid_mlx.config import get_config

    url = base_url.rstrip("/") + "/models"
    endpoint = urllib.parse.urlsplit(url)
    host = endpoint.hostname or ""
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    request = urllib.request.Request(url)
    key = api_key or os.environ.get("RAPID_MLX_API_KEY") or get_config().api_key
    if key and (endpoint.scheme == "https" or (endpoint.scheme == "http" and loopback)):
        request.add_unredirected_header("Authorization", f"Bearer {key}")
    return request


def probe_models(base_url: str) -> list[dict]:
    """Discover models, preserving authentication failures for CLI guidance.

    Environment credentials take precedence over the process server config.
    Only loopback HTTP or HTTPS endpoints may receive credentials, and a
    redirect never receives the bearer header. Other probe failures return [].
    """
    import json
    import urllib.error
    import urllib.request

    try:
        request = model_probe_request(base_url)
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
        entries = payload.get("data") if isinstance(payload, dict) else None
        return (
            [entry for entry in entries if isinstance(entry, dict)]
            if isinstance(entries, list)
            else []
        )
    except urllib.error.HTTPError as exc:
        exc.close()
        if exc.code in (401, 403):
            raise PermissionError("Server requires a valid API key") from None
        return []
    except (urllib.error.URLError, OSError, ValueError):
        return []
