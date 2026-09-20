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
