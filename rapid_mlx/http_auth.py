# SPDX-License-Identifier: Apache-2.0
"""Authentication helpers for local OpenAI-compatible HTTP clients."""

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
