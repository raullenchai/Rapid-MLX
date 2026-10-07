# SPDX-License-Identifier: Apache-2.0
"""Agent setup when the Rapid-MLX server is not running yet.

Configuring an agent before ``rapid-mlx serve`` is up is the normal order of
operations for a new user, so a refused connection is not a setup failure:
the config can be written for the intended model and port, and the user is
told the one command that starts a matching server.

There is deliberately no "wait for it to come up" poll. ``serve`` binds its
port only after the model has loaded, so a server that is still starting and
one that was never started both refuse the connection; polling would only add
latency to the common never-started case.
"""

from __future__ import annotations

import shlex
import urllib.error
from typing import Any

# Hosts a plain ``rapid-mlx serve`` on this machine answers (it binds
# 127.0.0.1 by default), mapped to the ``--host`` the command needs, if any.
_LOCAL_HOSTS = {"localhost": None, "127.0.0.1": None, "::1": "::1"}


class ServerNotRunningError(RuntimeError):
    """Nothing is listening at the configured server address."""


def is_connection_refused(exc: BaseException) -> bool:
    """True when *exc* means no process is listening at the address.

    Timeouts, HTTP errors and malformed responses are not included: those
    mean something answered (or hung), which is a real setup failure.
    """
    if isinstance(exc, ConnectionRefusedError):
        return True
    return isinstance(exc, urllib.error.URLError) and isinstance(
        exc.reason, ConnectionRefusedError
    )


def is_local_base_url(base_url: str) -> bool:
    """True when *base_url* names a server this machine would start itself.

    A refused remote endpoint is not "not started yet": there is no single
    local command that brings it up, so callers keep treating it as a failure.
    """
    from rapid_mlx.connect import _parse_base_url

    try:
        host, _port = _parse_base_url(base_url)
    except ValueError:
        return False
    return host in _LOCAL_HOSTS


def start_server_command(profile: Any, base_url: str, model_id: str) -> str:
    """The one ``rapid-mlx serve`` command that matches the written config."""
    from rapid_mlx.connect import _parse_base_url

    if model_id and model_id != "default":
        model = model_id
    elif getattr(profile, "recommended_models", None):
        model = profile.recommended_models[0]
    else:
        model = "<model>"
    parts = ["rapid-mlx", "serve", model if model == "<model>" else shlex.quote(model)]
    try:
        host, port = _parse_base_url(base_url)
    except ValueError:
        return " ".join(parts)
    # ``serve`` picks the first free port in 8000-8009 by default; pin the one
    # the agent was configured for so the two cannot drift apart.
    parts += ["--port", str(port)]
    bind_host = _LOCAL_HOSTS.get(host)
    if bind_host is not None:
        parts += ["--host", bind_host]
    return " ".join(parts)


def not_running_lines(profile: Any, base_url: str, model_id: str) -> list[str]:
    """Note printed after a config was saved for a server that is not up."""
    root = base_url.rstrip("/").removesuffix("/v1")
    return [
        f"  Server not running yet at {root}; the config is saved and will "
        "work once it starts.",
        f"  Start it with:  {start_server_command(profile, base_url, model_id)}",
    ]
