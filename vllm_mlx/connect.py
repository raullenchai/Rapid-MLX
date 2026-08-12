# SPDX-License-Identifier: Apache-2.0
"""Single source of truth (SSOT) for "Ready:" / "Connect:" server output.

The "server is up, now point your tools at it" experience is fragmented:
the serve lifespan prints a ``Ready:`` line, ``agents --setup`` writes a
tool-specific config, and the desktop app separately assembles its own
endpoint URLs. Any change to one is invisible to the others, so a user who
copies the banner URL into a config gets subtly different paths than the
one the desktop shows.

This module owns the *shape* of every endpoint and its human/machine
rendering. Both the serve banner and ``rapid-mlx connect`` render from the
same ``ServerEndpoints`` value, so they can never drift:

* ``ServerEndpoints`` — an immutable description of the ready base URL, the
  OpenAI and Anthropic endpoint paths, and the served model name.
* ``render_banner`` — the human-facing block (used by the serve lifespan
  and by ``rapid-mlx connect``).
* ``to_dict`` — the stable machine-readable JSON shape (used by
  ``rapid-mlx connect --json`` for the desktop and other tooling).

Constraints encoded here:

* The ready URL is the *base* (``http://host:port``) — it has no ``/v1``.
* OpenAI-compatible endpoints append ``/v1`` to the base.
* Anthropic-compatible endpoints ARE the base (no ``/v1`` — the Anthropic
  SDK joins ``/v1/messages`` itself; see ``launch/claude_code.py``).
* Socket-activation (``--listen-fd``) has no known address, so the SSOT
  renders an fd-shaped banner instead of guessing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ServerEndpoints:
    """Immutable description of a running server's connection points."""

    host: str
    port: int
    model: str | None = None
    # True when the server was started via socket activation and the bind
    # address is unknown (owned by the supervisor). When set, ``ready_url``
    # and friends are meaningless and callers must render the fd form.
    listen_fd: int | None = None

    @property
    def base_url(self) -> str:
        """Base ``http://host:port`` — the anchor every endpoint hangs off."""
        if self.listen_fd is not None:
            raise RuntimeError("socket-activation endpoint has no known base URL")
        return f"http://{self.host}:{self.port}"

    @property
    def openai_url(self) -> str:
        """OpenAI-compatible base (``/v1`` is appended by most SDKs)."""
        if self.listen_fd is not None:
            raise RuntimeError("socket-activation endpoint has no known base URL")
        return f"{self.base_url}/v1"

    @property
    def anthropic_url(self) -> str:
        """Anthropic-compatible base (no trailing ``/v1``)."""
        if self.listen_fd is not None:
            raise RuntimeError("socket-activation endpoint has no known base URL")
        return self.base_url

    def to_dict(self) -> dict[str, Any]:
        """Stable machine-readable shape for ``connect --json`` / Desktop."""
        if self.listen_fd is not None:
            return {
                "ready": f"inherited fd {self.listen_fd}",
                "openai": None,
                "anthropic": None,
                "model": self.model,
                "listen_fd": self.listen_fd,
            }
        return {
            "ready": self.base_url,
            "openai": self.openai_url,
            "anthropic": self.anthropic_url,
            "model": self.model,
        }

    def to_json(self) -> str:
        """JSON-encode :meth:`to_dict` for machine consumers."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# The ``--setup`` verbs printed in the human banner's "Connect:" section.
# These are the exact commands a user runs to point each tool at the
# server. ``openai-python`` writes no config file, so its entry is the
# ``rapid-mlx connect openai-python`` snippet-printing command.
_CONNECT_ROWS: list[tuple[str, str]] = [
    ("Claude Code", "rapid-mlx agents claude-code --setup"),
    ("Continue", "rapid-mlx agents continue --setup"),
    ("Python", "rapid-mlx connect openai-python"),
]


def render_banner(ep: ServerEndpoints, *, include_connect: bool = True) -> str:
    """Render the human "Ready:" / "OpenAI:" / "Connect:" block.

    Used by the serve lifespan (once warmup completes) and by
    ``rapid-mlx connect``. Rendered centrally so the served banner and the
    standalone ``connect`` output can never disagree about an endpoint.
    """
    lines: list[str] = []

    if ep.listen_fd is not None:
        lines.append(f"  Ready: inherited fd {ep.listen_fd}")
    else:
        lines.append(f"  Ready: {ep.base_url}")
        lines.append("")
        lines.append(f"  OpenAI:    {ep.openai_url}")
        lines.append(f"  Anthropic: {ep.anthropic_url}")
        lines.append(f"  Model:     {ep.model or '(not yet loaded)'}")

    if include_connect and ep.listen_fd is None:
        lines.append("")
        lines.append("  Connect:")
        width = max(len(a) for a, _ in _CONNECT_ROWS)
        for app, cmd in _CONNECT_ROWS:
            lines.append(f"    {app:<{width}}  {cmd}")

    return "\n".join(lines) + "\n"


def endpoints_from_bind(
    host: str | None,
    port: int | None,
    *,
    model: str | None = None,
    listen_fd: int | None = None,
) -> ServerEndpoints:
    """Build a :class:`ServerEndpoints` from bind source-of-truth fields.

    Mirrors the serve CLI's host/port-or-fd stash on ``ServerConfig``.
    ``host``/``port`` win when both are set; otherwise a non-None
    ``listen_fd`` selects the socket-activation shape.
    """
    if listen_fd is not None and (host is None or port is None):
        return ServerEndpoints(host="", port=0, model=model, listen_fd=listen_fd)
    return ServerEndpoints(host or "localhost", port or 8000, model=model)


def resolve_endpoints(
    *,
    host: str | None = None,
    port: int | None = None,
    model: str | None = None,
) -> ServerEndpoints:
    """Resolve connection info for ``rapid-mlx connect``.

    Preference order:

    1. Explicit ``--host`` / ``--port`` / ``--model`` flags.
    2. A populated ``ServerConfig`` (in-process serve) bind + model fields.
    3. Sensible defaults (``localhost:8000``, best-effort model probe of a
       running server).

    Never raises; a defaulted value is always returned so ``connect`` can
    print a useful banner even when no server is running yet.
    """
    from vllm_mlx.config import get_config

    cfg = get_config()

    out_host = host or cfg.bind_host or "localhost"
    if port is not None:
        out_port = port
    elif cfg.bind_port is not None:
        out_port = cfg.bind_port
    else:
        out_port = 8000

    out_model = model or cfg.model_alias or cfg.model_name

    # Try live detection only when nothing authoritative was supplied.
    if (out_model is None or out_host == "localhost" and out_port == 8000) and (
        cfg.bind_host is None and cfg.bind_port is None
    ):
        detected_model = _probe_running_model(out_host, out_port)
        if detected_model:
            out_model = detected_model

    return ServerEndpoints(out_host, out_port, model=out_model)


def _probe_running_model(host: str, port: int) -> str | None:
    """Best-effort: ask a running server which model it serves. Never raises."""
    import urllib.error
    import urllib.request

    base = f"http://{host}:{port}"
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=2.0) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    models = payload.get("data", [])
    if not isinstance(models, list) or not models:
        return None
    for m in models:
        mid = m.get("id") if isinstance(m, dict) else None
        if isinstance(mid, str) and "/" not in mid and mid != "default":
            return mid
    mid = models[0].get("id") if isinstance(models[0], dict) else None
    return mid if isinstance(mid, str) and mid else None
