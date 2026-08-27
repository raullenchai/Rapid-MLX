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
import shlex
from dataclasses import dataclass
from typing import Any


def _authority(host: str) -> str:
    """Render a host for a URI authority component, canonicalizing it.

    IPv6 literals (and scoped literals like ``fe80::1%en0``) MUST be wrapped
    in square brackets in a URI authority, e.g. ``http://[::1]:8000``. Uses
    the same lexical ``":" in host`` rule as the serve CLI's ``_is_ipv6_host``
    so zone-id scoped addresses are also detected.

    The zone-id separator ``%`` is percent-encoded as ``%25`` per RFC 6874
    (zone identifiers are delimited by ``%25`` in URIs), e.g.
    ``http://[fe80::1%25en0]:8000``.

    Accepts and normalizes non-canonical input rather than guessing:
    an already-bracketed literal (``[::1]``) is not double-bracketed, and an
    already-encoded ``%25`` is not percent-encoded a second time.
    """
    # Strip an existing surrounding bracket pair so we never double-bracket.
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]

    if ":" not in host:
        return host

    # Percent-encode a raw zone-id `%` as `%25` unless it already begins a
    # valid `%25` escape (case-insensitive). This keeps an already-encoded
    # ``fe80::1%25en0`` stable while encoding a bare ``fe80::1%en0``.
    if "%" in host:
        out: list[str] = []
        i = 0
        while i < len(host):
            if host[i] == "%" and host[i : i + 3].lower() != "%25":
                out.append("%25")
                i += 1
            else:
                out.append(host[i])
                i += 1
        host = "".join(out)

    return f"[{host}]"


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
        return f"http://{_authority(self.host)}:{self.port}"

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
# Each row is ``(app label, command prefix, OpenAI endpoint?)``. Every row —
# including ``openai-python`` — takes an OpenAI-style ``/v1`` base URL and
# gets ``--base-url <url>`` appended so the copied command targets the
# *actual* running server, not the localhost:8000 default the standalone
# process would otherwise assume (#2348). ``connect`` accepts ``--base-url``
# as the explicit way to receive that instance context, so the pasted command
# resolves the real host/port/model instead of printing placeholders.
_CONNECT_ROWS: list[tuple[str, str, bool]] = [
    ("Claude Code", "rapid-mlx agents claude-code --setup", True),
    ("Continue", "rapid-mlx agents continue --setup", True),
    ("Python", "rapid-mlx connect openai-python", True),
]


def render_banner(
    ep: ServerEndpoints, *, include_connect: bool = True, running: bool = True
) -> str:
    """Render the human "Ready:" / "OpenAI:" / "Connect:" block.

    Used by the serve lifespan (once warmup completes) and by
    ``rapid-mlx connect``. Rendered centrally so the served banner and the
    standalone ``connect`` output can never disagree about an endpoint.

    ``running`` is the serve default: the lifespan only calls this once the
    server is actually up. ``rapid-mlx connect`` passes ``running=False`` when
    a liveness probe found nothing on the target, so the banner does not claim
    "Ready:" for an address that refuses connections (#1999) — it says there is
    no server and how to start one, and drops the Connect cheat-sheet that
    would otherwise wire a client to a dead endpoint.
    """
    lines: list[str] = []

    if ep.listen_fd is not None:
        lines.append(f"  Ready: inherited fd {ep.listen_fd}")
    elif not running:
        port_hint = "" if ep.port == 8000 else f" --port {ep.port}"
        lines.append(f"  No rapid-mlx server on {ep.base_url}")
        lines.append(f"  Start one with:  rapid-mlx serve <model>{port_hint}")
        lines.append("")
        lines.append("  A server there would expose:")
        lines.append(f"  OpenAI:    {ep.openai_url}")
        lines.append(f"  Anthropic: {ep.anthropic_url}")
        return "\n".join(lines) + "\n"
    else:
        lines.append(f"  Ready: {ep.base_url}")
        lines.append("")
        lines.append(f"  OpenAI:    {ep.openai_url}")
        lines.append(f"  Anthropic: {ep.anthropic_url}")
        lines.append(f"  Model:     {ep.model or '(not yet loaded)'}")

    if include_connect and ep.listen_fd is None:
        lines.append("")
        lines.append("  Connect:")
        width = max(len(a) for a, _, _ in _CONNECT_ROWS)
        for app, cmd, needs_endpoint in _CONNECT_ROWS:
            rendered = cmd
            if needs_endpoint:
                rendered += f" --base-url {shlex.quote(ep.openai_url)}"
            lines.append(f"    {app:<{width}}  {rendered}")

    return "\n".join(lines) + "\n"


def probe_server_alive(host: str, port: int) -> bool:
    """Best-effort liveness check: is a rapid-mlx server answering on host:port?

    Hits ``/healthz`` — the middleware fast-path that answers without touching
    the engine, so it can't stall under load the way ``/health`` can. The route
    is rapid-mlx-specific, so we key on the response CODE rather than the body
    (which differs across the standard / DFlash / DDTree servers):

    * any 2xx, or an auth challenge (401/403 — a DDTree server can gate
      ``/healthz`` behind its API key) → a server that ``connect`` can point a
      client at;
    * any other status (404 from an unrelated HTTP server, 503 from a server
      that is up but draining and refusing new work, 5xx from a broken one) →
      not something to hand a client;
    * connection refused / timeout / DNS → nothing is running.

    Never raises.
    """
    import http.client
    import urllib.error
    import urllib.request

    url = f"http://{_authority(host)}:{port}/healthz"
    try:
        urllib.request.urlopen(url, timeout=2.0)
        return True
    except urllib.error.HTTPError as exc:
        return exc.code in (401, 403)
    # http.client.HTTPException (e.g. BadStatusLine) covers a port occupied by a
    # non-HTTP service (SSH, a database) that answers with garbage — not a
    # server to point a client at, and it must not crash ``connect``.
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError):
        return False


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


def _parse_base_url(base_url: str) -> tuple[str, int]:
    """Split an OpenAI-style base URL into ``(host, port)``.

    ``rapid-mlx connect`` accepts ``--base-url`` as the explicit way to carry
    the *live* server context across process boundaries (#2348). Without it a
    standalone ``connect`` process cannot know the port/model a ``serve``
    picked, so it would print the localhost:8000 default instead of the real
    endpoint. A trailing ``/v1`` (OpenAI-style, what the banner prints) is
    accepted and ignored for host/port derivation.

    Accepts and normalizes the same host shapes as :func:`_authority` —
    bare IPv4/hostnames and bracket-wrapped or scoped IPv6 literals — so a
    user can paste the banner URL verbatim. A missing port falls back to the
    rapid-mlx default (8000).

    ``connect`` targets a local ``http://`` server whose endpoints the SSOT
    models as ``http://host:port`` (+ a literal ``/v1`` for OpenAI) — it has no
    notion of a URL path prefix. So, mirroring the scheme check, any path other
    than empty or ``/v1`` is rejected with a clear error rather than silently
    rewriting a proxied base URL to the wrong API (codex #2348-R2). An explicit
    port is validated against the CLI ``_port_arg`` 1-65535 invariant, so ``:0``
    or out-of-range values fail loudly instead of retargeting the snippet.

    Raises ``ValueError`` on a non-http scheme, a URL with no host, an
    unexpected path, or an explicit out-of-range port.
    """
    from urllib.parse import urlsplit

    split = urlsplit(base_url, scheme="http")
    scheme = split.scheme or "http"
    if scheme != "http":
        raise ValueError(f"base-url must be an http URL, got {base_url!r}")
    host = split.hostname
    if not host:
        raise ValueError(f"base-url has no host, got {base_url!r}")
    # Empty path (``http://host:port``), a bare trailing slash (``/``), and
    # ``/v1`` (+ a single trailing slash) are the only paths the SSOT renders
    # losslessly; one trailing slash is a semantics-free spelling from SDK/config
    # tooling and is normalized away before comparing. Repeated slashes
    # (``//``, ``/v1//``) and real path-prefixes are rejected since they are
    # path-significant and this local-server tool does not model them.
    path = ""
    raw = split.path or ""
    if raw:
        path = raw[:-1] if raw.endswith("/") else raw
    if path not in ("", "/v1"):
        raise ValueError(f"base-url path must be empty or /v1, got {base_url!r}")
    # A scoped IPv6 literal is the only host that carries a ``%25`` zone-id
    # separator (``fe80::1%25en0`` -> raw ``fe80::1%en0``); :func:`_authority`
    # then re-encodes it consistently when rendering, so the round-trip is
    # stable. Detect IPv6 the same way ``_authority`` does (a ``:`` in the
    # host) and decode ``%25`` ONLY there — in an ordinary hostname ``%25`` is
    # a genuine encoded octet that must be left untouched, or the generated
    # URL would be malformed (codex #2348-R2/R3).
    if ":" in host:
        host = host.replace("%25", "%")
    port = split.port if split.port is not None else 8000
    if not (1 <= port <= 65535):
        raise ValueError(f"base-url port must be between 1 and 65535, got {port}")
    return host, port


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

    # Explicit --model wins outright; then config model fields; neither
    # present means we may fall back to probing a *running* server.
    out_model = model or cfg.model_alias or cfg.model_name

    # Probe (best-effort) only when no model was supplied anywhere AND no
    # authoritative bind config pins the target. The probe result must never
    # overwrite an explicit `--model` (flags > config > defaults).
    if out_model is None and cfg.bind_host is None and cfg.bind_port is None:
        detected_model = _probe_running_model(out_host, out_port)
        if detected_model:
            out_model = detected_model

    return ServerEndpoints(out_host, out_port, model=out_model)


def _probe_running_model(host: str, port: int) -> str | None:
    """Best-effort: ask a running server which model it serves. Never raises."""
    import urllib.error
    import urllib.request

    base = f"http://{_authority(host)}:{port}"
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
