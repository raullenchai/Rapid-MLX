# SPDX-License-Identifier: Apache-2.0
"""Bearer token + browser-origin gating for the web surface.

Threat model: the moment the user attaches a tunnel this port is on the
public internet, so nothing here may assume the tunnel protects it.

Two independent gates protect every API request:

1. **Bearer token** — always required. The CLI creates and persists one by
   default; ``--token`` may override it. Tunnel authentication is useful
   defence in depth, but cannot be the server's only trust boundary.
2. **Origin / fetch-metadata** — always on, and the one that matters
   against a browser confused deputy: any page the user visits can
   ``fetch()`` a loopback port; it cannot read the response cross-origin,
   but it can cause a side effect. A bearer does not replace this check,
   since a user could paste the token into a malicious page.
"""

from __future__ import annotations

import contextlib
import hmac
import os
import secrets
import stat
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_TOKEN_PATH = Path.home() / ".rapid-mlx" / "web-token"

_TOKEN_BYTES = 32


def generate_token() -> str:
    """Fresh URL-safe secret.

    URL-safe rather than hex so it pastes into a query string or a QR
    code without escaping.
    """
    return secrets.token_urlsafe(_TOKEN_BYTES)


def load_or_create_token(
    path: Path | None = None,
    *,
    override: str | None = None,
    rotate: bool = False,
) -> str:
    """Resolve the bearer for this run.

    The generated token is **persisted**, not rotated per launch: one party
    is a phone browser holding it in
    ``localStorage``, so rotating on every start would silently log the
    user out and the recovery path (walk to the Mac, read the new token,
    retype it) defeats remote access.

    Precedence: explicit ``override`` > existing file > freshly created.
    ``rotate`` forces a new secret even if the file exists.
    """
    if override:
        return override

    path = path or DEFAULT_TOKEN_PATH

    if not rotate and path.exists():
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            _harden_permissions(path)
            return existing

    token = generate_token()
    path.parent.mkdir(parents=True, exist_ok=True)
    # 0600 from the start rather than write-then-chmod: between those two
    # steps the secret is readable by every local user under a 022 umask.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("utf-8"))
    finally:
        os.close(fd)
    _harden_permissions(path)
    return token


def _harden_permissions(path: Path) -> None:
    """Force 0600 on a token file that already existed.

    A file from an older build, a backup or a hand copy can be 0644, and
    reading it silently would leave the secret world-readable.
    """
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return
    if mode != 0o600:
        with contextlib.suppress(OSError):
            path.chmod(0o600)


def extract_bearer(authorization: str | None) -> str | None:
    """Pull the credential out of an ``Authorization`` header.

    Scheme match is case-insensitive per RFC 7235.
    """
    if not authorization:
        return None
    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    credential = credential.strip()
    return credential or None


def token_matches(expected: str, presented: str | None) -> bool:
    """Constant-time comparison."""
    if not presented:
        return False
    return hmac.compare_digest(expected, presented)


def _normalise_authority(value: str) -> str:
    """Reduce a host[:port] to a comparable form.

    Browsers omit the port from ``Origin`` when it is the scheme default,
    while ``Host`` may or may not carry it. Dropping default ports makes
    the two comparable.
    """
    value = value.strip().lower()
    if value.endswith(":80") or value.endswith(":443"):
        value = value.rsplit(":", 1)[0]
    return value


def origin_is_allowed(
    origin: str | None,
    host_header: str | None,
    sec_fetch_site: str | None,
) -> bool:
    """Decide whether a browser-originated request may proceed.

    Rules, in order:

    * No ``Origin`` -> allow. Non-browser clients do not send one, and a
      browser always does on cross-origin requests and every non-GET.
    * ``Sec-Fetch-Site: same-origin``/``none`` -> allow, otherwise deny.
      It is the browser's own verdict and page JS cannot forge it.
    * Otherwise compare the ``Origin`` authority against ``Host``. Under a
      tunnel both carry the tunnel's hostname, so this needs no
      allow-list — the external hostname is not known in advance.
    """
    if origin is None:
        return True

    if sec_fetch_site is not None:
        return sec_fetch_site.strip().lower() in ("same-origin", "none")

    if not host_header:
        return False

    origin_authority = _normalise_authority(urlsplit(origin).netloc)
    if not origin_authority:
        # "null" origin — sandboxed iframe, file:// page, or a redirect
        # that stripped it. Not something a legitimate client produces.
        return False

    return origin_authority == _normalise_authority(host_header)


def content_type_is_json(content_type: str | None) -> bool:
    """Require ``application/json`` on request bodies.

    A CSRF control, not a parsing convenience: ``text/plain``,
    ``application/x-www-form-urlencoded`` and ``multipart/form-data`` are
    the CORS "simple" types a cross-origin page can send with no
    preflight. ``application/json`` is not, so requiring it forces a
    preflight that we then fail.
    """
    if not content_type:
        return False
    return content_type.split(";", 1)[0].strip().lower() == "application/json"


def content_type_is_multipart(content_type: str | None) -> bool:
    """Recognise a browser-generated multipart form with a boundary."""
    if not content_type:
        return False
    media_type, separator, parameters = content_type.partition(";")
    return (
        media_type.strip().lower() == "multipart/form-data"
        and bool(separator)
        and "boundary=" in parameters.lower()
    )
