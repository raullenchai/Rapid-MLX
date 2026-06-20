# SPDX-License-Identifier: Apache-2.0
"""Authentication and rate limiting middleware."""

import hashlib
import hmac

# ``ipaddress`` is imported by ``_subnet_bucket`` (rate-limit helper) and
# ``_is_loopback_client`` (kept as dead code for future control-plane
# gates after #728 was reverted). Canonicalising via ``ipaddress`` is the
# only safe way to identify loopback callers — string comparison misses
# ``::ffff:127.0.0.1`` and IPv6 spellings.
import ipaddress
import logging
import secrets
import threading
import time
from collections import defaultdict

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import get_config

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

_auth_warning_logged: bool = False

_RATE_LIMIT_HMAC_KEY = secrets.token_bytes(32)


class RateLimiter:
    """Simple in-memory rate limiter using sliding window.

    Stale-entry cleanup is amortized O(1) per request:

    * A separate ``_last_seen`` map records each client's most recent timestamp
      so the sweep is O(N) over the dict (no per-entry ``max(v)`` scan, which
      previously made the worst case quadratic when the dict held >100 active
      clients — each request walked every entry, each entry scanned its
      window-length timestamp list).
    * The sweep runs at most once per ``window_size`` interval rather than on
      every request, so a busy server with >100 active clients no longer pays
      O(N) per request just to walk the dict.
    * The current client's entry is always preserved during the sweep — only
      entries whose newest timestamp is strictly older than ``window_start``
      are deleted, so a client that made any request inside the window cannot
      have its counter wiped out by an unrelated concurrent client.
    """

    # Soft cap that gates cleanup. Kept at 100 to preserve historical behavior
    # (and the existing regression test in test_server.py).
    _CLEANUP_DICT_THRESHOLD = 100

    def __init__(self, requests_per_minute: int = 60, enabled: bool = False):
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled
        self.window_size = 60.0
        self._requests: dict[str, list[float]] = defaultdict(list)
        # ``_last_seen[k]`` mirrors ``max(_requests[k])`` for every tracked k,
        # so the sweep avoids the per-entry ``max(v)`` scan that drove the
        # quadratic worst case.
        self._last_seen: dict[str, float] = {}
        # Throttle is keyed off ``time.monotonic()`` so a backwards NTP step
        # cannot suppress sweeps far longer than ``window_size``. Wall time
        # is still used for the request-window math (timestamps must round-
        # trip with retry-after semantics). Initialized to ``-inf`` so the
        # first eligible request always sweeps, even on a freshly booted
        # system where ``time.monotonic()`` may itself be smaller than
        # ``window_size``.
        self._last_cleanup_mono: float = float("-inf")
        self._lock = threading.Lock()

    def _maybe_cleanup(self, window_start: float, client_id: str) -> None:
        """Sweep stale entries. Must be called with ``self._lock`` held.

        Throttled to at most once per ``window_size`` seconds (measured on
        the monotonic clock so NTP/wall-clock adjustments cannot starve it)
        and skipped entirely when the dict is small. The current ``client_id``
        is never evicted — removing it here would either be a wasted op (the
        caller is about to re-create it) or race-prone in the rare case the
        sweep encountered a brand-new defaultdict-inserted empty list for it.
        """
        if len(self._requests) <= self._CLEANUP_DICT_THRESHOLD:
            return
        now_mono = time.monotonic()
        if now_mono - self._last_cleanup_mono < self.window_size:
            return
        self._last_cleanup_mono = now_mono

        # Snapshot keys to avoid mutating during iteration.
        for k in list(self._requests.keys()):
            if k == client_id:
                # Never reap the entry that's about to be touched.
                continue
            last = self._last_seen.get(k)
            if last is None or last <= window_start:
                self._requests.pop(k, None)
                self._last_seen.pop(k, None)

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """Check if request is allowed. Returns (is_allowed, retry_after_seconds)."""
        if not self.enabled:
            return True, 0

        current_time = time.time()
        window_start = current_time - self.window_size

        with self._lock:
            self._maybe_cleanup(window_start, client_id)

            # Filter the current client's window. Strict ``>`` matches the
            # original semantics (timestamps exactly at window_start are out).
            timestamps = [t for t in self._requests[client_id] if t > window_start]
            self._requests[client_id] = timestamps

            if len(timestamps) >= self.requests_per_minute:
                # ``min`` matches the original semantics; timestamps are
                # append-ordered so this is also the head.
                oldest = min(timestamps)
                retry_after = int(oldest + self.window_size - current_time) + 1
                # Track the rejected request's time so an entry that's still
                # actively probing the limit isn't considered stale on the
                # next sweep.
                self._last_seen[client_id] = current_time
                return False, max(1, retry_after)

            timestamps.append(current_time)
            self._last_seen[client_id] = current_time
            return True, 0


# Global rate limiter (disabled by default, configured via --rate-limit)
rate_limiter = RateLimiter(requests_per_minute=60, enabled=False)


def configure_rate_limiter(
    requests_per_minute: int,
    *,
    enabled: bool = True,
) -> RateLimiter:
    """Configure the shared rate limiter object used by FastAPI dependencies."""
    with rate_limiter._lock:
        rate_limiter.requests_per_minute = requests_per_minute
        rate_limiter.enabled = enabled
        rate_limiter._requests.clear()
        rate_limiter._last_seen.clear()
        rate_limiter._last_cleanup_mono = float("-inf")
    return rate_limiter


def _extract_bearer_token(authorization: str | None) -> str | None:
    """Return the raw Bearer token from an Authorization header."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _bucket_id(raw: str) -> str:
    """HMAC-SHA256 with a per-process random key.

    The HMAC key prevents an attacker who knows the SHA-256 of a secret
    from mapping it back to the secret via a rainbow table of candidate
    values.  Each process restart also rotates the key, so even if a
    hash leaks across the wire it becomes useless after the process
    exits.
    """
    return hmac.new(_RATE_LIMIT_HMAC_KEY, raw.encode(), hashlib.sha256).hexdigest()[:16]


def _subnet_bucket(host: str) -> str:
    """Group IPv4 hosts into /24 and IPv6 hosts into /64 subnets."""
    try:
        addr = ipaddress.ip_address(host)
        if isinstance(addr, ipaddress.IPv4Address):
            network = ipaddress.ip_network(f"{addr}/24", strict=False)
        else:
            network = ipaddress.ip_network(f"{addr}/64", strict=False)
        return str(network.network_address)
    except ValueError:
        return host


def _rate_limit_client_id(request: Request) -> str:
    """Resolve the default client id for rate limiting."""
    authorization = request.headers.get("Authorization")
    if authorization:
        bearer_key = _extract_bearer_token(authorization)
        raw = bearer_key or authorization
        return _bucket_id(raw)

    if request.client and request.client.host:
        return _subnet_bucket(request.client.host)
    return "unknown"


def _anthropic_rate_limit_client_id(request: Request) -> str:
    """Resolve a stable client id for Anthropic-compatible API-key headers."""
    bearer_key = _extract_bearer_token(request.headers.get("Authorization"))
    if bearer_key:
        return _bucket_id(bearer_key)

    x_api_key = request.headers.get("x-api-key")
    if x_api_key:
        return _bucket_id(x_api_key)

    if request.client and request.client.host:
        return _subnet_bucket(request.client.host)
    return "unknown"


async def check_rate_limit(request: Request):
    """Rate limiting dependency for FastAPI."""
    client_id = _rate_limit_client_id(request)

    allowed, retry_after = rate_limiter.is_allowed(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


async def check_rate_limit_or_x_api_key(request: Request):
    """Rate limiting dependency for Anthropic-compatible API-key headers."""
    client_id = _anthropic_rate_limit_client_id(request)

    allowed, retry_after = rate_limiter.is_allowed(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


def _verify_api_key_values(*api_keys: str | None) -> bool:
    """Verify one or more API key values against the configured key."""
    global _auth_warning_logged

    cfg = get_config()

    if cfg.api_key is None:
        if not _auth_warning_logged:
            logger.debug(
                "No API key configured. Use --api-key to enable authentication."
            )
            _auth_warning_logged = True
        return True

    provided_keys = [api_key for api_key in api_keys if api_key]
    if not provided_keys:
        raise HTTPException(status_code=401, detail="API key required")
    if not all(
        secrets.compare_digest(api_key, cfg.api_key) for api_key in provided_keys
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key if authentication is enabled."""
    bearer_key = credentials.credentials if credentials is not None else None
    return _verify_api_key_values(bearer_key)


async def verify_api_key_or_x_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Verify OpenAI Bearer auth or Anthropic x-api-key auth."""
    bearer_key = credentials.credentials if credentials is not None else None
    return _verify_api_key_values(bearer_key, request.headers.get("x-api-key"))


# Loopback hosts the operator might appear under. We canonicalise via
# ``ipaddress`` to catch ``::ffff:127.0.0.1`` and the various IPv6 spellings
# of ``::1`` (``0000:0:0:0:0:0:0:1`` etc.) — a string-equality check would
# miss those and let a remote attacker who controls DNS / a reverse proxy
# forge ``127.0.0.1`` as a literal but fail any canonical form.
#
# Note: ``_is_loopback_client`` is kept as dead code in case a future PR
# reintroduces a control-plane gate. The companion ``verify_internal_admin``
# dependency was removed when operator-intent reverted #728 — single-machine
# UX means the destructive routes run on plain ``verify_api_key`` (no-op
# when ``--api-key`` unset). Cancel-envelope sanitization + scheduler
# ``abort_request`` correctness from #728 are kept; only the auth gate goes.
_LOOPBACK_LITERALS = frozenset({"127.0.0.1", "::1", "localhost"})


def _is_loopback_client(request: Request) -> bool:
    """True when ``request.client.host`` is a loopback address AND the
    request has no reverse-proxy fingerprint.

    Uses ``ipaddress.is_loopback`` so canonical IPv4 (``127.0.0.0/8``) and
    IPv6 (``::1``, ``::ffff:127.0.0.1``) variants all match — defending
    against an attacker who probes for non-canonical loopback spellings to
    see if the gate is implemented with a string compare.

    Codex PR #728 round-3 BLOCKING: a same-host reverse proxy (nginx
    ``proxy_pass http://127.0.0.1:8000``) makes every external client look
    like ``127.0.0.1`` to the Worker, so a naive loopback check is a hole.
    We harden by also REJECTING any request carrying a forwarding-trail
    header. The header set covers:

      * ``X-Forwarded-For`` / ``X-Forwarded-Host`` / ``X-Forwarded-Proto``
        (de-facto standard set added by nginx / Apache / HAProxy / ALB /
        Cloudflare).
      * ``Forwarded`` (RFC 7239 — set by Apache 2.4.10+).
      * ``Via`` (RFC 7230 — added by every HTTP/1.1-compliant proxy).
      * ``CF-Connecting-IP`` (Cloudflare-specific).
      * ``True-Client-IP`` (Akamai / Cloudflare Enterprise).

    A direct loopback caller from the same machine has none of these
    (loopback ``curl`` doesn't set them; same-process supervised processes
    don't either). The trade-off: an attacker who controls a header-
    stripping proxy can still spoof — but that's a much higher bar than
    "the default nginx config", and operators in that posture should set
    ``--api-key`` (production posture) instead.
    """
    client = request.client
    if client is None or not client.host:
        return False

    # Proxy-trail check first — cheaper than the IP parse, and any
    # forwarded header is an immediate disqualifier regardless of host.
    forwarded_headers = (
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "forwarded",
        "via",
        "cf-connecting-ip",
        "true-client-ip",
    )
    for h in forwarded_headers:
        if request.headers.get(h):
            return False

    host = client.host
    if host in _LOOPBACK_LITERALS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# ``verify_internal_admin`` removed per operator-intent revert of #728
# (single-machine UX, no API key gate on cache-clear / request-cancel).
# If a future PR needs to gate these routes again, reuse
# ``_is_loopback_client`` above + the standard ``verify_api_key`` /
# ``HTTPException`` pattern.
