# SPDX-License-Identifier: Apache-2.0
"""Authentication and rate limiting middleware."""

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


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, requests_per_minute: int = 60, enabled: bool = False):
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled
        self.window_size = 60.0
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """Check if request is allowed. Returns (is_allowed, retry_after_seconds)."""
        if not self.enabled:
            return True, 0

        current_time = time.time()
        window_start = current_time - self.window_size

        with self._lock:
            if len(self._requests) > 100:
                stale = [
                    k
                    for k, v in self._requests.items()
                    if not v or max(v) <= window_start
                ]
                for k in stale:
                    del self._requests[k]

            self._requests[client_id] = [
                t for t in self._requests[client_id] if t > window_start
            ]

            if len(self._requests[client_id]) >= self.requests_per_minute:
                oldest = min(self._requests[client_id])
                retry_after = int(oldest + self.window_size - current_time) + 1
                return False, max(1, retry_after)

            self._requests[client_id].append(current_time)
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
    return rate_limiter


def _extract_bearer_token(authorization: str | None) -> str | None:
    """Return the raw Bearer token from an Authorization header."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _rate_limit_client_id(request: Request) -> str:
    """Resolve the default client id for rate limiting."""
    authorization = request.headers.get("Authorization")
    if authorization:
        bearer_key = _extract_bearer_token(authorization)
        return bearer_key or authorization

    return request.client.host if request.client else "unknown"


def _anthropic_rate_limit_client_id(request: Request) -> str:
    """Resolve a stable client id for Anthropic-compatible API-key headers."""
    bearer_key = _extract_bearer_token(request.headers.get("Authorization"))
    if bearer_key:
        return bearer_key

    x_api_key = request.headers.get("x-api-key")
    if x_api_key:
        return x_api_key

    return request.client.host if request.client else "unknown"


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
