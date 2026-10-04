#!/usr/bin/env python3
"""Purge and verify the two canonical Gemma index URLs for the 0.15.6 release.

This is intentionally a one-purpose operation. URLs and expected bytes are
constants so a workflow dispatch cannot widen the purge target.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

API = "https://api.cloudflare.com/client/v4"
EXPECTED_ZONE = "rapidmlx.com"
EXPECTED_SIZE = 176_940
EXPECTED_SHA256 = "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13"
EXPECTED_TOTAL_SIZE = 15_340_981_404
INDEX_PATH = "mlx-community/gemma-4-26b-a4b-it-4bit/model.safetensors.index.json"
PURGE_URLS = (
    f"https://models.rapidmlx.com/{INDEX_PATH}",
    f"https://dl.models.rapidmlx.com/{INDEX_PATH}",
)
TIMEOUT = 30
USER_AGENT = "rapid-mlx-release-model-cdn-repair"


class OperationError(RuntimeError):
    """A fail-closed operational contract violation."""


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Reject every redirect before an authenticated follow-up can be sent."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise OperationError(f"redirect refused for authenticated API request ({code})")


class CanonicalIndexRedirect(urllib.request.HTTPRedirectHandler):
    """Allow only the canonical public index redirect, with fresh safe headers."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if (
            code != 302
            or req.get_method() != "GET"
            or req.full_url != PURGE_URLS[0]
            or newurl != PURGE_URLS[1]
            or req.get_header("Authorization") is not None
        ):
            raise OperationError(f"unexpected public index redirect ({code})")
        return urllib.request.Request(
            PURGE_URLS[1],
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        )


def _open(
    request: urllib.request.Request,
    redirect_handler: urllib.request.HTTPRedirectHandler,
):
    return urllib.request.build_opener(redirect_handler).open(request, timeout=TIMEOUT)


def _request(
    request: urllib.request.Request,
    redirect_handler: urllib.request.HTTPRedirectHandler,
    *,
    expected_status: int = 200,
) -> bytes:
    try:
        with _open(request, redirect_handler) as response:
            body = response.read(EXPECTED_SIZE + 1)
            status = response.status
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OperationError(f"request failed for {request.full_url}: {exc}") from exc
    if status != expected_status:
        raise OperationError(
            f"unexpected HTTP {status} for {request.full_url}; expected {expected_status}"
        )
    return body


def _api_json(
    path: str,
    token: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    raw = _request(request, RejectRedirects())
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperationError(f"Cloudflare returned invalid JSON for {path}") from exc
    if not isinstance(body, dict) or body.get("success") is not True:
        raise OperationError(f"Cloudflare rejected the request for {path}")
    return body


def verify_scope(token: str, zone_id: str) -> None:
    token_result = _api_json("/user/tokens/verify", token).get("result") or {}
    if token_result.get("status") != "active":
        raise OperationError("Cloudflare token is not active")

    zone_result = _api_json(f"/zones/{zone_id}", token).get("result") or {}
    if zone_result.get("name") != EXPECTED_ZONE:
        raise OperationError("configured Cloudflare zone is not rapidmlx.com")
    if zone_result.get("status") != "active":
        raise OperationError("rapidmlx.com Cloudflare zone is not active")


def purge(token: str, zone_id: str) -> None:
    _api_json(
        f"/zones/{zone_id}/purge_cache",
        token,
        method="POST",
        payload={"files": list(PURGE_URLS)},
    )


def verify_public_bytes() -> None:
    for url in PURGE_URLS:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        )
        body = _request(request, CanonicalIndexRedirect())
        if len(body) != EXPECTED_SIZE:
            raise OperationError(
                f"canonical index at {url} is {len(body)} bytes; expected {EXPECTED_SIZE}"
            )
        digest = hashlib.sha256(body).hexdigest()
        if digest != EXPECTED_SHA256:
            raise OperationError(
                f"canonical index at {url} has SHA-256 {digest}; expected {EXPECTED_SHA256}"
            )
        try:
            total_size = json.loads(body)["metadata"]["total_size"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise OperationError(
                f"canonical index at {url} has invalid metadata"
            ) from exc
        if total_size != EXPECTED_TOTAL_SIZE:
            raise OperationError(
                f"canonical index at {url} has metadata.total_size={total_size}; "
                f"expected {EXPECTED_TOTAL_SIZE}"
            )


def main() -> int:
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    zone_id = os.environ.get("CLOUDFLARE_ZONE_ID", "")
    if not token or not zone_id:
        print(
            "error: hosted Cloudflare token and zone id are required", file=sys.stderr
        )
        return 1
    try:
        verify_scope(token, zone_id)
        purge(token, zone_id)
        verify_public_bytes()
    except OperationError as exc:
        message = str(exc).replace(token, "<redacted>").replace(zone_id, "<redacted>")
        print(f"error: {message}", file=sys.stderr)
        return 1
    print(
        "targeted Gemma index CDN purge verified: both canonical URLs match "
        f"{EXPECTED_SHA256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
