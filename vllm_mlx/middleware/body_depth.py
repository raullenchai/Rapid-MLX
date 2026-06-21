# SPDX-License-Identifier: Apache-2.0
"""Request-body JSON nesting-depth cap (D-DEEP-JSON DoS defense).

Why this lives at the ASGI layer (not a FastAPI ``Depends`` or
``Request.body()`` check inside each handler):

* FastAPI dependency-injection runs AFTER Pydantic body validation —
  and Pydantic v2 body validation recurses one Python frame per
  JSON nesting level. A payload of ``{"a":{"a":…}}`` 1000 levels deep
  (~10 KB on the wire, well under the body-size cap) exhausted the
  recursion limit inside Pydantic and surfaced as HTTP 500 with a
  stack trace fragment on every body-binding route
  (``/v1/chat/completions``, ``/v1/completions``, ``/v1/embeddings``,
  ``/v1/messages``, ``/v1/responses``). Cross-confirmed against
  five chat-template parsers, all five 500'd identically — proving
  the surface was framework-level, not a model-specific path.

* Running the cap as ASGI middleware lets us reject in one shape:

    1. Read the body bytes (already gated by
       :class:`RequestBodyLimitMiddleware` for size).
    2. Parse with the C JSON accelerator — Python's ``json`` module
       does not use Python recursion, so the parse step itself is
       crash-safe even on extreme nesting.
    3. Walk iteratively (see
       :func:`vllm_mlx.utils.json_depth.json_nesting_depth_exceeds`)
       to measure structural depth.
    4. Reject with the OpenAI-shaped 400 envelope if depth exceeds
       ``RAPID_MLX_MAX_BODY_DEPTH`` (default 64).
    5. Otherwise, replay the buffered body to the downstream app via
       a synthetic ``receive`` so Pydantic still sees the original
       bytes.

The cap is read from the env at request time, so a test fixture that
mutates the env takes effect without rebuilding the FastAPI app — same
pattern :class:`RequestBodyLimitMiddleware` uses.

Path scope mirrors the body-size middleware: only ``/v1/...``,
``/internal/...``, and ``/anthropic/...`` POST/PUT/PATCH/DELETE are
gated, so ``/docs`` / ``/openapi.json`` / ``/healthz`` / ``/metrics``
pay no overhead. The audio transcriptions route is excluded — its
multipart payload is not JSON.
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any

from ..utils.json_depth import (
    json_nesting_depth_exceeds,
    resolve_max_body_depth,
)

logger = logging.getLogger(__name__)


_GUARDED_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_GUARDED_PREFIXES = ("/v1/", "/internal/", "/anthropic/")
# ``/v1/audio/transcriptions`` ships multipart form data, not JSON;
# the body-depth gate would have nothing to measure. The audio
# middleware (``vllm_mlx/routes/audio.py``) owns its own cap.
_EXCLUDED_PATHS = frozenset({"/v1/audio/transcriptions"})


def _path_is_guarded(path: str | None) -> bool:
    if not path:
        return False
    if path in _EXCLUDED_PATHS:
        return False
    return any(path.startswith(prefix) for prefix in _GUARDED_PREFIXES)


def _is_jsonish_content_type(headers) -> bool:
    """Return ``True`` iff the request advertises a JSON-shaped content
    type. We deliberately accept ``application/json``, ``application/
    *+json`` (vendor / draft profiles), and an empty / missing
    ``Content-Type`` (the OpenAI client historically omitted it on
    POST). Anything else (``multipart/form-data``, ``text/plain``,
    ``application/octet-stream``, ...) is left alone so we don't try
    to JSON-parse a binary upload."""
    ctype: str = ""
    for raw_name, raw_value in headers:
        if raw_name.lower() == b"content-type":
            try:
                ctype = raw_value.decode("latin-1").lower()
            except UnicodeDecodeError:
                ctype = ""
            break
    if not ctype:
        return True
    primary = ctype.split(";", 1)[0].strip()
    if primary == "application/json":
        return True
    if primary.startswith("application/") and primary.endswith("+json"):
        return True
    return False


async def _send_400_depth(send, *, max_depth: int) -> None:
    """Emit the OpenAI-shaped 400 envelope for a depth-cap rejection.

    Shape matches the rest of the rapid-mlx 400 surface
    (``middleware/exception_handlers.py::_decode_error_response``) so
    SDKs that key on ``error.code`` / ``error.type`` handle this gate
    the same as a malformed-JSON 400.

    The message names the env knob explicitly (``RAPID_MLX_MAX_BODY_DEPTH``)
    so an operator running into this on a legitimate payload knows
    exactly which lever to turn. No bytes from the request body are
    reflected — the only depth-determined field is the cap integer,
    which is a server-side constant.
    """
    body = _json.dumps(
        {
            "error": {
                "message": (
                    f"Request body JSON nesting depth exceeds the {max_depth}-level "
                    "server cap (set via RAPID_MLX_MAX_BODY_DEPTH)."
                ),
                "type": "invalid_request_error",
                "code": "request_body_too_deep",
                "param": None,
            }
        }
    ).encode("utf-8")
    try:
        await send(
            {
                "type": "http.response.start",
                "status": 400,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})
    except Exception:
        logger.debug("body-depth 400 send failed (client already disconnected)")


class RequestBodyDepthMiddleware:
    """ASGI middleware enforcing :data:`RAPID_MLX_MAX_BODY_DEPTH`.

    Runs AFTER :class:`RequestBodyLimitMiddleware` (so a giant body is
    bounced for size before we try to parse it) but BEFORE FastAPI's
    body-binding Pydantic validators (so a deeply-nested body cannot
    blow the recursion limit inside Pydantic).
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        if scope.get("method") not in _GUARDED_METHODS:
            return await self.app(scope, receive, send)
        path = scope.get("path")
        if not _path_is_guarded(path):
            return await self.app(scope, receive, send)
        max_depth = resolve_max_body_depth()
        if max_depth <= 0:
            return await self.app(scope, receive, send)
        if not _is_jsonish_content_type(scope.get("headers", ())):
            return await self.app(scope, receive, send)

        # Drain the body into memory. The size cap upstream has already
        # bounded ``len(body)`` to ``ServerConfig.max_request_bytes``
        # (8 MiB by default), so this is bounded memory — and we MUST
        # buffer to inspect the JSON structure before letting it reach
        # Pydantic.
        chunks: list[bytes] = []
        while True:
            msg = await receive()
            mtype = msg.get("type")
            if mtype == "http.request":
                chunk = msg.get("body", b"") or b""
                if chunk:
                    chunks.append(chunk)
                if not msg.get("more_body", False):
                    break
            elif mtype == "http.disconnect":
                # Client gave up before sending the whole body. Forward
                # the disconnect so the downstream app's cancellation
                # path runs — the depth gate has nothing to enforce on
                # a half-shipped body.
                return await self.app(scope, _replay_with_disconnect(chunks), send)
            else:
                # Unknown ASGI message — forward as-is and let the
                # downstream app decide what to do.
                chunks.append(b"")
        body = b"".join(chunks)

        if not body.strip():
            # Empty body: no JSON to parse, no depth to enforce.
            return await self.app(scope, _replay_buffered(body), send)

        try:
            parsed = _json.loads(body)
        except _json.JSONDecodeError:
            # Malformed JSON — leave it to FastAPI's
            # :class:`json.JSONDecodeError` handler (registered in
            # ``middleware/exception_handlers.py``) so the 400 shape
            # stays consistent. We just forward the buffered body and
            # let the downstream app try to parse it again. The double
            # parse is unavoidable: the depth gate has to know the
            # structure to measure it, and the downstream FastAPI body
            # reader insists on owning the parse itself.
            return await self.app(scope, _replay_buffered(body), send)

        if json_nesting_depth_exceeds(parsed, max_depth):
            await _send_400_depth(send, max_depth=max_depth)
            return

        return await self.app(scope, _replay_buffered(body), send)


def _replay_buffered(body: bytes):
    """Build a synthetic ``receive`` that ships ``body`` in one frame.

    The downstream app sees the same bytes the client sent — Pydantic
    re-parses from these bytes, so its view of the request is identical
    to the no-middleware case (apart from the depth check we already
    cleared).

    Once the body frame is consumed, subsequent ``receive()`` calls
    return ``http.disconnect`` so a downstream that polls
    ``request.is_disconnected()`` after reading the body sees the
    correct lifecycle. Mirrors Starlette's own
    ``Request._receive_after_body`` shape.
    """
    sent = {"value": False}

    async def receive():
        if not sent["value"]:
            sent["value"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    return receive


def _replay_with_disconnect(chunks: list[bytes]):
    """Build a synthetic ``receive`` that replays already-read chunks
    and then yields the ``http.disconnect`` that closed the original
    stream.

    Used on the early-disconnect path: we buffered partial chunks
    while reading the body, then saw ``http.disconnect`` — replay the
    partial body to the downstream app so its cancellation logic
    runs the same way it would have without the middleware in the
    path.
    """
    idx = {"value": 0}

    async def receive():
        i = idx["value"]
        if i < len(chunks):
            idx["value"] = i + 1
            more_body = (i + 1) < len(chunks)
            return {
                "type": "http.request",
                "body": chunks[i],
                "more_body": more_body,
            }
        return {"type": "http.disconnect"}

    return receive


def install_request_body_depth_middleware(app: Any) -> None:
    """Attach :class:`RequestBodyDepthMiddleware` to ``app``.

    Centralised for the same reason
    :func:`vllm_mlx.middleware.body_size.install_request_body_limit_middleware`
    is — keeps the wiring discoverable from this module and gives tests
    a single hook to call against a minimal app fixture."""
    app.add_middleware(RequestBodyDepthMiddleware)
