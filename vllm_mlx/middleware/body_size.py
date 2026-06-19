# SPDX-License-Identifier: Apache-2.0
"""Generic request-body size cap for all POST/PUT/PATCH routes.

Why this lives at the ASGI layer (not a FastAPI ``Depends`` or
``Request.body()`` check inside each handler):

* FastAPI dependency-injection runs AFTER Starlette has already invoked
  ``json.loads`` on the request body to populate the route's Pydantic
  model. By the time a ``Depends`` callable fires, the entire payload
  has been read off ``receive`` and JSON-decoded — a 100 MB body of
  random bytes has already cost a worker ~1 s of pure JSON parsing and
  is sitting in RAM (plus a second copy as the Pydantic model). The
  symptom F-007 documented (rapid-desktop#273): 10 MB body → ~60 s
  full-prefill hang, 100 MB → ~90 s. Both bodies were ASCII so JSON
  parsing succeeded; the worker then ran prefill against a multi-million-
  token prompt and the client gave up before the response started.

* Running the cap as ASGI middleware lets us reject in two ways:

    1. **Honest ``Content-Length`` fast path** — advertised length over
       cap → 413 with zero ``receive`` calls. Cheapest possible bounce.

    2. **Streaming slow path** — wrap ``receive`` so it tallies streamed
       body bytes. When the running total exceeds the cap we inject a
       synthetic ``http.disconnect`` (Starlette's JSON parser honors it,
       stops reading, raises) and the wrapper emits the 413 from the
       outside. This covers ``Transfer-Encoding: chunked`` and any
       client that omits or understates ``Content-Length``.

Mirrors :class:`vllm_mlx.routes.audio.AudioBodyLimitMiddleware` but
applies to a different scope — see ``_GUARDED_METHODS`` /
``_path_is_guarded``: every POST/PUT/PATCH/DELETE to ``/v1/...``
(chat, completions, embeddings, models, audio, anthropic, mcp, …).

The cap is read from ``ServerConfig.max_request_bytes`` at request
time, so ``rapid-mlx serve --max-request-bytes N`` (or the env var)
takes effect without restart in tests that mutate the config.
A value of ``0`` disables the cap entirely (escape hatch for
operators whose internal deployments have other DoS controls).
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any

from ..config.server_config import get_config

logger = logging.getLogger(__name__)


# Methods that can carry a body large enough to be a DoS surface.
# GET/HEAD/OPTIONS/TRACE have no meaningful body in our routes, so we
# skip the receive-wrapper overhead for them.
_GUARDED_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Path prefixes the cap applies to. We deliberately scope to ``/v1/``
# (and ``/internal/`` for parity) so internal probes like
# ``/docs``/``/openapi.json``/``/metrics`` are never gated — those
# endpoints have no body and the middleware would just add overhead.
_GUARDED_PREFIXES = ("/v1/", "/internal/", "/anthropic/")

# Paths owned by a more specific middleware that applies its own,
# domain-appropriate cap. We skip them here so the generic 8 MiB JSON
# cap doesn't trample a 25 MB Whisper upload that the audio middleware
# already validates. See ``vllm_mlx/routes/audio.py``.
_EXCLUDED_PATHS = frozenset({"/v1/audio/transcriptions"})


def _path_is_guarded(path: str | None) -> bool:
    if not path:
        return False
    if path in _EXCLUDED_PATHS:
        return False
    return any(path.startswith(prefix) for prefix in _GUARDED_PREFIXES)


def _resolve_limit() -> int:
    """Look up the current cap from the live ServerConfig singleton.

    Resolving per-request (not at middleware init) lets tests mutate
    ``ServerConfig.max_request_bytes`` between cases without rebuilding
    the FastAPI app. The cost is one attribute access on a dataclass,
    negligible against the per-request socket round-trip.
    """
    try:
        cap = int(get_config().max_request_bytes)
    except Exception:
        # Defensive fallback: if the config singleton is in some
        # half-initialised state during shutdown, fall back to the
        # dataclass default rather than crashing the request. 0 here
        # would mean "unlimited" — silently disabling the gate is a
        # worse outcome than a sane default.
        cap = 8 * 1024 * 1024
    return max(0, cap)


class RequestBodyLimitMiddleware:
    """ASGI middleware enforcing :attr:`ServerConfig.max_request_bytes`."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        if scope.get("method") not in _GUARDED_METHODS:
            return await self.app(scope, receive, send)
        if not _path_is_guarded(scope.get("path")):
            return await self.app(scope, receive, send)

        limit = _resolve_limit()
        if limit == 0:
            # Cap explicitly disabled by operator. Skip the wrapper to
            # avoid per-message overhead.
            return await self.app(scope, receive, send)

        # Fast path: honest Content-Length.
        advertised: int | None = None
        for raw_name, raw_value in scope.get("headers", ()):
            if raw_name.lower() == b"content-length":
                try:
                    advertised = int(raw_value.decode("latin-1"))
                except (UnicodeDecodeError, ValueError):
                    advertised = None
                break

        if advertised is not None and advertised > limit:
            await _send_413(
                send,
                advertised=advertised,
                limit=limit,
                streaming=False,
            )
            return

        # Slow path: chunked / no-Content-Length / lying Content-Length.
        # Wrap receive so we abort the moment the running total exceeds
        # the cap. We do NOT trust the Content-Length even when it's
        # under the cap — a lying client could advertise 1 KB and then
        # stream 100 MB, so the tally guards both.
        tripped = {"value": False}
        total = {"bytes": 0}

        async def bounded_receive():
            if tripped["value"]:
                # Signal disconnect so Starlette's JSON parser exits
                # cleanly. This is the same shape AudioBodyLimitMiddleware
                # uses — confirmed empirically that fastapi's
                # ``Request.json()`` raises ``ClientDisconnect`` when
                # receive yields disconnect, which the wrap-send below
                # turns into our documented 413.
                return {"type": "http.disconnect"}
            msg = await receive()
            if msg.get("type") == "http.request":
                body_len = len(msg.get("body", b"") or b"")
                total["bytes"] += body_len
                if total["bytes"] > limit:
                    tripped["value"] = True
                    return {"type": "http.disconnect"}
            return msg

        sent_413 = {"value": False}

        async def guarded_send(msg):
            if tripped["value"] and not sent_413["value"]:
                sent_413["value"] = True
                await _send_413(
                    send,
                    advertised=None,
                    limit=limit,
                    streaming=True,
                    streamed=total["bytes"],
                )
                return
            if sent_413["value"]:
                # Downstream tried to send after we already wrote 413;
                # drop the message to avoid double-write.
                return
            await send(msg)

        try:
            await self.app(scope, bounded_receive, guarded_send)
        except Exception:
            # If we tripped the cap, the downstream app raised because
            # of the synthetic disconnect we injected — translate that
            # into the documented 413. Anything else is a real error;
            # surface it normally.
            if not tripped["value"]:
                raise

        # Fallback 413 emission: catches the silent-drop-on-disconnect
        # path where Starlette returns without sending a response at
        # all after seeing our disconnect, and the swallowed-exception
        # path above.
        if tripped["value"] and not sent_413["value"]:
            sent_413["value"] = True
            await _send_413(
                send,
                advertised=None,
                limit=limit,
                streaming=True,
                streamed=total["bytes"],
            )


def _format_message(
    *,
    advertised: int | None,
    limit: int,
    streaming: bool,
    streamed: int | None = None,
) -> str:
    if streaming:
        return (
            f"Request body too large: streamed {streamed or 0} bytes "
            f"exceeded the {limit}-byte server cap "
            "(set via --max-request-bytes / RAPID_MLX_MAX_REQUEST_BYTES)"
        )
    return (
        f"Request body too large: Content-Length {advertised} bytes "
        f"exceeds the {limit}-byte server cap "
        "(set via --max-request-bytes / RAPID_MLX_MAX_REQUEST_BYTES)"
    )


async def _send_413(
    send,
    *,
    advertised: int | None,
    limit: int,
    streaming: bool,
    streamed: int | None = None,
) -> None:
    """Emit an OpenAI-shaped 413 JSON response from inside ASGI middleware.

    We hand-roll the response (rather than raise ``HTTPException``)
    because the FastAPI exception machinery needs the request body to
    be drained — exactly what we're trying to avoid. The envelope
    matches the OpenAI error shape used by the rest of the codebase
    (see ``vllm_mlx/server.py::_http_exception_handler``):

      {"error": {"message": ..., "type": "invalid_request_error",
                 "code": "request_too_large", "param": null}}
    """
    message = _format_message(
        advertised=advertised,
        limit=limit,
        streaming=streaming,
        streamed=streamed,
    )
    body = _json.dumps(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "request_too_large",
                "param": None,
            }
        }
    ).encode("utf-8")
    try:
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})
    except Exception:
        # Best-effort: if the connection is already torn down, log and
        # move on. Re-raising here would mask the real cap-trip cause
        # in the request log.
        logger.debug("body-size 413 send failed (client already disconnected)")


def install_request_body_limit_middleware(app: Any) -> None:
    """Attach :class:`RequestBodyLimitMiddleware` to ``app``.

    Centralised for the same reason the audio variant has its own
    install helper — keeps the wiring discoverable from this module
    rather than buried in app-construction code, and gives tests a
    single hook to call when standing up a minimal app fixture."""
    app.add_middleware(RequestBodyLimitMiddleware)
