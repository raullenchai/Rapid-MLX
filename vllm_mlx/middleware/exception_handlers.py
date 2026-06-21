# SPDX-License-Identifier: Apache-2.0
"""Unified FastAPI exception handlers for rapid-mlx.

The shapes here are the single source of truth — both
:mod:`vllm_mlx.server` (production) and the route-level test apps under
``tests/`` call :func:`install_exception_handlers` to wire them in.

The module intentionally has **no heavy imports** (no ``.engine``, no
``mlx``) so isolated route tests can import it without pulling the
whole engine stack into the fixture.

Fixes covered:

* F-161 / F-162 — malformed JSON bodies on ``/v1/messages``,
  ``/v1/messages/count_tokens``, and ``/v1/responses`` were producing
  HTTP 500 because ``await request.json()`` raises
  :class:`json.JSONDecodeError` and the only catch-all was the global
  ``Exception`` handler.
* F-094 / F-104 mitigation — the default FastAPI 422 echoes the
  offending value verbatim in ``detail[*].input``. We collapse to a
  400 with no value echo and strip the pydantic.dev help URL (F-163).
* H-17 — ``/v1/messages`` and ``/v1/responses`` construct their
  Pydantic request models manually (``AnthropicRequest(**body)`` /
  ``ResponsesRequest(**body)``) instead of binding them as FastAPI
  body parameters, so the resulting :class:`pydantic.ValidationError`
  never reached ``RequestValidationError``. The previous per-route
  ``raise HTTPException(status_code=400, detail=str(e))`` patches
  echoed the full Pydantic message — leaking the model class name,
  the pinned pydantic version (``errors.pydantic.dev/2.13/...``),
  and attacker-controlled ``input_value`` blobs. The dedicated
  ``pydantic.ValidationError`` handler below routes both routes
  through the same sanitized 400 envelope used by ``/v1/chat/
  completions`` and ``/v1/completions``.
"""

from __future__ import annotations

import json as _json
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

logger = logging.getLogger("rapid_mlx.exception_handlers")


def _sanitize_loc(loc: tuple) -> str:
    """Collapse a Pydantic ``loc`` tuple to a safe dotted path.

    Drops the synthetic ``"body"`` prefix FastAPI prepends. Keeps
    positional indices (``int``) as-is — they come from list/sequence
    positions and the attacker can't inject arbitrary bytes there.
    Replaces **every** string component with the placeholder
    ``<field>`` so attacker-controlled bytes are never reflected:

    * Pydantic v2 puts dict keys (``dict[str, T]`` types), JSON-
      pointer-style indices, and (with ``ConfigDict(extra="forbid")``)
      the rejected extra-field NAME directly into ``loc`` —
      indistinguishable from the schema-owned field names at the
      string level. Codex H-17 round-2 BLOCKING #1 caught that any
      shape-based whitelist (e.g. identifier regex) still leaks
      identifier-shaped attacker bytes like ``AWS_SECRET_ACCESS_KEY``.
    * Schema-owned field names are public API surface anyway, so
      losing them in the envelope costs at most an informational
      hint; the validator message itself ("Field required", "Input
      should be a valid list", ...) already conveys the failure mode.

    Net effect: the 400 still carries actionable structure (e.g.
    ``<field>.0.<field>: Input should be a valid integer``) and the
    Pydantic error message, without echoing a single byte the caller
    chose.
    """
    parts: list[str] = []
    for raw in loc:
        if raw == "body":
            continue
        if isinstance(raw, int):
            parts.append(str(raw))
            continue
        # All string components — whether schema-owned or attacker-
        # supplied — collapse to a constant placeholder. The Pydantic
        # error MESSAGE ("Field required", etc.) is schema-determined
        # and is still surfaced separately, so we lose only the
        # field-name hint, never any byte the attacker can choose.
        parts.append("<field>")
    return ".".join(parts)


def _decode_error_response(exc: _json.JSONDecodeError) -> JSONResponse:
    """Build the 400 envelope for a malformed-JSON request body.

    The message includes the structural reason from ``exc.msg``
    (e.g. ``Expecting value``, ``Expecting property name``) so clients
    can fix the bug — these strings are short and stable across Python
    versions, no secret leakage risk.
    """
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": f"Invalid JSON in request body: {exc.msg}",
                "type": "invalid_request_error",
                "code": "invalid_json",
                "param": None,
            }
        },
    )


def _validation_error_response(
    exc: RequestValidationError | PydanticValidationError,
) -> JSONResponse:
    """Build the 400 envelope for Pydantic body-validation failures.

    Strips ``detail[*].input`` (F-094 / F-104 secret-bounce vector)
    and drops the pydantic.dev help URL (F-163). Surfaces only the
    location + message so clients still get an actionable hint.

    Accepts both the FastAPI-wrapped :class:`RequestValidationError`
    (raised when a Pydantic model is bound as a FastAPI body parameter)
    and the raw :class:`pydantic.ValidationError` (raised when a route
    constructs the request model manually — e.g. ``/v1/messages`` and
    ``/v1/responses``). Both expose the same ``.errors()`` shape, so a
    single sanitizer covers both code paths (H-17).

    The ``loc`` is run through :func:`_sanitize_loc` so attacker-
    controlled dict keys / extra-field names (codex H-17 round-2
    finding) collapse to ``<field>`` instead of being echoed verbatim
    in the 400 message.
    """
    details = []
    for err in exc.errors():
        loc = _sanitize_loc(tuple(err.get("loc", ())))
        msg = err.get("msg", "validation error")
        details.append(f"{loc}: {msg}" if loc else msg)
    summary = "; ".join(details) or "Invalid request body"
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": f"Invalid request body: {summary}",
                "type": "invalid_request_error",
                "code": "invalid_request",
                "param": None,
            }
        },
    )


_HTTP_ERROR_TYPE_MAP = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    405: "invalid_request_error",
    409: "conflict_error",
    429: "rate_limit_error",
}


def _http_error_response(exc: StarletteHTTPException) -> JSONResponse:
    """Build the OpenAI-shaped envelope for a Starlette ``HTTPException``.

    Routes can opt in to a fully custom envelope by raising
    ``HTTPException(detail={"error": {...}})`` — the structured detail
    is passed through unchanged. Bare-string detail is wrapped in the
    legacy envelope so existing callers keep working.
    """
    detail = exc.detail
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=detail,
            headers=getattr(exc, "headers", None),
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": _HTTP_ERROR_TYPE_MAP.get(exc.status_code, "api_error"),
                "code": None,
                "param": None,
            }
        },
        headers=getattr(exc, "headers", None),
    )


def _generic_error_response() -> JSONResponse:
    """The unmodified-secret 500 envelope used for unhandled errors.

    Exception message / traceback go to the log; the client sees a
    generic message so we don't leak filesystem paths, model paths, or
    environment values to a probing attacker.
    """
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error"}},
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Register the rapid-mlx exception handlers on ``app``.

    Wiring is idempotent — re-registering the same exception class
    just overwrites the previous binding (FastAPI / Starlette behaviour).
    Tests and production both call this exactly once.
    """

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(
        request: Request,  # noqa: ARG001
        exc: StarletteHTTPException,
    ):
        return _http_error_response(exc)

    @app.exception_handler(_json.JSONDecodeError)
    async def _decode_handler(
        request: Request,  # noqa: ARG001
        exc: _json.JSONDecodeError,
    ):
        return _decode_error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        request: Request,  # noqa: ARG001
        exc: RequestValidationError,
    ):
        return _validation_error_response(exc)

    @app.exception_handler(PydanticValidationError)
    async def _pydantic_validation_handler(
        request: Request,
        exc: PydanticValidationError,
    ):
        # H-17: routes that build a Pydantic model manually
        # (``AnthropicRequest(**body)`` on /v1/messages,
        # ``ResponsesRequest(**body)`` on /v1/responses, plus the
        # adapter-layer ``ChatCompletionRequest`` constructions inside
        # those routes) raise the raw ``pydantic.ValidationError``.
        # Route it through the same sanitized envelope as the FastAPI-
        # bound bodies so the model class name, the pinned pydantic
        # version (``errors.pydantic.dev/2.13/...``), and the attacker-
        # supplied ``input_value`` stay out of the response body.
        #
        # Codex H-17 round-2 NIT #4: a global handler converts every
        # internal Pydantic bug into a client 400, which can mask
        # server-side defects. Log at WARNING with sanitized metadata
        # so operators can spot "this 400 actually came from a server-
        # side response-model construction failure".
        #
        # Codex H-17 round-3 BLOCKING: do NOT pass the raw exception
        # (``exc_info=exc`` or ``str(exc)``) — its string form embeds
        # ``input_value=...`` which can carry attacker-supplied
        # secrets, and this PR is explicitly preventing that
        # reflection. Operator log lines must be sanitized too;
        # otherwise an attacker can stuff secrets into a body field
        # and pivot them into the operator's log pipeline. We surface
        # only the per-error ``type`` codes (``missing``,
        # ``int_parsing``, ``extra_forbidden``, …) and the sanitized
        # ``loc`` path — both schema-determined.
        sanitized = [
            {
                "type": err.get("type", "validation_error"),
                "loc": _sanitize_loc(tuple(err.get("loc", ()))),
            }
            for err in exc.errors()
        ]
        logger.warning(
            "pydantic.ValidationError on %s %s — %d sanitized error(s): %s",
            request.method,
            request.url.path,
            len(sanitized),
            sanitized,
        )
        return _validation_error_response(exc)

    @app.exception_handler(Exception)
    async def _generic_handler(request: Request, exc: Exception):
        # Re-route the specific subclasses in case a TaskGroup /
        # thread boundary dispatches them here instead of through the
        # dedicated handlers above (FastAPI/Starlette occasionally
        # falls back to the generic handler on cancellation paths).
        if isinstance(exc, _json.JSONDecodeError):
            return _decode_error_response(exc)
        if isinstance(exc, RequestValidationError):
            return _validation_error_response(exc)
        if isinstance(exc, PydanticValidationError):
            return _validation_error_response(exc)
        if isinstance(exc, StarletteHTTPException):
            return _http_error_response(exc)
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return _generic_error_response()
