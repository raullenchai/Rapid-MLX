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
"""

from __future__ import annotations

import json as _json
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

logger = logging.getLogger("rapid_mlx.exception_handlers")


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


def _validation_error_response(exc: RequestValidationError) -> JSONResponse:
    """Build the 400 envelope for Pydantic body-validation failures.

    Strips ``detail[*].input`` (F-094 / F-104 secret-bounce vector)
    and drops the pydantic.dev help URL (F-163). Surfaces only the
    location + message so clients still get an actionable hint.
    """
    details = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
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
