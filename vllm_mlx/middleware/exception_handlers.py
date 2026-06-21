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


# D-ANTHRO-VALIDATION F1 — closed allowlist of schema-owned field names
# the loc sanitizer is allowed to ECHO instead of collapsing to
# ``<field>``. Pre-fix, every string ``loc`` component (even safe
# schema-owned ones like ``temperature`` / ``messages``) collapsed to
# the placeholder, producing a user-facing 400 like
# ``<field>: Field required`` that names nothing actionable. Sergei's
# F1 dogfood (Anthropic /v1/messages with ``temperature="hot"``) shows
# the leak: ``message: "Invalid request body: <field>: Input should
# be a valid number, ..."``.
#
# The H-17 round-2 finding (codex BLOCKING #1) is preserved by keeping
# the default-deny: only names that appear in this allowlist are
# echoed; everything else (attacker-supplied dict keys, extra-forbid
# field names, identifiers we don't recognize) still collapses to
# ``<field>``. The set is built from the public request-model surfaces
# (AnthropicRequest, ChatCompletionRequest, CompletionRequest,
# ResponsesRequest) plus the nested content-block models — every name
# below is schema-determined public-API surface, so echoing it leaks
# no attacker bytes.
#
# IMPORTANT: this list is a *closed* allowlist — adding a new field to
# a request model also requires adding it here if you want the
# validation error to name it. The default-deny means a forgotten
# entry just produces a less-informative ``<field>`` placeholder; it
# never opens a leak vector. The H-17 round-2 attacker shapes
# (``AWS_SECRET_ACCESS_KEY``, ``X-Forwarded-For``, ``../../etc/passwd``,
# 256-char identifiers) are NOT in this set and remain collapsed.
_SCHEMA_OWNED_FIELD_NAMES: frozenset[str] = frozenset(
    {
        # AnthropicRequest
        "max_tokens",
        "messages",
        "metadata",
        "model",
        "output_config",
        "stop_sequences",
        "stream",
        "system",
        "temperature",
        "thinking",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
        # ChatCompletionRequest (additional)
        "chat_template_kwargs",
        "enable_thinking",
        "frequency_penalty",
        "function_call",
        "functions",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "min_p",
        "n",
        "parallel_tool_calls",
        "presence_penalty",
        "reasoning_max_tokens",
        "repetition_penalty",
        "response_format",
        "seed",
        "stop",
        "stream_options",
        "timeout",
        "top_logprobs",
        "video_fps",
        "video_max_frames",
        # CompletionRequest (additional)
        "best_of",
        "echo",
        "prompt",
        "suffix",
        # ResponsesRequest (additional)
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "previous_response_id",
        "prompt_cache_key",
        "reasoning",
        "service_tier",
        "store",
        "text",
        # Nested Anthropic content-block / message fields
        # (``text``, ``input`` already declared at top-level above)
        "role",
        "content",
        "type",
        "source",
        "id",
        "name",
        "tool_use_id",
        "is_error",
        # Nested OpenAI Message / ContentPart fields
        "tool_call_id",
        "tool_calls",
        "audio_url",
        "image_url",
        "video",
        "video_url",
        # Nested output_config / reasoning / text.format fields
        "format",
        "effort",
        "schema",
        "description",
        "strict",
        "budget_tokens",
        # tool_choice nested
        "function",
        # Nested image source
        "media_type",
        "data",
        "url",
        # tool definitions
        "input_schema",
        "parameters",
    }
)


def _is_union_arm_discriminator(raw: str) -> bool:
    """Identify Pydantic v2 union-arm loc components.

    On a union field, Pydantic v2 appends the failing arm's *type
    descriptor* to ``loc`` so the validator can disambiguate which arm
    rejected the input. The descriptors are not user-controlled bytes
    (they're synthesised from the schema) but they're noisy — bare
    primitive names like ``"str"``, ``"int"``, ``"bool"``, ``"dict"``,
    or wrapped names like ``"list[function-after[...]]"`` /
    ``"nullable[...]"``. Filter them out entirely so the surfaced
    ``loc`` path stays readable: instead of
    ``messages.0.content.<field>: Input should be a valid string`` the
    user sees ``messages.0.content: Input should be a valid string``.

    The set covers Pydantic's primitive-arm names AND the bracketed
    composite-arm shapes (detected by structural prefix because the
    inner schema text varies per model).
    """
    if raw in {"str", "int", "float", "bool", "dict", "list", "bytes", "tuple"}:
        return True
    # Composite-arm shapes: ``list[...]``, ``dict[...]``, ``tuple[...]``,
    # ``nullable[...]``, ``function-after[...]``, ``function-before[...]``,
    # ``union[...]``. The bracket is the structural tell.
    if "[" in raw and raw.endswith("]"):
        return True
    return False


def _sanitize_loc(loc: tuple) -> str:
    """Collapse a Pydantic ``loc`` tuple to a safe dotted path.

    Drops the synthetic ``"body"`` prefix FastAPI prepends. Keeps
    positional indices (``int``) as-is — they come from list/sequence
    positions and the attacker can't inject arbitrary bytes there.
    Drops union-arm discriminator components (see
    :func:`_is_union_arm_discriminator`) so the user-visible path
    matches the request body shape, not Pydantic's internal dispatch
    metadata.

    For string components, applies a *closed allowlist* of schema-owned
    field names (see :data:`_SCHEMA_OWNED_FIELD_NAMES`):

    * Names in the allowlist are echoed verbatim — they're public API
      surface (declared on the request Pydantic models) and naming
      them in the error message gives clients an actionable hint.
    * Names NOT in the allowlist collapse to ``<field>`` so attacker-
      controlled bytes (dict keys on ``dict[str, T]`` fields,
      extra-forbid field names) are never reflected. This is the H-17
      round-2 safety contract preserved unchanged for unknown names.

    Pre-D-ANTHRO-VALIDATION (F1), EVERY string collapsed — so
    ``temperature="hot"`` produced ``<field>: Input should be a valid
    number`` and the client had no idea which field broke. The closed
    allowlist closes that informational gap without re-opening the
    H-17 leak vector (any name an attacker could choose is NOT in the
    allowlist and still collapses).
    """
    parts: list[str] = []
    for raw in loc:
        if raw == "body":
            continue
        if isinstance(raw, int):
            parts.append(str(raw))
            continue
        # Drop Pydantic union-arm discriminator metadata entirely —
        # it's noisy and not a real path the user can act on.
        if isinstance(raw, str) and _is_union_arm_discriminator(raw):
            continue
        # String component — echo if it's a known schema-owned field
        # name; otherwise collapse to the H-17 placeholder.
        if isinstance(raw, str) and raw in _SCHEMA_OWNED_FIELD_NAMES:
            parts.append(raw)
        else:
            parts.append("<field>")
    return ".".join(parts)


# D-ANTHRO-VALIDATION F1 — Anthropic /v1/messages routes use a
# different top-level error envelope than the OpenAI surfaces:
# ``{"type":"error","error":{...}}`` (with an explicit ``type`` key on
# the outer object) versus ``{"error":{...}}``. The Anthropic SDK
# routes errors by ``response.type == "error"``; without the wrapper a
# 400 looks like an unstructured response and the SDK falls back to a
# generic ``APIStatusError`` with no typed Anthropic error class.
#
# Detect Anthropic surfaces by request path so a single set of
# handlers covers every error path (validation, HTTPException, JSON
# decode, recursion, generic 500). Path matching is strict — an exact
# match on the root path OR a strict sub-path match. Codex round-1
# NIT: a bare ``startswith("/v1/messages")`` would also classify
# unrelated paths like ``/v1/messages-foo`` or ``/v1/messagesevil`` as
# Anthropic surfaces, so an attacker who can probe arbitrary paths
# would receive the Anthropic envelope on 404/405s. The explicit
# ``path == ROOT or path.startswith(ROOT + "/")`` shape rejects those
# while still covering the legitimate ``/v1/messages/count_tokens``
# sub-route.
_ANTHROPIC_ROOT_PATHS: tuple[str, ...] = ("/v1/messages",)


def _is_anthropic_path(request: Request | None) -> bool:
    """Return True if ``request`` targets an Anthropic-compat route."""
    if request is None:
        return False
    try:
        path = request.url.path
    except Exception:
        return False
    for root in _ANTHROPIC_ROOT_PATHS:
        if path == root or path.startswith(root + "/"):
            return True
    return False


def _wrap_for_anthropic(response: JSONResponse) -> JSONResponse:
    """Rewrap an OpenAI-shaped error envelope to the Anthropic shape.

    Input:  ``{"error":{...}}``
    Output: ``{"type":"error","error":{...}}``

    Idempotent: if the body already carries a top-level ``type=="error"``
    (e.g. a route opted into the Anthropic envelope explicitly via
    ``HTTPException(detail={"type":"error","error":{...}})``), the
    response is returned unchanged. Non-error bodies are also returned
    unchanged so non-error JSON responses can't accidentally pick up an
    ``error`` field.

    Headers from the original response are preserved EXCEPT for
    ``content-length`` / ``content-type`` (Starlette recomputes these
    when constructing the new ``JSONResponse`` — copying them verbatim
    would emit an ``h11._util.LocalProtocolError: Too much data for
    declared Content-Length`` when the wrapped body is longer than the
    original).
    """
    raw = getattr(response, "body", None)
    if raw is None:
        return response
    try:
        body = _json.loads(raw)
    except (_json.JSONDecodeError, TypeError):
        return response
    if not isinstance(body, dict):
        return response
    if body.get("type") == "error" and isinstance(body.get("error"), dict):
        return response
    if "error" not in body:
        return response
    wrapped = {"type": "error", "error": body["error"]}
    # Preserve any non-error sibling keys (none expected today but
    # forward-compatible) by surfacing them at the top level.
    for k, v in body.items():
        if k == "error":
            continue
        wrapped[k] = v
    # Carry over auth / rate-limit headers but skip length+type which
    # Starlette regenerates from the new body. h11's protocol checker
    # rejects a stale Content-Length when the wrapped body is longer
    # than the original (D-ANTHRO-VALIDATION first-pass repro).
    preserved_headers: dict[str, str] | None = None
    if response.headers:
        preserved_headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in ("content-length", "content-type")
        }
        if not preserved_headers:
            preserved_headers = None
    return JSONResponse(
        status_code=response.status_code,
        content=wrapped,
        headers=preserved_headers,
    )


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


def _recursion_error_response() -> JSONResponse:
    """The sanitized envelope used when a ``RecursionError`` reaches
    the framework boundary (D-TOOL-RECUR / D-DEEP-JSON defense-in-depth).

    The primary defense for both bugs is structural — an iterative
    chat-template walk (see
    :func:`vllm_mlx.utils.chat_template._walk_tools_iter`) plus a body-
    depth guard middleware (see
    :mod:`vllm_mlx.middleware.body_depth`) plus a per-tool-schema
    depth validator (see :class:`vllm_mlx.api.models.ToolDefinition`).
    None of those should let a ``RecursionError`` propagate. But:

    * The body-depth gate path-scopes to JSON content types only,
      so a future route that accepts JSON via a different content
      type could bypass it.
    * The per-tool-schema validator runs at request-model construction,
      so a code path that builds a ``ChatCompletionRequest`` from a
      programmatically-constructed dict (engine tests, internal
      adapters) skips it.
    * Pydantic / FastAPI / Starlette internals may add new recursive
      paths in a future release that we haven't audited.

    Surfacing a ``RecursionError`` as HTTP 500 with a stack trace
    fragment (the pre-fix shape) is both a DoS signal AND an info-leak
    — the pre-fix trace named ``_sanitize_tools_for_template._walk``
    on every parser, so an attacker could identify the function and
    the line. This handler returns the SAME shape as
    :func:`_generic_error_response` (HTTP 500 ``Internal server
    error``) so:

    * No stack trace ever reaches the client (info-leak closed).
    * We DON'T claim "request body too deep" when the cause might
      actually be an unrelated recursion bug elsewhere in the server
      (codex r4 BLOCKING — a misleading 400 on a server-side bug
      would mask the real failure mode and the client would retry).
      The user-facing message stays neutral so an SDK keying on
      ``error.message == "Internal server error"`` handles it the
      same as any other unhandled server-side fault.
    * The body-depth gate middleware still emits its own
      ``request_body_too_deep`` 400 from the depth-cap rejection
      path — clients DO see that more-actionable error when the
      cause was actually a deep body.

    We log the trace at WARNING level so an operator can spot a new
    recursion site we should put a structural fix on, regardless of
    whether the cause was body-depth-related or somewhere else.
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
        request: Request,
        exc: StarletteHTTPException,
    ):
        response = _http_error_response(exc)
        if _is_anthropic_path(request):
            response = _wrap_for_anthropic(response)
        return response

    @app.exception_handler(_json.JSONDecodeError)
    async def _decode_handler(
        request: Request,
        exc: _json.JSONDecodeError,
    ):
        response = _decode_error_response(exc)
        if _is_anthropic_path(request):
            response = _wrap_for_anthropic(response)
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        response = _validation_error_response(exc)
        if _is_anthropic_path(request):
            response = _wrap_for_anthropic(response)
        return response

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
        response = _validation_error_response(exc)
        if _is_anthropic_path(request):
            response = _wrap_for_anthropic(response)
        return response

    @app.exception_handler(RecursionError)
    async def _recursion_handler(
        request: Request,
        exc: RecursionError,  # noqa: ARG001
    ):
        # D-TOOL-RECUR / D-DEEP-JSON defense-in-depth — see
        # :func:`_recursion_error_response`. Log the trace at WARNING
        # so an operator can spot the new recursion site and add a
        # structural fix (the iterative walk + the depth guards are
        # the primary defenses; this handler should be unreachable in
        # production). Path + method only — no request-body bytes are
        # logged, mirroring the H-17 round-3 sanitization rule.
        logger.warning(
            "RecursionError on %s %s — caught at framework boundary, "
            "returning sanitized 500 (Internal server error). Add a "
            "structural fix (iterative walk / depth guard) for the new "
            "recursion site.",
            request.method,
            request.url.path,
            exc_info=True,
        )
        response = _recursion_error_response()
        if _is_anthropic_path(request):
            response = _wrap_for_anthropic(response)
        return response

    @app.exception_handler(Exception)
    async def _generic_handler(request: Request, exc: Exception):
        # Re-route the specific subclasses in case a TaskGroup /
        # thread boundary dispatches them here instead of through the
        # dedicated handlers above (FastAPI/Starlette occasionally
        # falls back to the generic handler on cancellation paths).
        anthropic = _is_anthropic_path(request)
        if isinstance(exc, _json.JSONDecodeError):
            response = _decode_error_response(exc)
            return _wrap_for_anthropic(response) if anthropic else response
        if isinstance(exc, RequestValidationError):
            response = _validation_error_response(exc)
            return _wrap_for_anthropic(response) if anthropic else response
        if isinstance(exc, PydanticValidationError):
            response = _validation_error_response(exc)
            return _wrap_for_anthropic(response) if anthropic else response
        if isinstance(exc, StarletteHTTPException):
            response = _http_error_response(exc)
            return _wrap_for_anthropic(response) if anthropic else response
        if isinstance(exc, RecursionError):
            # ``isinstance(RecursionError) before isinstance(Exception)``:
            # the dedicated handler above SHOULD catch this first, but
            # FastAPI's fallback chain occasionally lands here (same
            # rationale as the other ``isinstance`` rerouting above).
            logger.warning(
                "RecursionError on %s %s (via generic handler) — "
                "returning sanitized 500 (Internal server error).",
                request.method,
                request.url.path,
                exc_info=True,
            )
            response = _recursion_error_response()
            return _wrap_for_anthropic(response) if anthropic else response
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        response = _generic_error_response()
        return _wrap_for_anthropic(response) if anthropic else response
