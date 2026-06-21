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

# JSON-API paths whose clients are known to historically omit
# ``Content-Type`` (the OpenAI Python client < 0.27 did this; some
# bare-bones curl scripts still do). For these specific paths an
# absent header is treated as JSON for back-compat. Every OTHER
# guarded path requires an explicit ``application/json`` (or
# ``+json`` variant) so a future ``/v1/foo/upload`` that ships raw
# binary blobs without a ``Content-Type`` doesn't accidentally pay
# the JSON-parsing cost and risk a spurious ``request_body_too_deep``
# rejection (codex r3 NIT #3).
_JSON_CONTENT_TYPE_OPTIONAL_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/responses",
        "/anthropic/v1/messages",
    }
)


def _quick_depth_might_exceed(body: bytes, max_depth: int) -> bool:
    """Cheap byte-level upper bound on JSON nesting depth (codex r2 NIT).

    Walks the bytes once, tracking the simultaneous-open balance of
    ``{``/``[`` minus ``}``/``]`` OUTSIDE JSON string literals. The
    depth at any JSON node is bounded above by the simultaneous-open
    count, so a body whose balance never reaches ``max_depth + 1`` is
    GUARANTEED to be at or under the cap and the middleware can skip
    the full ``json.loads`` + iterative-walk pipeline.

    The function is intentionally one-sided: it returns ``True`` if
    the body MIGHT exceed the cap (caller MUST do the full check),
    ``False`` only when the body provably cannot. False positives
    fall back to the precise parse; a false negative would silently
    bypass the gate and reopen D-DEEP-JSON.

    codex r3 BLOCKING #1: an earlier version naively decremented on
    every ``}``/``]`` byte. That let an attacker prepend a string
    literal carrying ``]]]…]]]`` to drive the counter negative, hiding
    a genuinely deep JSON tree from the precise parser:

      ``{"x":"]]]]]]]]…]]]]]]]]","tree":<deep nest>}``

    The counter went negative on the string body, the peak of the
    structurally-deep portion was masked by the negative drift, and
    the heuristic incorrectly returned ``False``. Fix: track JSON
    string state so the close-bracket decrement only fires when we
    really are between structural tokens. ``\\`` inside a string
    escapes the next byte (covers ``\\"``, ``\\\\``, etc.); we don't
    interpret unicode escapes (``\\uXXXX``) because the JSON parser
    rejects malformed escapes downstream — the heuristic only needs
    to be correct on syntactically-valid JSON to be a safe upper
    bound, and on malformed JSON the precise parse takes over and
    bounces with 400.
    """
    cap = max_depth + 1  # depth > max_depth → fail
    depth = 0
    peak = 0
    in_string = False
    escape_next = False
    for byte in body:
        if escape_next:
            # The previous byte was a backslash inside a string —
            # consume this byte literally and clear the escape flag.
            escape_next = False
            continue
        if in_string:
            if byte == 0x5C:  # ``\\``
                escape_next = True
            elif byte == 0x22:  # ``"`` closes the string
                in_string = False
            # All other bytes inside a string (including ``[``,
            # ``]``, ``{``, ``}``) are inert.
            continue
        if byte == 0x22:  # ``"`` opens a string
            in_string = True
            continue
        if byte == 0x7B or byte == 0x5B:  # ``{`` or ``[``
            depth += 1
            if depth > peak:
                peak = depth
                if peak >= cap:
                    return True
        elif byte == 0x7D or byte == 0x5D:  # ``}`` or ``]``
            # Outside-string close: the structurally-correct
            # decrement. Inside-string closes are inert (handled
            # above), so an attacker cannot prepend stringified
            # closes to drive the counter negative and mask a deep
            # structural payload further along.
            depth -= 1
    return peak >= cap


def _path_is_guarded(path: str | None) -> bool:
    if not path:
        return False
    if path in _EXCLUDED_PATHS:
        return False
    return any(path.startswith(prefix) for prefix in _GUARDED_PREFIXES)


def _is_jsonish_content_type(headers, path: str | None) -> bool:
    """Return ``True`` iff the request advertises a JSON-shaped content
    type.

    Accepts ``application/json`` and ``application/*+json`` (vendor /
    draft profiles) unconditionally — those are unambiguous JSON
    signals.

    An empty / missing ``Content-Type`` is only accepted for the
    known-JSON paths listed in :data:`_JSON_CONTENT_TYPE_OPTIONAL_PATHS`
    (codex r3 NIT #3). The OpenAI Python client < 0.27 omitted the
    header on ``/v1/chat/completions``, and some curl scripts still
    do; we preserve that back-compat for the exact endpoints where
    a JSON body is the only legal shape. A future
    ``/v1/foo/upload`` that ships raw binary without a
    ``Content-Type`` is left alone so the depth gate never tries to
    parse it as JSON.

    Anything else (``multipart/form-data``, ``text/plain``,
    ``application/octet-stream``, …) is left alone so we don't try
    to JSON-parse a binary upload — same shape as the size cap.
    """
    ctype: str = ""
    for raw_name, raw_value in headers:
        if raw_name.lower() == b"content-type":
            try:
                ctype = raw_value.decode("latin-1").lower()
            except UnicodeDecodeError:
                ctype = ""
            break
    if not ctype:
        return path in _JSON_CONTENT_TYPE_OPTIONAL_PATHS
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
        if not _is_jsonish_content_type(scope.get("headers", ()), path):
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

        # Cheap byte-level heuristic (codex r2 NIT): the absolute
        # maximum structural depth of a JSON document is bounded above
        # by the number of consecutive opening brackets / braces in
        # the byte stream — every container opens with ``{`` or ``[``
        # and there's at most one open per nested level, so a body
        # that never accumulates ``max_depth`` simultaneous opens
        # CANNOT exceed the cap and we can skip the full parse + walk.
        # This zips through the bytes once with two integer counters,
        # which is dramatically cheaper than the ``json.loads`` →
        # tree-walk pipeline on the production hot path of normal-
        # depth payloads (chat/completion bodies sit at depth 3–5).
        # The heuristic over-counts when the bytes appear inside a
        # string literal (e.g. ``"a{{{"``) but only in the direction
        # that triggers a fallback to the full parse, never in the
        # direction that bypasses the gate.
        if not _quick_depth_might_exceed(body, max_depth):
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
        except RecursionError:
            # codex r1 BLOCKING #2: ``json.loads`` is implemented in
            # C and uses an internal recursion bound that, on
            # sufficiently extreme nesting (well past 1000 levels),
            # can itself raise ``RecursionError`` BEFORE
            # :func:`json_nesting_depth_exceeds` runs. Pre-fix, the
            # ``except _json.JSONDecodeError`` arm let the
            # ``RecursionError`` propagate and the request relied on
            # the global :class:`RecursionError` handler — which still
            # returns the same canonical envelope, but it means the
            # operator log records the trace at WARNING for every
            # such request instead of the body-depth gate's quiet
            # path. Catching the parser-level recursion here surfaces
            # it as the configured-cap rejection so the trace stays
            # off the WARNING log and the cap-naming message reaches
            # the client.
            await _send_400_depth(send, max_depth=max_depth)
            return

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

    codex r3 BLOCKING #2: every replayed chunk MUST carry
    ``more_body=True``. Pre-fix the LAST buffered chunk was emitted
    with ``more_body=False``, which signals "body fully on the wire"
    to the downstream app — a truncated upload would then be
    indistinguishable from a complete one to any handler that polls
    on ``more_body``. Post-fix the disconnect is the ONLY terminal
    event, exactly matching the upstream sequence the client
    actually shipped (partial chunks followed by an
    ``http.disconnect``), so a downstream that buffers via
    ``await request.body()`` sees the same Starlette
    ``ClientDisconnect`` it would have without the middleware in the
    path.
    """
    idx = {"value": 0}

    async def receive():
        i = idx["value"]
        if i < len(chunks):
            idx["value"] = i + 1
            # ALL replayed chunks carry ``more_body=True``; the
            # original sequence terminated with ``http.disconnect``,
            # not a final ``more_body=False`` frame, so the downstream
            # app must see the same disconnect-as-terminator shape.
            return {
                "type": "http.request",
                "body": chunks[i],
                "more_body": True,
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
