# SPDX-License-Identifier: Apache-2.0
"""Event constructor helpers — the only API call sites should use.

Phase 2 instrumentation calls ``emit.session_start(...)``, etc. These
helpers own the four things every call site needs to get right:

1. **Consent gate.** Every emit calls ``is_enabled()`` first. If the
   user has not opted in, the helper returns immediately and constructs
   no payload at all (so a future field addition cannot leak by mistake
   from a path that was supposed to be dark).

2. **Redaction.** Every user-provided string is funnelled through the
   primitives in ``redact.py``. Sites must not construct payloads by
   hand — they must call these helpers.

3. **Process-singleton queue.** Sites do not pass the queue around.
   They call ``get_queue()`` which lazily constructs + starts one. Tests
   reset via ``_reset_for_tests()``.

4. **Failure suppression.** Every helper wraps the body in a broad
   except so a telemetry bug cannot ever crash a user command. The
   only exception we let surface is ``KeyboardInterrupt`` (user
   intent) and ``SystemExit`` (programmatic intent).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from vllm_mlx import __version__ as _rapid_mlx_version  # noqa: N811
from vllm_mlx.telemetry.queue import TelemetryQueue
from vllm_mlx.telemetry.redact import (
    bucket_tokens,
    bucket_tps,
    bucket_ttft_ms,
    fingerprint_traceback,
    hash_flag_names,
    normalize_model_path,
    platform_info,
)
from vllm_mlx.telemetry.schema import SCHEMA_VERSION
from vllm_mlx.telemetry.state import get_or_create_client_id, is_enabled

# ---------------------------------------------------------------- singleton

_queue_lock = threading.Lock()
_queue: TelemetryQueue | None = None
_session_id: str | None = None


def get_queue() -> TelemetryQueue:
    """Return the process-singleton queue, constructing it on first use.

    Idempotent across threads. The flush daemon is started here, not
    in ``__init__``, so an import alone never spawns a background
    thread — only an actual emit does.
    """
    global _queue
    if _queue is not None:
        return _queue
    with _queue_lock:
        if _queue is None:
            _queue = TelemetryQueue()
            _queue.start()
    return _queue


def session_id() -> str:
    """Return a per-process random session id (created lazily, stable).

    Lives in this module rather than in ``state.py`` because it is
    process-scoped (not persisted) — the on-disk client_id is the only
    stable identity. Reset by ``_reset_for_tests`` so each test gets a
    fresh id.
    """
    global _session_id
    if _session_id is None:
        # uuid4 imported lazily so importing emit at module-scan time
        # does not touch /dev/urandom in CI fast-paths.
        import uuid

        _session_id = str(uuid.uuid4())
    return _session_id


def _reset_for_tests() -> None:
    """Wipe the singleton + session_id. Call from a test ``setUp``.

    Not part of the public API — tests-only seam. The queue's shutdown
    is best-effort; we do not block on the join.
    """
    global _queue, _session_id
    with _queue_lock:
        q = _queue
        _queue = None
        _session_id = None
    if q is not None:
        q.shutdown(timeout=0.1)


# ---------------------------------------------------------------- envelope


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _envelope(event: str) -> dict[str, Any]:
    """Build the per-emit common header.

    All four event types share this exact header — the optional
    payload field (session / request / error) is added by each emit
    helper below.
    """
    info = platform_info()
    return {
        "schema_version": SCHEMA_VERSION,
        "client_id": get_or_create_client_id(),
        "session_id": session_id(),
        "rapid_mlx_version": _rapid_mlx_version,
        "event": event,
        "timestamp": _utc_now_iso(),
        "platform": {
            "os": info["os"],
            "os_version": info["os_version"],
            "arch": info["arch"],
            "chip": info["chip"],
            "memory_gb": info["memory_gb"],
            "python_version": info["python_version"],
        },
    }


def _safe(fn: Any) -> Any:
    """Decorator: swallow any non-system exception from telemetry emit.

    Defined inline (not in ``utils``) so the suppression is visible at
    every emit site during code review. KeyboardInterrupt / SystemExit
    are intentionally NOT caught — user / programmatic intent always
    wins.

    ``functools.wraps`` is load-bearing for the signature-pin test in
    ``tests/test_telemetry_emit.py``: without it, ``inspect.signature``
    sees the bare ``(*args, **kwargs)`` of the wrapper and the test
    is silently void. Round 1 codex review caught this.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            fn(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return

    return wrapper


# ---------------------------------------------------------------- public API


@_safe
def session_start(
    *,
    subcommand: str,
    argv: list[str] | None = None,
    engine: str = "",
    models_loaded: list[str] | tuple[str, ...] = (),
) -> None:
    """Emit a ``session_start`` payload.

    Called by ``cli.py`` after argparse, before subcommand dispatch.
    ``argv`` carries the raw command-line tokens for flag-name
    redaction; values are never read.
    """
    if not is_enabled():
        return
    payload = _envelope("session_start")
    payload["session"] = {
        "subcommand": subcommand,
        "engine": engine,
        "flag_names": hash_flag_names(argv) if argv is not None else [],
        "models_loaded": [normalize_model_path(m) for m in models_loaded][:32],
    }
    get_queue().enqueue(payload)


@_safe
def session_end(
    *,
    subcommand: str,
    duration_seconds: int,
    engine: str = "",
    models_loaded: list[str] | tuple[str, ...] = (),
) -> None:
    """Emit a ``session_end`` payload.

    Called by an ``atexit`` handler registered in ``cli.py`` main. The
    duration is wall-clock from process start; subcommand is captured
    at start so atexit does not need to re-inspect argv.
    """
    if not is_enabled():
        return
    payload = _envelope("session_end")
    payload["session"] = {
        "subcommand": subcommand,
        "duration_seconds": int(max(0, duration_seconds)),
        "engine": engine,
        "flag_names": [],
        "models_loaded": [normalize_model_path(m) for m in models_loaded][:32],
    }
    get_queue().enqueue(payload)


# Allowlist of route identifiers ``request()`` accepts. Round 2 codex
# review caught that storing ``endpoint`` verbatim was a free-form
# escape hatch — a future caller threading a path with a query string
# (``/v1/chat?key=sk-...``) would have leaked it. Constrain to the
# small set of routes Phase 2.2 actually instruments; anything else
# collapses to ``"other"``.
_ALLOWED_ENDPOINTS: frozenset[str] = frozenset(
    {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/audio/transcriptions",
        "/v1/messages",
        "/v1/images/generations",
    }
)


def _normalize_endpoint(raw: str) -> str:
    """Map a caller-provided route to the small ``_ALLOWED_ENDPOINTS``
    set; everything else becomes ``"other"``.

    Strips query strings + fragments defensively in case a caller
    passes a full URL. The signature red-line test demands that no
    field on the payload carry caller-controlled free-form text — this
    is the enforcement.
    """
    if not isinstance(raw, str):
        return "other"
    path = raw.split("?", 1)[0].split("#", 1)[0]
    return path if path in _ALLOWED_ENDPOINTS else "other"


@_safe
def request(
    *,
    endpoint: str,
    model_alias: str,
    stream: bool,
    tool_call_used: bool,
    prompt_tokens: int,
    completion_tokens: int,
    ttft_ms: float,
    tps: float,
    status: int,
) -> None:
    """Emit a ``request`` payload (Phase 2.2 sites use this).

    Defined now so call sites can land without touching this module
    later. The Phase 2.0/2.1 PR does not call this — it ships dark.
    """
    if not is_enabled():
        return
    payload = _envelope("request")
    payload["request"] = {
        "endpoint": _normalize_endpoint(endpoint),
        "model_alias": normalize_model_path(model_alias),
        "stream": bool(stream),
        "tool_call_used": bool(tool_call_used),
        "prompt_tokens_bucket": bucket_tokens(prompt_tokens),
        "completion_tokens_bucket": bucket_tokens(completion_tokens),
        "ttft_ms_bucket": bucket_ttft_ms(ttft_ms),
        "tps_bucket": bucket_tps(tps),
        "status": int(status),
    }
    get_queue().enqueue(payload)


# Round 3 codex review: ``category`` + ``phase`` on ``error()`` were
# stored verbatim, the same free-form escape hatch the signature
# red-line test cannot catch (type is ``str``). A future caller
# threading exception text or user input would have leaked. The
# allowlists below are intentionally short — every NEW value requires
# editing this file, which puts a privacy review on the path.
_ALLOWED_ERROR_CATEGORIES: frozenset[str] = frozenset(
    {
        "model_load_failure",
        "parser_failure",
        "scheduler_failure",
        "request_failure",
        "lifespan_failure",
        "tool_call_failure",
    }
)
_ALLOWED_ERROR_PHASES: frozenset[str] = frozenset(
    {
        "startup",
        "warmup",
        "prefill",
        "decode",
        "stream",
        "shutdown",
        "chat",
        "serve",
    }
)


def _normalize_error_field(raw: str, allowed: frozenset[str]) -> str:
    if not isinstance(raw, str):
        return "other"
    return raw if raw in allowed else "other"


@_safe
def error(
    *,
    category: str,
    exc: BaseException,
    phase: str,
) -> None:
    """Emit an ``error`` payload (Phase 2.2 sites use this).

    ``exc`` is fingerprinted with ``fingerprint_traceback`` — only
    ``basename:func:lineno`` of each frame plus the exception class
    name participate. No message text, no module path.

    ``category`` and ``phase`` are constrained to a small allowlist
    (see ``_ALLOWED_ERROR_CATEGORIES`` / ``_ALLOWED_ERROR_PHASES``);
    anything else collapses to ``"other"``.
    """
    if not is_enabled():
        return
    payload = _envelope("error")
    payload["error"] = {
        "category": _normalize_error_field(category, _ALLOWED_ERROR_CATEGORIES),
        "fingerprint": fingerprint_traceback(exc),
        "phase": _normalize_error_field(phase, _ALLOWED_ERROR_PHASES),
    }
    get_queue().enqueue(payload)
