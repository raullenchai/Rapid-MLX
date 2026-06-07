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

import itertools
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

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


# Allowlist of subcommands ``session_start``/``session_end`` accept.
# Round 11 codex review caught that ``subcommand`` was the last
# free-form ``str`` slot on the privacy-boundary helpers, the same
# shape of escape hatch closed for ``endpoint`` / ``category`` /
# ``phase``. The set matches the subcommands wired into ``cli.py``;
# anything off-list (including a future internal subcommand) collapses
# to ``"other"``.
_ALLOWED_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "serve",
        "chat",
        "agents",
        "bench",
        "doctor",
        "models",
        "info",
        "ps",
        "pull",
        "rm",
        "upgrade",
        "share",
        "version",
        # ``telemetry`` is excluded from lifecycle emit in cli.py, but
        # external callers using these helpers directly can still pass
        # it — keep it on-list so it doesn't get redacted needlessly.
        "telemetry",
    }
)


def _normalize_subcommand(raw: str) -> str:
    if not isinstance(raw, str):
        return "other"
    return raw if raw in _ALLOWED_SUBCOMMANDS else "other"


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

    Round 6 codex review caught that the prior lazy init was unlocked,
    so two concurrent emit sites racing past the ``is None`` check
    would generate different uuids in the same process and the
    aggregation pipeline would see two sessions per real session. The
    queue's ``_queue_lock`` is the same one that guards singleton
    creation; reuse it so we don't introduce a second lock for the
    same "first-call-init" pattern.
    """
    global _session_id
    if _session_id is not None:
        return _session_id
    with _queue_lock:
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

    Round 14 codex review: the broad ``except Exception`` used to also
    swallow ``TypeError`` from a call-site that drifted out of sync
    with the helper signature (renamed kwarg, missing required arg).
    That turned a wiring bug into a silent "no telemetry," which the
    integration tests can't see. ``inspect.signature(fn).bind(...)``
    is evaluated BEFORE the broad catch so any signature mismatch
    surfaces as a normal ``TypeError`` at call time. The signature is
    bound once at decoration time so the per-call cost is just a
    method invocation.
    """
    import functools
    import inspect

    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        # Validate the call-site BEFORE the suppression block. A
        # TypeError here means a wiring bug; we want it visible.
        sig.bind(*args, **kwargs)
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
    models_loaded: Iterable[str] = (),
) -> None:
    """Emit a ``session_start`` payload.

    Called by ``cli.py`` after argparse, before subcommand dispatch.
    ``argv`` carries the raw command-line tokens for flag-name
    redaction; values are never read.

    Round 4 codex review removed the ``engine`` parameter. It was
    declared with a default of ``""`` and never threaded through from
    any call site, leaving a free-form ``str`` slot in the payload
    that the signature red-line test couldn't catch — the same shape
    of escape hatch closed in earlier rounds for ``endpoint`` /
    ``category`` / ``phase``. There is effectively one engine today
    (``BatchedEngine``) since PR #156 deleted ``SimpleEngine``, so
    the field carried no information either. Re-add as an enum if a
    second engine ever lands.
    """
    if not is_enabled():
        return
    payload = _envelope("session_start")
    # Round 8 codex review: the runtime payload must include every key
    # the schema-v1 ``SessionPayload`` dataclass advertises, even
    # default-valued ones, so external consumers parsing v1 envelopes
    # can rely on the keys being present. ``duration_seconds`` is
    # ``None`` on session_start (defined for session_end only). The
    # ``engine`` slot stays for v1 back-compat with no runtime value.
    payload["session"] = {
        "subcommand": _normalize_subcommand(subcommand),
        "duration_seconds": None,
        "flag_names": hash_flag_names(argv) if argv is not None else [],
        "engine": "",
        # Slice before normalize (round 7 codex catch): if a caller
        # ever hands us 1000 paths, redacting all of them just to
        # throw 968 away wastes work for no extra privacy.
        # Round 13 codex review: ``tuple(models_loaded)[:32]`` would
        # still materialize the entire input before capping. Use
        # ``itertools.islice`` so the slice itself is the cap — callers
        # that hand us a 1000-entry iterable only pay for the first 32.
        "models_loaded": [
            normalize_model_path(m) for m in itertools.islice(models_loaded, 32)
        ],
    }
    get_queue().enqueue(payload)


@_safe
def session_end(
    *,
    subcommand: str,
    duration_seconds: int,
    models_loaded: Iterable[str] = (),
) -> None:
    """Emit a ``session_end`` payload.

    Called by an ``atexit`` handler registered in ``cli.py`` main. The
    duration is wall-clock from process start; subcommand is captured
    at start so atexit does not need to re-inspect argv.
    """
    if not is_enabled():
        return
    payload = _envelope("session_end")
    # Same schema-completeness rationale as ``session_start`` above.
    # ``engine`` slot for v1 back-compat, ``flag_names`` empty for
    # session_end (argv parsing happened at session_start).
    payload["session"] = {
        "subcommand": _normalize_subcommand(subcommand),
        "duration_seconds": int(max(0, duration_seconds)),
        "flag_names": [],
        "engine": "",
        # Same itertools.islice cap as session_start (round 13 catch).
        "models_loaded": [
            normalize_model_path(m) for m in itertools.islice(models_loaded, 32)
        ],
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

    Round 13 codex review: a naive ``split("?")`` left a full URL like
    ``"https://host/v1/chat/completions"`` unmatched and recorded as
    ``"other"``, which silently hid one of the very leaks the allowlist
    is supposed to prevent. ``urlsplit`` extracts the path regardless of
    whether the caller passed ``"/v1/chat/completions"``, the same with
    a query string, or a full URL.

    The signature red-line test demands that no field on the payload
    carry caller-controlled free-form text — this is the enforcement.
    """
    if not isinstance(raw, str):
        return "other"
    try:
        path = urlsplit(raw).path
    except ValueError:
        return "other"
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
# Aligned with the design doc's "WHEN" column in
# ``docs/plans/telemetry-golden-profile.md`` — every Phase 2.2 call
# site has a slot here. Round 5 codex review caught that ``oom`` was
# missing, which would have made the planned scheduler.py MemoryError
# handler silently collapse its category to ``"other"`` and lose the
# triage signal.
_ALLOWED_ERROR_CATEGORIES: frozenset[str] = frozenset(
    {
        "model_load_failure",
        "parser_failure",
        "scheduler_failure",
        "request_failure",
        "lifespan_failure",
        "tool_call_failure",
        "tool_parse",
        "oom",
        "shutdown_traceback",
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
        "request",
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
