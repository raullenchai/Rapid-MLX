# SPDX-License-Identifier: Apache-2.0
"""DFlash server — dedicated single-user mode that bypasses BatchedEngine.

When DFlash is enabled, the CLI launches this server instead of the
standard ``vllm_mlx.server.app``. It hosts a minimal OpenAI-compatible
surface (``/healthz``, ``/v1/models``, ``/v1/chat/completions``) and routes
generation through mlx-vlm's ``stream_generate`` with the loaded DFlash
drafter.

Why a separate server (not a fork of the standard route)?
  - mlx-vlm's ``generate_step`` is a per-request Python generator with its
    own ``prompt_cache`` argument. BatchedEngine merges per-request KV
    caches into a ``BatchKVCache``. Grafting one onto the other would
    invent batched-DFlash that doesn't exist upstream and would risk
    regressing the non-DFlash path under attention layout changes.
  - DFlash today only validates on B=1 anyway (see PoC: 1.83-2.18× on
    Qwen3.5-27B-8bit; no batched-DFlash kernel exists in mlx-vlm 0.5.0).
  - A separate, opt-in server is a clean blast-radius boundary: turning
    on DFlash can never break a request that doesn't use it.

Current limitations (documented in README + ``rapid-mlx info``):
  - Single-user serial. Concurrent requests queue on an ``asyncio.Lock``.
  - No MCP, embeddings, or audio in this server (the standard server
    handles those).
  - No prefix cache (per-request KV cache built fresh each call).

These limitations are deliberate for v1 — the target user is someone
running ``rapid-mlx serve qwen3.5-27b-8bit --speculative-config
'{"method":"dflash"}'`` to get a ~2× speedup on code/long-form
completions on a single Apple Silicon box.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ModelsResponse,
    Usage,
)
from vllm_mlx.config import get_config
from vllm_mlx.engine import GenerationOutput
from vllm_mlx.service.helpers import (
    _parse_tool_calls_with_parser,
    _validate_tool_call_params,
)
from vllm_mlx.service.postprocessor import StreamingPostProcessor

from .eligibility import have_runtime
from .runtime import DFlashRuntime, load_runtime

logger = logging.getLogger(__name__)


# Global serial lock — DFlash is single-stream by design (mlx-vlm doesn't
# expose a batched DFlash kernel in 0.5.0). The second concurrent request
# waits its turn; this matches the PoC reality.
_dflash_lock = asyncio.Lock()


# Dedicated single-thread executor so every mlx-vlm call (drafter loading,
# generate, stream_generate's ``next``) executes on ONE thread for the
# lifetime of the process. Reason: mlx-lm 0.31.3+ keeps the GPU Stream
# in thread-local storage; iterating a generator across threads (which
# would happen if we used the default ThreadPoolExecutor with N workers)
# trips "There is no Stream(gpu, N) in current thread" mid-stream. Pinning
# to one worker preserves thread affinity and matches the serial-only
# contract enforced by ``_dflash_lock``.
_dflash_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="dflash-worker"
)

# Separate executor for prompt rendering (codex round-4 #4 → round-5 #1/#2).
# ``_render_prompt`` applies the chat template + tokenizes into a *string* —
# pure CPU tokenizer work that never creates mlx GPU arrays, so it does NOT
# need the thread-local GPU affinity ``_dflash_executor`` enforces. Keeping it
# OFF the single GPU worker is what makes it safe to offload: a render must
# never run between the token steps of a lock-owning generation (that would
# steal its deadline and touch its model/processor mid-generation). This pool
# is single-worker too, so renders serialize among themselves — cheap, and it
# sidesteps any tokenizer thread-safety question — while remaining fully
# independent of ``_dflash_lock`` and GPU generation.
#
# codex round-8 #5: an uncancellable render that overruns its deadline does NOT
# free its admission slot immediately. The endpoint retains the slot (claim +
# defer) until the render's ``concurrent.futures.Future`` actually completes
# and releases from its callback (codex round-7 #2). Holding the slot is what
# keeps ``max_concurrent_requests`` honest: a burst of cancelled/timed-out
# requests each keeps its render counted until it drains, so they cannot pile
# up unbounded on this pool's queue. (The slot is held only for accounting —
# the render still touches no GPU/serial resource, so nothing blocks the GPU.)
_dflash_render_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="dflash-render"
)


# F2 (codex round-2 #1): bound the producer→consumer SSE handoff queue so a
# stalled client cannot make the producer buffer an entire completion in
# memory. Once the queue is full, the producer waits at most
# ``_STREAM_BACKPRESSURE_TIMEOUT_SECONDS`` for the consumer to drain a slot;
# if the client is truly gone, generation is aborted and the generator +
# lease are cleaned up. The bound is chunk-count, not bytes — each queued
# item is one small SSE frame, so a modest count keeps buffered memory tiny
# without throttling a healthy fast reader.
#
# codex round-4 #3: ``_STREAM_BACKPRESSURE_TIMEOUT_SECONDS`` is an EXPLICIT,
# documented server-side backpressure cap, INDEPENDENT of the per-request
# ``timeout``. It applies even when ``timeout=0`` ("no request deadline"),
# because a client that has stopped reading must not be able to pin an
# unbounded in-memory completion (or the sole GPU lock) indefinitely — that
# is itself a denial-of-service vector, deadline or not. Crucially, hitting
# this cap no longer truncates the stream silently: the producer converts the
# backpressure abort into a terminal ``finish_reason="length"`` error frame +
# ``[DONE]`` (see ``_stream_completion``'s content-emit handler), so a client
# that is merely slow still receives an explicit explanation of why the
# stream ended. Only a client that has genuinely departed misses it — and
# that terminal ``_emit`` is itself swallowed rather than hanging.
_STREAM_QUEUE_MAXSIZE = 64
_STREAM_BACKPRESSURE_TIMEOUT_SECONDS = 30.0

# codex round-6 #2: grace window for a synchronous ``generator.close()`` before
# the lease stops awaiting it and detaches the close (retaining the lock/slot
# until the close future finishes). A normal close returns in microseconds, so
# this only ever triggers for a genuinely hung teardown — and even a tiny value
# would keep the fast path synchronous; 5s just avoids detaching on a briefly
# slow (but not hung) GPU cleanup under load.
_STREAM_GENERATOR_CLOSE_GRACE_SECONDS = 5.0


def _format_timeout_seconds(seconds: float) -> str:
    """Human-facing rendering of a timeout in seconds (codex round-6 #5).

    A fixed ``:.1f`` renders a valid sub-100 ms limit like ``0.01`` as
    "0.0 seconds", a misleading operational message. Use adaptive precision so
    small positive timeouts keep enough significant digits to be meaningful,
    while typical multi-second values stay clean.
    """
    if seconds <= 0:
        return "0 seconds"
    if seconds < 0.001:
        # codex round-7 #3: below the 3-decimal resolution ``:.3f`` would print
        # "0.000 seconds" — the very misleading zero this formatter exists to
        # avoid. Report a floor instead so the message stays truthful.
        return "<0.001 seconds"
    if seconds < 0.1:
        # e.g. 0.01 → "0.010", 0.005 → "0.005"
        return f"{seconds:.3f} seconds"
    if seconds < 1:
        return f"{seconds:.2f} seconds"
    return f"{seconds:.1f} seconds"


class _DFlashClientGoneError(Exception):
    """Raised inside the stream producer when a CONTENT-frame ``put`` blocks
    on a full SSE handoff queue past ``_STREAM_BACKPRESSURE_TIMEOUT_SECONDS``
    — i.e. the client has stopped reading. Signals the producer to abort
    generation and let the lease close the generator + release the GPU lock /
    admission slot, rather than buffering an unbounded completion for a client
    that is almost certainly gone. (Terminal frames use a longer/independent
    bound and never raise this — see ``_emit``.)"""


class _DFlashStreamDeadlineError(Exception):
    """Raised inside the stream producer when the REQUEST DEADLINE expires
    while a content frame is waiting for queue capacity (codex round-3 #2).

    Distinct from :class:`_DFlashClientGoneError`: the client may be perfectly
    healthy — the configured request ``timeout`` simply elapsed. This routes
    to the normal stream-timeout path (emit the ``finish_reason="length"``
    timeout notice + ``[DONE]``) instead of silently aborting, so the client
    always learns why the stream ended."""


class _DFlashAdmissionReservation:
    """One request slot from a :class:`_DFlashAdmission` gate.

    Ownership handoff (F1): the reservation is minted at the ASGI layer
    *before* the request body is parsed, so ``max_concurrent_requests``
    bounds the number of concurrently parsed bodies rather than only the
    number of requests already inside the endpoint. The middleware keeps
    a safety-net ``finally`` that force-releases the slot, but once the
    endpoint (stream lease / non-stream path) takes ownership via
    :meth:`claim` it is responsible for the release — including the
    deferred-until-worker-exits case — and the middleware net becomes a
    no-op. This prevents a double decrement of ``_reservations``.
    """

    def __init__(self, admission: _DFlashAdmission) -> None:
        self._admission = admission
        self._deferred = False
        self._released = False
        self._claimed = False

    def claim(self) -> None:
        """Mark the endpoint as the owner of this slot's release."""
        self._claimed = True

    @property
    def claimed(self) -> bool:
        return self._claimed

    def defer_release(self) -> None:
        """Keep the slot until a timed-out worker has actually stopped."""
        self._deferred = True

    def release(self, *, force: bool = False) -> None:
        if self._deferred and not force:
            return
        with self._admission._lock:
            if self._released:
                return
            self._released = True
            self._admission._reservations = max(0, self._admission._reservations - 1)


class _DFlashAdmission:
    """Bound DFlash's serial-lock queue before it consumes worker memory."""

    def __init__(self, max_concurrent_requests: int) -> None:
        self._max_concurrent_requests = max(0, int(max_concurrent_requests))
        self._reservations = 0
        self._lock = threading.Lock()

    def reserve(self) -> _DFlashAdmissionReservation:
        with self._lock:
            if (
                self._max_concurrent_requests > 0
                and self._reservations >= self._max_concurrent_requests
            ):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "DFlash is at its max_concurrent_requests admission "
                        "limit; retry shortly."
                    ),
                    headers={"Retry-After": "1"},
                )
            self._reservations += 1
        return _DFlashAdmissionReservation(self)


class _DFlashAdmissionMiddleware:
    """ASGI gate that reserves a DFlash slot BEFORE the body is parsed (F1).

    ``_DFlashAdmission`` was previously consulted inside the endpoint —
    i.e. after FastAPI had already streamed, JSON-decoded, and
    Pydantic-validated the full ``ChatCompletionRequest``. That bounded
    the number of requests *in generation* but not the number of
    concurrently parsed bodies: an authenticated burst could inflate
    unbounded parsed-body memory and only trip the 503 once each body was
    already resident. Reserving here — before ``self.app`` (and therefore
    before FastAPI's route body-binding) runs — makes
    ``max_concurrent_requests`` bound parsed-body memory too.

    The reservation is stashed in ``scope["state"]`` so the endpoint reuses
    the same slot (it calls :meth:`_DFlashAdmissionReservation.claim`
    and owns the release, including the deferred-worker-cleanup case). The
    middleware keeps a ``finally`` safety-net that force-releases the slot
    only when the endpoint never claimed it — e.g. the request was bounced
    by auth / rate-limit / body-size / validation before generation began,
    or never matched the chat route at all. This retains the reservation
    across the full streaming-response lifecycle without ever double
    decrementing the counter.
    """

    _GUARDED_METHOD = "POST"
    _GUARDED_PATH = "/v1/chat/completions"

    def __init__(self, app: Any, admission: _DFlashAdmission) -> None:
        self.app = app
        self._admission = admission

    async def __call__(self, scope, receive, send):
        if (
            scope.get("type") != "http"
            or scope.get("method") != self._GUARDED_METHOD
            or scope.get("path") != self._GUARDED_PATH
        ):
            return await self.app(scope, receive, send)

        # codex round-3 #3: reject unauthenticated / over-rate-limit clients
        # BEFORE reserving a slot. Otherwise a client that will ultimately be
        # 401'd or 429'd (e.g. one dribbling a slow body upload) still holds a
        # scarce DFlash slot for the whole request, denying service to
        # authorized callers. These checks are header-only — no body is read —
        # and they run OUTSIDE the admission gate. Because the chat route's
        # ``verify_api_key`` / ``check_rate_limit`` dependencies are dropped
        # for this path (this middleware is now the single source of truth on
        # ``/v1/chat/completions``), the rate limiter is consulted exactly
        # once per request, so counting stays correct.
        from starlette.requests import Request as _StarletteRequest

        from ...middleware.auth import (
            _extract_bearer_token,
            _rate_limit_client_id,
            _verify_api_key_values,
            rate_limiter,
        )

        request = _StarletteRequest(scope, receive)
        try:
            _verify_api_key_values(
                _extract_bearer_token(request.headers.get("Authorization"))
            )
        except HTTPException as exc:
            # codex round-5 #5: a 401 must carry the ``WWW-Authenticate:
            # Bearer`` challenge (RFC 6750). Forward any headers the
            # ``HTTPException`` attached, and — because the shared
            # ``_verify_api_key_values`` raises a bare 401 without the
            # challenge — inject it here for a 401 so the DFlash auth rejection
            # is spec-compliant rather than an unchallenged 401.
            challenge_headers = dict(exc.headers or {})
            if exc.status_code == 401:
                challenge_headers.setdefault("WWW-Authenticate", "Bearer")
            await _send_json_error(
                send,
                status_code=exc.status_code,
                message=str(exc.detail),
                error_type="invalid_request_error",
                code="invalid_api_key",
                extra_headers=challenge_headers,
            )
            return

        allowed, retry_after = rate_limiter.is_allowed(_rate_limit_client_id(request))
        if not allowed:
            await _send_json_error(
                send,
                status_code=429,
                message=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                error_type="rate_limit_error",
                code="rate_limit_exceeded",
                retry_after=str(retry_after),
            )
            return

        try:
            reservation = self._admission.reserve()
        except HTTPException as exc:
            await _send_json_error(
                send,
                status_code=503,
                message=str(exc.detail),
                error_type="service_unavailable",
                code="at_capacity",
                retry_after=(exc.headers or {}).get("Retry-After"),
            )
            return

        # Hand the slot to the endpoint via ASGI scope state. Starlette
        # seeds ``scope["state"]`` per-request; create it defensively for
        # bare-ASGI callers/tests that omit it.
        state = scope.setdefault("state", {})
        state["dflash_reservation"] = reservation
        try:
            await self.app(scope, receive, send)
        finally:
            # Safety net only. Once the endpoint claims the slot it owns
            # the release (possibly deferred until a timed-out worker
            # exits), so force-releasing here would double-decrement.
            if not reservation.claimed:
                reservation.release(force=True)


async def _send_json_error(
    send,
    *,
    status_code: int,
    message: str,
    error_type: str,
    code: str,
    retry_after: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> None:
    """Emit an OpenAI-shaped error JSON response from inside ASGI middleware.

    Hand-rolled (not ``HTTPException``) because these rejections happen
    before FastAPI's exception machinery is reachable — the request body has
    not been read. Used for the middleware's pre-admission 401 / 429 / 503
    (F1 + codex round-3 #3).

    ``extra_headers`` carries response headers the caught ``HTTPException``
    attached (codex round-5 #5) — notably ``WWW-Authenticate: Bearer`` on a
    401, the RFC 6750 challenge that FastAPI's own auth path emits and that a
    hand-rolled middleware rejection would otherwise drop.
    """
    body = json.dumps(
        {
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
                "param": None,
            }
        }
    ).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if retry_after:
        headers.append((b"retry-after", str(retry_after).encode("ascii")))
    if extra_headers:
        for name, value in extra_headers.items():
            # ``retry-after`` is handled above; skip a duplicate if both paths
            # supply it. Header names are ASCII-lowercased per ASGI convention.
            if name.lower() == "retry-after" and retry_after:
                continue
            headers.append((name.lower().encode("ascii"), str(value).encode("ascii")))
    try:
        await send(
            {"type": "http.response.start", "status": status_code, "headers": headers}
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})
    except Exception:  # noqa: BLE001 -- client already gone; nothing to emit
        logger.debug("DFlash %d send failed (client already disconnected)", status_code)


class _DFlashStreamLease:
    """Keep DFlash resources owned until a cancelled worker is cleaned up."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        reservation: _DFlashAdmissionReservation | None,
        deadline: float | None,
    ) -> None:
        self._loop = loop
        self._reservation = reservation
        self._deadline = deadline
        self.timed_out = False
        self._lock_acquired = False
        self._released = False
        self._cleanup_deferred = False
        self._active_future: asyncio.Future[Any] | None = None
        self._active_future_makes_generator = False
        self._generator: Any | None = None

    async def __aenter__(self) -> _DFlashStreamLease:
        if self._deadline is None:
            await _dflash_lock.acquire()
        else:
            remaining = self._deadline - self._loop.time()
            if remaining <= 0:
                self.timed_out = True
                return self
            try:
                await asyncio.wait_for(_dflash_lock.acquire(), timeout=remaining)
            except asyncio.TimeoutError:
                self.timed_out = True
                return self
        self._lock_acquired = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        future = self._active_future
        if future is not None and not future.done():
            self._defer_cleanup(future)
            return False
        if future is not None:
            # Cancellation can land after the executor future resolves but
            # before its caller stores a constructed generator.
            self._capture_generator(future)
            self.clear_future(future)
        await self._close_generator_and_release()
        return False

    def track_future(
        self, future: asyncio.Future[Any], *, makes_generator: bool = False
    ) -> None:
        self._active_future = future
        self._active_future_makes_generator = makes_generator

    def clear_future(self, future: asyncio.Future[Any]) -> None:
        if self._active_future is future:
            self._active_future = None
            self._active_future_makes_generator = False

    def set_generator(self, generator: Any) -> None:
        self._generator = generator

    def _capture_generator(self, future: asyncio.Future[Any]) -> None:
        if not self._active_future_makes_generator or self._generator is not None:
            return
        # F3: catch ``BaseException`` — a cancelled ``run_in_executor``
        # future raises ``CancelledError`` from ``.result()``, and
        # ``CancelledError`` subclasses ``BaseException`` (not
        # ``Exception``). The pre-fix ``except Exception`` let that escape
        # out of ``__aexit__`` before ``_close_generator_and_release`` /
        # ``_release`` ran, permanently leaking ``_dflash_lock`` when the
        # client cancelled during generator construction (the tiny window
        # after ``run_in_executor`` submits but before the single worker
        # thread picks the task up, so ``future.cancel()`` succeeds).
        try:
            candidate = future.result()
        except BaseException:  # noqa: BLE001 -- worker error/cancel has no generator to close
            return
        if not isinstance(candidate, Exception) and hasattr(candidate, "close"):
            self._generator = candidate

    def _defer_cleanup(self, future: asyncio.Future[Any]) -> None:
        if self._cleanup_deferred:
            return
        self._cleanup_deferred = True
        if self._reservation is not None:
            self._reservation.defer_release()

        def _on_worker_done(done: asyncio.Future[Any]) -> None:
            self._capture_generator(done)
            self._active_future = None
            self._active_future_makes_generator = False
            self._loop.create_task(self._close_generator_and_release())

        future.add_done_callback(_on_worker_done)

    async def _close_generator_and_release(self) -> None:
        generator = self._generator
        if generator is not None:

            def _close_gen() -> None:
                try:
                    generator.close()
                except Exception:  # noqa: BLE001 -- cleanup is best-effort
                    logger.debug(
                        "DFlash generator close raised; ignoring", exc_info=True
                    )

            close_future = self._loop.run_in_executor(_dflash_executor, _close_gen)
            try:
                # codex round-6 #2: bound the close wait. ``generator.close()``
                # runs the generator's finally blocks (mlx-vlm GPU teardown),
                # which normally return in microseconds — so on the common path
                # this await completes immediately and the lock/slot release
                # synchronously, preserving the fast-path contract. But a hung
                # close would otherwise block ``__aexit__`` — and therefore the
                # producer's terminal SSE (timeout notice + [DONE]) —
                # indefinitely. If the close overruns the grace window, DETACH
                # it (like an in-flight worker): keep the serial lock + slot
                # held until the close future actually completes (release fires
                # from its done-callback) while letting the response proceed.
                await asyncio.wait_for(
                    asyncio.shield(close_future),
                    timeout=_STREAM_GENERATOR_CLOSE_GRACE_SECONDS,
                )
            except asyncio.TimeoutError:
                if self._reservation is not None:
                    self._reservation.defer_release()
                close_future.add_done_callback(lambda _done: self._release())
                return
            except asyncio.CancelledError:
                # The client task can be cancelled a second time while it is
                # already unwinding. The queued close still owns the GPU
                # ordering, so release only from its completion callback.
                #
                # codex round-7 #1: DEFER the admission release too — otherwise
                # the reservation could be force-released here while
                # ``generator.close()`` (and thus the serial GPU lock) is still
                # in flight, freeing capacity for a new request to overlap
                # teardown. And RE-RAISE the ``CancelledError`` rather than
                # swallowing it: this coroutine runs inside the cancelled client
                # task's unwind, so returning normally would resurrect a
                # cancelled task and let producer work continue past
                # cancellation. The completion callback still owns the eventual
                # lock + slot release.
                if self._reservation is not None:
                    self._reservation.defer_release()
                close_future.add_done_callback(lambda _done: self._release())
                raise
        self._release()

    def _release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._lock_acquired:
            _dflash_lock.release()
            self._lock_acquired = False
        if self._reservation is not None:
            self._reservation.release(force=True)


@atexit.register
def _shutdown_dflash_executor() -> None:
    """Drain the DFlash workers on interpreter exit. Python registers an
    implicit atexit for ThreadPoolExecutor, but registering ours
    explicitly makes shutdown order deterministic and silences
    "unfinished thread" warnings during graceful uvicorn termination."""
    _dflash_executor.shutdown(wait=False, cancel_futures=True)
    _dflash_render_executor.shutdown(wait=False, cancel_futures=True)


def _build_app(
    *,
    model: Any,
    processor: Any,
    runtime: DFlashRuntime,
    served_model_name: str,
    default_max_tokens: int,
    cors_origins: list[str],
    no_thinking: bool = False,
    api_key: str | None = None,
    rate_limit: int = 0,
    max_request_bytes: int = 8 * 1024 * 1024,
    body_receive_timeout_seconds: float = 15.0,
    default_timeout: float = 1800.0,
    max_concurrent_requests: int = 256,
    cors_policy: Any | None = None,
    tool_call_parser: str | None = None,
    reasoning_parser_name: str | None = None,
) -> FastAPI:
    """Create the FastAPI application for DFlash mode.

    Per-app model state (``model``, ``processor``, ``runtime``,
    ``served_model_name``) is captured by closure. The security policy
    deliberately uses the shared server config and rate limiter, matching
    the unified server; DFlash therefore supports one active app per
    process.

    Note: ``_dflash_lock`` and ``_dflash_executor`` are *module-level*
    by design — every DFlash invocation must serialise through the
    same single-thread worker because mlx's GPU Stream is thread-local
    (see the ``_dflash_executor`` docstring at module top). A future
    multi-model deployment would still share that worker; one model
    can't run while another's generator is mid-step.
    """
    # DFlash owns a separate FastAPI application, so it cannot inherit the
    # unified server's dependencies or middleware implicitly. Copy the
    # already-resolved security settings into the shared config singleton
    # before wiring those common protections onto this app. In particular,
    # ``verify_api_key`` and ``RequestBodyLimitMiddleware`` read this
    # singleton at request time.
    cfg = get_config()
    cfg.api_key = api_key
    cfg.max_request_bytes = max(0, int(max_request_bytes))
    cfg.body_receive_timeout_seconds = max(0.0, float(body_receive_timeout_seconds))
    cfg.default_timeout = max(0.0, float(default_timeout))
    cfg.enable_auto_tool_choice = bool(tool_call_parser)
    cfg.tool_call_parser = tool_call_parser
    cfg.reasoning_parser_name = reasoning_parser_name

    from ...middleware.auth import (
        configure_rate_limiter,
    )
    from ...middleware.body_depth import install_request_body_depth_middleware
    from ...middleware.body_size import install_request_body_limit_middleware

    configure_rate_limiter(rate_limit, enabled=rate_limit > 0)

    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    from ...middleware.auth import _verify_api_key_values

    _bearer_scheme = HTTPBearer(auto_error=False)

    async def _verify_api_key_with_bearer_challenge(
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    ) -> bool:
        """``verify_api_key`` that attaches the RFC 6750 bearer challenge.

        codex round-5 #5 / round-6 #1: the shared ``verify_api_key`` raises a
        bare 401 (no ``WWW-Authenticate`` header). Any DFlash bearer-protected
        route that used it directly returned an unchallenged 401. This wrapper
        runs the same verification but re-raises the 401 WITH the challenge so
        ``/v1/models`` is RFC-compliant, mirroring the chat route's ASGI
        middleware — without editing the shared verifier (a different lane).
        """
        bearer_key = credentials.credentials if credentials is not None else None
        try:
            return _verify_api_key_values(bearer_key)
        except HTTPException as exc:
            if exc.status_code == 401:
                headers = dict(exc.headers or {})
                headers.setdefault("WWW-Authenticate", "Bearer")
                raise HTTPException(
                    status_code=exc.status_code,
                    detail=exc.detail,
                    headers=headers,
                ) from exc
            raise

    app = FastAPI(title="Rapid-MLX (DFlash)")
    # DFlash has one GPU worker and serializes generation with
    # ``_dflash_lock``. Bound its waiting room as well so a burst cannot
    # accumulate an unbounded number of requests and their parsed bodies.
    admission = _DFlashAdmission(max_concurrent_requests)
    app.state.dflash_admission = admission
    # D-ANTHRO-VALIDATION F11: install the shared exception handlers so
    # Pydantic validation errors return the canonical
    # ``{"error":{"type":"invalid_request_error","code":"invalid_request",
    # ...}}`` envelope at HTTP 400 instead of FastAPI's default 422 with
    # an unbounded ``detail`` array. Same handlers the main server uses.
    from ...middleware.exception_handlers import install_exception_handlers

    install_exception_handlers(app)
    # Match the main server's generic JSON request defenses. The body-size
    # middleware also enforces the configured body-receive idle timeout, so a
    # DFlash request cannot bypass the slow-client protection by taking this
    # dedicated app path.
    # Middleware nesting (``add_middleware`` wraps last-added OUTERMOST).
    # Target order, outer → inner:
    #   body-size (413/408)  →  admission (auth/rate/reserve)  →  body-depth
    #   →  route (Pydantic body bind)
    # Rationale:
    #   * body-size stays OUTERMOST so an oversized body is rejected with 413
    #     (and a slow upload with 408) as cheaply as possible — before any
    #     auth/rate/admission work (codex round-3 didn't ask to move these
    #     generic DoS gates, and 413-cheapest-first is the established
    #     contract the security test pins).
    #   * admission sits INSIDE body-size but OUTSIDE body-depth + the route,
    #     so its header-only auth + rate rejection (codex round-3 #3) runs
    #     before a slot is reserved, and the reservation is taken before
    #     body-depth reads the body and before FastAPI binds the Pydantic
    #     model (F1 — bounds parsed-body memory).
    # Add order to realize that nesting: body-depth (innermost), then
    # admission, then body-size (outermost) last.
    install_request_body_depth_middleware(app)
    app.add_middleware(_DFlashAdmissionMiddleware, admission=admission)
    install_request_body_limit_middleware(app)
    # F-090/F-091: register CORS only when an explicit origin allowlist is
    # configured. ``cors_origins=[]`` (the new default — see
    # ``vllm_mlx/server.py::configure_cors_from_env``) skips the middleware
    # entirely so preflight returns 405 and no ``Access-Control-*`` header
    # leaks. The dflash path mirrors the main server's stance.
    if cors_policy is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_policy.origins,
            allow_credentials=cors_policy.allow_credentials,
            allow_methods=cors_policy.methods,
            allow_headers=cors_policy.headers,
            max_age=cors_policy.max_age,
        )
    elif cors_origins:
        wildcard = "*" in cors_origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            # Fetch spec: wildcard + credentials is invalid; flip off
            # credentials when ``*`` is present so the response stays
            # browser-valid.
            allow_credentials=not wildcard,
            # F-091: previously ``["*"]`` (DELETE/GET/HEAD/OPTIONS/PATCH/
            # POST/PUT). The dflash server only serves the OpenAI-compat
            # chat surface, so POST/GET/OPTIONS is the correct allowlist.
            allow_methods=["POST", "GET", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-Rapid-MLX-Internal"],
            max_age=3600,
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "engine": "dflash",
            "mode": "single-user-serial",
            "drafter": runtime.drafter_repo,
        }

    @app.get(
        "/v1/models",
        dependencies=[Depends(_verify_api_key_with_bearer_challenge)],
    )
    async def list_models() -> ModelsResponse:
        return ModelsResponse(
            data=[
                ModelInfo(
                    id=served_model_name,
                    created=int(time.time()),
                    owned_by="rapid-mlx",
                )
            ]
        )

    # codex round-3 #3: auth + rate-limit for this route are enforced in
    # ``_DFlashAdmissionMiddleware`` BEFORE the admission slot is reserved and
    # before the body is read — NOT as route dependencies here. Keeping them
    # here too would consult the rate limiter twice per request (halving the
    # effective limit) and would run only AFTER a slot was already reserved.
    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        request: ChatCompletionRequest, http_request: Request
    ):
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        if request.n is not None and request.n > 1:
            raise HTTPException(status_code=400, detail="n > 1 is not supported")
        if request.tools and not cfg.tool_call_parser:
            raise HTTPException(
                status_code=400,
                detail=(
                    "DFlash tool calling requires an enabled tool-call parser. "
                    "Remove --no-tool-call-parser or configure one explicitly."
                ),
            )
        if request.tools and request.tool_choice not in (None, "auto"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "DFlash currently supports tool_choice='auto' only. "
                    "Restart without DFlash for forced or disabled tool choice."
                ),
            )
        # Surface unsupported params explicitly rather than silently
        # ignoring — silent-drop is the bug class that makes users think
        # they got logprobs / JSON-schema / etc. when they didn't.
        if request.logprobs:
            raise HTTPException(
                status_code=400,
                detail="logprobs is not supported in DFlash mode. Restart without DFlash.",
            )
        if request.response_format is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "response_format (structured output) is not supported "
                    "in DFlash mode. Restart without DFlash."
                ),
            )
        # F1: the slot was reserved at the ASGI layer
        # (``_DFlashAdmissionMiddleware``) BEFORE this body was parsed, so
        # admission bounds parsed-body memory too. Reuse that slot. A
        # direct-ASGI/test caller that bypasses the middleware won't have
        # seeded scope state, so fall back to reserving here (still gated).
        #
        # Ownership handoff (``claim``) is deliberately NOT taken here
        # (codex round-4 #1). If we claimed now, a client that disconnects
        # while Starlette is sending the ``StreamingResponse`` headers would
        # leave the stream iterator unstarted — its ``finally`` release never
        # runs — while the middleware's safety net refuses to release a
        # *claimed* slot, permanently leaking capacity. Instead each path
        # claims only at the point its release is guaranteed to fire:
        #   * non-stream: inside the ``finally`` below, which always runs;
        #   * stream: inside ``_stream_with_admission`` when the iterator
        #     actually begins. Until then the slot stays *unclaimed*, so the
        #     middleware's safety-net ``finally`` owns the release if response
        #     startup fails.
        reservation = getattr(http_request.state, "dflash_reservation", None)
        if reservation is None:
            reservation = admission.reserve()
        # Tracks the offloaded render's ``concurrent.futures.Future`` (the real
        # worker-thread state) so the failure/cancellation cleanup can decide
        # whether the slot may be freed now or must be held until an in-flight
        # render actually finishes (codex round-7 #2).
        render_future: concurrent.futures.Future[str] | None = None
        try:
            # F4 (codex round-2 #3): resolve the effective timeout with an
            # ``is None`` check, NOT ``request.timeout or default_timeout``.
            # ``or`` treats an explicit ``timeout=0`` as falsy and swaps in
            # ``default_timeout``, silently contradicting the ``timeout <= 0``
            # "no deadline" semantics both completion paths now honor. Only
            # an omitted (``None``) timeout should fall back to the default.
            #
            # codex round-4 #4: resolve this BEFORE rendering the prompt and
            # establish ONE absolute deadline here — prompt rendering (chat
            # template application + tokenization) can be non-trivial, and the
            # pre-fix code charged its cost to nobody: each completion helper
            # started its own ``deadline = now + timeout`` only AFTER rendering
            # returned, so a slow template could blow well past the client's
            # configured ``timeout`` unenforced. ``timeout=0`` still means "no
            # deadline" (``request_deadline is None``).
            effective_timeout = (
                request.timeout if request.timeout is not None else default_timeout
            )
            # codex round-5 #6: keep the ORIGINAL configured timeout for
            # human-facing diagnostics. ``effective_timeout`` is rewritten below
            # to the post-render REMAINING budget (so the absolute deadline
            # spans render + generation), but a 504 message should still say
            # "timed out after 60s" for a 60s request that spent 20s rendering
            # — not "after 40s" — or users chase a phantom shorter limit.
            request_timeout_for_diagnostics = effective_timeout
            loop = asyncio.get_running_loop()
            request_deadline = (
                loop.time() + effective_timeout
                if effective_timeout and effective_timeout > 0
                else None
            )

            # Render chat messages into a single prompt string via mlx-vlm's
            # processor. We pass through the model's chat template so the
            # tokenizer-side reasoning/tool markers match what the model was
            # trained on; no rapid-mlx-side prompt mutation happens here.
            #
            # Resolve enable_thinking (#387). The dflash app captures its own
            # ``no_thinking`` by closure rather than going through the
            # ServerConfig singleton, so we apply that override first then
            # delegate the request-side precedence (chat_template_kwargs >
            # request.enable_thinking > None) to the shared extractor — same
            # source of truth as the OpenAI/anthropic helper, but without the
            # ``cfg.no_thinking`` consult that doesn't apply to dflash.
            from ...service.helpers import _extract_thinking_from_request

            if no_thinking:
                enable_thinking: bool | None = False
            else:
                enable_thinking = _extract_thinking_from_request(request)
            effective_thinking = False if enable_thinking is None else enable_thinking
            if effective_thinking and not cfg.reasoning_parser_name:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "DFlash thinking output requires a reasoning parser. "
                        "Remove --no-reasoning-parser or configure one explicitly."
                    ),
                )

            # codex round-4 #4 → round-5 #1/#2: OFFLOAD rendering so a heavy
            # chat template cannot block the event loop (starving other
            # connections, health checks, and — critically — the very deadline
            # enforcement we rely on elsewhere), and bound it by the request
            # deadline so an expensive render on a tight budget fails fast with
            # a 504 rather than running unbounded.
            #
            # It runs on ``_dflash_render_executor`` — a pool SEPARATE from the
            # single GPU worker — NOT ``_dflash_executor``. Using the GPU worker
            # (the round-4 mistake) let a later request's render run between the
            # token steps of the lock-owning generation, consuming its deadline
            # and touching its model/processor mid-generation. Rendering
            # produces only a string (pure CPU tokenizer work, no mlx GPU
            # arrays), so it needs neither the GPU thread affinity nor
            # ``_dflash_lock``. Because this render holds NO serial/GPU resource,
            # an uncancellable overrun can be abandoned and its admission slot
            # released immediately (round-5 #2) — nothing serial is left running
            # behind it, unlike a timed-out generation worker.
            def _render() -> str:
                return _render_prompt(
                    processor,
                    model,
                    request,
                    enable_thinking=effective_thinking,
                )

            # codex round-8 #1: validate the remaining budget BEFORE submitting
            # any work. Submitting first — then checking ``render_budget <= 0``
            # — could still start an uncancellable tokenizer render on an
            # already-expired deadline before returning the 504. Check first;
            # only submit when there is time to render.
            if request_deadline is not None and request_deadline - loop.time() <= 0:
                raise HTTPException(
                    status_code=504,
                    detail=(
                        "DFlash request deadline elapsed before prompt "
                        "rendering could start."
                    ),
                )

            # codex round-7 #2: submit via the executor DIRECTLY so we hold the
            # ``concurrent.futures.Future`` — which tracks the ACTUAL worker
            # thread. (``loop.run_in_executor`` returns an asyncio future whose
            # ``.cancel()`` marks itself done while the thread keeps running, so
            # its ``.done()`` cannot tell "still executing" from "finished" — the
            # exact ambiguity that let the slot be freed under a live render.)
            # ``asyncio.wrap_future`` gives an awaitable view; cancelling that
            # view never touches the underlying thread future, so
            # ``render_cf.done()`` in the cleanup below reflects true completion.
            render_cf = _dflash_render_executor.submit(_render)
            render_future = render_cf  # tracked for the cleanup handler
            render_awaitable = asyncio.wrap_future(render_cf)
            if request_deadline is None:
                prompt = await render_awaitable
            else:
                render_budget = request_deadline - loop.time()
                try:
                    prompt = await asyncio.wait_for(
                        asyncio.shield(render_awaitable), timeout=render_budget
                    )
                except asyncio.TimeoutError as exc:
                    # ``cancel()`` stops the render only if it is still QUEUED;
                    # a render already running on the pool keeps going (the
                    # cleanup handler below holds the slot until it drains).
                    render_cf.cancel()
                    raise HTTPException(
                        status_code=504,
                        detail=(
                            "DFlash prompt rendering exceeded the request "
                            "timeout of "
                            f"{_format_timeout_seconds(request_timeout_for_diagnostics)}."
                        ),
                    ) from exc

            max_tokens = (
                request.max_tokens
                if request.max_tokens is not None
                else default_max_tokens
            )
            temperature = (
                request.temperature if request.temperature is not None else 0.0
            )
            top_p = request.top_p if request.top_p is not None else 1.0

            gen_kwargs = dict(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                draft_model=runtime.drafter,
                draft_kind=runtime.kind,
            )

            # Pass the REMAINING budget (post-render) to the completion helper,
            # not the original timeout, so the single absolute deadline
            # established above spans render + generation. ``0`` keeps its "no
            # deadline" meaning ONLY when the request had no deadline to begin
            # with; a request that DID set a deadline which rendering fully
            # consumed must NOT collapse into the no-deadline sentinel (that
            # would let generation run unbounded) — surface a 504 instead.
            if request_deadline is None:
                effective_timeout = 0.0
            else:
                remaining_budget = request_deadline - loop.time()
                if remaining_budget <= 0:
                    raise HTTPException(
                        status_code=504,
                        detail=(
                            "DFlash request deadline elapsed during prompt "
                            "rendering (timeout "
                            f"{_format_timeout_seconds(request_timeout_for_diagnostics)})."
                        ),
                    )
                effective_timeout = remaining_budget
        except BaseException:
            # codex round-7 #2: if we are unwinding while the offloaded render
            # is STILL in flight (timeout/cancel/disconnect), releasing the slot
            # now would let repeated cancelled requests each leave a render
            # running/queued on the single-worker render pool while freeing
            # capacity — an unbounded pileup that bypasses
            # ``max_concurrent_requests``. Try to cancel it; if it was still
            # queued the cancel succeeds and we release immediately, but if it
            # is already running (cancel returns False) DEFER the slot release
            # to the render's completion so the admission gate keeps counting it
            # until the work actually drains. A finished render releases at once.
            if render_future is not None and not render_future.done():
                render_future.cancel()
            if render_future is not None and not render_future.done():
                # Still running (uncancellable) — hold the slot until it drains.
                # CLAIM ownership so the ASGI middleware safety net does not
                # force-release the (as-yet unclaimed) slot out from under the
                # still-running render; the completion callback then owns the
                # single deferred release.
                reservation.claim()
                reservation.defer_release()
                render_future.add_done_callback(
                    lambda _f: reservation.release(force=True)
                )
            else:
                reservation.release()
            raise

        if request.stream:
            return StreamingResponse(
                _stream_with_admission(
                    _stream_completion(
                        prompt=prompt,
                        request=request,
                        served_model_name=served_model_name,
                        gen_kwargs=gen_kwargs,
                        model=model,
                        processor=processor,
                        timeout=effective_timeout,
                        timeout_label=request_timeout_for_diagnostics,
                        deadline=request_deadline,
                        admission_reservation=reservation,
                        enable_thinking=effective_thinking,
                    ),
                    reservation,
                ),
                media_type="text/event-stream",
            )

        # Non-stream path: take ownership now. Unlike the streaming path,
        # this coroutine is awaited directly, so the ``finally`` below is
        # guaranteed to run (no header-send-then-abort gap). Claiming here is
        # required because ``_non_stream_completion`` may ``defer_release`` the
        # slot until a timed-out worker actually exits; if the slot were still
        # unclaimed the middleware safety net would force-release it out from
        # under the still-running worker.
        reservation.claim()
        try:
            return await _non_stream_completion(
                prompt=prompt,
                request=request,
                served_model_name=served_model_name,
                gen_kwargs=gen_kwargs,
                model=model,
                processor=processor,
                timeout=effective_timeout,
                timeout_label=request_timeout_for_diagnostics,
                deadline=request_deadline,
                admission_reservation=reservation,
                enable_thinking=effective_thinking,
            )
        finally:
            reservation.release()

    return app


def _render_prompt(
    processor: Any,
    model: Any,
    request: ChatCompletionRequest,
    *,
    enable_thinking: bool | None = None,
) -> str:
    """Apply the model's chat template via mlx-vlm's helper.

    mlx-vlm's ``apply_chat_template`` mirrors mlx-lm's but accepts the
    multimodal kwargs the VLM models need (we pass ``num_images=0`` since
    DFlash-eligible aliases are text-only Qwen3.5/3.6 variants today).

    ``enable_thinking`` resolution (caller-side; we just thread through):
      None  → disable thinking because DFlash has no reasoning parser.
      True  → force chain-of-thought on.
      False → force chain-of-thought off (server --no-thinking or per-
              request ``enable_thinking=false`` body field).
    """
    from mlx_vlm.prompt_utils import apply_chat_template

    messages = []
    for m in request.messages:
        message = m.model_dump(exclude_none=True)
        content = m.content
        if isinstance(content, list):
            # Multimodal payload — DFlash server is text-only. Collapse
            # text parts; non-text parts (image/audio/video) are
            # dropped. A 400 would surprise users mid-prompt, but a
            # silent drop hides "why is my model ignoring the image?"
            # debugging — so we degrade with a visible WARN log per
            # request that hits this path.
            text_pieces = []
            dropped_kinds: list[str] = []
            for part in content:
                part_type = part.type if hasattr(part, "type") else part.get("type", "")
                if part_type == "text":
                    text_pieces.append(
                        part.text if hasattr(part, "text") else part.get("text", "")
                    )
                elif part_type:
                    dropped_kinds.append(part_type)
            if dropped_kinds:
                logger.warning(
                    "DFlash server is text-only; dropped %d non-text "
                    "content part(s) of type(s) %s. The request will be "
                    "served using text parts only — switch to the standard "
                    "server without DFlash for full multimodal support.",
                    len(dropped_kinds),
                    sorted(set(dropped_kinds)),
                )
            content = "".join(text_pieces)
        message["content"] = content
        messages.append(message)

    # Default to final-answer mode. Explicit thinking opt-in is post-processed
    # by the same reasoning pipeline as the standard server.
    effective_thinking = False if enable_thinking is None else enable_thinking
    template_kwargs: dict[str, Any] = {
        "num_images": 0,
        "num_audios": 0,
        "enable_thinking": effective_thinking,
    }
    if request.tools:
        from ...api.tool_calling import convert_tools_for_template

        template_kwargs["tools"] = convert_tools_for_template(request.tools)

    return apply_chat_template(
        processor,
        model.config,
        messages,
        **template_kwargs,
    )


async def _stream_with_admission(
    stream: AsyncIterator[bytes], reservation: _DFlashAdmissionReservation
) -> AsyncIterator[bytes]:
    """Release an admission slot even when Starlette cancels an SSE stream.

    Ownership handoff happens here, at the first body of this generator —
    i.e. only once Starlette has begun consuming the ``StreamingResponse``
    (codex round-4 #1). Claiming any earlier (e.g. in the endpoint before
    returning the response) risks a leak: if sending the response headers
    fails, this generator is never iterated, its ``finally`` release never
    runs, and the middleware safety net would refuse to release a *claimed*
    slot. By claiming here, an unstarted stream stays unclaimed and the
    middleware ``finally`` reclaims the slot. Once claimed, the ``finally``
    below owns the release for the whole stream lifecycle.
    """
    reservation.claim()
    try:
        async for chunk in stream:
            yield chunk
    finally:
        # ``StreamingResponse`` runs its background task only on a normal
        # return. Closing the inner generator here covers client disconnects
        # and send failures too, so its GPU cleanup runs before the slot is
        # made available again.
        #
        # codex round-5 #4: ``reservation.release()`` MUST run even if
        # ``aclose()`` raises ``asyncio.CancelledError`` — which subclasses
        # ``BaseException``, NOT ``Exception``, so an ``except Exception`` would
        # let it skip the release and permanently leak the slot on a client
        # cancellation. Put the release in its own unconditional ``finally`` so
        # it fires regardless of how ``aclose()`` unwinds, then let cancellation
        # propagate.
        aclose = getattr(stream, "aclose", None)
        try:
            if aclose is not None:
                await aclose()
        except Exception:  # noqa: BLE001 -- release remains mandatory
            logger.debug(
                "DFlash stream close raised; releasing admission", exc_info=True
            )
        finally:
            reservation.release()


async def _stream_completion(
    *,
    prompt: str,
    request: ChatCompletionRequest,
    served_model_name: str,
    gen_kwargs: dict[str, Any],
    model: Any,
    processor: Any,
    timeout: float | None = None,
    timeout_label: float | None = None,
    deadline: float | None = None,
    admission_reservation: _DFlashAdmissionReservation | None = None,
    enable_thinking: bool = False,
) -> AsyncIterator[bytes]:
    """Stream OpenAI-format chunks. Generation happens under the serial
    lock; chunks are forwarded as ``data: ...\\n\\n`` SSE events.

    ``deadline`` is an ABSOLUTE ``loop.time()`` instant (codex round-8 #2). The
    endpoint establishes one deadline spanning render + generation and passes
    it here directly; reconstructing a fresh deadline from a relative
    ``timeout`` at this point would silently hand back the wall-clock time spent
    between the endpoint and the first stream step (returning ``StreamingResponse``
    + header send). When ``deadline`` is omitted (direct callers/tests) it is
    derived from ``timeout`` as before.

    ``timeout`` is the ENFORCED remaining budget (post prompt-render);
    ``timeout_label`` is the ORIGINAL configured request timeout used only in
    human-facing timeout messages (codex round-5 #6), so a client sees the
    limit it actually asked for rather than the render-reduced remainder. When
    omitted (direct callers/tests) it mirrors ``timeout``."""
    from mlx_vlm import stream_generate

    if timeout_label is None:
        timeout_label = timeout

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    postprocessor: StreamingPostProcessor | None = None
    if request.tools or enable_thinking:
        postprocessor = StreamingPostProcessor(
            get_config(),
            tools_requested=bool(request.tools),
            enable_thinking=enable_thinking,
            request=request.model_dump(),
        )
        postprocessor.reset()

    # First chunk — role marker. Emitted by the consumer loop below,
    # before the producer task is started, so the client sees the role
    # delta before any GPU lock is touched (matches the prior contract).
    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": served_model_name,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }

    # Track max_tokens so we can report ``finish_reason="length"`` when
    # generation was truncated (OpenAI clients distinguish "stop"
    # = natural end / stop sequence from "length" = token-budget hit;
    # presenting "stop" for a truncated reply misleads downstream tools).
    _max_tokens = gen_kwargs.get("max_tokens")

    # Resolve the model's EOS token id (best-effort). Used by the
    # length-vs-stop disambiguation below; falls back to None when the
    # processor doesn't expose a tokenizer (the heuristic then degrades
    # to pure token-count comparison).
    _eos_ids: set[int] = set()
    _tok = getattr(processor, "tokenizer", processor)
    _eos = getattr(_tok, "eos_token_id", None)
    if isinstance(_eos, int):
        _eos_ids.add(_eos)
    elif isinstance(_eos, (list, tuple, set)):
        _eos_ids.update(int(t) for t in _eos if isinstance(t, int))

    loop = asyncio.get_running_loop()
    # codex round-8 #2: prefer the absolute ``deadline`` the endpoint passed
    # (which already spans render + generation on this loop's clock). Only
    # derive one from the relative ``timeout`` when a direct caller/test omits
    # it — otherwise the clock would be re-based here, refunding the time spent
    # returning the response and sending headers.
    if deadline is None and timeout and timeout > 0:
        deadline = loop.time() + timeout
    timed_out = object()
    lease = _DFlashStreamLease(loop, admission_reservation, deadline)

    # F2: decouple generation from socket writes. Generation runs in a
    # dedicated producer task that OWNS the lease (serial GPU lock +
    # admission slot) and pushes ready-to-send SSE byte chunks onto this
    # queue. This outer coroutine only drains the queue and yields to the
    # client socket. If the client applies backpressure (stops reading),
    # the outer ``yield`` suspends — but the producer keeps running and
    # its deadline checks keep firing, so a stalled reader can no longer
    # pin ``_dflash_lock`` past ``timeout``. Lease acquisition/release is
    # therefore independent of downstream socket writes.
    #
    # codex round-2 #1: the queue is BOUNDED so a stalled client cannot make
    # the producer buffer the whole completion. When the queue fills, the
    # producer's ``put`` blocks; if it stays blocked past the backpressure
    # timeout, the client is treated as gone and generation is aborted with
    # the generator + lease cleaned up.
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)

    # codex round-4 #3: terminal frames (the ``finish_reason`` notice + [DONE])
    # that could NOT be pushed through the bounded queue — because it stayed
    # full past the backpressure cap — are stashed here for the CONSUMER to
    # emit DIRECTLY to the socket after it finishes draining. This guarantees a
    # connected-but-slow client always receives an explicit stream terminator
    # even when the queue-based handoff itself hit backpressure; the consumer
    # owns the socket and will deliver these to any client still reading.
    pending_terminal: list[bytes] = []

    async def _produce() -> None:
        """Own the lease, generate, and push SSE chunks onto the queue.

        Always terminates the queue with a ``None`` sentinel so the
        consumer's drain loop can exit even on error/cancel. Runs under
        ``async with lease`` so the serial lock + admission slot are
        released (or deferred until a timed-out/cancelled worker exits)
        the moment generation finishes, regardless of consumer speed.
        """
        finish_reason = "stop"
        total_completion_tokens = 0
        prompt_tokens = 0
        # Track the last token id to disambiguate "hit max_tokens but the
        # final token was actually EOS" — without this we'd falsely flag a
        # natural-stop response as truncated when it lands on exactly the
        # budget. None means "no token observed yet".
        last_token_id: int | None = None
        error_message: str | None = None
        postprocess_spilled = False

        async def _emit(item: bytes, *, terminal: bool = False) -> None:
            """Put one SSE frame on the bounded queue.

            CONTENT frames (``terminal=False``) are emitted while the lease
            holds the GPU lock, so a full queue means the lock is pinned. The
            wait is bounded by BOTH:
            * the request deadline (codex round-2 #2): a full queue must not
              let a stalled client retain the sole GPU lock past the configured
              ``timeout``; and
            * the backpressure timeout (codex round-2 #1): even with no
              deadline, an indefinitely-stalled client must not make the
              producer buffer an unbounded completion.
            On timeout we tell the two apart (codex round-3 #2):
            * the request DEADLINE elapsed → raise ``_DFlashStreamDeadlineError``
              so the loop emits the normal timeout notice + ``[DONE]`` (the
              client is told why the stream ended); vs
            * pure BACKPRESSURE (no deadline, or backpressure shorter than the
              remaining deadline) → raise ``_DFlashClientGoneError`` to abort
              silently (the client is gone; nothing to tell it).

            TERMINAL frames (final finish_reason/usage chunk, timeout/error
            notice, ``[DONE]``) are emitted AFTER the lease is released, so they
            cannot pin the GPU lock. They wait only on the backpressure timeout
            — NEVER the deadline — so an already-blown deadline can't suppress
            the explanation of how the stream ended. A truly-gone client can
            still not wedge them thanks to the backpressure cap."""
            backpressure = _STREAM_BACKPRESSURE_TIMEOUT_SECONDS
            deadline_left = (
                None
                if (terminal or deadline is None)
                else max(0.0, deadline - loop.time())
            )
            wait = (
                backpressure
                if deadline_left is None
                else min(backpressure, deadline_left)
            )
            try:
                await asyncio.wait_for(queue.put(item), timeout=wait)
            except asyncio.TimeoutError as exc:
                # Deadline expiry is signalled ONLY when the deadline is what
                # actually bounded the wait (it is <= the backpressure cap).
                if deadline_left is not None and deadline_left <= backpressure:
                    raise _DFlashStreamDeadlineError from exc
                raise _DFlashClientGoneError from exc

        async def _emit_postprocessed_event(
            event: Any, *, terminal: bool = False
        ) -> None:
            """Translate one shared postprocessor event to OpenAI SSE."""
            nonlocal finish_reason, postprocess_spilled

            if event.finish_reason is not None:
                finish_reason = event.finish_reason
            delta: dict[str, Any] = {}
            if event.content:
                delta["content"] = event.content
            if event.reasoning:
                delta["reasoning_content"] = event.reasoning
            if event.tool_calls:
                delta["tool_calls"] = event.tool_calls
            if not delta:
                return
            piece = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }
                ],
            }
            frame = f"data: {json.dumps(piece)}\n\n".encode()
            if terminal and postprocess_spilled:
                pending_terminal.append(frame)
                return
            try:
                await _emit(frame, terminal=terminal)
            except _DFlashClientGoneError:
                if not terminal:
                    raise
                postprocess_spilled = True
                pending_terminal.append(frame)

        async def _await_worker(func: Any, *, makes_generator: bool = False) -> Any:
            """Wait without cancelling an mlx operation on deadline expiry."""
            if lease.timed_out:
                return timed_out
            # F5: recheck the remaining deadline BEFORE submitting the
            # executor job. Acquiring the serial lock (and prior token
            # steps) can consume the whole budget; submitting first — as
            # the pre-fix code did — would start expensive, non-preemptible
            # GPU work for an already-expired request and keep the lock
            # held until it finished. Check first; only submit if time
            # remains.
            if deadline is not None and deadline - loop.time() <= 0:
                return timed_out
            future = loop.run_in_executor(_dflash_executor, func)
            lease.track_future(future, makes_generator=makes_generator)
            if deadline is None:
                # F3: shield so a client cancellation during this bare
                # await cannot tear a non-preemptible GPU step mid-flight
                # while leaving the lock leaked. The shield keeps the
                # underlying worker running; the raised ``CancelledError``
                # unwinds into ``lease.__aexit__``, which defers cleanup
                # until the worker completes and then releases the lock.
                # ``clear_future`` runs only on the normal (uncancelled)
                # path — on cancel we leave ``_active_future`` set so
                # ``__aexit__`` sees the in-flight future and routes it
                # through deferred cleanup.
                result = await asyncio.shield(future)
                lease.clear_future(future)
                return result
            remaining = deadline - loop.time()
            if remaining <= 0:
                # F2 (codex round-2 #2): the deadline is already gone. Do
                # NOT block on the non-preemptible worker — a hung token
                # step would make the timeout ineffective indefinitely.
                # Return the timeout marker IMMEDIATELY, leaving
                # ``_active_future`` tracked so the lease's ``__aexit__``
                # routes the still-running worker through deferred cleanup
                # (retaining the lock + admission slot until the worker
                # actually stops, then closing its generator). Mark the
                # lease timed-out so any further ``_await_worker`` call
                # short-circuits.
                lease.timed_out = True
                return timed_out
            done, _pending = await asyncio.wait({future}, timeout=remaining)
            if done:
                result = future.result()
                lease.clear_future(future)
                return result

            # F2 (codex round-2 #2): the wait elapsed before the worker
            # finished. Same policy as above — surface the timeout
            # immediately and defer the in-flight worker's cleanup to the
            # lease instead of blocking the response on a non-preemptible
            # step. ``_active_future`` stays tracked for ``__aexit__``.
            lease.timed_out = True
            return timed_out

        finish_reason = "stop"
        async with lease:
            # mlx-vlm's stream_generate is a sync generator — run it in a
            # thread pool so we don't block the FastAPI event loop. Iterate
            # by polling with ``run_in_executor`` per chunk. We're already
            # inside a coroutine, so use ``get_running_loop`` (the 3.10+
            # idiom; ``get_event_loop`` is deprecated for in-coroutine use).
            # The executor MUST be ``_dflash_executor`` (single-thread) so
            # consecutive ``next(gen)`` calls land on the same worker —
            # mlx's GPU Stream is thread-local and a hand-off across worker
            # threads would crash mid-generation.
            # Create the generator on the same worker that will drive it,
            # not on the event-loop thread — otherwise the first ``next``
            # crosses a thread boundary just like the rest.
            #
            # Wrap construction in a sentinel pattern too: if
            # ``stream_generate`` raises at setup time (OOM, missing kernel,
            # bad arg) the exception would otherwise propagate out of the
            # async generator and leave the SSE client hanging without a
            # ``[DONE]``. Surfacing it as an error SSE keeps the contract
            # the same as the mid-stream error path below.
            def _make_gen():
                try:
                    return stream_generate(model, processor, prompt, **gen_kwargs)
                except Exception as e:  # noqa: BLE001 — surface upstream; outer code converts to error SSE
                    return e

            gen_or_err = await _await_worker(_make_gen, makes_generator=True)
            # ``_await_worker`` now returns the ``timed_out`` sentinel
            # immediately on deadline expiry (codex round-2 #2) — never a
            # tuple. When it times out during construction we have no
            # generator to drive; the lease's deferred cleanup owns closing
            # the (possibly in-flight) worker.
            if gen_or_err is timed_out:
                error_message = f"DFlash stream timed out after {_format_timeout_seconds(timeout_label)}."
                finish_reason = "length"
                gen = None
            elif isinstance(gen_or_err, Exception):
                logger.exception(
                    "DFlash stream_generate raised at construction: %s",
                    gen_or_err,
                    exc_info=gen_or_err,
                )
                error_message = f"{type(gen_or_err).__name__}: {gen_or_err}"
                # OpenAI ChatCompletion only accepts {stop, length,
                # tool_calls, content_filter, function_call}. The error
                # block on the final SSE chunk carries the abort details
                # for clients.
                finish_reason = "length"
                gen = None
            else:
                gen = gen_or_err
                lease.set_generator(gen)

            # Sentinels distinguish "generator exhausted" (None) from
            # "generator raised mid-stream" (an Exception instance).
            # Catching only StopIteration would let any other mlx-vlm error
            # propagate through run_in_executor, abort the response
            # coroutine, and leave the SSE client hanging without a final
            # ``[DONE]`` — the client then either times out or holds the
            # connection forever.
            def _next_chunk():
                try:
                    return next(gen)
                except StopIteration:
                    return None
                except Exception as e:  # noqa: BLE001 — surface upstream; loop converts to error SSE
                    return e

            while gen is not None:
                chunk = await _await_worker(_next_chunk)
                if chunk is timed_out:
                    # codex round-2 #2: deadline hit mid-stream. Surface the
                    # timeout immediately; the still-running token step is
                    # tracked by the lease and cleaned up (lock + slot held
                    # until it stops) via ``__aexit__``'s deferred path.
                    error_message = f"DFlash stream timed out after {_format_timeout_seconds(timeout_label)}."
                    finish_reason = "length"
                    break
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    logger.exception(
                        "DFlash stream_generate raised mid-stream: %s",
                        chunk,
                        exc_info=chunk,
                    )
                    error_message = f"{type(chunk).__name__}: {chunk}"
                    # See above for OpenAI spec literal-set rationale.
                    finish_reason = "length"
                    break
                # Always sync token counts from the chunk — even when text
                # is empty (mlx-vlm occasionally emits trailing flush
                # chunks carrying the final token counters but no
                # incremental text). Skipping the update would leave the
                # final usage block with stale numbers.
                total_completion_tokens = chunk.generation_tokens
                prompt_tokens = chunk.prompt_tokens
                _ct = getattr(chunk, "token", None)
                if isinstance(_ct, int):
                    last_token_id = _ct
                if not chunk.text:
                    continue
                # F2: push to the bounded queue instead of yielding to the
                # socket directly, so the lease (and its lock) is never held
                # across a suspended socket write. ``_emit`` on a content frame
                # can raise on two timeouts, and BOTH now route through the
                # terminal notice path (codex round-3 #2 + round-4 #3) so the
                # stream ALWAYS ends with an explicit error + ``[DONE]`` rather
                # than a silent truncation:
                #   * ``_DFlashStreamDeadlineError`` — the request ``timeout``
                #     elapsed while a content frame waited for queue capacity.
                #   * ``_DFlashClientGoneError`` — the queue stayed full past
                #     the ``_STREAM_BACKPRESSURE_TIMEOUT_SECONDS`` server-side
                #     backpressure cap. This bound applies even when
                #     ``timeout=0`` ("no request deadline"), because an
                #     unbounded in-memory completion for a client that has
                #     stopped reading is itself a DoS. A client that is merely
                #     slow (not gone) still gets a terminal explanation; a truly
                #     departed client's terminal ``_emit`` will itself hit the
                #     cap and be swallowed in ``_drive_producer`` — no hang
                #     either way.
                try:
                    if postprocessor is None:
                        piece = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": served_model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk.text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        await _emit(f"data: {json.dumps(piece)}\n\n".encode())
                    else:
                        output = GenerationOutput(
                            text=chunk.text,
                            new_text=chunk.text,
                            prompt_tokens=chunk.prompt_tokens,
                            completion_tokens=chunk.generation_tokens,
                            finished=False,
                            finish_reason=None,
                        )
                        for event in postprocessor.process_chunk(output):
                            await _emit_postprocessed_event(event)
                except _DFlashStreamDeadlineError:
                    error_message = f"DFlash stream timed out after {_format_timeout_seconds(timeout_label)}."
                    finish_reason = "length"
                    break
                except _DFlashClientGoneError:
                    error_message = (
                        "DFlash stream aborted: the client did not read "
                        "buffered output within "
                        f"{_STREAM_BACKPRESSURE_TIMEOUT_SECONDS:.0f}s of "
                        "server-side backpressure."
                    )
                    finish_reason = "length"
                    break

        # Length-truncation detection — mlx-vlm's GenerationResult has no
        # ``finish_reason`` field, so we infer "length" by comparing the
        # completion token count to the budget. Only set when we exited the
        # loop normally (StopIteration), not when the generator errored or
        # produced fewer tokens (natural stop).
        #
        # Subtle case: if the model emitted EOS exactly at ``max_tokens``,
        # the stop was natural and reporting "length" would mislead clients
        # into auto-continuing (only to get an immediate EOS again). Check
        # the last token id against the resolved EOS set to keep the
        # classification honest in this edge case.
        if (
            finish_reason == "stop"
            and _max_tokens is not None
            and total_completion_tokens >= _max_tokens
            and last_token_id not in _eos_ids
        ):
            finish_reason = "length"

        if postprocessor is not None and error_message is None:
            finished_output = GenerationOutput(
                text="",
                new_text="",
                prompt_tokens=prompt_tokens,
                completion_tokens=total_completion_tokens,
                finished=True,
                finish_reason=finish_reason,
            )
            terminal_events = postprocessor.process_chunk(finished_output)
            terminal_events.extend(postprocessor.finalize())
            for event in terminal_events:
                await _emit_postprocessed_event(event, terminal=True)

        # Final chunk — finish_reason + usage. If we broke out of the loop
        # because the underlying generator raised, attach an OpenAI-style
        # error block so the client gets a readable failure instead of
        # silent truncation.
        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": served_model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": prompt_tokens + total_completion_tokens,
            },
        }
        if error_message is not None:
            final["error"] = {
                "type": "dflash_runtime_error",
                "message": error_message,
            }
        final_frame = f"data: {json.dumps(final)}\n\n".encode()
        done_frame = b"data: [DONE]\n\n"
        # codex round-4 #3: terminal frames must reach a connected client even
        # if the queue is still full from a backpressure abort. Try the queue
        # first (normal fast path); on backpressure, hand the remaining
        # terminal frames to the consumer to emit DIRECTLY to the socket after
        # it drains, rather than dropping them. A truly-departed client's
        # consumer is already cancelled, so the direct emit is a harmless
        # no-op; a slow-but-present client gets its terminator.
        #
        # codex round-5 #3: ORDER is a contract — the [DONE] terminator must
        # arrive LAST, after the final finish_reason/error frame. Once ANY
        # terminal frame spills to ``pending_terminal`` (queue full), every
        # subsequent frame MUST go there too, even if the queue drains in
        # between; otherwise a later ``[DONE]`` could be enqueued and delivered
        # ahead of the still-pending final frame. ``spilled`` latches that.
        spilled = postprocess_spilled
        for frame in (final_frame, done_frame):
            if spilled:
                pending_terminal.append(frame)
                continue
            try:
                await _emit(frame, terminal=True)
            except _DFlashClientGoneError:
                spilled = True
                pending_terminal.append(frame)

    async def _drive_producer() -> None:
        """Run ``_produce``; swallow the client-gone abort.

        codex round-3 #1: this NO LONGER relies on pushing a queue sentinel
        to terminate the consumer (a full bounded queue could drop it, hanging
        a slow-but-resuming client). The consumer instead terminates from this
        task's completion state — see the drain loop below. We still push a
        best-effort sentinel to wake a consumer that is blocked in
        ``queue.get()`` right now, but correctness no longer depends on it.
        """
        try:
            await _produce()
        except _DFlashClientGoneError:
            # codex round-2 #1/#3: the client stopped reading (bounded queue
            # stayed full past the backpressure / deadline bound). ``_produce``
            # already unwound ``async with lease`` (generator closed, lock +
            # slot released or deferred). Nothing more to emit.
            logger.debug(
                "DFlash stream aborted: client stopped reading within the "
                "backpressure/deadline bound"
            )
        finally:
            # Best-effort wake for a consumer currently blocked on
            # ``queue.get()``. If the queue is full the sentinel is dropped,
            # but the consumer's producer-completion check still terminates
            # it, so no hang. ``put_nowait`` never blocks.
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    # F2: start generation in its own task BEFORE the first client read.
    # ``ensure_future`` schedules the producer eagerly so its deadline
    # checks and lease lifecycle run independently of when (or whether) the
    # consumer pulls the next chunk. This coroutine only shuttles
    # already-formatted SSE bytes from the queue to the socket, so client
    # backpressure suspends THIS coroutine — never the lease-holding
    # producer. Creating it before the role-marker yield (rather than after)
    # also guarantees ``producer`` is defined for the ``finally`` cleanup
    # even if the client reads exactly the role marker and then disconnects.
    producer = asyncio.ensure_future(_drive_producer())
    try:
        # Role marker first — emitted before any generated chunk, matching
        # the prior contract. The producer may already be mid-generation by
        # now; its output waits in the queue behind this marker.
        yield f"data: {json.dumps(first)}\n\n".encode()
        # codex round-3 #1: drive termination from producer-completion state,
        # not solely a queue sentinel. The top-of-loop check
        # ``producer.done() and queue.empty()`` guarantees termination even if
        # the sentinel was dropped: the sentinel is only ever dropped when the
        # queue is FULL (``put_nowait`` raises only on a full queue), and a
        # full queue means ``await queue.get()`` returns immediately — it can
        # never block forever. Once the producer is done and every buffered
        # item has been yielded, ``queue.empty()`` is True and the loop exits.
        while True:
            if producer.done() and queue.empty():
                break
            item = await queue.get()
            if item is None:
                # Best-effort sentinel — just re-loop; the top-of-loop check
                # performs the actual termination decision.
                continue
            yield item
        # codex round-4 #3: emit any terminal frames the producer could not
        # push through a backpressure-full queue. We own the socket and the
        # producer has finished, so these go straight to the client — the
        # bounded queue never blocks them. A departed client's generator is
        # already closed, so this yields into a no-op teardown.
        for frame in pending_terminal:
            yield frame
        # Surface a producer crash (not one of the sentinel-wrapped
        # generation errors, which are already emitted as error SSE) so it
        # is logged rather than silently swallowed.
        await producer
    finally:
        # Client disconnect / ``aclose`` cancels this generator. Cancel the
        # producer so its ``async with lease`` unwinds and releases (or
        # defers) the serial lock + admission slot; then await it so the
        # lease cleanup actually completes before we return.
        if not producer.done():
            producer.cancel()
        try:
            await producer
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 -- lease cleanup is what matters here
            logger.debug("DFlash stream producer raised on teardown", exc_info=True)


async def _non_stream_completion(
    *,
    prompt: str,
    request: ChatCompletionRequest,
    served_model_name: str,
    gen_kwargs: dict[str, Any],
    model: Any,
    processor: Any,
    timeout: float = 1800.0,
    timeout_label: float | None = None,
    deadline: float | None = None,
    admission_reservation: _DFlashAdmissionReservation | None = None,
    enable_thinking: bool = False,
) -> ChatCompletionResponse:
    """Run generation under the serial lock and enforce a safe deadline.

    ``mlx_vlm.generate`` cannot be preempted once it is running on the
    dedicated worker. On timeout we return a 504, but keep both the serial
    lock and admission slot until the worker exits; otherwise a second call
    could overlap the first GPU operation on the same thread.

    ``deadline`` is an ABSOLUTE ``loop.time()`` instant (codex round-8 #2)
    spanning render + generation; when omitted (direct callers/tests) it is
    derived from the relative ``timeout``. ``timeout`` is the ENFORCED remaining
    budget (post prompt-render); ``timeout_label`` is the ORIGINAL configured
    request timeout reported in the 504 message (codex round-5 #6). Omitted →
    mirrors ``timeout``.
    """
    from mlx_vlm import generate

    if timeout_label is None:
        timeout_label = timeout

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # Keep the helper usable by direct programmatic callers and existing
    # focused tests. The request route always supplies its real admission
    # reservation; this private fallback only matters outside the ASGI path.
    if admission_reservation is None:
        admission_reservation = _DFlashAdmission(0).reserve()

    loop = asyncio.get_running_loop()
    # F4: unify zero-timeout policy with the streaming path. ``timeout <= 0``
    # means "no deadline" on BOTH completion paths (matches the main
    # server's ``default_timeout`` semantics and ``_stream_completion``'s
    # ``timeout and timeout > 0`` gate). Previously a 0 here built an
    # already-expired deadline and returned an immediate 504, while the
    # same value disabled the streaming deadline — identical requests
    # behaving oppositely by path.
    #
    # codex round-8 #2: prefer the absolute ``deadline`` the endpoint passed;
    # only derive from the relative ``timeout`` when a direct caller omitted it.
    if deadline is None and timeout and timeout > 0:
        deadline = loop.time() + timeout
    lock_acquired = False
    worker_future: asyncio.Future[Any] | None = None

    def _release_after_worker(_future: asyncio.Future[Any]) -> None:
        """Run in the event loop after a timed-out worker exits."""
        _dflash_lock.release()
        admission_reservation.release(force=True)

    def _defer_worker_cleanup() -> None:
        nonlocal lock_acquired
        assert worker_future is not None
        admission_reservation.defer_release()
        worker_future.add_done_callback(_release_after_worker)
        # The callback above is now responsible for the serial lock. Do not
        # release it in this coroutine's finally block.
        lock_acquired = False

    def _remaining() -> float | None:
        """Seconds left before the deadline, or ``None`` for no deadline."""
        return None if deadline is None else deadline - loop.time()

    try:
        remaining = _remaining()
        if remaining is not None and remaining <= 0:
            raise asyncio.TimeoutError
        if remaining is None:
            await _dflash_lock.acquire()
        else:
            await asyncio.wait_for(_dflash_lock.acquire(), timeout=remaining)
        lock_acquired = True

        # mlx-vlm's ``generate`` blocks; offload to the dedicated
        # single-thread DFlash executor so every mlx-vlm call lands on
        # the same worker (matches ``_stream_completion`` — see the
        # _dflash_executor comment at module top).
        #
        # Wrap in a sentinel pattern so generate-time errors (OOM, bad
        # arg, drafter mismatch) come back as a clean HTTP 500 with a
        # readable detail string rather than a raw stack trace. Mirrors
        # the stream path's error handling.
        def _generate_safely():
            try:
                return generate(model, processor, prompt, **gen_kwargs)
            except Exception as e:  # noqa: BLE001 — surface as HTTPException below
                return e

        # F5: recompute and validate the remaining deadline BEFORE
        # submitting to the executor. Acquiring the serial lock can itself
        # consume the whole budget; submitting first (as the pre-fix code
        # did) would start expensive GPU work for an already-expired
        # request and hold the lock until it finished. Check first, submit
        # only if time remains.
        remaining = _remaining()
        if remaining is not None and remaining <= 0:
            raise asyncio.TimeoutError
        worker_future = loop.run_in_executor(_dflash_executor, _generate_safely)
        if remaining is None:
            result = await asyncio.shield(worker_future)
        else:
            result = await asyncio.wait_for(
                asyncio.shield(worker_future), timeout=remaining
            )
    except asyncio.TimeoutError as exc:
        if lock_acquired and worker_future is not None:
            _defer_worker_cleanup()
        raise HTTPException(
            status_code=504,
            detail=f"DFlash request timed out after {_format_timeout_seconds(timeout_label)}.",
        ) from exc
    except asyncio.CancelledError:
        if lock_acquired and worker_future is not None:
            _defer_worker_cleanup()
        raise
    finally:
        if lock_acquired:
            _dflash_lock.release()

    if isinstance(result, Exception):
        logger.exception(
            "DFlash non-stream generate raised: %s", result, exc_info=result
        )
        raise HTTPException(
            status_code=500,
            detail=f"DFlash runtime error: {type(result).__name__}: {result}",
        )

    # OpenAI distinguishes "stop" (natural end / stop sequence) from
    # "length" (token-budget hit). mlx-vlm doesn't surface that on
    # GenerationResult, so infer from token-count vs requested budget.
    #
    # Known v1 limitation: unlike the streaming path which can read
    # ``chunk.token`` and check against EOS, ``mlx_vlm.generate``
    # returns only the concatenated text + token counts. If the model
    # emits EOS at exactly ``max_tokens`` the non-stream response will
    # still report ``finish_reason="length"`` (false truncation). A
    # client that auto-continues will issue one more request that
    # immediately returns EOS — annoying but not corrupt. Fix requires
    # an upstream mlx-vlm change to expose the final token id; tracked
    # as a v2 follow-up.
    _max_tokens = gen_kwargs.get("max_tokens")
    finish_reason = (
        "length"
        if _max_tokens is not None and result.generation_tokens >= _max_tokens
        else "stop"
    )

    content: str | None = result.text
    reasoning_content: str | None = None
    cfg = get_config()
    if cfg.reasoning_parser_name:
        from ...reasoning import get_parser

        parser_cls = get_parser(cfg.reasoning_parser_name)
        reasoning_parser = parser_cls()
        reasoning_content, parsed_content = reasoning_parser.extract_reasoning(
            result.text,
            enable_thinking=enable_thinking,
        )
        if parsed_content is not None:
            content = parsed_content
        elif reasoning_content is not None:
            content = None

    tool_calls = None
    if request.tools:
        content, tool_calls = _parse_tool_calls_with_parser(content or "", request)
        if tool_calls:
            _validate_tool_call_params(tool_calls, request.tools)
            finish_reason = "tool_calls"

    return ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=created,
        model=served_model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(
                    role="assistant",
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.generation_tokens,
            total_tokens=result.prompt_tokens + result.generation_tokens,
        ),
    )


def run_dflash_server(
    *,
    main_model_repo: str,
    drafter_repo: str,
    host: str,
    port: int,
    served_model_name: str,
    default_max_tokens: int,
    cors_origins: list[str],
    uvicorn_log_level: str,
    no_thinking: bool = False,
    api_key: str | None = None,
    rate_limit: int = 0,
    max_request_bytes: int = 8 * 1024 * 1024,
    body_receive_timeout_seconds: float = 15.0,
    default_timeout: float = 1800.0,
    max_concurrent_requests: int = 256,
    cors_policy: Any | None = None,
    tool_call_parser: str | None = "hermes",
    reasoning_parser_name: str | None = "qwen3",
) -> None:
    """Load the model + DFlash drafter via mlx-vlm and start uvicorn.

    The mlx-vlm load path is mandatory: the DFlash hooks
    (``capture_layer_ids``, ``_dflash_rounds``) live on the mlx-vlm
    model classes, not mlx-lm's. Loading via ``mlx_lm.load`` would give
    us a model without the hooks and DFlash would silently fall back to
    AR — exactly the kind of "silent regression" the eligibility gate
    is meant to prevent. We surface a clear error if mlx-vlm is missing
    or too old.

    Eligibility re-check: even though the CLI's ``serve_command`` gates
    on the alias before calling here, a *programmatic* caller (e.g. a
    notebook or test harness) can bypass the CLI entirely. We re-run
    the path-detectable gates (4-bit quant via repo-name heuristic;
    non-empty drafter). MoE detection requires the AliasProfile (an
    ``is_moe`` flag aliases.json maintains by hand) and is therefore
    only enforced via the CLI entrypoint — callers serving an
    arbitrary ``main_model_repo`` programmatically are responsible for
    not pointing it at a MoE model. Documented in CALLERS.md.
    """
    if not have_runtime():
        raise RuntimeError(
            "DFlash server requires mlx-vlm 0.5.0+ — install with "
            "pip install 'rapid-mlx[dflash]'. Homebrew installs the "
            "text-only package; switch to an isolated uv tool install "
            "for optional extras."
        )

    # Belt-and-suspenders eligibility re-check for programmatic callers
    # (the CLI's serve_command already gates on the alias upstream, but
    # we don't want to depend on it being the only entrypoint).
    from .eligibility import (
        DFlashUnavailable,
        _looks_like_4bit,  # noqa: PLC2701 — internal helper
    )

    if _looks_like_4bit(main_model_repo):
        raise DFlashUnavailable(
            f"DFlash cannot run on a 4-bit quantized model "
            f"(main_model_repo={main_model_repo!r}); upstream PoC measured "
            "regression to 0.63-0.96× on Qwen3.5-4B-MLX-4bit. Use the "
            "8-bit variant."
        )
    if not drafter_repo:
        raise DFlashUnavailable(
            "DFlash requires a non-empty drafter_repo — pass the DFlash "
            "drafter HF path (e.g. 'z-lab/Qwen3.5-27B-DFlash')."
        )

    import uvicorn
    from mlx_vlm import load

    # CRITICAL: load model + drafter on the dedicated DFlash executor
    # thread (not the main thread). mlx-lm 0.31.3+ keeps GPU streams in
    # thread-local storage, so weights loaded on thread A cannot be
    # evaluated on thread B — generate() raises ``RuntimeError: There
    # is no Stream(gpu, N) in current thread``. By pinning load AND all
    # subsequent generate() calls to the same single-worker executor,
    # streams stay reachable for the lifetime of the process.
    def _load_all():
        t0 = time.perf_counter()
        m, p = load(main_model_repo)
        logger.info("DFlash: main model loaded in %.1fs", time.perf_counter() - t0)
        rt = load_runtime(drafter_repo)
        return m, p, rt

    logger.info("DFlash: loading main model via mlx-vlm: %s", main_model_repo)
    model, processor, runtime = _dflash_executor.submit(_load_all).result()

    app = _build_app(
        model=model,
        processor=processor,
        runtime=runtime,
        served_model_name=served_model_name,
        default_max_tokens=default_max_tokens,
        cors_origins=cors_origins,
        no_thinking=no_thinking,
        api_key=api_key,
        rate_limit=rate_limit,
        max_request_bytes=max_request_bytes,
        body_receive_timeout_seconds=body_receive_timeout_seconds,
        default_timeout=default_timeout,
        max_concurrent_requests=max_concurrent_requests,
        cors_policy=cors_policy,
        tool_call_parser=tool_call_parser,
        reasoning_parser_name=reasoning_parser_name,
    )

    print()
    host_display = "localhost" if host == "0.0.0.0" else host
    print(f"  Ready: http://{host_display}:{port}/v1  (DFlash mode)")
    print(f"  Docs:  http://{host_display}:{port}/docs")
    print()

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=uvicorn_log_level,
        timeout_keep_alive=30,
    )
