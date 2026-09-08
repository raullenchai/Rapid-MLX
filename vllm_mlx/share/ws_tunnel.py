# SPDX-License-Identifier: Apache-2.0
"""Pure-Python WebSocket reverse tunnel for ``rapid-mlx share``.

Connects to the rapidserver Worker (defaults to
``wss://rapidserver.quicksilverpro.io/up``), receives HTTP requests
reverse-multiplexed over a single WebSocket, forwards them to the
local ``rapid-mlx serve`` on ``127.0.0.1``, and streams responses back.

Replaces the prior frpc binary + control-plane control flow: the
rapid-mlx process is now end-to-end Python — no external binary
download, no cross-account control-plane HTTP call, no operator-side
relay server.

Topology recap:

    rapid-mlx share (this module)  ─WSS─▶  rapidserver Worker  ◀─HTTPS─  chat frontend
                                          │  (Cloudflare edge,    │      (BCG /app/
                                          │   ``rapidserver.       │      fetch)
                                          │   quicksilverpro.io``) │
                                          ▼                        │
                                   Durable Object per id            │
                                   multiplexes inbound HTTP         │
                                   over the WS frames               │

Protocol (JSON text frames):

    worker → client:
        {"t":"req", "id":<reqId>, "method":<str>, "path":<str>,
         "headers":<obj>, "body":<base64>}
        {"t":"abort", "id":<reqId>}

    client → worker:
        {"t":"ready", "v":1}   (sent once on connect; byte-identical in
                                plain and pool mode — the pool claim
                                rides the upgrade Authorization header)
        {"t":"head", "id":<reqId>, "status":<int>, "headers":<obj>}
        {"t":"chunk", "id":<reqId>, "data":<base64>}
        {"t":"end", "id":<reqId>}
        {"t":"err", "id":<reqId>, "msg":<str>}

One WS connection multiplexes many concurrent HTTP requests via the
``id`` field. ``path`` is the path the user's local serve should see
(the ``/r/<id>`` prefix is stripped on the worker side before
forwarding into the WS frame).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import http.client
import inspect
import json
import logging
import secrets
import socket
import threading
import time
import urllib.parse
from collections.abc import Callable, Mapping
from typing import Any

try:
    import websockets
except ImportError as exc:  # pragma: no cover — declared as install dep
    raise ImportError(
        "rapid-mlx share requires the ``websockets`` package "
        "(it ships in rapid-mlx core deps; pip install websockets if missing)"
    ) from exc

log = logging.getLogger(__name__)


DEFAULT_RAPIDSERVER_WSS = "wss://rapidserver.quicksilverpro.io/up"

# Per-request fetch timeout. LLM streams legitimately run >5 min (long
# context, slow model); we still want a ceiling so a wedged local serve
# eventually surfaces as a tunnel error rather than hanging the chat
# client forever. 30 min matches the longest sustained generation we
# observed in eval suites.
LOCAL_FETCH_TIMEOUT_SECONDS = 1800

# Body-chunk size for the local fetch → WS forwarding loop. We pair this
# with ``resp.read1`` (not ``read``): on a chunked-transfer SSE stream,
# the blocking ``read(n)`` waits until either ``n`` bytes have
# accumulated or EOF, which silently batches every token of a short
# response into a single WS frame at the END of generation. ``read1``
# returns whatever has landed in the buffer right now — so each SSE
# ``data: ...\n\n`` flush forwards immediately.
CHUNK_SIZE = 4096

# A pre-registration abort marker is only meaningful within the race
# window it was created for (abort frame lands while the fetch worker
# is between connect() and registration — milliseconds). Relays are
# allowed to reuse request ids; a marker older than this cannot refer
# to the request currently registering under that id, and MUST NOT
# discard it. 5 s is orders of magnitude above any scheduling delay
# while staying far below id-reuse timescales.
_ABORT_MARKER_STALE_SECONDS = 5.0


def _hard_close(conn: http.client.HTTPConnection) -> None:
    """Close a loopback connection such that a thread blocked in
    ``getresponse()``/``read1`` on it WAKES UP.

    A bare ``conn.close()`` closes the fd but does not reliably
    interrupt a concurrent blocking ``recv`` on macOS/Darwin — the
    worker stays parked and its slot stays busy. ``shutdown()`` first
    both sends the FIN that serve's ``_disconnect_guard`` needs and
    unblocks the local reader (EOF).

    ``sock`` is captured ONCE: the fetch worker may set
    ``conn.sock = None`` (inside its own ``close()``) between the two
    reads, and calling ``shutdown`` through the re-read attribute
    would die on AttributeError in whichever thread runs the drain."""
    sock = conn.sock
    if sock is not None:
        with contextlib.suppress(OSError, AttributeError):
            sock.shutdown(socket.SHUT_RDWR)
    try:
        conn.close()
    except Exception:  # noqa: BLE001 — best-effort teardown
        pass


class OneShotHTTPConnection(http.client.HTTPConnection):
    """An ``HTTPConnection`` that a teardown ``close()`` cannot undo.

    The stock ``request()`` AUTO-RECONNECTS when ``sock`` is None — so
    an abort that lands in the window between the fetch worker's
    registration in ``_active`` and its ``conn.request()`` would have
    its cancellation silently undone: close() clears the socket, the
    aborted request's ``putrequest`` sees ``sock is None``, dials a
    fresh loopback connection, and the cancelled generation runs
    anyway. Marking the connection dead at close() makes the redial
    raise instead, exactly as if the serve had vanished."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dead = False

    def close(self) -> None:
        self.dead = True
        super().close()

    def connect(self) -> None:
        if self.dead:
            raise OSError("connection aborted before the request was sent")
        super().connect()


def new_tunnel_id() -> str:
    """Mint a fresh tunnel id. 22 urlsafe-base64 chars (128 bits) — well
    above the worker's 8-char minimum, below the 64-char ceiling.

    Charset matches what the worker validates (``[A-Za-z0-9_-]{8,64}``).
    ``token_urlsafe(16)`` returns 22 chars after b64 stripping.
    """
    return secrets.token_urlsafe(16)


def public_url_for(tunnel_id: str, relay_url: str = DEFAULT_RAPIDSERVER_WSS) -> str:
    """Derive the HTTPS reverse-proxy URL chat frontends should hit
    from the WSS relay URL + tunnel id. Used by ``share_command`` to
    build the banner before the tunnel actually comes up — keeps the
    "tunnel id is opaque to the URL building logic" contract clean.

    Maps the WSS scheme to HTTPS (``wss://`` → ``https://``,
    ``ws://`` → ``http://``) and replaces the upgrade path ``/up`` with
    the per-tunnel reverse-proxy prefix ``/r/<id>``.
    """
    parsed = urllib.parse.urlparse(relay_url)
    scheme = {"wss": "https", "ws": "http"}.get(parsed.scheme, "https")
    # Use ``netloc`` rather than rebuilding from hostname so an
    # operator-set non-default port (rare; smoke / local dev) survives.
    return f"{scheme}://{parsed.netloc}/r/{tunnel_id}"


class TunnelClient:
    """One WS connection multiplexing many concurrent HTTP requests.

    Designed to be driven by a background thread from a synchronous
    caller (``share_command``). Use ``run_in_thread`` for that.

    Lifecycle:
        connect → ready_event set → relay HTTP frames → close

    Errors during ``run`` are stashed on ``error`` so the parent can
    decide how to surface them — same shape as the prior
    ``subprocess.Popen`` model where ``returncode`` was the failure
    signal.
    """

    def __init__(
        self,
        *,
        local_port: int,
        tunnel_id: str | None = None,
        relay_url: str = DEFAULT_RAPIDSERVER_WSS,
        ready_event: threading.Event | None = None,
        share_key: str | None = None,
        override_authorization: str | None = None,
        inject_stream_usage: bool = False,
    ) -> None:
        """
        QuickSilver-pool mode knobs (all default to the historical plain
        ``share`` behaviour — a default-constructed client sends the same
        bytes and forwards the same headers it always did):

        * ``share_key`` — when set, the WS upgrade carries
          ``Authorization: Bearer <share_key>`` so a keyed relay can
          prove node ownership at handshake time, without the key ever
          touching the connect URL (and therefore never in exception
          reprs / WS logs) or any post-upgrade frame.
        * ``override_authorization`` — every forwarded request has its
          ``Authorization`` header replaced with ``Bearer <value>``.
          Pool traffic arrives bearing a QuickSilver-side credential the
          local serve has never seen; share mints its own loopback
          bearer, so the tunnel is the only place that knows both.
        * ``inject_stream_usage`` — streaming chat/completions bodies
          get ``stream_options.include_usage = true`` injected server-of-
          origin-side: the pool ledger reads ``usage`` off the final SSE
          frame, which rapid-mlx serve only emits when asked.
        """
        self.local_port = local_port
        self.tunnel_id = tunnel_id or new_tunnel_id()
        self.relay_url = relay_url
        self.ready_event = ready_event or threading.Event()
        self._share_key = share_key
        self._override_authorization = override_authorization
        self._inject_stream_usage = inject_stream_usage
        # Set by ``run`` on its event loop; used by ``_sync_send`` to
        # post messages back from per-request threads.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._send_queue: asyncio.Queue[str] | None = None
        self._closed = asyncio.Event()
        self._tasks: set[asyncio.Task[Any]] = set()
        # Populated when ``run`` exits with an exception. Cleared on
        # success (clean WS close — same as Ctrl-C on the parent).
        self.error: BaseException | None = None
        # HTTP status of a failed WS handshake (401 = keyed relay
        # rejected our share_key). Surfaced separately from ``error``
        # because the exception repr may embed the connect URI.
        self.error_status: int | None = None
        # WS close code of a cleanly-closed session (or of the
        # ConnectionClosed exception). A keyed claim's rejection is
        # normally an HTTP 401 at the upgrade (``error_status``); this
        # covers the defensive case where a relay instead accepts the
        # socket and drops it with a policy close (conventionally 1008).
        # The supervisor treats such a close as terminal like a 401.
        self.close_code: int | None = None
        # Live local fetches, keyed by tunnel request id. Populated for
        # every client (cheap bookkeeping) — the QuickSilver relay uses
        # it to cancel generations for aborted requests; the count backs
        # the pool heartbeat's ``inflight`` field.
        self._active: dict[str, http.client.HTTPConnection] = {}
        # Aborts that arrive before their req's fetch has registered in
        # ``_active`` (the fetch task only starts on the next loop
        # tick). Without this the abort is a silent no-op and the
        # generation runs to completion against nobody. Values are
        # ``time.monotonic()`` stamps, not bare membership — see
        # ``_ABORT_MARKER_STALE_SECONDS``.
        self._aborted: dict[str, float] = {}
        # Recently completed fetches (id → completion stamp, FIFO
        # capped). Membership is ABORT OWNERSHIP: the latest epoch of
        # this id is retired, so an abort arriving now was in flight
        # from a dead request — dropping it is what keeps a reused id
        # from being poisoned. In-flight epochs count in ``_pending``.
        self._completed: dict[str, float] = {}
        # Epochs dispatched but not finished (id → in-flight count),
        # incremented when a ``req`` frame is dispatched and released
        # when its ``_handle_request`` task completes. An abort for a
        # pending id belongs to that epoch even before its fetch
        # registers — ownership by state, not by timestamps.
        self._pending: dict[str, int] = {}
        self._active_lock = threading.Lock()
        # Caller-visible "tunnel died after banner" sentinel. Set when
        # the WS closes unexpectedly; the parent's monitor loop polls it.
        self.closed_event = threading.Event()

    @property
    def inflight(self) -> int:
        """Requests currently being fetched from the local serve."""
        with self._active_lock:
            return len(self._active)

    def _greeting(self) -> dict[str, Any]:
        """The first frame on every connection: always the historical
        ``{"t":"ready","v":1}``, byte-identical in plain and pool mode.
        A pool share_key is NOT carried here — the QuickSilver relay
        authenticates the tunnel claim at the WS *upgrade* (hashing the
        ``Authorization`` header) and treats this frame as a no-op, so
        the key rides the upgrade request's ``Authorization`` header
        (see ``_auth_headers``), never the URL and never this frame."""
        return {"t": "ready", "v": 1}

    def _auth_headers(self) -> dict[str, str] | None:
        """Extra HTTP headers for the WS upgrade. In pool mode the
        server-minted share_key proves node ownership as a ``Bearer``
        credential the relay hashes at upgrade time — kept out of the
        URL (which would leak into exception reprs / launchd logs) and
        out of any post-upgrade frame (the relay authenticates the
        upgrade, not a later message). ``None`` for plain share so a
        default client sends the exact upgrade request it always did."""
        if self._share_key is None:
            return None
        return {"Authorization": f"Bearer {self._share_key}"}

    @staticmethod
    def _connect_kwargs(headers: dict[str, str] | None) -> dict[str, Any]:
        """``websockets.connect`` kwargs, threading pool-auth headers
        through the parameter the installed major actually exposes: the
        asyncio client (default top-level ``connect`` in websockets
        >=14) takes ``additional_headers``; the legacy client (<=13)
        takes ``extra_headers``. Repo floor is >=12 — the same span the
        close-code extraction in ``run`` already straddles. Prefer
        ``additional_headers`` unless the signature positively shows
        only the legacy name."""
        kwargs: dict[str, Any] = {"max_size": None}
        if headers:
            try:
                params: Mapping[str, inspect.Parameter] = inspect.signature(
                    websockets.connect
                ).parameters
            except (TypeError, ValueError):
                params = {}
            header_kw = "additional_headers"
            if (
                params
                and "additional_headers" not in params
                and "extra_headers" in params
            ):
                header_kw = "extra_headers"
            kwargs[header_kw] = headers
        return kwargs

    @property
    def public_url(self) -> str:
        """Chat-frontend-facing reverse-proxy URL."""
        return public_url_for(self.tunnel_id, self.relay_url)

    # ─────────────────────────── public API ────────────────────────────

    async def run(self) -> None:
        """Connect once + serve forever. Returns when the WS closes or
        ``stop()`` is called. Raises if the initial connect fails — the
        banner must NOT print in that case.

        The relay URL includes the tunnel id as a query parameter (the
        worker validates id shape before upgrading), so a hostile load
        balancer can't strip the upgrade context.
        """
        self._loop = asyncio.get_running_loop()
        self._send_queue = asyncio.Queue()
        # urlencoded, not f-string-interpolated: a node id containing
        # &/# would otherwise inject extra query params (or a fragment)
        # into the relay request instead of staying one ``id`` value.
        uri = self.relay_url + "?" + urllib.parse.urlencode({"id": self.tunnel_id})
        ws = None
        try:
            # ``max_size=None`` removes the 1-MiB frame ceiling so a
            # legitimately-large chat prompt doesn't get dropped. The
            # base64 inflation of multimodal prompts (images encoded
            # in user messages) easily reaches several MiB on modern
            # VLM apps. The connect is CANCELLABLE: a handshake that
            # stalls past the supervisor's ready-window must die with
            # ``stop()`` instead of surfacing minutes later as a second
            # live tunnel the supervisor no longer tracks.
            ws = await self._connect_cancellable(uri, self._auth_headers())
            # Plain protocol greeting only. The pool share_key (if any)
            # was already proven on the upgrade's Authorization header —
            # never the URL, never this frame — so a rejected/revoked
            # key fails the handshake as an HTTP 401 (surfaced via
            # ``error_status``), not as a post-upgrade close.
            await ws.send(json.dumps(self._greeting()))
            self.ready_event.set()
            sender = asyncio.create_task(self._sender_loop(ws))
            try:
                async for raw in ws:
                    if not isinstance(raw, str):
                        # Binary frames aren't part of the protocol;
                        # silently drop so a buggy peer can't crash
                        # the loop.
                        continue
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    self._dispatch_inbound(msg)
            finally:
                sender.cancel()
                self._closed.set()
                for t in list(self._tasks):
                    t.cancel()
            # Clean peer close: surface the close code the relay
            # chose. Primary key rejection is an HTTP 401 at the upgrade
            # (below), but a relay that instead drops the claim after
            # accepting the socket signals it here — capture it so the
            # supervisor's terminal check can act on it.
            self.close_code = getattr(ws, "close_code", None)
        except Exception as exc:
            self.error = exc
            # ConnectionClosed family carries the close code, but the
            # attribute shape depends on the websockets major (repo
            # floor is >=12): modern exposes it on ``rcvd`` and
            # deprecates ``.code``; legacy exposes ``.code`` directly.
            # Fall through rcvd → code → ws.close_code so a revoked-key
            # 1008 can never read as "unknown close" and feed the
            # reconnect loop.
            code: Any = None
            rcvd = getattr(exc, "rcvd", None)
            if rcvd is not None:
                code = getattr(rcvd, "code", None)
            elif not hasattr(exc, "rcvd"):
                code = getattr(exc, "code", None)
            if not isinstance(code, int) and ws is not None:
                code = getattr(ws, "close_code", None)
            if isinstance(code, int):
                self.close_code = code
            # A rejected keyed claim surfaces as an HTTP 401 during the
            # WS handshake. Record the status separately: the caller
            # must not pattern-match on ``str(exc)`` — for keyed
            # clients the repr can embed the connect URI. websockets
            # exposes it two ways across versions: directly on the
            # exception (legacy InvalidStatusCode) and on
            # ``exc.response`` (modern InvalidStatus) — read both.
            status = getattr(exc, "status_code", None)
            if not isinstance(status, int):
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
            if isinstance(status, int):
                self.error_status = status
            raise
        finally:
            # Give cancelled req tasks a beat to unwind BEFORE the
            # drain — then the drain itself refuses-to-see any late
            # registration (below in _perform_local_fetch), because a
            # queued to_thread worker can still be between task-start
            # and registry-insert no matter how long we await here.
            if self._tasks:
                with contextlib.suppress(Exception):
                    await asyncio.wait(list(self._tasks), timeout=2.0)
            # Drain in-flight loopback connections BEFORE declaring the
            # tunnel closed: cancelling the asyncio wrappers does not
            # stop the ``to_thread`` workers blocked on serve — without
            # this their generations keep running (and serving stale
            # responses into a dead WS) after the tunnel is gone.
            self._drain_active()
            if ws is not None:
                with contextlib.suppress(Exception):
                    await ws.close()
            self.closed_event.set()

    def _drain_active(self) -> None:
        with self._active_lock:
            conns = list(self._active.values())
            self._active.clear()
        for conn in conns:
            _hard_close(conn)

    async def _connect_cancellable(
        self, uri: str, headers: dict[str, str] | None = None
    ) -> Any:
        """``websockets.connect`` awaited, but aborted if ``stop()``
        lands while the handshake is still in flight. Without this the
        connect task is uncancellable and a timed-out attempt can
        connect AFTER the supervisor abandoned it — an untracked tunnel
        proxying requests with a stale credential state."""
        connect_task = asyncio.ensure_future(
            websockets.connect(uri, **self._connect_kwargs(headers))
        )
        closed_task = asyncio.ensure_future(self._closed.wait())
        try:
            done, _ = await asyncio.wait(
                {connect_task, closed_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if closed_task in done:
                connect_task.cancel()
                abandoned = None
                with contextlib.suppress(BaseException):
                    await connect_task
                    abandoned = connect_task.result()
                if abandoned is not None:
                    # The handshake completed in the same tick stop()
                    # landed — close it instead of orphaning a live
                    # socket behind a dead run().
                    with contextlib.suppress(Exception):
                        await abandoned.close()
                raise ConnectionError("share tunnel stopped during connect")
            return connect_task.result()
        finally:
            closed_task.cancel()
            if not connect_task.done():
                connect_task.cancel()

    def run_in_thread(self) -> threading.Thread:
        """Run the asyncio loop in a dedicated thread. Returns the
        thread; caller can ``thread.join()`` for cleanup. The
        ``ready_event`` fires the moment the WS handshake completes
        (and the protocol greeting is sent) — block-wait on that
        before printing the banner.
        """

        def _entry() -> None:
            try:
                asyncio.run(self.run())
            except Exception as exc:
                # ``run`` already stashed it on ``self.error``; the
                # outer ``asyncio.run`` would otherwise re-raise into
                # the thread's unhandled-exception sink and print a
                # traceback during clean Ctrl-C shutdowns.
                if self.error is None:
                    self.error = exc

        t = threading.Thread(
            target=_entry,
            name="rapid-mlx-share-ws-tunnel",
            daemon=True,
        )
        t.start()
        return t

    def stop(self) -> None:
        """Signal a graceful shutdown. Idempotent. Safe to call from
        any thread — the actual WS close happens on the asyncio loop.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._closed.set)
        except RuntimeError:
            # Loop already shut down by its own ``finally`` — fine.
            pass

    # ───────────────────────── private machinery ───────────────────────

    async def _sender_loop(self, ws: Any) -> None:
        """Drain the ``_send_queue`` into the live WS. A dedicated task
        keeps the per-request fan-in straightforward (per-request
        threads enqueue via ``_sync_send``; this one task serializes
        the writes to satisfy the websockets-library single-writer
        contract).
        """
        assert self._send_queue is not None
        while True:
            msg = await self._send_queue.get()
            try:
                await ws.send(msg)
            except Exception:
                # WS died mid-send; the outer ``async for`` will see
                # the close on its next iteration. Drop quietly.
                return

    def _release_epoch(self, req_id: str) -> None:
        """One epoch of ``req_id`` finished. Keep the id in ``_pending``
        while other epochs of the same id are still in flight — a
        blanket pop would make an abort for the survivor look like a
        never-seen id (and a completion marker for the finished one
        would then be the only thing standing between a late abort and
        poisoning the survivor)."""
        remaining = self._pending.get(req_id, 1) - 1
        if remaining > 0:
            self._pending[req_id] = remaining
        else:
            self._pending.pop(req_id, None)

    def _dispatch_inbound(self, msg: dict[str, Any]) -> None:
        t = msg.get("t")
        if t == "req":
            req_id = msg.get("id")
            if isinstance(req_id, str):
                # Epoch opens at DISPATCH, on the loop thread that also
                # reads aborts: from here until _release_epoch, an abort
                # naming this id is unambiguously for THIS request even
                # if its fetch worker has not reached the registry yet.
                # No window in which a legitimate abort can be mistaken
                # for a late one (or vice versa).
                with self._active_lock:
                    self._pending[req_id] = self._pending.get(req_id, 0) + 1
            task = asyncio.create_task(self._handle_request(msg))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        elif t == "abort":
            # The pool relay sends this when the downstream client
            # disconnected or the request timed out before headers.
            # The tunnel↔serve connection is OURS — a remote disconnect
            # never propagates to 127.0.0.1 on its own — so we must
            # close the local socket explicitly. serve's
            # ``_disconnect_guard`` sees the break and force-aborts the
            # scheduler request, freeing the slot. (The pre-pool code
            # assumed a TCP RST would arrive on its own; it does not.)
            req_id = msg.get("id")
            if isinstance(req_id, str):
                with self._active_lock:
                    conn = self._active.get(req_id)
                    if conn is not None:
                        # Epoch live AND registered: close the local
                        # socket, which is how cancellation reaches
                        # serve at all (see comment above).
                        pass
                    elif self._pending.get(req_id):
                        # Epoch in flight but its fetch worker has not
                        # reached the registry yet — ownership is settled
                        # by the dispatch-side counter, so this abort is
                        # unambiguously for the live epoch. Mark it; the
                        # fetch pops the marker and skips instead of
                        # starting a generation nobody listens for.
                        self._mark_aborted(req_id)
                    elif req_id in self._completed:
                        # The id's last epoch is RETIRED. This abort was
                        # in flight from a request that is already gone;
                        # nothing left to cancel, and recording it would
                        # discard the NEXT request that reuses the id.
                        pass
                    else:
                        # Never-seen id: the req frame is still behind
                        # this abort on the wire. Mark it, same as the
                        # in-flight case.
                        self._mark_aborted(req_id)
                if conn is not None:
                    _hard_close(conn)

    def _mark_aborted(self, req_id: str) -> None:
        # Caller holds ``_active_lock``. Timestamped so a fetch that
        # arrives long after the race window (relay id reuse) can tell a
        # live cancellation from a stale marker — see
        # ``_ABORT_MARKER_STALE_SECONDS``. Capped so multi-day nodes
        # cannot creep on stray aborts; a needed marker only ever lives
        # one loop tick.
        if len(self._aborted) >= 4096:
            # FIFO — dict preserves insertion order, so this evicts the
            # oldest marker.
            self._aborted.pop(next(iter(self._aborted)))
        self._aborted[req_id] = time.monotonic()

    async def _handle_request(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("id")
        if not isinstance(req_id, str):
            return
        # Epoch opened at DISPATCH (see _dispatch_inbound); this task
        # owns its release no matter which path out it takes — the
        # finally covers the err-returns and fetch failures alike.
        try:
            await self._serve_request(msg, req_id)
        finally:
            with self._active_lock:
                self._release_epoch(req_id)

    async def _serve_request(self, msg: dict[str, Any], req_id: str) -> None:
        method = msg.get("method", "GET")
        path = msg.get("path", "/")
        headers: dict[str, str] = msg.get("headers") or {}
        body_b64: str = msg.get("body") or ""
        try:
            body = base64.b64decode(body_b64) if body_b64 else b""
        except (ValueError, TypeError) as exc:
            await self._send(
                {"t": "err", "id": req_id, "msg": f"bad body encoding: {exc}"}
            )
            return

        if self._override_authorization is not None:
            # Pool traffic bears a QuickSilver credential our loopback
            # serve has never seen — ANY method (GET /v1/models, DELETE
            # /v1/requests/…) must arrive with the bearer the serve
            # actually minted, or serve's auth gate 401s it.
            # Case-insensitive replace: the relay may forward any
            # casing of the inbound header name.
            headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
            headers["Authorization"] = f"Bearer {self._override_authorization}"

        if self._inject_stream_usage and method == "POST":
            new_body = _inject_stream_options_usage(path, body)
            if new_body is not body:
                body = new_body
                # The relay forwarded a Content-Length matching the
                # original bytes; http.client recomputes it for a bytes
                # body, so drop the stale value instead of shipping a
                # header/body mismatch.
                headers = {
                    k: v for k, v in headers.items() if k.lower() != "content-length"
                }

        # ``http.client`` is synchronous; run it in a worker thread so
        # the asyncio loop stays responsive while the response streams
        # back from the local serve.
        try:
            await asyncio.to_thread(
                self._perform_local_fetch, req_id, method, path, headers, body
            )
        except Exception as exc:  # noqa: BLE001 — surfaced to the chat client
            await self._send({"t": "err", "id": req_id, "msg": str(exc)[:200]})

    def _perform_local_fetch(
        self,
        req_id: str,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        """Sync fetch + chunked WS forwarding. Runs in ``to_thread``."""
        conn = OneShotHTTPConnection(
            "127.0.0.1", self.local_port, timeout=LOCAL_FETCH_TIMEOUT_SECONDS
        )
        with self._active_lock:
            # ``_closed`` is set by every loop-exit path BEFORE the
            # drain runs, so a worker scheduled just before teardown
            # that only NOW reaches registration refuses to register:
            # the drain's snapshot would miss it and the generation
            # would serve into a dead tunnel. (asyncio.Event.is_set()
            # only reads a flag — safe from this worker thread.)
            if self._closed.is_set():
                conn.close()
                return
            marked_at = self._aborted.pop(req_id, None)
            if marked_at is not None:
                if time.monotonic() - marked_at <= _ABORT_MARKER_STALE_SECONDS:
                    # The relay cancelled before we ever registered — the
                    # downstream is gone; starting the generation would
                    # burn a pool slot for nobody. Skip without sending.
                    conn.close()
                    return
                # Else: a stale marker from an abort of a PREVIOUS
                # request that reused this id — the relay's counter
                # wrapped, not our cancellation. Serve normally.
            if req_id in self._active:
                # A live generation already owns this id (relay sent a
                # duplicate while the first is in flight). Overwriting
                # would orphan the first: unabortable, and the teardown
                # drain would miss it. Refuse the NEWCOMER instead.
                conn.close()
                return
            self._active[req_id] = conn
        try:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            self._sync_send(
                {
                    "t": "head",
                    "id": req_id,
                    "status": resp.status,
                    "headers": dict(resp.getheaders()),
                }
            )
            while True:
                # ``read1`` (not ``read``) — return whatever's currently
                # buffered without waiting to fill CHUNK_SIZE. See the
                # CHUNK_SIZE comment above for why; perf impact is the
                # entire SSE streaming UX through the tunnel.
                chunk = resp.read1(CHUNK_SIZE)
                if not chunk:
                    break
                self._sync_send(
                    {
                        "t": "chunk",
                        "id": req_id,
                        "data": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            self._sync_send({"t": "end", "id": req_id})
        finally:
            with self._active_lock:
                # Only drop OUR registration — an abort already closed
                # the conn, and a retried request id must not evict a
                # newer connection. Discard any late abort marker in
                # the same breath: after this point there is nothing
                # left for it to cancel (bounded memory).
                if self._active.get(req_id) is conn:
                    del self._active[req_id]
                self._aborted.pop(req_id, None)
                if len(self._completed) >= 4096:
                    self._completed.pop(next(iter(self._completed)))
                self._completed[req_id] = time.monotonic()
            conn.close()

    async def _send(self, obj: Any) -> None:
        if self._send_queue is None:
            return
        await self._send_queue.put(json.dumps(obj))

    def _sync_send(self, obj: Any) -> None:
        """Enqueue from a worker thread. Schedules the put on the
        asyncio loop the queue belongs to — ``asyncio.Queue`` is not
        itself thread-safe.
        """
        loop = self._loop
        q = self._send_queue
        if loop is None or q is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(q.put_nowait, json.dumps(obj))
        except RuntimeError:
            # Loop already torn down (e.g. share parent shutting down
            # mid-request). Drop quietly — the chat-side stream will
            # see a tunnel error.
            pass


_USAGE_INJECTION_PATHS = frozenset({"/v1/chat/completions", "/v1/completions"})


def _inject_stream_options_usage(path: str, body: bytes) -> bytes:
    """Force ``stream_options.include_usage=true`` on streaming request
    bodies so serve stamps ``usage`` on the final SSE frame (the pool
    ledger's primary billing signal). Non-streaming bodies are left
    byte-identical (their responses already carry usage); bodies that
    don't parse as a JSON object are passed through untouched — a
    malformed body is serve's problem to report, not ours to rewrite.
    """
    root = path.split("?", 1)[0].rstrip("/")
    if root not in _USAGE_INJECTION_PATHS or not body:
        return body
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict) or not payload.get("stream"):
        return body
    options = payload.get("stream_options")
    if not isinstance(options, dict):
        options = {}
    if options.get("include_usage") is True:
        return body  # already asked for; don't re-encode
    options["include_usage"] = True
    payload["stream_options"] = options
    return json.dumps(payload).encode("utf-8")


def wait_for_public_url(
    public_url: str,
    bearer: str,
    *,
    timeout: float = 30.0,
    log_fn: Callable[[str], None] = log.debug,
) -> bool:
    """Probe ``<public_url>/v1/models`` to confirm the relay is reachable
    and the local serve is answering through the tunnel.

    The check rides on top of the WS tunnel we just opened, so a passing
    probe transitively proves:
        1. The WS is connected and the worker is forwarding inbound.
        2. The local serve is healthy and bearer-auth is wired up.

    Used by ``share_command`` after the tunnel reports ready but
    before printing the banner — same role as the prior frpc-era
    ``_wait_for_public_url``.
    """
    import urllib.error
    import urllib.request

    url = public_url.rstrip("/") + "/v1/models"
    deadline = time.monotonic() + timeout
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "rapid-mlx-share",
            "Authorization": f"Bearer {bearer}",
        },
    )
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310
                if r.status == 200:
                    return True
        except urllib.error.HTTPError as exc:
            # 503 = DO has no WS attached yet (race against tunnel
            # ready). 401 = auth not enforced — shouldn't happen, but
            # if it does we shouldn't open the share to the public.
            log_fn(f"probe got HTTP {exc.code}")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            # ``TimeoutError`` is raised bare by ``urlopen`` on
            # connect-then-stall (not a ``URLError`` subclass since
            # 3.10). Same gotcha the prior frpc-era helper hit; see
            # ``vllm_mlx/share/cli.py`` for the full rationale.
            pass
        time.sleep(1)
    return False
