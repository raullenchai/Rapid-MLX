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
        {"t":"ready", "v":1}                              (sent once on connect)
        {"t":"ready", "v":1, "key":"qspsk-…"}   (pool mode: keyed claim)
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
import http.client
import json
import logging
import secrets
import threading
import time
import urllib.parse
from collections.abc import Callable
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

        * ``share_key`` — when set, the greeting frame becomes
          ``{"t":"ready","v":1,"key":<share_key>}`` so a keyed relay can
          prove node ownership without the key ever touching the
          connect URL (and therefore never in exception reprs / WS logs).
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
        # Live local fetches, keyed by tunnel request id. Populated for
        # every client (cheap bookkeeping) — the QuickSilver relay uses
        # it to cancel generations for aborted requests; the count backs
        # the pool heartbeat's ``inflight`` field.
        self._active: dict[str, http.client.HTTPConnection] = {}
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
        """The first frame on every connection. Byte-identical to the
        historical ``{"t":"ready","v":1}`` unless a pool share_key is
        set (method, not inline, so tests can pin the plain-share
        byte-compat claim directly)."""
        greeting: dict[str, Any] = {"t": "ready", "v": 1}
        if self._share_key is not None:
            greeting["key"] = self._share_key
        return greeting

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
        uri = f"{self.relay_url}?id={self.tunnel_id}"
        try:
            # ``max_size=None`` removes the 1-MiB frame ceiling so a
            # legitimately-large chat prompt doesn't get dropped. The
            # base64 inflation of multimodal prompts (images encoded
            # in user messages) easily reaches several MiB on modern
            # VLM apps.
            async with websockets.connect(uri, max_size=None) as ws:
                # Keyed claim (QuickSilver pool): the share_key rides
                # the first frame, never the URL — keeping it out of
                # exception reprs and websockets-library logs.
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
        except Exception as exc:
            self.error = exc
            # A rejected keyed claim surfaces as an HTTP 401 during the
            # WS handshake. Record the status separately: the caller
            # must not pattern-match on ``str(exc)`` — for keyed
            # clients the repr can embed the connect URI.
            status = getattr(exc, "status_code", None)
            if isinstance(status, int):
                self.error_status = status
            raise
        finally:
            self.closed_event.set()

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

    def _dispatch_inbound(self, msg: dict[str, Any]) -> None:
        t = msg.get("t")
        if t == "req":
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
                    try:
                        conn.close()
                    except OSError:
                        pass

    async def _handle_request(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("id")
        if not isinstance(req_id, str):
            return
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

        if self._override_authorization is not None and method == "POST":
            # Pool traffic bears a QuickSilver credential our loopback
            # serve has never seen. Swap in the bearer share minted for
            # THIS serve (case-insensitive replace — the relay may
            # forward any casing of the inbound header name).
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

    def _perform_local_fetch(
        self,
        req_id: str,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        """Sync fetch + chunked WS forwarding. Runs in ``to_thread``."""
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.local_port, timeout=LOCAL_FETCH_TIMEOUT_SECONDS
        )
        with self._active_lock:
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
                # newer connection.
                if self._active.get(req_id) is conn:
                    del self._active[req_id]
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
