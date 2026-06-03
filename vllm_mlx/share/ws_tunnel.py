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

# Body-chunk size for the local fetch → WS forwarding loop. Smaller →
# lower latency per SSE event but more JSON+base64 overhead. 4 KiB is
# the same default ``http.client.HTTPResponse.read`` uses internally,
# matching real-world MLX serve flush boundaries (one SSE event ≈ a few
# hundred bytes).
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
    ) -> None:
        self.local_port = local_port
        self.tunnel_id = tunnel_id or new_tunnel_id()
        self.relay_url = relay_url
        self.ready_event = ready_event or threading.Event()
        # Set by ``run`` on its event loop; used by ``_sync_send`` to
        # post messages back from per-request threads.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._send_queue: asyncio.Queue[str] | None = None
        self._closed = asyncio.Event()
        self._tasks: set[asyncio.Task[Any]] = set()
        # Populated when ``run`` exits with an exception. Cleared on
        # success (clean WS close — same as Ctrl-C on the parent).
        self.error: BaseException | None = None
        # Caller-visible "tunnel died after banner" sentinel. Set when
        # the WS closes unexpectedly; the parent's monitor loop polls it.
        self.closed_event = threading.Event()

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
                await ws.send(json.dumps({"t": "ready", "v": 1}))
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
            # We trust the worker to have already cancelled the inbound
            # HTTP request stream; the corresponding ``to_thread`` will
            # finish on its own (the local serve sees a TCP RST when
            # the response is dropped). No bookkeeping needed.
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
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.local_port, timeout=LOCAL_FETCH_TIMEOUT_SECONDS
        )
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
                chunk = resp.read(CHUNK_SIZE)
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
