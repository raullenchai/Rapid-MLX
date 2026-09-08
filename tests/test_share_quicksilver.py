# SPDX-License-Identifier: Apache-2.0
"""QuickSilver compute-pool share mode — provider-spec acceptance tests.

Maps §8 of the provider spec (client acceptance checklist) onto the
pieces that are testable without a live pay.*/relay: registration +
cache lifecycle, credential hygiene, the keyed tunnel client knobs
(ready-frame claim, abort → cancel, usage injection), heartbeat
taxonomy, and the service-install refusal without cache. The parts
that genuinely need the server (§8 bullets 4-5 end-to-end) are covered
here at the wire-frame level instead.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
import plistlib
import stat
import subprocess
import threading
import urllib.error
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.share import cli as share_cli
from vllm_mlx.share import quicksilver as qs
from vllm_mlx.share import ws_tunnel

SHARE_KEY = "qspsk-" + "k" * 32
PROVIDER_KEY = "qsppk-" + "p" * 32


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """HOME (quicksilver cache + LaunchAgents) and the share state dir
    both point at tmp_path; the process-local secret registry starts
    and stays empty per test."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(qs.PROVIDER_KEY_ENV_VAR, raising=False)
    state = tmp_path / "share-state"
    state.mkdir()
    monkeypatch.setattr(share_cli, "_state_dir", lambda: state)
    qs._SECRETS.clear()
    yield tmp_path
    qs._SECRETS.clear()


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model="qwen3.6-35b",
        _original_alias=None,
        port=18765,
        thinking=False,
        cors_origins=None,
        rate_limit=None,
        chat_frontend=None,
        quicksilver=True,
        provider_key=None,
        quicksilver_model=None,
        worker="test-worker",
        reregister=False,
        install_service=False,
        quicksilver_api=None,
        _passthrough=[],
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@contextlib.contextmanager
def _enter(*ctxs):
    """Enter N context managers in one statement (tests here stack 9-11
    patches; indexing a tuple instead reads like a footgun)."""
    with contextlib.ExitStack() as stack:
        for c in ctxs:
            stack.enter_context(c)
        yield


def _http_error(code: int, body: bytes = b"{}") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://pay.test",
        code,
        "err",
        {},
        io.BytesIO(body),  # type: ignore[arg-type]
    )


class _FakeResp:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._data = json.dumps(payload).encode()
        self.status = status

    def read(self, size: int = -1) -> bytes:
        # honour the size hint like a real fp — _bounded_read caps it
        if size is None or size < 0:
            return self._data
        return self._data[:size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _register_payload(**over) -> dict:
    payload = {
        "node_id": "qspnode-1a2b3c4d",
        "share_key": SHARE_KEY,
        "model": "qwen3.6-35b",
        "worker": "test-worker",
        "relay_url": "wss://rapidserver.quicksilverpro.io/up",
        "pool_base": "https://rapidserver.quicksilverpro.io/pool/v1",
        "heartbeat_url": "https://rapidserver.quicksilverpro.io/hb",
        "heartbeat_interval_s": 10,
        "payout_account": "a***@x***",
    }
    payload.update(over)
    return payload


# ─────────────────────── ws_tunnel: keyed-claim knobs ───────────────────────


def test_tunnel_client_class_shape_intact():
    """The pool-mode edits inserted a module-level function mid-class
    once, silently orphaning ``_perform_local_fetch``/``_send``/
    ``_sync_send`` into dead code (unit stubs hid it). Pin that the
    request-path methods are real class members, not instance attrs."""
    for name in ("_perform_local_fetch", "_send", "_sync_send", "_dispatch_inbound"):
        assert callable(getattr(ws_tunnel.TunnelClient, name, None)), name


def test_greeting_plain_is_byte_identical_to_history():
    client = ws_tunnel.TunnelClient(local_port=1)
    assert json.dumps(client._greeting()) == '{"t": "ready", "v": 1}'


def test_greeting_is_plain_even_in_pool_mode():
    # The share_key does NOT ride the greeting frame — the relay
    # authenticates the WS upgrade, not a later message, so the frame
    # stays byte-identical to plain share in both modes.
    client = ws_tunnel.TunnelClient(local_port=1, share_key=SHARE_KEY)
    assert json.dumps(client._greeting()) == '{"t": "ready", "v": 1}'
    assert SHARE_KEY not in client.public_url


def test_share_key_rides_the_upgrade_authorization_header_not_url():
    # §6: the key must reach the relay as a Bearer header on the WS
    # upgrade — never in the URL (which leaks into exception reprs /
    # launchd logs) and never in a post-upgrade frame.
    seen: list = []

    class _FakeWS:
        close_code = None

        async def send(self, msg):
            # The greeting that went out must carry no key.
            assert "key" not in json.loads(msg)

        async def close(self):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    async def fake_connect(uri, **kw):
        seen.append((uri, kw))
        return _FakeWS()

    client = ws_tunnel.TunnelClient(
        local_port=1, relay_url="wss://r.test/up", share_key=SHARE_KEY
    )
    with patch.object(ws_tunnel.websockets, "connect", fake_connect):
        asyncio.run(client.run())
    assert len(seen) == 1
    uri, kw = seen[0]
    assert SHARE_KEY not in uri
    # Installed websockets (asyncio client) takes ``additional_headers``;
    # ``_connect_kwargs`` falls back to ``extra_headers`` only on a
    # legacy major. Accept whichever the client selected.
    hdrs = kw.get("additional_headers") or kw.get("extra_headers")
    assert hdrs == {"Authorization": f"Bearer {SHARE_KEY}"}


def test_abort_frame_closes_the_registered_connection():
    client = ws_tunnel.TunnelClient(local_port=1)
    conn = MagicMock()
    client._active["req-1"] = conn
    client._dispatch_inbound({"t": "abort", "id": "req-1"})
    conn.close.assert_called_once()


def test_abort_unknown_id_is_harmless():
    client = ws_tunnel.TunnelClient(local_port=1)
    client._dispatch_inbound({"t": "abort", "id": "ghost"})  # must not raise


def test_inflight_counts_active_fetches():
    client = ws_tunnel.TunnelClient(local_port=1)
    assert client.inflight == 0
    client._active["a"] = MagicMock()
    client._active["b"] = MagicMock()
    assert client.inflight == 2


def _run_handle_request(client: ws_tunnel.TunnelClient, msg: dict) -> dict:
    captured: dict = {}

    def _fake_fetch(req_id, method, path, headers, body):
        captured.update(
            req_id=req_id, method=method, path=path, headers=headers, body=body
        )

    client._perform_local_fetch = _fake_fetch  # type: ignore[method-assign]
    asyncio.run(client._handle_request(msg))
    return captured


def _req_msg(body_obj: dict | None = None, **over) -> dict:
    body = b"" if body_obj is None else json.dumps(body_obj).encode()
    import base64

    msg = {
        "t": "req",
        "id": "r1",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": {"Content-Type": "application/json", "X-Keep": "1"},
        "body": base64.b64encode(body).decode(),
    }
    msg.update(over)
    return msg


def test_plain_client_forwards_headers_and_body_untouched():
    client = ws_tunnel.TunnelClient(local_port=1)
    got = _run_handle_request(client, _req_msg({"stream": True}))
    assert got["headers"] == {"Content-Type": "application/json", "X-Keep": "1"}
    assert json.loads(got["body"]) == {"stream": True}


def test_override_authorization_replaces_any_inbound_auth_header():
    client = ws_tunnel.TunnelClient(local_port=1, override_authorization="local-bearer")
    msg = _req_msg({"stream": True})
    msg["headers"]["authorization"] = "Bearer qsp-some-pool-key"
    got = _run_handle_request(client, msg)
    lowered = {k.lower(): v for k, v in got["headers"].items()}
    assert lowered["authorization"] == "Bearer local-bearer"
    assert "qsp-some-pool-key" not in json.dumps(got["headers"])
    assert lowered["content-type"] == "application/json"  # others survive


def test_usage_injected_for_streaming_and_content_length_stripped():
    client = ws_tunnel.TunnelClient(local_port=1, inject_stream_usage=True)
    msg = _req_msg({"model": "m", "stream": True})
    msg["headers"]["Content-Length"] = "1234"  # stale after injection
    got = _run_handle_request(client, msg)
    assert json.loads(got["body"])["stream_options"]["include_usage"] is True
    assert "content-length" not in {k.lower() for k in got["headers"]}


def test_usage_not_injected_for_non_stream_or_other_paths():
    client = ws_tunnel.TunnelClient(local_port=1, inject_stream_usage=True)
    got = _run_handle_request(client, _req_msg({"model": "m", "stream": False}))
    assert "stream_options" not in json.loads(got["body"])
    got = _run_handle_request(client, _req_msg({"input": "x"}, path="/v1/embeddings"))
    assert json.loads(got["body"]) == {"input": "x"}


def test_usage_injection_never_corrupts_unparsable_bodies():
    client = ws_tunnel.TunnelClient(local_port=1, inject_stream_usage=True)
    import base64

    raw = b"not-json"
    msg = _req_msg()
    msg["body"] = base64.b64encode(raw).decode()
    got = _run_handle_request(client, msg)
    assert got["body"] == raw


def _real_invalid_status(code: int):
    """Build the exception the INSTALLED websockets raises on a
    rejected handshake — the library moved the status from
    ``exc.status_code`` to ``exc.response.status_code`` and a synthetic
    exception with the old shape would pass while the real 401 path
    stayed broken."""
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    return InvalidStatus(Response(code, "Unauthorized", Headers(), b""))


def test_error_status_captured_from_real_invalidstatus():
    exc = _real_invalid_status(401)
    assert not hasattr(exc, "status_code")  # guards the test itself

    client = ws_tunnel.TunnelClient(local_port=1)
    with (
        patch.object(ws_tunnel.websockets, "connect", side_effect=exc),
        pytest.raises(type(exc)),
    ):
        asyncio.run(client.run())
    assert client.error_status == 401


def test_error_status_captured_from_legacy_status_code_shape():
    class _RejectedError(Exception):
        status_code = 401

    client = ws_tunnel.TunnelClient(local_port=1)
    with (
        patch.object(
            ws_tunnel.websockets, "connect", side_effect=_RejectedError("rejected")
        ),
        pytest.raises(_RejectedError),
    ):
        asyncio.run(client.run())
    assert client.error_status == 401


def test_stop_during_connect_kills_the_pending_handshake():
    """A connect that hangs past the supervisor's ready-window must die
    with ``stop()`` — otherwise it surfaces minutes later as an
    untracked second tunnel proxying on stale state."""
    import threading
    import time as _time

    pending = asyncio.Event()  # never set — the handshake "hangs"

    async def fake_connect(uri, **kw):
        await pending.wait()
        raise AssertionError("connect must have been cancelled")

    client = ws_tunnel.TunnelClient(local_port=1, share_key="qspsk-x")
    outcome: list[BaseException | None] = []

    def run_and_expect_error():
        try:
            with patch.object(ws_tunnel.websockets, "connect", fake_connect):
                asyncio.run(client.run())
        except BaseException as exc:  # noqa: BLE001 — inspected below
            outcome.append(exc)

    t = threading.Thread(target=run_and_expect_error)
    t.start()
    deadline = _time.monotonic() + 2
    while client._loop is None and _time.monotonic() < deadline:
        _time.sleep(0.01)
    client.stop()
    t.join(timeout=5)
    assert not t.is_alive(), "run() outlived stop() during connect"
    assert outcome and isinstance(outcome[0], ConnectionError)
    assert "stopped during connect" in str(client.error)
    assert client.closed_event.is_set()


def test_one_shot_connection_close_poisons_redial():
    """Stock HTTPConnection.request() silently RE-DIALS when close()
    nulled the socket — an abort landing between registration and
    request would otherwise have its cancellation undone."""
    conn = ws_tunnel.OneShotHTTPConnection("127.0.0.1", 1)
    conn.close()
    assert conn.dead is True
    with pytest.raises(OSError, match="aborted"):
        conn.connect()


def test_abort_via_dispatch_poisons_connection():
    client = ws_tunnel.TunnelClient(local_port=1)
    conn = ws_tunnel.OneShotHTTPConnection("127.0.0.1", 1)
    client._active["req-1"] = conn
    client._dispatch_inbound({"t": "abort", "id": "req-1"})
    assert conn.dead is True


def test_hard_close_survives_concurrent_sock_none():
    """The fetch worker may clear conn.sock between the two attribute
    reads of the naive pattern — the drain must read once."""

    class _RaceConn:
        def __init__(self) -> None:
            self._sock = MagicMock()
            self._reads = 0
            self.closed = False

        @property
        def sock(self):
            self._reads += 1
            return None if self._reads > 1 else self._sock

        def close(self):
            self.closed = True

    race = _RaceConn()
    ws_tunnel._hard_close(race)  # must not raise AttributeError
    race._sock.shutdown.assert_called_once()
    assert race.closed


def _magic_conn():
    conn = MagicMock()
    conn.sock = MagicMock()
    return conn


def test_hard_close_shuts_down_socket_before_close():
    """A bare close() does not reliably wake a thread blocked in recv
    on Darwin; shutdown() does (and is what carries the FIN serve's
    disconnect guard waits for)."""
    conn = _magic_conn()
    ws_tunnel._hard_close(conn)
    conn.sock.shutdown.assert_called_once()
    conn.close.assert_called_once()


def test_hard_close_tolerates_dead_socket():
    conn = MagicMock()
    conn.sock = None
    ws_tunnel._hard_close(conn)  # must not raise
    conn.close.assert_called_once()


def test_abort_wakes_blocked_worker_via_shutdown():
    client = ws_tunnel.TunnelClient(local_port=1)
    conn = _magic_conn()
    client._active["req-1"] = conn
    client._dispatch_inbound({"t": "abort", "id": "req-1"})
    conn.sock.shutdown.assert_called_once()
    conn.close.assert_called_once()


def test_clean_close_captures_relay_close_code():
    """A post-upgrade policy close (defensive path: primary key
    rejection is an HTTP 401 at the upgrade, but a relay MAY instead
    drop a bad claim after accepting the socket) must surface its close
    code for the supervisor's terminal check."""

    class _FakeWS:
        close_code = 1008
        close_reason = "policy violation"

        async def send(self, msg):
            pass

        async def close(self):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    async def fake_connect(uri, **kw):
        return _FakeWS()

    client = ws_tunnel.TunnelClient(local_port=1, share_key=SHARE_KEY)
    with patch.object(ws_tunnel.websockets, "connect", fake_connect):
        asyncio.run(client.run())
    assert client.close_code == 1008
    assert client.error is None


def test_real_connection_closed_error_surfaces_close_code():
    """A policy close arrives as a REAL websockets ConnectionClosedError
    whose code lives on ``exc.rcvd.code`` (modern) / ``exc.code``
    (legacy, deprecated in 17). The supervisor's terminal check reads
    client.close_code — the extraction must match the library's actual
    shape or a revoked key enters the reconnect loop."""
    import websockets.exceptions as ws_exc
    from websockets.protocol import Close

    class _FakeWS:
        close_code = None  # prove the code came from the exception

        async def send(self, msg):
            pass

        async def close(self):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ws_exc.ConnectionClosedError(
                rcvd=Close(1008, "policy violation"), sent=None
            )

    async def fake_connect(uri, **kw):
        return _FakeWS()

    client = ws_tunnel.TunnelClient(local_port=1, share_key=SHARE_KEY)
    with (
        patch.object(ws_tunnel.websockets, "connect", fake_connect),
        pytest.raises(ws_exc.ConnectionClosedError),
    ):
        asyncio.run(client.run())
    assert client.close_code == 1008
    assert client.error is not None


def test_connect_uri_percent_encodes_tunnel_id():
    """The connect URL is query-interpolated — an id carrying & or #
    (server-supplied in pool mode) must stay ONE ``id`` value, not
    inject extra params or a fragment into the relay request."""
    seen: list = []

    class _FakeWS:
        close_code = None

        async def send(self, msg):
            pass

        async def close(self):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    async def fake_connect(uri, **kw):
        seen.append(uri)
        return _FakeWS()

    client = ws_tunnel.TunnelClient(
        local_port=1, tunnel_id="qspnode-a&b=c#d", relay_url="wss://r.test/up"
    )
    with patch.object(ws_tunnel.websockets, "connect", fake_connect):
        asyncio.run(client.run())
    assert len(seen) == 1
    parsed = urllib.parse.urlparse(seen[0])
    assert parsed.path == "/up" and not parsed.fragment
    assert urllib.parse.parse_qs(parsed.query) == {"id": ["qspnode-a&b=c#d"]}


def test_fetch_registration_refused_after_shutdown_begins():
    """A to_thread worker scheduled pre-teardown that only reaches
    registration after the drain must refuse, not register into a
    tunnel nobody will ever drain again."""
    client = ws_tunnel.TunnelClient(local_port=1)
    client._closed.set()
    conn = MagicMock()
    with patch.object(ws_tunnel, "OneShotHTTPConnection", return_value=conn):
        client._perform_local_fetch("r9", "POST", "/v1/chat/completions", {}, b"")
    conn.request.assert_not_called()
    conn.close.assert_called_once()
    assert client._active == {}


def test_teardown_drains_inflight_connections():
    """Task cancellation cannot stop the to_thread fetch workers — the
    run() finally must hard-close every registered connection, or
    generations keep running against a dead tunnel."""

    class _FakeWS:
        async def send(self, msg):
            pass

        async def close(self):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    async def fake_connect(uri, **kw):
        return _FakeWS()

    client = ws_tunnel.TunnelClient(local_port=1)
    conn = _magic_conn()
    client._active["live-req"] = conn
    with patch.object(ws_tunnel.websockets, "connect", fake_connect):
        asyncio.run(client.run())
    conn.sock.shutdown.assert_called_once()
    conn.close.assert_called_once()
    assert client._active == {}


def test_abort_before_fetch_registration_skips_generation():
    """The abort can beat the req task to the ``_active`` registry (the
    fetch worker registers a loop tick late). A silently-dropped abort
    would burn a pool slot for a downstream that already left."""
    client = ws_tunnel.TunnelClient(local_port=1)
    client._dispatch_inbound({"t": "abort", "id": "r1"})
    assert "r1" in client._aborted

    conn = MagicMock()
    with patch.object(ws_tunnel, "OneShotHTTPConnection", return_value=conn):
        client._perform_local_fetch("r1", "POST", "/v1/chat/completions", {}, b"")
    conn.request.assert_not_called()
    conn.close.assert_called_once()
    assert client._aborted == {} and client._active == {}


def test_stale_abort_marker_does_not_discard_reused_request_id():
    """Relays may reuse request ids (per-gateway counters that reset on
    reconnect). An abort that arrived after its own fetch had already
    finished left a stray marker; a LATER legitimate request carrying
    the same id must still be served, not silently dropped."""
    client = ws_tunnel.TunnelClient(local_port=1)
    client._dispatch_inbound({"t": "abort", "id": "r1"})
    # Age the marker past the race window it could possibly describe.
    client._aborted["r1"] -= ws_tunnel._ABORT_MARKER_STALE_SECONDS + 1

    conn = MagicMock()
    with (
        patch.object(ws_tunnel, "OneShotHTTPConnection", return_value=conn),
        patch.object(client, "_sync_send"),
    ):
        conn.getresponse.return_value.getheaders.return_value = []
        conn.getresponse.return_value.read1.return_value = b""
        client._perform_local_fetch("r1", "GET", "/v1/models", {}, b"")
    conn.request.assert_called_once()  # served, not discarded
    assert client._aborted == {} and client._active == {}


def test_fresh_abort_marker_still_wins_on_id_reuse():
    """The stale-window exemption must not weaken the real race: a
    marker inside the window still skips the generation."""
    client = ws_tunnel.TunnelClient(local_port=1)
    client._dispatch_inbound({"t": "abort", "id": "r1"})
    conn = MagicMock()
    with patch.object(ws_tunnel, "OneShotHTTPConnection", return_value=conn):
        client._perform_local_fetch("r1", "POST", "/v1/chat/completions", {}, b"")
    conn.request.assert_not_called()


def test_completed_id_reuse_not_cancelled_by_inflight_abort():
    """The hard reuse case: request r1 finishes, ITS abort arrives late
    (in flight when the gateway's counter wrapped), and a brand-new
    legitimate r1 is already in the window. The late abort must be
    dropped, the new request served — the pre-registration skip only
    applies to ids that were never completed."""
    client = ws_tunnel.TunnelClient(local_port=1)

    def _served_fetch(conn: MagicMock) -> None:
        with (
            patch.object(ws_tunnel, "OneShotHTTPConnection", return_value=conn),
            patch.object(client, "_sync_send"),
        ):
            conn.getresponse.return_value.getheaders.return_value = []
            conn.getresponse.return_value.read1.return_value = b""
            client._perform_local_fetch("r1", "GET", "/v1/models", {}, b"")

    first = MagicMock()
    _served_fetch(first)
    assert first.request.called and "r1" in client._completed

    client._dispatch_inbound({"t": "abort", "id": "r1"})  # late — dropped
    assert "r1" not in client._aborted

    second = MagicMock()
    _served_fetch(second)
    second.request.assert_called_once()  # reused id served, not skipped


def test_abort_for_pending_id_owns_marker_despite_retired_ancestor():
    """Ownership is by EPOCH STATE, not timestamp: the id's previous
    epoch is retired AND completed, but a new epoch is in flight — the
    abort belongs to the live epoch and must mark, even though the
    retired-ancestor rule alone would have dropped it."""
    client = ws_tunnel.TunnelClient(local_port=1)
    client._pending["r1"] = 1
    client._completed["r1"] = 0.0  # an ancestor epoch retired
    client._dispatch_inbound({"t": "abort", "id": "r1"})
    assert "r1" in client._aborted


def test_release_epoch_keeps_id_while_other_epochs_in_flight():
    client = ws_tunnel.TunnelClient(local_port=1)
    client._pending["r1"] = 2
    client._release_epoch("r1")
    assert client._pending.get("r1") == 1
    client._release_epoch("r1")
    assert "r1" not in client._pending


def test_duplicate_active_id_is_refused_not_overwritten():
    """Relay protocol violation (duplicate id while the first is still
    generating): overwriting _active would orphan the first —
    unabortable, invisible to the teardown drain. The newcomer is
    refused instead."""
    client = ws_tunnel.TunnelClient(local_port=1)
    existing = MagicMock()
    client._active["r1"] = existing
    newcomer = MagicMock()
    with patch.object(ws_tunnel, "OneShotHTTPConnection", return_value=newcomer):
        client._perform_local_fetch("r1", "POST", "/v1/chat/completions", {}, b"")
    newcomer.request.assert_not_called()
    newcomer.close.assert_called_once()
    assert client._active["r1"] is existing  # original stays owned


def test_late_abort_after_completion_is_harmless_and_bounded():
    """An abort for a freshly-finished fetch is dropped (nothing to
    cancel, marker would poison id reuse); aborts for never-seen ids
    still record markers, capped at 4096 so multi-day nodes can't
    creep."""
    client = ws_tunnel.TunnelClient(local_port=1)
    conn = MagicMock()
    with (
        patch.object(ws_tunnel, "OneShotHTTPConnection", return_value=conn),
        patch.object(client, "_sync_send"),
    ):
        conn.getresponse.return_value.getheaders.return_value = []
        conn.getresponse.return_value.read1.return_value = b""
        client._perform_local_fetch("r1", "GET", "/v1/models", {}, b"")
    assert client._active == {}
    client._dispatch_inbound({"t": "abort", "id": "r1"})  # too late, dropped
    assert "r1" not in client._aborted and "r1" in client._completed
    for i in range(5000):  # unknown ids still mark; cap holds
        client._dispatch_inbound({"t": "abort", "id": f"x{i}"})
    assert len(client._aborted) <= 4096


def test_override_authorization_applies_to_get_requests():
    """The serve's auth gate guards EVERY method — a GET /v1/models
    carrying the pool credential would 401 locally if the swap were
    POST-only."""
    client = ws_tunnel.TunnelClient(local_port=1, override_authorization="local-bearer")
    msg = _req_msg(method="GET")
    msg["headers"]["Authorization"] = "Bearer qsp-pool-key"
    got = _run_handle_request(client, msg)
    lowered = {k.lower(): v for k, v in got["headers"].items()}
    assert lowered["authorization"] == "Bearer local-bearer"
    assert "qsp-pool-key" not in json.dumps(got["headers"])


# ─────────────────────────── catalog resolution §5.4 ───────────────────────────


@pytest.mark.parametrize(
    "typed, expected",
    [
        ("qwen3.6-35b", ("qwen3.6-35b", "qwen3.6-35b")),
        ("qwen3.8-27b", ("qwen3.8-27b", "qwen3.8-27b-4bit")),
        ("qwen3.8-27b-4bit", ("qwen3.8-27b", "qwen3.8-27b-4bit")),
        (
            "nemotron-3.5-lightning-30b-4bit",
            ("nemotron-3.5-lightning", "nemotron-3.5-lightning-30b-4bit"),
        ),
    ],
)
def test_resolve_catalog_table(typed, expected):
    assert (
        qs.resolve_catalog(_make_args(model=typed, quicksilver_model=None)) == expected
    )


def test_resolve_catalog_explicit_flag_wins_and_unknown_passes_through():
    args = _make_args(model="my-thing", quicksilver_model="brand-new-model")
    assert qs.resolve_catalog(args) == ("brand-new-model", "my-thing")


def test_resolve_catalog_unknown_without_flag_errors():
    with pytest.raises(qs.QuickSilverError, match="not a known QuickSilver catalog id"):
        qs.resolve_catalog(_make_args(model="my-thing", quicksilver_model=None))


# ───────────────────────────── cache §5.1 ─────────────────────────────


def test_cache_roundtrip_is_0600_and_complete(tmp_path):
    payload = dict(_register_payload(), alias="qwen3.6-35b")
    path = qs._save_cache("qwen3.6-35b", payload)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # Only allowlisted fields persist (pool_base is in the wire
    # response but not in _CACHE_ALLOWED_KEYS — see the leak test).
    loaded = qs._load_cache("qwen3.6-35b", "qwen3.6-35b")
    assert loaded == {k: v for k, v in payload.items() if k in qs._CACHE_ALLOWED_KEYS}
    assert "pool_base" not in loaded


def test_cache_save_never_persists_unexpected_response_fields(tmp_path):
    """The registration response is server-controlled — an echoed
    provider key or surprise token must not become a resident secret
    (§1: the provider key is never written down)."""
    payload = dict(
        _register_payload(),
        alias="qwen3.6-35b",
        provider_key=PROVIDER_KEY,
        api_token="secret-token-xyz",
    )
    path = qs._save_cache("qwen3.6-35b", payload)
    raw = path.read_text(encoding="utf-8")
    assert PROVIDER_KEY not in raw
    assert "secret-token-xyz" not in raw
    assert "provider_key" not in raw and "api_token" not in raw
    assert qs._load_cache("qwen3.6-35b", "qwen3.6-35b")["share_key"] == SHARE_KEY


def test_cache_alias_mismatch_is_not_reused(tmp_path):
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="alias-a"))
    assert qs._load_cache("qwen3.6-35b", "alias-b") is None


def test_cache_missing_field_is_not_reused(tmp_path):
    payload = dict(_register_payload(), alias="qwen3.6-35b")
    del payload["heartbeat_url"]
    qs._save_cache("qwen3.6-35b", payload)
    assert qs._load_cache("qwen3.6-35b", "qwen3.6-35b") is None


def test_cache_corrupt_file_is_not_reused(tmp_path):
    p = qs._cache_path("qwen3.6-35b")
    p.write_text("{not json", encoding="utf-8")
    assert qs._load_cache("qwen3.6-35b", "qwen3.6-35b") is None


@pytest.mark.parametrize(
    "field",
    ["node_id", "share_key", "model", "relay_url", "heartbeat_url", "alias"],
)
def test_cache_nonstring_wire_field_is_not_reused(tmp_path, field):
    """Disk is attacker-adjacent: a cache whose share_key is a number or
    whose relay_url is a nested object must be rejected on load, not
    rendered, URL-parsed, or shoved into the greeting frame verbatim."""
    payload = dict(_register_payload(), alias="qwen3.6-35b")
    payload[field] = {"nested": [1, 2]}
    qs._save_cache("qwen3.6-35b", payload)
    assert qs._load_cache("qwen3.6-35b", "qwen3.6-35b") is None


def test_cache_missing_interval_field_is_not_reused(tmp_path):
    payload = dict(_register_payload(), alias="qwen3.6-35b")
    del payload["heartbeat_interval_s"]
    qs._save_cache("qwen3.6-35b", payload)
    assert qs._load_cache("qwen3.6-35b", "qwen3.6-35b") is None


def test_load_cache_dir_failure_is_actionable(tmp_path, monkeypatch):
    """_cache_path CREATES the cache dir — a HOME under which it cannot
    be made (read-only volume, path collision) must be the redacted
    exit-2 line, not a traceback and not a silent prompt for a
    provider key the operator may not have."""
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("HOME", str(blocker / "sub"))
    with pytest.raises(qs.QuickSilverError, match="cache directory"):
        qs._load_cache("qwen3.6-35b", "qwen3.6-35b")


def test_load_cache_unreadable_file_is_actionable(tmp_path):
    """EACCES on an EXISTING cache is not 'no cache' — falling through
    would start an unasked-for re-registration."""
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    path = qs._cache_path("qwen3.6-35b")
    path.chmod(0o000)
    try:
        with pytest.raises(qs.QuickSilverError, match="read the node cache"):
            qs._load_cache("qwen3.6-35b", "qwen3.6-35b")
    finally:
        path.chmod(0o600)


def test_save_cache_failure_removes_credential_bearing_tmp(tmp_path):
    """The write is tmp→replace; if the replace fails (full disk) a
    SECOND copy of the share-key sits on disk as the .tmp file. It must
    be unlinked, and the OSError must surface as the redacted exit
    path, not a raw traceback."""
    payload = dict(_register_payload(), alias="qwen3.6-35b")
    with (
        patch.object(qs.os, "replace", side_effect=OSError("No space left on device")),
        pytest.raises(qs.QuickSilverError, match="could not write node cache"),
    ):
        qs._save_cache("qwen3.6-35b", payload)
    leftovers = [p.name for p in qs._cache_dir().iterdir() if ".tmp-" in p.name]
    assert leftovers == []
    raw = b"".join(p.read_bytes() for p in qs._cache_dir().iterdir())
    assert SHARE_KEY.encode() not in raw


def test_cache_path_rejects_path_shaped_ids():
    with pytest.raises(qs.QuickSilverError):
        qs._cache_path("../evil")


# ─────────────────────────── registration §3.1 ───────────────────────────


def test_register_success_sets_alias_and_ua():
    calls: list = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        return _FakeResp(_register_payload())

    with patch.object(qs, "_open", fake_urlopen):
        out = qs.register_node(
            "https://pay.test", PROVIDER_KEY, "qwen3.6-35b", "al", "w"
        )
    assert out["alias"] == "al"
    req = calls[0]
    assert req.headers["User-agent"] == qs._user_agent()  # urllib title-cases
    assert req.headers["Authorization"] == f"Bearer {PROVIDER_KEY}"
    body = json.loads(req.data)
    assert body["model"] == "qwen3.6-35b" and body["alias"] == "al"


@pytest.mark.parametrize(
    "code, hint",
    [
        (401, "provider key"),
        (403, "not pool-eligible"),
        (409, "different account"),
        (422, "unknown/unsupported"),
    ],
)
def test_register_terminal_codes(code, hint):
    def fake_urlopen(req, timeout=None):
        raise _http_error(code)

    with (
        patch.object(qs, "_open", fake_urlopen),
        pytest.raises(qs.QuickSilverError, match=hint),
    ):
        qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a", "w")


def test_register_retries_429_then_succeeds():
    calls: list = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        if len(calls) == 1:
            raise _http_error(429)
        return _FakeResp(_register_payload())

    with (
        patch.object(qs, "_open", fake_urlopen),
        patch.object(qs, "_register_retry_sleep") as sleep,
    ):
        out = qs.register_node(
            "https://pay.test", PROVIDER_KEY, "qwen3.6-35b", "al", "w"
        )
    assert out["node_id"] == "qspnode-1a2b3c4d"
    assert sleep.call_count == 1
    # The retry must re-POST the ORIGINAL registration payload — an
    # earlier revision shadowed the request `body` with the 429's
    # response body, so attempt 2 shipped the server's own error JSON.
    assert len(calls) == 2
    for req in calls:
        sent = json.loads(req.data)
        assert sent["model"] == "qwen3.6-35b"
        assert sent["alias"] == "al"
        assert "hardware" in sent


def test_register_error_detail_never_leaks_secrets():
    def fake_urlopen(req, timeout=None):
        raise _http_error(
            422, json.dumps({"error": {"message": f"bad {SHARE_KEY}"}}).encode()
        )

    qs._register_secret(SHARE_KEY)
    with (
        patch.object(qs, "_open", fake_urlopen),
        pytest.raises(qs.QuickSilverError) as ei,
    ):
        qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a", "w")
    assert SHARE_KEY not in str(ei.value)


def test_register_400_is_terminal_without_retrying():
    """Only 429/5xx/transport are transient. A permanent 4xx (or a
    3xx from the no-redirect opener) must surface immediately, not
    hammer the API for the five-minute retry budget."""

    def one(code: int) -> tuple[int, object]:
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            raise _http_error(code)

        with (
            patch.object(qs, "_open", fake_urlopen),
            patch.object(qs, "_register_retry_sleep") as sleep,
            pytest.raises(qs.QuickSilverError, match=f"HTTP {code}"),
        ):
            qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a", "w")
        return calls["n"], sleep

    for code in (400, 404):
        n, sleep = one(code)
        assert n == 1
        sleep.assert_not_called()


def test_register_unparseable_2xx_is_actionable_error():
    class _GarbageResp:
        status = 200

        def read(self, size: int = -1) -> bytes:
            return b"<html>nginx is not json</html>"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    with (
        patch.object(qs, "_open", lambda req, timeout=None: _GarbageResp()),
        pytest.raises(qs.QuickSilverError, match="unparseable"),
    ):
        qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a", "w")


def test_register_oversized_success_response_is_terminal_not_buffered():
    """A hostile pay.* streaming an unbounded 2xx body must hit the
    size cap and exit terminal — no buffering the blob, no retry."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        return _FakeResp(
            _register_payload(junk="x" * (qs._REGISTER_RESPONSE_MAX_BYTES + 16))
        )

    with (
        patch.object(qs, "_open", fake_urlopen),
        patch.object(qs, "_register_retry_sleep"),
        pytest.raises(qs.QuickSilverError, match="too large"),
    ):
        qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a", "w")
    assert calls["n"] == 1


def test_register_oversized_error_body_reports_code_without_blob():
    """The error-body cap protects memory AND stderr: an oversized body
    is dropped, the actionable code/hint still surfaces."""
    blob = b"e" * (qs._ERROR_BODY_MAX_BYTES + 1)

    def fake_urlopen(req, timeout=None):
        raise _http_error(422, blob)

    with (
        patch.object(qs, "_open", fake_urlopen),
        pytest.raises(qs.QuickSilverError) as ei,
    ):
        qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a", "w")
    msg = str(ei.value)
    assert "422" in msg or "unknown/unsupported" in msg
    assert "eeee" not in msg


def test_register_model_echo_mismatch_is_terminal():
    """The response must echo the catalog id we asked for — binding a
    different model would serve/bill under a pool model nobody
    registered for."""

    def fake_urlopen(req, timeout=None):
        return _FakeResp(_register_payload(model="someone-elses-model"))

    with (
        patch.object(qs, "_open", fake_urlopen),
        pytest.raises(qs.QuickSilverError, match="does not match"),
    ):
        qs.register_node("https://pay.test", PROVIDER_KEY, "qwen3.6-35b", "a", "w")


def test_wire_urls_reject_credential_bearing_components():
    """userinfo/query/fragment are the classic key-in-URL channels —
    the v1 claim rides the ready frame, so anything there is hostile
    (or at least an unprintable leak candidate)."""
    for relay in (
        "wss://rapidserver.quicksilverpro.io/up?key=qspsk-leak",
        "wss://user:qspsk-leak@rapidserver.quicksilverpro.io/up",
        "wss://rapidserver.quicksilverpro.io/up#qspsk-leak",
    ):
        with pytest.raises(qs.QuickSilverError, match="userinfo, query, or fragment"):
            qs._validate_wire_urls(
                _register_payload(relay_url=relay), _API_ORIGIN, source="t"
            )


def test_url_opener_refuses_redirects():
    """urllib REPLAYS the Authorization header when following a 3xx —
    on register that would ship the provider key to the redirect
    target. The opener must surface the 3xx instead of following."""
    req = urllib.request.Request(
        "https://pay.quicksilverpro.io/v1/pool/nodes/register",
        headers={"Authorization": f"Bearer {PROVIDER_KEY}"},
    )
    for handler in qs._URL_OPENER.handlers:
        if isinstance(handler, qs._NoRedirectHandler):
            break
    else:
        raise AssertionError("no-redirect handler not installed on the opener")
    assert (
        handler.redirect_request(req, None, 302, "Found", None, "https://evil.test")
        is None
    )


def test_register_302_is_terminal_single_call():
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(302)

    with (
        patch.object(qs, "_open", fake_urlopen),
        pytest.raises(qs.QuickSilverError, match="HTTP 302"),
    ):
        qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a", "w")
    assert calls["n"] == 1


def test_api_base_validation():
    assert (
        qs._validate_api_base("https://pay.quicksilverpro.io/")
        == "https://pay.quicksilverpro.io"
    )
    assert qs._validate_api_base("http://127.0.0.1:9999") == "http://127.0.0.1:9999"
    for bad in (
        "http://evil.example.com",
        "ftp://pay.example.com",
        "https://pay.example.com/api/v2",
        "https://",
        "https://pay.example.com?x=1",
    ):
        with pytest.raises(qs.QuickSilverError):
            qs._validate_api_base(bad)
    with pytest.raises(qs.QuickSilverError, match="userinfo"):
        qs._validate_api_base("https://user:secret@pay.quicksilverpro.io")


def test_provider_key_resolution_order(monkeypatch):
    monkeypatch.setenv(qs.PROVIDER_KEY_ENV_VAR, "qsppk-env")
    assert qs._resolve_provider_key(_make_args(provider_key=None)) == "qsppk-env"
    assert (
        qs._resolve_provider_key(_make_args(provider_key="qsppk-flag")) == "qsppk-flag"
    )


@pytest.mark.parametrize("blank", ["   ", "\t\n"])
def test_provider_key_whitespace_only_rejected_without_network(monkeypatch, blank):
    """ "Empty but truthy" must not become `Bearer ` — the remote 401 it
    produces reads like a revoked key, not like the operator error it
    is. Reject locally, before any request."""
    monkeypatch.setenv(qs.PROVIDER_KEY_ENV_VAR, blank)

    def _no_net(*a, **k):
        raise AssertionError("network touched with a blank key")

    with (
        patch.object(qs, "_open", _no_net),
        pytest.raises(SystemExit) as ei,
    ):
        qs.run_share(_make_args())
    assert ei.value.code == 2
    with pytest.raises(qs.QuickSilverError):
        qs._resolve_provider_key(_make_args(provider_key=blank))


def test_provider_key_noninteractive_without_key_is_actionable(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    with pytest.raises(qs.QuickSilverError, match=qs.PROVIDER_KEY_ENV_VAR):
        qs._resolve_provider_key(_make_args(provider_key=None))


# ────────────────── wire URL origin validation (credential leak) ──────────────────


_API_ORIGIN = "https://pay.quicksilverpro.io"


def test_wire_urls_trusted_origins_and_loopback_pass():
    qs._validate_wire_urls(_register_payload(), _API_ORIGIN, source="t")
    qs._validate_wire_urls(
        _register_payload(
            relay_url="ws://127.0.0.1:9/off", heartbeat_url="http://localhost:9/hb"
        ),
        "http://127.0.0.1:9",
        source="t",
    )
    # the --quicksilver-api origin itself is trusted (test/staging APIs)
    qs._validate_wire_urls(
        _register_payload(
            relay_url="wss://api.staging.example/up",
            heartbeat_url="https://api.staging.example/hb",
        ),
        "https://api.staging.example",
        source="t",
    )


def test_wire_urls_reject_cleartext_off_loopback():
    # A ws:// relay would carry the Authorization share-key in clear.
    with pytest.raises(qs.QuickSilverError, match="relay_url must be wss"):
        qs._validate_wire_urls(
            _register_payload(relay_url="ws://rapidserver.quicksilverpro.io/up"),
            _API_ORIGIN,
            source="t",
        )
    # An http:// heartbeat would carry the Authorization share-key.
    with pytest.raises(qs.QuickSilverError, match="heartbeat_url must be https"):
        qs._validate_wire_urls(
            _register_payload(heartbeat_url="http://hb.quicksilverpro.io/hb"),
            _API_ORIGIN,
            source="t",
        )


def test_wire_urls_reject_foreign_hosts_even_over_tls():
    # wss/https alone is not enough — a compromised API pointing the
    # credential at a stranger must be refused on origin.
    with pytest.raises(qs.QuickSilverError, match="not a QuickSilver origin"):
        qs._validate_wire_urls(
            _register_payload(relay_url="wss://relay.evil.test/up"),
            _API_ORIGIN,
            source="t",
        )
    with pytest.raises(qs.QuickSilverError, match="not a QuickSilver origin"):
        qs._validate_wire_urls(
            _register_payload(heartbeat_url="https://collector.evil.test/hb"),
            _API_ORIGIN,
            source="t",
        )


def test_wire_urls_reject_lookalike_domains():
    for relay in (
        "wss://evil-quicksilverpro.io/up",
        "wss://quicksilverpro.io.evil.test/up",
        "wss://xquicksilverpro.io/up",
    ):
        with pytest.raises(qs.QuickSilverError, match="not a QuickSilver origin"):
            qs._validate_wire_urls(
                _register_payload(relay_url=relay), _API_ORIGIN, source="t"
            )


def test_run_share_hostile_cached_relay_never_connects(capsys):
    """A cache written by a compromised/buggy API is just as dangerous
    as a hostile wire response — validation must gate the tunnel, and
    the only repair is --reregister."""
    hostile = dict(
        _register_payload(relay_url="wss://collector.evil.test/up"),
        alias="qwen3.6-35b",
    )
    qs._save_cache("qwen3.6-35b", hostile)

    def _no_net(*a, **k):
        raise AssertionError("network touched")

    with (
        patch.object(qs, "_open", _no_net),
        patch.object(qs.ws_tunnel, "TunnelClient") as tunnel_cls,
        pytest.raises(SystemExit) as ei,
    ):
        qs.run_share(_make_args())
    assert ei.value.code == 2
    tunnel_cls.assert_not_called()
    err = capsys.readouterr().err
    assert "--reregister" in err
    assert SHARE_KEY not in err


def test_run_share_model_mismatch_response_never_cached_or_served(capsys):
    hostile = _register_payload(model="someone-elses-model")
    with (
        patch.object(qs, "_open", lambda req, timeout=None: _FakeResp(hostile)),
        patch.object(qs.ws_tunnel, "TunnelClient") as tunnel_cls,
        pytest.raises(SystemExit) as ei,
    ):
        qs.run_share(_make_args(provider_key=PROVIDER_KEY))
    assert ei.value.code == 2
    tunnel_cls.assert_not_called()
    assert qs._load_cache("qwen3.6-35b", "qwen3.6-35b") is None


def test_run_share_hostile_register_response_not_cached(capsys):
    hostile = _register_payload(relay_url="ws://collector.evil.test/up")
    with (
        patch.object(qs, "_open", lambda req, timeout=None: _FakeResp(hostile)),
        patch.object(qs.ws_tunnel, "TunnelClient") as tunnel_cls,
        pytest.raises(SystemExit) as ei,
    ):
        qs.run_share(_make_args(provider_key=PROVIDER_KEY))
    assert ei.value.code == 2
    tunnel_cls.assert_not_called()
    assert qs._load_cache("qwen3.6-35b", "qwen3.6-35b") is None  # never persisted
    assert "relay_url" in capsys.readouterr().err


# ───────────────────────────── heartbeat §3.3 ───────────────────────────


def _hb(**over) -> qs._Heartbeat:
    return qs._Heartbeat("https://hb.test/hb", SHARE_KEY, 10, lambda: 3)


def test_heartbeat_200_ok_carries_inflight_and_never_key_in_body():
    hb = _hb()
    reqs = []

    def fake_urlopen(req, timeout=None):
        reqs.append(req)
        return _FakeResp({"ok": True})

    with patch.object(qs, "_open", fake_urlopen):
        hb._beat_once()
    body = json.loads(reqs[0].data)
    assert body == {"inflight": 3, "client": qs._user_agent()}
    assert SHARE_KEY not in json.dumps(body)
    assert reqs[0].headers["Authorization"] == f"Bearer {SHARE_KEY}"


def test_heartbeat_404_beats_on_and_logs_once(caplog):
    hb = _hb()

    def fake_urlopen(req, timeout=None):
        raise _http_error(404)

    with (
        patch.object(qs, "_open", fake_urlopen),
        caplog.at_level(logging.INFO, logger="vllm_mlx.share.quicksilver"),
    ):
        hb._beat_once()
        hb._beat_once()
    assert not hb.fatal.is_set()
    assert sum("keep beating" in r.message for r in caplog.records) == 1


def test_heartbeat_401_is_fatal():
    hb = _hb()

    def fake_urlopen(req, timeout=None):
        raise _http_error(401)

    with patch.object(qs, "_open", fake_urlopen):
        hb._beat_once()
    assert hb.fatal.is_set()


def test_heartbeat_transport_error_is_transparent():
    hb = _hb()

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("unreachable")

    with patch.object(qs, "_open", fake_urlopen):
        hb._beat_once()  # must not raise, must not go fatal
    assert not hb.fatal.is_set()


@pytest.mark.parametrize("code", [401, 404, 500])
def test_heartbeat_http_error_response_is_closed(code):
    """An HTTPError IS the response. The heartbeat beats every 10 s for
    the node's whole life — leaving the error socket unclosed leaks one
    per rejected beat, forever."""
    hb = _hb()
    fp = io.BytesIO(b"{}")
    err = urllib.error.HTTPError("https://hb.test/hb", code, "err", {}, fp)

    def fake_urlopen(req, timeout=None):
        raise err

    with patch.object(qs, "_open", fake_urlopen):
        hb._beat_once()
    assert fp.closed


def test_register_http_error_response_is_closed():
    """Same discipline on the registration path: the retry loop can
    burn through many rejections in one session (429/5xx), and each one
    must hand its socket back."""
    fp = io.BytesIO(b'{"detail":"not pool-eligible"}')
    err = urllib.error.HTTPError("https://pay.test", 403, "err", {}, fp)

    def fake_urlopen(req, timeout=None):
        raise err

    with (
        patch.object(qs, "_open", fake_urlopen),
        pytest.raises(qs.QuickSilverError, match="403"),
    ):
        qs.register_node(
            "https://pay.test", PROVIDER_KEY, "qwen3.6-35b", "qwen3.6-35b", "w"
        )
    assert fp.closed


def test_heartbeat_stop_join_window_covers_request_timeout():
    """A beat mid-flight in _open keeps sending the node credential;
    stop() must outlast the beat's own socket timeout or run_share
    returns with a still-beating thread."""
    hb = _hb()
    hb._thread = MagicMock()
    hb.stop()
    hb._thread.join.assert_called_once()
    join_kwargs = hb._thread.join.call_args.kwargs
    assert join_kwargs["timeout"] >= qs._BEAT_REQUEST_TIMEOUT


def test_heartbeat_stop_warns_when_thread_survives_join(caplog):
    """The join window already exceeds the beat's socket timeout — a
    thread still alive past it means a credential-bearing beat may fire
    one final request after run_share returns. Surface it, don't claim
    a guarantee that failed."""
    hb = _hb()
    hb._thread = MagicMock()
    hb._thread.is_alive.return_value = True
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.share.quicksilver"):
        hb.stop()
    assert "one final request" in caplog.text


# ─────────────────────────── run_share integration ───────────────────────────


def _fake_tunnel(
    ready=True, closed=False, error=None, error_status=None, close_code=None
):
    t = MagicMock()
    t.error = error
    t.error_status = error_status
    t.close_code = close_code
    t.inflight = 0
    # ``ready_event.wait(30)`` must not actually block 30 s in tests —
    # stub the wait to report the configured state instantly.
    ready_event = MagicMock()
    ready_event.wait.return_value = bool(ready)
    t.ready_event = ready_event
    closed_event = threading.Event()
    if closed:
        closed_event.set()
    t.closed_event = closed_event
    thread = MagicMock()
    thread.is_alive.return_value = False
    t.run_in_thread.return_value = thread
    return t


def _fake_heartbeat_class(instance=None):
    hb = instance or MagicMock()
    hb.enabled = threading.Event()
    hb.fatal = threading.Event()
    hb.interval = 10.0
    return hb


def _serve_proc():
    p = MagicMock()
    p.poll.return_value = None
    p.wait.return_value = 0
    return p


def _patched_run_env(tunnel_factory):
    """Patches shared by every run_share integration test."""
    serve = _serve_proc()

    def _ctrl_c_on_first(*_a, **_k):
        raise KeyboardInterrupt

    return serve, _ctrl_c_on_first


def _run_patches(serve, tunnel_cls, sleep_side_effect):
    return (
        patch.object(share_cli, "_spawn_serve", return_value=serve),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(
            share_cli, "_resolve_served_model_name", return_value="served-model"
        ),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch.object(qs.ws_tunnel, "TunnelClient", new=tunnel_cls),
        patch.object(qs, "_warmup"),
        patch.object(qs, "_supervisor_sleep", side_effect=sleep_side_effect),
    )


def test_run_share_first_run_registers_caches_and_prints_keyless_banner(capsys):
    tunnel = _fake_tunnel()
    serve, ctrl_c = _patched_run_env(None)
    ctxs = _run_patches(serve, lambda **kw: tunnel, ctrl_c) + (
        patch.object(
            qs,
            "_open",
            lambda req, timeout=None: _FakeResp(_register_payload()),
        ),
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    args = _make_args(provider_key=PROVIDER_KEY)
    with _enter(*ctxs):
        qs.run_share(args)

    out = capsys.readouterr()
    combined = out.out + out.err
    # §8 bullet 1: banner shows node + masked payout, serves the model.
    assert "qspnode-1a2b3c4d" in out.out
    assert "a***@x***" in out.out
    assert "QuickSilver" in out.out
    # §8 bullet 7: no credential, anywhere in the process output.
    assert SHARE_KEY not in combined
    assert PROVIDER_KEY not in combined
    # Cache written, complete, 0600, bound to alias.
    cache = qs._load_cache("qwen3.6-35b", "qwen3.6-35b")
    assert cache is not None and cache["node_id"] == "qspnode-1a2b3c4d"
    assert stat.S_IMODE(qs._cache_path("qwen3.6-35b").stat().st_mode) == 0o600
    # Ctrl-C path tears both down.
    tunnel.stop.assert_called()
    serve.terminate.assert_called_once()


def test_run_share_serve_uses_served_model_name_catalog_id():
    """§5.4 + readiness: the pool addresses the node by its catalog id while the
    serve alias differs, so serve is spawned with --served-model-name=<catalog
    id>. Without it serve echoes the alias and the relay readiness probe
    (response `model` must equal the pool model) never passes → the node
    connects + heartbeats but stays unroutable and all traffic falls to cloud."""
    tunnel = _fake_tunnel()
    serve, ctrl_c = _patched_run_env(None)
    spawn = MagicMock(return_value=serve)
    ctxs = (
        patch.object(share_cli, "_spawn_serve", spawn),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(
            share_cli, "_resolve_served_model_name", return_value="qwen3.6-35b"
        ),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch.object(qs.ws_tunnel, "TunnelClient", new=lambda **kw: tunnel),
        patch.object(qs, "_warmup"),
        patch.object(qs, "_supervisor_sleep", side_effect=ctrl_c),
        patch.object(
            qs, "_open", lambda req, timeout=None: _FakeResp(_register_payload())
        ),
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with _enter(*ctxs):
        qs.run_share(_make_args(provider_key=PROVIDER_KEY))
    assert spawn.call_count == 1
    extra = spawn.call_args.kwargs["extra_args"]
    assert "--served-model-name" in extra
    assert extra[extra.index("--served-model-name") + 1] == "qwen3.6-35b"


def test_top_level_cli_accepts_known_quicksilver_catalog_id(monkeypatch):
    """The global alias guard must let the pool resolver map catalog ids."""
    from vllm_mlx import cli as top_cli

    called = MagicMock()
    monkeypatch.setattr(
        top_cli.sys,
        "argv",
        ["rapid-mlx", "share", "qwen3.8-27b", "--quicksilver"],
    )
    with patch.object(qs, "run_share", called):
        top_cli.main()
    called.assert_called_once()
    assert called.call_args.args[0].model == "qwen3.8-27b"


def test_register_body_carries_worker():
    """The register request names the per-machine worker (account.worker)."""
    seen = {}

    def fake_open(req, timeout):
        seen["body"] = json.loads(req.data.decode())
        return _FakeResp(_register_payload())

    with patch.object(qs, "_open", fake_open):
        qs.register_node(
            "https://pay.test", PROVIDER_KEY, "qwen3.6-35b", "qwen3.6-35b", "mac-studio"
        )
    assert seen["body"]["worker"] == "mac-studio"
    assert seen["body"]["model"] == "qwen3.6-35b"


def test_resolve_worker_defaults_to_hostname(monkeypatch):
    monkeypatch.setattr(qs.socket, "gethostname", lambda: "Some.Host-01")
    assert qs._resolve_worker(_make_args(worker=None)) == "Some.Host-01"
    # explicit --worker wins and is sanitized to the id charset
    assert qs._resolve_worker(_make_args(worker="mac mini!")) == "macmini"


def test_load_cache_worker_mismatch_forces_reregister(tmp_path, monkeypatch):
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    # same catalog+alias but a DIFFERENT machine (worker) => different node => miss
    assert qs._load_cache("qwen3.6-35b", "qwen3.6-35b", "other-machine") is None
    # matching worker => hit
    assert qs._load_cache("qwen3.6-35b", "qwen3.6-35b", "test-worker") is not None


def test_run_share_cached_run_never_calls_register(capsys):
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    tunnel = _fake_tunnel()
    serve, ctrl_c = _patched_run_env(None)

    def _no_net(*a, **k):
        raise AssertionError("network touched on cached path")

    ctxs = _run_patches(serve, lambda **kw: tunnel, ctrl_c) + (
        patch.object(qs, "_open", _no_net),
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with _enter(*ctxs):
        qs.run_share(_make_args())
    assert "cached node" in capsys.readouterr().err


def test_run_share_reregister_overwrites_rotated_key():
    old = dict(_register_payload(share_key="qspsk-oldoldold"), alias="qwen3.6-35b")
    qs._save_cache("qwen3.6-35b", old)
    tunnel = _fake_tunnel()
    serve, ctrl_c = _patched_run_env(None)

    seen: list = []

    def fake_urlopen(req, timeout=None):
        seen.append(req)
        return _FakeResp(_register_payload(share_key="qspsk-rotated"))

    ctxs = _run_patches(serve, lambda **kw: tunnel, ctrl_c) + (
        patch.object(qs, "_open", fake_urlopen),
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with _enter(*ctxs):
        qs.run_share(_make_args(reregister=True, provider_key=PROVIDER_KEY))
    assert len(seen) == 1
    assert qs._load_cache("qwen3.6-35b", "qwen3.6-35b")["share_key"] == "qspsk-rotated"


def test_run_share_serves_with_max_seqs_2_and_no_rate_limit():
    tunnel = _fake_tunnel()
    serve, ctrl_c = _patched_run_env(None)
    spawned: dict = {}

    def fake_spawn(**kw):
        spawned.update(kw)
        return serve

    ctxs = _run_patches(serve, lambda **kw: tunnel, ctrl_c) + (
        patch.object(
            qs,
            "_open",
            lambda req, timeout=None: _FakeResp(_register_payload()),
        ),
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    # the later _spawn_serve patch (capture) wins over _run_patches' stub
    with _enter(*ctxs, patch.object(share_cli, "_spawn_serve", side_effect=fake_spawn)):
        qs.run_share(_make_args(provider_key=PROVIDER_KEY))
    extra = spawned["extra_args"]
    assert extra[:2] == ["--max-num-seqs", "2"]
    assert "--rate-limit" not in extra
    assert "--no-thinking" in extra


def test_run_share_explicit_thinking_does_not_force_disable():
    tunnel = _fake_tunnel()
    serve, ctrl_c = _patched_run_env(None)
    spawned: dict = {}

    def fake_spawn(**kw):
        spawned.update(kw)
        return serve

    ctxs = _run_patches(serve, lambda **kw: tunnel, ctrl_c) + (
        patch.object(
            qs,
            "_open",
            lambda req, timeout=None: _FakeResp(_register_payload()),
        ),
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with _enter(*ctxs, patch.object(share_cli, "_spawn_serve", side_effect=fake_spawn)):
        qs.run_share(_make_args(provider_key=PROVIDER_KEY, thinking=True))
    assert "--no-thinking" not in spawned["extra_args"]


def test_run_share_respects_user_max_seqs_override():
    tunnel = _fake_tunnel()
    serve, ctrl_c = _patched_run_env(None)
    spawned: dict = {}

    def fake_spawn(**kw):
        spawned.update(kw)
        return serve

    ctxs = _run_patches(serve, lambda **kw: tunnel, ctrl_c) + (
        patch.object(
            qs,
            "_open",
            lambda req, timeout=None: _FakeResp(_register_payload()),
        ),
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with _enter(*ctxs, patch.object(share_cli, "_spawn_serve", side_effect=fake_spawn)):
        qs.run_share(
            _make_args(
                provider_key=PROVIDER_KEY,
                _passthrough=["--max-num-seqs", "8"],
            )
        )
    # Injection skipped — the user's own --max-num-seqs passthrough wins; the
    # catalog-id --served-model-name is still added (pool addresses by catalog id).
    assert spawned["extra_args"] == [
        "--no-thinking",
        "--served-model-name",
        "qwen3.6-35b",
        "--max-num-seqs",
        "8",
    ]


def test_run_share_scrubs_env_provider_key_before_serve_spawn(monkeypatch):
    """§1/§6: serve loads third-party model code and inherits
    os.environ wholesale. The account key is resolved by POP, so by
    spawn time (and by every earlier probe child) it must be gone from
    the environment — memory-only, in the supervisor alone."""
    monkeypatch.setenv(qs.PROVIDER_KEY_ENV_VAR, PROVIDER_KEY)
    tunnel = _fake_tunnel()
    serve, ctrl_c = _patched_run_env(None)
    seen: list = []

    def fake_spawn(**kw):
        seen.append(os.environ.get(qs.PROVIDER_KEY_ENV_VAR))
        return serve

    ctxs = _run_patches(serve, lambda **kw: tunnel, ctrl_c) + (
        patch.object(
            qs,
            "_open",
            lambda req, timeout=None: _FakeResp(_register_payload()),
        ),
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with _enter(*ctxs, patch.object(share_cli, "_spawn_serve", side_effect=fake_spawn)):
        qs.run_share(_make_args())  # provider key arrives via env, not flag
    assert seen == [None]


def test_cached_run_scrubs_unused_env_provider_key(monkeypatch):
    """Cached runs never resolve the key — but an exported one would
    otherwise sit in os.environ for the whole session and ride into the
    serve child. Scrub even when unused."""
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    monkeypatch.setenv(qs.PROVIDER_KEY_ENV_VAR, PROVIDER_KEY)
    tunnel = _fake_tunnel()
    serve, ctrl_c = _patched_run_env(None)
    seen: list = []

    def fake_spawn(**kw):
        seen.append(os.environ.get(qs.PROVIDER_KEY_ENV_VAR))
        return serve

    def _no_net(*a, **k):
        raise AssertionError("network touched on cached path")

    ctxs = _run_patches(serve, lambda **kw: tunnel, ctrl_c) + (
        patch.object(qs, "_open", _no_net),
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with _enter(*ctxs, patch.object(share_cli, "_spawn_serve", side_effect=fake_spawn)):
        qs.run_share(_make_args())
    assert seen == [None]
    assert qs.PROVIDER_KEY_ENV_VAR not in os.environ


def test_run_share_reaps_serve_proc_after_kill():
    """SIGKILL only signals — without a follow-up wait(), a serve that
    ignored SIGTERM lingers as a zombie child for the life of the
    node process (which KeepAlive keeps alive forever)."""
    tunnel = _fake_tunnel()
    serve, ctrl_c = _patched_run_env(None)
    # terminate() → wait(5) times out → kill() → wait() must be called
    # a second time to reap.
    serve.wait.side_effect = [subprocess.TimeoutExpired("serve", 5), 0]
    ctxs = _run_patches(serve, lambda **kw: tunnel, ctrl_c) + (
        patch.object(
            qs,
            "_open",
            lambda req, timeout=None: _FakeResp(_register_payload()),
        ),
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with _enter(*ctxs):
        qs.run_share(_make_args(provider_key=PROVIDER_KEY))
    serve.terminate.assert_called_once()
    serve.kill.assert_called_once_with()
    assert serve.wait.call_count == 2


def test_run_share_ws_401_is_terminal_no_spin():
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    tunnel = _fake_tunnel(
        ready=False, closed=True, error=Exception("401"), error_status=401
    )
    serve, _ = _patched_run_env(None)
    made: list = []

    def factory(**kw):
        made.append(kw)
        return tunnel

    ctxs = _run_patches(serve, factory, lambda *a, **k: None) + (
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with pytest.raises(SystemExit) as ei, _enter(*ctxs):
        qs.run_share(_make_args())
    assert ei.value.code == 1
    assert len(made) == 1  # tried exactly once — no retry spin on revoked key


def test_run_share_tunnel_drop_reconnects_then_dies_on_401():
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    first = _fake_tunnel(closed=True)  # up then immediately dropped
    second = _fake_tunnel(
        ready=False, closed=True, error=Exception("401"), error_status=401
    )
    serve, _ = _patched_run_env(None)
    tunnels = iter([first, second])
    sleeps: list = []

    def fake_sleep(secs):
        sleeps.append(secs)

    ctxs = _run_patches(serve, lambda **kw: next(tunnels), fake_sleep) + (
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with pytest.raises(SystemExit) as ei, _enter(*ctxs):
        qs.run_share(_make_args())
    assert ei.value.code == 1
    assert sleeps == [1.0]  # one backoff between attempts


def test_run_share_ws_1008_close_is_terminal_no_spin():
    """A revoked key normally fails as an HTTP 401 at the upgrade
    (see ``test_run_share_ws_401_is_terminal_no_spin``), but a relay
    that instead drops the claim post-upgrade with a 1008 policy close
    must hit the same no-spin terminal exit (§5.5)."""
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    tunnel = _fake_tunnel(ready=True, closed=True, close_code=1008)
    serve, _ = _patched_run_env(None)
    made: list = []

    def factory(**kw):
        made.append(kw)
        return tunnel

    ctxs = _run_patches(serve, factory, lambda *a, **k: None) + (
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with pytest.raises(SystemExit) as ei, _enter(*ctxs):
        qs.run_share(_make_args())
    assert ei.value.code == 1
    assert len(made) == 1  # no retry spin on a post-upgrade rejection


def test_run_share_instant_drop_streak_warns_but_keeps_retrying(capsys):
    """Five relay-accepted-then-instantly-dropped sessions with NO
    rejection signal (no 401, no 1008) reads as relay instability. The
    node must warn loudly and keep trying — exiting would turn a relay
    incident into lost earnings until an operator notices."""
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    serve, _ = _patched_run_env(None)
    made: list = []
    limit = qs._FAST_REJECT_STREAK_LIMIT

    def factory(**kw):
        made.append(kw)
        # connected, closed_event pre-set, unclassifiable close
        return _fake_tunnel(ready=True, closed=True, close_code=1001)

    sleeps: list[float] = []

    def sleep_then_ctrl_c(seconds, **_k):
        sleeps.append(seconds)
        if len(made) > limit + 2:
            raise KeyboardInterrupt

    ctxs = _run_patches(serve, factory, sleep_then_ctrl_c) + (
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with _enter(*ctxs):
        qs.run_share(_make_args())  # Ctrl-C after past-threshold retries
    err = capsys.readouterr().err
    assert "WARNING" in err and "instability" in err
    # Went PAST the streak threshold without exiting …
    assert len(made) > limit
    # … and warned exactly once, not once per drop.
    assert err.count("WARNING") == 1
    assert sleeps[:4] == [1.0, 2.0, 4.0, 8.0]


def test_run_share_1008_close_is_terminal_despite_instant_drop_shape(capsys):
    """The explicit-signal path must still exit: an immediate drop
    carrying close 1008 IS a rejection, streak heuristic irrelevant."""
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    serve, _ = _patched_run_env(None)

    def factory(**kw):
        return _fake_tunnel(ready=True, closed=True, close_code=1008)

    ctxs = _run_patches(serve, factory, lambda *a, **k: None) + (
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with pytest.raises(SystemExit) as ei, _enter(*ctxs):
        qs.run_share(_make_args())
    assert ei.value.code == 1
    assert "1008" in capsys.readouterr().err


def test_run_share_server_echoed_share_key_is_rejected_and_never_printed(capsys):
    """A share key reflected into node_id would otherwise enter the tunnel URL."""
    leaky = _register_payload(
        node_id=f"qspnode-{SHARE_KEY}", payout_account=f"a***@x*** {SHARE_KEY}"
    )
    with (
        patch.object(qs, "_open", lambda req, timeout=None: _FakeResp(leaky)),
        pytest.raises(SystemExit) as exc,
    ):
        qs.run_share(_make_args(provider_key=PROVIDER_KEY))
    assert exc.value.code == 2
    out = capsys.readouterr()
    combined = out.out + out.err
    assert SHARE_KEY not in combined
    assert not qs._cache_path("qwen3.6-35b").exists()


def test_register_response_cannot_persist_provider_key_in_allowed_field():
    hostile = _register_payload(payout_account=f"echo:{PROVIDER_KEY}")
    with (
        patch.object(qs, "_open", lambda req, timeout=None: _FakeResp(hostile)),
        pytest.raises(SystemExit) as exc,
    ):
        qs.run_share(_make_args(provider_key=PROVIDER_KEY))
    assert exc.value.code == 2
    assert not qs._cache_path("qwen3.6-35b").exists()


def test_cached_encoded_share_key_in_node_id_is_rejected_before_tunnel(capsys):
    encoded = urllib.parse.quote(SHARE_KEY, safe="").replace("-", "%2D")
    qs._save_cache(
        "qwen3.6-35b",
        dict(
            _register_payload(node_id=f"node-{encoded}"),
            alias="qwen3.6-35b",
        ),
    )
    with pytest.raises(SystemExit) as exc:
        qs.run_share(_make_args())
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert SHARE_KEY not in (captured.out + captured.err)


def test_run_share_rejects_nonstring_wire_fields():
    leaky = _register_payload(node_id={"evil": "object"})
    with (
        patch.object(qs, "_open", lambda req, timeout=None: _FakeResp(leaky)),
        pytest.raises(SystemExit) as ei,
    ):
        qs.run_share(_make_args(provider_key=PROVIDER_KEY))
    assert ei.value.code == 2
    assert qs._load_cache("qwen3.6-35b", "qwen3.6-35b") is None


def test_run_share_heartbeat_fatal_exits_nonzero(capsys):
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    tunnel = _fake_tunnel()
    serve, _ = _patched_run_env(None)
    hb = _fake_heartbeat_class()
    hb.fatal.set()

    ctxs = _run_patches(serve, lambda **kw: tunnel, lambda *a, **k: None) + (
        patch.object(qs, "_Heartbeat", return_value=hb),
    )
    with pytest.raises(SystemExit) as ei, _enter(*ctxs):
        qs.run_share(_make_args())
    assert ei.value.code == 1
    assert "--reregister" in capsys.readouterr().err


def test_run_share_key_never_reaches_tunnel_error_printing(capsys):
    # A tunnel error whose repr embeds the share key must print redacted.
    qs._register_secret(SHARE_KEY)
    qs._save_cache(
        "qwen3.6-35b",
        dict(_register_payload(), alias="qwen3.6-35b", share_key=SHARE_KEY),
    )
    tunnel = _fake_tunnel(
        ready=False, closed=True, error=Exception(f"rejected uri with {SHARE_KEY}")
    )
    serve, _ = _patched_run_env(None)
    ctxs = _run_patches(serve, lambda **kw: tunnel, lambda *a, **k: None) + (
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )

    def sleep_then_ctrl_c(secs):
        raise KeyboardInterrupt

    # Ctrl-C during the reconnect backoff → clean exit 0 (no SystemExit);
    # what we're pinning here is that the error repr printed before the
    # backoff went out REDACTED despite embedding the share key.
    # ctxs already pins TunnelClient + _Heartbeat; the trailing sleep
    # patch (Ctrl-C mid-backoff) overrides _run_patches' no-op sleep.
    with _enter(
        *ctxs,
        patch.object(qs, "_supervisor_sleep", side_effect=sleep_then_ctrl_c),
    ):
        qs.run_share(_make_args())
    combined = capsys.readouterr()
    assert SHARE_KEY not in combined.out + combined.err
    assert "reconnecting" in combined.err  # the drop path did print its line


# ─────────────────────────── install-service §5.6 ───────────────────────────


def test_install_service_refuses_without_cache_exits_2():
    with pytest.raises(SystemExit) as ei:
        qs.run_share(_make_args(install_service=True))
    assert ei.value.code == 2


def test_install_service_writes_keyless_plist(capsys, tmp_path):
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    qs.run_share(_make_args(install_service=True))
    plist_path = (
        tmp_path / "Library/LaunchAgents/com.quicksilver.node.qwen3.6-35b.plist"
    )
    assert plist_path.exists()
    data = plist_path.read_bytes()
    assert b"com.quicksilver.node.qwen3.6-35b" in data
    assert b"--quicksilver" in data
    assert SHARE_KEY.encode() not in data and b"--provider-key" not in data
    out = capsys.readouterr().out
    assert "launchctl bootstrap" in out


def test_install_service_plist_keeps_registration_origin(capsys):
    """A node minted against a custom API, installed WITHOUT the
    override baked in, would restart against the default origin and
    reject its own cached relay/heartbeat hosts in a KeepAlive loop."""
    qs._save_cache(
        "qwen3.6-35b",
        dict(
            _register_payload(
                relay_url="wss://pay.staging.example/up",
                heartbeat_url="https://pay.staging.example/hb",
            ),
            alias="qwen3.6-35b",
            api_base="https://pay.staging.example",
        ),
    )
    qs.run_share(_make_args(install_service=True))
    plist_path = (
        Path.home() / "Library/LaunchAgents/com.quicksilver.node.qwen3.6-35b.plist"
    )
    argv = plistlib.loads(plist_path.read_bytes())["ProgramArguments"]
    assert "--quicksilver-api" in argv
    assert argv[argv.index("--quicksilver-api") + 1] == "https://pay.staging.example"


def test_install_service_refuses_wire_urls_job_cannot_validate():
    """The resident job re-runs the wire-URL gate every boot against
    ITS effective origin. A cache with sibling staging hosts
    (relay.staging.example under pay.staging.example) passes today's
    install but fails every start — KeepAlive hot-loops a doomed
    restart at ThrottleInterval=10s. Refuse at install instead."""
    qs._save_cache(
        "qwen3.6-35b",
        dict(
            _register_payload(
                relay_url="wss://relay.staging.example/up",
                heartbeat_url="https://hb.staging.example/hb",
            ),
            alias="qwen3.6-35b",
            api_base="https://pay.staging.example",
        ),
    )
    with pytest.raises(SystemExit) as ei:
        qs.run_share(_make_args(install_service=True))
    assert ei.value.code == 2
    plist_path = (
        Path.home() / "Library/LaunchAgents/com.quicksilver.node.qwen3.6-35b.plist"
    )
    assert not plist_path.exists()


def test_install_service_bakes_declared_flags_into_plist():
    """The KeepAlive job must reproduce the interactive run's declared
    behavior — --thinking/--port/--cors-origins/--rate-limit/
    --chat-frontend/origin override all serialize; --reregister and
    credentials never do."""
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    qs.run_share(
        _make_args(
            install_service=True,
            thinking=True,
            port=19999,
            cors_origins=["http://localhost:3000", "*"],
            rate_limit=60,
            chat_frontend="https://chat.example",
            quicksilver_api="https://pay.staging.example",
            reregister=True,  # one-shot action — must NOT be baked in
        )
    )
    argv = plistlib.loads(
        (
            Path.home() / "Library/LaunchAgents/com.quicksilver.node.qwen3.6-35b.plist"
        ).read_bytes()
    )["ProgramArguments"]

    def value_after(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert "--thinking" in argv
    assert value_after("--port") == "19999"
    assert value_after("--rate-limit") == "60"
    assert value_after("--chat-frontend") == "https://chat.example"
    assert value_after("--quicksilver-api") == "https://pay.staging.example"
    cors = argv[argv.index("--cors-origins") + 1 :]
    assert cors[:2] == ["http://localhost:3000", "*"]
    assert "--reregister" not in argv
    assert "--provider-key" not in argv and PROVIDER_KEY not in argv


def test_install_service_refuses_serve_passthrough():
    """Passthrough is free-form serve config — possibly credentials.
    Refuse loudly; silently dropping it would install a job serving
    different behavior than the command it claims to reproduce."""
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    with pytest.raises(SystemExit) as ei:
        qs.run_share(
            _make_args(install_service=True, _passthrough=["--force-spec-decode"])
        )
    assert ei.value.code == 2
    plist_path = (
        Path.home() / "Library/LaunchAgents/com.quicksilver.node.qwen3.6-35b.plist"
    )
    assert not plist_path.exists()


def test_install_service_log_fs_errors_are_actionable():
    """mkdir/touch/chmod sit on the resident-job critical path — an
    OSError (full disk, read-only volume) must exit 2 with a redacted
    line, not a raw traceback."""
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    with (
        patch.object(Path, "touch", side_effect=OSError("No space left on device")),
        pytest.raises(SystemExit) as ei,
    ):
        qs.run_share(_make_args(install_service=True))
    assert ei.value.code == 2


def test_install_service_refuses_unvalidatable_cached_origin():
    """An unparseable/illegal cached api_base used to be dropped with a
    bare try/pass — the job would then restart against the DEFAULT
    origin and KeepAlive-hot-loop rejecting its own wire hosts. Refuse
    the install instead, and write no plist."""
    qs._save_cache(
        "qwen3.6-35b",
        dict(_register_payload(), alias="qwen3.6-35b", api_base="not-a-url"),
    )
    with pytest.raises(SystemExit) as ei:
        qs.run_share(_make_args(install_service=True))
    assert ei.value.code == 2
    plist_path = (
        Path.home() / "Library/LaunchAgents/com.quicksilver.node.qwen3.6-35b.plist"
    )
    assert not plist_path.exists()


def test_cached_run_validates_against_cached_registration_origin(capsys):
    """Without an explicit --quicksilver-api, the cached origin is
    authoritative — a staging cache must not be judged by the default
    origin's allowlist (which would reject relay.staging.example)."""
    staging = dict(
        _register_payload(
            relay_url="wss://pay.staging.example/up",
            heartbeat_url="https://pay.staging.example/hb",
        ),
        alias="qwen3.6-35b",
        api_base="https://pay.staging.example",
    )
    qs._save_cache("qwen3.6-35b", staging)
    # The same wire URLs MUST be rejected when the cache carries no
    # origin binding and the default (production) origin is assumed.
    unbound = {k: v for k, v in staging.items() if k != "api_base"}
    qs._save_cache("qwen3.6-35b", unbound)
    with pytest.raises(SystemExit) as ei:
        qs.run_share(_make_args())
    assert ei.value.code == 2

    qs._save_cache("qwen3.6-35b", staging)
    tunnel = _fake_tunnel()
    serve, ctrl_c = _patched_run_env(None)

    def _no_net(*a, **k):
        raise AssertionError("network touched on cached path")

    ctxs = _run_patches(serve, lambda **kw: tunnel, ctrl_c) + (
        patch.object(qs, "_open", _no_net),
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with _enter(*ctxs):
        qs.run_share(_make_args())  # no --quicksilver-api passed
    assert "cached node" in capsys.readouterr().err


# ─────────────────────────── heartbeat interval validation ───────────────────────────


@pytest.mark.parametrize("bad", ["fast", float("nan"), float("inf"), 0, -5, None])
def test_bad_heartbeat_interval_fails_before_serve_starts(bad):
    qs._save_cache(
        "qwen3.6-35b",
        dict(_register_payload(heartbeat_interval_s=bad), alias="qwen3.6-35b"),
    )
    with pytest.raises(SystemExit) as ei:
        qs.run_share(_make_args())
    assert ei.value.code == 2


def test_bad_heartbeat_interval_from_register_response_fails_fast():
    hostile = _register_payload(heartbeat_interval_s="soon")
    with (
        patch.object(qs, "_open", lambda req, timeout=None: _FakeResp(hostile)),
        pytest.raises(SystemExit) as ei,
    ):
        qs.run_share(_make_args(provider_key=PROVIDER_KEY))
    assert ei.value.code == 2


# ─────────────────────────── explicit --rate-limit ───────────────────────────


def test_run_share_forwards_explicit_rate_limit_but_never_injects_default():
    """Default (None) must stay uncapped (gateway meters pool traffic);
    an explicit value is the user's call and must reach serve."""

    def run_with(rate_limit):
        spawned: dict = {}

        def fake_spawn(**kw):
            spawned.update(kw)
            return _serve_proc()

        def ctrl_c(*_a, **_k):
            raise KeyboardInterrupt

        ctxs = _run_patches(_serve_proc(), lambda **kw: _fake_tunnel(), ctrl_c) + (
            patch.object(
                qs,
                "_open",
                lambda req, timeout=None: _FakeResp(_register_payload()),
            ),
            patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
            patch.object(share_cli, "_spawn_serve", side_effect=fake_spawn),
        )
        with _enter(*ctxs):
            qs.run_share(_make_args(provider_key=PROVIDER_KEY, rate_limit=rate_limit))
        return spawned["extra_args"]

    extra = run_with(None)
    assert extra[:2] == ["--max-num-seqs", "2"]
    assert "--rate-limit" not in extra

    extra = run_with(60)
    assert extra[extra.index("--rate-limit") + 1] == "60"


# ─────────────────────────── plain-share byte compat ───────────────────────────


def test_plain_share_default_rate_limit_still_120():
    """§8: plain share behavior is unchanged — the argparse default moved
    to None (so quicksilver can tell it apart), so the plain path must
    re-derive 120 on its own."""
    seen: dict = {}

    def fake_spawn(**kw):
        seen.update(kw)
        p = MagicMock()
        p.poll.return_value = None
        p.wait.return_value = 0
        return p

    args = _make_args(quicksilver=False, rate_limit=None)

    def ctrl_c(*_a, **_k):
        raise KeyboardInterrupt

    tunnel = _fake_tunnel()
    with (
        patch.object(share_cli, "_maybe_confirm_download"),
        patch.object(share_cli, "_spawn_serve", side_effect=fake_spawn),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli.ws_tunnel, "wait_for_public_url", return_value=True),
        patch.object(share_cli, "_resolve_served_model_name", return_value="m"),
        patch("time.sleep", side_effect=ctrl_c),
    ):
        share_cli.share_command(args)
    extra = seen["extra_args"]
    assert "--rate-limit" in extra
    assert extra[extra.index("--rate-limit") + 1] == "120"


# ─────────────── changed-line gate: defensive lifecycle contracts ───────────────


def test_cache_non_object_is_not_reused():
    path = qs._cache_path("qwen3.6-35b")
    path.write_text("[]", encoding="utf-8")
    assert qs._load_cache("qwen3.6-35b", "qwen3.6-35b") is None


def test_save_cache_preserves_primary_error_when_tmp_cleanup_fails():
    payload = dict(_register_payload(), alias="qwen3.6-35b")
    with (
        patch.object(qs.os, "replace", side_effect=OSError("disk full")),
        patch.object(qs.os, "unlink", side_effect=OSError("cleanup failed")),
        pytest.raises(qs.QuickSilverError, match="could not write node cache"),
    ):
        qs._save_cache("qwen3.6-35b", payload)


def test_resolve_serve_hf_path_uses_profile_and_falls_back():
    import vllm_mlx.model_aliases as aliases

    with patch.object(aliases, "resolve_profile", return_value={"hf_path": "org/repo"}):
        assert qs._resolve_serve_hf_path("alias") == "org/repo"
    with patch.object(aliases, "resolve_profile", return_value=None):
        assert qs._resolve_serve_hf_path("raw/repo") == "raw/repo"


def test_wire_urls_require_a_host_even_with_secure_scheme():
    with pytest.raises(qs.QuickSilverError, match="include a host"):
        qs._validate_wire_urls(
            _register_payload(relay_url="wss:///up"), _API_ORIGIN, source="cache"
        )


def test_hardware_info_is_best_effort(monkeypatch):
    import vllm_mlx

    monkeypatch.delattr(vllm_mlx, "__version__")
    with patch.object(qs.subprocess, "run", side_effect=OSError("no sysctl")):
        assert qs._hardware_info() == {}


def test_hardware_info_parses_sysctl_results_on_every_platform():
    chip = subprocess.CompletedProcess([], 0, stdout="Apple M3 Max\n", stderr="")
    ram = subprocess.CompletedProcess([], 0, stdout=str(64 * 2**30), stderr="")
    with patch.object(qs.subprocess, "run", side_effect=[chip, ram]):
        info = qs._hardware_info()
    assert info["chip"] == "Apple M3 Max"
    assert info["ram_gb"] == 64


def test_provider_key_interactive_prompt(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    with patch.object(qs.getpass, "getpass", return_value=" qsppk-prompt "):
        assert qs._resolve_provider_key(_make_args()) == "qsppk-prompt"
    with (
        patch.object(qs.getpass, "getpass", return_value="   "),
        pytest.raises(qs.QuickSilverError, match="empty provider key"),
    ):
        qs._resolve_provider_key(_make_args())


def test_open_uses_the_no_redirect_opener():
    req = urllib.request.Request("https://pay.test")
    with patch.object(qs._URL_OPENER, "open", return_value="response") as opened:
        assert qs._open(req, 3.5) == "response"
    opened.assert_called_once_with(req, timeout=3.5)


def test_register_transport_retry_exhaustion_and_response_shapes():
    with (
        patch.object(qs, "_open", side_effect=urllib.error.URLError("offline")),
        patch.object(qs.time, "monotonic", side_effect=[0.0, 999.0]),
        pytest.raises(qs.QuickSilverError, match="unreachable"),
    ):
        qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a", "w")

    with (
        patch.object(qs, "_open", return_value=_FakeResp([])),
        pytest.raises(qs.QuickSilverError, match="non-object"),
    ):
        qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a", "w")

    payload = _register_payload(payout_account={"hostile": True})
    with patch.object(qs, "_open", return_value=_FakeResp(payload)):
        out = qs.register_node(
            "https://pay.test", PROVIDER_KEY, "qwen3.6-35b", "a", "w"
        )
    assert "payout_account" not in out


def test_heartbeat_thread_controls_and_inflight_failure():
    hb = _hb()
    with patch.object(qs.threading, "Thread") as thread_cls:
        hb.start()
    thread_cls.assert_called_once()
    thread_cls.return_value.start.assert_called_once()

    with patch.object(hb, "_beat_once") as beat:
        hb.beat_now()
    beat.assert_called_once()

    stopped = _hb()
    stopped._stop.set()
    stopped._beat_once()

    bad = qs._Heartbeat("https://hb.test", SHARE_KEY, 1, lambda: 1 / 0)
    requests = []

    def record(req, timeout=None):
        requests.append(req)
        return _FakeResp({})

    with patch.object(qs, "_open", record):
        bad._beat_once()
    assert json.loads(requests[0].data)["inflight"] == 0


def test_heartbeat_run_pauses_then_stops_on_fatal():
    hb = _hb()
    hb.enabled.set()
    hb._stop.wait = MagicMock(return_value=False)  # type: ignore[method-assign]
    beat = MagicMock(side_effect=lambda: hb.fatal.set())
    hb._beat_once = beat  # type: ignore[method-assign]
    hb._run()
    beat.assert_called_once()


def test_heartbeat_run_pauses_while_tunnel_is_disabled():
    hb = _hb()
    hb._stop.wait = MagicMock(return_value=False)  # type: ignore[method-assign]
    hb.enabled.is_set = MagicMock(side_effect=[False, True])  # type: ignore[method-assign]
    hb._beat_once = MagicMock(side_effect=lambda: hb.fatal.set())  # type: ignore[method-assign]
    hb._run()
    hb._beat_once.assert_called_once()


def test_warmup_warns_on_http_failure_and_exception(capsys):
    with patch.object(qs, "_open", return_value=_FakeResp({}, status=503)):
        qs._warmup(1, "local-key", "m")
    assert "HTTP 503" in capsys.readouterr().err
    with patch.object(qs, "_open", side_effect=OSError("offline")):
        qs._warmup(1, "local-key", "m")
    assert "warm-up failed" in capsys.readouterr().err


def test_install_service_catalog_alias_and_plist_write_error():
    qs._save_cache(
        "qwen3.8-27b",
        dict(
            _register_payload(model="qwen3.8-27b", worker="test-worker"),
            alias="qwen3.8-27b-4bit",
        ),
    )
    args = _make_args(
        model="qwen3.8-27b-4bit",
        quicksilver_model="qwen3.8-27b",
        install_service=True,
    )
    qs.run_share(args)
    plist = plistlib.loads(
        (
            Path.home() / "Library/LaunchAgents/com.quicksilver.node.qwen3.8-27b.plist"
        ).read_bytes()
    )
    argv = plist["ProgramArguments"]
    assert argv[argv.index("--quicksilver-model") + 1] == "qwen3.8-27b"

    with (
        patch.object(Path, "write_bytes", side_effect=OSError("read only")),
        pytest.raises(qs.QuickSilverError, match="could not write"),
    ):
        qs.install_service(args, "qwen3.8-27b", "qwen3.8-27b-4bit")


def test_ws_connect_kwargs_legacy_and_uninspectable():
    import inspect

    client = ws_tunnel.TunnelClient(local_port=1)
    legacy = MagicMock()
    legacy.__signature__ = inspect.Signature(
        [inspect.Parameter("extra_headers", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    with patch.object(ws_tunnel.websockets, "connect", legacy):
        assert "extra_headers" in client._connect_kwargs({"A": "B"})
    with patch.object(ws_tunnel.inspect, "signature", side_effect=TypeError("opaque")):
        assert "additional_headers" in client._connect_kwargs({"A": "B"})


def test_hard_close_suppresses_close_failure_and_one_shot_can_connect():
    import http.client

    conn = MagicMock()
    conn.sock = None
    conn.close.side_effect = RuntimeError("already dead")
    ws_tunnel._hard_close(conn)
    live = ws_tunnel.OneShotHTTPConnection("127.0.0.1", 1)
    with patch.object(http.client.HTTPConnection, "connect") as connect:
        live.connect()
    connect.assert_called_once()


def test_dispatch_request_tracks_epoch_and_completed_map_is_bounded():
    async def exercise():
        client = ws_tunnel.TunnelClient(local_port=1)
        client._serve_request = MagicMock(  # type: ignore[method-assign]
            return_value=asyncio.sleep(0)
        )
        client._dispatch_inbound(_req_msg({"stream": False}))
        assert client._pending["r1"] == 1
        await asyncio.gather(*list(client._tasks))
        assert "r1" not in client._pending

        client._completed = {f"old-{i}": 0.0 for i in range(4096)}
        conn = MagicMock()
        with (
            patch.object(ws_tunnel, "OneShotHTTPConnection", return_value=conn),
            patch.object(client, "_sync_send"),
        ):
            conn.getresponse.return_value.getheaders.return_value = []
            conn.getresponse.return_value.read1.return_value = b""
            client._perform_local_fetch("fresh", "GET", "/v1/models", {}, b"")
        assert len(client._completed) == 4096 and "fresh" in client._completed

    asyncio.run(exercise())


def test_usage_injection_keeps_already_requested_body_identical():
    body = json.dumps(
        {"stream": True, "stream_options": {"include_usage": True}}
    ).encode()
    assert ws_tunnel._inject_stream_options_usage("/v1/chat/completions", body) is body


def test_connect_cancellable_closes_same_tick_socket_and_cancels_on_wait_error():
    class Socket:
        closed = False

        async def close(self):
            self.closed = True

    async def same_tick():
        socket = Socket()

        async def connect(*_a, **_k):
            return socket

        client = ws_tunnel.TunnelClient(local_port=1)
        client._closed.set()
        with (
            patch.object(ws_tunnel.websockets, "connect", connect),
            pytest.raises(ConnectionError, match="stopped during connect"),
        ):
            await client._connect_cancellable("wss://r.test")
        assert socket.closed

    async def wait_error():
        pending = asyncio.Event()

        async def connect(*_a, **_k):
            await pending.wait()

        client = ws_tunnel.TunnelClient(local_port=1)
        with (
            patch.object(ws_tunnel.websockets, "connect", connect),
            patch.object(asyncio, "wait", side_effect=RuntimeError("wait failed")),
            pytest.raises(RuntimeError, match="wait failed"),
        ):
            await client._connect_cancellable("wss://r.test")

    asyncio.run(same_tick())
    asyncio.run(wait_error())


def test_cached_invalid_api_origin_is_actionable_before_serve():
    qs._save_cache(
        "qwen3.6-35b",
        dict(_register_payload(), alias="qwen3.6-35b", api_base="not-a-url"),
    )
    with pytest.raises(SystemExit) as exc:
        qs.run_share(_make_args())
    assert exc.value.code == 2


def test_run_share_port_health_and_auth_startup_failures(capsys):
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))

    with (
        patch.object(share_cli, "_maybe_confirm_download"),
        patch.object(share_cli, "_pick_port", side_effect=RuntimeError("no port")),
        pytest.raises(SystemExit) as exc,
    ):
        qs.run_share(_make_args())
    assert exc.value.code == 1 and "no port" in capsys.readouterr().err

    for failed, expected in (
        ("health", "before becoming ready"),
        ("auth", "authenticated"),
    ):
        serve = _serve_proc()
        with (
            patch.object(share_cli, "_maybe_confirm_download"),
            patch.object(share_cli, "_pick_port", return_value=18765),
            patch.object(share_cli, "_spawn_serve", return_value=serve),
            patch.object(
                share_cli, "_wait_for_healthz", return_value=failed != "health"
            ),
            patch.object(share_cli, "_verify_auth_gate", return_value=failed != "auth"),
            pytest.raises(SystemExit) as exc,
        ):
            qs.run_share(_make_args())
        assert exc.value.code == 1 and expected in capsys.readouterr().err


def test_run_share_signal_handler_and_cleanup_defenses():
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    tunnel = _fake_tunnel()
    tunnel.run_in_thread.return_value.is_alive.return_value = True
    serve, ctrl_c = _patched_run_env(None)
    serve.wait.side_effect = [subprocess.TimeoutExpired("serve", 5), OSError("gone")]
    handlers = []

    def signal_side_effect(_sig, handler):
        handlers.append(handler)
        if len(handlers) == 2:
            raise ValueError("not main thread")
        if len(handlers) == 3:
            raise TypeError("invalid restore")
        return "original"

    ctxs = _run_patches(serve, lambda **kw: tunnel, ctrl_c) + (
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
        patch.object(qs.signal, "signal", side_effect=signal_side_effect),
    )
    with _enter(*ctxs):
        qs.run_share(_make_args())
    with pytest.raises(KeyboardInterrupt):
        handlers[0](None, None)
    tunnel.run_in_thread.return_value.join.assert_called()
    serve.kill.assert_called_once()


def test_run_share_cleanup_tolerates_terminate_oserror():
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    tunnel = _fake_tunnel()
    serve, ctrl_c = _patched_run_env(None)
    serve.terminate.side_effect = OSError("already gone")
    ctxs = _run_patches(serve, lambda **kw: tunnel, ctrl_c) + (
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with _enter(*ctxs):
        qs.run_share(_make_args())


def test_run_share_propagates_serve_exit_and_resets_after_healthy_connection():
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    exited = _serve_proc()
    exited.poll.return_value = 7
    tunnel = _fake_tunnel()
    ctxs = _run_patches(exited, lambda **kw: tunnel, lambda *_a, **_k: None) + (
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with pytest.raises(SystemExit) as exc, _enter(*ctxs):
        qs.run_share(_make_args())
    assert exc.value.code == 7

    healthy_drop = _fake_tunnel(closed=True)
    serve = _serve_proc()
    sleeps = []

    def sleep_then_stop(seconds):
        sleeps.append(seconds)
        raise KeyboardInterrupt

    ctxs = _run_patches(serve, lambda **kw: healthy_drop, sleep_then_stop) + (
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
        patch.object(
            qs.time,
            "monotonic",
            side_effect=[100.0, 100.0 + qs._FAST_REJECT_WINDOW_SECONDS + 1],
        ),
    )
    with _enter(*ctxs):
        qs.run_share(_make_args())
    assert sleeps == [1.0]


def test_ws_run_ignores_non_protocol_frames_and_cancels_request_tasks():
    class Frames:
        close_code = None

        def __init__(self):
            self.frames = iter([b"binary", "not-json", json.dumps(_req_msg())])

        async def send(self, _msg):
            pass

        async def close(self):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.frames)
            except StopIteration:
                raise StopAsyncIteration from None

    async def exercise():
        client = ws_tunnel.TunnelClient(local_port=1)

        async def pending(_msg):
            await asyncio.Event().wait()

        client._handle_request = pending  # type: ignore[method-assign]

        async def connect(*_a, **_k):
            return Frames()

        with patch.object(ws_tunnel.websockets, "connect", connect):
            await client.run()
        assert client.closed_event.is_set()

    asyncio.run(exercise())


def test_ws_exception_falls_back_to_socket_close_code():
    class PeerError(Exception):
        rcvd = None

    class Socket:
        close_code = 1008

        async def send(self, _msg):
            pass

        async def close(self):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise PeerError("peer failed")

    async def connect(*_a, **_k):
        return Socket()

    client = ws_tunnel.TunnelClient(local_port=1)
    with (
        patch.object(ws_tunnel.websockets, "connect", connect),
        pytest.raises(PeerError),
    ):
        asyncio.run(client.run())
    assert client.close_code == 1008
