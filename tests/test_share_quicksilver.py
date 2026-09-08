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
import plistlib
import stat
import threading
import urllib.error
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

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _register_payload(**over) -> dict:
    payload = {
        "node_id": "qspnode-1a2b3c4d",
        "share_key": SHARE_KEY,
        "model": "qwen3.6-35b",
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


def test_greeting_with_share_key_carries_key_not_url():
    client = ws_tunnel.TunnelClient(local_port=1, share_key=SHARE_KEY)
    assert client._greeting() == {"t": "ready", "v": 1, "key": SHARE_KEY}
    # The claim URL must not gain the key (§6: no key in logs/reprs).
    assert SHARE_KEY not in client.public_url


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
    """Post-upgrade key rejection reaches the client as a close frame
    (greeting-key design → no HTTP 401 possible). The code must
    surface for the supervisor's terminal check."""

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
    assert client._aborted == set() and client._active == {}


def test_late_abort_after_completion_is_harmless_and_bounded():
    """An abort for an already-finished fetch has nothing to cancel;
    the stray marker is recorded (set is capped at 4096 internally so
    multi-day nodes can't creep) but never consumed."""
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
    client._dispatch_inbound({"t": "abort", "id": "r1"})  # too late, harmless
    assert client._aborted == {"r1"}
    for i in range(5000):  # cap holds
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
        out = qs.register_node("https://pay.test", PROVIDER_KEY, "qwen3.6-35b", "al")
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
        qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a")


def test_register_retries_429_then_succeeds():
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(429)
        return _FakeResp(_register_payload())

    with (
        patch.object(qs, "_open", fake_urlopen),
        patch("time.sleep") as sleep,
    ):
        out = qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a")
    assert out["node_id"] == "qspnode-1a2b3c4d"
    assert sleep.call_count == 1


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
        qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a")
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
            patch("time.sleep") as sleep,
            pytest.raises(qs.QuickSilverError, match=f"HTTP {code}"),
        ):
            qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a")
        return calls["n"], sleep

    for code in (400, 404):
        n, sleep = one(code)
        assert n == 1
        sleep.assert_not_called()


def test_register_unparseable_2xx_is_actionable_error():
    class _GarbageResp:
        status = 200

        def read(self):
            return b"<html>nginx is not json</html>"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    with (
        patch.object(qs, "_open", lambda req, timeout=None: _GarbageResp()),
        pytest.raises(qs.QuickSilverError, match="unparseable"),
    ):
        qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a")


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
        qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a")
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


def test_provider_key_resolution_order(monkeypatch):
    monkeypatch.setenv(qs.PROVIDER_KEY_ENV_VAR, "qsppk-env")
    assert qs._resolve_provider_key(_make_args(provider_key=None)) == "qsppk-env"
    assert (
        qs._resolve_provider_key(_make_args(provider_key="qsppk-flag")) == "qsppk-flag"
    )


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
    # A ws:// relay would carry the greeting-frame share-key in clear.
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
        patch("time.sleep", side_effect=sleep_side_effect),
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
    # Injection skipped — the user's own --max-num-seqs passthrough wins.
    assert spawned["extra_args"] == ["--max-num-seqs", "8"]


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
    """The greeting-frame key design makes HTTP-401-at-upgrade
    impossible — a revoked key must be caught post-upgrade via the
    policy close code, with the same no-spin exit (§5.5)."""
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


def test_run_share_instant_drop_streak_becomes_terminal(capsys):
    """Five relay-accepted-then-instantly-dropped sessions in a row is
    indistinguishable from rejection — stop instead of spinning under
    KeepAlive forever."""
    qs._save_cache("qwen3.6-35b", dict(_register_payload(), alias="qwen3.6-35b"))
    serve, _ = _patched_run_env(None)
    made: list = []

    def factory(**kw):
        made.append(kw)
        # connected, closed_event pre-set, unclassifiable close
        return _fake_tunnel(ready=True, closed=True, close_code=1001)

    ctxs = _run_patches(serve, factory, lambda *a, **k: None) + (
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with pytest.raises(SystemExit) as ei, _enter(*ctxs):
        qs.run_share(_make_args())
    assert ei.value.code == 1
    assert len(made) == qs._FAST_REJECT_STREAK_LIMIT
    assert "rejected" in capsys.readouterr().err


def test_run_share_server_echoed_key_never_printed(capsys):
    """node_id/payout_account are server-controlled text — the secret
    registry must be armed before ANY of it reaches a sink (§8 bullet
    7 holds even against a malicious API)."""
    leaky = _register_payload(
        node_id=f"qspnode-{SHARE_KEY}", payout_account=f"a***@x*** {SHARE_KEY}"
    )
    serve, ctrl_c = _patched_run_env(None)
    ctxs = _run_patches(serve, lambda **kw: _fake_tunnel(), ctrl_c) + (
        patch.object(qs, "_open", lambda req, timeout=None: _FakeResp(leaky)),
        patch.object(qs, "_Heartbeat", return_value=_fake_heartbeat_class()),
    )
    with _enter(*ctxs):
        qs.run_share(_make_args(provider_key=PROVIDER_KEY))
    out = capsys.readouterr()
    combined = out.out + out.err
    assert SHARE_KEY not in combined
    assert "[redacted]" in combined


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
    with _enter(*ctxs, patch("time.sleep", side_effect=sleep_then_ctrl_c)):
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
                relay_url="wss://relay.staging.example/up",
                heartbeat_url="https://hb.staging.example/hb",
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
