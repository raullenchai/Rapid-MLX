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
import stat
import threading
import urllib.error
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


def test_error_status_captured_from_failed_handshake():
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
    assert qs._load_cache("qwen3.6-35b", "qwen3.6-35b") == payload


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

    with patch.object(qs.urllib.request, "urlopen", fake_urlopen):
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
        patch.object(qs.urllib.request, "urlopen", fake_urlopen),
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
        patch.object(qs.urllib.request, "urlopen", fake_urlopen),
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
        patch.object(qs.urllib.request, "urlopen", fake_urlopen),
        pytest.raises(qs.QuickSilverError) as ei,
    ):
        qs.register_node("https://pay.test", PROVIDER_KEY, "m", "a")
    assert SHARE_KEY not in str(ei.value)


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


# ───────────────────────────── heartbeat §3.3 ───────────────────────────


def _hb(**over) -> qs._Heartbeat:
    return qs._Heartbeat("https://hb.test/hb", SHARE_KEY, 10, lambda: 3)


def test_heartbeat_200_ok_carries_inflight_and_never_key_in_body():
    hb = _hb()
    reqs = []

    def fake_urlopen(req, timeout=None):
        reqs.append(req)
        return _FakeResp({"ok": True})

    with patch.object(qs.urllib.request, "urlopen", fake_urlopen):
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
        patch.object(qs.urllib.request, "urlopen", fake_urlopen),
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

    with patch.object(qs.urllib.request, "urlopen", fake_urlopen):
        hb._beat_once()
    assert hb.fatal.is_set()


def test_heartbeat_transport_error_is_transparent():
    hb = _hb()

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("unreachable")

    with patch.object(qs.urllib.request, "urlopen", fake_urlopen):
        hb._beat_once()  # must not raise, must not go fatal
    assert not hb.fatal.is_set()


# ─────────────────────────── run_share integration ───────────────────────────


def _fake_tunnel(ready=True, closed=False, error=None, error_status=None):
    t = MagicMock()
    t.error = error
    t.error_status = error_status
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
            qs.urllib.request,
            "urlopen",
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
        patch.object(qs.urllib.request, "urlopen", _no_net),
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
        patch.object(qs.urllib.request, "urlopen", fake_urlopen),
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
            qs.urllib.request,
            "urlopen",
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
            qs.urllib.request,
            "urlopen",
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
