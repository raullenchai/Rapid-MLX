# SPDX-License-Identifier: Apache-2.0
"""Adversarial pool observers: real HTTP parsing, deterministic clocks/events."""

import http.client
import io
import threading
import time
import urllib.request
from email.message import Message
from unittest.mock import MagicMock

import pytest

from rapid_mlx import _download_gate as gate
from rapid_mlx.share import cli
from rapid_mlx.share import quicksilver as qs
from tests import test_share_progress as progress_tests
from tests.test_share_progress import metrics, metrics_response

clock = progress_tests.clock
status = progress_tests.status


class WireSocket:
    def __init__(self, wire):
        self.wire = wire

    def makefile(self, *args):
        return io.BytesIO(self.wire)


def wire_response(body, *, code=200, content_type="text/plain", length=None):
    wire = (
        f"HTTP/1.1 {code} Test\r\nContent-Type: {content_type}\r\n"
        f"Content-Length: {len(body) if length is None else length}\r\n\r\n"
    ).encode() + body
    response = http.client.HTTPResponse(WireSocket(wire))
    response.begin()
    return response


@pytest.mark.parametrize(
    "case", ["valid", "partial", "non200", "html", "huge", "encoding", "missing-type"]
)
def test_real_http_response_validation(monkeypatch, capsys, case):
    body = metrics(2, 18, 60, 2).encode()
    kwargs = {}
    if case == "partial":
        kwargs["length"] = len(body) + 10  # All four families, but truncated on wire.
    elif case == "non200":
        kwargs["code"] = 503
    elif case == "html":
        kwargs["content_type"] = "text/html"
    elif case == "huge":
        body += b"x" * (256 * 1024)
    elif case == "encoding":
        body += b"\xff"
    response = wire_response(body, **kwargs)
    if case == "missing-type":
        del response.headers["Content-Type"]
    monkeypatch.setattr(qs, "_open", lambda *a, **kw: response)
    reporter = qs._PoolStatus(1, lambda: True)
    assert reporter.baseline == ((2, 18, 60, 2) if case == "valid" else None)
    assert response.isclosed()
    assert capsys.readouterr() == ("", "")


def test_metrics_uses_local_auth_and_refuses_real_urllib_redirect(monkeypatch, capsys):
    requests = []

    class HTTP(urllib.request.HTTPHandler):
        def http_open(self, req):
            requests.append(req)
            headers = Message()
            headers["Location"] = "http://hostile.invalid/steal"
            response = urllib.response.addinfourl(
                io.BytesIO(b"private"), headers, req.full_url, 302
            )
            response.msg = "Found"
            return response

    monkeypatch.setattr(
        qs, "_URL_OPENER", urllib.request.build_opener(HTTP(), qs._NoRedirectHandler())
    )
    reporter = qs._PoolStatus(18765, lambda: True, api_key="local-private-key")
    assert reporter.baseline is None
    assert len(requests) == 1
    assert requests[0].full_url == "http://127.0.0.1:18765/metrics"
    assert requests[0].get_header("Authorization") == "Bearer local-private-key"
    assert capsys.readouterr() == ("", "")


def test_stalled_scrape_deadline_and_only_one_inflight(monkeypatch, capsys):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def drip(*a, **kw):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return metrics_response(metrics().encode())

    monkeypatch.setattr(qs, "_open", drip)
    try:
        reporter = qs._PoolStatus(1, lambda: True)
        assert entered.is_set()
        assert reporter.baseline is None  # Constructor returns before drip stops.
        assert reporter._scrape_thread.is_alive()
        assert reporter._scrape_thread.daemon
        for _ in range(100):
            assert reporter._read() is None
        assert calls == [1]
    finally:
        release.set()
    reporter._scrape_thread.join(2)
    assert not reporter._scrape_thread.is_alive()
    assert capsys.readouterr() == ("", "")


def test_shutdown_never_joins_status_scrape(monkeypatch):
    monkeypatch.setattr(qs._PoolStatus, "_read", lambda self: (0, 0, 0, 0))
    reporter = qs._PoolStatus(1, lambda: True)
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        assert release.wait(5)
        return (1, 10, 20, 1)

    monkeypatch.setattr(reporter, "_read", blocked)
    hb = qs._Heartbeat(
        "https://heartbeat.invalid",
        "share-private-key",
        3600,
        lambda: 0,
        status=reporter,
    )
    hb.start()
    scraper = threading.Thread(target=reporter.poll, args=(True,), daemon=True)
    scraper.start()
    try:
        assert entered.wait(2)
        hb.stop()
        assert scraper.is_alive()  # Shutdown returned while the scrape is blocked.
        assert not hb._thread.is_alive()
        reporter._thread.join(2)
        assert not reporter._thread.is_alive()
    finally:
        release.set()
        scraper.join(2)
    assert reporter.last_print is None


@pytest.mark.parametrize("inflight_poll", [False, True])
def test_final_status_scrape_and_poll_share_one_shutdown_bound(
    monkeypatch, capsys, inflight_poll
):
    monkeypatch.setattr(
        qs, "_open", lambda *a, **kw: metrics_response(metrics().encode())
    )
    reporter = qs._PoolStatus(1, lambda: True)
    reporter._stop = threading.Event()
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        release.wait(5)
        return (1, 10, 20, 1)

    # Exercise the real deadline/daemon implementation, including a peer that
    # returns valid metrics only after shutdown has already skipped the scrape.
    monkeypatch.setattr(reporter, "_scrape", blocked)
    poller = None
    if inflight_poll:
        poller = threading.Thread(target=reporter.poll, args=(True,))
        poller.start()
        assert entered.wait(2)
    try:
        started = time.monotonic()
        reporter.stop(MagicMock(poll=MagicMock(return_value=None)))
        elapsed = time.monotonic() - started
        assert elapsed < 1.25  # One-second deadline, with scheduler tolerance.
        assert entered.is_set()
        assert reporter._stop.is_set()
        assert reporter._scrape_thread.daemon
        assert reporter._scrape_thread.is_alive()
        assert reporter.last_print is None
        assert capsys.readouterr() == ("", "")
    finally:
        release.set()
        reporter._scrape_thread.join(2)
        if poller is not None:
            poller.join(2)
    assert capsys.readouterr() == ("", "")


def test_final_status_skips_scrape_when_poll_holds_lock(status, monkeypatch, capsys):
    reporter, _ = status
    reporter._stop = threading.Event()
    entered, release = threading.Event(), threading.Event()

    def blocked_read(**kw):
        entered.set()
        assert release.wait(5)
        return (2, 18, 60, 2)

    read = MagicMock(side_effect=blocked_read)
    monkeypatch.setattr(reporter, "_read", read)
    poller = threading.Thread(target=reporter.poll, args=(True,))
    poller.start()
    try:
        assert entered.wait(2)
        started = time.perf_counter()
        reporter.stop(MagicMock(poll=MagicMock(return_value=None)))
        assert time.perf_counter() - started < 1.25
        assert poller.is_alive()
        assert reporter._stop.is_set()
        read.assert_called_once_with()
    finally:
        release.set()
        poller.join(2)
    assert not poller.is_alive()
    assert not reporter._poll_lock.locked()
    assert reporter._totals == (0, 0, 0, 0)
    assert reporter.last_print is None
    assert capsys.readouterr() == ("", "")


def test_download_initial_cache_scan_is_bounded_and_reused(
    monkeypatch, tmp_path, clock, capsys
):
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked(self):
        calls.append(1)
        entered.set()
        release.wait(5)
        return 3 * 1024**3

    monkeypatch.setattr(cli._DownloadProgress, "_cache_size", blocked)
    reporter = None
    try:
        started = time.perf_counter()  # The progress clock is deterministic.
        reporter = cli._DownloadProgress("org/repo", 15 * 1024**3)
        assert time.perf_counter() - started < 0.5
        assert entered.is_set()
        assert reporter._scan_thread.daemon and reporter._scan_thread.is_alive()
        assert capsys.readouterr().err == (
            "Model download: approximately 15.0 GiB in total. "
            "Cache size every 30s until ready.\n"
        )
        clock[0] = 130
        reporter.poll()
        assert calls == [1]
        assert "model cache size unavailable" in capsys.readouterr().err
    finally:
        release.set()
        if reporter is not None:
            reporter._scan_thread.join(2)
    clock[0] = 160
    reporter.poll()
    assert "model cache size 3.0 GiB" in capsys.readouterr().err


@pytest.mark.parametrize("source", ["callback", "read", "stderr"])
def test_reporter_exceptions_never_emit_private_traceback(
    status, monkeypatch, capsys, source
):
    reporter, _ = status
    secret = "prompt completion customer-id api-key node-key share-key"

    def fail(*a, **kw):
        raise RuntimeError(secret)

    stop = MagicMock()
    stop.wait.side_effect = [False, True]
    stop.is_set.return_value = False
    reporter._stop = stop
    if source == "callback":
        monkeypatch.setattr(reporter, "_connected_fn", fail)
    elif source == "read":
        monkeypatch.setattr(reporter, "_read", fail)
    else:
        monkeypatch.setattr(reporter, "_read", lambda: (2, 18, 60, 2))
        with monkeypatch.context() as scoped:
            scoped.setattr(qs.sys, "stderr", MagicMock(write=fail))
            reporter._run()
        assert capsys.readouterr() == ("", "")
        return
    reporter._run()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_unexpected_http_exception_is_silent(monkeypatch, capsys, error_type):
    def fail(*a, **kw):
        raise error_type("request text customer secret")

    monkeypatch.setattr(qs, "_open", fail)
    reporter = qs._PoolStatus(1, lambda: True)
    reporter.poll(True)
    assert reporter.baseline is None
    assert capsys.readouterr() == ("", "")


def test_metrics_thread_start_failure_is_silent_and_retryable(
    status, monkeypatch, capsys
):
    reporter, _ = status
    start = MagicMock(side_effect=RuntimeError("cannot start new thread"))
    with monkeypatch.context() as scoped:
        scoped.setattr(qs.threading.Thread, "start", start)
        assert reporter._read() is None
    start.assert_called_once_with()
    assert not reporter._scrape_thread.is_alive()
    assert reporter._scrape_result is None
    assert reporter._read() == reporter.baseline
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("reason", ["deadline", "stop"])
def test_metrics_join_abandons_blocked_transport(
    status, clock, monkeypatch, capsys, reason
):
    reporter, _ = status
    entered, release = threading.Event(), threading.Event()
    calls, joins = [], []
    real_join = threading.Thread.join
    stop = threading.Event()

    def transport(*a, **kw):
        calls.append(1)
        entered.set()
        release.wait()  # The finally block releases it; no timed sleep or race.
        return metrics_response(metrics(2, 18, 60, 2).encode())

    def join(thread, timeout=None):
        assert entered.wait(0.2)
        real_join(thread, timeout=0)
        joins.append(timeout)
        # Advance at the actual join boundary with a real, blocked scrape thread.
        if reason == "deadline":
            clock[0] += timeout + 0.001
        else:
            stop.set()

    monkeypatch.setattr(qs, "_open", transport)
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(qs.threading.Thread, "join", join)
            assert reporter._read(timeout=0.01, stop=stop) is None
        assert joins == [pytest.approx(0.01)]
        assert stop.is_set() == (reason == "stop")
        assert reporter._scrape_thread.is_alive()
        assert reporter._scrape_thread.daemon
        assert reporter._read() is None  # Never start a second blocked scrape.
        assert calls == [1]
        assert reporter.baseline == (1, 8, 40, 1)
        assert capsys.readouterr() == ("", "")
    finally:
        release.set()
        real_join(reporter._scrape_thread, timeout=0.2)
    assert not reporter._scrape_thread.is_alive()
    assert reporter._scrape_result == (2, 18, 60, 2)
    assert capsys.readouterr() == ("", "")


def test_final_status_skips_scrape_when_lock_acquisition_uses_deadline(
    status, clock, monkeypatch, capsys
):
    reporter, _ = status
    reporter._stop = threading.Event()
    lock = reporter._poll_lock

    def acquire(*, timeout):
        assert lock.acquire(blocking=False)
        clock[0] += timeout  # Acquiring the poll lock spent the entire bound.
        return True

    delayed_lock = MagicMock(wraps=lock)
    delayed_lock.acquire.side_effect = acquire
    read = MagicMock(wraps=reporter._read)
    monkeypatch.setattr(reporter, "_poll_lock", delayed_lock)
    monkeypatch.setattr(reporter, "_read", read)
    reporter.stop(MagicMock(poll=MagicMock(return_value=None)))
    delayed_lock.acquire.assert_called_once_with(timeout=1.0)
    delayed_lock.release.assert_called_once_with()
    assert not lock.locked()
    assert reporter._stop.is_set()
    read.assert_not_called()
    assert capsys.readouterr() == ("", "")


def test_double_start_stop_is_one_shot(status):
    reporter, _ = status
    hb = qs._Heartbeat(
        "https://heartbeat.invalid", "key", 3600, lambda: 0, status=reporter
    )
    hb.start()
    first_status, first_heartbeat = reporter._thread, hb._thread
    try:
        hb.start()
        reporter.start(threading.Event())
        assert reporter._thread is first_status
        assert hb._thread is first_heartbeat
    finally:
        hb.stop()
        hb.stop()
        first_status.join(2)
    hb.start()
    assert not first_status.is_alive()
    assert not first_heartbeat.is_alive()
    assert first_status.daemon and first_heartbeat.daemon


def test_status_start_failure_does_not_prevent_heartbeat(status, monkeypatch):
    reporter, _ = status

    def fail(*a):
        raise RuntimeError("secret thread creation error")

    monkeypatch.setattr(reporter, "start", fail)
    hb = qs._Heartbeat(
        "https://heartbeat.invalid", "key", 3600, lambda: 0, status=reporter
    )
    hb.start()
    try:
        assert hb._thread.is_alive()
    finally:
        hb.stop()


def test_counter_reset_preserves_pool_totals_and_recovers_rate(status, clock, capsys):
    reporter, snapshot = status
    snapshot[0] = metrics(101, 1008, 2040, 101)
    reporter.poll(True)
    assert "100 requests served, 1,000 output tokens" in capsys.readouterr().err
    snapshot[0] = metrics()  # Process/ledger reset.
    clock[0] = 160
    reporter.poll(True)
    assert capsys.readouterr().err == ""
    snapshot[0] = metrics(1, 5, 30, 1)
    clock[0] = 170
    reporter.poll(True)
    output = capsys.readouterr().err
    assert "101 requests served, 1,005 output tokens" in output
    assert "average 20.1 tok/s" in output
    assert reporter._totals == (101, 1005, 2030, 101)


@pytest.mark.parametrize("field", [0, 1, 3])
def test_fractional_counters_are_rejected(field):
    values = [1, 1, 1, 1]
    values[field] = 1e-300
    assert qs._PoolStatus._parse(metrics(*values)) is None


def test_malformed_required_sample_is_rejected():
    assert (
        qs._PoolStatus._parse(metrics() + "rapid_mlx_model_requests_total{\n") is None
    )


def test_large_finite_numbers_stay_bounded_and_private(status, clock, capsys):
    reporter, snapshot = status
    snapshot[0] = metrics(1e300, 1e300, 1e300, 1)
    reporter.poll(True)
    output = capsys.readouterr().err
    assert "QuickSilver status:" in output
    assert len(output) < 1500
    assert "private model" not in output
    assert "inf" not in output.lower()
    assert "nan" not in output.lower()


@pytest.mark.parametrize("field", [0, 1, 2, 3])
def test_status_total_overflow_after_counter_reset_preserves_state(
    status, clock, capsys, field
):
    reporter, snapshot = status
    values = [2, 18, 60, 2]
    values[field] = 1e308
    snapshot[0] = metrics(*values)
    reporter.poll(True)
    capsys.readouterr()
    clock[0] += 10
    snapshot[0] = metrics()  # Reset the server counters, preserving local totals.
    reporter.poll(True)
    capsys.readouterr()
    before = (
        reporter._totals,
        reporter._previous,
        reporter.last_counts,
        reporter.last_print,
    )
    clock[0] += 10
    snapshot[0] = metrics(*values)
    assert reporter._parse(snapshot[0]) is not None  # Each sample is finite.
    reporter.poll(True)  # The sum of the two finite samples overflows.
    assert (
        reporter._totals,
        reporter._previous,
        reporter.last_counts,
        reporter.last_print,
    ) == before
    assert capsys.readouterr() == ("", "")
    snapshot[0] = metrics(1, 2, 3, 1)
    clock[0] += 60
    reporter.poll(True)
    assert reporter._previous == (1, 2, 3, 1)
    output = capsys.readouterr().err.lower()
    assert "quicksilver status:" in output
    assert "inf" not in output and "nan" not in output


def test_days_of_non_tty_status_are_flushed_and_throttled(status, clock, monkeypatch):
    reporter, snapshot = status
    stream = MagicMock()
    stream.isatty.return_value = False
    monkeypatch.setattr(qs.sys, "stderr", stream)
    for now in range(100, 100 + 3 * 86400, 10):
        clock[0] = now
        snapshot[0] = metrics(now, now * 5, now * 20, now)
        reporter.poll(now % 20 == 0)  # Reconnect storm plus continually growing counts.
    writes = "".join(call.args[0] for call in stream.write.call_args_list)
    assert writes.count("QuickSilver status:") == 3 * 1440
    assert stream.flush.call_count == 3 * 1440
    assert "\r" not in writes
    assert "private model" not in writes
    assert all(len(line) < 300 for line in writes.splitlines())


def test_cache_scan_cannot_block_health_supervisor(
    monkeypatch, tmp_path, clock, capsys
):
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    reporter = cli._DownloadProgress("org/repo", None)
    capsys.readouterr()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return 10

    monkeypatch.setattr(reporter, "_cache_size", blocked)
    try:
        clock[0] = 130
        reporter.poll()
        assert entered.is_set()
        assert reporter._scan_thread.daemon and reporter._scan_thread.is_alive()
        assert "model cache size unavailable" in capsys.readouterr().err
        clock[0] = 160
        reporter.poll()
        assert calls == [1]
        proc = MagicMock(poll=MagicMock(return_value=7))
        assert not cli._wait_for_healthz(1, proc, progress=reporter)
    finally:
        release.set()
        reporter._scan_thread.join(2)
    assert "private" not in capsys.readouterr().err


@pytest.mark.parametrize("estimate", [None, 0, "error"])
def test_failed_or_empty_estimate_still_reports_unknown(
    monkeypatch, tmp_path, capsys, estimate
):
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(gate, "is_repo_cached", lambda repo: False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)

    def estimated(repo):
        if estimate == "error":
            raise RuntimeError("repo customer secret")
        return estimate

    monkeypatch.setattr(gate, "estimate_download_size_bytes", estimated)
    reporter = cli._maybe_confirm_download("org/repo", pool_mode=True)
    assert reporter is not None
    assert reporter.estimated is None
    assert (
        capsys.readouterr().err
        == "Model download: estimated size unknown. Cache size every 30s until ready.\n"
    )


def test_unreadable_cache_probe_does_not_abort_pool_share(
    monkeypatch, tmp_path, capsys
):
    """A cache probe that raises (unreadable or disconnected HF cache) is
    treated as "not cached": share continues and still reports progress."""
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)

    def probe(repo):
        raise OSError("volume private path")

    monkeypatch.setattr(gate, "is_repo_cached", probe)
    monkeypatch.setattr(gate, "estimate_download_size_bytes", lambda repo: None)
    reporter = cli._maybe_confirm_download("org/repo", pool_mode=True)
    assert reporter is not None
    err = capsys.readouterr().err
    assert err.startswith("Model download: estimated size unknown.")
    assert "private" not in err


@pytest.mark.parametrize("layout", ["home", "hub", "symlink"])
def test_hf_cache_env_is_resolved_in_fresh_process(tmp_path, layout):
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        env.pop(name, None)
    hf_home = tmp_path / "home"
    hf_home.mkdir()
    env["HF_HOME"] = str(hf_home)
    expected = hf_home / "hub"
    if layout == "hub":
        expected = tmp_path / "override"
        env["HF_HUB_CACHE"] = str(expected)
    elif layout == "symlink":
        target = tmp_path / "other-volume-layout"
        target.mkdir()
        expected.symlink_to(target, target_is_directory=True)
    blob = expected / "models--org--repo" / "blobs" / "weights.incomplete"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x" * 123)
    code = """
from pathlib import Path
from rapid_mlx.share.cli import _DownloadProgress
import sys
reporter = _DownloadProgress('org/repo', None)
assert reporter.cache_dir == Path(sys.argv[1]) / 'models--org--repo'
assert reporter._cache_size() == 123
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(expected)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "estimated size unknown" in result.stderr


def test_real_partial_cache_resumes_but_complete_cache_is_silent(
    monkeypatch, tmp_path, capsys
):
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--org--repo"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    refs = repo / "refs"
    refs.mkdir()
    (refs / "main").write_text("a" * 40)
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first shard")
    assert not gate.is_repo_cached("org/repo")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    estimate = MagicMock(return_value=None)
    monkeypatch.setattr(gate, "estimate_download_size_bytes", estimate)
    reporter = cli._maybe_confirm_download("org/repo", pool_mode=True)
    assert reporter is not None
    assert reporter._cache_size() >= len(b"first shard")
    assert "Model download:" in capsys.readouterr().err
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"second shard")
    assert gate.is_repo_cached("org/repo")
    estimate.reset_mock()
    assert cli._maybe_confirm_download("org/repo", pool_mode=True) is None
    estimate.assert_not_called()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("override", ["non-tty", "auto-pull", "chat-spawn"])
def test_plain_share_preserves_early_returns(monkeypatch, capsys, override):
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.delenv("RAPID_MLX_CHAT_SPAWN", raising=False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: override != "non-tty")
    if override != "non-tty":
        monkeypatch.setenv(
            "RAPID_MLX_AUTO_PULL"
            if override == "auto-pull"
            else "RAPID_MLX_CHAT_SPAWN",
            "1",
        )
    cached, estimate = MagicMock(), MagicMock()
    monkeypatch.setattr(gate, "is_repo_cached", cached)
    monkeypatch.setattr(gate, "estimate_download_size_bytes", estimate)
    assert cli._maybe_confirm_download("org/repo") is None
    cached.assert_not_called()
    estimate.assert_not_called()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGINT", "serve-exit"])
@pytest.mark.parametrize("phase", ["warmup", "supervisor"])
def test_real_signal_reaps_serve_with_metrics_inflight(tmp_path, signal_name, phase):
    import subprocess
    import sys

    code = r"""
import os, signal, subprocess, threading, time
from pathlib import Path
from unittest.mock import patch
from rapid_mlx.share import quicksilver as qs
from tests.test_share_quicksilver import _make_args, _register_payload, _patched_run_env, _fake_tunnel, _run_patches, _enter
from tests.test_share_progress import metrics_response, metrics
import sys
os.environ["HOME"] = sys.argv[1]
# xdist can leave SIGINT ignored in its worker process. Python preserves an
# ignored disposition across exec, so make this child behave like a foreground
# CLI before asking QuickSilver to handle a real Ctrl-C.
signal.signal(signal.SIGINT, signal.default_int_handler)
entered = threading.Event()
reads = [0]
stop_forever = threading.Event()
scrape_finished = threading.Event()
triggered_at = []
warmup_left_at = []
warmup = sys.argv[3] == 'warmup'

def transport(req, **kw):
    if not req.full_url.endswith("/metrics"):
        return metrics_response(b"")
    reads[0] += 1
    if reads[0] == 1:
        return metrics_response(metrics(1, 8, 40, 1).encode())
    if not warmup and reads[0] == 2:
        return metrics_response(metrics(2, 16, 80, 2).encode())
    entered.set()
    stop_forever.wait()  # A deliberately unresponsive endpoint; daemon only.
    scrape_finished.set()

reporters = []
original_status = qs._PoolStatus

def make_status(*a, **kw):
    status = original_status(*a, **kw)
    wait_for_warmup = status.wait_for_warmup
    def accounting_wait(*a, **kw):
        try:
            wait_for_warmup(*a, **kw)
        finally:
            warmup_left_at.append(time.monotonic())
    status.wait_for_warmup = accounting_wait
    reporters.append(status)
    return status

serve = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True)
real_terminate, real_wait = serve.terminate, serve.wait
terminated = [0]
tunnel = _fake_tunnel()
def trigger(*a):
    if not warmup:
        threading.Thread(target=reporters[0].poll, args=(True,), daemon=True).start()
    assert entered.wait(3)
    triggered_at.append(time.monotonic())
    if sys.argv[2] == "serve-exit":
        real_terminate()
        real_wait(timeout=3)
        return
    os.kill(os.getpid(), getattr(signal, sys.argv[2]))

def terminate():
    terminated[0] += 1
    real_terminate()
    os.kill(os.getpid(), signal.SIGTERM)  # Teardown must ignore a second TERM.

serve.terminate = terminate
cache = dict(_register_payload(), alias='qwen3.6-35b')
contexts = _run_patches(serve, lambda **kw:tunnel, trigger) + (
    patch.object(qs, '_load_cache', return_value=cache),
    patch('rapid_mlx.share.cli._state_dir', return_value=Path(sys.argv[1])),
    patch.object(qs, '_PoolStatus', side_effect=make_status),
    patch.object(qs, '_open', side_effect=transport),
    patch.object(original_status, '_scrape', original_status._scrape),
    patch.object(qs, '_WARMUP_ACCOUNTING_WAIT_SECONDS', 3.0),
)
if warmup:
    contexts += (patch.object(qs, '_warmup', side_effect=lambda *a:
        threading.Thread(target=trigger, daemon=True).start()),)
# _run_patches installs a scrape stub; the last patch restores the real method.
try:
    with _enter(*contexts):
        try:
            qs.run_share(_make_args())
        except SystemExit as exc:
            assert sys.argv[2] == 'serve-exit' and exc.code != 0
        assert triggered_at and time.monotonic() - triggered_at[0] < 2.5
finally:
    if serve.poll() is None:
        real_terminate()
    real_wait(timeout=3)
assert terminated[0] == (0 if sys.argv[2] == 'serve-exit' else 1)
assert serve.poll() is not None
if warmup:
    limit = 1.25 if sys.argv[2] == 'serve-exit' else 0.25
    assert warmup_left_at[0] - triggered_at[0] < limit
    assert reporters[0]._stop is None
    assert reporters[0]._thread is None
    tunnel.run_in_thread.assert_not_called()
else:
    assert reporters[0]._stop.is_set()
    assert reporters[0]._scrape_thread.is_alive()
assert reporters[0]._scrape_thread.daemon
assert not scrape_finished.is_set()  # Transport is still blocked after cleanup.
print('reaped without waiting for metrics')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), signal_name, phase],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "reaped without waiting for metrics" in result.stdout
    assert "QuickSilver status:" not in result.stderr
    assert "Traceback" not in result.stderr


def test_background_scan_keeps_last_completed_observation(
    monkeypatch, tmp_path, clock, capsys
):
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    reporter = cli._DownloadProgress("org/repo", None)
    release = threading.Event()
    monkeypatch.setattr(
        reporter, "_cache_size", lambda: release.wait(5) and 2 * 1024**3
    )
    clock[0] = 130
    reporter.poll()
    assert "unavailable" in capsys.readouterr().err
    release.set()
    reporter._scan_thread.join(2)
    release.clear()
    try:
        clock[0] = 160
        reporter.poll()
        assert reporter._scan_thread.is_alive()
        assert "model cache size 2.0 GiB" in capsys.readouterr().err
    finally:
        release.set()
        reporter._scan_thread.join(2)


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_daemon_callback_control_exceptions_are_silent(
    status, monkeypatch, capsys, error_type
):
    reporter, _ = status
    stop = MagicMock()
    stop.wait.return_value = False
    reporter._stop = stop

    def fail():
        raise error_type("private customer text")

    monkeypatch.setattr(reporter, "_connected_fn", fail)
    reporter._run()
    assert capsys.readouterr() == ("", "")


def test_offline_uncached_gate_has_sign_of_life(tmp_path):
    import os
    import subprocess
    import sys

    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_CACHE=str(tmp_path))
    code = """
from rapid_mlx.share.cli import _maybe_confirm_download
reporter = _maybe_confirm_download('pr4516-test/unknown-model', pool_mode=True)
assert reporter is not None
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "Model download:" in result.stderr
    assert "estimated size unknown" in result.stderr


def test_health_wait_has_no_deadline_for_days_of_preparation(
    monkeypatch, tmp_path, clock, capsys
):
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    reporter = cli._DownloadProgress("org/repo", None)
    capsys.readouterr()
    responses = [0]

    def health(*a, **kw):
        responses[0] += 1
        if responses[0] < 4:
            raise TimeoutError()
        return metrics_response(b"")

    monkeypatch.setattr(cli.urllib.request, "urlopen", health)
    monkeypatch.setattr(
        cli.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + 86400)
    )
    proc = MagicMock(poll=MagicMock(return_value=None))
    assert cli._wait_for_healthz(1, proc, progress=reporter)
    output = capsys.readouterr().err
    assert "259200s elapsed" in output
    assert output.count("Preparing model:") == 3
    assert "\r" not in output


class AccountingClockStop:
    """Advance the test clock only at the real accounting loop's wait boundary."""

    def __init__(self, clock, on_wait=lambda: None):
        self.clock = clock
        self.on_wait = on_wait
        self.waits = []

    def is_set(self):
        return False

    def wait(self, timeout):
        self.waits.append(timeout)
        self.clock[0] += timeout
        self.on_wait()
        return False


def test_warmup_accounting_skips_scrape_when_clock_passes_deadline(
    status, monkeypatch, capsys
):
    reporter, _ = status
    read = MagicMock(wraps=reporter._read)
    clock = MagicMock(
        side_effect=[100.0, 100.0 + qs._WARMUP_ACCOUNTING_WAIT_SECONDS + 1]
    )
    monkeypatch.setattr(reporter, "_read", read)
    monkeypatch.setattr(qs.time, "monotonic", clock)
    reporter.wait_for_warmup(threading.Event())
    assert clock.call_count == 2
    read.assert_not_called()
    assert reporter.baseline is reporter._previous is None
    assert not reporter.baseline_at_start
    assert capsys.readouterr() == ("", "")


def test_warmup_late_accounting_within_bound_is_excluded(monkeypatch, clock, capsys):
    snapshot = [metrics()]
    monkeypatch.setattr(
        qs, "_open", lambda *a, **kw: metrics_response(snapshot[0].encode())
    )
    started = threading.Event()
    reporter = qs._PoolStatus(1, lambda: True, requests_started_fn=started.is_set)

    def account_after_one_second():
        if clock[0] >= 101:
            snapshot[0] = metrics(1, 8, 40, 1)

    stop = AccountingClockStop(clock, account_after_one_second)
    reporter.wait_for_warmup(stop)
    assert stop.waits == [0.5, 0.5]
    assert reporter.baseline == (1, 8, 40, 1)
    started.set()  # First relay dispatch, still in flight.
    reporter.poll(True)
    assert capsys.readouterr() == ("", "")
    snapshot[0] = metrics(2, 18, 60, 2)
    clock[0] += 10
    reporter.poll(True)
    assert capsys.readouterr().err == (
        "QuickSilver status: 1 request served, 10 output tokens since start, "
        "average 20.0 tok/s, connected.\n"
    )


@pytest.mark.parametrize("initially_available", [False, True])
def test_warmup_unavailable_metrics_skip_or_retry_accounting_wait(
    monkeypatch, clock, initially_available
):
    reads = []

    def transport(*a, **kw):
        reads.append(1)
        if initially_available and len(reads) == 1:
            return metrics_response(metrics().encode())
        raise TimeoutError()

    monkeypatch.setattr(qs, "_open", transport)
    reporter = qs._PoolStatus(1, lambda: True)
    stop = AccountingClockStop(clock)
    reporter.wait_for_warmup(stop)
    assert stop.waits == ([0.5] * 20 if initially_available else [])
    assert clock[0] == (110 if initially_available else 100)
    assert len(reads) == (21 if initially_available else 1)
    assert reporter.baseline is reporter._previous is None
    assert not reporter.baseline_at_start


def test_warmup_transient_metrics_failure_retries_and_excludes_warmup(
    monkeypatch, clock, capsys
):
    responses = iter([metrics(), None, metrics(1, 8, 40, 1)])

    def transport(*a, **kw):
        sample = next(responses)
        if sample is None:
            raise TimeoutError()
        return metrics_response(sample.encode())

    monkeypatch.setattr(qs, "_open", transport)
    started = threading.Event()
    reporter = qs._PoolStatus(1, lambda: True, requests_started_fn=started.is_set)
    stop = AccountingClockStop(clock)
    reporter.wait_for_warmup(stop)
    assert stop.waits == [0.5]
    assert clock[0] == 100.5
    assert reporter.baseline == reporter._previous == (1, 8, 40, 1)
    assert reporter.baseline_at_start
    started.set()
    snapshot = [metrics(1, 8, 40, 1)]
    monkeypatch.setattr(
        qs, "_open", lambda *a, **kw: metrics_response(snapshot[0].encode())
    )
    reporter.poll(True)
    assert capsys.readouterr() == ("", "")
    snapshot[0] = metrics(2, 18, 60, 2)
    clock[0] += 10
    reporter.poll(True)
    assert capsys.readouterr().err == (
        "QuickSilver status: 1 request served, 10 output tokens since start, "
        "average 20.0 tok/s, connected.\n"
    )


def test_warmup_metrics_failure_for_whole_bound_uses_labelled_late_baseline(
    status, monkeypatch, clock, capsys
):
    reporter, snapshot = status
    monkeypatch.setattr(qs, "_open", MagicMock(side_effect=TimeoutError()))
    stop = AccountingClockStop(clock)
    reporter.wait_for_warmup(stop)
    assert stop.waits == [0.5] * 20
    assert clock[0] == 110
    assert reporter.baseline is reporter._previous is None
    assert not reporter.baseline_at_start
    snapshot[0] = metrics(3, 28, 80, 3)  # Omit work before metrics recover.
    monkeypatch.setattr(
        qs, "_open", lambda *a, **kw: metrics_response(snapshot[0].encode())
    )
    reporter.poll(True)
    assert reporter.baseline == reporter._previous == (3, 28, 80, 3)
    assert reporter._totals == (0, 0, 0, 0)
    assert capsys.readouterr() == ("", "")
    snapshot[0] = metrics(4, 38, 100, 4)
    clock[0] += 10
    reporter.poll(True)
    assert capsys.readouterr().err == (
        "QuickSilver status: 1 request served, 10 output tokens "
        "since metrics became available, average 20.0 tok/s, connected.\n"
    )


@pytest.mark.parametrize("blocked_scrape", [False, True])
def test_shutdown_interrupts_warmup_accounting_wait(monkeypatch, blocked_scrape):
    monkeypatch.setattr(
        qs, "_open", lambda *a, **kw: metrics_response(metrics().encode())
    )
    reporter = qs._PoolStatus(1, lambda: True)
    entered = threading.Event()
    release = threading.Event()

    class Stop(threading.Event):
        def wait(self, timeout=None):
            entered.set()
            return super().wait(timeout)

    stop = Stop()
    if blocked_scrape:

        def transport(*a, **kw):
            entered.set()
            release.wait(5)
            return metrics_response(metrics().encode())

        monkeypatch.setattr(qs, "_open", transport)
    waiter = threading.Thread(target=reporter.wait_for_warmup, args=(stop,))
    waiter.start()
    try:
        assert entered.wait(2)
        stop.set()
        waiter.join(0.25)
        assert not waiter.is_alive()
        assert reporter.baseline == reporter._previous == (0, 0, 0, 0)
        assert reporter.baseline_at_start
        if blocked_scrape:
            assert reporter._scrape_thread.is_alive()
            assert reporter._scrape_thread.daemon
    finally:
        stop.set()
        release.set()
        waiter.join(2)
        reporter._scrape_thread.join(2)


@pytest.mark.parametrize("exit_during", ["before-scrape", "scrape", "wait"])
def test_serve_exit_interrupts_warmup_accounting_wait(
    status, monkeypatch, clock, exit_during
):
    reporter, _ = status
    serve = MagicMock(poll=MagicMock(return_value=None))

    def read(**kw):
        if exit_during == "scrape":
            clock[0] += kw["timeout"]
            serve.poll.return_value = 7
        return None

    read = MagicMock(side_effect=read)
    monkeypatch.setattr(reporter, "_read", read)
    stop = AccountingClockStop(clock)
    if exit_during == "before-scrape":
        serve.poll.return_value = 7
    elif exit_during == "wait":
        stop.on_wait = lambda: setattr(serve.poll, "return_value", 7)
    reporter.wait_for_warmup(stop, serve_proc=serve)
    assert read.call_count == (0 if exit_during == "before-scrape" else 1)
    assert stop.waits == ([0.5] if exit_during == "wait" else [])
    assert clock[0] == {"before-scrape": 100, "scrape": 101, "wait": 100.5}[exit_during]
    assert reporter.baseline == reporter._previous == (1, 8, 40, 1)
    assert reporter.baseline_at_start


def test_timed_out_warmup_after_bound_and_relay_dispatch_is_known_residual(
    monkeypatch, clock, capsys
):
    snapshot = [metrics()]
    monkeypatch.setattr(
        qs, "_open", lambda *a, **kw: metrics_response(snapshot[0].encode())
    )
    started = threading.Event()
    reporter = qs._PoolStatus(1, lambda: True, requests_started_fn=started.is_set)
    monkeypatch.setattr(qs, "_open", MagicMock(side_effect=TimeoutError()))
    qs._warmup(1, "local-key", "local-model")
    assert "warm-up failed" in capsys.readouterr().err
    monkeypatch.setattr(
        qs, "_open", lambda *a, **kw: metrics_response(snapshot[0].encode())
    )
    stop = AccountingClockStop(clock)
    reporter.wait_for_warmup(stop)
    assert clock[0] == 110
    assert stop.waits == [0.5] * 20
    assert reporter.baseline == (0, 0, 0, 0)
    started.set()
    snapshot[0] = metrics(1, 8, 40, 1)  # Warm-up only; pool work still in flight.
    reporter.poll(True)
    assert capsys.readouterr().err == (
        "QuickSilver status: 1 request served, 8 output tokens since start, "
        "average 40.0 tok/s, connected.\n"
    )
    snapshot[0] = metrics(2, 18, 60, 2)
    clock[0] += 60
    reporter.poll(True)
    assert "2 requests served, 18 output tokens" in capsys.readouterr().err


def test_warmup_accounting_bound_includes_scrape_latency(monkeypatch, clock):
    monkeypatch.setattr(
        qs, "_open", lambda *a, **kw: metrics_response(metrics().encode())
    )
    reporter = qs._PoolStatus(1, lambda: True)

    def transport(*a, **kw):
        clock[0] += 0.75
        return metrics_response(metrics().encode())

    monkeypatch.setattr(qs, "_open", transport)
    stop = AccountingClockStop(clock)
    reporter.wait_for_warmup(stop)
    assert clock[0] == 110
    assert sum(stop.waits) == 4


def test_real_sigterm_interrupts_warmup_accounting_wait(monkeypatch):
    """Share's SIGTERM handler raises KeyboardInterrupt on the main thread;
    the warm-up accounting wait must abort at once, not run out its bound."""
    import os
    import signal
    import threading
    import time

    from rapid_mlx.share import quicksilver as qs

    sample = (1.0, 8.0, 40.0, 1.0)
    monkeypatch.setattr(qs._PoolStatus, "_read", lambda self, **kw: sample)
    reporter = qs._PoolStatus(18765, lambda: True)

    def term_handler(signum, frame):  # same shape as run_share's handler
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, term_handler)
    timer = threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGTERM))
    started = time.monotonic()
    try:
        timer.start()
        with pytest.raises(KeyboardInterrupt):
            reporter.wait_for_warmup()  # counter never advances: full bound
    finally:
        timer.cancel()
        signal.signal(signal.SIGTERM, previous)
    assert time.monotonic() - started < 3.0
    assert qs._WARMUP_ACCOUNTING_WAIT_SECONDS >= 5
