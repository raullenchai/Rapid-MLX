# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``rapid-mlx bench <model> --tier all``.

``--tier all`` runs smoke → speed → harness sequentially against a
single booted server. The hard contract: if smoke fails, abort the
rest of the run and report only the smoke result (there's no point
benching a model that can't say "4"). These tests stub the individual
tier implementations so we can verify the orchestration without
running real tier work.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from vllm_mlx.bench.tier_runner import TierResult, run_tier


@contextlib.contextmanager
def _fake_serve(model, port=None, **kwargs):
    yield {
        "base_url": f"http://127.0.0.1:{port}/v1",
        "port": port,
        # boot_time_ms threads through to ``_run_smoke`` for the
        # schema-v2 ``smoke_result.boot_time_ms`` field; pinned for
        # reproducible test output.
        "boot_time_ms": 1234.5,
    }


@pytest.fixture
def patch_serve_only():
    """Stub only the server boot; tier implementations remain real-but-mocked."""

    def _free_port(lo, hi):
        return 8500

    with (
        patch(
            "vllm_mlx.bench.tier_runner._find_free_port_in_range",
            side_effect=_free_port,
        ),
        patch("vllm_mlx.bench._server.serve", _fake_serve),
    ):
        yield


def test_all_runs_smoke_speed_harness_in_order(patch_serve_only, capsys):
    """Happy path: all 3 tiers run in smoke → speed → harness order."""
    call_order: list[str] = []

    def _smoke_stub(model, base_url, boot_time_ms=None):
        call_order.append("smoke")
        return TierResult(name="smoke", passed=True, duration_s=0.5, detail="PASS")

    def _speed_stub(model, base_url, sampled=False):
        call_order.append("speed")
        return TierResult(
            name="speed", passed=True, duration_s=10.0, detail="PASS tps=42.0"
        )

    def _harness_stub(model, base_url):
        call_order.append("harness")
        return TierResult(
            name="harness", passed=True, duration_s=30.0, detail="5/5 pass"
        )

    with (
        patch("vllm_mlx.bench.tier_runner._run_smoke", _smoke_stub),
        patch("vllm_mlx.bench.tier_runner._run_speed", _speed_stub),
        patch("vllm_mlx.bench.tier_runner._run_harness", _harness_stub),
    ):
        rc = run_tier(model="qwen3.5-4b-4bit", tier="all")

    assert rc == 0
    assert call_order == ["smoke", "speed", "harness"], (
        f"--tier all must run smoke → speed → harness; got {call_order}"
    )

    captured = capsys.readouterr()
    assert "OK: 3/3 tiers passed" in captured.out


def test_all_aborts_after_smoke_failure(patch_serve_only, capsys):
    """If smoke fails, --tier all must NOT run speed or harness."""
    call_order: list[str] = []

    def _smoke_fail(model, base_url, boot_time_ms=None):
        call_order.append("smoke")
        return TierResult(
            name="smoke",
            passed=False,
            duration_s=0.5,
            detail="FAIL no '4' in response",
        )

    def _speed_should_not_run(model, base_url, sampled=False):
        call_order.append("speed")
        return TierResult(
            name="speed", passed=True, duration_s=10.0, detail="should not run"
        )

    def _harness_should_not_run(model, base_url):
        call_order.append("harness")
        return TierResult(
            name="harness",
            passed=True,
            duration_s=30.0,
            detail="should not run",
        )

    with (
        patch("vllm_mlx.bench.tier_runner._run_smoke", _smoke_fail),
        patch("vllm_mlx.bench.tier_runner._run_speed", _speed_should_not_run),
        patch("vllm_mlx.bench.tier_runner._run_harness", _harness_should_not_run),
    ):
        rc = run_tier(model="qwen3.5-4b-4bit", tier="all")

    assert rc == 1, "--tier all with smoke FAIL must exit 1"
    assert call_order == ["smoke"], (
        f"--tier all must abort after smoke fail; got {call_order}"
    )

    captured = capsys.readouterr()
    assert "Aborting --tier all" in captured.out
    assert "smoke failed" in captured.out


def test_all_continues_past_speed_failure(patch_serve_only, capsys):
    """Speed failure does NOT abort the sweep — harness still runs."""
    call_order: list[str] = []

    def _smoke_stub(model, base_url, boot_time_ms=None):
        call_order.append("smoke")
        return TierResult(name="smoke", passed=True, duration_s=0.5)

    def _speed_fail(model, base_url, sampled=False):
        call_order.append("speed")
        return TierResult(
            name="speed", passed=False, duration_s=10.0, detail="FAIL HTTP 500"
        )

    def _harness_stub(model, base_url):
        call_order.append("harness")
        return TierResult(name="harness", passed=True, duration_s=30.0)

    with (
        patch("vllm_mlx.bench.tier_runner._run_smoke", _smoke_stub),
        patch("vllm_mlx.bench.tier_runner._run_speed", _speed_fail),
        patch("vllm_mlx.bench.tier_runner._run_harness", _harness_stub),
    ):
        rc = run_tier(model="qwen3.5-4b-4bit", tier="all")

    # Speed failed → overall fail, but harness must still have run.
    assert rc == 1
    assert call_order == ["smoke", "speed", "harness"]


def test_all_boots_server_exactly_once(capsys):
    """The server boot context manager must be entered exactly once."""
    boot_count = {"n": 0}

    @contextlib.contextmanager
    def _counting_serve(model, port=None, **kwargs):
        boot_count["n"] += 1
        yield {
            "base_url": f"http://127.0.0.1:{port}/v1",
            "port": port,
            "boot_time_ms": 1234.5,
        }

    def _free_port(lo, hi):
        return 8500

    def _stub(model, port, **kwargs):
        return TierResult(name="x", passed=True, duration_s=0.1)

    def _smoke_stub(model, base_url, boot_time_ms=None):
        return TierResult(name="smoke", passed=True, duration_s=0.1)

    def _speed_stub(model, base_url, sampled=False):
        return TierResult(name="speed", passed=True, duration_s=0.1)

    def _harness_stub(model, base_url):
        return TierResult(name="harness", passed=True, duration_s=0.1)

    with (
        patch(
            "vllm_mlx.bench.tier_runner._find_free_port_in_range",
            side_effect=_free_port,
        ),
        patch("vllm_mlx.bench._server.serve", _counting_serve),
        patch("vllm_mlx.bench.tier_runner._run_smoke", _smoke_stub),
        patch("vllm_mlx.bench.tier_runner._run_speed", _speed_stub),
        patch("vllm_mlx.bench.tier_runner._run_harness", _harness_stub),
    ):
        run_tier(model="qwen3.5-4b-4bit", tier="all")

    assert boot_count["n"] == 1, (
        f"--tier all must boot the server exactly once; booted {boot_count['n']}"
    )


def test_all_with_base_url_skips_server_boot(capsys):
    """--base-url path must NOT boot a server; just attach."""
    boot_count = {"n": 0}

    @contextlib.contextmanager
    def _counting_serve(model, port=None, **kwargs):
        boot_count["n"] += 1
        yield {
            "base_url": f"http://127.0.0.1:{port}/v1",
            "port": port,
            "boot_time_ms": 1234.5,
        }

    def _smoke_stub(model, base_url, boot_time_ms=None):
        return TierResult(name="smoke", passed=True, duration_s=0.1)

    def _speed_stub(model, base_url, sampled=False):
        return TierResult(name="speed", passed=True, duration_s=0.1)

    def _harness_stub(model, base_url):
        return TierResult(name="harness", passed=True, duration_s=0.1)

    # Stub the urlopen call so the attach health-check passes.
    class _FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(url, timeout=3):
        return _FakeResp()

    with (
        patch("vllm_mlx.bench._server.serve", _counting_serve),
        patch("vllm_mlx.bench.tier_runner._run_smoke", _smoke_stub),
        patch("vllm_mlx.bench.tier_runner._run_speed", _speed_stub),
        patch("vllm_mlx.bench.tier_runner._run_harness", _harness_stub),
        patch("urllib.request.urlopen", _fake_urlopen),
    ):
        rc = run_tier(
            model="qwen3.5-4b-4bit",
            tier="all",
            base_url="http://127.0.0.1:8000",
        )

    assert rc == 0
    assert boot_count["n"] == 0, (
        f"--base-url path must skip server boot; booted {boot_count['n']}"
    )
    captured = capsys.readouterr()
    assert "attached to existing server" in captured.out


def test_speed_tier_fails_when_server_returns_zero_tokens(capsys):
    """HTTP 200 with empty content + zero usage tokens must FAIL the tier.

    Regression coverage for codex PR #621 BLOCKING: previous revision
    returned passed=True unconditionally after HTTP 200, masking the
    "server unhealthy but route reachable" silent-failure class.
    """
    from vllm_mlx.bench.tier_runner import _run_speed

    class _FakeResp:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    class _FakeClient:
        def __init__(self, *a, **k):
            self.gets = []
            self.posts = []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            return _FakeResp({"data": [{"id": "test-model"}]})

        def post(self, url, json=None):
            # Server returns HTTP 200 but emits nothing — usage zero AND
            # message content empty. This is the silent regression class.
            return _FakeResp(
                {
                    "choices": [{"message": {"content": ""}}],
                    "usage": {"completion_tokens": 0},
                }
            )

    with patch("httpx.Client", _FakeClient):
        r = _run_speed(model="qwen3.5-4b-4bit", base_url="http://127.0.0.1:8500/v1")

    assert r.passed is False, (
        "speed tier must FAIL when server emits zero tokens and empty content"
    )
    assert "no completion tokens" in r.detail.lower() or "unhealthy" in r.detail.lower()


def test_all_with_dead_base_url_fails_fast(capsys):
    """A dead --base-url surfaces a clear error and exits non-zero."""
    import urllib.error

    def _dead_urlopen(url, timeout=3):
        raise urllib.error.URLError("connection refused")

    with patch("urllib.request.urlopen", _dead_urlopen):
        rc = run_tier(
            model="qwen3.5-4b-4bit",
            tier="all",
            base_url="http://127.0.0.1:9999",
        )

    assert rc == 1
    captured = capsys.readouterr()
    assert "not reachable" in captured.out
