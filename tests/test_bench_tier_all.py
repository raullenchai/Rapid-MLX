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
    yield {"base_url": f"http://127.0.0.1:{port}/v1", "port": port}


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
        patch("vllm_mlx.doctor.server.serve", _fake_serve),
    ):
        yield


def test_all_runs_smoke_speed_harness_in_order(patch_serve_only, capsys):
    """Happy path: all 3 tiers run in smoke → speed → harness order."""
    call_order: list[str] = []

    def _smoke_stub(model, port):
        call_order.append("smoke")
        return TierResult(
            name="smoke", passed=True, duration_s=0.5, detail="PASS"
        )

    def _speed_stub(model, port, sampled=False):
        call_order.append("speed")
        return TierResult(
            name="speed", passed=True, duration_s=10.0, detail="PASS tps=42.0"
        )

    def _harness_stub(model, port):
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

    def _smoke_fail(model, port):
        call_order.append("smoke")
        return TierResult(
            name="smoke",
            passed=False,
            duration_s=0.5,
            detail="FAIL no '4' in response",
        )

    def _speed_should_not_run(model, port, sampled=False):
        call_order.append("speed")
        return TierResult(
            name="speed", passed=True, duration_s=10.0, detail="should not run"
        )

    def _harness_should_not_run(model, port):
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

    def _smoke_stub(model, port):
        call_order.append("smoke")
        return TierResult(name="smoke", passed=True, duration_s=0.5)

    def _speed_fail(model, port, sampled=False):
        call_order.append("speed")
        return TierResult(
            name="speed", passed=False, duration_s=10.0, detail="FAIL HTTP 500"
        )

    def _harness_stub(model, port):
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
        yield {"base_url": f"http://127.0.0.1:{port}/v1", "port": port}

    def _free_port(lo, hi):
        return 8500

    def _stub(model, port, **kwargs):
        return TierResult(name="x", passed=True, duration_s=0.1)

    def _smoke_stub(model, port):
        return TierResult(name="smoke", passed=True, duration_s=0.1)

    def _speed_stub(model, port, sampled=False):
        return TierResult(name="speed", passed=True, duration_s=0.1)

    def _harness_stub(model, port):
        return TierResult(name="harness", passed=True, duration_s=0.1)

    with (
        patch(
            "vllm_mlx.bench.tier_runner._find_free_port_in_range",
            side_effect=_free_port,
        ),
        patch("vllm_mlx.doctor.server.serve", _counting_serve),
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
        yield {"base_url": f"http://127.0.0.1:{port}/v1", "port": port}

    def _smoke_stub(model, port):
        return TierResult(name="smoke", passed=True, duration_s=0.1)

    def _speed_stub(model, port, sampled=False):
        return TierResult(name="speed", passed=True, duration_s=0.1)

    def _harness_stub(model, port):
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
        patch("vllm_mlx.doctor.server.serve", _counting_serve),
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
