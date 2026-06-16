# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``rapid-mlx bench <model> --tier harness``.

Harness tier instantiates ``AgentTestRunner`` once per first-class
harness (codex, opencode, hermes, aider, langchain) and aggregates the
5 outcomes. These tests verify the iteration order, that AgentTestRunner
is invoked exactly five times against the shared booted server, and
that a single harness failure surfaces as tier=FAIL while leaving the
other 4 still runnable.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.bench.tier_runner import HARNESS_PROFILES, run_tier


@contextlib.contextmanager
def _fake_serve(model, port=None, **kwargs):
    yield {"base_url": f"http://127.0.0.1:{port}/v1", "port": port}


def _make_fake_report(*, passed=10, failed=0, errored=0, skipped=2, results=None):
    """Build a TestReport-shaped mock."""
    report = MagicMock()
    report.passed = passed
    report.failed = failed
    report.errored = errored
    report.skipped = skipped
    report.results = results or []
    return report


@pytest.fixture
def patch_harness_environment():
    """Stub the server boot + AgentTestRunner so the tier runs in-process."""

    def _free_port(lo, hi):
        return 8500

    # Track which profiles got passed to AgentTestRunner, in order.
    invocations: list[str] = []

    def _fake_runner_init(profile, base_url, model_id=None, **kwargs):
        invocations.append(profile.name)
        r = MagicMock()
        r.run.return_value = _make_fake_report()
        return r

    # Make get_profile return an object with a .name attribute matching input.
    def _fake_get_profile(name):
        p = MagicMock()
        p.name = name
        p.display_name = name.title()
        return p

    with (
        patch(
            "vllm_mlx.bench.tier_runner._find_free_port_in_range",
            side_effect=_free_port,
        ),
        patch("vllm_mlx.doctor.server.serve", _fake_serve),
        patch("vllm_mlx.agents.get_profile", _fake_get_profile),
        patch(
            "vllm_mlx.agents.testing.AgentTestRunner", side_effect=_fake_runner_init
        ),
    ):
        yield invocations


def test_harness_invokes_all_five_in_documented_order(
    patch_harness_environment, capsys
):
    """The 5 harnesses must run in the documented order."""
    rc = run_tier(model="qwen3.5-4b-4bit", tier="harness")
    invocations = patch_harness_environment

    assert rc == 0, f"all-pass harness sweep should exit 0; got {rc}"
    assert tuple(invocations) == HARNESS_PROFILES, (
        f"harness order mismatch: got {invocations}, want {HARNESS_PROFILES}"
    )

    captured = capsys.readouterr()
    # Each harness name should appear in the per-tier detail block.
    for name in HARNESS_PROFILES:
        assert name in captured.out, f"harness {name} missing from output"


def test_harness_single_failure_marks_tier_failed(capsys):
    """If hermes fails, tier exits 1 but other harnesses still run."""

    def _free_port(lo, hi):
        return 8500

    invocations: list[str] = []

    def _fake_get_profile(name):
        p = MagicMock()
        p.name = name
        p.display_name = name.title()
        return p

    # Build a runner factory that fails specifically for hermes.
    def _runner_factory(profile, base_url, model_id=None, **kwargs):
        invocations.append(profile.name)
        r = MagicMock()
        if profile.name == "hermes":
            from vllm_mlx.agents.testing import TestStatus

            bad_result = MagicMock()
            bad_result.name = "single_tool_call"
            bad_result.message = "tool name mismatch"
            bad_result.status = TestStatus.FAIL
            r.run.return_value = _make_fake_report(
                passed=4, failed=1, errored=0, skipped=2, results=[bad_result]
            )
        else:
            r.run.return_value = _make_fake_report()
        return r

    with (
        patch(
            "vllm_mlx.bench.tier_runner._find_free_port_in_range",
            side_effect=_free_port,
        ),
        patch("vllm_mlx.doctor.server.serve", _fake_serve),
        patch("vllm_mlx.agents.get_profile", _fake_get_profile),
        patch(
            "vllm_mlx.agents.testing.AgentTestRunner", side_effect=_runner_factory
        ),
    ):
        rc = run_tier(model="qwen3.5-4b-4bit", tier="harness")

    assert rc == 1, "harness with hermes FAIL should exit 1"
    assert tuple(invocations) == HARNESS_PROFILES, (
        "all 5 harnesses must still run even when one fails"
    )

    captured = capsys.readouterr()
    assert "[FAIL] tier=harness" in captured.out
    # Hermes's failing detail must be surfaced for actionable signal.
    assert "FAIL hermes" in captured.out
    assert "tool name mismatch" in captured.out


def test_harness_crash_in_runner_does_not_abort_sweep(capsys):
    """An exception inside one runner records FAIL and continues."""

    def _free_port(lo, hi):
        return 8500

    invocations: list[str] = []

    def _fake_get_profile(name):
        p = MagicMock()
        p.name = name
        p.display_name = name.title()
        return p

    def _runner_factory(profile, base_url, model_id=None, **kwargs):
        invocations.append(profile.name)
        r = MagicMock()
        if profile.name == "opencode":
            r.run.side_effect = RuntimeError("simulated parser crash")
        else:
            r.run.return_value = _make_fake_report()
        return r

    with (
        patch(
            "vllm_mlx.bench.tier_runner._find_free_port_in_range",
            side_effect=_free_port,
        ),
        patch("vllm_mlx.doctor.server.serve", _fake_serve),
        patch("vllm_mlx.agents.get_profile", _fake_get_profile),
        patch(
            "vllm_mlx.agents.testing.AgentTestRunner", side_effect=_runner_factory
        ),
    ):
        rc = run_tier(model="qwen3.5-4b-4bit", tier="harness")

    # Crash → tier-level FAIL, but every harness was visited.
    assert rc == 1
    assert tuple(invocations) == HARNESS_PROFILES
    captured = capsys.readouterr()
    assert "simulated parser crash" in captured.out


def test_harness_missing_profile_marks_as_failure(capsys):
    """If get_profile returns None, that slot fails but sweep continues."""

    def _free_port(lo, hi):
        return 8500

    def _fake_get_profile(name):
        # Return None for langchain only — simulates a missing profile.
        if name == "langchain":
            return None
        p = MagicMock()
        p.name = name
        return p

    def _runner_factory(profile, base_url, model_id=None, **kwargs):
        r = MagicMock()
        r.run.return_value = _make_fake_report()
        return r

    with (
        patch(
            "vllm_mlx.bench.tier_runner._find_free_port_in_range",
            side_effect=_free_port,
        ),
        patch("vllm_mlx.doctor.server.serve", _fake_serve),
        patch("vllm_mlx.agents.get_profile", _fake_get_profile),
        patch(
            "vllm_mlx.agents.testing.AgentTestRunner", side_effect=_runner_factory
        ),
    ):
        rc = run_tier(model="qwen3.5-4b-4bit", tier="harness")

    assert rc == 1
    captured = capsys.readouterr()
    assert "langchain" in captured.out
    assert "not found" in captured.out.lower()
