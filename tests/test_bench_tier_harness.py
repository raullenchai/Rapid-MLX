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
    yield {
        "base_url": f"http://127.0.0.1:{port}/v1",
        "port": port,
        # Schema-v2 ``smoke_result.boot_time_ms`` source — pinned so
        # tier=harness tests that incidentally hit the smoke probe
        # (e.g. tier=all) still see deterministic output.
        "boot_time_ms": 1234.5,
    }


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
    """Stub the server boot + AgentTestRunner so the tier runs in-process.

    Also stubs ``_health_check`` to always return True — without this,
    the post-#682 harness session would conclude the (mock) server is
    dead before each profile and reboot endlessly in-test, masking
    iteration-logic regressions. Cascade-restart behavior gets its own
    dedicated test (``test_harness_dead_server_between_profiles_reboots``).
    """

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
        patch("vllm_mlx.bench._server.serve", _fake_serve),
        patch("vllm_mlx.agents.get_profile", _fake_get_profile),
        patch("vllm_mlx.agents.testing.AgentTestRunner", side_effect=_fake_runner_init),
        patch("vllm_mlx.bench.tier_runner._health_check", return_value=True),
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
        patch("vllm_mlx.bench._server.serve", _fake_serve),
        patch("vllm_mlx.agents.get_profile", _fake_get_profile),
        patch("vllm_mlx.agents.testing.AgentTestRunner", side_effect=_runner_factory),
        patch("vllm_mlx.bench.tier_runner._health_check", return_value=True),
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
        patch("vllm_mlx.bench._server.serve", _fake_serve),
        patch("vllm_mlx.agents.get_profile", _fake_get_profile),
        patch("vllm_mlx.agents.testing.AgentTestRunner", side_effect=_runner_factory),
        patch("vllm_mlx.bench.tier_runner._health_check", return_value=True),
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
        patch("vllm_mlx.bench._server.serve", _fake_serve),
        patch("vllm_mlx.agents.get_profile", _fake_get_profile),
        patch("vllm_mlx.agents.testing.AgentTestRunner", side_effect=_runner_factory),
        patch("vllm_mlx.bench.tier_runner._health_check", return_value=True),
    ):
        rc = run_tier(model="qwen3.5-4b-4bit", tier="harness")

    assert rc == 1
    captured = capsys.readouterr()
    assert "langchain" in captured.out
    assert "not found" in captured.out.lower()


# ---------------------------------------------------------------------------
# Cascade-fail regression: dead server between profiles must reboot, not
# tank every later profile with ECONNREFUSED. See issue #682.
# ---------------------------------------------------------------------------


def test_harness_dead_server_between_profiles_reboots(capsys):
    """A dead /health between two profiles triggers a reboot.

    Simulates the production failure: codex passes, then the in-process
    server dies (OOM on a slow model). Pre-fix every later profile
    raised ``server_check: Rapid-MLX server not running``. Post-fix the
    session detects the dead /health and boots a fresh ``serve()`` so
    opencode/hermes/aider/langchain still get their fair shot.
    """
    import time as _time

    def _free_port(lo, hi):
        # Each restart picks a NEW port so we can count reboots.
        _free_port.calls += 1
        return 8500 + _free_port.calls

    _free_port.calls = 0  # type: ignore[attr-defined]

    serve_calls: list[int] = []

    @contextlib.contextmanager
    def _serve_recording(model, port=None, **kwargs):
        serve_calls.append(port)
        yield {
            "base_url": f"http://127.0.0.1:{port}/v1",
            "port": port,
            "boot_time_ms": 100.0,
        }

    invocations: list[str] = []

    def _runner_factory(profile, base_url, model_id=None, **kwargs):
        invocations.append(profile.name)
        r = MagicMock()
        r.run.return_value = _make_fake_report()
        return r

    def _fake_get_profile(name):
        p = MagicMock()
        p.name = name
        p.display_name = name.title()
        return p

    # Health-check sequence: True for codex's pre-check; False right
    # AFTER codex (server died); then True again so the reboot succeeds
    # AND every later profile sees a healthy server. The sequence
    # length matches the order ``_run_harness`` probes: 1 per profile.
    health_sequence = iter([True, False, True, True, True])

    def _stub_health(*args, **kwargs):
        try:
            return next(health_sequence)
        except StopIteration:
            return True

    with (
        patch(
            "vllm_mlx.bench.tier_runner._find_free_port_in_range",
            side_effect=_free_port,
        ),
        patch("vllm_mlx.bench._server.serve", _serve_recording),
        patch("vllm_mlx.agents.get_profile", _fake_get_profile),
        patch("vllm_mlx.agents.testing.AgentTestRunner", side_effect=_runner_factory),
        patch("vllm_mlx.bench.tier_runner._health_check", side_effect=_stub_health),
    ):
        t0 = _time.time()
        rc = run_tier(model="qwen3.5-4b-4bit", tier="harness")
        _ = _time.time() - t0  # touched for clarity; harness sweep is mocked

    # All 5 profiles must have been visited despite the mid-sweep death.
    assert tuple(invocations) == HARNESS_PROFILES, (
        "cascade fix must keep iterating after a server reboot; "
        f"got {invocations}"
    )

    # At least 2 serve() calls: initial boot + 1 reboot between profiles.
    assert len(serve_calls) >= 2, (
        f"expected initial boot + at least one reboot; got {len(serve_calls)} "
        f"serve() invocations"
    )

    captured = capsys.readouterr()
    # The session must announce the reboot so gauntlet operators see it.
    assert "rebooted" in captured.out or "restart" in captured.out.lower(), (
        f"expected reboot notice in tier output; got:\n{captured.out}"
    )
    # Tier exit code is 0 because we recovered cleanly.
    assert rc == 0, f"recovered sweep should pass; got rc={rc}"


def test_harness_dead_server_no_reboot_when_attached_url(capsys):
    """With ``--base-url``, the session can't restart — surfaces a FAIL.

    User attached to an externally-managed server. If THAT server dies
    mid-sweep we have no business spawning our own replacement (it would
    listen on a different port the user isn't pointing at). Each
    affected profile records a FAIL with a clear "cannot restart
    attached servers" note instead of cascading ECONNREFUSED.
    """
    invocations: list[str] = []

    def _runner_factory(profile, base_url, model_id=None, **kwargs):
        invocations.append(profile.name)
        r = MagicMock()
        r.run.return_value = _make_fake_report()
        return r

    def _fake_get_profile(name):
        p = MagicMock()
        p.name = name
        p.display_name = name.title()
        return p

    # ``_serve_or_attach`` uses ``urllib.request.urlopen`` (NOT
    # ``_health_check``) for its initial sanity ping, so every
    # ``_health_check`` call here is a per-profile probe. We want all
    # of them to return False so each profile records a server-not-
    # healthy FAIL.
    def _stub_health(*args, **kwargs):
        return False

    # Also stub the attach-time urlopen so ``_serve_or_attach`` lets us
    # in. We just need a urlopen that returns a 200-status object.
    class _FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return b""

    with (
        patch("urllib.request.urlopen", return_value=_FakeResp()),
        patch("vllm_mlx.agents.get_profile", _fake_get_profile),
        patch("vllm_mlx.agents.testing.AgentTestRunner", side_effect=_runner_factory),
        patch("vllm_mlx.bench.tier_runner._health_check", side_effect=_stub_health),
    ):
        rc = run_tier(
            model="qwen3.5-4b-4bit",
            tier="harness",
            base_url="http://127.0.0.1:9999/v1",
        )

    captured = capsys.readouterr()
    # Every profile records a server-not-healthy FAIL.
    assert "cannot restart attached" in captured.out, (
        f"expected attach-mode skip notice; got:\n{captured.out}"
    )
    # No AgentTestRunner.run() ever got dispatched because the server
    # was unhealthy before every profile.
    assert invocations == [], (
        f"attached + unhealthy → no profile should run; got {invocations}"
    )
    assert rc == 1


def test_harness_profile_timeout_does_not_block_next_profile(capsys):
    """A hung profile is killed by the per-profile timeout and the sweep continues.

    Reproduces the cause of the cascade fail: codex's e2e_file_read hung
    for 156s on a slow model. Pre-fix the runner waited indefinitely.
    Post-fix the per-profile deadline fires, the hung profile records a
    "timed out" FAIL, and the next profile (opencode, etc.) starts
    immediately.
    """
    import threading

    invocations: list[str] = []

    def _runner_factory(profile, base_url, model_id=None, **kwargs):
        invocations.append(profile.name)
        r = MagicMock()
        if profile.name == "codex":
            # Block the worker thread for longer than the timeout so the
            # deadline fires.
            def _hang():
                threading.Event().wait()  # blocks forever

            r.run.side_effect = _hang
        else:
            r.run.return_value = _make_fake_report()
        return r

    def _fake_get_profile(name):
        p = MagicMock()
        p.name = name
        p.display_name = name.title()
        return p

    def _free_port(lo, hi):
        return 8500

    with (
        patch(
            "vllm_mlx.bench.tier_runner._find_free_port_in_range",
            side_effect=_free_port,
        ),
        patch("vllm_mlx.bench._server.serve", _fake_serve),
        patch("vllm_mlx.agents.get_profile", _fake_get_profile),
        patch("vllm_mlx.agents.testing.AgentTestRunner", side_effect=_runner_factory),
        patch("vllm_mlx.bench.tier_runner._health_check", return_value=True),
        # Cap the per-profile wall-clock at 1s so the test is fast.
        patch("vllm_mlx.bench.tier_runner.HARNESS_PROFILE_TIMEOUT_S", 1),
    ):
        rc = run_tier(model="qwen3.5-4b-4bit", tier="harness")

    captured = capsys.readouterr()
    # Every profile still got tried — the hung codex didn't block the
    # next four.
    assert tuple(invocations) == HARNESS_PROFILES, (
        "per-profile timeout must let the sweep continue; "
        f"got {invocations}"
    )
    # Codex must surface as a FAIL with the timeout marker.
    assert "timed out" in captured.out, (
        f"expected per-profile timeout marker; got:\n{captured.out}"
    )
    assert "FAIL codex" in captured.out
    # Tier exits 1 because of the codex FAIL.
    assert rc == 1
