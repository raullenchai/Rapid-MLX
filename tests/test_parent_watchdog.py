# SPDX-License-Identifier: Apache-2.0
"""Tests for the parent-PID watchdog (rapid-desktop issue #449).

Covers:

  * ``resolve_expected_ppid`` precedence (CLI > env > None) and the
    ``ppid <= 1`` early-out that keeps the watchdog safe to install
    unconditionally.
  * Install-time short-circuit: if the live PPID is already wrong at
    install time (supervisor died between spawn and serve_command),
    the orphan callback fires SYNCHRONOUSLY and no thread is created.
  * Loop-time detection: when the simulated live PPID flips mid-loop,
    the watchdog fires the callback exactly once and the thread exits.
  * Default ``on_orphan`` writes the canonical marker so dogfood log
    scrapers can grep one consistent string.

Test isolation
--------------

The default ``on_orphan`` SIGKILLs the test runner, so every test that
exercises the loop substitutes a no-op callback. ``ENV_VAR`` is
restored via ``monkeypatch`` so the suite doesn't leak watchdog state
into later tests in the same pytest process.
"""

from __future__ import annotations

import os
import threading
import time
from unittest.mock import patch

import pytest

from vllm_mlx import _parent_watchdog as pwd


class TestResolveExpectedPpid:
    def test_returns_none_when_neither_source_set(self, monkeypatch):
        monkeypatch.delenv(pwd.ENV_VAR, raising=False)
        assert pwd.resolve_expected_ppid(None) is None

    def test_cli_flag_wins_over_env(self, monkeypatch):
        monkeypatch.setenv(pwd.ENV_VAR, "1234")
        assert pwd.resolve_expected_ppid(9999) == 9999

    def test_env_used_when_cli_omitted(self, monkeypatch):
        monkeypatch.setenv(pwd.ENV_VAR, "5678")
        assert pwd.resolve_expected_ppid(None) == 5678

    def test_zero_or_negative_cli_is_noop(self, monkeypatch):
        monkeypatch.delenv(pwd.ENV_VAR, raising=False)
        assert pwd.resolve_expected_ppid(0) is None
        assert pwd.resolve_expected_ppid(-5) is None
        assert pwd.resolve_expected_ppid(1) is None  # launchd/init — no parent to watch

    def test_malformed_env_ignored(self, monkeypatch):
        monkeypatch.setenv(pwd.ENV_VAR, "not-a-number")
        assert pwd.resolve_expected_ppid(None) is None

    def test_empty_env_ignored(self, monkeypatch):
        monkeypatch.setenv(pwd.ENV_VAR, "")
        assert pwd.resolve_expected_ppid(None) is None


class TestInstallParentWatchdog:
    def test_noop_when_expected_ppid_is_none(self):
        assert pwd.install_parent_watchdog(None) is None

    def test_noop_when_expected_ppid_is_pid_1(self):
        # PID 1 (launchd / init) is the "no real parent" sentinel —
        # comparing getppid() against 1 would either always-match
        # (we're already a launchd grandchild and there's no death to
        # detect) or never-match (we have a real parent but the caller
        # accidentally passed 1). Either way, refusing to install is
        # the safe move.
        assert pwd.install_parent_watchdog(1) is None

    def test_fires_synchronously_when_install_time_ppid_already_wrong(self):
        """Catch the race where the supervisor died between spawn and
        install. The watchdog must NOT silently wait for the next poll —
        the operator already has no answer for "where did my server go"
        and a 2 s blind spot would make the marker line look like it
        appeared spontaneously."""
        fired: list[tuple[int, int]] = []
        # Expected PPID is "the real PPID + 1" so the comparison fails
        # immediately. Using getppid() + 1 keeps the test portable: any
        # CI host that happens to have PID 1 as the test runner's parent
        # (containerized pytest) is still covered.
        real_ppid = os.getppid()
        wrong_ppid = real_ppid + 1
        # Guard against the (extremely unlikely) case that wrong_ppid
        # happens to equal the live PPID after wraparound — bump again.
        if wrong_ppid <= 1:
            wrong_ppid = real_ppid + 2

        result = pwd.install_parent_watchdog(
            wrong_ppid, on_orphan=lambda exp, obs: fired.append((exp, obs))
        )
        assert result is None  # synchronous fire, no thread
        assert len(fired) == 1
        assert fired[0][0] == wrong_ppid
        assert fired[0][1] == real_ppid

    def test_loop_detects_ppid_change(self, monkeypatch):
        """Simulate the post-SIGKILL re-parent: live PPID flips to 1
        (launchd) while the expected PPID stays at the supervisor's old
        PID. The watchdog must fire on the very next poll."""
        real_ppid = os.getppid()
        fired = threading.Event()
        captured: list[tuple[int, int]] = []

        # Toggle: getppid() first returns the real value (so install
        # succeeds), then returns 1 after the flip to simulate launchd
        # adopting the orphan.
        call_count = {"n": 0}

        def fake_getppid():
            call_count["n"] += 1
            # First call happens inside install_parent_watchdog's
            # install-time short-circuit check. Return real_ppid so the
            # install proceeds to spawning a thread. Subsequent calls
            # (from the loop body) return 1 to simulate the orphan.
            if call_count["n"] == 1:
                return real_ppid
            return 1

        def on_orphan(expected, observed):
            captured.append((expected, observed))
            fired.set()

        with patch.object(pwd.os, "getppid", side_effect=fake_getppid):
            thread = pwd.install_parent_watchdog(
                real_ppid, interval=0.05, on_orphan=on_orphan
            )
            assert thread is not None
            # The loop fires immediately on the second poll (no initial
            # sleep). 2 s ceiling guards against runaway tests on
            # severely overloaded CI; in practice this resolves in <50 ms.
            assert fired.wait(timeout=2.0), "watchdog never fired"
        # Thread is daemon — even if join times out the suite proceeds.
        thread.join(timeout=1.0)
        assert not thread.is_alive(), "watchdog thread did not exit after firing"
        assert len(captured) == 1
        assert captured[0] == (real_ppid, 1)

    def test_loop_exits_when_stop_event_set(self):
        """The thread exposes a ``_rapid_mlx_stop_event`` attribute so
        a graceful-shutdown caller (or test) can disarm the watchdog
        without waiting for the next poll OR triggering the orphan
        path. Important for the future ``execve`` self-replacement
        scenario (e.g. socket-activation handoff)."""
        real_ppid = os.getppid()
        thread = pwd.install_parent_watchdog(
            real_ppid, interval=0.05, on_orphan=lambda e, o: None
        )
        assert thread is not None
        try:
            stop_event = thread._rapid_mlx_stop_event  # type: ignore[attr-defined]
            assert isinstance(stop_event, threading.Event)
            stop_event.set()
            thread.join(timeout=2.0)
            assert not thread.is_alive()
        finally:
            # Defensive — make sure the thread cannot outlive the test
            # even if assertions above raise.
            if thread.is_alive():
                getattr(thread, "_rapid_mlx_stop_event", threading.Event()).set()
                thread.join(timeout=1.0)

    def test_callback_runs_only_once_per_install(self):
        """Once the orphan callback fires, the loop returns — no spam
        on subsequent poll intervals. Pre-fix a buggy implementation
        could call the supervisor 100x while the lifespan drain ran."""
        real_ppid = os.getppid()
        fired = threading.Event()
        call_count = {"n": 0}

        def on_orphan(expected, observed):
            call_count["n"] += 1
            fired.set()

        # Simulate the always-orphan condition: after install-time
        # check, getppid always returns 1.
        getppid_calls = {"n": 0}

        def fake_getppid():
            getppid_calls["n"] += 1
            if getppid_calls["n"] == 1:
                return real_ppid
            return 1

        with patch.object(pwd.os, "getppid", side_effect=fake_getppid):
            thread = pwd.install_parent_watchdog(
                real_ppid, interval=0.02, on_orphan=on_orphan
            )
            assert thread is not None
            assert fired.wait(timeout=2.0)
            # Give the loop a chance to (incorrectly) fire again. If it
            # had been written without an exit-after-fire, this 200 ms
            # window at 20 ms cadence would record ~10 extra fires.
            time.sleep(0.2)
        thread.join(timeout=1.0)
        assert call_count["n"] == 1


class TestDefaultOnOrphan:
    def test_writes_canonical_marker_to_stderr(self, capsys):
        """Dogfood postmortems grep for the single marker line. Pin the
        string so a future refactor that changes the wording also
        updates whatever log scraper depends on it."""
        with patch.object(pwd.os, "kill") as mock_kill, patch.object(
            pwd.time, "sleep"
        ), patch.object(pwd.os, "_exit") as mock_exit:
            # ``_exit`` is what terminates the call; we patch it so the
            # test runner survives. The function would normally never
            # return.
            mock_exit.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                pwd._default_on_orphan(expected_ppid=12345, observed_ppid=1)
        err = capsys.readouterr().err
        assert "[rapid-mlx] parent watchdog" in err
        assert "12345" in err
        assert "observed PPID 1" in err
        # SIGTERM first, SIGKILL second (matches the rapid-desktop
        # terminationGrace contract).
        assert mock_kill.call_count >= 1

    def test_sigkill_after_grace_window(self):
        """Defense against a lifespan drain that hangs: the second kill
        is the un-catchable SIGKILL, sent after the same 5 s grace the
        desktop side waits on its own terminateChild path."""
        import signal as _signal

        kills: list[int] = []

        def fake_kill(pid, sig):
            kills.append(sig)

        with patch.object(pwd.os, "kill", side_effect=fake_kill), patch.object(
            pwd.time, "sleep"
        ) as mock_sleep, patch.object(pwd.os, "_exit", side_effect=SystemExit(0)):
            with pytest.raises(SystemExit):
                pwd._default_on_orphan(expected_ppid=9999, observed_ppid=1)
        assert kills == [_signal.SIGTERM, _signal.SIGKILL]
        # Exactly 5 s of sleep between SIGTERM and SIGKILL.
        assert any(args[0] == 5.0 for args, _ in mock_sleep.call_args_list)
