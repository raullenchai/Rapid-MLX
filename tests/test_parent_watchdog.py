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


class TestServeCommandWiring:
    """Source-pinned contract tests for the CLI wiring.

    Codex round-1 MAJOR #3: the helper itself is well-tested, but a
    future refactor that drops the ``install_parent_watchdog`` call
    from ``serve_command`` OR moves it AFTER the multi-minute model
    download would silently re-introduce the orphan-during-download
    bug. Source-grep the contract so the regression has to walk
    through this test on the way out.
    """

    def _serve_command_body(self) -> str:
        from pathlib import Path

        cli_file = Path(__file__).resolve().parents[1] / "vllm_mlx" / "cli.py"
        source = cli_file.read_text()
        start = source.index("def serve_command(")
        end = source.find("\ndef ", start + 1)
        return source[start : end if end != -1 else len(source)]

    def test_serve_command_installs_watchdog(self):
        body = self._serve_command_body()
        assert "install_parent_watchdog(" in body, (
            "serve_command no longer calls install_parent_watchdog — the "
            "orphan-sidecar mitigation for rapid-desktop #449 is gone. "
            "Restore the helper invocation at the top of serve_command."
        )

    def test_watchdog_installs_before_model_download(self):
        """The install MUST happen BEFORE ``_ensure_model_downloaded`` so
        an operator who kills the supervisor while the (possibly multi-
        minute) HF snapshot download is in flight still gets a clean
        reap. Pre-install, the orphan would hold both the partial download
        AND the future model RAM."""
        body = self._serve_command_body()
        idx_install = body.find("install_parent_watchdog(")
        idx_download = body.find("_ensure_model_downloaded(")
        assert idx_install != -1, (
            "install_parent_watchdog missing from serve_command — see "
            "test_serve_command_installs_watchdog for the actionable hint."
        )
        assert idx_download != -1, (
            "_ensure_model_downloaded fixture moved; update this test."
        )
        assert idx_install < idx_download, (
            "rapid-desktop #449 regression: install_parent_watchdog fires "
            "AFTER _ensure_model_downloaded. Move the install to the TOP "
            "of serve_command so it arms before the model download starts."
        )

    def test_watchdog_installs_before_audio_mode_fork(self):
        """Same invariant for the audio-mode fork — ``_serve_audio_mode``
        returns out of ``serve_command`` without re-installing the
        watchdog, so the install MUST happen above the
        ``_resolve_audio_model_for_serve`` branch or audio-only
        operators get no orphan protection at all."""
        body = self._serve_command_body()
        idx_install = body.find("install_parent_watchdog(")
        idx_audio = body.find("_resolve_audio_model_for_serve(")
        assert idx_install != -1
        if idx_audio != -1:
            assert idx_install < idx_audio, (
                "rapid-desktop #449 regression: install_parent_watchdog "
                "fires AFTER the audio-mode fork — audio aliases (kokoro, "
                "whisper, etc.) would skip the watchdog install entirely. "
                "Move the install to the TOP of serve_command."
            )


class TestInternalSpawnersStampWatchdog:
    """Source-pinned contract tests for in-tree supervisors that
    themselves spawn ``rapid-mlx serve`` children (codex round-1 MAJOR
    #1+#2): ``rapid-mlx share`` and ``rapid-mlx chat``. Without the
    stamp, a SIGKILL on the parent CLI invocation leaves the child
    serve as an orphan — the exact bug rapid-desktop #449 reports for
    the desktop case, applied to the local CLI."""

    def test_chat_spawn_stamps_watchdog_ppid(self):
        from pathlib import Path

        cli_file = Path(__file__).resolve().parents[1] / "vllm_mlx" / "cli.py"
        source = cli_file.read_text()
        # Locate _spawn_chat_server body. The function is small (~150
        # lines) so a single-window scan is enough; pin the marker on
        # the env-dict write so a refactor that splits the function
        # still trips this test.
        start = source.index("def _spawn_chat_server(")
        end = source.find("\ndef ", start + 1)
        body = source[start : end if end != -1 else len(source)]
        assert "RAPID_MLX_WATCHDOG_PPID" in body, (
            "rapid-desktop #449 sibling regression: _spawn_chat_server "
            "no longer stamps RAPID_MLX_WATCHDOG_PPID on the child env. "
            "Without it, SIGKILL of `rapid-mlx chat` orphans the serve "
            "subprocess. Restore the env stamp."
        )

    def test_share_spawn_stamps_watchdog_ppid(self):
        from pathlib import Path

        share_file = (
            Path(__file__).resolve().parents[1] / "vllm_mlx" / "share" / "cli.py"
        )
        source = share_file.read_text()
        assert "RAPID_MLX_WATCHDOG_PPID" in source, (
            "rapid-desktop #449 sibling regression: rapid-mlx share's "
            "serve-spawn no longer stamps RAPID_MLX_WATCHDOG_PPID on the "
            "child env. Without it, SIGKILL of `rapid-mlx share` leaves "
            "an orphan serve holding the bearer-gated port. Restore the "
            "env stamp."
        )

    def test_bench_serve_spawn_stamps_watchdog_ppid(self):
        """Codex r2 MAJOR: ``rapid-mlx bench --tier ...`` boots a
        serve child via ``vllm_mlx/bench/_server.py``. Same SIGKILL-of-
        supervisor orphan story applies; pin the env stamp here too."""
        from pathlib import Path

        bench_file = (
            Path(__file__).resolve().parents[1] / "vllm_mlx" / "bench" / "_server.py"
        )
        source = bench_file.read_text()
        assert "RAPID_MLX_WATCHDOG_PPID" in source, (
            "rapid-desktop #449 sibling regression: bench/_server.py "
            "no longer stamps RAPID_MLX_WATCHDOG_PPID on the spawned "
            "serve child. SIGKILL of `rapid-mlx bench` would orphan "
            "the model server. Restore the env stamp."
        )

    def test_all_internal_spawners_use_direct_assignment_not_setdefault(self):
        """Codex r2 MAJOR: ``setdefault`` is the wrong primitive for
        the watchdog stamp.

        If the parent CLI invocation was itself launched under a
        supervisor that exported ``RAPID_MLX_WATCHDOG_PPID=<gp_pid>``,
        ``setdefault`` would preserve the grandparent's PID. The serve
        child would then compare ``os.getppid()`` (= our immediate
        PID) against the grandparent's PID, mismatch on the FIRST
        poll, and self-terminate the freshly-booted server. The
        spawner owns the watchdog relationship for the child it just
        spawned — overwrite (direct ``env[KEY] = ...``) is the only
        correct shape."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        offenders: list[str] = []
        for spawner in (
            repo_root / "vllm_mlx" / "cli.py",
            repo_root / "vllm_mlx" / "share" / "cli.py",
            repo_root / "vllm_mlx" / "bench" / "_server.py",
        ):
            source = spawner.read_text()
            # We only care about the watchdog stamp itself, not any
            # other ``setdefault`` calls in the file. Grep for both
            # forms on the same key, skipping comment lines (which
            # legitimately mention ``setdefault`` in the rationale).
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "setdefault" in stripped and "RAPID_MLX_WATCHDOG_PPID" in stripped:
                    offenders.append(f"{spawner.name}: {stripped}")
        assert not offenders, (
            "rapid-desktop #449 sibling regression: a spawner used "
            "setdefault for the watchdog stamp. A grandparent's stale "
            "RAPID_MLX_WATCHDOG_PPID would be inherited and the serve "
            "child would self-terminate immediately. Use direct "
            "assignment instead. Offending lines:\n  " + "\n  ".join(offenders)
        )


class TestDefaultOnOrphan:
    def test_writes_canonical_marker_to_stderr(self, capsys):
        """Dogfood postmortems grep for the single marker line. Pin the
        string so a future refactor that changes the wording also
        updates whatever log scraper depends on it."""
        with (
            patch.object(pwd.os, "kill") as mock_kill,
            patch.object(pwd.time, "sleep"),
            patch.object(pwd.os, "_exit") as mock_exit,
        ):
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

        with (
            patch.object(pwd.os, "kill", side_effect=fake_kill),
            patch.object(pwd.time, "sleep") as mock_sleep,
            patch.object(pwd.os, "_exit", side_effect=SystemExit(0)),
            pytest.raises(SystemExit),
        ):
            pwd._default_on_orphan(expected_ppid=9999, observed_ppid=1)
        assert kills == [_signal.SIGTERM, _signal.SIGKILL]
        # Exactly 5 s of sleep between SIGTERM and SIGKILL.
        assert any(args[0] == 5.0 for args, _ in mock_sleep.call_args_list)
