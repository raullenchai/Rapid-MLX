# SPDX-License-Identifier: Apache-2.0
"""Tests for ``vllm_mlx._tempfile_safe.managed_tempfile_path`` (GH #719).

Covers:

1. Happy path — file exists inside the context, is unlinked on exit.
2. Exception inside the body — unlink still runs (try/finally semantics).
3. ``release()`` hands ownership off and SUPPRESSES the unlink.
4. The atexit fallback fires when ``sys.exit()`` (or any other path
   that bypasses ``__exit__``) leaves the body. This is the
   regression that #719 reported on ``rapid-mlx chat``.
5. The internal registry doesn't accumulate dead entries after
   normal context exits — important for long-lived parents that
   create + tear down many tempfiles (chat REPL ``/model`` swaps).
6. Idempotent: calling ``release()`` twice, or releasing after the
   context already exited cleanly, is safe.

The leak regression test (#5 below) runs the chat REPL in a fresh
subprocess and asserts the post-exit count delta in ``$TMPDIR`` is
zero, which is the user-visible contract from the bug report.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from vllm_mlx import _tempfile_safe
from vllm_mlx._tempfile_safe import managed_tempfile_path


def _count_chat_logs() -> int:
    """Return the number of ``rapid-mlx-chat-*.log`` stragglers."""
    tmp = Path(tempfile.gettempdir())
    try:
        return sum(1 for name in os.listdir(tmp) if name.startswith("rapid-mlx-chat-"))
    except OSError:
        return 0


def test_happy_path_unlinks_on_exit():
    """File exists inside the block, gone after."""
    with managed_tempfile_path(prefix="ut-happy-", suffix=".tmp") as h:
        assert os.path.exists(h.path)
        # Path-like interface: ``open(handle)`` works.
        with open(h, "w") as f:
            f.write("hello")
        assert Path(h.path).read_text() == "hello"
    assert not os.path.exists(h.path)


def test_exception_in_body_still_unlinks():
    """try/finally — unlink runs even when the body raises."""
    sentinel: list[str] = []
    with (
        pytest.raises(RuntimeError, match="boom"),
        managed_tempfile_path(prefix="ut-exc-", suffix=".tmp") as h,
    ):
        sentinel.append(h.path)
        assert os.path.exists(h.path)
        raise RuntimeError("boom")
    assert not os.path.exists(sentinel[0])


def test_release_suppresses_unlink():
    """``release()`` hands ownership over; the helper steps back."""
    with managed_tempfile_path(prefix="ut-rel-", suffix=".tmp") as h:
        path = h.path
        h.release()
        assert h.released is True
    # File should still exist — caller is now responsible.
    assert os.path.exists(path)
    # Manual cleanup so we don't pollute $TMPDIR.
    os.unlink(path)


def test_release_idempotent():
    """Double-release is a no-op (the second call is a noop)."""
    with managed_tempfile_path(prefix="ut-relx2-", suffix=".tmp") as h:
        path = h.path
        h.release()
        h.release()
    assert os.path.exists(path)
    os.unlink(path)


def test_registry_does_not_leak_after_normal_exit():
    """The internal registry must shrink on normal exit.

    Long-lived parents (chat REPL ``/model`` swaps) create + tear
    down many tempfiles. If the registry kept stale paths around,
    the eventual ``atexit`` reap would walk a giant list of
    already-unlinked names — harmless but wasteful.
    """
    baseline = _tempfile_safe._pending_snapshot()
    for _ in range(10):
        with managed_tempfile_path(prefix="ut-noreg-", suffix=".tmp"):
            pass
    after = _tempfile_safe._pending_snapshot()
    assert after == baseline


def test_registry_tracks_path_inside_context():
    """While the context is open, the path IS in the registry —
    so the atexit fallback would reap it if interpreter exited."""
    baseline = _tempfile_safe._pending_snapshot()
    with managed_tempfile_path(prefix="ut-track-", suffix=".tmp") as h:
        assert h.path in _tempfile_safe._pending_snapshot()
        assert _tempfile_safe._pending_snapshot() - baseline == {h.path}
    assert _tempfile_safe._pending_snapshot() == baseline


def test_release_removes_from_registry_immediately():
    """``release()`` should drop the path from the registry so the
    atexit fallback doesn't try to unlink a file the caller is now
    tracking under their own lifecycle."""
    with managed_tempfile_path(prefix="ut-relreg-", suffix=".tmp") as h:
        path = h.path
        assert path in _tempfile_safe._pending_snapshot()
        h.release()
        assert path not in _tempfile_safe._pending_snapshot()
    os.unlink(path)


def test_atexit_fallback_reaps_paths_not_cleaned_by_context_exit():
    """Codex round-1 finding #3 — exercise the atexit hook directly.

    The ``with`` block's ``finally`` covers normal exit, exceptions, and
    ``SystemExit`` — so a test that uses the public context manager and
    then exits cannot distinguish "atexit reaped it" from "``__exit__``
    reaped it". To prove the atexit fallback actually works, we have to
    leave a path in the module-level registry WITHOUT going through the
    context manager's ``finally``.

    Strategy: in a fresh subprocess, call ``mkstemp`` directly, register
    the path via the module's internal registry (the same pathway
    ``managed_tempfile_path`` uses), then exit normally. Only the atexit
    hook can reap that path; if the hook is broken, the file persists
    after the subprocess exits.
    """
    with tempfile.TemporaryDirectory() as td:
        marker = Path(td) / "marker.txt"
        script = textwrap.dedent(
            f"""
            import os, sys, tempfile
            sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
            from vllm_mlx import _tempfile_safe

            fd, path = tempfile.mkstemp(prefix="ut-atexit-", suffix=".tmp", dir={td!r})
            os.close(fd)
            # Bypass the context manager: register the path directly and
            # arm the shared atexit hook the same way the helper does on
            # first use. If the hook is broken, the file survives the
            # subprocess. This is the ONLY test in this file that
            # actually exercises the atexit path in isolation.
            _tempfile_safe._ensure_atexit_registered()
            with _tempfile_safe._pending_lock:
                _tempfile_safe._pending_paths.add(path)
            with open({str(marker)!r}, "w") as f:
                f.write(path)
            # Normal exit. ``__exit__`` does NOT run for this path
            # (we never entered the context manager). atexit is the
            # ONLY path that can reap it.
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        leaked_path = marker.read_text().strip()
        assert leaked_path
        assert not os.path.exists(leaked_path), (
            f"atexit hook failed to reap {leaked_path}"
        )


def test_systemexit_inside_context_body_triggers_context_finally():
    """Companion to the atexit test above.

    Spawn a subprocess that calls ``sys.exit(0)`` from inside a
    ``with managed_tempfile_path(...)`` block. ``SystemExit`` triggers
    ``__exit__`` (i.e. the ``finally`` inside the ``@contextmanager``
    wrapper), so the path is reaped via the normal exit path — atexit
    plays no role here. This documents the contract for the chat REPL,
    whose ``sys.exit(1)`` on readiness failure depends on this guarantee.
    """
    with tempfile.TemporaryDirectory() as td:
        marker = Path(td) / "marker.txt"
        script = textwrap.dedent(
            f"""
            import os, sys
            sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
            from vllm_mlx._tempfile_safe import managed_tempfile_path

            with managed_tempfile_path(prefix="ut-exit-", suffix=".tmp", dir={td!r}) as h:
                with open({str(marker)!r}, "w") as f:
                    f.write(h.path)
                sys.exit(0)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        leaked_path = marker.read_text().strip()
        assert leaked_path
        assert not os.path.exists(leaked_path), (
            f"SystemExit path leaked: {leaked_path} still exists"
        )


def test_os_exit_is_documented_to_skip_cleanup_negative_control():
    """Negative control — ``os._exit`` MUST leak.

    Documents the limitation called out in the module docstring:
    ``os._exit`` skips both ``__exit__`` and ``atexit``, so the file
    survives the subprocess. If a future "fix" started reaping this
    case (e.g. a SIGCHLD-based janitor), we'd want to know so the
    docstring can be updated. Asserting the file IS present makes
    the negative control real.
    """
    with tempfile.TemporaryDirectory() as td:
        marker = Path(td) / "marker.txt"
        script = textwrap.dedent(
            f"""
            import os, sys
            sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
            from vllm_mlx._tempfile_safe import managed_tempfile_path

            with managed_tempfile_path(prefix="ut-osexit-", suffix=".tmp", dir={td!r}) as h:
                with open({str(marker)!r}, "w") as f:
                    f.write(h.path)
                os._exit(0)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        leaked_path = marker.read_text().strip()
        assert leaked_path
        # ``os._exit`` skips atexit + __exit__: file must remain.
        assert os.path.exists(leaked_path), (
            "negative-control regression: os._exit no longer leaks. "
            "If intentional, update the helper's docstring."
        )
        # Manual cleanup so $TMPDIR doesn't accumulate.
        os.unlink(leaked_path)


def test_setup_window_exception_does_not_leak_path(monkeypatch, tmp_path):
    """Codex round-2 BLOCKING: exceptions between ``mkstemp`` and the
    yield must still unlink the just-created file.

    The leak window: ``mkstemp`` has created the file on disk, but the
    path has not yet been added to ``_pending_paths`` (so atexit won't
    see it) and the yielded-context ``finally`` hasn't started yet
    (so ``__exit__`` won't see it either). A SIGINT/SIGTERM landing in
    that window — or any setup-phase exception such as
    ``_ensure_atexit_registered`` raising — would otherwise leak the
    file permanently.

    Inject the failure by patching ``_ensure_atexit_registered`` to
    raise. Assert the file was cleaned up before the exception
    propagated, and that the registry is in the same state as
    before the call.
    """
    from vllm_mlx import _tempfile_safe

    baseline = _tempfile_safe._pending_snapshot()

    def _boom() -> None:
        raise KeyboardInterrupt("simulated SIGINT during setup")

    monkeypatch.setattr(_tempfile_safe, "_ensure_atexit_registered", _boom)

    captured_path: list[str] = []

    # Also stub mkstemp to capture the path before it's wiped, so we
    # can verify the unlink actually happened.
    real_mkstemp = tempfile.mkstemp

    def _spy_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        captured_path.append(path)
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", _spy_mkstemp)

    with (
        pytest.raises(KeyboardInterrupt, match="simulated SIGINT during setup"),
        managed_tempfile_path(prefix="ut-setupfail-", suffix=".tmp", dir=str(tmp_path)),
    ):
        pytest.fail("should never reach the body")

    assert captured_path, "mkstemp was not invoked"
    leaked = captured_path[0]
    assert not os.path.exists(leaked), (
        f"setup-window leak: {leaked} survived a setup-phase exception"
    )
    # Registry should be unchanged.
    assert _tempfile_safe._pending_snapshot() == baseline


def test_cleanup_unlinks_before_discarding_from_registry(monkeypatch, tmp_path):
    """Codex round-3 BLOCKING: ordering inside the cleanup ``finally``.

    The original order — ``_pending_paths.discard(path)`` first, then
    ``os.unlink(path)`` — had a window in which a ``BaseException``
    (Ctrl-C, SIGTERM-induced ``SystemExit``) landing between the two
    statements would leave the file on disk WITHOUT a registry entry.
    The atexit fallback could never see it.

    Fix: unlink first, discard second. If unlink raises a
    ``BaseException``, the path stays in the registry and atexit
    reaps it.

    Test: patch ``os.unlink`` to raise ``KeyboardInterrupt`` exactly
    once, drive the context-exit cleanup path, and assert the path
    is still in ``_pending_paths`` so atexit can reap it.
    """
    from vllm_mlx import _tempfile_safe

    real_unlink = os.unlink
    raised = {"n": 0}

    def _boom_unlink(p):
        if raised["n"] == 0 and "ut-clean-" in os.path.basename(p):
            raised["n"] += 1
            raise KeyboardInterrupt("simulated Ctrl-C during unlink")
        return real_unlink(p)

    monkeypatch.setattr(os, "unlink", _boom_unlink)

    captured_path: list[str] = []
    with (
        pytest.raises(KeyboardInterrupt, match="Ctrl-C during unlink"),
        managed_tempfile_path(
            prefix="ut-clean-", suffix=".tmp", dir=str(tmp_path)
        ) as h,
    ):
        captured_path.append(h.path)
        assert os.path.exists(h.path)

    assert captured_path, "context manager never yielded a handle"
    leaked = captured_path[0]
    # File survived the interrupted unlink — that's expected.
    assert os.path.exists(leaked), (
        "test setup error: unlink wasn't actually intercepted"
    )
    # BLOCKING fix: path must still be in the registry so atexit
    # can reap it. Pre-fix the discard ran first → registry was
    # empty → atexit blind.
    assert leaked in _tempfile_safe._pending_snapshot(), (
        f"registry ordering regression: {leaked} dropped from "
        f"_pending_paths before unlink completed, atexit can no "
        f"longer reap it"
    )
    # Manual cleanup to avoid polluting $TMPDIR.
    monkeypatch.undo()
    real_unlink(leaked)
    with _tempfile_safe._pending_lock:
        _tempfile_safe._pending_paths.discard(leaked)


def _count_in_dir(d: str) -> int:
    """Count ``rapid-mlx-chat-*.log`` files in ``d``."""
    try:
        return sum(1 for name in os.listdir(d) if name.startswith("rapid-mlx-chat-"))
    except OSError:
        return 0


def test_chat_command_does_not_leak_tempfile_on_keyboard_interrupt(tmp_path):
    """End-to-end regression for GH #719.

    The original bug: ``rapid-mlx chat`` leaked one zero-byte
    ``rapid-mlx-chat-*.log`` per invocation if anything raised
    between the ``NamedTemporaryFile(...).name`` call and the
    proc-registration step inside ``_spawn_chat_server``.

    Reproduce by injecting a KeyboardInterrupt at the ``print(...)``
    that announces the log path — the exact window the leak lived in.

    Run the chat command in a fresh subprocess with ``TMPDIR`` pointed
    at the test's ``tmp_path``. After the subprocess fully exits (so
    ``atexit`` has run), the parent counts the files. This is the
    user-visible contract from the bug report: after ``rapid-mlx chat``
    returns, the temp dir should not have a new straggler.
    """
    tmpdir = str(tmp_path)
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
        from unittest.mock import patch
        from vllm_mlx import cli

        import builtins
        real_print = builtins.print
        def killing_print(*args, **kwargs):
            s = " ".join(str(a) for a in args) if args else ""
            if "Starting server" in s:
                raise KeyboardInterrupt("simulated")
            return real_print(*args, **kwargs)

        with patch.object(cli, "_ensure_model_downloaded"), \\
             patch("builtins.print", killing_print):
            ns = type("Args", (), {{}})()
            ns.base_url = None
            ns.port = None
            ns.system = None
            ns.think = False
            ns.max_tokens = 50
            ns.temperature = 0.0
            ns.ready_timeout = 5
            ns.response_timeout = 5
            ns.model = "qwen3.5-4b-4bit"
            try:
                cli.chat_command(ns)
            except (SystemExit, KeyboardInterrupt):
                pass
        """
    )
    env = os.environ.copy()
    env["TMPDIR"] = tmpdir
    before = _count_in_dir(tmpdir)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    after = _count_in_dir(tmpdir)
    delta = after - before
    assert delta == 0, (
        f"GH #719 regression: rapid-mlx chat leaked {delta} "
        f"tempfile(s) on KeyboardInterrupt during spawn. "
        f"Files in {tmpdir}: {os.listdir(tmpdir)}"
    )


def test_chat_command_does_not_leak_tempfile_on_spawn_readiness_failure(tmp_path):
    """The other leak vector: ``_wait_for_chat_server`` raises, the
    parent prints a friendly error + ``sys.exit(1)``. In the original
    code the log file persisted because the early-exit path didn't
    explicitly unlink. ``_teardown_proc``'s zero-byte unlink covers
    this case via the atexit chain, but only when the spawn made it
    far enough to register on ``_active_procs``. With the helper
    wrapping the whole spawn, the leak is closed regardless.

    Run in subprocess with ``TMPDIR`` pinned to the test directory so
    we observe the post-atexit state (the contract that the bug
    reporter exercised).
    """
    tmpdir = str(tmp_path)
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
        from unittest.mock import patch, MagicMock
        from vllm_mlx import cli

        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.wait.return_value = 0

        def fake_spawn(model, log_path, served_name=None, *, register_in=None, log_handle=None):
            # Codex round-5 #4: explicitly REQUIRE the chat callsite to
            # pass log_handle. Without this assertion, the test could
            # silently pass even if the callsite stopped wiring the
            # managed handle through (then the context manager's own
            # finally would unlink the path and the count delta would
            # still be zero — false-negative).
            assert log_handle is not None, (
                "chat callsite regressed: log_handle was not passed "
                "through to _spawn_chat_server"
            )
            assert log_handle.released is False, (
                "chat callsite released the handle BEFORE the spawn"
            )
            log_fh = open(log_path, "w")
            fake_proc._rapid_mlx_log = log_fh
            fake_proc._rapid_mlx_log_path = log_path
            if register_in is not None:
                register_in.append(fake_proc)
            # Mirror the real spawn's hand-off so the chat REPL's
            # ``_teardown_proc`` is the one that decides when to unlink.
            log_handle.release()
            assert log_handle.released is True
            return fake_proc, "http://127.0.0.1:9999"

        def fake_wait(base_url, proc, timeout_s=600):
            raise RuntimeError("simulated readiness fail")

        with patch.object(cli, "_ensure_model_downloaded"), \\
             patch.object(cli, "_spawn_chat_server", side_effect=fake_spawn), \\
             patch.object(cli, "_wait_for_chat_server", side_effect=fake_wait):
            ns = type("Args", (), {{}})()
            ns.base_url = None
            ns.port = None
            ns.system = None
            ns.think = False
            ns.max_tokens = 50
            ns.temperature = 0.0
            ns.ready_timeout = 5
            ns.response_timeout = 5
            ns.model = "qwen3.5-4b-4bit"
            try:
                cli.chat_command(ns)
            except SystemExit:
                pass
        """
    )
    env = os.environ.copy()
    env["TMPDIR"] = tmpdir
    before = _count_in_dir(tmpdir)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    after = _count_in_dir(tmpdir)
    delta = after - before
    assert delta == 0, (
        f"GH #719 regression: rapid-mlx chat leaked {delta} "
        f"tempfile(s) on _wait_for_chat_server failure. "
        f"Files in {tmpdir}: {os.listdir(tmpdir)}"
    )
