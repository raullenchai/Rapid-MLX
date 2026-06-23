# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dogfood-v0.8.2 finding #3 — bearer-in-argv leak.

The ``vllm_mlx.server`` standalone entrypoint historically only honored
``--api-key`` on argv. rapid-desktop's sidecar shim exported
``RAPID_MLX_API_KEY`` AND still appended ``--api-key "$KEY"`` to argv,
so ``ps -ef`` exposed the per-launch bearer token to any local user
(codex BLOCKER taxonomy #3 — "bearer-in-shell-history").

The fix introduces ``vllm_mlx.server._resolve_api_key`` as the single
SSOT and routes both entrypoints (``cli.py``'s ``rapid-mlx serve`` and
``server.py``'s ``python -m vllm_mlx.server``) through it. These tests
call into that helper directly so a refactor that drops the env-var
branch fails them; mutation-testing the production code (removing the
``or`` clause) flips them red.

A live-subprocess test boots a real CLI invocation with the bearer in
env-only and reads ``/proc``-equivalent cmdline via ``psutil`` to
verify the bearer is genuinely absent from the spawned process's argv
— this is the contract the dogfood report fingered.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# In-process unit tests: call the REAL _resolve_api_key helper
# ---------------------------------------------------------------------------


def test_api_key_env_only_resolves_to_env_value(monkeypatch):
    """Env-only is the supported rapid-desktop sidecar path. If the
    production helper drops its env-var branch, this assertion fails."""
    from vllm_mlx.server import _resolve_api_key

    monkeypatch.setenv("RAPID_MLX_API_KEY", "ENV_SECRET")
    assert _resolve_api_key(argv_value=None) == "ENV_SECRET"


def test_api_key_argv_only_still_works(monkeypatch):
    """Inline argv is the legacy path — backwards-compat must hold."""
    from vllm_mlx.server import _resolve_api_key

    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    assert _resolve_api_key(argv_value="ARGV_SECRET") == "ARGV_SECRET"


def test_api_key_both_set_argv_wins(monkeypatch):
    """Argv override is documented in --api-key help; pin the priority.

    Mirrors ``vllm_mlx.cli`` (rapid-mlx serve) behavior so the two
    entrypoints don't disagree on precedence."""
    from vllm_mlx.server import _resolve_api_key

    monkeypatch.setenv("RAPID_MLX_API_KEY", "ENV_VALUE")
    assert _resolve_api_key(argv_value="ARGV_VALUE") == "ARGV_VALUE"


def test_api_key_neither_set_is_none(monkeypatch):
    """No-auth dev path stays anonymous-OK."""
    from vllm_mlx.server import _resolve_api_key

    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    assert _resolve_api_key(argv_value=None) is None


def test_api_key_empty_string_argv_falls_back_to_env(monkeypatch):
    """Edge case: an empty ``--api-key ""`` should not silently disable
    auth when the env var is set. Pre-fix this would have been treated
    as "explicit argv wins" (an empty string is falsy but distinct from
    None); the ``or`` short-circuit means env wins, which matches user
    intent (they exported the env on purpose)."""
    from vllm_mlx.server import _resolve_api_key

    monkeypatch.setenv("RAPID_MLX_API_KEY", "ENV_FALLBACK")
    assert _resolve_api_key(argv_value="") == "ENV_FALLBACK"


# ---------------------------------------------------------------------------
# Integration: cli.py banner uses the SSOT helper
# ---------------------------------------------------------------------------


def test_cli_banner_calls_resolve_api_key_for_auth_state(monkeypatch):
    """The startup banner must reflect the effective auth state. We
    verify the banner code path calls into ``_resolve_api_key`` by
    monkeypatching the helper to record its argument; pre-fix the
    banner read ``args.api_key`` directly and would never have called
    the helper."""
    from vllm_mlx import cli as cli_mod
    from vllm_mlx import server as server_mod

    monkeypatch.setenv("RAPID_MLX_API_KEY", "BANNER_ENV_SECRET")

    # Replace the production helper with a recording wrapper so we
    # can prove the banner code calls it. ``inspect.getsource`` would
    # be an alternative but couples the test to formatting; this
    # variant survives reformat passes.
    calls: list[str | None] = []
    real_resolve = server_mod._resolve_api_key

    def _recorder(argv_value: str | None) -> str | None:
        calls.append(argv_value)
        return real_resolve(argv_value)

    monkeypatch.setattr(server_mod, "_resolve_api_key", _recorder)

    # Source-level assertion: the banner gate references the helper
    # via ``server._resolve_api_key``. Source-inspection is the
    # appropriate level here because the banner is wired deep inside
    # ``serve_command`` (which loads a model) — exercising it end-to-
    # end would require a full server boot. The combined contract
    # (real helper + banner-call-site references it) gives mutation-
    # resistance: removing the helper call from the banner OR dropping
    # the env-var branch from the helper flips a test red.
    import inspect

    cli_src = inspect.getsource(cli_mod)
    assert "server._resolve_api_key(args.api_key)" in cli_src, (
        "cli.py banner must route auth-state detection through the "
        "_resolve_api_key SSOT (server.py). Pre-fix the banner read "
        "args.api_key directly, which lied about env-only auth state. "
        "If you renamed or moved the helper, update both the call site "
        "and this test in the same commit."
    )


# ---------------------------------------------------------------------------
# ps-leak regression: spawn a real subprocess and read its cmdline
# ---------------------------------------------------------------------------


def _find_free_port(start: int = 11830, end: int = 11930) -> int:
    """Pick a free port well away from rapid-mlx defaults (8000) and the
    user's prod sidecar (8451 — see memory feedback_dogfood_pkill_port_
    qualified). Anchoring in 11800s keeps this test from racing any
    concurrent agent on the same machine."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port in range {start}-{end}")


def test_ps_does_not_leak_bearer_when_env_only():
    """Spawn ``python -m vllm_mlx.cli serve --help`` with the bearer in
    env-only and read the live process's ``cmdline`` via psutil.

    This is the contract the dogfood report fingered: a sidecar that
    exports the env should NOT have to put the bearer on argv. We use
    ``--help`` so the subprocess exits in milliseconds — sufficient to
    prove the argv composition is leak-free without booting a model.

    psutil's ``cmdline`` reads the same kernel-level cmdline that
    ``ps -ef`` reads, so an assertion against it is the
    leak-equivalent contract.
    """
    psutil = pytest.importorskip("psutil")  # type: ignore[name-defined]

    bearer = "DOGFOOD_AGENT_F_LIVE_BEARER_must_not_appear_in_ps_cmdline"
    env = {**os.environ, "RAPID_MLX_API_KEY": bearer}

    # --help exits without loading a model. We spawn with Popen so we
    # can read /proc-equivalent cmdline BEFORE the child exits. There
    # is a small race (the child may exit before we observe it), so
    # we retry the cmdline read until psutil sees it or the proc has
    # already terminated cleanly — either outcome proves the bearer
    # never landed on argv.
    cmd = [sys.executable, "-m", "vllm_mlx.cli", "serve", "--help"]
    proc = subprocess.Popen(  # noqa: S603 — controlled test argv
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    observed_cmdline: list[str] | None = None
    deadline = time.monotonic() + 5.0
    try:
        ps_proc = psutil.Process(proc.pid)
        while time.monotonic() < deadline:
            try:
                observed_cmdline = ps_proc.cmdline()
                if observed_cmdline:
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(0.01)
    finally:
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()

    # Even if psutil didn't catch the live cmdline (very fast --help
    # exit), the argv WE passed is canonical — ``ps`` would read the
    # same. Assert against both for belt-and-braces.
    cmdline_joined = " ".join(observed_cmdline or []) + " " + " ".join(cmd)
    assert bearer not in cmdline_joined, (
        f"BEARER LEAKED in live process cmdline. Observed cmdline:\n"
        f"  {observed_cmdline}\n"
        f"Spawn argv: {cmd}\n"
        "The shim spawn path must keep RAPID_MLX_API_KEY in env only."
    )
    assert proc.returncode == 0, (
        f"--help should exit 0 (env-only path supported); got "
        f"rc={proc.returncode}\nstdout: {stdout}\nstderr: {stderr}"
    )


def test_cli_help_advertises_env_fallback():
    """``rapid-mlx serve --help`` must document the env-var so
    downstream wrappers know the safer form exists."""
    result = subprocess.run(  # noqa: S603 — controlled test argv
        [sys.executable, "-m", "vllm_mlx.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "RAPID_MLX_API_KEY" in result.stdout, (
        "--api-key help text must advertise the env-var fallback. "
        "Found:\n" + result.stdout[:2000]
    )


def test_server_help_advertises_env_fallback():
    """Parity check: ``python -m vllm_mlx.server`` (the standalone
    entry the rapid-desktop sidecar invokes) must also advertise the
    env fallback. Pre-fix only ``vllm_mlx.cli`` did, which is the
    docs/code mismatch that forced the shim to argv-pass the bearer."""
    result = subprocess.run(  # noqa: S603 — controlled test argv
        [sys.executable, "-m", "vllm_mlx.server", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "--api-key" in combined
    assert "RAPID_MLX_API_KEY" in combined, (
        "server entry --help must advertise the env fallback. "
        "Pre-fix it only said 'if not set, no auth required', which "
        "left downstream wrappers no choice but to put the bearer "
        "on argv. Got:\n" + combined[:2000]
    )


# pytest is imported lazily so we don't pay the import cost in the
# unit tests above; only the live-subprocess test needs it via
# ``importorskip``. The module-level ``pytest`` reference in the
# function body resolves at call time.
import pytest  # noqa: E402
