# SPDX-License-Identifier: Apache-2.0
"""Assert ``rapid-mlx doctor`` is *purely* an env-health probe.

The contract for the doctor subcommand (PR #4 of the doctor refactor
series) is:

* runs in ≤ 5 seconds wall-clock,
* **never** instantiates an engine,
* **never** loads a model,
* **never** opens a server port.

The previous doctor's smoke tier did all three (model_load check
launched a real ``BatchedEngine``). Hard-pin the new contract so a
future refactor that re-adds model-touching code surfaces in CI
instead of in a user's stuck terminal.
"""

from __future__ import annotations

import io
import socket
import sys
import time
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Module-level: doctor must NOT pull in heavy model-machinery on import.
# ---------------------------------------------------------------------------


def test_doctor_module_does_not_import_engine_or_server():
    """Importing the doctor module must not drag in BatchedEngine,
    mlx.core, vllm_mlx.engine, or the FastAPI server. The old doctor
    smoke tier did pull these in for the model_load check; the new
    env-health doctor must not.

    Runs in a fresh subprocess because (a) sibling tests in
    ``test_doctor_env_health.py`` already imported the doctor modules
    in this process, and (b) clearing only the blocked modules would
    leave the doctor's cached imports in place — so a future regression
    that adds ``from vllm_mlx import engine`` at the top of
    ``doctor/env_health.py`` would be invisible to an in-process check
    (the cached import never re-runs). Codex review round 1 caught
    this; the subprocess form gives a clean module table.
    """
    import subprocess

    # Block both exact names AND the ``mlx*`` family (mlx, mlx.core, mlx_lm,
    # mlx_vlm). Codex review round 2: the previous set missed ``mlx.core``,
    # so importing mlx at module load would have slipped past silently.
    probe = (
        "import importlib, sys; "
        "blocked_exact = {'vllm_mlx.engine', 'vllm_mlx.server', "
        "'vllm_mlx.api.server'}; "
        "blocked_prefixes = ('mlx.', 'mlx_lm', 'mlx_vlm'); "
        "importlib.import_module('vllm_mlx.doctor'); "
        "importlib.import_module('vllm_mlx.doctor.cli'); "
        "importlib.import_module('vllm_mlx.doctor.env_health'); "
        "loaded = set(sys.modules); "
        "leaked_exact = blocked_exact & loaded; "
        "leaked_prefix = {m for m in loaded "
        "if m == 'mlx' or m.startswith(blocked_prefixes)}; "
        "leaked = sorted(leaked_exact | leaked_prefix); "
        "sys.exit('LEAKED:' + ','.join(leaked)) if leaked else None"
    )
    result = subprocess.run(  # noqa: S603 — args constructed by us
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        f"doctor module imports leaked heavy machinery: "
        f"{result.stderr.strip() or result.stdout.strip()}"
    )


# ---------------------------------------------------------------------------
# Runtime: run_all() and CLI dispatch must not touch model machinery.
# ---------------------------------------------------------------------------


def test_run_all_does_not_call_load_model():
    """``run_all()`` must not invoke ``vllm_mlx.server.load_model``."""
    from vllm_mlx.doctor import env_health

    with mock.patch("vllm_mlx.server.load_model", autospec=True) as load_mock:
        env_health.run_all()
    assert load_mock.call_count == 0, (
        f"doctor called load_model {load_mock.call_count} times; "
        "env-health must never load a model."
    )


def test_run_all_does_not_open_any_socket():
    """No probe may open a listening socket. The HF network probe makes
    a single outbound HTTPS request (which we mock here) — and nothing
    else may bind a port.

    Patches ``socket.socket.bind`` / ``socket.socket.listen`` to fail
    loudly if any code path tries to bring up a server-like endpoint.
    """
    from vllm_mlx.doctor import env_health

    real_bind = socket.socket.bind
    real_listen = socket.socket.listen

    def explode_bind(self, *args, **kwargs):  # pragma: no cover — assertion
        pytest.fail(
            f"doctor opened a socket.bind({args!r}) — env-health must not "
            "bind any port."
        )

    def explode_listen(self, *args, **kwargs):  # pragma: no cover — assertion
        pytest.fail("doctor called socket.listen — env-health must not serve.")

    # Network probe is mocked at a higher level: replace the probe with a
    # no-op so we don't actually touch huggingface.co.
    with (
        mock.patch.object(
            env_health,
            "_probe_hf",
            return_value=(env_health.CheckStatus.OK, "mocked"),
        ),
        mock.patch.object(socket.socket, "bind", explode_bind),
        mock.patch.object(socket.socket, "listen", explode_listen),
    ):
        env_health.run_all()

    # If we got here, no bind / listen happened. Restore (mock.patch did this).
    assert socket.socket.bind is real_bind
    assert socket.socket.listen is real_listen


def test_doctor_runtime_under_five_seconds():
    """End-to-end run_all() + render must complete inside the contract.

    Network probe is mocked so the test doesn't depend on real connectivity;
    the budget is the structural runtime (filesystem walks, version lookups,
    socket-free fast paths)."""
    from vllm_mlx.doctor import env_health
    from vllm_mlx.doctor.cli import render

    with mock.patch.object(
        env_health,
        "_probe_hf",
        return_value=(env_health.CheckStatus.OK, "mocked"),
    ):
        t0 = time.perf_counter()
        report = env_health.run_all()
        buf = io.StringIO()
        render(report, stream=buf)
        elapsed = time.perf_counter() - t0

    # 5 s is the user-facing contract; the unit test is well under because
    # the HF cache walk is the only thing that can be slow on a real box.
    # Keep the test bound generous so CI flake on a slow runner doesn't
    # red-flag a non-issue, but tight enough to catch a 10× regression.
    assert elapsed < 5.0, f"doctor took {elapsed:.2f}s — exceeds 5 s contract"


# ---------------------------------------------------------------------------
# CLI dispatch: removed tiers must hard-redirect, not crash.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("legacy_tier", ["smoke", "check", "full", "benchmark"])
def test_doctor_legacy_tier_subcommand_redirects(legacy_tier: str, capsys):
    """``rapid-mlx doctor smoke|check|full|benchmark`` exits 2 with a
    pointer to ``rapid-mlx bench --tier <tier>``. PR #2 added a
    deprecation shim; this PR removes the shim and turns the same
    invocation into a clean redirect (exit 2 = "you typed something
    that's no longer valid")."""
    from argparse import Namespace

    from vllm_mlx.doctor.cli import doctor_command

    args = Namespace(tier=legacy_tier, verbose=False)
    with pytest.raises(SystemExit) as exc:
        doctor_command(args)
    assert exc.value.code == 2

    captured = capsys.readouterr()
    msg = captured.err
    assert "removed" in msg
    assert f"bench <model> --tier {legacy_tier}" in msg
