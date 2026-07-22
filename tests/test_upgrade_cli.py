# SPDX-License-Identifier: Apache-2.0
"""Tests for the `rapid-mlx upgrade --dry-run` flag.

Dogfood-driven: a real user typing the Homebrew-muscle-memory `--dry-run`
on 0.9.3 hit `error: unrecognized arguments`. 0.9.4 adds the flag and
this test pins the contract — printed plan, no subprocess.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx import _version_check as vc
from vllm_mlx.cli import upgrade_command


def _stub_brew_with_upgrade_available(monkeypatch):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.9.3")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.9.4")
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="brew",
            binary_path="/opt/homebrew/bin/rapid-mlx",
            upgrade_command="brew upgrade raullenchai/rapid-mlx/rapid-mlx",
            upgrade_argv=["brew", "upgrade", "raullenchai/rapid-mlx/rapid-mlx"],
        ),
    )


def test_dry_run_does_not_invoke_subprocess(monkeypatch, capsys):
    _stub_brew_with_upgrade_available(monkeypatch)
    args = SimpleNamespace(yes=False, dry_run=True)
    with (
        patch("subprocess.run") as run,
        patch("builtins.input") as inp,
    ):
        upgrade_command(args)
        run.assert_not_called()
        inp.assert_not_called()
    out = capsys.readouterr().out
    assert "Current:  rapid-mlx 0.9.3" in out
    assert "Latest:   rapid-mlx 0.9.4" in out
    assert "brew upgrade raullenchai/rapid-mlx/rapid-mlx" in out
    assert "(dry-run — not executed" in out


def test_dry_run_short_circuits_before_yes_prompt(monkeypatch, capsys):
    """`--dry-run -y` is well-defined: dry-run wins (no surprise mutation)."""
    _stub_brew_with_upgrade_available(monkeypatch)
    args = SimpleNamespace(yes=True, dry_run=True)
    with patch("subprocess.run") as run:
        upgrade_command(args)
        run.assert_not_called()


def test_non_dry_run_still_calls_subprocess(monkeypatch):
    """Regression guard: adding --dry-run must not skip the real path."""
    _stub_brew_with_upgrade_available(monkeypatch)
    args = SimpleNamespace(yes=True, dry_run=False)
    fake_result = MagicMock(returncode=0)
    with (
        patch("subprocess.run", return_value=fake_result) as run,
        pytest.raises(SystemExit) as exc,
    ):
        upgrade_command(args)
    run.assert_called_once_with(
        ["brew", "upgrade", "raullenchai/rapid-mlx/rapid-mlx"], check=False
    )
    assert exc.value.code == 0


def test_dry_run_returns_silently_when_already_up_to_date(monkeypatch, capsys):
    """If current == latest, upgrade_command returns before consulting
    install method. --dry-run should not change that — still a clean
    return, no subprocess."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.9.4")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.9.4")
    args = SimpleNamespace(yes=False, dry_run=True)
    with patch("subprocess.run") as run:
        upgrade_command(args)
        run.assert_not_called()
    out = capsys.readouterr().out
    assert "Already up to date" in out
    assert "dry-run" not in out  # no point printing dry-run if there's nothing to do


# ---------------------------------------------------------------------------
# `update` alias — muscle-memory parity with npm/brew/claude/rustup.
# Exercised via the real argparse (subprocess `--help`) so we pin that the
# alias is registered AND accepts the same flags, without importing the giant
# CLI module in-process (which would drag in torch/mlx-vlm). Mirrors the
# `_serve_help_stdout` pattern in test_mtp_cli_wiring.py.
# ---------------------------------------------------------------------------


def _cli_help_stdout(*argv: str):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-m", "vllm_mlx.cli", *argv],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_update_alias_is_registered_and_routes_to_upgrade():
    """`rapid-mlx update --help` exits 0 (alias registered) and its help is
    the upgrade help — an unregistered subcommand would exit 2 instead."""
    proc = _cli_help_stdout("update", "--help")
    assert proc.returncode == 0, proc.stderr
    # --help of the alias renders the upgrade parser: same flags, same
    # description note pointing back at `upgrade`. Collapse whitespace
    # first — argparse hard-wraps the description to terminal width.
    normalized = " ".join(proc.stdout.split())
    assert "--dry-run" in normalized
    assert "'rapid-mlx update' is an alias for 'upgrade'" in normalized


def test_update_alias_accepts_upgrade_flags():
    """The alias shares the parser, so `-y`/`--dry-run` parse identically."""
    proc = _cli_help_stdout("update", "-y", "--dry-run", "--help")
    assert proc.returncode == 0, proc.stderr
