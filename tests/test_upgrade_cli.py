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
