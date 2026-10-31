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

from rapid_mlx import _version_check as vc
from rapid_mlx.cli import upgrade_command


def _stub_brew_with_upgrade_available(monkeypatch):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.9.3")
    monkeypatch.setattr(vc, "installed_version_in_fresh_process", lambda info: "0.9.4")
    monkeypatch.setattr(
        vc,
        "get_release_info",
        lambda force_refresh=False, need_brew=False: vc.ReleaseInfo("0.9.4"),
    )
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="brew",
            binary_path="/opt/homebrew/bin/rapid-mlx",
            upgrade_command="brew upgrade rapid-mlx",
            upgrade_argv=["brew", "upgrade", "rapid-mlx"],
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
    assert "brew upgrade rapid-mlx" in out
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
    run.assert_called_once_with(["brew", "upgrade", "rapid-mlx"], check=False)
    assert exc.value.code == 0


@pytest.mark.parametrize(
    ("method", "command", "argv"),
    [
        ("uv", "uv tool upgrade rapid-mlx", ["uv", "tool", "upgrade", "rapid-mlx"]),
        ("pipx", "pipx upgrade rapid-mlx", ["pipx", "upgrade", "rapid-mlx"]),
    ],
)
def test_missing_manager_command_exits_cleanly(
    monkeypatch, capsys, method, command, argv
):
    """A removed manager must produce an actionable error, not a traceback."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.9.3")
    monkeypatch.setattr(
        vc,
        "get_release_info",
        lambda force_refresh=False, need_brew=False: vc.ReleaseInfo("0.9.4"),
    )
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method=method,
            binary_path=f"/manager/rapid-mlx/{method}/bin/rapid-mlx",
            upgrade_command=command,
            upgrade_argv=argv,
        ),
    )
    missing = FileNotFoundError(2, "No such file or directory", argv[0])

    with (
        patch("subprocess.run", side_effect=missing),
        pytest.raises(SystemExit) as exc,
    ):
        upgrade_command(SimpleNamespace(yes=True, dry_run=False))

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert f"Upgrade command not found: {argv[0]}" in out
    assert f"Reinstall {method}" in out


def test_global_pipx_prints_manual_command_without_executing(monkeypatch, capsys):
    """Global pipx needs sudo, so the CLI must never execute its empty argv."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.9.3")
    monkeypatch.setattr(
        vc,
        "get_release_info",
        lambda force_refresh=False, need_brew=False: vc.ReleaseInfo("0.9.4"),
    )
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="unknown",
            binary_path="/opt/pipx/venvs/rapid-mlx/bin/rapid-mlx",
            upgrade_command="sudo pipx upgrade --global rapid-mlx",
            upgrade_argv=[],
        ),
    )

    with patch("subprocess.run") as run:
        upgrade_command(SimpleNamespace(yes=True, dry_run=False))

    run.assert_not_called()
    out = capsys.readouterr().out
    assert "sudo pipx upgrade --global rapid-mlx" in out
    assert "run the command above manually" in out


def test_dry_run_returns_silently_when_already_up_to_date(monkeypatch, capsys):
    """If current == latest, upgrade_command returns before consulting
    install method. --dry-run should not change that — still a clean
    return, no subprocess."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.9.4")
    monkeypatch.setattr(
        vc,
        "get_release_info",
        lambda force_refresh=False, need_brew=False: vc.ReleaseInfo("0.9.4"),
    )
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
        [sys.executable, "-m", "rapid_mlx.cli", *argv],
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


@pytest.mark.parametrize(
    "method,guidance",
    [
        ("pip", "python3.12 -m venv"),
        ("install_sh", "curl -fsSL https://rapidmlx.com/install.sh | bash"),
        ("uv", "uv tool upgrade --python 3.12 rapid-mlx"),
        ("pipx", "pipx reinstall --python python3.12 rapid-mlx"),
    ],
)
@pytest.mark.parametrize("dry_run", [False, True])
def test_python_floor_guidance(monkeypatch, capsys, method, guidance, dry_run):
    monkeypatch.setattr(vc, "_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(
        vc, "get_release_info", lambda **kw: vc.ReleaseInfo("1.0.1", ">=3.12")
    )
    monkeypatch.setattr(vc, "_python_supports", lambda spec: False)
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(method, "upgrade cmd", ["upgrade"]),
    )
    with patch("subprocess.run", return_value=MagicMock(returncode=1)) as run:
        if dry_run:
            upgrade_command(SimpleNamespace(yes=True, dry_run=True))
        else:
            with pytest.raises(SystemExit) as exc:
                upgrade_command(SimpleNamespace(yes=True, dry_run=False))
            assert exc.value.code == 1
        if dry_run or method == "install_sh":
            run.assert_not_called()
        else:
            run.assert_called_once_with(["upgrade"], check=False)
    out = capsys.readouterr().out
    assert "requires Python >=3.12" in out
    assert "`upgrade cmd` cannot install 1.0.1 on this Python. To move to 1.0.1:" in out
    assert guidance in out


@pytest.mark.parametrize(
    "method,verified,code,expected",
    [
        ("pip", "1.0.0", 1, "still 1.0.0"),
        ("pip", "1.0.1", 0, "now 1.0.1"),
        ("brew", "1.0.0", 1, "brew update && brew upgrade rapid-mlx"),
        (
            "pip",
            None,
            1,
            "Could not confirm the installed version; run `rapid-mlx --version` to check.",
        ),
    ],
)
def test_upgrade_verifies_result(monkeypatch, capsys, method, verified, code, expected):
    monkeypatch.setattr(vc, "_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(
        vc,
        "get_release_info",
        lambda **kw: vc.ReleaseInfo("1.0.1", brew_stable="1.0.1"),
    )
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(method, "upgrade", ["upgrade"]),
    )
    monkeypatch.setattr(vc, "installed_version_in_fresh_process", lambda info: verified)
    with (
        patch("subprocess.run", return_value=MagicMock(returncode=0)) as run,
        pytest.raises(SystemExit) as exc,
    ):
        upgrade_command(SimpleNamespace(yes=True, dry_run=False))
    run.assert_called_once()
    assert exc.value.code == code
    assert expected in capsys.readouterr().out


@pytest.mark.parametrize("dry_run", [False, True])
def test_brew_lag_does_not_run(monkeypatch, capsys, dry_run):
    monkeypatch.setattr(vc, "_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(
        vc,
        "get_release_info",
        lambda **kw: vc.ReleaseInfo("1.0.1", brew_stable="1.0.0"),
    )
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo("brew", "brew upgrade rapid-mlx", ["brew"]),
    )
    with patch("subprocess.run") as run:
        if dry_run:
            upgrade_command(SimpleNamespace(yes=True, dry_run=True))
        else:
            with pytest.raises(SystemExit) as exc:
                upgrade_command(SimpleNamespace(yes=True, dry_run=False))
            assert exc.value.code == 1
    run.assert_not_called()
    assert "Homebrew has not published it yet" in capsys.readouterr().out


@pytest.mark.parametrize("verified,code", [("1.0.1", 0), ("1.0.0", 1)])
def test_brew_upgrade_targets_formula_version(monkeypatch, capsys, verified, code):
    monkeypatch.setattr(vc, "_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(
        vc,
        "get_release_info",
        lambda **kw: vc.ReleaseInfo("1.0.2", brew_stable="1.0.1"),
    )
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo("brew", "brew upgrade rapid-mlx", ["brew"]),
    )
    monkeypatch.setattr(vc, "installed_version_in_fresh_process", lambda info: verified)
    with (
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        pytest.raises(SystemExit) as exc,
    ):
        upgrade_command(SimpleNamespace(yes=True, dry_run=False))
    assert exc.value.code == code
    out = capsys.readouterr().out
    assert (
        "Homebrew offers 1.0.1 now; 1.0.2 will follow once the formula is updated."
        in out
    )
    if code:
        assert "(latest is 1.0.1)" in out


def test_upgrade_propagates_command_failure(monkeypatch):
    _stub_brew_with_upgrade_available(monkeypatch)
    with (
        patch("subprocess.run", return_value=MagicMock(returncode=7)),
        pytest.raises(SystemExit) as exc,
    ):
        upgrade_command(SimpleNamespace(yes=True, dry_run=False))
    assert exc.value.code == 7


def test_upgrade_reports_python_guidance_if_floor_changes(monkeypatch, capsys):
    monkeypatch.setattr(vc, "_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(
        vc, "get_release_info", lambda **kw: vc.ReleaseInfo("1.0.1", ">=3.12")
    )
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo("uv", "upgrade", ["upgrade"]),
    )
    monkeypatch.setattr(vc, "_python_supports", MagicMock(side_effect=[True, False]))
    monkeypatch.setattr(vc, "installed_version_in_fresh_process", lambda info: "1.0.0")
    with (
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        pytest.raises(SystemExit) as exc,
    ):
        upgrade_command(SimpleNamespace(yes=True, dry_run=False))
    assert exc.value.code == 1
    assert "uv tool upgrade --python 3.12 rapid-mlx" in capsys.readouterr().out


@pytest.mark.parametrize("method", ["pip", "uv", "pipx"])
@pytest.mark.parametrize("verified,code", [("1.0.1", 0), ("1.0.0", 1), (None, 1)])
def test_python_floor_tries_compatible_upgrade(
    monkeypatch, capsys, method, verified, code
):
    monkeypatch.setattr(vc, "_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(
        vc, "get_release_info", lambda **kw: vc.ReleaseInfo("1.0.2", ">=3.12")
    )
    monkeypatch.setattr(vc, "_python_supports", lambda spec: False)
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(method, "upgrade", ["upgrade"]),
    )
    monkeypatch.setattr(vc, "installed_version_in_fresh_process", lambda info: verified)
    with (
        patch("subprocess.run", return_value=MagicMock(returncode=0)) as run,
        pytest.raises(SystemExit) as exc,
    ):
        upgrade_command(SimpleNamespace(yes=True, dry_run=False))
    run.assert_called_once_with(["upgrade"], check=False)
    assert exc.value.code == code
    out = capsys.readouterr().out
    if code == 0:
        assert (
            "  ✓ rapid-mlx is now 1.0.1, the newest release this Python could install. 1.0.2 needs Python >=3.12; follow the steps above to move to it."
            in out
        )
    elif verified is None:
        assert "Could not confirm the installed version" in out
        assert "This Python already has the newest release it can run." not in out
    else:
        assert "This Python already has the newest release it can run." in out


@pytest.mark.parametrize("method", ["pip", "uv", "pipx"])
def test_python_floor_upgrade_failure(monkeypatch, capsys, method):
    monkeypatch.setattr(vc, "_installed_version", lambda: "1.0.0")
    monkeypatch.setattr(
        vc, "get_release_info", lambda **kw: vc.ReleaseInfo("1.0.2", ">=3.12")
    )
    monkeypatch.setattr(vc, "_python_supports", lambda spec: False)
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(method, "upgrade", ["upgrade"]),
    )
    with (
        patch("subprocess.run", return_value=MagicMock(returncode=7)),
        pytest.raises(SystemExit) as exc,
    ):
        upgrade_command(SimpleNamespace(yes=True, dry_run=False))
    assert exc.value.code == 7
    out = capsys.readouterr().out
    assert "Upgrade command failed" in out
    assert "This Python already has the newest release it can run." not in out


def test_upgrade_interrupt_exits_130(monkeypatch, capsys):
    _stub_brew_with_upgrade_available(monkeypatch)
    with (
        patch("subprocess.run", side_effect=KeyboardInterrupt),
        pytest.raises(SystemExit) as exc,
    ):
        upgrade_command(SimpleNamespace(yes=True, dry_run=False))
    assert exc.value.code == 130
    assert "Interrupted." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("method", "binary", "label"),
    [("install_sh", "/some/bin/rapid-mlx", "one-line installer"), ("pip", None, "pip")],
)
def test_upgrade_install_display(monkeypatch, capsys, method, binary, label):
    import sys

    _stub_brew_with_upgrade_available(monkeypatch)
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(method, "upgrade", ["upgrade"], binary),
    )
    upgrade_command(SimpleNamespace(yes=False, dry_run=True))
    out = capsys.readouterr().out
    assert f"  Install:  {label} ({binary or sys.executable})" in out
    assert "unknown path" not in out
