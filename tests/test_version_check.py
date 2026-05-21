# SPDX-License-Identifier: Apache-2.0
"""Tests for the staleness-warning helper.

The helper is opt-in (TTY+no-CI), cache-aware, and fail-silent on
network errors. Tests pin those guarantees so a future "let's add a
real call" change can't accidentally break the CLI on an offline
laptop.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx import _version_check as vc

# --- _parse_version ---------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.6.14", (0, 6, 14)),
        ("v0.6.14", (0, 6, 14)),  # leading v stripped
        ("1.0.0", (1, 0, 0)),
        ("0.6.14.dev3", (0, 6, 14)),  # dev suffix tolerated, takes patch
    ],
)
def test_parse_version_accepts_typical(raw, expected):
    assert vc._parse_version(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "0.6",  # missing patch
        "abc",
        "0.6.x",
    ],
)
def test_parse_version_rejects_garbage(raw):
    assert vc._parse_version(raw) is None


# --- staleness_warning logic (no network) -----------------------------


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache at tmp + force interactive mode + no fetch."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(vc, "_cache_path", lambda: cache_dir / "version_check.json")
    # Disable the disabled() short-circuit so logic runs.
    monkeypatch.setattr(vc, "_disabled", lambda: False)
    # Block real network — every test MUST stub _fetch_latest_from_github.
    monkeypatch.setattr(
        vc,
        "_fetch_latest_from_github",
        lambda: pytest.fail("real network call leaked into test"),
    )
    return cache_dir


def _seed_cache(cache_dir: Path, latest: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "version_check.json").write_text(
        json.dumps({"latest": latest, "ts": 9999})
    )


def test_warns_when_2_or_more_patch_behind(isolated_cache, monkeypatch):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.14")
    _seed_cache(isolated_cache, "0.6.16")

    msg = vc.staleness_warning()
    assert msg is not None
    assert "0.6.14" in msg
    assert "0.6.16" in msg
    assert "rapid-mlx upgrade" in msg


def test_silent_when_only_1_patch_behind(isolated_cache, monkeypatch):
    """1 patch behind is normal noise — minor bug-fix releases happen.
    We only want to nag when feature releases are missed (≥2 lag).
    """
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.15")
    _seed_cache(isolated_cache, "0.6.16")

    assert vc.staleness_warning() is None


def test_silent_when_current(isolated_cache, monkeypatch):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.16")
    _seed_cache(isolated_cache, "0.6.16")

    assert vc.staleness_warning() is None


def test_silent_when_dev_ahead(isolated_cache, monkeypatch):
    """Devs running their own builds ahead of main shouldn't get a
    warning that confuses them about phantom 'latest' releases."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.7.0")
    _seed_cache(isolated_cache, "0.6.16")

    assert vc.staleness_warning() is None


def test_silent_across_minor_boundary(isolated_cache, monkeypatch):
    """If user is on 0.6.x and 0.7.x is out, that's a minor bump — they
    might be intentionally pinning the 0.6 line. Don't auto-suggest a
    cross-minor upgrade."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.10")
    _seed_cache(isolated_cache, "0.7.0")

    assert vc.staleness_warning() is None


def test_silent_when_offline(tmp_path, monkeypatch):
    """No cache + GitHub fetch fails → no warning, no exception."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(vc, "_cache_path", lambda: cache_dir / "version_check.json")
    monkeypatch.setattr(vc, "_disabled", lambda: False)
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.14")
    monkeypatch.setattr(vc, "_fetch_latest_from_github", lambda: None)

    assert vc.staleness_warning() is None


def test_silent_when_disabled(monkeypatch):
    monkeypatch.setattr(vc, "_disabled", lambda: True)
    # Even with stub installed/cache that would warn, disabled wins.
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.14")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.16")

    assert vc.staleness_warning() is None


def test_silent_when_dev_build_unparseable(isolated_cache, monkeypatch):
    """``rapid-mlx`` not installed (running from source tree without
    install) → ``pkg_version`` raises and we return None — no warning."""
    monkeypatch.setattr(vc, "_installed_version", lambda: None)

    assert vc.staleness_warning() is None


# --- _disabled honors RAPID_MLX_DISABLE_VERSION_CHECK ----------------


def test_disabled_via_env(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_DISABLE_VERSION_CHECK", "1")
    assert vc._disabled() is True


def test_disabled_in_ci(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_DISABLE_VERSION_CHECK", raising=False)
    monkeypatch.setenv("CI", "true")
    assert vc._disabled() is True


# --- print_staleness_warning_if_any never raises ---------------------


def test_print_helper_swallows_all_exceptions(monkeypatch, capsys):
    def boom():
        raise RuntimeError("simulated GitHub outage")

    monkeypatch.setattr(vc, "staleness_warning", boom)
    # Must not raise — the CLI must never break because of a staleness
    # check. capsys just makes sure we don't pollute stdout either.
    vc.print_staleness_warning_if_any()
    captured = capsys.readouterr()
    assert captured.out == ""


# --- staleness warning recommends `rapid-mlx upgrade` ----------------


def test_warning_message_recommends_upgrade_subcommand(isolated_cache, monkeypatch):
    """The banner must point users at our own upgrade subcommand.

    Pre-0.6.31 we suggested raw ``brew upgrade rapid-mlx`` — wrong formula
    path (the tap is ``raullenchai/tap/rapid-mlx``) AND it stranded pip /
    install.sh users. The new flow centralises the install-method detection
    in ``rapid-mlx upgrade``, so the warning just needs to point there.
    """
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.20")
    _seed_cache(isolated_cache, "0.6.30")

    msg = vc.staleness_warning()
    assert msg is not None
    assert "rapid-mlx upgrade" in msg


# --- detect_install_method() -----------------------------------------


def test_detect_install_method_brew(monkeypatch):
    """A brew install resolves through realpath into ``/opt/homebrew/Cellar/``.

    The detector must spot that and return the *tap-qualified* formula path
    — ``brew upgrade rapid-mlx`` alone doesn't know about external taps and
    fails with ``Error: rapid-mlx not installed`` for users on the tap.
    """
    fake_binary = "/opt/homebrew/bin/rapid-mlx"
    fake_realpath = "/opt/homebrew/Cellar/rapid-mlx/0.6.20/bin/rapid-mlx"
    monkeypatch.setattr("shutil.which", lambda _name: fake_binary)
    monkeypatch.setattr(
        "os.path.realpath",
        lambda p: fake_realpath if p == fake_binary else p,
    )

    info = vc.detect_install_method()
    assert info.method == "brew"
    assert info.upgrade_command == "brew upgrade raullenchai/tap/rapid-mlx"
    assert info.upgrade_argv == ["brew", "upgrade", "raullenchai/tap/rapid-mlx"]
    assert info.binary_path == fake_binary


def test_detect_install_method_brew_linux(monkeypatch):
    """Linux Homebrew installs to ``/home/linuxbrew/.linuxbrew/`` — must
    detect there too, otherwise Linux-via-brew users get the pip command."""
    fake_binary = "/home/linuxbrew/.linuxbrew/bin/rapid-mlx"
    fake_realpath = "/home/linuxbrew/.linuxbrew/Cellar/rapid-mlx/0.6.20/bin/rapid-mlx"
    monkeypatch.setattr("shutil.which", lambda _name: fake_binary)
    monkeypatch.setattr(
        "os.path.realpath",
        lambda p: fake_realpath if p == fake_binary else p,
    )

    info = vc.detect_install_method()
    assert info.method == "brew"


def test_detect_install_method_install_sh(tmp_path, monkeypatch):
    """install.sh drops the binary in ``~/.local/bin`` — re-running the
    script is the only sane upgrade path for this install class.
    """
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    fake_binary = str(local_bin / "rapid-mlx")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("shutil.which", lambda _name: fake_binary)
    monkeypatch.setattr("os.path.realpath", lambda p: p)

    info = vc.detect_install_method()
    assert info.method == "install_sh"
    assert "install.sh" in info.upgrade_command


def test_detect_install_method_install_sh_via_symlink(tmp_path, monkeypatch):
    """install.sh actually creates a venv under ``~/.rapid-mlx/`` and
    symlinks the entry point into ``~/.local/bin/rapid-mlx``. ``realpath``
    resolves through the symlink, so a check that *only* looked at the
    resolved path classified install.sh users as 'pip' and silently
    suggested the wrong upgrade command. Pin the symlink case explicitly.
    """
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    venv_bin = home / ".rapid-mlx" / "bin"
    venv_bin.mkdir(parents=True)
    fake_binary = str(local_bin / "rapid-mlx")
    fake_realpath = str(venv_bin / "rapid-mlx")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr("shutil.which", lambda _name: fake_binary)
    monkeypatch.setattr(
        "os.path.realpath",
        lambda p: fake_realpath if p == fake_binary else p,
    )

    info = vc.detect_install_method()
    assert info.method == "install_sh"
    assert "install.sh" in info.upgrade_command
    # Pipe needs a shell — wrapped as ``bash -c <pipe>``, never `shell=True`.
    assert info.upgrade_argv[:2] == ["bash", "-c"]


def test_detect_install_method_pip_uses_sys_executable(monkeypatch):
    """When the binary path doesn't match brew or install.sh, fall back to
    pip — and use ``sys.executable -m pip`` so the upgrade lands in the
    same Python env that's currently running the CLI (matters when the
    user has multiple python3 installs).
    """
    import sys

    monkeypatch.setattr("shutil.which", lambda _name: "/some/other/path/rapid-mlx")
    monkeypatch.setattr("os.path.realpath", lambda p: p)

    info = vc.detect_install_method()
    assert info.method == "pip"
    assert info.upgrade_command.startswith(sys.executable)
    assert info.upgrade_command.endswith("-m pip install --upgrade rapid-mlx")
    # argv form is shell-safe even if sys.executable contains spaces — that
    # was a P0 in deepseek review (subprocess shell=True path injection).
    assert info.upgrade_argv == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "rapid-mlx",
    ]


def test_detect_install_method_no_binary_falls_back_to_pip(monkeypatch):
    """When ``rapid-mlx`` isn't on PATH (e.g. invoked via
    ``python -m vllm_mlx.cli``), default to pip so the upgrade subcommand
    still works."""
    monkeypatch.setattr("shutil.which", lambda _name: None)

    info = vc.detect_install_method()
    assert info.method == "pip"
    assert info.binary_path is None


# --- prompt_upgrade_if_available ------------------------------------------


@pytest.fixture
def interactive(monkeypatch):
    """Enable the prompt path: TTY on stdin+stderr, not disabled, not in CI."""
    monkeypatch.delenv("RAPID_MLX_DISABLE_VERSION_CHECK", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(vc.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(vc.sys.stderr, "isatty", lambda: True)


def test_prompt_returns_false_when_disabled(monkeypatch, interactive):
    monkeypatch.setenv("RAPID_MLX_DISABLE_VERSION_CHECK", "1")
    # Disabled MUST short-circuit before fetching anything.
    monkeypatch.setattr(
        vc, "get_latest_version",
        lambda force_refresh=False: pytest.fail("network leaked on disabled"),
    )
    assert vc.prompt_upgrade_if_available() is False


def test_prompt_returns_false_when_stdin_not_tty(monkeypatch, interactive):
    monkeypatch.setattr(vc.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        vc, "get_latest_version",
        lambda force_refresh=False: pytest.fail("network leaked on non-TTY"),
    )
    assert vc.prompt_upgrade_if_available() is False


def test_prompt_returns_false_when_already_current(monkeypatch, interactive):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.62")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.62")
    with patch("builtins.input") as inp:
        assert vc.prompt_upgrade_if_available() is False
        inp.assert_not_called()


def test_prompt_returns_false_when_local_ahead(monkeypatch, interactive):
    """Dev build one bump ahead of latest — never prompt downward."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.63")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.62")
    with patch("builtins.input") as inp:
        assert vc.prompt_upgrade_if_available() is False
        inp.assert_not_called()


def test_prompt_returns_false_when_dev_build_unparseable(monkeypatch, interactive):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.62.dev1+gabcdef")
    monkeypatch.setattr(vc, "_parse_version", lambda s: None if "dev" in s else (0, 6, 62))
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.62")
    with patch("builtins.input") as inp:
        assert vc.prompt_upgrade_if_available() is False
        inp.assert_not_called()


def test_prompt_returns_false_when_offline(monkeypatch, interactive):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: None)
    with patch("builtins.input") as inp:
        assert vc.prompt_upgrade_if_available() is False
        inp.assert_not_called()


def test_prompt_returns_false_when_user_declines(monkeypatch, interactive):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.62")
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="pip",
            upgrade_command="pip install -U rapid-mlx",
            upgrade_argv=["pip", "install", "-U", "rapid-mlx"],
        ),
    )
    with (
        patch("builtins.input", return_value="n"),
        patch("subprocess.run") as run,
    ):
        assert vc.prompt_upgrade_if_available() is False
        run.assert_not_called()


def test_prompt_returns_true_and_runs_upgrade_on_accept(monkeypatch, interactive):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.62")
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="brew",
            upgrade_command="brew upgrade raullenchai/tap/rapid-mlx",
            upgrade_argv=["brew", "upgrade", "raullenchai/tap/rapid-mlx"],
        ),
    )
    fake_result = MagicMock(returncode=0)
    # Empty answer == default Y.
    with (
        patch("builtins.input", return_value=""),
        patch("subprocess.run", return_value=fake_result) as run,
    ):
        assert vc.prompt_upgrade_if_available() is True
        run.assert_called_once_with(
            ["brew", "upgrade", "raullenchai/tap/rapid-mlx"], check=False
        )


def test_prompt_crosses_minor_boundary(monkeypatch, interactive):
    """``staleness_warning`` stays silent across minor bumps, but the
    interactive prompt opts in — user can still say no."""
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.62")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.7.0")
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="pip",
            upgrade_command="pip",
            upgrade_argv=["pip"],
        ),
    )
    with patch("builtins.input", return_value="n") as inp:
        assert vc.prompt_upgrade_if_available() is False
        inp.assert_called_once()


def test_prompt_never_raises(monkeypatch, interactive):
    """A bug in detect_install_method or _installed_version must never crash
    the CLI — silently skip and return False so serve continues to boot.
    """

    def boom():
        raise RuntimeError("simulated bug")

    monkeypatch.setattr(vc, "_installed_version", boom)
    # Must not raise.
    assert vc.prompt_upgrade_if_available() is False


def test_prompt_returns_false_on_keyboard_interrupt(monkeypatch, interactive):
    monkeypatch.setattr(vc, "_installed_version", lambda: "0.6.61")
    monkeypatch.setattr(vc, "get_latest_version", lambda force_refresh=False: "0.6.62")
    monkeypatch.setattr(
        vc,
        "detect_install_method",
        lambda: vc.InstallInfo(
            method="pip", upgrade_command="pip", upgrade_argv=["pip"]
        ),
    )
    with (
        patch("builtins.input", side_effect=KeyboardInterrupt()),
        patch("subprocess.run") as run,
    ):
        assert vc.prompt_upgrade_if_available() is False
        run.assert_not_called()
