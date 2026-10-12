"""The terminal-only first serve command on CLI discovery paths."""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rapid_mlx import cli, front_door


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def fixed_ram(monkeypatch):
    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: 16.0)
    monkeypatch.setattr(
        "rapid_mlx._version_check.print_staleness_warning_if_any", lambda: None
    )


@pytest.mark.parametrize("argv", [["--help"], ["help"]])
def test_top_level_help_ends_with_policy_pick(argv, monkeypatch):
    out = Terminal()
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", *argv])
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(cli, "_start_quiet_lifecycle", lambda *_a, **_kw: None)
    if argv == ["--help"]:
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
    else:
        cli.main()
    assert out.getvalue().rstrip().endswith("Next: rapid-mlx serve qwen3.5-4b-4bit")


def test_bare_terminal_screen_ends_with_policy_pick():
    state = front_door.FrontDoorState(
        version="1.0",
        chip=None,
        ram_gb=16.0,
        picks=[{"role": "smart", "alias": "qwen3.5-4b-4bit"}],
        selected="qwen3.5-4b-4bit",
    )
    assert front_door.render_screen(state).endswith(
        "Next: rapid-mlx serve qwen3.5-4b-4bit"
    )
    assert "Next:" not in front_door.render_non_tty("1.0", 16.0, state.picks)


def test_bare_fallback_help_ends_with_policy_pick(monkeypatch):
    out = Terminal()
    monkeypatch.setattr(sys, "argv", ["rapid-mlx"])
    monkeypatch.setattr(sys, "stdout", out)

    def unavailable(**_kw):
        raise front_door.FrontDoorUnavailableError()

    monkeypatch.setattr(front_door, "run_bare", unavailable)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert out.getvalue().rstrip().endswith("Next: rapid-mlx serve qwen3.5-4b-4bit")


@pytest.mark.parametrize("cached", [False, True])
def test_models_and_ls_terminal_views_end_with_hint(cached, monkeypatch):
    out = Terminal()
    if cached:
        monkeypatch.setattr(cli, "_print_cached_models", lambda: print("Cached"))
        monkeypatch.setattr(
            "rapid_mlx.byom.imports.print_imports_section", lambda: None
        )
    with patch.object(sys, "stdout", out):
        cli.models_command(SimpleNamespace(cached=cached, json=False))
    assert out.getvalue().rstrip().endswith("Next: rapid-mlx serve qwen3.5-4b-4bit")


def test_ls_dispatch_ends_with_hint(monkeypatch):
    out = Terminal()
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "ls"])
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(cli, "_print_cached_models", lambda: print("Cached"))
    monkeypatch.setattr("rapid_mlx.byom.imports.print_imports_section", lambda: None)
    cli.main()
    assert out.getvalue().rstrip().endswith("Next: rapid-mlx serve qwen3.5-4b-4bit")


@pytest.mark.parametrize(
    "ram_gb,alias,expected",
    [
        (16.0, "qwen3.5-4b-4bit", "qwen3.5-4b-4bit"),
        (24.0, "qwen3.5-4b-4bit", "qwen3.5-4b-4bit"),
        (
            24.0,
            "mlx-community/Qwen3.5-4B-MLX-4bit",
            "bonsai-27b-2bit",
        ),
        (16.0, "qwen3.5-27b-4bit", "qwen3.5-4b-4bit"),
        (16.0, "bonsai-27b-2bit", "qwen3.5-4b-4bit"),
        (8.0, "qwen3.5-4b-4bit", "lfm2.5-2.6b-4bit"),
    ],
)
def test_info_hint_uses_alias_only_when_it_fits(ram_gb, alias, expected, monkeypatch):
    out = Terminal()
    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: ram_gb)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.estimate_repo_size_bytes", lambda *_a: None
    )
    monkeypatch.setattr(cli, "_print_dflash_status", lambda *_a: None)
    monkeypatch.setattr(cli, "_print_ddtree_status", lambda *_a: None)
    with patch.object(sys, "stdout", out):
        cli.info_command(SimpleNamespace(model=alias))
    assert out.getvalue().rstrip().endswith(f"Next: rapid-mlx serve {expected}")


@pytest.mark.parametrize("cached", [False, True])
def test_models_json_terminal_output_is_valid_json(cached):
    out = Terminal()
    with patch.object(sys, "stdout", out):
        cli.models_command(SimpleNamespace(cached=cached, json=True))
    assert isinstance(json.loads(out.getvalue()), (list, dict))
    assert "Next:" not in out.getvalue()


def test_non_tty_and_json_outputs_have_no_hint(monkeypatch):
    out = io.StringIO()
    with patch.object(sys, "stdout", out):
        cli.models_command(SimpleNamespace(cached=False, json=True))
    assert "Next:" not in out.getvalue()
    out = io.StringIO()
    with patch.object(sys, "stdout", out):
        cli.models_command(SimpleNamespace(cached=False, json=False))
        cli.info_command(SimpleNamespace(model="qwen3.5-4b-4bit"))
    assert "Next:" not in out.getvalue()
    out = Terminal()
    with patch.object(sys, "stdout", out):
        cli.info_command(SimpleNamespace(model="qwen3.5-4b-4bit", json=True))
    assert "Next:" not in out.getvalue()


def test_piped_help_has_no_hint(monkeypatch):
    out = io.StringIO()
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "--help"])
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(cli, "_start_quiet_lifecycle", lambda *_a, **_kw: None)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert "Next:" not in out.getvalue()


@pytest.mark.parametrize("argv", [["version"], ["--version"], ["-V"]])
def test_version_has_no_hint(argv, monkeypatch):
    out = Terminal()
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", *argv])
    monkeypatch.setattr(sys, "stdout", out)
    if argv == ["version"]:
        cli.main()
    else:
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
    assert out.getvalue().strip().startswith("rapid-mlx ")
    assert "Next:" not in out.getvalue()
