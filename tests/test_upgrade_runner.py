# SPDX-License-Identifier: Apache-2.0
"""Upgrade output filtering and uv configuration compatibility."""

import re
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rapid_mlx import _upgrade_runner as runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def clean_uv(monkeypatch, tmp_path):
    monkeypatch.delenv("UV_CONFIG_FILE", raising=False)
    monkeypatch.delenv("UV_NO_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CONFIG_DIRS", str(tmp_path / "xdg"))
    monkeypatch.setattr(runner, "_SYSTEM_CONFIG_DIR", tmp_path / "etc")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return tmp_path


def test_filtered_pip_output_preserves_other_warnings_and_exit_status(capfd):
    script = "import sys; sys.stderr.write(\"WARNING: transformers 5.15.1 does not provide the extra 'tokenizers'\\nWARNING: another issue\\n\"); sys.exit(3)"
    assert runner._run_filtered([sys.executable, "-c", script]) == 3
    assert capfd.readouterr().err == "WARNING: another issue\n"


def test_filtered_pip_output_replaces_invalid_utf8(capfd):
    script = "import sys; sys.stderr.buffer.write(b'bad: \\xff\\n')"
    assert runner._run_filtered([sys.executable, "-c", script]) == 0
    assert capfd.readouterr().err == "bad: \ufffd\n"


def test_pip_routes_to_filter(monkeypatch):
    filtered = Mock(return_value=3)
    monkeypatch.setattr(runner, "_run_filtered", filtered)
    argv = [sys.executable, "-m", "pip", "install"]
    assert runner.run_upgrade_command(argv) == 3
    filtered.assert_called_once_with(argv)


def test_uv_supplies_temporary_metadata(monkeypatch, clean_uv):
    def run(argv, **kwargs):
        assert argv == ["/bin/uv", "tool", "upgrade", "rapid-mlx"]
        assert (
            Path(kwargs["env"]["UV_CONFIG_FILE"]).read_text()
            == runner.UV_DEPENDENCY_METADATA
        )
        return SimpleNamespace(returncode=4)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.run_upgrade_command(["/bin/uv", "tool", "upgrade", "rapid-mlx"]) == 4


@pytest.mark.parametrize(
    "config",
    [
        "env",
        "no_config",
        "user",
        "project",
        "bad_project",
        "bad_tool",
        "system",
        "xdg",
        "parent",
        "symlink",
    ],
)
def test_uv_respects_existing_configuration(monkeypatch, clean_uv, config):
    if config == "env":
        monkeypatch.setenv("UV_CONFIG_FILE", "")
    elif config == "no_config":
        monkeypatch.setenv("UV_NO_CONFIG", "")
    elif config in ("user", "system", "xdg"):
        base = {
            "user": clean_uv / "config",
            "system": clean_uv / "etc",
            "xdg": clean_uv / "xdg",
        }[config]
        (base / "uv").mkdir(parents=True)
        (base / "uv" / "uv.toml").touch()
    elif config == "parent":
        (clean_uv / "uv.toml").touch()
    elif config == "symlink":
        Path("uv.toml").symlink_to("missing")
    else:
        Path("pyproject.toml").write_text(
            {
                "project": "[tool.uv]\n",
                "bad_project": "[invalid",
                "bad_tool": "tool = 1\n",
            }[config]
        )
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(runner.subprocess, "run", run)
    argv = ["uv", "tool", "upgrade", "rapid-mlx"]
    assert runner.run_upgrade_command(argv) == 0
    run.assert_called_once_with(argv)


def test_metadata_stays_in_sync_with_installer_and_pin():
    metadata = tomllib.loads(runner.UV_DEPENDENCY_METADATA)["dependency-metadata"][0]
    installer = (ROOT / "install.sh").read_text()
    heredoc = installer.split("<<'EOF'\n[[dependency-metadata]]", 1)[1].split(
        "\nEOF", 1
    )[0]
    installer_metadata = tomllib.loads("[[dependency-metadata]]" + heredoc)[
        "dependency-metadata"
    ][0]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pin = next(
        dep
        for dep in project["project"]["dependencies"]
        if dep.startswith("mlx-video-with-audio==")
    )
    assert metadata == installer_metadata
    assert metadata["version"] == re.match(
        r"mlx-video-with-audio==([^;\s]+)", pin
    ).group(1)


def test_other_commands_use_plain_subprocess(monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=7))
    monkeypatch.setattr(runner.subprocess, "run", run)
    argv = ["brew", "upgrade", "rapid-mlx"]
    assert runner.run_upgrade_command(argv) == 7
    run.assert_called_once_with(argv, check=False)
