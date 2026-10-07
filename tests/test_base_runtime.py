# SPDX-License-Identifier: Apache-2.0
"""Guidance for runtimes that now ship in the base ``rapid-mlx`` install."""

from __future__ import annotations

import io
import sys
from types import SimpleNamespace

import pytest

import rapid_mlx
from rapid_mlx.runtime import base_runtime, optional_runtime


@pytest.mark.parametrize("extra", ["vision", "image", "video", "dflash", "mtp"])
def test_bundled_extras_repair_the_pinned_base_package(extra) -> None:
    assert base_runtime.is_base_runtime_extra(extra)
    assert base_runtime.runtime_install_spec(extra, "1.2.3") == "rapid-mlx==1.2.3"


@pytest.mark.parametrize("extra", ["audio", "embeddings", "chat", "computer-use"])
def test_opt_in_extras_keep_the_extra_specifier(extra) -> None:
    assert not base_runtime.is_base_runtime_extra(extra)
    assert (
        base_runtime.runtime_install_spec(extra, "1.2.3")
        == f"rapid-mlx[{extra}]==1.2.3"
    )


def _detect(monkeypatch, method: str) -> None:
    monkeypatch.setattr(
        "rapid_mlx._version_check.detect_install_method",
        lambda: SimpleNamespace(method=method),
    )


def test_bundled_hint_says_the_runtime_ships_with_rapid_mlx(monkeypatch) -> None:
    _detect(monkeypatch, "pip")
    hint = optional_runtime.optional_extra_install_hint(
        "vision", version="1.2.3", include_paths=False
    )
    assert hint == (
        "The vision runtime ships with rapid-mlx but is missing or damaged in "
        "this environment. Repair the install with:\n"
        "    python -m pip install rapid-mlx==1.2.3"
    )
    assert "[vision]" not in hint


def test_opt_in_hint_is_unchanged(monkeypatch) -> None:
    _detect(monkeypatch, "pip")
    hint = optional_runtime.optional_extra_install_hint(
        "audio", version="1.2.3", include_paths=False
    )
    assert hint == (
        "Install the optional runtime with:\n"
        "    python -m pip install 'rapid-mlx[audio]==1.2.3'"
    )


def test_brew_bundled_repair_switches_to_an_isolated_base_install(
    monkeypatch,
) -> None:
    _detect(monkeypatch, "brew")
    command = optional_runtime.optional_extra_repair_command("image", version="1.2.3")
    assert "does not carry this runtime" in command
    assert "optional extras" not in command
    assert command.endswith(
        "brew uninstall rapid-mlx && uv tool install rapid-mlx==1.2.3"
    )


def test_bundled_prompt_offers_a_repair_without_a_size(monkeypatch) -> None:
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        optional_runtime, "_read_posix_prompt_response", lambda *_args: "n"
    )
    result = optional_runtime._prompt_to_install("image")
    assert result is optional_runtime.PromptResult.DECLINED
    assert (
        stderr.getvalue() == "Repair rapid-mlx now (restores its image runtime)? [y/N] "
    )


def test_python_upgrade_hint_points_at_a_supported_interpreter() -> None:
    hint = base_runtime.python_upgrade_hint()
    assert "Python 3.11 or newer" in hint
    assert (
        f"uv tool install --force --python 3.12 'rapid-mlx=={rapid_mlx.__version__}'"
        in hint
    )


def test_desktop_sidecar_bundled_hint_asks_for_an_app_reinstall(monkeypatch) -> None:
    monkeypatch.setattr(optional_runtime, "_running_in_desktop_sidecar", lambda: True)
    hint = optional_runtime.optional_extra_install_hint("video", version="1.2.3")
    assert "Reinstall Rapid-MLX Desktop" in hint
    assert "pip install" not in hint


def test_desktop_sidecar_opt_in_hint_is_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(optional_runtime, "_running_in_desktop_sidecar", lambda: True)
    _detect(monkeypatch, "pip")
    hint = optional_runtime.optional_extra_install_hint(
        "audio", version="1.2.3", include_paths=False
    )
    assert hint.startswith("Install the optional runtime with:")
