# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for multimodal runtime fail-fast handling (#2860)."""

from __future__ import annotations

import plistlib
import sys
from types import ModuleType

import pytest

from vllm_mlx.models import mllm


def test_vision_runtime_reports_incompatible_mlx_vlm_version(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_vlm", ModuleType("mlx_vlm"))
    monkeypatch.setattr(mllm, "version", lambda _distribution: "0.7.0")

    status, detail = mllm.vision_runtime_status()

    assert status is mllm.VisionRuntimeStatus.INCOMPATIBLE
    assert detail == "0.7.0"


def test_cli_incompatible_runtime_is_actionable_and_not_reported_as_oom(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.INCOMPATIBLE, "installed 0.7.0"),
    )
    monkeypatch.setattr(mllm, "_managed_desktop_runtime", lambda: False)
    monkeypatch.setattr(mllm.sys, "executable", "/active/runtime/bin/python")

    with pytest.raises(SystemExit) as exc_info:
        mllm.require_mlx_vlm_or_exit("publisher/vision-model")

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "installed 0.7.0" in stderr
    assert "not a Metal out-of-memory error" in stderr
    assert "/active/runtime/bin/python -m pip" in stderr
    assert f"mlx-vlm=={mllm.VALIDATED_MLX_VLM_VERSION}" in stderr


def test_engine_guard_reports_missing_runtime_with_model_context(monkeypatch):
    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.ABSENT, "mlx_vlm"),
    )

    with pytest.raises(ImportError) as exc_info:
        mllm._require_mlx_vlm("publisher/vision-model")

    message = str(exc_info.value)
    assert "publisher/vision-model" in message
    assert "optional `mlx-vlm` dependency" in message


def test_validated_runtime_version_is_accepted(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_vlm", ModuleType("mlx_vlm"))
    monkeypatch.setattr(
        mllm, "version", lambda _distribution: mllm.VALIDATED_MLX_VLM_VERSION
    )

    status, detail = mllm.vision_runtime_status()

    assert status is mllm.VisionRuntimeStatus.OK
    assert detail is None


def test_runtime_override_repair_hint_requires_removal_after_app_install(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/alice")
    monkeypatch.setattr(
        mllm.sys,
        "executable",
        "/Users/alice/Library/Application Support/Rapid/runtime-override/"
        "rapid-mlx/python/bin/python3.12",
    )

    hint = mllm._vision_install_hint()

    assert "Install the current Rapid-MLX Desktop.app first" in hint
    assert "then remove" in hint
    assert "runtime-override/rapid-mlx" in hint
    assert " -m pip install" not in hint


def test_runtime_override_repair_hint_uses_active_custom_home(monkeypatch):
    monkeypatch.setenv("HOME", "/tmp/dogfood-home")
    monkeypatch.setattr(
        mllm.sys,
        "executable",
        "/tmp/dogfood-home/Library/Application Support/Rapid/runtime-override/"
        "rapid-mlx/python/bin/python3.12",
    )

    hint = mllm._vision_install_hint()

    assert "/tmp/dogfood-home/Library/Application Support/Rapid/" in hint
    assert "~/Library/Application Support" not in hint


def test_noncanonical_override_path_is_not_managed(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/alice")
    monkeypatch.setattr(
        mllm.sys,
        "executable",
        "/Users/alice/Library/Application Support/Rapid/runtime-override/bin/python",
    )

    assert mllm._managed_desktop_runtime_kind() is None


def test_runtime_override_stays_managed_without_home(monkeypatch):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setattr(
        mllm.sys,
        "executable",
        "/tmp/dogfood/Library/Application Support/Rapid/runtime-override/"
        "rapid-mlx/python/bin/python3.12",
    )

    assert mllm._managed_desktop_runtime_kind() == "runtime-override"
    assert "pip-install into" in mllm._vision_install_hint()


def test_managed_runtime_missing_dependency_never_recommends_pip(monkeypatch, tmp_path):
    app = tmp_path / "Rapid-MLX Desktop.app"
    info = app / "Contents" / "Info.plist"
    info.parent.mkdir(parents=True)
    with info.open("wb") as handle:
        plistlib.dump({"CFBundleIdentifier": "com.rapidmlx.rapid"}, handle)
    monkeypatch.setattr(
        mllm.sys,
        "executable",
        str(app / "Contents/Resources/rapid-mlx/python/bin/python3.12"),
    )

    hint = mllm._vlm_broken_install_hint("PIL")

    assert "Reinstall Rapid-MLX Desktop.app" in hint
    assert "pip install" not in hint


def test_unrelated_app_runtime_is_not_managed(monkeypatch, tmp_path):
    app = tmp_path / "Unrelated.app"
    info = app / "Contents" / "Info.plist"
    info.parent.mkdir(parents=True)
    with info.open("wb") as handle:
        plistlib.dump({"CFBundleIdentifier": "com.example.unrelated"}, handle)
    monkeypatch.setattr(
        mllm.sys,
        "executable",
        str(app / "Contents/Resources/rapid-mlx/python/bin/python3.12"),
    )

    assert mllm._managed_desktop_runtime_kind() is None


def test_embedded_runtime_without_valid_plist_is_not_managed(monkeypatch, tmp_path):
    app = tmp_path / "Rapid-MLX Desktop.app"
    executable = app / "Contents/Resources/rapid-mlx/python/bin/python3.12"
    executable.parent.mkdir(parents=True)
    monkeypatch.setattr(mllm.sys, "executable", str(executable))

    assert mllm._managed_desktop_runtime_kind() is None


def test_vision_runtime_reports_missing_distribution_metadata(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "mlx_vlm", object())
    monkeypatch.setattr(
        mllm,
        "version",
        lambda _name: (_ for _ in ()).throw(mllm.PackageNotFoundError()),
    )

    assert mllm.vision_runtime_status() == (
        mllm.VisionRuntimeStatus.BROKEN,
        "mlx-vlm version metadata unavailable",
    )


def test_require_mlx_vlm_rejects_incompatible_runtime(monkeypatch):
    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.INCOMPATIBLE, "0.7.0"),
    )
    monkeypatch.setattr(mllm, "_vision_install_hint", lambda: "repair runtime")

    with pytest.raises(ImportError, match="installed mlx-vlm '0.7.0'"):
        mllm._require_mlx_vlm("publisher/vision-model")


def test_standalone_repair_commands_shell_quote_python_path(monkeypatch):
    monkeypatch.setattr(
        mllm.sys, "executable", "/Users/alice/My Runtime/bin/python's preview"
    )

    install_hint = mllm._vision_install_hint()
    dependency_hint = mllm._vlm_broken_install_hint("PIL")

    quoted = "'/Users/alice/My Runtime/bin/python'\"'\"'s preview'"
    assert f"{quoted} -m pip" in install_hint
    assert f"{quoted} -m pip" in dependency_hint
