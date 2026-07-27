#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the manifest-driven release model fleet."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "release_fleet.py"


@pytest.fixture(scope="module")
def fleet():
    spec = importlib.util.spec_from_file_location("release_fleet", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_scope_covers_routinely_feasible_families(fleet):
    assert fleet.models_for_scope("release") == (
        "qwen3.5-4b-4bit",
        "qwen3.5-35b-4bit",
        "qwen3.6-27b-4bit",
        "gemma-4-12b-4bit",
        "deepseek-r1-32b-4bit",
        "gpt-oss-20b-mxfp4-q8",
    )


def test_toolchain_scope_adds_ultra_only_family(fleet):
    release_models = set(fleet.models_for_scope("release"))
    toolchain_models = set(fleet.models_for_scope("toolchain"))
    assert toolchain_models == release_models | {"hy3-preview-4bit"}


def test_release_scope_covers_each_architecture_risk_class(fleet):
    release_classes = {
        family.coverage_class
        for family in fleet.load_fleet()
        if "release" in family.scopes
    }
    assert release_classes >= fleet.REQUIRED_RELEASE_CLASSES


def test_toolchain_snapshot_detects_lock_only_version_bump(fleet):
    pyproject = '[project]\ndependencies = ["mlx>=0.31"]'
    old_lock = '[[package]]\nname = "mlx"\nversion = "0.31.0"'
    new_lock = '[[package]]\nname = "mlx"\nversion = "0.32.0"'
    assert fleet._toolchain_snapshot(pyproject, old_lock) != fleet._toolchain_snapshot(
        pyproject, new_lock
    )


def test_toolchain_snapshot_detects_same_version_artifact_change(fleet):
    pyproject = '[project]\ndependencies = ["mlx>=0.31"]'
    old_lock = (
        '[[package]]\nname = "mlx"\nversion = "0.31.0"\n'
        'wheels = [{ url = "mlx-0.31.whl", hash = "sha256:old" }]'
    )
    new_lock = (
        '[[package]]\nname = "mlx"\nversion = "0.31.0"\n'
        'wheels = [{ url = "mlx-0.31.whl", hash = "sha256:new" }]'
    )
    assert fleet._toolchain_snapshot(pyproject, old_lock) != fleet._toolchain_snapshot(
        pyproject, new_lock
    )


def test_toolchain_snapshot_detects_direct_requirement_bump(fleet):
    lock = '[[package]]\nname = "mlx-lm"\nversion = "0.31.3"'
    old_pyproject = '[project]\ndependencies = ["mlx-lm>=0.31.3"]'
    new_pyproject = '[project]\ndependencies = ["mlx-lm>=0.32.0"]'
    assert fleet._toolchain_snapshot(old_pyproject, lock) != fleet._toolchain_snapshot(
        new_pyproject, lock
    )


def test_toolchain_snapshot_ignores_metadata_mentions(fleet):
    lock = '[[package]]\nname = "rich"\nversion = "14.0"'
    old_pyproject = '[project]\nname = "rapid-mlx"\nkeywords = ["llm"]'
    new_pyproject = (
        '[project]\nname = "rapid-mlx"\n'
        'description = "Rapid-MLX inference"\nkeywords = ["llm", "mlx"]'
    )
    assert fleet._toolchain_snapshot(old_pyproject, lock) == fleet._toolchain_snapshot(
        new_pyproject, lock
    )


def test_explicit_scope_does_not_require_git_lookup(fleet):
    assert fleet.resolve_scope(requested="release", base_ref=None) == "release"
    assert fleet.resolve_scope(requested="toolchain", base_ref=None) == "toolchain"


def test_auto_scope_compares_with_tag_before_head(fleet, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["git", "describe"]:
            return subprocess.CompletedProcess(command, 0, "v0.11.0\n", "")
        return subprocess.CompletedProcess(command, 0, command[-1], "")

    monkeypatch.setattr(fleet.subprocess, "run", fake_run)
    monkeypatch.setattr(
        fleet,
        "_toolchain_snapshot",
        lambda pyproject, lock: {"pyproject": pyproject, "lock": lock},
    )

    assert fleet.resolve_scope(requested="auto", base_ref=None) == "toolchain"
    assert calls[0][-1] == "HEAD^"
    assert calls[0][3:5] == ["--match", "v[0-9]*"]
    assert calls[1][-1] == "v0.11.0:pyproject.toml"
    assert calls[2][-1] == "HEAD:pyproject.toml"
    assert calls[3][-1] == "v0.11.0:uv.lock"
    assert calls[4][-1] == "HEAD:uv.lock"


def test_auto_scope_handles_untracked_lock_deterministically(fleet, monkeypatch):
    def fake_run(command, **kwargs):
        if command[:2] == ["git", "show"] and command[-1].endswith(":uv.lock"):
            return subprocess.CompletedProcess(command, 128, "", "missing")
        return subprocess.CompletedProcess(
            command,
            0,
            (fleet.REPO_ROOT / "pyproject.toml").read_text(),
            "",
        )

    monkeypatch.setattr(fleet.subprocess, "run", fake_run)
    assert fleet.resolve_scope(requested="auto", base_ref="v0.11.0") == "release"


def test_auto_scope_keeps_normal_fleet_when_no_tag_is_available(fleet, monkeypatch):
    monkeypatch.setattr(
        fleet.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 128, "", ""),
    )
    assert fleet.resolve_scope(requested="auto", base_ref=None) == "release"


def test_release_gauntlet_invokes_manifest_driven_sweep():
    script = (REPO_ROOT / "scripts" / "release_check_m3.sh").read_text()
    assert "bash scripts/coherence_sweep.sh" in script
    assert 'FLEET_SCOPE="${FLEET_SCOPE:-auto}"' in script
