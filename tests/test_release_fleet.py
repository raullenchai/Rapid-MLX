#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the manifest-driven release model fleet."""

from __future__ import annotations

import importlib.util
import json
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


def test_reasoning_distill_participates_in_coherence_all_scopes(fleet):
    # DeepSeek-R1-Distill participates in the coherence sweep now that the gate
    # is reasoning-aware (#1323): it is served with thinking enabled and scored
    # on its concluded answer. It must appear in both release and toolchain
    # coherence scopes and be flagged `reasoning_distill`.
    assert "deepseek-r1-32b-4bit" in fleet.models_for_scope("release")
    assert "deepseek-r1-32b-4bit" in fleet.models_for_scope("toolchain")
    deepseek = next(f for f in fleet.load_fleet() if f.name == "deepseek")
    assert deepseek.reasoning_distill is True


def test_coherence_excluded_family_still_provides_artifact_matrix(fleet):
    # A reasoning-distill family stays a full release fleet member with its
    # scope-independent artifact-acceptance matrix cell intact.
    deepseek = next(f for f in fleet.load_fleet() if f.name == "deepseek")
    assert deepseek.artifact_matrix is not None
    assert deepseek.artifact_matrix.get("model") == "deepseek-r1-32b-4bit"


def test_release_scope_covers_each_architecture_risk_class(fleet):
    release_classes = {
        family.coverage_class
        for family in fleet.load_fleet()
        if "release" in family.scopes and family.coherence_enabled
    }
    assert release_classes >= fleet.REQUIRED_RELEASE_CLASSES


def test_reasoning_distill_flag_defaults_false_and_rejects_non_bool(fleet, tmp_path):
    # Absent flag -> not reasoning-distill; a non-bool value is rejected like
    # the other manifest field validators.
    base = {
        "coherence_model": "m",
        "coverage_class": "small_dense",
        "scopes": ["release", "toolchain"],
    }
    ok = tmp_path / "ok.json"
    ok.write_text(
        json.dumps(
            {
                "schema": 1,
                "families": {
                    "small_dense": base,
                    "hybrid_moe": {
                        **base,
                        "coherence_model": "m2",
                        "coverage_class": "hybrid_moe",
                    },
                    "large_dense": {
                        **base,
                        "coherence_model": "m3",
                        "coverage_class": "large_dense",
                    },
                    "large_moe": {
                        **base,
                        "coherence_model": "m4",
                        "coverage_class": "large_moe",
                    },
                    "multimodal": {
                        **base,
                        "coherence_model": "m5",
                        "coverage_class": "multimodal",
                    },
                },
            }
        )
    )
    families = fleet.load_fleet(ok)
    assert all(not family.reasoning_distill for family in families)

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "schema": 1,
                "families": {"small_dense": {**base, "reasoning_distill": "no"}},
            }
        )
    )
    with pytest.raises(ValueError, match="reasoning_distill must be a boolean"):
        fleet.load_fleet(bad)

    legacy = tmp_path / "legacy.json"
    legacy_data = json.loads(fleet.DEFAULT_MANIFEST.read_text())
    legacy_data["families"]["deepseek"]["coherence"] = False
    legacy.write_text(json.dumps(legacy_data))
    legacy_family = next(
        family for family in fleet.load_fleet(legacy) if family.name == "deepseek"
    )
    assert legacy_family.coherence_enabled is False
    assert "deepseek-r1-32b-4bit" not in fleet.models_for_scope("release", path=legacy)


def test_reasoning_distill_classifier_resolves_alias_and_hf_path(fleet):
    assert fleet.is_reasoning_distill_model("deepseek-r1-32b-4bit")
    assert fleet.is_reasoning_distill_model(
        "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit"
    )
    assert not fleet.is_reasoning_distill_model("qwen3.5-4b-4bit")


def test_force_text_lane_classifier_is_explicit_and_resolves_hf_path(fleet):
    gemma = next(f for f in fleet.load_fleet() if f.name == "gemma4")
    assert gemma.coherence_force_text_lane is True
    assert fleet.coherence_forces_text_lane("gemma-4-12b-4bit")
    assert fleet.coherence_forces_text_lane("mlx-community/gemma-4-12B-it-4bit")
    assert not fleet.coherence_forces_text_lane("qwen3.5-4b-4bit")


def test_force_text_lane_flag_defaults_false_and_rejects_non_bool(fleet, tmp_path):
    data = json.loads(fleet.DEFAULT_MANIFEST.read_text())
    del data["families"]["gemma4"]["coherence_force_text_lane"]
    defaulted = tmp_path / "defaulted.json"
    defaulted.write_text(json.dumps(data))
    assert all(not f.coherence_force_text_lane for f in fleet.load_fleet(defaulted))

    data["families"]["gemma4"]["coherence_force_text_lane"] = "yes"
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="coherence_force_text_lane must be a boolean"):
        fleet.load_fleet(invalid)


def test_reasoning_distill_classifier_reports_infrastructure_failure(
    fleet, monkeypatch, capsys
):
    monkeypatch.setattr(
        fleet,
        "is_reasoning_distill_model",
        lambda _model: (_ for _ in ()).throw(ValueError("broken manifest")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["release_fleet.py", "is-reasoning-distill", "deepseek-r1-32b-4bit"],
    )
    assert fleet.main() == 2
    assert "broken manifest" in capsys.readouterr().err


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


def test_coherence_sweep_avoids_empty_array_expansion_on_macos_bash():
    script = (REPO_ROOT / "scripts" / "coherence_sweep.sh").read_text()
    assert 'gate_command=("$PY" evals/coherence_gate.py)' in script
    assert 'gate_command=("$PY" evals/coherence_gate.py --reasoning-distill)' in script
    assert '"${GATE_ARGS[@]}"' not in script


def test_reasoning_distill_sweep_uses_supported_serve_flag():
    script = (REPO_ROOT / "scripts" / "coherence_sweep.sh").read_text()
    assert "SERVE_ARGS+=(--reasoning)" in script
    assert "SERVE_ARGS+=(--thinking)" not in script


def test_coherence_sweep_pins_text_only_lane():
    """G0a scores text and must not require the optional vision dependency."""
    script = (REPO_ROOT / "scripts" / "coherence_sweep.sh").read_text()
    assert 'scripts/release_fleet.py forces-text-lane "$MODEL"' in script
    assert "SERVE_ARGS+=(--no-mllm)" in script


def test_coherence_sweep_boot_wait_is_progress_aware_and_hard_bounded():
    script = (REPO_ROOT / "scripts" / "coherence_sweep.sh").read_text()
    assert 'BOOT_STALL_S="${COHERENCE_BOOT_STALL_S:-180}"' in script
    assert 'BOOT_HARD_S="${COHERENCE_BOOT_HARD_S:-1800}"' in script
    assert 'log_size=$(wc -c < "$LOG"' in script
    assert "SECONDS - last_progress" in script
    assert "SECONDS - boot_started" in script
    assert "for _ in $(seq 1 180)" not in script
