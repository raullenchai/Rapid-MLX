#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the non-inference helpers in release_artifact_matrix.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "release_artifact_matrix.py"


@pytest.fixture(scope="module")
def matrix():
    spec = importlib.util.spec_from_file_location("release_artifact_matrix", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_find_release_wheel_requires_exactly_one_candidate(matrix, tmp_path):
    wheel = tmp_path / "rapid_mlx-0.10.9-py3-none-any.whl"
    wheel.write_bytes(b"placeholder")
    assert matrix.find_release_wheel(tmp_path) == wheel.resolve()


def test_find_release_wheel_rejects_missing_candidate(matrix, tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        matrix.find_release_wheel(tmp_path)


def test_find_release_wheel_rejects_ambiguous_candidates(matrix, tmp_path):
    (tmp_path / "rapid_mlx-0.10.8-py3-none-any.whl").write_bytes(b"one")
    (tmp_path / "rapid_mlx-0.10.9-py3-none-any.whl").write_bytes(b"two")
    with pytest.raises(ValueError, match="0.10.8.*0.10.9"):
        matrix.find_release_wheel(tmp_path)


def test_clean_env_drops_source_injection_variables(matrix, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/unsafe/source")
    monkeypatch.setenv("PYTHONHOME", "/unsafe/home")
    monkeypatch.setenv("PIP_TARGET", "/unsafe/target")
    env = matrix._clean_env()
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert "PIP_TARGET" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["RAPID_MLX_DISABLE_VERSION_CHECK"] == "1"
    assert env["RAPID_MLX_TELEMETRY"] == "0"


def test_validate_families_json_allows_a_nonempty_diagnostic_subset(matrix):
    assert matrix.validate_families_json('["qwen36", "gptoss"]') == (
        "qwen36",
        "gptoss",
    )


@pytest.mark.parametrize(
    "value, message",
    [
        ("[]", "non-empty"),
        ('["qwen36", "qwen36"]', "duplicates"),
        ('["qwen36", "unknown"]', "unknown family"),
        ('["qwen36", 3]', "only family-name strings"),
        ('{"family": "qwen36"}', "JSON array"),
        ("not-json", "JSON array"),
    ],
)
def test_validate_families_json_rejects_invalid_selection(matrix, value, message):
    with pytest.raises(ValueError, match=message):
        matrix.validate_families_json(value)


def test_validate_families_json_requires_all_families_for_publication(matrix):
    all_families = list(matrix.FAMILY_CONFIGS)
    assert matrix.validate_families_json(
        str(all_families).replace("'", '"'), require_all_families=True
    ) == tuple(all_families)

    with pytest.raises(ValueError, match="publication requires every release family"):
        matrix.validate_families_json('["qwen36"]', require_all_families=True)


def test_family_configs_cover_the_release_eligible_families(matrix):
    assert set(matrix.FAMILY_CONFIGS) == {"qwen36", "gemma4", "deepseek", "gptoss"}
    assert matrix.FAMILY_CONFIGS["gemma4"].extras == ("vision",)


def test_cli_smoke_covers_base_commands_but_not_optional_chat(matrix):
    assert set(matrix.CLI_SMOKE_SCRIPTS) == {
        "rapid-mlx",
        "rapid-mlx-bench",
        "vllm-mlx",
        "vllm-mlx-bench",
    }


def test_parser_rejects_unknown_family(matrix):
    parser = matrix._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dist-dir", "dist", "--family", "unknown"])
