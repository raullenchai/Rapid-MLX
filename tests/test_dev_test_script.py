# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the local developer test wrapper."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


def test_unit_suite_timeout_covers_current_three_minute_runtime(monkeypatch):
    script = Path(__file__).parents[1] / "scripts" / "dev_test.py"
    spec = importlib.util.spec_from_file_location("rapid_mlx_dev_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    observed: dict[str, object] = {}

    def fake_run(cmd, label, timeout=600):
        observed.update(cmd=cmd, label=label, timeout=timeout)
        return True

    monkeypatch.setattr(module, "run", fake_run)

    assert module.run_unit() is True
    assert observed["timeout"] == 300


def test_lint_checks_and_formats_the_whole_repository(monkeypatch):
    script = Path(__file__).parents[1] / "scripts" / "dev_test.py"
    spec = importlib.util.spec_from_file_location("rapid_mlx_dev_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    commands: list[list[str]] = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0),
    )

    def fake_run(cmd, label, timeout=600):
        commands.append(cmd)
        return True

    monkeypatch.setattr(module, "run", fake_run)
    assert module.run_lint() is True
    assert commands == [
        [module.PY, "-m", "ruff", "check", "."],
        [module.PY, "-m", "ruff", "format", "--check", "."],
    ]


def test_github_ci_uses_the_same_whole_repository_ruff_scope():
    workflow = (Path(__file__).parents[1] / ".github/workflows/ci.yml").read_text()
    assert "run: ruff check ." in workflow
    assert "run: ruff format --check ." in workflow
    assert "ruff check vllm_mlx/ tests/" not in workflow


def test_repository_format_gate_excludes_markdown_examples():
    config = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    format_config = config.split("[tool.ruff.format]", 1)[1].split(
        "[tool.ruff.lint]", 1
    )[0]
    assert '"*.md"' in format_config
