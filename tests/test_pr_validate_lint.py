# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lint`` pr_validate step.

The contract that matters here — and the reason this file exists — is that
both ruff invocations pass ``--force-exclude``. ruff only applies the
``[tool.ruff].exclude`` / ``[tool.ruff.format].exclude`` config to files it
DISCOVERS itself; when handed explicit paths (which this step always does)
it ignores those excludes unless ``--force-exclude`` is set. Without the
flag, vendored files pyproject deliberately excludes (deepseek_v4.py,
gemma4_vendored/*, hy_v3.py, ...) get checked anyway and falsely fail the
step on any PR that merely touches them.
"""

from __future__ import annotations

import subprocess

import pytest

from scripts.pr_validate.context import Context
from scripts.pr_validate.steps.lint import LintStep


@pytest.fixture
def ctx_factory(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fake'\n")
    monkeypatch.chdir(tmp_path)

    def _make(files_changed: list[str]) -> Context:
        # Materialize the changed files so the step's exists() filter keeps
        # them (it drops paths that no longer exist in the working tree).
        for rel in files_changed:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x = 1\n")
        ctx = Context(pr_number=1)
        ctx.files_changed = files_changed
        ctx.work_dir = tmp_path / "work"
        return ctx

    return _make


def _capture_ruff_cmds(monkeypatch) -> list[list[str]]:
    """Record every argv the lint step hands to ruff, forcing a clean exit
    so both invocations run (the step short-circuits nothing)."""
    cmds: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("scripts.pr_validate.steps.lint.subprocess.run", fake_run)
    return cmds


class TestForceExclude:
    def test_both_ruff_invocations_pass_force_exclude(self, ctx_factory, monkeypatch):
        # The regression guard: a vendored file that pyproject excludes must
        # not be able to fail the step. That only holds if BOTH `ruff check`
        # and `ruff format --check` are given `--force-exclude` so ruff
        # honors the exclude config on the explicit paths.
        cmds = _capture_ruff_cmds(monkeypatch)
        ctx = ctx_factory(["vllm_mlx/models/deepseek_v4.py"])

        res = LintStep().run(ctx)
        assert res.status == "pass"

        check_cmd = next(c for c in cmds if c[:2] == ["ruff", "check"])
        format_cmd = next(c for c in cmds if c[:2] == ["ruff", "format"])

        assert "--force-exclude" in check_cmd
        assert "--force-exclude" in format_cmd
        # And the format invocation keeps its --check (report-only, no write).
        assert "--check" in format_cmd

    def test_force_exclude_precedes_the_file_paths(self, ctx_factory, monkeypatch):
        # ruff treats a value after a positional path as another path; the
        # flag must sit before the file list so it's parsed as an option.
        cmds = _capture_ruff_cmds(monkeypatch)
        ctx = ctx_factory(["vllm_mlx/models/hy_v3.py"])

        LintStep().run(ctx)

        for cmd in cmds:
            fx = cmd.index("--force-exclude")
            first_path = next(i for i, a in enumerate(cmd) if a.endswith(".py"))
            assert fx < first_path


class TestShouldRun:
    def test_runs_on_python_change(self, ctx_factory):
        ctx = ctx_factory(["vllm_mlx/models/hy_v3.py"])
        assert LintStep().should_run(ctx) is True

    def test_skips_docs_only(self, ctx_factory):
        ctx = ctx_factory(["README.md", "docs/guide.md"])
        assert LintStep().should_run(ctx) is False


class TestRegistration:
    def test_registered(self):
        from scripts.pr_validate.runner import STEPS

        assert "lint" in [s.name for s in STEPS]
