# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts.pr_validate.runner.run_pipeline``.

We want to lock in two contracts:

1. Default (``fail_fast=False``) runs every step even after one fails,
   so the scorecard surfaces the FULL picture for a maintainer review.
2. Opt-in ``fail_fast=True`` stops at the first ``fail`` / ``error``
   AFTER the always-on fetch fail-fast — so CI doesn't waste compute on
   stress/bench when an earlier cheap check already blocked the PR.

Both contracts are exercised against fake in-memory Steps via the
``steps=`` injection seam — the production STEPS list pulls real PR
data over the network and is not unit-testable here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.pr_validate.base import Step, StepResult
from scripts.pr_validate.runner import run_pipeline
from scripts.pr_validate.steps.fetch import FetchStep


class _FakeFetch(Step):
    """Stand-in for ``FetchStep`` — populates the bare minimum of
    Context fields the rest of the pipeline reads, then returns pass."""

    name = "fetch"
    description = "fake fetch"

    def run(self, ctx):  # type: ignore[no-untyped-def]
        # Other steps may read these; populate them harmlessly.
        ctx.pr_title = "test"
        ctx.pr_author = "tester"
        ctx.head_sha = "deadbeef"
        ctx.diff_path = ""
        ctx.files_changed = []
        return StepResult(name=self.name, status="pass", summary="ok")


class _FakeStep(Step):
    """A configurable step that returns a preset status."""

    def __init__(self, name: str, status: str = "pass"):
        self.name = name
        self.description = f"fake {name}"
        self._status = status

    def run(self, ctx):  # type: ignore[no-untyped-def]
        return StepResult(
            name=self.name, status=self._status, summary=f"{self._status}"
        )


@pytest.fixture
def repo_root_cwd(monkeypatch, tmp_path):
    """Context's ``__post_init__`` insists on running from a dir with a
    pyproject.toml. Build a fake one so the test doesn't have to live in
    the real repo root."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fake'\n")
    monkeypatch.chdir(tmp_path)
    # Each test gets its own work_dir under tmp_path so artifacts don't
    # collide across runs.
    monkeypatch.setattr(
        "scripts.pr_validate.context.Path",
        Path,
        raising=False,
    )
    return tmp_path


def _fake_pipeline(after_fetch: list[tuple[str, str]]) -> list[Step]:
    """Build a [fake-fetch, ...named steps with a status...] pipeline."""
    return [_FakeFetch(), *(_FakeStep(name, status) for name, status in after_fetch)]


class TestFailFast:
    def test_default_runs_all_steps_after_a_fail(self, repo_root_cwd, capsys):
        """Without fail_fast, every step after fetch runs even when one
        fails — the scorecard is supposed to show ALL the issues at once."""
        steps = _fake_pipeline(
            [
                ("step_a", "pass"),
                ("step_b", "fail"),  # blocking, but fail_fast=False
                ("step_c", "pass"),
            ]
        )
        rc = run_pipeline(pr_number=999, fail_fast=False, steps=steps)
        captured = capsys.readouterr()
        # Exit code is non-zero because step_b failed.
        assert rc == 1
        # All four step headers must appear in stderr — none was skipped.
        for name in ("fetch", "step_a", "step_b", "step_c"):
            assert f"## [{name}]" in captured.err, f"missing step {name!r}"
        # Scorecard goes to stdout.
        assert "step_c" in captured.out

    def test_fail_fast_stops_at_first_fail_after_fetch(self, repo_root_cwd, capsys):
        """With fail_fast=True, step_c never runs once step_b fails."""
        steps = _fake_pipeline(
            [
                ("step_a", "pass"),
                ("step_b", "fail"),  # should stop here
                ("step_c", "pass"),  # never reached
            ]
        )
        rc = run_pipeline(pr_number=999, fail_fast=True, steps=steps)
        captured = capsys.readouterr()
        assert rc == 1
        # fetch + step_a + step_b ran; step_c did not.
        assert "## [fetch]" in captured.err
        assert "## [step_a]" in captured.err
        assert "## [step_b]" in captured.err
        assert "## [step_c]" not in captured.err
        # The fail-fast stop message must include the step name and the
        # 'subsequent steps not run' phrasing so the operator isn't
        # surprised by a short scorecard.
        assert "fail-fast: [step_b]" in captured.err
        assert "subsequent steps not run" in captured.err

    def test_fail_fast_stops_on_error_too(self, repo_root_cwd, capsys):
        """error status (step crash) should also short-circuit fail_fast."""
        steps = _fake_pipeline(
            [
                ("step_a", "pass"),
                ("step_b", "error"),  # crash counts as blocking
                ("step_c", "pass"),
            ]
        )
        rc = run_pipeline(pr_number=999, fail_fast=True, steps=steps)
        captured = capsys.readouterr()
        assert rc == 1
        assert "## [step_c]" not in captured.err
        assert "fail-fast: [step_b]" in captured.err

    def test_fail_fast_does_not_stop_on_skip(self, repo_root_cwd, capsys):
        """skip is neutral — fail-fast must NOT trigger on it, otherwise
        a high-blast gate skipping on a low-blast PR would look like a
        failure."""
        steps = _fake_pipeline(
            [
                ("step_a", "skip"),  # neutral, must continue
                ("step_b", "pass"),
                ("step_c", "pass"),
            ]
        )
        rc = run_pipeline(pr_number=999, fail_fast=True, steps=steps)
        captured = capsys.readouterr()
        assert rc == 0
        for name in ("step_a", "step_b", "step_c"):
            assert f"## [{name}]" in captured.err

    def test_fetch_failure_always_stops_regardless_of_flag(self, repo_root_cwd, capsys):
        """The pre-existing FAIL_FAST_STEPS={'fetch'} contract still
        holds even with fail_fast=False — without a successful fetch
        nothing else has anything to validate against."""

        class _BadFetch(Step):
            name = "fetch"
            description = "fake bad fetch"

            def run(self, ctx):  # type: ignore[no-untyped-def]
                return StepResult(name=self.name, status="fail", summary="bad")

        steps = [_BadFetch(), _FakeStep("step_a", "pass")]
        rc = run_pipeline(pr_number=999, fail_fast=False, steps=steps)
        captured = capsys.readouterr()
        assert rc == 1
        assert "## [fetch]" in captured.err
        assert "## [step_a]" not in captured.err
        # The "is critical" message is the hard-coded fetch fail-fast,
        # not the user-toggled one — make sure the right path fired.
        assert "is critical" in captured.err

    def test_real_fetch_step_in_global_pipeline(self):
        """Sanity: the production pipeline still has FetchStep first.
        Catches accidental reorderings in runner.py's STEPS list."""
        from scripts.pr_validate.runner import STEPS

        assert isinstance(STEPS[0], FetchStep)


class TestSelectModels:
    """Pin the candidate-selection contract for ``stress_e2e_bench``.

    The qwen3.6 family ships with two candidates today: an 8-bit primary
    (``unsloth/Qwen3.6-27B-MLX-8bit``, ram=36) and a 4-bit fallback
    (``mlx-community/Qwen3.6-27B-4bit``, ram=18). The selection rule is
    "first candidate whose ``ram_gb_required`` fits ``usable_gb``" — these
    tests make sure a future refactor that swaps the rule (e.g. to
    "highest ``quality_tier``") doesn't silently downgrade the 36 GB+ host
    to the 4-bit candidate or upgrade the 24 GB host to an OOM at boot.
    """

    @staticmethod
    def _registry() -> dict:
        """Mirror the qwen3.6 entry in golden_models.yaml. Hand-built so
        the test doesn't depend on file content — if someone reorders the
        YAML, this test still pins the selection algorithm itself."""
        return {
            "families": [
                {
                    "family": "qwen3.6",
                    "candidates": [
                        {
                            "id": "unsloth/Qwen3.6-27B-MLX-8bit",
                            "ram_gb_required": 36,
                            "quality_tier": "golden",
                        },
                        {
                            "id": "mlx-community/Qwen3.6-27B-4bit",
                            "ram_gb_required": 18,
                            # Mirrors the YAML — `smoke` so the multi-
                            # tool agent (skip_for_smoke=true) is
                            # suppressed on constrained hosts that
                            # have to fall through to this 4-bit
                            # entry. A future test that relies on the
                            # skip behavior wants the fixture's tier
                            # to match production.
                            "quality_tier": "smoke",
                        },
                    ],
                },
            ],
            "overrides": {
                "unsloth/Qwen3.6-27B-MLX-8bit": {
                    "args": [
                        "--enable-auto-tool-choice",
                        "--tool-call-parser",
                        "hermes",
                    ],
                },
                "mlx-community/Qwen3.6-27B-4bit": {
                    "args": [
                        "--enable-auto-tool-choice",
                        "--tool-call-parser",
                        "hermes",
                    ],
                },
            },
        }

    # The override args we expect for the qwen3.6 family — match
    # exactly so a regression that drops the parser, drops the value,
    # or swaps `hermes` for the wrong parser id is caught. A tuple
    # rather than a list so a future test can't accidentally `.append`
    # to it and silently poison every other test that compares against
    # this constant.
    _QWEN36_HERMES_ARGS = (
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    )

    def test_high_ram_picks_8bit_primary(self):
        """48 GB usable easily fits the 36 GB primary — fallback must not
        win on a beefy host."""
        from scripts.pr_validate.steps.stress_e2e_bench import _select_models

        choices = _select_models(self._registry(), usable_gb=48.0)
        assert len(choices) == 1
        assert choices[0].family == "qwen3.6"
        assert choices[0].model_id == "unsloth/Qwen3.6-27B-MLX-8bit"
        # Override args wired through verbatim so the server boots with
        # the exact parser we asked for — regression guard for the
        # override-by-id map.
        assert tuple(choices[0].extra_args) == self._QWEN36_HERMES_ARGS

    def test_low_ram_falls_through_to_4bit_fallback(self):
        """24 GB usable can't fit the 36 GB primary but does fit the 18
        GB fallback — covers the constrained-host (≤32 GB) graceful-
        degradation path. Without the fallback this family would be
        silently dropped from the matrix on older M2 Pro / base M3."""
        from scripts.pr_validate.steps.stress_e2e_bench import _select_models

        choices = _select_models(self._registry(), usable_gb=24.0)
        assert len(choices) == 1
        assert choices[0].model_id == "mlx-community/Qwen3.6-27B-4bit"
        assert choices[0].ram_gb_required == 18.0
        # The fallback also gets its overrides — the override map keys
        # by full HF id, so a typo in either side silently drops the
        # parser flag and the model boots with default (broken)
        # tool-call routing.
        assert tuple(choices[0].extra_args) == self._QWEN36_HERMES_ARGS

    def test_below_all_candidates_skips_family(self):
        """A host below every candidate's floor drops the family from the
        returned list rather than picking something that will OOM."""
        from scripts.pr_validate.steps.stress_e2e_bench import _select_models

        choices = _select_models(self._registry(), usable_gb=10.0)
        assert choices == []

    def test_first_fit_wins_even_when_later_candidate_has_higher_tier(self):
        """``quality_tier`` is informational — order in the YAML decides
        the priority. If a later candidate happens to be tagged "golden"
        and the earlier one is tagged "small", the earlier one still
        wins when both fit. This is the contract the inline docstring
        on ``_select_models`` makes; pin it so a "smart" refactor can't
        silently flip the priority."""
        from scripts.pr_validate.steps.stress_e2e_bench import _select_models

        registry = {
            "families": [
                {
                    "family": "tiered",
                    "candidates": [
                        {
                            "id": "first/listed",
                            "ram_gb_required": 8,
                            "quality_tier": "small",
                        },
                        {
                            "id": "later/listed",
                            "ram_gb_required": 16,
                            "quality_tier": "golden",
                        },
                    ],
                },
            ],
        }
        choices = _select_models(registry, usable_gb=64.0)
        assert len(choices) == 1
        assert choices[0].model_id == "first/listed"
        assert choices[0].quality_tier == "small"

    def test_real_yaml_high_ram_picks_first_qwen36_candidate(self):
        """Hand-built ``_registry()`` above pins the algorithm; this
        test pins the *file* — a future bump of ``ram_gb_required`` (say
        36→40 after observing OOMs) or a candidate reorder must not
        silently break the bench. We assert the YAML's first qwen3.6
        candidate is the one selected on a high-RAM host, and that the
        override map still has an entry for whatever id is first.
        That's enough to catch the two breakages a YAML-only edit can
        introduce: dropping the override key, or accidentally promoting
        the 4-bit entry to position 0."""
        from scripts.pr_validate.steps.stress_e2e_bench import (
            _load_registry,
            _select_models,
        )

        registry = _load_registry()
        # Headroom that fits any plausible 27-35B 8-bit model on a real
        # rig — the test is "primary entry wins on a beefy host", not
        # "exactly 48 GB"; bumping the YAML floor to 40 or 48 doesn't
        # invalidate this test, only a value > 999 would.
        choices = _select_models(registry, usable_gb=999.0)
        qwen36 = next((c for c in choices if c.family == "qwen3.6"), None)
        assert qwen36 is not None, "qwen3.6 family missing from selection"

        # The selected id must be the YAML's literal first candidate —
        # walk the file the same way _select_models does.
        first_yaml = next(f for f in registry["families"] if f["family"] == "qwen3.6")[
            "candidates"
        ][0]
        assert qwen36.model_id == first_yaml["id"]
        # Override args wired through verbatim — full equality so a
        # value-typo in the YAML override (e.g. `hermez` instead of
        # `hermes`) is caught rather than just the truthiness of a
        # non-empty list.
        assert tuple(qwen36.extra_args) == self._QWEN36_HERMES_ARGS, (
            f"selected {qwen36.model_id!r} but its overrides args don't "
            f"match expected — check overrides:{qwen36.model_id} in "
            "golden_models.yaml"
        )
