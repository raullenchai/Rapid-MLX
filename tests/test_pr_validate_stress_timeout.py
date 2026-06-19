# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-candidate ``stress_timeout_s`` override on the
``stress_e2e_bench`` pr_validate step.

Issue #664: the previous hardcoded ``timeout=900`` (15 min) for
``scripts/stress_test.py`` was sized for autoregressive models. Text-
diffusion models (e.g. ``mlx-community/diffusiongemma-26B-A4B-it-4bit``)
denoise all positions in parallel per step, so per-token wall-clock is
materially higher and the long-generation sub-test can blow the 15-min
budget. The fix adds an optional ``stress_timeout_s`` field on
``ModelChoice`` (loaded from ``golden_models.yaml``) that overrides the
default. This test locks in:

* ``_effective_stress_timeout`` returns the override when set.
* ``_effective_stress_timeout`` falls back to 900 when unset.
* ``_run_stress`` propagates that value to ``subprocess.run``'s
  ``timeout=`` kwarg (the actual regression site).
* The diffusiongemma entry in ``golden_models.yaml`` ships with
  ``stress_timeout_s: 1800`` (config-drift gate).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.pr_validate.steps import stress_e2e_bench
from scripts.pr_validate.steps.stress_e2e_bench import (
    DEFAULT_STRESS_TIMEOUT_S,
    ModelChoice,
    _effective_stress_timeout,
    _run_stress,
)


def _make_choice(stress_timeout_s: int | None = None) -> ModelChoice:
    return ModelChoice(
        family="diffusion-gemma",
        model_id="mlx-community/diffusiongemma-26B-A4B-it-4bit",
        ram_gb_required=20.0,
        quality_tier="smoke",
        extra_args=[],
        stress_timeout_s=stress_timeout_s,
    )


def test_effective_timeout_uses_override_when_set() -> None:
    """A candidate carrying ``stress_timeout_s`` (e.g. the diffusion
    family) gets its override, not the autoregressive default."""
    choice = _make_choice(stress_timeout_s=1800)
    assert _effective_stress_timeout(choice) == 1800


def test_effective_timeout_falls_back_to_default_when_unset() -> None:
    """Autoregressive candidates leave ``stress_timeout_s`` at ``None``
    and must keep the tight 900s default — bumping the default would
    weaken regression detection (see issue #664)."""
    choice = _make_choice(stress_timeout_s=None)
    assert _effective_stress_timeout(choice) == 900
    assert _effective_stress_timeout(choice) == DEFAULT_STRESS_TIMEOUT_S


def test_run_stress_passes_override_to_subprocess(tmp_path: Path) -> None:
    """The regression site: ``_run_stress`` must forward the resolved
    timeout to ``subprocess.run``. Mock subprocess and capture the
    kwargs."""
    choice = _make_choice(stress_timeout_s=1800)
    ctx = SimpleNamespace(
        artifact_path=lambda name: tmp_path / name,
        repo_root=tmp_path,
    )

    captured: dict = {}

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="3 passed", stderr="")

    with patch.object(stress_e2e_bench.subprocess, "run", side_effect=fake_run):
        result = _run_stress(ctx, choice)

    assert captured["timeout"] == 1800
    assert result["status"] == "pass"


def test_run_stress_falls_back_to_default_when_unset(tmp_path: Path) -> None:
    """Symmetric: unset ``stress_timeout_s`` → ``subprocess.run`` sees
    900, not None / 0 / a different default."""
    choice = _make_choice(stress_timeout_s=None)
    ctx = SimpleNamespace(
        artifact_path=lambda name: tmp_path / name,
        repo_root=tmp_path,
    )

    captured: dict = {}

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="3 passed", stderr="")

    with patch.object(stress_e2e_bench.subprocess, "run", side_effect=fake_run):
        _run_stress(ctx, choice)

    assert captured["timeout"] == 900


def test_golden_models_yaml_diffusiongemma_has_override() -> None:
    """Config-drift gate: the diffusiongemma candidate in
    ``golden_models.yaml`` MUST carry ``stress_timeout_s: 1800``. If
    someone reverts the YAML without reverting the dataclass plumbing,
    this test catches it before the next high-blast PR hangs for 15 min."""
    import yaml

    registry_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "pr_validate"
        / "golden_models.yaml"
    )
    registry = yaml.safe_load(registry_path.read_text())

    diffusion_candidate = None
    for family in registry.get("families", []):
        if family.get("family") == "diffusion-gemma":
            for cand in family.get("candidates", []):
                if cand.get("id") == "mlx-community/diffusiongemma-26B-A4B-it-4bit":
                    diffusion_candidate = cand
                    break
            break

    assert diffusion_candidate is not None, (
        "diffusion-gemma family / candidate missing from golden_models.yaml"
    )
    assert diffusion_candidate.get("stress_timeout_s") == 1800, (
        "diffusiongemma stress_timeout_s must be 1800s (text-diffusion "
        "denoise per step is ~2x autoregressive per-token wall-clock — "
        "see issue #664)"
    )


def test_select_models_picks_up_stress_timeout_from_yaml(tmp_path: Path) -> None:
    """The loader (``_select_models``) must thread ``stress_timeout_s``
    from the YAML dict into the ``ModelChoice``. Otherwise the YAML
    change above is silently dropped at the data boundary."""
    registry = {
        "families": [
            {
                "family": "diffusion-gemma",
                "candidates": [
                    {
                        "id": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
                        "ram_gb_required": 20,
                        "quality_tier": "smoke",
                        "stress_timeout_s": 1800,
                    }
                ],
            },
            {
                "family": "ar-model",
                "candidates": [
                    {
                        "id": "fake/ar-model",
                        "ram_gb_required": 10,
                        "quality_tier": "golden",
                        # no stress_timeout_s → must stay None and fall
                        # through to 900s at the timeout site.
                    }
                ],
            },
        ],
        "overrides": {},
    }
    choices = stress_e2e_bench._select_models(registry, usable_gb=100.0)
    by_family = {c.family: c for c in choices}

    assert by_family["diffusion-gemma"].stress_timeout_s == 1800
    assert by_family["ar-model"].stress_timeout_s is None
    assert _effective_stress_timeout(by_family["ar-model"]) == 900
