# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the mlx/mlx-lm/mlx-vlm version-bound guard (#1248).

Pure-logic — no network, no GPU, no git. Covers dependency extraction (across
every pyproject source), the fail-closed change detector, strict-mode failure,
and the attestation check.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load the script directly (scripts/ is not an importable package).
_SPEC = importlib.util.spec_from_file_location(
    "check_mlx_bound_move",
    Path(__file__).resolve().parent.parent / "scripts" / "check_mlx_bound_move.py",
)
guard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(guard)


_CORE = """\
[project]
name = "rapid-mlx"
# The 'mlx' keyword must NOT be mistaken for a dependency (regression guard).
keywords = ["llm", "mlx", "apple-silicon"]
dependencies = [
    "numpy>=1.0",
    "mlx>=0.31.2,<0.32",
    "mlx-lm>=0.31.3,<0.32",
    "transformers>=5.0.0,<5.13",
]

[project.optional-dependencies]
vision = [
    "mlx-vlm>=0.6.3,!=0.6.4,<0.7",
    "mlx-vlm>=0.6.3,!=0.6.4,<0.7; platform_system == 'Darwin'",
]
"""


def _pp(*deps: str) -> str:
    """Wrap dependency strings in a minimal valid pyproject for extraction tests."""
    body = ",\n    ".join(f'"{d}"' for d in deps)
    return f'[project]\nname = "x"\ndependencies = [\n    {body},\n]\n'


class TestExtractMlxBounds:
    def test_extracts_each_guarded_package_distinctly(self):
        from packaging.specifiers import SpecifierSet

        bounds = guard.extract_mlx_bounds(_CORE)
        # Compare semantically — extract_mlx_bounds stores the normalized
        # ``str(SpecifierSet)`` (order-independent), not the source order.
        assert {SpecifierSet(s) for s in bounds["mlx"]} == {
            SpecifierSet(">=0.31.2,<0.32")
        }
        assert {SpecifierSet(s) for s in bounds["mlx-lm"]} == {
            SpecifierSet(">=0.31.3,<0.32")
        }
        # Both mlx-vlm entries share the same specifier (marker ignored) -> one.
        assert {SpecifierSet(s) for s in bounds["mlx-vlm"]} == {
            SpecifierSet(">=0.6.3,!=0.6.4,<0.7")
        }

    def test_ignores_non_mlx_and_prefix_lookalikes(self):
        text = _pp("mlx-audio>=0.2.9,<0.4.4", "transformers<5.13", "mlxfoo==1.0")
        bounds = guard.extract_mlx_bounds(text)
        assert bounds["mlx"] == set()
        assert bounds["mlx-lm"] == set()
        assert bounds["mlx-vlm"] == set()

    def test_keyword_mlx_is_not_a_dependency(self):
        # The ``keywords = ["mlx", ...]`` entry in _CORE must be ignored.
        assert "" not in guard.extract_mlx_bounds(_CORE)["mlx"]

    def test_mlx_does_not_swallow_mlx_lm_or_mlx_vlm(self):
        text = _pp("mlx>=0.31.2", "mlx-lm>=0.31.3", "mlx-vlm>=0.6.3")
        bounds = guard.extract_mlx_bounds(text)
        assert bounds["mlx"] == {">=0.31.2"}
        assert bounds["mlx-lm"] == {">=0.31.3"}
        assert bounds["mlx-vlm"] == {">=0.6.3"}

    def test_pep503_name_aliases_are_canonicalized(self):
        # mlx_vlm / MLX-VLM / mlx.vlm are all valid spellings of mlx-vlm and must
        # NOT slip past the guard (canonicalize_name).
        for alias in ("mlx_vlm>=0.7", "MLX-VLM>=0.7", "mlx.vlm>=0.7"):
            bounds = guard.extract_mlx_bounds(_pp(alias))
            assert bounds["mlx-vlm"] == {">=0.7"}, alias

    def test_scans_dependency_groups_and_build_system(self):
        from packaging.specifiers import SpecifierSet

        text = (
            '[build-system]\nrequires = ["mlx>=0.31.2,<0.32"]\n\n'
            '[dependency-groups]\ndev = ["mlx-lm>=0.31.3,<0.32"]\n'
        )
        bounds = guard.extract_mlx_bounds(text)
        assert {SpecifierSet(s) for s in bounds["mlx"]} == {
            SpecifierSet(">=0.31.2,<0.32")
        }
        assert {SpecifierSet(s) for s in bounds["mlx-lm"]} == {
            SpecifierSet(">=0.31.3,<0.32")
        }


class TestStrictMode:
    def test_malformed_guarded_requirement_raises_in_strict(self):
        text = _pp("mlx-lm>=0.31.3", "mlx >>>= broken")
        with pytest.raises(guard.MalformedGuardedRequirementError):
            guard.extract_mlx_bounds(text, strict=True)

    def test_malformed_guarded_requirement_skipped_when_not_strict(self):
        text = _pp("mlx-lm>=0.31.3", "mlx >>>= broken")
        bounds = guard.extract_mlx_bounds(text)  # non-strict
        assert bounds["mlx-lm"] == {">=0.31.3"}

    def test_malformed_non_guarded_requirement_never_raises(self):
        # A broken NON-mlx requirement is not our gate's concern.
        text = _pp("mlx-lm>=0.31.3", "some-pkg >>>= broken")
        assert guard.extract_mlx_bounds(text, strict=True)["mlx-lm"] == {">=0.31.3"}

    def test_malformed_guarded_alias_raises_in_strict(self):
        # A malformed alias spelling of a guarded package must still fail closed.
        text = _pp("mlx_vlm >>>= broken")
        with pytest.raises(guard.MalformedGuardedRequirementError):
            guard.extract_mlx_bounds(text, strict=True)

    def test_malformed_toml_raises_in_strict(self):
        with pytest.raises(Exception):
            guard.extract_mlx_bounds("this is [ not valid = toml", strict=True)

    def test_malformed_toml_yields_empty_when_not_strict(self):
        bounds = guard.extract_mlx_bounds("this is [ not valid = toml")
        assert bounds == {p: set() for p in guard.GUARDED_PACKAGES}


class TestDetectBoundChanges:
    def test_identical_is_no_change(self):
        assert guard.detect_bound_changes(_CORE, _CORE) == []

    def test_reordered_specifier_is_not_a_change(self):
        reordered = _CORE.replace(
            '"mlx-vlm>=0.6.3,!=0.6.4,<0.7"', '"mlx-vlm<0.7,>=0.6.3,!=0.6.4"'
        )
        assert guard.detect_bound_changes(_CORE, reordered) == []

    def test_marker_only_change_is_not_a_change(self):
        # Only the version specifier is coherence-relevant; markers are ignored.
        toggled = _CORE.replace(
            "\"mlx-vlm>=0.6.3,!=0.6.4,<0.7; platform_system == 'Darwin'\"",
            '"mlx-vlm>=0.6.3,!=0.6.4,<0.7"',
        )
        assert guard.detect_bound_changes(_CORE, toggled) == []

    # --- every real specifier change is flagged (fail-closed) ---

    def test_raising_upper_bound_is_flagged(self):
        bumped = _CORE.replace('"mlx-lm>=0.31.3,<0.32"', '"mlx-lm>=0.31.3,<0.33"')
        changes = guard.detect_bound_changes(_CORE, bumped)
        assert len(changes) == 1 and changes[0].startswith("mlx-lm:")

    def test_removing_cap_entirely_is_flagged(self):
        uncapped = _CORE.replace('"mlx>=0.31.2,<0.32"', '"mlx>=0.31.2"')
        assert any(
            c.startswith("mlx:") for c in guard.detect_bound_changes(_CORE, uncapped)
        )

    def test_pure_floor_bump_is_flagged(self):
        # BLOCKING-class case: >=0.31.2 -> >=0.32 forces an untested minor.
        old = _CORE.replace('"mlx>=0.31.2,<0.32"', '"mlx>=0.31.2"')
        new = _CORE.replace('"mlx>=0.31.2,<0.32"', '"mlx>=0.32"')
        assert any(c.startswith("mlx:") for c in guard.detect_bound_changes(old, new))

    def test_dropping_ne_exclusion_is_flagged(self):
        deexcluded = _CORE.replace(
            '"mlx-vlm>=0.6.3,!=0.6.4,<0.7"', '"mlx-vlm>=0.6.3,<0.7"'
        )
        assert any(
            c.startswith("mlx-vlm:")
            for c in guard.detect_bound_changes(_CORE, deexcluded)
        )

    def test_adding_a_cap_is_flagged(self):
        # Fail-closed: even tightening is gated (its attestation is a one-liner).
        uncapped = _CORE.replace('"mlx>=0.31.2,<0.32"', '"mlx>=0.31.2"')
        assert any(
            c.startswith("mlx:") for c in guard.detect_bound_changes(uncapped, _CORE)
        )

    def test_skewing_one_of_multiple_vlm_occurrences_is_flagged(self):
        skewed = _CORE.replace(
            "\"mlx-vlm>=0.6.3,!=0.6.4,<0.7; platform_system == 'Darwin'\"",
            "\"mlx-vlm>=0.6.3,!=0.6.4,<0.8; platform_system == 'Darwin'\"",
        )
        assert any(
            c.startswith("mlx-vlm:") for c in guard.detect_bound_changes(_CORE, skewed)
        )


class TestAttestation:
    def test_label_grants_attestation(self):
        assert guard._attestation_ok("", "some-label, mlx-coherence-swept", False)

    def test_trailer_with_note_grants_attestation(self):
        body = "Fixes stuff.\n\nCoherence-Sweep: https://ci/run/123 all families green"
        assert guard._attestation_ok(body, "", False)

    def test_empty_trailer_does_not_grant(self):
        assert not guard._attestation_ok("Coherence-Sweep:   ", "", False)

    def test_no_attestation_is_false(self):
        assert not guard._attestation_ok("just a normal body", "bug,p0", False)

    def test_forced_env_grants(self):
        assert guard._attestation_ok("", "", True)

    def test_label_match_is_case_insensitive_and_trimmed(self):
        assert guard._attestation_ok("", "  MLX-Coherence-Swept  ", False)
