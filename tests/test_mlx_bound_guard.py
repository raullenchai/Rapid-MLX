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
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

# Load the script directly (scripts/ is not an importable package).
_SPEC = importlib.util.spec_from_file_location(
    "check_mlx_bound_move",
    Path(__file__).resolve().parent.parent / "scripts" / "check_mlx_bound_move.py",
)
guard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(guard)

_REPO_ROOT = Path(__file__).resolve().parent.parent


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


def test_desktop_sidecar_uses_validated_mlx_vlm_bound():
    """The signed Desktop and every fresh CLI install use one VLM runtime."""
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    vision_specs = [
        Requirement(spec)
        for spec in pyproject["project"]["optional-dependencies"]["vision"]
        if Requirement(spec).name == "mlx-vlm"
    ]
    assert len(vision_specs) == 1

    scripts = _REPO_ROOT / "apps" / "rapid-mac" / "scripts"
    sidecar = (scripts / "build-sidecar.sh").read_text()
    constraints = [
        Requirement(line)
        for line in (scripts / "sidecar-constraints.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    matches = [spec for spec in constraints if spec.name == "mlx-vlm"]
    assert len(matches) == 1, "expected exactly one mlx-vlm sidecar constraint"
    assert '--constraint "$SIDECAR_CONSTRAINTS"' in sidecar

    desktop_spec = matches[0]
    # Both surfaces deliberately pin one validated version. A range here caused
    # fresh pip installs to backtrack to 0.6.3 while Desktop stayed on 0.6.16.
    assert desktop_spec.specifier == Requirement("mlx-vlm==0.6.16").specifier
    assert vision_specs[0].specifier == desktop_spec.specifier


def test_image_extra_tracks_mlx_032_compatible_mflux_line():
    """The image extra must remain resolvable with the validated core runtime.

    mflux 0.18.x requires ``mlx<0.32``.  Once core moved to MLX 0.32.1, leaving
    the old mflux floor made ``pip install rapid-mlx[image]`` unsatisfiable.
    Lock both sides of that compatibility boundary into the package metadata.
    """
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    core_specs = [
        Requirement(spec)
        for spec in pyproject["project"]["dependencies"]
        if Requirement(spec).name == "mlx"
    ]
    image_specs = [
        Requirement(spec)
        for spec in pyproject["project"]["optional-dependencies"]["image"]
        if Requirement(spec).name == "mflux"
    ]

    assert len(core_specs) == 1
    assert len(image_specs) == 1
    assert Version("0.32.1") in core_specs[0].specifier
    assert Version("0.32.0") not in core_specs[0].specifier
    assert Version("0.33.0") not in core_specs[0].specifier
    assert Version("0.19.0") in image_specs[0].specifier
    assert Version("0.18.1") not in image_specs[0].specifier
    assert Version("0.20.0") not in image_specs[0].specifier


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


class TestMergifyAttestationWorkflow:
    """The queue may reuse only exact-head guard results from trusted batches."""

    @staticmethod
    def _job() -> dict:
        workflow = yaml.load(
            (_REPO_ROOT / ".github/workflows/ci.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        return workflow["jobs"]["mlx-bound-guard"]

    def test_candidate_resolver_is_limited_to_trusted_same_repo_mergify_prs(self):
        job = self._job()
        checkout = next(
            step
            for step in job["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        )
        setup = next(
            step
            for step in job["steps"]
            if step.get("name") == "Install Mergify CLI for trusted queue metadata"
        )
        queue_info = next(
            step for step in job["steps"] if step.get("id") == "queue-info"
        )
        resolver = next(
            step for step in job["steps"] if step.get("id") == "mergify-attestation"
        )

        condition = setup["if"]
        assert "github.event.pull_request.user.login == 'mergify[bot]'" in condition
        assert (
            "github.event.pull_request.head.repo.full_name == github.repository"
            in condition
        )
        assert "startsWith(github.head_ref, 'mergify/merge-queue/')" in condition
        assert checkout["with"]["fetch-depth"] == "0"
        assert checkout["with"]["ref"] == ("${{ github.event.pull_request.head.sha }}")
        assert setup["uses"].startswith("Mergifyio/setup-cli@")
        assert len(setup["uses"].split("@", 1)[1]) == 40
        assert setup["with"]["mergify_cli_version"] == "2026.8.28.1"
        assert queue_info["if"] == condition
        assert queue_info["run"] == "mergify ci queue-info"
        assert resolver["if"] == "steps.queue-info.outcome == 'success'"
        assert (
            resolver["env"]["QUEUE_METADATA"]
            == "${{ steps.queue-info.outputs.queue_metadata }}"
        )
        assert resolver["env"]["REAL_BASE_SHA"] == (
            "${{ github.event.pull_request.base.sha }}"
        )
        assert resolver["env"]["CANDIDATE_SHA"] == (
            "${{ github.event.pull_request.head.sha }}"
        )
        assert resolver["run"] == "python scripts/check_mergify_mlx_attestation.py"
        assert job["permissions"] == {
            "contents": "read",
            "pull-requests": "read",
        }

    def test_guard_uses_the_immutable_event_base_and_resolver_output(self):
        job = self._job()
        guard_step = next(
            step
            for step in job["steps"]
            if step.get("name") == "Guard mlx/mlx-lm/mlx-vlm version bounds"
        )
        assert (
            guard_step["env"]["MLX_BOUND_ATTESTED"]
            == "${{ steps.mergify-attestation.outputs.attested }}"
        )
        assert guard_step["env"]["MLX_BOUND_BASE_REF"] == (
            "${{ github.event.pull_request.base.sha }}"
        )


class TestDesktopManifestSynced:
    """The shipped Desktop third-party manifest must track the core runtime.

    ``apps/rapid-mac/THIRD_PARTY.md`` lists the bundled engine dependencies that
    a Mac build presents to users and license/security auditors. A dependency
    bound change in ``pyproject.toml`` (gated by this file's guard) must not
    leave that manifest stale: the installed Mac build would then bundle a
    different MLX than the manifest advertises.
    """

    @staticmethod
    def _third_party_mlx_table() -> dict[str, str]:
        """Return {component: declared range} for the Desktop manifest table."""
        text = (_REPO_ROOT / "apps" / "rapid-mac" / "THIRD_PARTY.md").read_text()
        out: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            name = cells[0]
            declared = cells[1]
            if name in ("mlx", "mlx-lm", "transformers"):
                out[name] = declared.strip("`")
        return out

    def test_desktop_mlx_ranges_match_core_manifest(self):
        pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
        core = {
            Requirement(spec).name: str(Requirement(spec).specifier)
            for spec in pyproject["project"]["dependencies"]
            if Requirement(spec).name in ("mlx", "mlx-lm", "transformers")
        }
        third = self._third_party_mlx_table()

        # The manifest must list the same guarded runtime packages as core.
        assert set(core) == set(third), f"manifest missing/extra: {third}"
        for pkg, ranges in core.items():
            # Backticks around the manifest cell are tolerated; compare the
            # version set semantically so formatting changes do not trip this.
            declared = third[pkg]
            assert SpecifierSet(declared) == SpecifierSet(ranges), (
                f"{pkg} manifest {declared!r} != core {ranges!r}"
            )

    def test_manifest_mlx_advertises_the_0321_floor(self):
        """Regression anchor: the shipped bundle claims MLX >=0.32.1,<0.33."""
        table = self._third_party_mlx_table()
        spec = SpecifierSet(table["mlx"])
        assert Version("0.31.2") not in spec
        assert Version("0.32.1") in spec
        assert Version("0.33.0") not in spec
