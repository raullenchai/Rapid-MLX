#!/usr/bin/env python3
"""Classify changed paths into stable CI lanes.

The workflow deliberately keeps this policy in tested Python instead of a
large, fragile GitHub Actions expression.  Unknown paths fail safe by selecting
both product lanes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal


@dataclass(frozen=True)
class Lanes:
    engine: bool
    desktop: bool
    docs_only: bool

    def as_outputs(self) -> dict[str, str]:
        return {
            "engine": str(self.engine).lower(),
            "desktop": str(self.desktop).lower(),
            "docs_only": str(self.docs_only).lower(),
        }


LinuxMatrixMode = Literal["py311", "full"]


@dataclass(frozen=True)
class ValidationPolicy:
    """Lane selection plus the Linux interpreter breadth for this diff."""

    lanes: Lanes
    linux_matrix_mode: LinuxMatrixMode
    linux_matrix_reason: str
    source_canary_tests: tuple[str, ...] = ()
    source_preflight: bool = False

    def as_outputs(self) -> dict[str, str]:
        outputs = self.lanes.as_outputs()
        outputs.update(
            {
                "linux_matrix_mode": self.linux_matrix_mode,
                "linux_matrix_reason": self.linux_matrix_reason,
                "test_matrix": json.dumps(
                    linux_test_matrix(self.linux_matrix_mode), separators=(",", ":")
                ),
                "source_preflight": str(self.source_preflight).lower(),
                "source_canary": str(bool(self.source_canary_tests)).lower(),
                "source_canary_tests": " ".join(self.source_canary_tests),
            }
        )
        return outputs


_ENGINE_ROOTS = {
    "rapid_mlx",
    "videox_fun_mlx",
    "tests",
    "scripts",
    "bench",
    "community-benchmarks",
    "evals",
    "examples",
    "harness",
    "reports",
}
_ENGINE_FILES = {
    "Makefile",
    "config/mypy-error-baseline.txt",
    "config/mypy-requirements.txt",
    "pyproject.toml",
    "uv.lock",
    "pytest.ini",
    "requirements.txt",
    "install.sh",
}
_DESKTOP_PREFIX = "apps/rapid-mac/"
_DESKTOP_SUPPORT_PREFIXES = ("tests/fixtures/ax_baseline/",)
_DESKTOP_SUPPORT = {
    "scripts/check_rapid_mac_ax_identifiers.py",
    "scripts/select_gui_flows.py",
    "tests/test_rapid_mac_ax_identifiers.py",
    "tests/test_rapid_mac_xcui_target.py",
    "tests/test_ax_baseline.py",
    "tests/test_ax_baseline_os_variance.py",
    "tests/test_gui_control_behavior_contract.py",
    "tests/test_gui_preflight_contract.py",
    "tests/test_gui_golden_ci_coverage.py",
    "tests/test_gui_flow_routing.py",
    "tests/test_gui_walk_completeness.py",
    "tests/test_fake_sidecar_image_catalog.py",
}
_DOC_ROOTS = {"docs"}
_DOC_FILES = {
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "LICENSE",
}

# Start with an exact allowlist of leaf metadata whose changes do not alter the
# shared CLI, API, engine, runtime, dependency, collection, authentication, or
# download control planes. A directory allowlist is deliberately avoided:
# telemetry/model_id.py, for example, participates in authenticated Hub access.
# Everything outside this set keeps the full supported-Python matrix. The merge
# candidate also forces the full matrix regardless of this result.
_PY311_ENGINE_PATHS = {
    "rapid_mlx/telemetry/events.json",
    "rapid_mlx/telemetry/registry.py",
}

# A filename heuristic cannot reliably tell whether a test is cheap or whether
# it protects a critical product surface (for example, "dflash", "toolchoice",
# and "wheel" are all critical without saying "kernel", "tool", or
# "packaging"). Keep this first slice exact and grow it only with reviewed path
# contracts.
_PY311_TEST_PATHS = {
    "tests/test_telemetry_registry.py",
    "tests/test_telemetry_registry_drift.py",
}

# Exact presentation/metadata contracts only. Shared CLI parsing, engine,
# authentication, model downloads and test support never enter this route.
_SOURCE_CANARY_AREAS = (
    (
        {"rapid_mlx/cli_help.py", "tests/test_cli_help_groups.py"},
        (
            "tests/test_cli_help_groups.py",
            "tests/test_cli_parser_snapshot.py",
            "tests/test_cli_parser_types.py",
        ),
    ),
    (
        {"rapid_mlx/_banner.py", "tests/test_cli_cheetah_banner.py"},
        (
            "tests/test_cli_cheetah_banner.py",
            "tests/test_cli_help_groups.py",
            "tests/test_cli_parser_snapshot.py",
        ),
    ),
    (
        {"rapid_mlx/telemetry/chip.py", "tests/test_telemetry_chip.py"},
        (
            "tests/test_telemetry_chip.py",
            "tests/test_chip_tier.py",
            "tests/test_telemetry_registry_drift.py",
        ),
    ),
    (
        _PY311_ENGINE_PATHS | _PY311_TEST_PATHS,
        ("tests/test_telemetry_registry.py", "tests/test_telemetry_registry_drift.py"),
    ),
    (
        # Test-only maintenance of these CPU/loopback contracts does not
        # change consent, transport, lifecycle or inference behavior. Run both
        # complete files; their production dependencies remain outside the map.
        {"tests/test_telemetry_track.py", "tests/test_telemetry_v1_retired.py"},
        ("tests/test_telemetry_track.py", "tests/test_telemetry_v1_retired.py"),
    ),
)


def source_canary_tests(paths: set[str]) -> tuple[str, ...]:
    allowed = set().union(*(area for area, _tests in _SOURCE_CANARY_AREAS))
    if not paths or not paths <= allowed:
        return ()
    return tuple(
        sorted(
            {
                test
                for area, tests in _SOURCE_CANARY_AREAS
                if paths & area
                for test in tests
            }
        )
    )


_FULL_MATRIX_REASONS = {
    ".github": "ci-control",
    "config": "dependency-or-policy",
    "scripts": "ci-or-build-control",
    "videox_fun_mlx": "shared-runtime",
}
_FULL_MATRIX_FILES = _ENGINE_FILES | {
    ".mergify.yml",
    ".coveragerc",
}


def linux_test_matrix(mode: LinuxMatrixMode) -> dict[str, list[dict[str, object]]]:
    """Return an explicit matrix so route changes cannot alter shard count."""
    versions = ("3.11",) if mode == "py311" else ("3.10", "3.11", "3.12")
    return {
        "include": [
            {"python-version": version, "shard": shard}
            for version in versions
            for shard in (1, 2, 3)
        ]
    }


def _normalized_paths(paths: Iterable[str]) -> set[str]:
    return {path.strip().removeprefix("./") for path in paths if path.strip()}


def _is_py311_test_path(path: str) -> bool:
    return path in _PY311_TEST_PATHS


def _linux_matrix_decision(
    paths: set[str], lanes: Lanes
) -> tuple[LinuxMatrixMode, str]:
    if not paths:
        return "full", "missing-diff"
    if not lanes.engine:
        # The engine matrix is skipped. Keeping the dormant value full makes a
        # future classifier/condition regression fail safe.
        return "full", "no-engine-lane"

    parsed_paths = [(path, PurePosixPath(path)) for path in sorted(paths)]
    if any(
        not pure.parts or pure.is_absolute() or ".." in pure.parts
        for _path, pure in parsed_paths
    ):
        return "full", "invalid-path"

    product_paths = [
        path
        for path, pure in parsed_paths
        if not (pure.parts[:1] == ("docs",) or path in _DOC_FILES)
    ]
    for path in product_paths:
        pure = PurePosixPath(path)
        if path in _FULL_MATRIX_FILES:
            return "full", "dependency-or-policy"
        root = pure.parts[0]
        if root in _FULL_MATRIX_REASONS:
            return "full", _FULL_MATRIX_REASONS[root]
        if _is_py311_test_path(path):
            continue
        if root == "tests":
            return "full", "test-support-or-fixture"
        if path in _PY311_ENGINE_PATHS:
            continue
        # Desktop and unknown paths are deliberately not neutral in a mixed
        # diff. Cross-lane and shared/core changes retain compatibility breadth.
        return "full", "shared-core-or-unmapped"
    return "py311", "leaf-engine-only"


def classify(paths: Iterable[str]) -> Lanes:
    normalized = _normalized_paths(paths)
    if not normalized:
        # A missing/invalid diff must never turn validation into a no-op.
        return Lanes(engine=True, desktop=True, docs_only=False)

    engine = False
    desktop = False
    docs_only = True

    for path in normalized:
        pure = PurePosixPath(path)
        root = pure.parts[0] if pure.parts else ""
        is_doc = root in _DOC_ROOTS or path in _DOC_FILES
        docs_only &= is_doc

        if (
            path.startswith(_DESKTOP_PREFIX)
            or path.startswith(_DESKTOP_SUPPORT_PREFIXES)
            or path in _DESKTOP_SUPPORT
        ):
            desktop = True
            continue

        if root == ".github":
            # Workflow/policy changes validate every lane they can affect.
            engine = True
            desktop = True
            continue

        if root in _ENGINE_ROOTS or path in _ENGINE_FILES:
            engine = True
            continue

        if not is_doc:
            # Fail closed for new top-level product areas.
            engine = True
            desktop = True

    return Lanes(engine=engine, desktop=desktop, docs_only=docs_only)


# Existing controller/collection regressions do not all use a test_ci_ name.
# Keep their families and the audited standalone guards on full source checks.
_SOURCE_PREFLIGHT_CONTROL_TEST_PREFIXES = (
    "test_ci_",
    "test_classify_ci_",
    "test_queue_",
    "test_check_",
    "test_mergify_",
    "test_release_",
    "test_pr_validate_",
    "test_probe_release_",
    "test_validate_release_",
    "test_github_action_",
)
_SOURCE_PREFLIGHT_CONTROL_TESTS = {
    "test_integration_collection_policy.py",
    "test_dev_test_script.py",
    "test_train_gates_matches_ci.py",
    "test_no_mlx_marker_contract.py",
    "test_mlx_bound_guard.py",
    "test_desktop_promotion.py",
    "test_community_benchmark_release_provenance.py",
}

# This generated golden has one ordinary CPU consumer which rebuilds and pins
# the complete argparse surface. Keep the exception exact: other fixtures can
# configure collection, runtime, integration, or platform-specific behavior.
_SOURCE_PREFLIGHT_FIXTURES = {"tests/fixtures/cli_parser_snapshot.json"}


def _source_preflight_paths(paths: set[str], lanes: Lanes) -> bool:
    """CPU source prefilter only; combined candidates still enforce every gate.

    Restrict the opt-in to engine code, ordinary CPU tests and the type debt file.
    Controllers, collection support, cross-product and unknown paths self-check
    in full. This is deliberately broader than a mapped regression contract.
    """
    if not paths or not lanes.engine or lanes.desktop:
        return False
    for path in paths:
        pure = PurePosixPath(path)
        if not pure.parts or pure.is_absolute() or ".." in pure.parts:
            return False
        # The existing type-check job still enforces the shrink-only budget.
        # Interpreter/toolchain pins and collection configuration stay full.
        if path == "config/mypy-error-baseline.txt":
            continue
        if path in _DOC_FILES or pure.parts[0] in _DOC_ROOTS:
            continue
        if path in _SOURCE_PREFLIGHT_FIXTURES:
            continue
        if len(pure.parts) < 2:
            return False
        if pure.parts[0] == "rapid_mlx":
            continue
        if pure.parts[0] == "tests" and (
            len(pure.parts) == 2
            or (len(pure.parts) == 3 and pure.parts[1] == "headless_mlx")
        ):
            name = pure.name
            if (
                name.startswith("test_")
                and name.endswith(".py")
                and not name.startswith(_SOURCE_PREFLIGHT_CONTROL_TEST_PREFIXES)
                and not name.endswith(("_workflow.py", "_workflows.py"))
                and name not in _SOURCE_PREFLIGHT_CONTROL_TESTS
            ):
                continue
        return False
    return True


def classify_policy(
    paths: Iterable[str],
    *,
    force_full: bool = False,
    force_reason: str = "promoted-head",
    source_canary: bool = False,
    source_preflight: bool = False,
) -> ValidationPolicy:
    normalized = _normalized_paths(paths)
    lanes = classify(normalized)
    mode, reason = _linux_matrix_decision(normalized, lanes)
    if force_full:
        mode, reason = "full", force_reason
    tests = source_canary_tests(normalized) if source_canary and not force_full else ()
    if source_canary and not force_full and not tests:
        mode, reason = "full", "source-canary-unmapped"
    preflight = (
        source_preflight
        and not force_full
        and not tests
        and _source_preflight_paths(normalized, lanes)
    )
    if preflight:
        mode, reason = "py311", "source-preflight"
    return ValidationPolicy(lanes, mode, reason, tests, preflight)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="Changed repository-relative paths")
    parser.add_argument("--paths-file", type=argparse.FileType("r"))
    parser.add_argument("--github-output", type=argparse.FileType("a"))
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--force-reason", default="promoted-head")
    parser.add_argument("--source-canary", action="store_true")
    parser.add_argument("--source-preflight", action="store_true")
    args = parser.parse_args()

    paths = list(args.paths)
    if args.paths_file:
        paths.extend(args.paths_file.read().splitlines())
    outputs = classify_policy(
        paths,
        force_full=args.force_full,
        force_reason=args.force_reason,
        source_canary=args.source_canary,
        source_preflight=args.source_preflight,
    ).as_outputs()

    if args.github_output:
        for key, value in outputs.items():
            print(f"{key}={value}", file=args.github_output)
    else:
        print(json.dumps(outputs, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
