#!/usr/bin/env python3
"""Classify changed paths into stable CI lanes.

The workflow deliberately keeps this policy in tested Python instead of a
large, fragile GitHub Actions expression.  Unknown paths fail safe by selecting
both product lanes. Ordinary PR diffs with an engine lane use Python 3.12
across all three shards, preserving the coverage interpreter. Missing/invalid
diffs and dormant engine lanes keep the full matrix; promoted heads force full.
Non-PR events bypass this classifier in ci.yml and keep the full matrix.
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


LinuxMatrixMode = Literal["py312", "full"]


@dataclass(frozen=True)
class ValidationPolicy:
    """PR lane selection and interpreter breadth, with promoted-head override."""

    lanes: Lanes
    linux_matrix_mode: LinuxMatrixMode
    linux_matrix_reason: str

    def as_outputs(self) -> dict[str, str]:
        outputs = self.lanes.as_outputs()
        outputs.update(
            {
                "linux_matrix_mode": self.linux_matrix_mode,
                "linux_matrix_reason": self.linux_matrix_reason,
                "test_matrix": json.dumps(
                    linux_test_matrix(self.linux_matrix_mode), separators=(",", ":")
                ),
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
    "config/pr-validate-deps.txt",
    "config/requirements-ci-linux.txt",
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
    "RELEASE.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "LICENSE",
}


def linux_test_matrix(mode: LinuxMatrixMode) -> dict[str, list[dict[str, object]]]:
    """Return an explicit matrix so route changes cannot alter shard count."""
    versions = ("3.12",) if mode == "py312" else ("3.12", "3.13")
    return {
        "include": [
            {"python-version": version, "shard": shard}
            for version in versions
            for shard in (1, 2, 3)
        ]
    }


def _normalized_paths(paths: Iterable[str]) -> set[str]:
    return {path.strip().removeprefix("./") for path in paths if path.strip()}


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

    return "py312", "pr-single-python"


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


def classify_policy(
    paths: Iterable[str],
    *,
    force_full: bool = False,
    force_reason: str = "promoted-head",
) -> ValidationPolicy:
    """Select the ordinary PR matrix, or full breadth for promoted heads."""
    normalized = _normalized_paths(paths)
    lanes = classify(normalized)
    mode, reason = _linux_matrix_decision(normalized, lanes)
    if force_full:
        mode, reason = "full", force_reason
    return ValidationPolicy(lanes, mode, reason)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="Changed repository-relative paths")
    parser.add_argument("--paths-file", type=argparse.FileType("r"))
    parser.add_argument("--github-output", type=argparse.FileType("a"))
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--force-reason", default="promoted-head")
    args = parser.parse_args()

    paths = list(args.paths)
    if args.paths_file:
        paths.extend(args.paths_file.read().splitlines())
    outputs = classify_policy(
        paths,
        force_full=args.force_full,
        force_reason=args.force_reason,
    ).as_outputs()

    if args.github_output:
        for key, value in outputs.items():
            print(f"{key}={value}", file=args.github_output)
    else:
        print(json.dumps(outputs, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
