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


_ENGINE_ROOTS = {
    "vllm_mlx",
    "tests",
    "scripts",
    "benchmarks",
    "examples",
}
_ENGINE_FILES = {
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


def classify(paths: Iterable[str]) -> Lanes:
    normalized = {path.strip().removeprefix("./") for path in paths if path.strip()}
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="Changed repository-relative paths")
    parser.add_argument("--paths-file", type=argparse.FileType("r"))
    parser.add_argument("--github-output", type=argparse.FileType("a"))
    args = parser.parse_args()

    paths = list(args.paths)
    if args.paths_file:
        paths.extend(args.paths_file.read().splitlines())
    outputs = classify(paths).as_outputs()

    if args.github_output:
        for key, value in outputs.items():
            print(f"{key}={value}", file=args.github_output)
    else:
        print(json.dumps(outputs, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
