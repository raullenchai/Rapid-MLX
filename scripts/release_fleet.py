#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read and validate the release-fleet manifest.

The manifest is the source of truth for release-time model-family coverage.
Normal releases sweep one real representative for every routinely feasible
family. Changes to the MLX toolchain expand the sweep to the Ultra-only Hy3
family as well.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = Path(__file__).with_name("release_fleet.json")
VALID_SCOPES = frozenset({"release", "toolchain"})
TOOLCHAIN_PACKAGES = ("mlx", "mlx-lm", "mlx-vlm", "mlx-audio")
REQUIRED_RELEASE_CLASSES = frozenset(
    {"small_dense", "hybrid_moe", "large_dense", "large_moe", "multimodal"}
)


@dataclass(frozen=True)
class FleetFamily:
    name: str
    coherence_model: str
    coverage_class: str
    scopes: tuple[str, ...]
    artifact_matrix: dict[str, Any] | None


def load_fleet(path: Path = DEFAULT_MANIFEST) -> tuple[FleetFamily, ...]:
    """Load the manifest and reject coverage-silencing mistakes."""

    data = json.loads(path.read_text())
    if data.get("schema") != 1:
        raise ValueError("release fleet manifest must use schema 1")
    raw_families = data.get("families")
    if not isinstance(raw_families, dict) or not raw_families:
        raise ValueError("release fleet manifest needs a non-empty families object")

    families: list[FleetFamily] = []
    coherence_models: set[str] = set()
    for name, raw in raw_families.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise ValueError("every release fleet family needs a non-empty name/object")
        model = raw.get("coherence_model")
        coverage_class = raw.get("coverage_class")
        scopes = raw.get("scopes")
        artifact = raw.get("artifact_matrix")
        if (
            not isinstance(model, str)
            or not model
            or any(character.isspace() for character in model)
        ):
            raise ValueError(f"{name}: coherence_model must be a non-empty string")
        if model in coherence_models:
            raise ValueError(f"{name}: duplicate coherence_model {model!r}")
        coherence_models.add(model)
        if not isinstance(coverage_class, str) or not coverage_class:
            raise ValueError(f"{name}: coverage_class must be a non-empty string")
        if (
            not isinstance(scopes, list)
            or not scopes
            or not all(isinstance(scope, str) for scope in scopes)
        ):
            raise ValueError(f"{name}: scopes must be a non-empty string list")
        unknown_scopes = set(scopes) - VALID_SCOPES
        if unknown_scopes:
            raise ValueError(f"{name}: unknown scopes {sorted(unknown_scopes)!r}")
        if "toolchain" not in scopes:
            raise ValueError(
                f"{name}: every family must participate in toolchain scope"
            )
        if artifact is not None and not isinstance(artifact, dict):
            raise ValueError(f"{name}: artifact_matrix must be an object")
        families.append(
            FleetFamily(
                name=name,
                coherence_model=model,
                coverage_class=coverage_class,
                scopes=tuple(scopes),
                artifact_matrix=artifact,
            )
        )
    release_classes = {
        family.coverage_class for family in families if "release" in family.scopes
    }
    missing_classes = REQUIRED_RELEASE_CLASSES - release_classes
    if missing_classes:
        raise ValueError(
            f"release fleet is missing coverage classes {sorted(missing_classes)!r}"
        )
    return tuple(families)


def models_for_scope(scope: str, *, path: Path = DEFAULT_MANIFEST) -> tuple[str, ...]:
    if scope not in VALID_SCOPES:
        raise ValueError(f"unknown release fleet scope {scope!r}")
    return tuple(
        family.coherence_model for family in load_fleet(path) if scope in family.scopes
    )


def _canonical_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _toolchain_snapshot(
    pyproject_text: str, lock_text: str
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Return direct requirements and complete lock records for MLX packages."""

    pyproject = tomllib.loads(pyproject_text)
    project = pyproject.get("project", {})
    requirement_groups = [project.get("dependencies", [])]
    requirement_groups.extend(project.get("optional-dependencies", {}).values())

    requirements: dict[str, set[str]] = {
        package: set() for package in TOOLCHAIN_PACKAGES
    }
    for group in requirement_groups:
        for requirement in group:
            if not isinstance(requirement, str):
                continue
            match = re.match(r"([a-zA-Z0-9][a-zA-Z0-9._-]*)", requirement)
            if not match:
                continue
            package = _canonical_package_name(match.group(1))
            if package in requirements:
                requirements[package].add(requirement)

    locked: dict[str, set[str]] = {package: set() for package in TOOLCHAIN_PACKAGES}
    for entry in tomllib.loads(lock_text).get("package", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            continue
        package = _canonical_package_name(entry["name"])
        if package not in locked:
            continue
        locked[package].add(json.dumps(entry, sort_keys=True, separators=(",", ":")))

    return {
        package: (tuple(sorted(requirements[package])), tuple(sorted(locked[package])))
        for package in TOOLCHAIN_PACKAGES
    }


def resolve_scope(*, requested: str, base_ref: str | None) -> str:
    if requested != "auto":
        if requested not in VALID_SCOPES:
            raise ValueError(f"unknown release fleet scope {requested!r}")
        return requested

    if base_ref is None:
        describe = subprocess.run(
            [
                "git",
                "describe",
                "--tags",
                "--match",
                "v[0-9]*",
                "--abbrev=0",
                "HEAD^",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if describe.returncode != 0 or not describe.stdout.strip():
            # A shallow/source-archive checkout has no reliable comparison
            # point. Keep the normal fleet instead of silently adding an
            # Ultra-only model solely because Git metadata is unavailable.
            return "release"
        base_ref = describe.stdout.strip()

    def tracked_file(ref: str, filename: str) -> str | None:
        result = subprocess.run(
            ["git", "show", f"{ref}:{filename}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout if result.returncode == 0 else None

    # Compare committed release inputs. Developer-only untracked lockfiles must
    # not make the same candidate select different fleets locally and in CI.
    base_project = tracked_file(base_ref, "pyproject.toml")
    current_project = tracked_file("HEAD", "pyproject.toml")
    if base_project is None or current_project is None:
        return "toolchain"

    base_snapshot = _toolchain_snapshot(
        base_project, tracked_file(base_ref, "uv.lock") or ""
    )
    current_snapshot = _toolchain_snapshot(
        current_project,
        tracked_file("HEAD", "uv.lock") or "",
    )
    return "toolchain" if base_snapshot != current_snapshot else "release"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    models = subparsers.add_parser("models", help="print coherence models for a scope")
    models.add_argument(
        "--scope", choices=("auto", *sorted(VALID_SCOPES)), default="auto"
    )
    models.add_argument("--base-ref")
    models.add_argument("--format", choices=("shell", "json", "lines"), default="shell")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    scope = resolve_scope(requested=args.scope, base_ref=args.base_ref)
    models = models_for_scope(scope)
    if args.format == "json":
        print(json.dumps(models))
    elif args.format == "lines":
        print("\n".join(models))
    else:
        print(" ".join(models))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
