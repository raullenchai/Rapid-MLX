#!/usr/bin/env python3
"""Fail closed on incompatible installed distributions in the Desktop sidecar."""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path

SIDECAR_CONSTRAINTS_FILE = Path(__file__).with_name("sidecar-constraints.txt")


def emit_constraints() -> str:
    """Return the canonical pip constraints used by release-shaped installs."""

    return SIDECAR_CONSTRAINTS_FILE.read_text()


def find_errors(site_packages: Path) -> list[str]:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    distributions = list(metadata.distributions(path=[str(site_packages)]))
    if not distributions:
        return [f"no installed distributions found in {site_packages}"]
    installed = {
        canonicalize_name(dist.metadata["Name"]): dist.version for dist in distributions
    }
    errors: list[str] = []
    for dist in distributions:
        owner = dist.metadata["Name"]
        for raw in dist.requires or ():
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            actual = installed.get(canonicalize_name(requirement.name))
            if actual is None or not requirement.specifier:
                continue
            if actual in requirement.specifier:
                continue
            errors.append(
                f"{owner} requires {requirement}, but bundled "
                f"{requirement.name}=={actual}"
            )
    return sorted(errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("site_packages", type=Path, nargs="?")
    parser.add_argument(
        "--emit-constraints",
        action="store_true",
        help="write the release-tested pip constraint set to stdout",
    )
    args = parser.parse_args()
    if args.emit_constraints:
        if args.site_packages is not None:
            parser.error("site_packages cannot be combined with --emit-constraints")
        print(emit_constraints(), end="")
        return 0
    if args.site_packages is None:
        parser.error("site_packages is required unless --emit-constraints is used")
    if not args.site_packages.is_dir():
        parser.error(f"site-packages directory does not exist: {args.site_packages}")
    errors = find_errors(args.site_packages)
    if errors:
        raise SystemExit(
            "incompatible bundled distributions:\n  " + "\n  ".join(errors)
        )
    print("==> bundled distribution metadata constraints: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
