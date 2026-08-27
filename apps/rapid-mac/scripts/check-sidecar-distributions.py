#!/usr/bin/env python3
"""Fail closed on incompatible installed distributions in the Desktop sidecar."""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def is_validated_compatibility_exception(
    *, owner: str, owner_version: str, requirement: Requirement, actual: str
) -> bool:
    """Return true only for the release-tested reduced vision runtime.

    mlx-vlm 0.6.16 declares Transformers >=5.14 for its complete feature set.
    Rapid's shipped Qwen/Gemma image lanes do not exercise that newer surface;
    they are validated by real image inference with Transformers 5.12.1.  This
    is deliberately an exact tuple rather than a package-wide metadata bypass.

    Remove this exception after replacing Rapid's Transformers <5.13 cap with
    !=5.13.0 and completing the tracked 5.15.x coherence sweep. Every other
    owner/version, dependency, declared specifier, or actual version remains a
    hard error.
    """

    return (
        canonicalize_name(owner) == "mlx-vlm"
        and owner_version == "0.6.16"
        and canonicalize_name(requirement.name) == "transformers"
        and str(requirement.specifier) == ">=5.14.0"
        and actual == "5.12.1"
    )


def find_errors(site_packages: Path) -> list[str]:
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
            if is_validated_compatibility_exception(
                owner=owner,
                owner_version=dist.version,
                requirement=requirement,
                actual=actual,
            ):
                print(
                    "==> validated compatibility exception: "
                    "mlx-vlm 0.6.16 with transformers 5.12.1"
                )
                continue
            errors.append(
                f"{owner} requires {requirement}, but bundled "
                f"{requirement.name}=={actual}"
            )
    return sorted(errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("site_packages", type=Path)
    args = parser.parse_args()
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
