#!/usr/bin/env python3
"""Print the requirements the Desktop sidecar installs for rapid-mlx.

The engine's base install carries the vision (mlx-vlm + torch/OpenCV), image
(mflux) and video (mlx-video + imageio) runtimes. The sidecar cannot afford that dependency cascade
under its bundle-size cap, so it installs rapid-mlx itself with ``--no-deps``
and resolves only this list: the base requirements minus every package named
by the ``[vision]`` / ``[image]`` / ``[video]`` alias extras, plus the
requested extras. The later build steps then add the reduced, validated
mlx-vlm + Pillow, mflux and minimal video runtimes exactly as before.

Accepts either the engine source tree (reads ``pyproject.toml``) or a built
wheel (reads its ``METADATA``), so candidate-wheel builds use the very
metadata that ships. Must run under the sidecar's own interpreter: markers
are evaluated for the bundle's Python/platform.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

# The sidecar's bare build interpreter has pip (stripped only after assembly)
# but no standalone ``packaging``; pip's vendored copy is always present.
from pip._vendor.packaging.requirements import Requirement

HEAVY_ALIAS_EXTRAS = ("vision", "image", "video")
_EXTRA_MARKER = re.compile(r"""\bextra\s*==\s*["']([^"']+)["']""")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _from_pyproject(source: Path) -> tuple[list[str], dict[str, list[str]]]:
    import tomllib

    project = tomllib.loads((source / "pyproject.toml").read_text())["project"]
    return list(project["dependencies"]), {
        extra: list(reqs)
        for extra, reqs in project.get("optional-dependencies", {}).items()
    }


def _from_wheel(wheel: Path) -> tuple[list[str], dict[str, list[str]]]:
    with zipfile.ZipFile(wheel) as archive:
        name = next(
            entry
            for entry in archive.namelist()
            if entry.endswith(".dist-info/METADATA") and entry.count("/") == 1
        )
        metadata = Parser().parsestr(archive.read(name).decode("utf-8"))
    base: list[str] = []
    extras: dict[str, list[str]] = {}
    for raw in metadata.get_all("Requires-Dist") or ():
        owner = _EXTRA_MARKER.search(raw)
        if owner:
            extras.setdefault(owner.group(1), []).append(raw)
        else:
            base.append(raw)
    return base, extras


def _applies(raw: str, extra: str = "") -> Requirement | None:
    requirement = Requirement(raw)
    if requirement.marker and not requirement.marker.evaluate({"extra": extra}):
        return None
    requirement.marker = None
    return requirement


def sidecar_requirements(source: Path, extras: list[str]) -> list[str]:
    if source.suffix == ".whl":
        base, optional = _from_wheel(source)
    else:
        base, optional = _from_pyproject(source)
    missing = [
        extra for extra in (*HEAVY_ALIAS_EXTRAS, *extras) if extra not in optional
    ]
    if missing:
        raise SystemExit(f"rapid-mlx metadata has no extra(s): {', '.join(missing)}")
    excluded = {
        _canonical(Requirement(raw).name)
        for extra in HEAVY_ALIAS_EXTRAS
        for raw in optional[extra]
    }
    selected: list[str] = []
    for raw in base:
        requirement = _applies(raw)
        if requirement and _canonical(requirement.name) not in excluded:
            selected.append(str(requirement))
    for extra in extras:
        for raw in optional[extra]:
            requirement = _applies(raw, extra)
            if requirement:
                selected.append(str(requirement))
    return list(dict.fromkeys(selected))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="engine source tree or wheel")
    parser.add_argument(
        "--extras", default="", help="comma-separated extras to include"
    )
    args = parser.parse_args(argv)
    extras = [extra for extra in args.extras.split(",") if extra]
    for requirement in sidecar_requirements(args.source, extras):
        print(requirement)
    return 0


if __name__ == "__main__":
    sys.exit(main())
