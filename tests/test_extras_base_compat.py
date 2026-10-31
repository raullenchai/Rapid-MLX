"""Extras must not impose upper bounds that downgrade base dependencies."""

import importlib.metadata as metadata
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

PROJECT = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())[
    "project"
]
BASE = [Requirement(item) for item in PROJECT["dependencies"]]


def _active(req, extra=""):
    return not req.marker or req.marker.evaluate({"extra": extra})


def _base_closure():
    pending = [req.name for req in BASE if _active(req)]
    names = set()
    while pending:
        name = canonicalize_name(pending.pop())
        if name in names:
            continue
        names.add(name)
        try:
            requirements = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue
        pending.extend(
            req.name for item in requirements if _active(req := Requirement(item))
        )
    return names


def _probes(*specifiers):
    for specifier in specifiers:
        for spec in specifier:
            version = Version(spec.version.rstrip(".*"))
            release = (version.release + (0, 0, 0))[:3]
            major, minor, patch = release
            yield version
            try:
                yield Version(f"{version.base_version}.dev0")
            except InvalidVersion:
                pass
            for index in range(len(version.release) - 1, -1, -1):
                if version.release[index] > 0:
                    below = list(version.release[: index + 1])
                    below[index] -= 1
                    yield Version(".".join(map(str, [*below, 999])))
                    break
            yield Version(f"{major}.{minor}.{patch + 1}")
            yield Version(f"{major}.{minor + 1}.0")
            yield Version(f"{major + 1}.0.0")


def _narrowing_versions(
    base_spec: SpecifierSet, extra_spec: SpecifierSet
) -> list[Version]:
    upper = SpecifierSet(
        ",".join(
            str(s) for s in extra_spec if s.operator in {"<", "<=", "==", "===", "~="}
        )
    )
    return [
        version
        for version in _probes(base_spec, extra_spec)
        if base_spec.contains(version, prereleases=True)
        and not upper.contains(version, prereleases=True)
    ]


def test_narrowing_versions():
    assert _narrowing_versions(SpecifierSet("<2"), SpecifierSet("<1.5"))
    assert not _narrowing_versions(SpecifierSet(">=0.5.3,<0.6"), SpecifierSet("<0.6"))


CLOSURE = _base_closure() - {"rapid-mlx"}
DIRECT = {canonicalize_name(req.name): req for req in BASE if _active(req)}
CASES = [
    (extra, req)
    for extra, items in PROJECT["optional-dependencies"].items()
    for item in items
    if _active(req := Requirement(item), extra)
    and canonicalize_name(req.name) in CLOSURE
    and any(spec.operator in {"<", "<=", "==", "===", "~="} for spec in req.specifier)
]


@pytest.mark.parametrize(("extra", "req"), CASES, ids=lambda value: str(value))
def test_extra_does_not_narrow_base(extra, req):
    name = canonicalize_name(req.name)
    base = DIRECT.get(name)
    message = f"extra {extra}, package {name}: base {base.specifier if base else '<undeclared>'}, extra {req.specifier}"
    assert base is not None, message
    narrowing = _narrowing_versions(base.specifier, req.specifier)
    assert not narrowing, f"{message}; versions {narrowing}"
