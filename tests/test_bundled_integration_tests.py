# SPDX-License-Identifier: Apache-2.0
"""Keep bundled integration tests in sync with agent profile declarations."""

import fnmatch
import importlib.util
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _declared_specific_tests(profile: dict) -> list[str]:
    """Return every ``specific_tests`` name, including version overrides."""
    specs = [profile.get("testing") or {}]
    specs.extend((v or {}).get("testing") or {} for v in profile.get("versions") or [])
    return [s["specific_tests"] for s in specs if s.get("specific_tests")]


def test_bundled_integration_tests_match_profiles():
    package = REPO_ROOT / "rapid_mlx"
    bundled = package / "_integration_tests"
    referenced = set()
    for profile_path in sorted((package / "agents" / "profiles").glob("*.yaml")):
        profile = yaml.safe_load(profile_path.read_text())
        for name in _declared_specific_tests(profile):
            assert (bundled / name).is_file(), f"{profile_path.name}: missing {name}"
            referenced.add(name)

    bundled_names = {
        path.name
        for path in bundled.iterdir()
        if path.name != "__init__.py" and (path.is_file() or path.is_symlink())
    }
    assert bundled_names == referenced


def test_declared_specific_tests_reads_version_overrides():
    profile = {
        "testing": {"specific_tests": "a.py"},
        "versions": [
            {"range": ">=1", "testing": {"specific_tests": "b.py"}},
            {"range": "<1"},
        ],
    }
    assert _declared_specific_tests(profile) == ["a.py", "b.py"]


def test_package_data_ships_every_bundled_integration_test():
    """The wheel must carry each bundled test, not just the source tree."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    globs = pyproject["tool"]["setuptools"]["package-data"]["rapid_mlx"]
    bundled = REPO_ROOT / "rapid_mlx" / "_integration_tests"
    for path in bundled.iterdir():
        if path.name == "__init__.py" or not (path.is_file() or path.is_symlink()):
            continue
        rel = f"_integration_tests/{path.name}"
        assert any(fnmatch.fnmatch(rel, g) for g in globs), f"{rel} not packaged"


def _profile_referenced_tests() -> set[str]:
    profiles = REPO_ROOT / "rapid_mlx" / "agents" / "profiles"
    names: set[str] = set()
    for profile_path in profiles.glob("*.yaml"):
        names.update(_declared_specific_tests(yaml.safe_load(profile_path.read_text())))
    return names


@pytest.mark.slow
def test_real_wheel_ships_exactly_the_profile_referenced_tests(tmp_path):
    """A real setuptools wheel build carries every profile-referenced test."""
    if importlib.util.find_spec("setuptools.build_meta") is None:
        pytest.skip("setuptools is not installed")
    source = tmp_path / "source"
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "pyproject.toml", "README.md", "LICENSE", "NOTICE"]
        + ["rapid_mlx", "vllm_mlx", "videox_fun_mlx"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    for raw_path in tracked:
        if not raw_path:
            continue
        relative = Path(os.fsdecode(raw_path))
        if not (REPO_ROOT / relative).exists():
            continue
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, target)  # follows bundle symlinks
    out = tmp_path / "dist"
    out.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, setuptools.build_meta as b; b.build_wheel(sys.argv[1])",
            str(out),
        ],
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    )
    (wheel,) = out.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        shipped = {
            Path(name).name
            for name in archive.namelist()
            if name.startswith("rapid_mlx/_integration_tests/")
            and not name.endswith("/__init__.py")
        }
    assert shipped == _profile_referenced_tests()
