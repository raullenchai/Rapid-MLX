# SPDX-License-Identifier: Apache-2.0
"""Keep tracked license and attribution files in distributed packages."""

import fnmatch
import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on the 3.10 floor
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("rapid_mlx", "vllm_mlx", "videox_fun_mlx")

if (
    shutil.which("git") is None
    or subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=REPO_ROOT,
        capture_output=True,
    ).returncode
):
    pytest.skip(
        "License packaging checks require a git checkout", allow_module_level=True
    )


def _tracked_license_files() -> set[tuple[str, str]]:
    files = set()
    for package in PACKAGES:
        tracked = subprocess.run(
            ["git", "ls-files", "-z", package],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout.split(b"\0")
        for raw_path in tracked:
            if not raw_path:
                continue
            path = Path(os.fsdecode(raw_path))
            name = path.name.upper()
            if path.suffix.lower() != ".py" and any(
                token in name for token in ("LICENSE", "NOTICE", "COPYING")
            ):
                files.add((package, path.relative_to(package).as_posix()))
    return files


def test_package_data_ships_every_tracked_license_file():
    tracked = _tracked_license_files()
    assert tracked and ("rapid_mlx", "spec_decode/mtp/LICENSE-MTPLX") in tracked
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    missing = sorted(
        f"{package}/{relative}"
        for package, relative in tracked
        if not any(
            fnmatch.fnmatch(relative, glob) for glob in package_data.get(package, [])
        )
    )
    assert not missing, (
        "Tracked license files missing from package-data:\n" + "\n".join(missing)
    )


@pytest.mark.slow
def test_real_distributions_ship_every_tracked_license_file(tmp_path):
    if importlib.util.find_spec("setuptools.build_meta") is None:
        pytest.skip("setuptools is not installed")
    source = tmp_path / "source"
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "NOTICE",
            *PACKAGES,
        ],
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
            "import sys, setuptools.build_meta as b; "
            "out = sys.argv[1]; b.build_wheel(out); b.build_sdist(out)",
            str(out),
        ],
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    )
    (wheel,) = out.glob("*.whl")
    (sdist,) = out.glob("*.tar.gz")
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
    with tarfile.open(sdist) as archive:
        sdist_names = set(archive.getnames())
    top = sdist.name.removesuffix(".tar.gz")
    missing = []
    inventory = _tracked_license_files()
    assert inventory and ("rapid_mlx", "spec_decode/mtp/LICENSE-MTPLX") in inventory
    for package, relative in sorted(inventory):
        path = f"{package}/{relative}"
        if path not in wheel_names:
            missing.append(f"wheel: {path}")
        if f"{top}/{path}" not in sdist_names:
            missing.append(f"sdist: {path}")
    assert not missing, (
        "Tracked license files missing from distributions:\n" + "\n".join(missing)
    )
