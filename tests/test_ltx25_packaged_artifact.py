# SPDX-License-Identifier: Apache-2.0
"""The built wheel really ships a working, self-contained LTX-2.5 runtime.

This is the end-to-end artifact contract behind ``pip install rapid-mlx``:
the wheel is built, installed into a fresh interpreter outside the source
tree, and the installed distribution's own runtime preflight must resolve the
vendored packages as trusted. The sdist must rebuild to the same wheel.

Marked ``slow`` because it builds distributions, resolves the full dependency
set in a fresh venv, and therefore needs network access. CI runs it in a
dedicated Apple-Silicon step.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_PACKAGES = ("ltx_core_mlx", "ltx_pipelines_mlx")
_BINARY_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".npz",
    ".npy",
    ".onnx",
    ".mlx",
    ".gguf",
)


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "RAPID_MLX_LTX25_RUNTIME"):
        environment.pop(name, None)
    return environment


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 1800) -> str:
    result = subprocess.run(
        command,
        cwd=cwd or REPO_ROOT,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"{' '.join(command)} failed with exit {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def _single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    assert len(matches) == 1, f"expected one {pattern} in {directory}: {matches}"
    return matches[0]


def _ltx_members(names: list[str]) -> set[str]:
    return {name for name in names if name.split("/", 1)[0] in VENDORED_PACKAGES}


def _assert_no_binaries(names: list[str]) -> None:
    offenders = [name for name in names if name.lower().endswith(_BINARY_SUFFIXES)]
    assert not offenders, (
        f"model weights or binaries leaked into the artifact: {offenders}"
    )


@pytest.mark.slow
@pytest.mark.requires_mlx
def test_wheel_ships_a_trusted_ltx25_runtime(tmp_path: Path) -> None:
    if sys.platform != "darwin":
        pytest.skip("MLX wheels are macOS-only")
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build and install the artifact")

    wheel_dir = tmp_path / "wheel"
    _run([uv, "build", "--wheel", "--out-dir", str(wheel_dir), str(REPO_ROOT)])
    wheel = _single(wheel_dir, "*.whl")

    with zipfile.ZipFile(wheel) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        _assert_no_binaries(names)
        ltx = _ltx_members(names)
        for package in VENDORED_PACKAGES:
            assert f"{package}/__init__.py" in ltx
            assert f"{package}/_vendored.py" in ltx
            assert f"{package}/LICENSE" in ltx
            assert f"{package}/NOTICE" in ltx
        assert (
            "ltx_core_mlx/text_encoders/gemma/encoders/prompts/"
            "gemma_t2v_system_prompt.txt" in ltx
        )
        compressed = sum(info.compress_size for info in infos if info.filename in ltx)
        # The two packages are ~1 MB of source; the compressed contribution
        # is a small fraction of the wheel. Pin the rough order of magnitude
        # so an accidental weight/blob inclusion is caught here too.
        assert 100_000 <= compressed <= 800_000, compressed

    venv = tmp_path / "venv"
    _run([uv, "venv", "--python", "3.12", str(venv)], cwd=tmp_path)
    python = venv / "bin" / "python"
    assert python.is_file()
    _run(
        [uv, "pip", "install", "--python", str(python), str(wheel)],
        cwd=tmp_path,
    )
    _run([uv, "pip", "check", "--python", str(python)], cwd=tmp_path)

    _run(
        [
            str(python),
            "-c",
            "import ltx_core_mlx, ltx_pipelines_mlx.cli; print('imported')",
        ],
        cwd=tmp_path,
    )
    preflight = _run(
        [
            str(python),
            "-c",
            (
                "from rapid_mlx.video.ltx25 import ltx25_runtime\n"
                "runtime = ltx25_runtime()\n"
                "print(runtime.state, runtime.source, runtime.executable)\n"
                "assert runtime.state == 'trusted', runtime\n"
                "assert runtime.source == 'vendored', runtime\n"
            ),
        ],
        cwd=tmp_path,
    )
    assert preflight.split()[:2] == ["trusted", "vendored"], preflight

    sdist_dir = tmp_path / "sdist"
    _run([uv, "build", "--sdist", "--out-dir", str(sdist_dir), str(REPO_ROOT)])
    sdist = _single(sdist_dir, "*.tar.gz")
    with tarfile.open(sdist) as archive:
        # The sdist records directory entries; compare files only, like the
        # wheel's namelist.
        sdist_names = [
            member.name for member in archive.getmembers() if member.isfile()
        ]
    _assert_no_binaries(sdist_names)
    sdist_ltx = {
        name.split("/", 1)[1]
        for name in sdist_names
        if "/" in name and name.split("/", 1)[1].split("/", 1)[0] in VENDORED_PACKAGES
    }
    assert sdist_ltx == ltx, "sdist and wheel disagree about the vendored LTX-2.5 files"

    extracted = tmp_path / "sdist-source"
    extracted.mkdir()
    with tarfile.open(sdist) as archive:
        archive.extractall(extracted, filter="data")
    source_root = _single(extracted, "rapid_mlx-*")
    rebuilt_dir = tmp_path / "rebuilt"
    _run(
        [uv, "build", "--wheel", "--out-dir", str(rebuilt_dir), str(source_root)],
        cwd=tmp_path,
    )
    rebuilt = _single(rebuilt_dir, "*.whl")

    with (
        zipfile.ZipFile(wheel) as original_archive,
        zipfile.ZipFile(rebuilt) as rebuilt_archive,
    ):
        rebuilt_names = set(rebuilt_archive.namelist())
        assert _ltx_members(rebuilt_names) == ltx
        for name in sorted(ltx):
            assert original_archive.read(name) == rebuilt_archive.read(name), (
                f"{name} changed when rebuilding the wheel from the sdist"
            )
