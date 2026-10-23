# SPDX-License-Identifier: Apache-2.0
"""Exercise package-manager upgrades without network or inference dependencies."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _wheel(directory, version):
    path = directory / f"rapid_mlx-{version}-py3-none-any.whl"
    metadata = f"rapid_mlx-{version}.dist-info"
    files = {
        "upgrade_fixture.py": f"def main():\n    print('{version}')\n",
        f"{metadata}/METADATA": f"Metadata-Version: 2.1\nName: rapid-mlx\nVersion: {version}\n",
        f"{metadata}/WHEEL": "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{metadata}/entry_points.txt": "[console_scripts]\nrapid-mlx = upgrade_fixture:main\n",
    }
    files[f"{metadata}/RECORD"] = (
        "".join(f"{name},,\n" for name in files) + f"{metadata}/RECORD,,\n"
    )
    with zipfile.ZipFile(path, "w") as wheel:
        for name, contents in files.items():
            wheel.writestr(name, contents)
    return path


@pytest.fixture
def upgrade_root():
    # Keep launcher shebangs below the platform length limit.
    with tempfile.TemporaryDirectory(prefix="upgrade-") as directory:
        yield Path(directory)


@pytest.mark.parametrize("manager", ["pip", "uv"])
def test_real_environment_upgrade_and_index_failure(upgrade_root, manager):
    tmp_path = upgrade_root
    uv = shutil.which("uv")
    if manager == "uv" and not uv:
        pytest.skip("uv is required to exercise a uv-created environment")
    home = tmp_path / "home"
    environment = home / ".rapid-mlx"
    home.mkdir()
    command = (
        [uv, "venv", "--python", sys.executable, str(environment)]
        if manager == "uv"
        else [sys.executable, "-m", "venv", str(environment)]
    )
    subprocess.run(command, check=True, capture_output=True)
    python = str(environment / "bin/python")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    old = _wheel(wheels, "0.0.1")
    installer = (
        [uv, "pip", "install", "--python", python]
        if manager == "uv"
        else [python, "-m", "pip", "install"]
    )
    env = dict(
        os.environ, PIP_NO_INDEX="1", UV_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1"
    )
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    subprocess.run(
        [*installer, "--no-index", str(old)],
        env=env,
        cwd=home,
        check=True,
        capture_output=True,
    )
    _wheel(wheels, "0.0.2")
    # Detect from the real launcher, metadata, interpreter, and pip availability.
    probe = (
        "import json, sys; from pathlib import Path; "
        "sys.path.insert(0, sys.argv[1]); "
        "from rapid_mlx._version_check import detect_install_method; "
        "sys.path.pop(0); "
        "Path.home = lambda: Path(sys.argv[2]); "
        "info = detect_install_method(); "
        "print(json.dumps([info.method, info.upgrade_argv]))"
    )
    env["PATH"] = str(environment / "bin") + os.pathsep + os.environ["PATH"]
    result = subprocess.run(
        [python, "-c", probe, str(ROOT), str(home)],
        env=env,
        cwd=home,
        check=True,
        capture_output=True,
        text=True,
    )
    method, argv = json.loads(result.stdout)
    assert method == "install_sh"
    assert ("--python" in argv) == (manager == "uv")
    env.update(PIP_FIND_LINKS=str(wheels), UV_FIND_LINKS=str(wheels))
    subprocess.run(
        [*argv, "--no-index", "--find-links", str(wheels)],
        env=env,
        cwd=home,
        check=True,
        capture_output=True,
    )
    version = subprocess.check_output(
        [
            python,
            "-I",
            "-c",
            "from importlib.metadata import version; print(version('rapid-mlx'))",
        ],
        text=True,
        cwd=home,
        env=env,
    )
    assert version.strip() == "0.0.2"
    # An unavailable index must fail without fetching another installer/source.
    # Force a missing requested package in the same upgrade command.
    missing = [
        *argv[:-1],
        "rapid-mlx==0.0.3",
        "--no-index",
        "--find-links",
        str(wheels),
    ]
    failure = subprocess.run(missing, env=env, cwd=home, capture_output=True, text=True)
    assert failure.returncode != 0
    assert "0.0.3" in failure.stderr
