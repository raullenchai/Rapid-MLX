# SPDX-License-Identifier: Apache-2.0
"""Run upgrades while avoiding the known stale dependency metadata advisory."""

import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

STALE_EXTRA_ADVISORY = re.compile(
    r"^WARNING: transformers \S+ does not provide the extra 'tokenizers'$"
)

UV_DEPENDENCY_METADATA = """\
[[dependency-metadata]]
name = "mlx-video-with-audio"
version = "0.1.36"
requires-dist = ["mlx>=0.22.0", "numpy", "safetensors", "huggingface_hub", "transformers", "tqdm",
                 "opencv-python>=4.12.0.88", "Pillow>=10.3.0", "mlx-vlm", "soundfile>=0.12.0",
                 "pytest; extra == \\"dev\\""]
"""

_SYSTEM_CONFIG_DIR = Path("/etc")


def _exists_or_symlink(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def uv_config_absent(cwd: Path) -> bool:
    """Mirror the installer's conservative check for existing uv configuration."""
    if "UV_CONFIG_FILE" in os.environ or "UV_NO_CONFIG" in os.environ:
        return False
    config_home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    config_dirs = [config_home, str(_SYSTEM_CONFIG_DIR)]
    config_dirs.extend(os.environ.get("XDG_CONFIG_DIRS", "").split(":"))
    if any(
        _exists_or_symlink(Path(directory) / "uv" / "uv.toml")
        for directory in config_dirs
        if directory
    ):
        return False
    for directory in (cwd, *cwd.parents):
        if _exists_or_symlink(directory / "uv.toml"):
            return False
        project = directory / "pyproject.toml"
        if _exists_or_symlink(project):
            try:
                with project.open("rb") as stream:
                    doc = tomllib.load(stream)
            except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
                return False
            if "tool" in doc and (
                not isinstance(doc["tool"], dict) or "uv" in doc["tool"]
            ):
                return False
    return True


def _run_filtered(argv: list[str]) -> int:
    process = subprocess.Popen(
        argv, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace"
    )
    try:
        assert process.stderr is not None
        for line in process.stderr:
            if not STALE_EXTRA_ADVISORY.fullmatch(line.removesuffix("\n")):
                sys.stderr.write(line)
                sys.stderr.flush()
        return process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise
    finally:
        if process.stderr is not None:
            process.stderr.close()


def run_upgrade_command(argv: list[str]) -> int:
    """Run an upgrade, respecting existing user configuration and output."""
    if argv[1:3] == ["-m", "pip"]:
        return _run_filtered(argv)
    if Path(argv[0]).name == "uv":
        if uv_config_absent(Path.cwd()):
            with tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "uv.toml"
                config.write_text(UV_DEPENDENCY_METADATA, encoding="utf-8")
                return subprocess.run(
                    argv, env={**os.environ, "UV_CONFIG_FILE": str(config)}
                ).returncode
        return subprocess.run(argv).returncode
    return subprocess.run(argv, check=False).returncode
