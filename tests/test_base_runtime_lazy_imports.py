# SPDX-License-Identifier: Apache-2.0
"""Startup must stay lazy now that vision/image runtimes ship in the base install.

``pip install rapid-mlx`` now carries mlx-vlm, mflux, torch and OpenCV. The CLI
front door (``--help`` / ``--version`` / parser construction) must not import
any of them: a top-level import would add seconds to every command for every
user. The child process installs an import hook that *materialises* a stub for
each heavy package and records it, so the guard goes red even on hosts where
the real packages are absent (the no-MLX Linux lane).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

_HEAVY = ("torch", "torchvision", "cv2", "mlx_vlm", "mflux", "mlx_video", "imageio")

_CHILD = textwrap.dedent(
    """
    import importlib.abc
    import importlib.machinery
    import json
    import sys
    import types

    HEAVY = set(json.loads(sys.argv[1]))
    ARGV = json.loads(sys.argv[2])
    imported = []

    class _Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return types.ModuleType(spec.name)

        def exec_module(self, module):
            imported.append(module.__name__)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.partition(".")[0] in HEAVY:
                return importlib.machinery.ModuleSpec(name, _Loader())
            return None

    sys.meta_path.insert(0, _Finder())
    sys.argv = ["rapid-mlx", *ARGV]
    try:
        from rapid_mlx.cli import cli_entrypoint

        if ARGV:
            cli_entrypoint()
    except SystemExit:
        pass
    finally:
        sys.__stdout__.flush()
        sys.stderr.write("\\nHEAVY_IMPORTS=" + json.dumps(sorted(imported)) + "\\n")
    """
)


def _heavy_imports(tmp_path, argv: list[str]) -> list[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("RAPID_MLX_", "_ARGCOMPLETE"))
    }
    env["HOME"] = str(tmp_path)
    env["NO_COLOR"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(_HEAVY), json.dumps(argv)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
    )
    marker = [
        line for line in proc.stderr.splitlines() if line.startswith("HEAVY_IMPORTS=")
    ]
    assert marker, proc.stderr[-2000:]
    return json.loads(marker[-1].removeprefix("HEAVY_IMPORTS="))


@pytest.mark.parametrize(
    "argv",
    [[], ["--help"], ["--version"], ["serve", "--help"]],
    ids=["import", "help", "version", "serve-help"],
)
def test_cli_front_door_never_imports_bundled_heavy_runtimes(tmp_path, argv):
    assert _heavy_imports(tmp_path, argv) == []


def test_guard_detects_a_heavy_import(tmp_path, monkeypatch):
    """The stub hook must observe a real import, or the guard is vacuous."""
    child = _CHILD.replace(
        "from rapid_mlx.cli import cli_entrypoint",
        "import torch  # noqa: F401\n        from rapid_mlx.cli import cli_entrypoint",
    )
    monkeypatch.setattr(sys.modules[__name__], "_CHILD", child)
    assert _heavy_imports(tmp_path, []) == ["torch"]
