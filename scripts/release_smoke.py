#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Clean-room install + import smoke test for releases.

Catches the class of bug that #408 shipped: code that imports cleanly on
the dev machine because the local mlx (or any other runtime dep) has a
symbol/API that hasn't appeared in a released wheel yet. The dev gates
(`make smoke/check/full`, `pr_validate`, codex review) all run against
the dev mlx and silently agree with each other.

This script builds rapid-mlx from the working tree, installs it into a
brand-new venv with **only PyPI wheels** for the runtime deps, then
imports the modules that the published entrypoints would import. An
``AttributeError`` / ``ImportError`` here means we are about to ship a
release that no user can ``pip install`` and use.

Usage:
    python3 scripts/release_smoke.py            # build wheel, smoke-import it
    python3 scripts/release_smoke.py --version  # post-tag: smoke `pip install rapid-mlx==X` from PyPI

Exit code: 0 on success, 1 on any failure. Designed to be the last gate
before pushing a ``chore: bump version to X.Y.Z`` commit.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Modules whose import-time side effects must succeed against a clean
# install. ``vllm_mlx.scheduler`` was the surface that #408 broke; the
# others are added because every `rapid-mlx serve ...` invocation
# imports them before binding a port, and any future shim-style code
# added near them will fail the same way.
IMPORT_TARGETS = (
    "vllm_mlx",
    "vllm_mlx.scheduler",
    "vllm_mlx.server",
    "vllm_mlx.cli",
)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, stream output, raise on non-zero."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def smoke(install_spec: str, *, source: str) -> None:
    venv = Path(tempfile.mkdtemp(prefix="rapid-mlx-release-smoke-"))
    try:
        print(f"[release-smoke] clean venv: {venv}")
        run([sys.executable, "-m", "venv", str(venv)])
        py = venv / "bin" / "python"
        run([str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])

        print(f"[release-smoke] installing {source}: {install_spec}")
        run([str(py), "-m", "pip", "install", "--quiet", install_spec])

        print("[release-smoke] importing release surfaces in clean venv:")
        for mod in IMPORT_TARGETS:
            print(f"    import {mod}", flush=True)
            run([str(py), "-c", f"import {mod}"])
        print("[release-smoke] OK — every release surface imports cleanly.")
    finally:
        shutil.rmtree(venv, ignore_errors=True)


def build_wheel() -> Path:
    dist = REPO_ROOT / "dist"
    if dist.exists():
        shutil.rmtree(dist)
    print("[release-smoke] building wheel from working tree")
    run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)], cwd=REPO_ROOT
    )
    wheels = sorted(dist.glob("rapid_mlx-*.whl"))
    if not wheels:
        sys.exit("[release-smoke] FAIL: no rapid_mlx wheel built")
    if len(wheels) > 1:
        print(
            f"[release-smoke] WARN: multiple wheels in dist, using newest: {wheels[-1].name}"
        )
    return wheels[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        help="If set, install `rapid-mlx==<VERSION>` from PyPI instead of building locally. "
        "Use post-tag to verify the published wheel.",
    )
    args = parser.parse_args()

    try:
        if args.version:
            smoke(f"rapid-mlx=={args.version}", source="PyPI")
        else:
            wheel = build_wheel()
            smoke(str(wheel), source="local wheel")
    except subprocess.CalledProcessError as exc:
        print(
            f"\n[release-smoke] FAIL: command exited {exc.returncode}\n"
            "    A release shipped from this state would crash on `import vllm_mlx.*`\n"
            "    for any user whose runtime deps differ from the dev env.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
