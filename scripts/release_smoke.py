#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Clean-room install + import smoke test for releases.

Catches the class of bug that #408 shipped: code that imports cleanly on
the dev machine because the local mlx (or any other runtime dep) has a
symbol/API that hasn't appeared in a released wheel yet. The dev gates
(`make smoke/check/full`, `pr_validate`, codex review) all run against
the dev mlx and silently agree with each other.

This script creates a brand-new venv and either:
  - pip-installs the working tree (pip handles PEP 517 build internally,
    so no `build` or `hatchling` dep is required on the dev env), or
  - pip-installs `rapid-mlx==X.Y.Z` from PyPI (post-tag verification).

Then it imports the modules that the published entrypoints would import.
An ``AttributeError`` / ``ImportError`` here means we are about to ship a
release that no user can ``pip install`` and use.

Usage:
    python3 scripts/release_smoke.py            # install working tree in clean venv, smoke-import
    python3 scripts/release_smoke.py --version 0.6.55  # post-tag: install from PyPI

Exit code: 0 on success, 1 on any failure. Designed to be the last gate
before pushing a ``chore: bump version to X.Y.Z`` commit.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _clean_subprocess_env() -> dict[str, str]:
    """Build an env dict that won't let the host shadow the clean venv.

    PYTHONPATH / PYTHONUSERBASE / PIP_TARGET / PIP_PREFIX can all inject
    host-site packages into a child interpreter even when it's the venv's
    own ``python``. Stripping them keeps the gate's "clean-room" promise
    honest. We keep PATH / HOME / LANG plus the proxy/cert vars pip needs
    in corporate-network environments (SSL_CERT_FILE etc.) so the gate
    is runnable behind a custom CA bundle.
    """
    keep = {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        # pip / requests TLS verification (corporate proxies, custom CAs)
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        # pip network / index configuration (corporate mirrors)
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_TRUSTED_HOST",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
    return {k: v for k, v in os.environ.items() if k in keep}


REPO_ROOT = Path(__file__).resolve().parent.parent

# Modules whose import-time side effects must succeed against a clean
# install. ``vllm_mlx.scheduler`` was the surface that #408 broke;
# the others cover every base ``[project.scripts]`` entrypoint
# (``rapid-mlx``, ``vllm-mlx``, ``vllm-mlx-bench``) plus the server
# surface that every ``rapid-mlx serve`` invocation imports before
# binding a port. ``vllm_mlx.gradio_app`` is intentionally excluded —
# it's the ``vllm-mlx-chat`` entrypoint, which lives behind the
# ``chat`` extra and is allowed to fail-import on the base install.
IMPORT_TARGETS = (
    "vllm_mlx",
    "vllm_mlx.scheduler",
    "vllm_mlx.server",
    "vllm_mlx.cli",
    "vllm_mlx.benchmark",
)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, stream output, raise on non-zero."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def smoke(install_spec: str, *, source: str) -> None:
    # ``venv`` is pre-bound to None so the finally can run safely no
    # matter where mkdtemp / venv / pip raises (DeepSeek r1 #1, r2).
    venv: Path | None = None
    env = _clean_subprocess_env()
    try:
        venv = Path(tempfile.mkdtemp(prefix="rapid-mlx-release-smoke-"))
        print(f"[release-smoke] clean venv: {venv}")
        run([sys.executable, "-m", "venv", str(venv)], env=env)
        py = venv / "bin" / "python"
        run([str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], env=env)

        # ``pip install <local-path>`` invokes the PEP 517 build backend
        # declared in ``pyproject.toml`` directly — no separate ``build``
        # package needed on the dev env. The wheel that gets built and
        # installed is bit-identical to what would land on PyPI.
        print(f"[release-smoke] installing {source}: {install_spec}")
        run([str(py), "-m", "pip", "install", "--quiet", install_spec], env=env)

        # Run the import probes from inside the venv, NOT the repo root:
        # ``python -c`` puts cwd at sys.path[0], so if we ran from
        # REPO_ROOT the in-tree ``vllm_mlx/`` would shadow the wheel we
        # just installed and the gate could pass against a broken
        # published artifact.
        print("[release-smoke] importing release surfaces in clean venv:")
        for mod in IMPORT_TARGETS:
            print(f"    import {mod}", flush=True)
            run([str(py), "-c", f"import {mod}"], cwd=str(venv), env=env)
        print("[release-smoke] OK — every release surface imports cleanly.")
    finally:
        if venv is not None:
            shutil.rmtree(venv, ignore_errors=True)


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
            smoke(str(REPO_ROOT), source="working tree")
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
