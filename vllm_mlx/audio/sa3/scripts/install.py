#!/usr/bin/env python3
"""Internal install plumbing — do NOT invoke directly. Use ../install.sh from the project root.

Called by install.sh after it has set up uv + .venv. Responsibilities:
1. (Optionally) pip-install requirements.txt — skipped when INSTALL_SKIP_PIP=1
   is set (install.sh sets this since uv already handled deps).
2. Ask which DiT bundles to download, then fetch them from HuggingFace.
3. Print example commands the user can copy-paste.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

# This file lives in <project>/scripts/. SCRIPT_DIR points at the project
# root (where models/, requirements.txt, .venv/, sa3 wrapper live).
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # `from weights import …`

MIN_PY = (3, 10)
SKIP_PIP = os.environ.get("INSTALL_SKIP_PIP") == "1"


def step(msg: str) -> None:
    print(f"\n\033[1;36m→ {msg}\033[0m")


def check_environment() -> None:
    """Bail out early with a clear message if the interpreter is wrong."""
    py = sys.version_info
    if (py.major, py.minor) < MIN_PY:
        print(
            f"\n\033[1;31merror\033[0m: this script is running under Python "
            f"{py.major}.{py.minor}, but MLX requires Python "
            f"{MIN_PY[0]}.{MIN_PY[1]}+.\n\n"
            f"Re-run install.py with a Python {MIN_PY[0]}.{MIN_PY[1]}+ interpreter, e.g.:\n"
            f"    /path/to/python3.11 install.py\n\n"
            f"On macOS you can get one via:\n"
            f"    brew install python@3.11\n"
            f"    /opt/homebrew/bin/python3.11 install.py\n"
            f"or with pyenv:\n"
            f"    pyenv install 3.11.10 && pyenv local 3.11.10 && python install.py\n",
            file=sys.stderr,
        )
        sys.exit(1)
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print(
            f"\n\033[1;33mwarning\033[0m: this stack is Apple-Silicon-only "
            f"(MLX is Metal-backed). Detected {platform.system()}/{platform.machine()} "
            f"— pip install will probably fail when it tries to fetch MLX.",
            file=sys.stderr,
        )


def pip_install_requirements() -> None:
    if SKIP_PIP:
        step("Dependencies already installed by install.sh (uv) — skipping pip step")
        return
    step("Installing Python dependencies")
    req = SCRIPT_DIR / "requirements.txt"

    # Prefer pip; fall back to uv if pip isn't available in this interpreter
    # (e.g. user is running install.py directly inside a uv-created venv
    # that wasn't seeded with pip).
    pip_available = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
    ).returncode == 0

    if pip_available:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(req)]
    elif _have("uv"):
        cmd = ["uv", "pip", "install", "--python", sys.executable, "-r", str(req)]
    else:
        print(
            f"\n\033[1;31merror\033[0m: neither pip nor uv is available in this "
            f"interpreter ({sys.executable}).\n"
            f"Install pip with `python -m ensurepip --upgrade`, or use ./install.sh "
            f"which sets everything up via uv.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def _have(binary: str) -> bool:
    """True if the given binary is on PATH."""
    from shutil import which
    return which(binary) is not None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="SA3 MLX install — non-interactive bundle downloader + post-install help.",
        epilog="Normally invoked via ../install.sh, not directly.",
    )
    ap.add_argument("--download", default="",
                    metavar="BUNDLES",
                    help="Comma-separated list of bundles to pre-download "
                         "(medium, sm-music, sm-sfx). Without this flag, "
                         "nothing is downloaded — sa3_mlx.py will fetch any "
                         "missing weights from HuggingFace on first use.")
    cli = ap.parse_args()

    check_environment()
    pip_install_requirements()

    # Import after pip install so a fresh checkout works.
    from weights import DIT_BUNDLES, SHARED, BUNDLE_SIZES, bundle_status, ensure_local

    step("Current weights status")
    for name in DIT_BUNDLES:
        present, total = bundle_status(name)
        mark = "✓" if present == total else " "
        print(f"  [{mark}] {name:9s}  {present}/{total} files present   ({BUNDLE_SIZES[name]})")

    chosen: list[str] = []
    if cli.download.strip():
        chosen = [b.strip() for b in cli.download.split(",") if b.strip()]
        unknown = [b for b in chosen if b not in DIT_BUNDLES]
        if unknown:
            print(f"\n\033[1;31merror\033[0m: unknown bundle(s): {', '.join(unknown)}. "
                  f"Choices: {', '.join(DIT_BUNDLES)}", file=sys.stderr)
            sys.exit(1)

    if chosen:
        step(f"Downloading {len(chosen)} bundle(s): {', '.join(chosen)}")
        seen: set[str] = set()
        for name in chosen:
            print(f"\n[{name}]")
            items = DIT_BUNDLES[name] + SHARED
            for rel, _hf in items:
                if rel in seen:
                    continue
                seen.add(rel)
                ensure_local(rel)
    else:
        step("No --download set — weights will auto-download on first ./sa3 use")
        print(f"  To pre-download instead, pass:  {sys.executable.split('/')[-1]} install.py --download medium")
        print(f"                              or:  ./install.sh --download medium,sm-music")

    from examples import print_example_commands, BOLD, GREEN, RESET
    print_example_commands(header=f"{BOLD}{GREEN}✓ Install complete.{RESET}  Try these commands:")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
