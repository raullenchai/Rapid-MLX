"""Shared, colored "Try these commands" block.

Used by:
- scripts/install.py   — printed at the end of install.sh
- scripts/sa3_mlx.py   — appended to `./sa3 --help`

Examples render with the user's `./sa3` wrapper when present (falling back to
`uv run python` or bare `python`), and only show entries for DiT bundles that
the user actually has installed. Bundles not yet on disk get a "re-run
install.sh, or just use them — weights auto-download" note at the bottom.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# This file lives in <project>/scripts/. SCRIPT_DIR points at the project
# root (where ./sa3, ./install.sh, models/, .venv/ live).
SCRIPT_DIR = Path(__file__).resolve().parent.parent


# ── ANSI colours (safe no-ops when stdout isn't a TTY) ──────────────────────
def _c(code: str) -> str:
    return code if sys.stdout.isatty() else ""

BOLD   = _c("\033[1m")
CYAN   = _c("\033[1;36m")
GREEN  = _c("\033[1;32m")
YELLOW = _c("\033[1;33m")
DIM    = _c("\033[2m")
RESET  = _c("\033[0m")


def _have(binary: str) -> bool:
    from shutil import which
    return which(binary) is not None


def _py_invocation() -> tuple[str, str, bool]:
    """Return (command-to-print, tip-or-empty, is_wrapper).

    Prefer the ./sa3 wrapper if present, else `uv run python scripts/sa3_mlx.py`,
    else fall back to a bare `python scripts/sa3_mlx.py`.
    """
    wrapper = SCRIPT_DIR / "sa3"
    if wrapper.exists() and os.access(wrapper, os.X_OK):
        return "./sa3", "the ./sa3 wrapper handles .venv + uv automatically", True
    if _have("uv"):
        return "uv run python", "uv run finds .venv/ automatically — no activation needed", False
    venv_dir = SCRIPT_DIR / ".venv"
    if Path(sys.prefix).resolve() == venv_dir.resolve():
        return ".venv/bin/python", "source .venv/bin/activate   # to run `python` directly", False
    return "python", "", False


def print_example_commands(header: str | None = None) -> None:
    """Print the categorized example-commands block.

    `header` is the title line inside the rule block. Defaults to a neutral
    "Examples:" for --help; install.sh passes "✓ Install complete. …".
    """
    from weights import bundle_status

    py, tip, is_wrapper = _py_invocation()
    prefix = py if is_wrapper else f"{py} scripts/sa3_mlx.py"

    def hdr(text: str) -> None:
        print(f"\n  {CYAN}{text}{RESET}")
    def cmd(args: str, comment: str = "") -> None:
        line = f"{prefix} {args}"
        if comment:
            print(f"    {GREEN}$ {line}{RESET}  {DIM}# {comment}{RESET}")
        else:
            print(f"    {GREEN}$ {line}{RESET}")

    if header is None:
        header = f"{BOLD}Examples:{RESET}"

    print(f"\n{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    print(f"  {header}")
    print(f"{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")

    if tip:
        print(f"\n  {DIM}tip: {tip}{RESET}")

    have = {name: bundle_status(name) == (4, 4) for name in ("medium", "sm-music", "sm-sfx")}

    # ── Basic generation ─────────────────────────────────────────────
    hdr("🎵 Generate audio from a prompt")
    if have["medium"]:
        cmd('--prompt "A beautiful piano arpeggio grows into a cinematic climax" \\\n'
            f'        --dit medium --decoder same-l --seconds 30 --out piano.wav',
            "highest quality (~5s for 10s clip on M1)")
    if have["sm-music"]:
        cmd('--prompt "lofi house loop, 120 BPM" \\\n'
            f'        --dit sm-music --decoder same-s --seconds 15 --out lofi.wav',
            "fast music generation (~1s wall for 10s clip)")
    if have["sm-sfx"]:
        cmd('--prompt "footsteps on gravel, then a door slamming" \\\n'
            f'        --dit sm-sfx --decoder same-s --seconds 8 --out sfx.wav',
            "sound-effect generation")

    # ── Playback ─────────────────────────────────────────────────────
    hdr("▶  Play immediately after generation")
    one_dit = "medium" if have["medium"] else ("sm-music" if have["sm-music"] else "sm-sfx")
    one_dec = "same-l" if one_dit == "medium" else "same-s"
    cmd(f'--prompt "ambient drone" --dit {one_dit} --decoder {one_dec} \\\n'
        f'        --seconds 10 --out drone.wav --play',
        "writes WAV + plays via afplay (Ctrl-C stops both)")

    # ── Audio-to-audio + inpaint ─────────────────────────────────────
    hdr("🎚️  Audio-to-audio & inpainting (requires an input WAV)")
    cmd(f'--prompt "jazz fusion with electric piano" --dit {one_dit} --decoder {one_dec} \\\n'
        f'        --init-audio funk.wav --init-noise-level 0.7 --out funk_jazz.wav',
        "variation: 0.4-0.8 typical, higher = more change")
    cmd(f'--prompt "explosive drum break" --dit {one_dit} --decoder {one_dec} \\\n'
        f'        --init-audio funk.wav --inpaint-range "4,7" --out funk_drums.wav',
        "regenerate seconds 4-7, keep rest")

    # ── CFG & negative prompts ───────────────────────────────────────
    hdr("🎯 Steer with CFG + negative prompts")
    cmd(f'--prompt "ambient drone" --cfg 3.0 \\\n'
        f'        --negative-prompt "drums, vocals, distortion" \\\n'
        f'        --dit {one_dit} --decoder {one_dec} --out clean_drone.wav',
        "cfg > 1.0 toward prompt, neg pushes away")

    # Bundles not installed (offer the on-demand path)
    missing = [name for name, ok in have.items() if not ok]
    if missing:
        print(f"\n  {YELLOW}note:{RESET} bundles not installed: {', '.join(missing)}.")
        print(f"        Re-run {BOLD}./install.sh{RESET} to pick them up, or just use them in"
              f" {BOLD}./sa3{RESET} —")
        print(f"        weights auto-download from HuggingFace on first use.")

    print(f"\n{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}\n")
