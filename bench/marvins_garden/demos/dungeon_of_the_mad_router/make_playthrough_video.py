#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a dungeon playthrough log into a terminal-style mp4.

Usage:
  python make_playthrough_video.py <run.log> <out.mp4> [--fps 30]

The log is the captured stdout of `dungeon_of_the_mad_router.py --auto`.
Frames reveal the transcript line by line (a scrolling 33-line terminal
window); resolution lines hold longer, like a real cast. Colors are re-
applied from the log's markers after ANSI stripping.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
BG = (18, 22, 19)
PANEL = (24, 29, 25)
FG = (200, 200, 192)
DIM = (138, 146, 138)
CYAN = (127, 213, 224)
GREEN = (134, 222, 148)
RED = (232, 116, 104)
YELLOW = (226, 198, 120)
WHITE = (245, 245, 240)
LH = 20
FONT = "/System/Library/Fonts/Menlo.ttc"

HOLD_LONG = 1.15   # after a resolution line
HOLD_TURN = 0.55   # after lane-panel lines
HOLD_TYPE = 0.30   # ordinary line (typing feel)


def clean(line: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", line.rstrip("\n"))


def color_of(line: str) -> tuple[int, int, int]:
    if line.startswith("═") or line.startswith("you>"):
        return WHITE
    if line.startswith("│ "):
        return CYAN
    if any(k in line for k in ("YOU WIN", "cleared", "+8GB")):
        return GREEN
    if any(k in line for k in ("strikes you", "GAME OVER", "blocks your")):
        return RED
    if line.startswith("[") or line.startswith("  (") or "HP " in line[:40]:
        return YELLOW
    if line.startswith("  ") or line.startswith("'"):
        return DIM
    return FG


def hold_of(prev: str) -> float:
    if any(k in prev for k in ("YOU WIN", "GAME OVER", "cleared", "strikes you",
                               "blocks your", "The Oracle opens", "exiles you")):
        return HOLD_LONG
    if prev.startswith("│ "):
        return HOLD_TURN
    return HOLD_TYPE


def render_frame(out_path: Path, window: list[str], step_hint: str) -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 44], fill=PANEL)
    font_h = ImageFont.truetype(FONT, 16)
    font_b = ImageFont.truetype(FONT, 15)
    d.text((18, 12), "THE DUNGEON OF THE MAD ROUTER — Marvin's Garden playthrough (--auto, 0 tokens generated)",
           font=font_h, fill=GREEN)
    d.text((W - 330, 12), step_hint, font=font_h, fill=DIM)
    y = 58
    for line in window:
        if y + LH > H - 12:
            break
        d.text((18, y), line[:110], font=font_b, fill=color_of(line))
        y += LH
    d.rectangle([0, H - 8, W, H], fill=PANEL)
    img.save(out_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args(argv)

    lines = [clean(l) for l in Path(args.log).read_text(errors="replace").splitlines()]
    lines = [l for l in lines if l.strip() != "" or True]
    # drop the stderr-style "summoning/loading" first lines if present
    lines = [l for l in lines if not l.startswith("summoning")]

    frames: list[tuple[list[str], float, str]] = []
    window: list[str] = []
    room = 0
    for i, line in enumerate(lines):
        window.append(line)
        if len(window) > 33:
            window = window[-33:]
        if line.startswith("["):
            room += 1
        hint = f"room {min(room, 5)}/5 · turn {sum(1 for l in lines[:i+1] if l.startswith('you>'))}"
        frames.append((list(window), hold_of(line), hint))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        concat = td / "cast.ffconcat"
        with concat.open("w") as fh:
            fh.write("ffconcat version 1.0\n")
            for i, (win, hold, hint) in enumerate(frames):
                png = td / f"f{i:04d}.png"
                render_frame(png, win, hint)
                fh.write(f"file '{png}'\nduration {hold:.2f}\n")
            # concat demuxer repeats the last frame's duration; add a tail hold
            fh.write(f"file '{(td / f'f{len(frames)-1:04d}')}'\nduration 2.0\n")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
            "-vf", f"fps={args.fps},format=yuv420p", "-c:v", "libx264",
            "-preset", "medium", "-crf", "21", "-movflags", "+faststart",
            args.out,
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {args.out} ({len(frames)} events)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
