#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a match JSONL (from play_match.py) into the deliverable video.

Same visual language as the web demo: LEFT panel = Marvin decision bar chart
(probabilities per option, latency, oracle agreement), RIGHT = battle state.
Pillow frames → ffmpeg h264, 1280x720.

  python replay_video.py match.jsonl --out spire_demo.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
PANEL = 430
FONT = "/System/Library/Fonts/Menlo.ttc"
_f = lambda s, sz: ImageFont.truetype(FONT, sz, index=(1 if sz > 14 else 0))

C_BG, C_PANEL, C_TXT, C_DIM = (16, 20, 24), (10, 13, 17), (216, 222, 230), (93, 107, 120)
C_GREEN, C_BLUE, C_RED, C_GOLD = (127, 208, 160), (61, 107, 143), (224, 123, 123), (224, 181, 109)


def render_frame(step: dict, out_path: Path):
    img = Image.new("RGB", (W, H), C_BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, PANEL, H], fill=C_PANEL)
    d.line([PANEL, 0, PANEL, H], fill=(35, 42, 51), width=2)

    f12, f14, f18, f26 = _f("m", 12), _f("m", 14), _f("m", 18), _f("m", 26)
    st, dec = step["state"], step.get("decision", {})

    # ── left panel ──
    d.text((20, 18), "MARVIN · decision readout", font=f18, fill=C_GREEN)
    lat = dec.get("latency_ms")
    d.text((20, 48), f"{lat:.0f} ms" if lat else "—", font=f26, fill=C_GREEN)
    d.text((20, 88), "single forward · 0 generated tokens", font=f12, fill=C_DIM)
    agree = dec.get("oracle_agrees")
    if agree is not None:
        d.text((20, 106), ("✓ matches rollout oracle" if agree else "✗ diverges from oracle"),
               font=f14, fill=C_GREEN if agree else C_RED)

    probs = step.get("probabilities") or {}
    y = 140
    chosen = dec.get("chosen")
    for key, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        pick = key == chosen
        d.text((20, y), ("▶ " if pick else "  ") + key[:30], font=f14,
               fill=C_TXT if pick else C_DIM)
        d.rectangle([250, y + 2, 380, y + 18], fill=(26, 33, 42))
        d.rectangle([250, y + 2, 250 + max(2, int(130 * p)), y + 18],
                    fill=C_GREEN if pick else C_BLUE)
        d.text((388, y), f"{100 * p:5.1f}%", font=f14, fill=C_TXT if pick else C_DIM)
        y += 26
    # facts strip
    y = max(y + 16, 470)
    d.text((20, y), f"enemy intent: {st['incoming']} dmg/turn", font=f14, fill=C_GOLD)
    d.text((20, y + 22), f"your block: {st['player']['block']}   energy: {st['player']['energy']}/3",
           font=f14, fill=C_DIM)

    # ── right: battle ──
    x0 = PANEL + 40
    d.text((x0, 24), f"SLAY THE SPIRE — floor battle · turn {st['turn'] + 1}", font=f18, fill=C_TXT)
    y = 70
    for m in st["monsters"]:
        d.rounded_rectangle([x0, y, x0 + 380, y + 74], 8, outline=(42, 51, 64), width=2)
        d.text((x0 + 14, y + 10), f"{m['name']}", font=f18, fill=C_RED)
        d.text((x0 + 14, y + 40), f"hp {m['hp']}/{m['max_hp']}   block {m['block']}", font=f14, fill=C_DIM)
        if m["max_hp"]:
            d.rectangle([x0 + 14, y + 62, x0 + 14 + int(350 * max(0, m["hp"]) / m["max_hp"]), y + 68], fill=C_RED)
        y += 88
    p = st["player"]
    d.rounded_rectangle([x0, y, x0 + 380, y + 74], 8, outline=(42, 51, 64), width=2)
    d.text((x0 + 14, y + 10), "YOU (Ironclad)", font=f18, fill=C_GREEN)
    d.text((x0 + 14, y + 40), f"hp {p['hp']}/{p['max_hp']}   block {p['block']}   energy {p['energy']}/3",
           font=f14, fill=C_DIM)
    if p["max_hp"]:
        d.rectangle([x0 + 14, y + 62, x0 + 14 + int(350 * max(0, p["hp"]) / p["max_hp"]), y + 68], fill=C_GREEN)
    y += 96
    for i in range(0, len(st["hand"]), 4):
        row = st["hand"][i:i + 4]
        for j, c in enumerate(row):
            cx = x0 + j * 200
            label = f"{c['name']}{'+' if c['upgraded'] else ''}\ncost {c['cost']}"
            d.rounded_rectangle([cx, y, cx + 188, y + 56], 6, fill=(27, 35, 46),
                                outline=(127, 166, 80) if c["upgraded"] else (51, 64, 79), width=2)
            d.multiline_text((cx + 10, y + 10), label, font=f14, fill=C_TXT)
        y += 66

    img.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match")
    ap.add_argument("--out", default="spire_demo.mp4")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--hold-last", type=float, default=2.5)
    args = ap.parse_args()
    steps = [json.loads(l) for l in open(args.match) if l.strip() and "state" in l]
    summaries = [json.loads(l) for l in open(args.match) if l.strip() and "summary" in l]
    tmp = Path("/tmp/spire_frames")
    tmp.mkdir(exist_ok=True)
    for i, s in enumerate(steps):
        render_frame(s, tmp / f"f{i:04d}.png")
    hold_last = int(args.fps * args.hold_last)
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
           "-i", str(tmp / "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-t", f"{len(steps) / args.fps + args.hold_last}", args.out]
    # simpler: concat last frame holds via duplicate input frame rate trick
    cmd = ["ffmpeg", "-y",
           "-framerate", str(args.fps), "-start_number", "0",
           "-i", str(tmp / "f%04d.png"),
           "-loop", "1", "-framerate", str(args.fps), "-i", str(tmp / f"f{len(steps)-1:04d}.png"),
           "-filter_complex", f"[0:v][1:v]concat=n=2:v=1[out]", "-map", "[out]",
           "-frames:v", f"{len(steps) + hold_last}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", args.out]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"{args.out}: {len(steps)} decisions, "
          f"{sum(1 for s in summaries if s.get('victory'))}/{len(summaries)} victories")


if __name__ == "__main__":
    main()
