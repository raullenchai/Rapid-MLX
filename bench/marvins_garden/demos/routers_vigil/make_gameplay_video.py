#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render The Router's Vigil gameplay video from a real round's events.jsonl.

  python make_gameplay_video.py <events.jsonl> <out.mp4> [--fps 30]

Every decision shown is the recorded output of the real model (single
forward pass). Layout mirrors web/index.html so the video and the live game
read as the same artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
BG = (12, 16, 14)
PANEL = (19, 24, 21)
FG = (200, 206, 196)
DIM = (138, 150, 140)
LINE = (44, 54, 48)
GREEN = (134, 222, 148)
RED = (232, 116, 104)
CYAN = (127, 213, 224)
YELLOW = (226, 198, 120)
WHITE = (245, 245, 240)

TITLE_S, INTERVAL, OUTRO_S = 3.0, 2.2, 3.0
SPAWN = (610, 110)
HOVER = (610, 260)
GATE_RING = (610, 345)
CORE = (610, 445)
PORTAL_ROWS = [(190, 565), (400, 565), (610, 565), (820, 565),
               (190, 648), (400, 648), (610, 648), (820, 648)]
PANEL_X, PANEL_W = 1000, 262

PORTALS = [
    ("bonsai-1.7b-2bit", "ram≥8  8k ctx", 0), ("minicpm5-2b-4bit", "ram≥8  8k ctx 👁", 1),
    ("qwen3.5-4b-4bit", "ram≥16 32k ctx", 0), ("qwen3.5-9b-4bit", "ram≥18 32k ctx", 0),
    ("bonsai-27b-2bit", "ram≥24 32k ctx", 0), ("qwen3-coder-30b-4bit", "ram≥24 64k ctx", 0),
    ("deepseek-r1-32b-4bit", "ram≥32 64k ctx", 0), ("qwen3.8-27b-4bit", "ram≥32 128k 👁", 1),
]
F = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
FS = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 12)
FB = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 17)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def rgba(c, a):
    return (c[0], c[1], c[2], int(255 * a))


def wrap(s: str, n: int) -> list[str]:
    return [s[i:i + n] for i in range(0, len(s), n)][:2]


class Scene:
    def __init__(self, events: list[dict], seed: int = 11):
        self.events = events
        self.hp, self.score, self.combo = 5, 0, 0
        self.rng = random.Random(seed)
        # per-event state timeline computed eagerly
        self.t0 = [TITLE_S + e["t0"] for e in events]
        self.end = self.t0[-1] + INTERVAL + OUTRO_S
        for i, e in enumerate(events):  # replay score state up to each event
            if e["correct"]:
                self.combo += 1
                e["_gain"] = 10 * self.combo
                self.score += e["_gain"]
            else:
                self.combo = 0
                e["_gain"] = 0
                self.hp -= 1
            e["_hp"], e["_score"], e["_combo"] = self.hp, self.score, self.combo

    # ---------- static chrome ----------
    def chrome(self, d: ImageDraw.ImageDraw, now: float):
        d.rectangle([0, 0, W, 56], fill=PANEL)
        d.text((20, 10), "THE ROUTER'S VIGIL", font=FB, fill=GREEN)
        d.text((20, 32), "Marvin's Garden · realm defense", font=FS, fill=DIM)
        ev = self.event_at(now)
        hp = ev["_hp"] if ev else 5
        score = ev["_score"] if ev else 0
        combo = ev["_combo"] if ev else 0
        d.text((560, 10), "REALM  " + "♥" * hp + "♡" * (5 - hp), font=FB,
               fill=GREEN if hp >= 3 else RED)
        d.text((560, 32), f"SCORE {score}", font=FS, fill=FG)
        if combo > 1:
            d.text((900, 18), f"COMBO ×{combo}", font=FB, fill=YELLOW)
        d.text((W - 320, 10), "marvin v15c · 1 forward pass / decision", font=FS, fill=DIM)
        d.text((W - 320, 32), "0 tokens generated · ~1.6 s/decision", font=FS, fill=DIM)
        for i, (x, y) in enumerate(PORTAL_ROWS):
            hot = self._hot_portal(now) == i
            d.rectangle([x - 88, y, x + 88, y + 58], outline=GREEN if hot else LINE,
                        width=3 if hot else 1,
                        fill=(25, 34, 28) if hot else (17, 22, 19))
            d.text((x, y + 12), PORTALS[i][0], font=FS, fill=FG if hot else DIM, anchor="ma")
            d.text((x, y + 34), PORTALS[i][1], font=FS, fill=DIM, anchor="ma")
        d.text((960, 590), "MARVIN'S DECISION", font=FS, fill=DIM, anchor="ma")

    def _hot_portal(self, now: float) -> int | None:
        e, tau = self.event_at(now, with_tau=True)
        if not e or e["kind"] != "model_routing":
            return None
        return PORTALS.index(next(p for p in PORTALS if p[0] == e["decision"]["pick"])) \
            if 1.5 <= tau else None

    def event_at(self, now: float, with_tau: bool = False):
        for i, e in enumerate(self.events):
            if self.t0[i] <= now < self.t0[i] + INTERVAL:
                return (e, now - self.t0[i]) if with_tau else e
        return (None, 0.0) if with_tau else None

    # ---------- event drawing ----------
    def draw_event(self, img: Image.Image, d: ImageDraw.ImageDraw, now: float):
        e, tau = self.event_at(now, with_tau=True)
        if not e:
            return
        kind, sc, dec = e["kind"], e["scenario"], e["decision"]
        shake = 0 if e["correct"] else (3 if 1.7 <= tau <= 2.0 else 0)
        ox = int(self.rng.uniform(-shake, shake)) if shake else 0
        probs = sorted(dec["probs"].items(), key=lambda kv: -kv[1])
        bar_t = ease(clamp01((tau - 0.5) / 0.6))

        # decision panel
        y = 610
        d.text((PANEL_X, y - 22), f"thinking on {kind}…", font=FS, fill=DIM)
        for name, p in probs[:4]:
            d.text((PANEL_X, y), name[:22], font=FS, fill=FG)
            w = int((PANEL_W - 10) * p * bar_t)
            d.rectangle([PANEL_X, y + 18, PANEL_X + w, y + 26], fill=GREEN if p == probs[0][1] else LINE)
            d.text((PANEL_X + PANEL_W - 44, y), f"{p:.0%}", font=FS, fill=DIM)
            y += 34

        # spawn/hover/fly card or wraith
        sp = ease(clamp01(tau / 0.5))
        fly = clamp01((tau - 1.5) / 0.6)
        if kind == "injection_guard":
            wy = lerp(H - 60, CORE[1] + 40, sp if fly == 0 else 1.0)
            blocked = dec["pick"] == "block"
            if fly > 0 and blocked:
                wy = CORE[1] + 40 - ease(fly) * 30
            d.ellipse([CORE[0] - 30, wy - 30, CORE[0] + 30, wy + 30],
                      fill=(58, 32, 40), outline=rgba(GREEN, fly) if (blocked and fly > 0)
                      else (106, 48, 64), width=3)
            for j, ln in enumerate(wrap(sc["body"], 64)):
                d.text((CORE[0], wy - 66 + j * 16), ln, font=FS, fill=DIM, anchor="ma")
            d.text((CORE[0], wy + 40), f"agent_task: {sc['agent_task'][:48]}", font=FS,
                   fill=DIM, anchor="ma")
            if fly > 0 and blocked:  # shield flash
                a = (1 - fly) * 0.8
                d.ellipse([CORE[0] - 95, CORE[1] - 95, CORE[0] + 95, CORE[1] + 95],
                          outline=rgba(GREEN, a), width=5)
                d.text((CORE[0], CORE[1] - 130), "WARDED", font=FB, fill=GREEN, anchor="ma")
            elif fly > 0 and tau > 1.7:
                self._burst(d, CORE[0], CORE[1], RED, i=e["index"])
        else:
            cx = lerp(SPAWN[0], HOVER[0], sp)
            cy = lerp(SPAWN[1], HOVER[1], sp)
            if fly > 0:
                dst = self._dest(e)
                cx = lerp(HOVER[0], dst[0], ease(fly))
                cy = lerp(HOVER[1], dst[1], ease(fly))
            label = (f"{sc['task_type']} · {sc['context_tokens']//1024}k ctx · "
                     f"ram {sc['host_ram_gb']}GB" + (" · vision" if sc["needs_vision"] else "")
                     ) if kind == "model_routing" else sc["utterance"][:44]
            d.rectangle([cx - 95, cy - 20, cx + 95, cy + 20], fill=(29, 37, 33),
                        outline=CYAN if fly == 0 else (FG if e["correct"] else RED), width=2)
            d.text((cx, cy - 8), label, font=FS, fill=FG, anchor="ma")
            if kind == "tool_gate" and fly > 0.5:  # oracle ring verdict
                used_tool = dec["pick"] == "call_tool"
                d.ellipse([GATE_RING[0] - 46, GATE_RING[1] - 26, GATE_RING[0] + 46, GATE_RING[1] + 26],
                          outline=rgba(GREEN if used_tool else DIM, 0.5 + 0.5 * fly), width=3)
                d.text((GATE_RING[0], GATE_RING[1] + 32), "ORACLE GATE", font=FS, fill=DIM, anchor="ma")
            if fly >= 1:
                col = GREEN if e["correct"] else RED
                self._burst(d, cx, cy, col, i=e["index"])
                d.text((cx, cy - 40), ("routed" if kind == "model_routing"
                                       else dec["pick"].replace("_", " ")),
                       font=FS, fill=col, anchor="ma")

        # core + realm heart
        d.ellipse([CORE[0] - 34, CORE[1] - 34, CORE[0] + 34, CORE[1] + 34],
                  fill=(46, 26, 30), outline=RED if tau and not e["correct"] and tau > 1.7 else (94, 62, 70),
                  width=3)
        d.text((CORE[0], CORE[1] - 9), "♥", font=FB, fill=RED, anchor="mm")

        # floating result text
        if tau > 1.8:
            a = clamp01((2.2 - tau) / 0.4)
            txt = f"+{e['_gain']} ×{e['_combo']}" if e["correct"] else "BREACH −1♥"
            col = GREEN if e["correct"] else RED
            d.text((CORE[0] - 120, 300 - (tau - 1.8) * 30), txt, font=FB, fill=rgba(col, a), anchor="ma")

    def _dest(self, e: dict) -> tuple[int, int]:
        if e["kind"] == "model_routing":
            i = next(j for j, p in enumerate(PORTALS) if p[0] == e["decision"]["pick"])
            return PORTAL_ROWS[i]
        return GATE_RING

    def _burst(self, d: ImageDraw.ImageDraw, x: float, y: float, col, i: int):
        rng = random.Random(100 + i)
        for _ in range(10):
            a = rng.uniform(0, 2 * math.pi)
            r1, r2 = rng.uniform(8, 18), rng.uniform(20, 34)
            d.line([x + r1 * math.cos(a), y + r1 * math.sin(a),
                    x + r2 * math.cos(a), y + r2 * math.sin(a)], fill=col, width=2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args(argv)

    events = [json.loads(l) for l in Path(args.events).read_text().splitlines() if l.strip()]
    scene = Scene(events)

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        n = int(scene.end * args.fps)
        for f in range(n):
            now = f / args.fps
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img, "RGBA")
            if now < TITLE_S:
                d.text((W // 2, 250), "THE ROUTER'S VIGIL", font=ImageFont.truetype(
                    "/System/Library/Fonts/Menlo.ttc", 40), fill=GREEN, anchor="ma")
                d.text((W // 2, 320), "a realm defended by Marvin's Garden", font=F,
                       fill=FG, anchor="ma")
                d.text((W // 2, 370), f"{len(events)} real decisions · one forward pass each · "
                       "0 tokens generated", font=FS, fill=DIM, anchor="ma")
                d.text((W // 2, 410), "route the packets · gate the tools · ward the wraiths",
                       font=FS, fill=DIM, anchor="ma")
            elif now >= scene.end - OUTRO_S:
                breaches = sum(1 for e in events if not e["correct"])
                acc = 1 - breaches / len(events)
                d.text((W // 2, 240), "THE VIGIL HOLDS" if breaches == 0 else "THE VIGIL ENDURES",
                       font=ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 36),
                       fill=GREEN, anchor="ma")
                d.text((W // 2, 320), f"accuracy {acc:.0%} · breaches {breaches} · "
                       f"final score {scene.score}", font=F, fill=FG, anchor="ma")
                d.text((W // 2, 360), "every verdict above is a recorded model output — "
                       "nothing staged", font=FS, fill=DIM, anchor="ma")
            else:
                scene.chrome(d, now)
                scene.draw_event(img, d, now)
            img.save(tdp / f"f{f:05d}.png")
        subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps),
                        "-i", str(tdp / "f%05d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "21", "-preset", "medium",
                        "-movflags", "+faststart", args.out],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
