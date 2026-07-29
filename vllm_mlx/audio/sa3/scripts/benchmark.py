#!/usr/bin/env python3
"""Benchmark — wall time + peak RAM for SA3 across model sizes and clip lengths.

Runs each (DiT, decoder, seconds) cell as a fresh ./sa3 subprocess so peak
RAM is isolated per run. Weights are pre-warmed via ensure_local() before
timing starts, so HF download time is NOT charged against the benchmark.

Usage:
    uv run --no-project python scripts/benchmark.py
    # or
    ./sa3 --help    # see the regular CLI; this file is separate
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from weights import ensure_local  # noqa: E402

# ── what to benchmark ──────────────────────────────────────────────────────
CONFIGS = [
    ("sm-music", "same-s"),
    ("medium",   "same-l"),
]
SECONDS = [5, 30, 120, 380]
PROMPT = "Impending tribal, epic orchestral buildup"
SEED = 42

# Weights we need on disk before benchmarks start. T5Gemma is shared.
PREFLIGHT_WEIGHTS = [
    "models/mlx/t5gemma_f16.npz",
    "models/mlx/dit_sm-music_f16.npz",
    "models/mlx/dit_medium_f16.npz",
    "models/mlx/same_s_decoder_f32.npz",
    "models/mlx/same_l_decoder_f32.npz",
]


# ── hardware info ──────────────────────────────────────────────────────────
def hwinfo() -> str:
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        chip = platform.processor() or platform.machine() or "unknown"
    try:
        mem_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True).strip())
        mem_gb = f"{mem_bytes / (1024**3):.0f} GB"
    except Exception:
        mem_gb = "?"
    return f"{chip}  ·  {mem_gb} RAM"


# ── output parsing ─────────────────────────────────────────────────────────
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_DONE = re.compile(
    r"done\s+([\d.]+)s\s+wall\s+→\s+([\d.]+)s\s+audio\s+→\s+"
    r"([\d.]+)×\s+realtime\s+peak\s+RAM\s+([\d.]+)\s+(MB|GB)"
)

def parse_done(stdout: str) -> dict | None:
    plain = _ANSI.sub("", stdout)
    m = _DONE.search(plain)
    if not m:
        return None
    return {
        "wall_s":   float(m.group(1)),
        "audio_s":  float(m.group(2)),
        "realtime": float(m.group(3)),
        "peak_ram": f"{m.group(4)} {m.group(5)}",
    }


# ── single benchmark cell ──────────────────────────────────────────────────
def run_one(dit: str, decoder: str, seconds: int) -> dict | None:
    out = Path(tempfile.gettempdir()) / f"bench_{dit}_{decoder}_{seconds}s.wav"
    cmd = [
        str(PROJECT_DIR / "sa3"),
        "--prompt", PROMPT,
        "--dit", dit,
        "--decoder", decoder,
        "--seconds", str(seconds),
        "--seed", str(SEED),
        "--out", str(out),
    ]
    label = f"  {dit:9s} {decoder:7s} {seconds:>4}s"
    print(f"{label}  →  running …", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(PROJECT_DIR),
            timeout=20 * 60,  # 20-minute hard cap per cell
        )
    except subprocess.TimeoutExpired:
        print(f"{label}  →  TIMEOUT after {time.time()-t0:.0f}s")
        return None
    elapsed = time.time() - t0
    out.unlink(missing_ok=True)

    if proc.returncode != 0:
        last_err = (proc.stderr.strip().split("\n") or ["(no stderr)"])[-1][:120]
        print(f"{label}  →  FAILED  exit={proc.returncode}  {last_err}")
        return None
    parsed = parse_done(proc.stdout)
    if not parsed:
        print(f"{label}  →  FAILED  (could not parse 'done' line after {elapsed:.0f}s)")
        return None
    print(
        f"{label}  →  {parsed['wall_s']:7.2f}s wall   "
        f"{parsed['realtime']:5.2f}× RT   peak {parsed['peak_ram']}"
    )
    return {"dit": dit, "decoder": decoder, "seconds": seconds, **parsed}


# ── ASCII table ────────────────────────────────────────────────────────────
def render_table(rows: list[dict]) -> str:
    headers = ["model", "decoder", "seconds", "wall (s)", "×realtime", "peak RAM"]
    widths  = [9, 7, 7, 9, 9, 10]

    def hline(l, m, r):
        return l + m.join("─" * (w + 2) for w in widths) + r
    def row(cells):
        parts = []
        for i, c in enumerate(cells):
            # right-align numeric columns (idx 2-5)
            parts.append(c.rjust(widths[i]) if i >= 2 else c.ljust(widths[i]))
        return "│ " + " │ ".join(parts) + " │"

    lines = [hline("┌", "┬", "┐"), row(headers), hline("├", "┼", "┤")]
    prev_dit = None
    for r in rows:
        # Visual separator when the model changes
        if prev_dit is not None and r["dit"] != prev_dit:
            lines.append(hline("├", "┼", "┤"))
        prev_dit = r["dit"]
        lines.append(row([
            r["dit"], r["decoder"], str(r["seconds"]),
            f"{r['wall_s']:.2f}",
            f"{r['realtime']:.2f}×",
            r["peak_ram"],
        ]))
    lines.append(hline("└", "┴", "┘"))
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────
def main() -> int:
    print()
    print("━" * 72)
    print("  Benchmark — Stable Audio 3 (MLX) on Apple Silicon")
    print("━" * 72)
    print(f"  Hardware:  {hwinfo()}")
    print(f"  Prompt:    {PROMPT!r}")
    print(f"  Seed:      {SEED}")
    print(f"  Cells:     {len(CONFIGS)} configs × {len(SECONDS)} durations = "
          f"{len(CONFIGS) * len(SECONDS)} runs")
    print()

    print("→ Preflight: ensuring all weights are local (downloads NOT timed)")
    for rel in PREFLIGHT_WEIGHTS:
        ensure_local(rel)
    print("  done.\n")

    print("→ Running benchmarks (each subprocess is isolated for clean peak-RAM)")
    rows: list[dict] = []
    for dit, decoder in CONFIGS:
        for s in SECONDS:
            r = run_one(dit, decoder, s)
            if r:
                rows.append(r)
    print()

    if not rows:
        print("No successful runs — nothing to tabulate.")
        return 1

    print("━" * 72)
    print(f"  Results  ({len(rows)}/{len(CONFIGS)*len(SECONDS)} cells succeeded)")
    print(f"  Hardware: {hwinfo()}")
    print("━" * 72)
    print()
    print(render_table(rows))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
