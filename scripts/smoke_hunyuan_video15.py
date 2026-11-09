#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-weight HunyuanVideo 1.5 480p T2V smoke; Apple Silicon, 64 GB+ RAM.

Uses the default Hugging Face cache and the same persistent worker as serving.
Downloads ~53 GB if absent. See docs/guides/video-generation.md for model terms.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapid_mlx.video.hunyuan_video15 import REPO  # noqa: E402
from rapid_mlx.video.hunyuan_video15.backend import HunyuanVideo15Engine  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=REPO,
        help="Pinned Hub checkpoint or local Diffusers snapshot",
    )
    parser.add_argument(
        "--prompt",
        default="A red fox walks through a snowy forest, cinematic tracking shot.",
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    engine = HunyuanVideo15Engine(args.model)
    previous = None

    def progress(stage, done, total):
        nonlocal previous
        message = (stage, done, total)
        if message != previous:
            print(f"{stage}: {done}/{total}", flush=True)
            previous = message

    try:
        engine.generate_sync(
            output_path=args.output,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            frames=args.frames,
            fps=24,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            on_progress=progress,
        )
        print(json.dumps(engine.timings, indent=2))
        print(f"Wrote {args.output.resolve()}")
    finally:
        asyncio.run(engine.close())


if __name__ == "__main__":
    main()
