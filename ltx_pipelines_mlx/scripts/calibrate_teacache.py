"""Calibrate TeaCache polyfit coefficients for LTX-2 stage 1.

Runs N reference generations through ``TI2VidTwoStagesPipeline.generate_two_stage``
with caching disabled, captures per-step (gate_signal, block_residual) for
the conditioned pass, computes input/output L1 deltas in fp32, and fits a
degree-4 polynomial mapping input delta → output delta. Writes the
coefficients (and a copy-pasteable preset snippet) to disk.

Memory: rolling per-step state only (no per-step buffering of full tensors).
Suitable for 32 GB hosts running the q8 model + Gemma text encoder.

Usage:
    python -m ltx_pipelines_mlx.scripts.calibrate_teacache \\
        --model-dir dgrauet/ltx-2.3-mlx-q8 \\
        --prompts prompts.txt \\
        --num-steps 30 \\
        --out coefficients.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines_mlx.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline


def _rel_l1_fp32(curr: mx.array, prev: mx.array) -> float:
    """Relative L1 distance with fp32 reductions: mean|curr-prev| / mean|prev|."""
    curr32 = curr.astype(mx.float32)
    prev32 = prev.astype(mx.float32)
    diff = mx.mean(mx.abs(curr32 - prev32))
    base = mx.mean(mx.abs(prev32))
    return float((diff / base).item())


class _StreamingCalibrator:
    """Rolls (prev_modulated, prev_residual) per stream and accumulates deltas."""

    def __init__(self):
        self._prev_gate: mx.array | None = None
        self._prev_v_res: mx.array | None = None
        self._prev_a_res: mx.array | None = None
        self.delta_in: list[float] = []
        self.delta_res_video: list[float] = []
        self.delta_res_audio: list[float] = []

    def __call__(self, step_idx: int, gate: mx.array, v_res: mx.array, a_res: mx.array) -> None:
        if self._prev_gate is not None:
            self.delta_in.append(_rel_l1_fp32(gate, self._prev_gate))
            self.delta_res_video.append(_rel_l1_fp32(v_res, self._prev_v_res))
            self.delta_res_audio.append(_rel_l1_fp32(a_res, self._prev_a_res))
        self._prev_gate = gate
        self._prev_v_res = v_res
        self._prev_a_res = a_res

    def reset_for_next_prompt(self) -> None:
        self._prev_gate = None
        self._prev_v_res = None
        self._prev_a_res = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, help="HF repo or local dir, e.g. dgrauet/ltx-2.3-mlx-q8")
    p.add_argument("--prompts", required=True, type=Path, help="Path to a text file, one prompt per line")
    p.add_argument("--num-steps", type=int, default=30, help="Stage 1 denoising steps (default 30)")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=704)
    p.add_argument("--num-frames", type=int, default=97)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--stg-scale", type=float, default=0.0)
    p.add_argument("--polyfit-degree", type=int, default=4)
    p.add_argument(
        "--default-rel-l1-thresh",
        type=float,
        default=0.15,
        help="Starting threshold to ship with the preset (user can tune).",
    )
    p.add_argument("--out", type=Path, default=Path("ltx2_teacache_calibration.json"))
    p.add_argument(
        "--hq",
        action="store_true",
        help=(
            "Calibrate the HQ res_2s path (TI2VidTwoStagesHQPipeline) instead of the "
            "Euler path. res_2s has different per-step dynamics so its "
            "coefficients are not interchangeable with the Euler ones."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.prompts.exists():
        print(f"prompts file not found: {args.prompts}", file=sys.stderr)
        return 2

    prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    if not prompts:
        print("prompts file is empty", file=sys.stderr)
        return 2

    pipeline_label = "HQ res_2s" if args.hq else "Euler"
    print(f"Calibrating TeaCache for LTX-2 stage 1 ({pipeline_label}): {len(prompts)} prompts x {args.num_steps} steps")

    pipeline_class = TI2VidTwoStagesHQPipeline if args.hq else TI2VidTwoStagesPipeline
    pipeline = pipeline_class(model_dir=args.model_dir)
    calibrator = _StreamingCalibrator()

    for i, prompt in enumerate(prompts):
        print(f"[{i + 1}/{len(prompts)}] {prompt[:80]}…")
        calibrator.reset_for_next_prompt()

        # tap fires on each stage 1 step's conditioned pass via
        # guided_denoise_loop. Stage 2 uses denoise_loop (no tap),
        # so its 3 steps don't contribute deltas — the calibrator only
        # observes stage 1.
        pipeline.generate_two_stage(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed + i,
            stage1_steps=args.num_steps,
            stage2_steps=3,
            cfg_scale=args.cfg_scale,
            stg_scale=args.stg_scale,
            tap=calibrator,
        )

    if not calibrator.delta_in:
        print("No deltas captured — did the tap fire? See plan follow-up.", file=sys.stderr)
        return 1

    coeffs = np.polyfit(calibrator.delta_in, calibrator.delta_res_video, deg=args.polyfit_degree).tolist()

    out_payload = {
        "name": "ltx2",
        "coefficients": coeffs,
        "rel_l1_thresh": args.default_rel_l1_thresh,
        "deltas": {
            "delta_in": calibrator.delta_in,
            "delta_res_video": calibrator.delta_res_video,
            "delta_res_audio": calibrator.delta_res_audio,
        },
        "calibration_meta": {
            "num_prompts": len(prompts),
            "num_steps": args.num_steps,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "model_dir": args.model_dir,
            "polyfit_degree": args.polyfit_degree,
            "delta_in_count": len(calibrator.delta_in),
        },
    }
    args.out.write_text(json.dumps(out_payload, indent=2))

    print(f"\n  Wrote {args.out}")
    print(
        "\nThe inline polyfit at degree "
        f"{args.polyfit_degree} produced these coefficients (may be unstable — "
        "use scripts/fit_teacache_poly.py for a robust fit that picks the "
        "lowest stable degree from the saved deltas):\n"
    )
    print("LTX2_TEACACHE_COEFFICIENTS = [")
    for c in coeffs:
        print(f"    {c!r},")
    print("]")
    print(f"LTX2_TEACACHE_THRESH = {args.default_rel_l1_thresh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
