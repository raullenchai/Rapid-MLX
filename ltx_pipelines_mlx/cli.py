"""Command-line interface for ltx-2-mlx.

Usage:
    ltx-2-mlx generate --prompt "a cat walking" --output out.mp4
    ltx-2-mlx generate --prompt "animate this" --image photo.jpg -o anim.mp4
    ltx-2-mlx generate --prompt "a scene" --two-stage -o hires.mp4
    ltx-2-mlx generate --prompt "a scene" --two-stages-hq --stage1-steps 20 -o hq.mp4
    ltx-2-mlx a2v --prompt "music video" --audio music.wav -o a2v.mp4
    ltx-2-mlx retake --prompt "new scene" --video source.mp4 --start 1 --end 3 -o retake.mp4
    ltx-2-mlx extend --prompt "continue" --video source.mp4 --extend-frames 2 -o extended.mp4
    ltx-2-mlx keyframe --prompt "transition" --start img1.png --end img2.png -o kf.mp4
    ltx-2-mlx ic-lora --prompt "scene" --lora lora.safetensors 1.0 --video-conditioning depth.mp4 1.0 -o out.mp4
    ltx-2-mlx enhance --prompt "a cat walking" --mode t2v
    ltx-2-mlx info --model dgrauet/ltx-2.3-mlx-q8
    ltx-2-mlx train --config training_config.yaml
    ltx-2-mlx preprocess --videos ./my_videos --model dgrauet/ltx-2.3-mlx-q8 -o ./preprocessed
"""

from __future__ import annotations

import argparse
import sys
import time

DEFAULT_MODEL = "dgrauet/ltx-2.3-mlx-q8"
DEFAULT_GEMMA = "mlx-community/gemma-3-12b-it-4bit"

# Distilled LoRA filenames per checkpoint version. LTX-2.5 ships
# ``ltx-2.5-22b-distilled-lora-450.safetensors`` in its model dir; the 2.3
# port uses ``ltx-2.3-22b-distilled-lora-384.safetensors``. The dev
# transformer name (``transformer-dev.safetensors``) is identical in both.
DISTILLED_LORA_2_3 = "ltx-2.3-22b-distilled-lora-384.safetensors"
DISTILLED_LORA_2_5 = "ltx-2.5-22b-distilled-lora-450.safetensors"


def _default_distilled_lora(model_dir: str) -> str:
    """Return the distilled LoRA filename matching the checkpoint version.

    Uses :func:`ltx_pipelines_mlx.utils._orchestration.detect_model_version`
    on the (local or resolved) model dir: ``2.5.x`` picks the 450-step LoRA,
    anything else keeps the 2.3 name. Unknown/unreadable dirs fall back to
    the 2.3 default, preserving legacy behaviour.
    """
    from ltx_pipelines_mlx.utils._orchestration import detect_model_version, resolve_model_dir

    # ``model_dir`` may still be a Hugging Face repo ID at CLI argument
    # resolution time. Resolve it before reading the version marker; otherwise
    # every remote checkpoint falls back to the legacy 2.3 LoRA filename.
    version = detect_model_version(resolve_model_dir(model_dir))
    if version.startswith("2.5"):
        return DISTILLED_LORA_2_5
    return DISTILLED_LORA_2_3


def _add_base_args(parser: argparse.ArgumentParser) -> None:
    """Add base arguments shared by all subcommands (prompt, output, model, seed)."""
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt")
    parser.add_argument("--output", "-o", required=True, help="Output video path (.mp4)")
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL, help=f"Model weights (HF repo or path, default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--gemma", default=DEFAULT_GEMMA, help=f"Gemma model for text encoding (default: {DEFAULT_GEMMA})"
    )
    parser.add_argument("--seed", "-s", type=int, default=-1, help="Random seed (-1 = random)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")


def _build_tile_count_config(args: argparse.Namespace):
    """Return a TileCountConfig from --tile-frames / --tile-spatial / --tile-overlap.

    Returns None if both frames and spatial are 1 (no tiling).
    """
    frames_n = getattr(args, "tile_frames", 1)
    spatial_n = getattr(args, "tile_spatial", 1)
    overlap = getattr(args, "tile_overlap", 2)
    if frames_n <= 1 and spatial_n <= 1:
        return None
    from ltx_core_mlx.model.video_vae.tiling import DimensionTilingConfig, TileCountConfig

    return TileCountConfig(
        frames=DimensionTilingConfig(num_tiles=frames_n, overlap=overlap if frames_n > 1 else 0),
        height=DimensionTilingConfig(num_tiles=spatial_n, overlap=overlap if spatial_n > 1 else 0),
        width=DimensionTilingConfig(num_tiles=spatial_n, overlap=overlap if spatial_n > 1 else 0),
    )


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    """Add generation-specific arguments (dimensions, steps) on top of base args."""
    _add_base_args(parser)
    parser.add_argument("--height", "-H", type=int, default=480, help="Video height (default: 480)")
    parser.add_argument("--width", "-W", type=int, default=704, help="Video width (default: 704)")
    parser.add_argument("--frames", "-f", type=int, default=97, help="Number of frames (default: 97)")
    parser.add_argument(
        "--frame-rate",
        type=float,
        required=True,
        help=(
            "Output frame rate (mandatory; matches upstream `frame_rate=` on every pipeline). "
            "LTX-2.3 was trained at 24; values far from that drift out of distribution."
        ),
    )
    parser.add_argument(
        "--tile-frames",
        type=int,
        default=1,
        help=(
            "Number of temporal tiles for modality tiling (default: 1 = no tiling). "
            "Each tile is denoised independently and blended back. Trades wall-clock "
            "for peak memory. Combine with --low-ram for max savings."
        ),
    )
    parser.add_argument(
        "--tile-spatial",
        type=int,
        default=1,
        help=(
            "Number of spatial tiles per axis (height and width). 2 = 2x2 = 4 spatial "
            "tiles. Combined with --tile-frames N gives N*S*S tiles total. Default: 1."
        ),
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=2,
        help=(
            "Token-grid overlap between adjacent tiles (default: 2). Higher overlap "
            "= smoother blend but more redundant compute. Ignored when both "
            "--tile-frames and --tile-spatial are 1."
        ),
    )
    parser.add_argument(
        "--low-ram",
        action="store_true",
        help=(
            "Stream transformer blocks from mmap'd safetensors via "
            "mx.compile + per-block sync + Metal heap release. Cuts "
            "transformer peak Metal ~75%% (e.g. q8 ~10-12 GB -> ~2.8 GB). "
            "Targets 16 GB Macs (q8) and 32 GB Macs (bf16). Supported "
            "on generate (one-stage / --two-stage / --two-stages-hq), a2v, "
            "keyframe, and ic-lora. Compatible with generate's --lora flag "
            "via per-block BlockLoraSource bind-time fusion. Two-stage at "
            "LoRA strength 1.0 swaps to the pre-fused "
            "transformer-distilled.safetensors at the stage 1->2 transition; "
            "custom strengths trigger bind-time LoRA fusion (slower but "
            "supports any strength)."
        ),
    )


def main() -> None:
    """Entry point for the ltx-2-mlx CLI."""
    parser = argparse.ArgumentParser(
        prog="ltx-2-mlx",
        description="LTX-2.3 video generation on Apple Silicon (MLX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  ltx-2-mlx generate --prompt "a sunset" --output sunset.mp4
  ltx-2-mlx generate --prompt "animate" --image photo.jpg -o anim.mp4
  ltx-2-mlx generate --prompt "a scene" --two-stage -o hires.mp4
  ltx-2-mlx a2v --prompt "music video" --audio music.wav -o a2v.mp4
  ltx-2-mlx retake --prompt "new scene" --video source.mp4 --start 1 --end 3 -o out.mp4
  ltx-2-mlx extend --prompt "continue" --video source.mp4 --extend-frames 2 -o out.mp4
  ltx-2-mlx keyframe --prompt "transition" --start img1.png --end img2.png -o out.mp4
  ltx-2-mlx ic-lora --prompt "scene" --lora lora.safetensors 1.0 --video-conditioning depth.mp4 1.0 -o out.mp4
  ltx-2-mlx enhance --prompt "a cat walking" --mode t2v
  ltx-2-mlx info --model dgrauet/ltx-2.3-mlx-q4
""",
    )
    sub = parser.add_subparsers(dest="command")

    # --- generate (T2V / I2V / two-stage / HQ) ---
    gen = sub.add_parser("generate", help="Generate video from text (T2V) or image (I2V)")
    _add_generation_args(gen)
    from ltx_pipelines_mlx.utils.args import ImageAction as _ImageAction

    gen.add_argument(
        "--image",
        "-i",
        action=_ImageAction,
        nargs="+",
        dest="images",
        default=None,
        metavar="ARG",
        help=(
            "Reference image for I2V. Form: PATH [FRAME_IDX STRENGTH [CRF]]. "
            "PATH alone defaults to FRAME_IDX=0 STRENGTH=1.0 (legacy single-image I2V). "
            "Repeatable to anchor multiple frames (e.g. --image foo.jpg 0 1.0 "
            "--image foo.jpg 96 1.0 to anchor both ends and preserve identity). "
            "FRAME_IDX is the pixel frame index (0-based); 0 replaces the first latent "
            "frame directly, others guide via VideoConditionByKeyframeIndex. "
            "Mirrors upstream LTX_2_3 --image."
        ),
    )
    gen.add_argument("--steps", type=int, default=None, help="Denoising steps for one-stage (default: 8)")
    gen.add_argument(
        "--two-stage",
        action="store_true",
        help="Two-stage pipeline: dev model + CFG at half-res, upscale, distilled LoRA refine (requires q8 model)",
    )
    gen.add_argument(
        "--two-stages-hq",
        action="store_true",
        dest="two_stages_hq",
        help="HQ two-stage pipeline (res_2s sampler for stage 1). Mirrors upstream TI2VidTwoStagesHQPipeline.",
    )
    gen.add_argument(
        "--distilled",
        action="store_true",
        help="Distilled two-stage pipeline (half-res distilled + upscale + distilled refine, no CFG). Mirrors upstream DistilledPipeline.",
    )
    gen.add_argument(
        "--one-stage",
        action="store_true",
        dest="one_stage",
        help="Dev model + CFG one-stage at full target resolution (no upsampler, no stage 2). Mirrors upstream TI2VidOneStagePipeline. Higher quality than --distilled at small resolutions; slower than --two-stage at large.",
    )
    gen.add_argument("--stage1-steps", type=int, default=None, help="Stage 1 steps (default: 30 standard, 15 HQ)")
    gen.add_argument("--stage2-steps", type=int, default=None, help="Stage 2 steps (default: 3)")
    gen.add_argument("--cfg-scale", type=float, default=None, help="CFG guidance scale (default: 3.0)")
    gen.add_argument(
        "--stg-scale",
        type=float,
        default=None,
        help="STG guidance scale (default: 1.0 standard, 0.0 HQ — upstream LTX_2_3_PARAMS)",
    )
    gen.add_argument(
        "--dev-transformer",
        default="transformer-dev.safetensors",
        help="Dev transformer filename (default: transformer-dev.safetensors)",
    )
    gen.add_argument(
        "--distilled-lora",
        default=None,
        help=(
            "Distilled LoRA filename for stage 2 (default: version-aware — "
            "ltx-2.5-22b-distilled-lora-450.safetensors for LTX-2.5 checkpoints, "
            "ltx-2.3-22b-distilled-lora-384.safetensors for LTX-2.3)"
        ),
    )
    gen.add_argument(
        "--distilled-lora-strength", type=float, default=1.0, help="Distilled LoRA strength for stage 2 (default: 1.0)"
    )
    gen.add_argument(
        "--enable-teacache",
        action="store_true",
        help=(
            "Two-stage (--two-stage / --two-stages-hq) only: enable TeaCache stage-1 "
            "acceleration (opt-in, ~1.46x speedup on Euler at default thresh; "
            "see CLAUDE.md)"
        ),
    )
    gen.add_argument(
        "--teacache-thresh",
        type=float,
        default=None,
        help=(
            "Override TeaCache rel_l1_thresh (default 0.5; higher = more skipping = "
            "faster but lossier). Ignored unless --enable-teacache is set."
        ),
    )
    gen.add_argument("--enhance-prompt", action="store_true", help="Enhance prompt using Gemma before generation")
    gen.add_argument(
        "--lora",
        action="append",
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        help=(
            "LoRA weights and strength (repeatable). PATH can be a local .safetensors file "
            "or a HuggingFace repo ID. Example: --lora my_lora.safetensors 1.0"
        ),
    )

    # --- a2v (Audio-to-Video) ---
    a2v = sub.add_parser("a2v", help="[beta] Generate video from audio + text prompt")
    _add_generation_args(a2v)
    a2v.add_argument("--audio", "-a", required=True, help="Input audio file (WAV/MP3/etc.)")
    a2v.add_argument("--audio-start", type=float, default=0.0, help="Audio start time in seconds (default: 0)")
    a2v.add_argument(
        "--one-stage",
        action="store_true",
        help=(
            "Experimental single-stage A2V at target resolution. Best for low-resolution drafts; "
            "the standard two-stage path remains the quality/default path."
        ),
    )
    a2v.add_argument("--steps", type=int, default=None, help="One-stage denoising steps (default: 16)")
    a2v.add_argument("--stage1-steps", type=int, default=None, help="Stage 1 steps (default: 30)")
    a2v.add_argument("--stage2-steps", type=int, default=None, help="Stage 2 steps (default: 3)")
    a2v.add_argument("--cfg-scale", type=float, default=None, help="CFG guidance scale (default: 3.0)")
    a2v.add_argument(
        "--stg-scale", type=float, default=None, help="STG guidance scale (default: 1.0 — upstream LTX_2_3_PARAMS)"
    )
    a2v.add_argument(
        "--distilled-lora",
        default=None,
        help=(
            "Distilled LoRA filename for stage 2 (default: version-aware — "
            "ltx-2.5-22b-distilled-lora-450.safetensors for LTX-2.5 checkpoints, "
            "ltx-2.3-22b-distilled-lora-384.safetensors for LTX-2.3)"
        ),
    )
    a2v.add_argument(
        "--distilled-lora-strength", type=float, default=1.0, help="Distilled LoRA strength for stage 2 (default: 1.0)"
    )
    a2v.add_argument(
        "--image",
        "-i",
        action=_ImageAction,
        nargs="+",
        dest="images",
        default=None,
        metavar="ARG",
        help=(
            "Reference image for I2V. Form: PATH [FRAME_IDX STRENGTH [CRF]]. "
            "Repeatable to anchor multiple frames. PATH alone defaults to "
            "FRAME_IDX=0 STRENGTH=1.0 (legacy single-image)."
        ),
    )

    # --- retake ---
    ret = sub.add_parser("retake", help="[beta] Regenerate a time segment of an existing video")
    _add_base_args(ret)
    ret.add_argument("--video", "-v", required=True, help="Source video file")
    ret.add_argument("--start", type=int, required=True, help="Start latent frame index (inclusive)")
    ret.add_argument("--end", type=int, required=True, help="End latent frame index (exclusive)")
    ret.add_argument("--steps", type=int, default=None, help="Denoising steps (default: 30)")
    ret.add_argument("--cfg-scale", type=float, default=None, help="CFG guidance scale (default: 3.0)")
    ret.add_argument(
        "--stg-scale", type=float, default=None, help="STG guidance scale (default: 1.0 — upstream LTX_2_3_PARAMS)"
    )
    ret.add_argument("--no-regen-audio", action="store_true", help="Preserve original audio (don't regenerate)")

    # --- extend ---
    ext = sub.add_parser("extend", help="[beta] Add frames before or after an existing video")
    _add_base_args(ext)
    ext.add_argument("--video", "-v", required=True, help="Source video file")
    ext.add_argument("--extend-frames", type=int, required=True, help="Number of latent frames to add")
    ext.add_argument("--direction", choices=["before", "after"], default="after", help="Direction (default: after)")
    ext.add_argument("--steps", type=int, default=None, help="Denoising steps (default: 30)")
    ext.add_argument("--cfg-scale", type=float, default=None, help="CFG guidance scale (default: 3.0)")
    ext.add_argument(
        "--stg-scale", type=float, default=None, help="STG guidance scale (default: 1.0 — upstream LTX_2_3_PARAMS)"
    )

    # --- keyframe ---
    kf = sub.add_parser("keyframe", help="Interpolate between keyframe images")
    _add_generation_args(kf)
    kf.add_argument("--start", required=True, help="Start keyframe image path")
    kf.add_argument("--end", required=True, help="End keyframe image path")
    kf.add_argument(
        "--start-strength",
        type=float,
        default=1.0,
        help="Start keyframe conditioning strength in [0, 1] (default: 1.0 — upstream-iso)",
    )
    kf.add_argument(
        "--end-strength",
        type=float,
        default=1.0,
        help="End keyframe conditioning strength in [0, 1] (default: 1.0 — upstream-iso). Lower values give the prompt more freedom to diverge from the end fixture.",
    )
    kf.add_argument("--stage1-steps", type=int, default=None, help="Stage 1 denoising steps")
    kf.add_argument("--stage2-steps", type=int, default=None, help="Stage 2 denoising steps")
    kf.add_argument("--cfg-scale", type=float, default=None, help="Override CFG scale (default: 3.0 video, 7.0 audio)")
    kf.add_argument("--stg-scale", type=float, default=None, help="Override STG scale (default: 1.0)")
    kf.add_argument(
        "--dev-transformer",
        default="transformer-dev.safetensors",
        help="Dev transformer filename (default: transformer-dev.safetensors)",
    )
    kf.add_argument(
        "--distilled-lora",
        default=None,
        help=(
            "Distilled LoRA filename for stage 2 refinement (default: version-aware — "
            "ltx-2.5-22b-distilled-lora-450.safetensors for LTX-2.5 checkpoints, "
            "ltx-2.3-22b-distilled-lora-384.safetensors for LTX-2.3)"
        ),
    )
    kf.add_argument("--lora-strength", type=float, default=1.0, help="Distilled LoRA strength (default: 1.0)")

    # --- ic-lora ---
    ic = sub.add_parser("ic-lora", help="Generate video with IC-LoRA control conditioning")
    _add_generation_args(ic)
    ic.add_argument(
        "--lora",
        action="append",
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=True,
        help=(
            "IC-LoRA weights and strength (repeatable). PATH can be a local .safetensors file "
            "or a HuggingFace repo ID (e.g. Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control). "
            "Example: --lora Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control 1.0"
        ),
    )
    ic.add_argument(
        "--video-conditioning",
        action="append",
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=True,
        help="Reference control video and strength (repeatable). Example: --video-conditioning depth.mp4 1.0",
    )
    ic.add_argument(
        "--image",
        "-i",
        action=_ImageAction,
        nargs="+",
        dest="images",
        default=None,
        metavar="ARG",
        help=("Reference image for I2V. Form: PATH [FRAME_IDX STRENGTH [CRF]]. Repeatable to anchor multiple frames."),
    )
    ic.add_argument("--stage1-steps", type=int, default=None, help="Stage 1 denoising steps")
    ic.add_argument("--stage2-steps", type=int, default=None, help="Stage 2 denoising steps")
    ic.add_argument(
        "--conditioning-strength",
        type=float,
        default=1.0,
        help="IC-LoRA conditioning attention strength 0.0-1.0 (default: 1.0)",
    )
    ic.add_argument("--skip-stage-2", action="store_true", help="Skip stage 2 upsampling (half resolution output)")

    # --- lipdub ---
    ld = sub.add_parser(
        "lipdub",
        help="[experimental] Lip-dub a reference video: re-generate visuals matching the reference audio.",
    )
    _add_base_args(ld)
    ld.add_argument("--height", "-H", type=int, default=480, help="Output height (default: 480)")
    ld.add_argument("--width", "-W", type=int, default=704, help="Output width (default: 704)")
    ld.add_argument(
        "--low-ram",
        action="store_true",
        help="Stream transformer blocks (same semantics as ic-lora --low-ram).",
    )
    ld.add_argument(
        "--reference-video",
        required=True,
        help="Reference video (provides visuals + audio for the lip-dub).",
    )
    ld.add_argument(
        "--reference-strength",
        type=float,
        default=1.0,
        help="Reference video conditioning strength (default: 1.0).",
    )
    ld.add_argument(
        "--lora",
        action="append",
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=True,
        help="Lip-dub IC-LoRA (exactly one required). PATH can be local or HF repo ID.",
    )
    ld.add_argument("--stage1-steps", type=int, default=None)
    ld.add_argument("--stage2-steps", type=int, default=None)
    ld.add_argument("--image", action="append", help="I2V anchor image path (repeatable, combined with --image-frame/--image-strength)")
    ld.add_argument("--image-frame", action="append", type=int, default=[], help="Frame index for each --image (default: 0)")
    ld.add_argument("--image-strength", action="append", type=float, default=[], help="Strength for each --image (default: 1.0)")

    # --- hdr-ic-lora ---
    hdr = sub.add_parser(
        "hdr-ic-lora",
        help="Generate HDR video via IC-LoRA with LogC3 inverse (saves SDR mp4 + .hdr.npz)",
    )
    _add_generation_args(hdr)
    hdr.add_argument(
        "--lora",
        action="append",
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=True,
        help=(
            "HDR IC-LoRA weights and strength (repeatable). PATH can be local .safetensors "
            "or a HuggingFace repo ID (e.g. Lightricks/LTX-2.3-22b-IC-LoRA-HDR). "
            "Auto-detects HDR transform from safetensors metadata."
        ),
    )
    hdr.add_argument(
        "--video-conditioning",
        action="append",
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        default=None,
        help="Optional reference control video(s) and strength (repeatable). Pure T2V HDR if omitted.",
    )
    hdr.add_argument(
        "--image",
        "-i",
        action=_ImageAction,
        nargs="+",
        dest="images",
        default=None,
        metavar="ARG",
        help=("Reference image for I2V. Form: PATH [FRAME_IDX STRENGTH [CRF]]. Repeatable to anchor multiple frames."),
    )
    hdr.add_argument("--stage1-steps", type=int, default=None, help="Stage 1 denoising steps")
    hdr.add_argument("--stage2-steps", type=int, default=None, help="Stage 2 denoising steps")
    hdr.add_argument(
        "--conditioning-strength",
        type=float,
        default=1.0,
        help="IC-LoRA conditioning attention strength 0.0-1.0 (default: 1.0)",
    )
    hdr.add_argument("--skip-stage-2", action="store_true", help="Skip stage 2 upsampling (half resolution output)")

    # --- enhance ---
    enh = sub.add_parser("enhance", help="Enhance a prompt using Gemma (no video generation)")
    enh.add_argument("--prompt", "-p", required=True, help="Prompt to enhance")
    enh.add_argument("--mode", choices=["t2v", "i2v"], default="t2v", help="Prompt mode (default: t2v)")
    enh.add_argument("--gemma", default=DEFAULT_GEMMA, help=f"Gemma model (default: {DEFAULT_GEMMA})")
    enh.add_argument("--seed", "-s", type=int, default=10, help="Random seed (default: 10)")

    # --- info ---
    info = sub.add_parser("info", help="Show model info and memory estimate")
    info.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model weights (HF repo or path)")

    # --- train ---
    trn = sub.add_parser("train", help="Train a LoRA or full model (requires ltx-trainer-mlx)")
    trn.add_argument("--config", "-c", required=True, help="Path to training config YAML file")
    trn.add_argument(
        "--low-ram",
        action="store_true",
        help="Enable gradient checkpointing (recompute blocks in backward) to fit the dev model on 64 GB",
    )

    # --- preprocess ---
    pre = sub.add_parser("preprocess", help="Preprocess videos into latents + conditions for training")
    pre.add_argument("--videos", "-v", required=True, help="Directory containing video files (mp4/mov/avi)")
    pre.add_argument("--output", "-o", required=True, help="Output directory for preprocessed data")
    pre.add_argument(
        "--model", "-m", default=DEFAULT_MODEL, help=f"Model weights for VAE encoding (default: {DEFAULT_MODEL})"
    )
    pre.add_argument("--gemma", default=DEFAULT_GEMMA, help=f"Gemma model for text encoding (default: {DEFAULT_GEMMA})")
    pre.add_argument("--height", "-H", type=int, default=None, help="Resize height (default: keep original)")
    pre.add_argument("--width", "-W", type=int, default=None, help="Resize width (default: keep original)")
    pre.add_argument(
        "--max-frames", type=int, default=97, help="Max frames per video (must satisfy frames %% 8 == 1, default: 97)"
    )
    pre.add_argument("--captions", default=None, help="Directory with .txt caption files (same stems as videos)")
    pre.add_argument("--caption-ext", default=".txt", help="Caption file extension (default: .txt)")
    pre.add_argument(
        "--with-audio",
        action="store_true",
        help="Also encode each clip's audio into audio_latents/ for joint audio-video training",
    )
    pre.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help="Override fps written into latents + used to size audio tokens (default: probed per clip)",
    )

    # --- slice ---
    slc = sub.add_parser("slice", help="Slice long videos into normalized training clips (audio retained)")
    slc.add_argument("sources", nargs="+", help="Video files or directories to slice")
    slc.add_argument("--output", "-o", required=True, help="Root output directory (one subfolder per source)")
    slc.add_argument("--interval", type=float, default=4.0, help="Clip length in seconds (default: 4.0)")
    slc.add_argument("--timecodes", default=None, help="Optional start,end timecode list file (overrides --interval)")
    slc.add_argument("--res", default="384x384", help="Target resolution WxH, both divisible by 32 (default: 384x384)")
    slc.add_argument("--fps", type=float, default=24.0, help="Output frame rate (default: 24.0)")
    slc.add_argument(
        "--fit",
        choices=["crop", "pad"],
        default="crop",
        help="Aspect handling: crop=center-crop (default), pad=letterbox",
    )
    slc.add_argument("--min-length", type=float, default=2.0, help="Drop clips shorter than this (default: 2.0s)")
    slc.add_argument(
        "--max-clips", type=int, default=None, help="Cap clips per source (build a pool to cull; default: all)"
    )
    slc.add_argument(
        "--sample",
        choices=["even", "sequential"],
        default="even",
        help="When --max-clips is set: even=spread across source (default), sequential=first N",
    )
    slc.add_argument("--skip-start", type=float, default=0.0, help="Skip N seconds at the start of each source (intro)")
    slc.add_argument("--skip-end", type=float, default=0.0, help="Skip N seconds at the end of each source (outro)")
    slc.add_argument(
        "--caption-template", default=None, help="If set, write this text to a .txt next to each clip for editing"
    )
    slc.add_argument("--crf", type=int, default=18, help="x264 quality, lower = better (default: 18)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Resolve seed=-1 to a random value and always print the concrete seed so
    # callers (video-lab UI, logs) can reproduce runs.
    if hasattr(args, "seed"):
        if args.seed is None or args.seed < 0:
            import random

            args.seed = random.randint(0, 2**31 - 1)
        if not getattr(args, "quiet", False):
            print(f"Seed: {args.seed}", flush=True)

    commands = {
        "generate": _cmd_generate,
        "a2v": _cmd_a2v,
        "retake": _cmd_retake,
        "extend": _cmd_extend,
        "keyframe": _cmd_keyframe,
        "ic-lora": _cmd_ic_lora,
        "lipdub": _cmd_lipdub,
        "hdr-ic-lora": _cmd_hdr_ic_lora,
        "enhance": _cmd_enhance,
        "info": _cmd_info,
        "train": _cmd_train,
        "preprocess": _cmd_preprocess,
        "slice": _cmd_slice,
    }
    commands[args.command](args)


# =============================================================================
# Generate (T2V / I2V / Two-stage / HQ)
# =============================================================================


def _cmd_generate(args: argparse.Namespace) -> None:
    """Generate a video from a text prompt (and optionally a reference image)."""
    t0 = time.time()

    prompt = _maybe_enhance_prompt(args)

    # Version-aware distilled LoRA default: pick the filename that exists in
    # the checkpoint's model dir (2.5 → lora-450, 2.3 → lora-384) unless the
    # user passed an explicit --distilled-lora.
    if args.distilled_lora is None:
        args.distilled_lora = _default_distilled_lora(args.model)

    lora_paths = [(path, float(strength)) for path, strength in args.lora] if args.lora else []

    if args.enable_teacache and not (args.two_stages_hq or args.two_stage):
        raise SystemExit(
            "--enable-teacache requires --two-stage (or --two-stages-hq, but HQ is not yet "
            "supported). The one-stage distilled path does not benefit from "
            "TeaCache (only 8 denoising steps)."
        )

    if sum(map(bool, (args.two_stages_hq, args.two_stage, args.distilled, args.one_stage))) > 1:
        raise SystemExit("Choose at most one of --two-stage, --two-stages-hq, --distilled, --one-stage.")

    if args.one_stage:
        from ltx_pipelines_mlx.ti2vid_one_stage import TI2VidOneStagePipeline

        if not args.quiet:
            print("Mode: One-Stage Dev (Euler + CFG at full resolution)")
            print(f"  Model: {args.model}")

        pipe = TI2VidOneStagePipeline(
            model_dir=args.model,
            gemma_model_id=args.gemma,
            low_memory=True,
            low_ram_streaming=getattr(args, "low_ram", False),
            dev_transformer=args.dev_transformer,
            tile_count=_build_tile_count_config(args),
        )
        pipe.verbose = not args.quiet
        if lora_paths:
            pipe._pending_loras = lora_paths
        kwargs: dict = dict(
            prompt=prompt,
            output_path=args.output,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            frame_rate=args.frame_rate,
            seed=args.seed,
            images=args.images,
        )
        # The dev one-stage pipeline uses `num_steps` (single sampler), not stage1/stage2.
        if args.stage1_steps is not None:
            kwargs["num_steps"] = args.stage1_steps
        elif args.steps is not None:
            kwargs["num_steps"] = args.steps
        if args.cfg_scale is not None:
            kwargs["cfg_scale"] = args.cfg_scale
        if args.stg_scale is not None:
            kwargs["stg_scale"] = args.stg_scale
        pipe.generate_and_save(**kwargs)

    elif args.distilled:
        from ltx_pipelines_mlx.distilled import DistilledPipeline

        if not args.quiet:
            print("Mode: Distilled Two-Stage (half-res + upscale + distilled refine)")
            print(f"  Model: {args.model}")

        pipe = DistilledPipeline(
            model_dir=args.model,
            gemma_model_id=args.gemma,
            low_memory=True,
            low_ram_streaming=getattr(args, "low_ram", False),
            tile_count=_build_tile_count_config(args),
        )
        pipe.verbose = not args.quiet
        if lora_paths:
            pipe._pending_loras = lora_paths
        kwargs: dict = dict(
            prompt=prompt,
            output_path=args.output,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            frame_rate=args.frame_rate,
            seed=args.seed,
            images=args.images,
        )
        if args.stage1_steps is not None:
            kwargs["stage1_steps"] = args.stage1_steps
        if args.stage2_steps is not None:
            kwargs["stage2_steps"] = args.stage2_steps
        pipe.generate_and_save(**kwargs)

    elif args.two_stages_hq or args.two_stage:
        if args.two_stages_hq:
            from ltx_pipelines_mlx.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline as PipeClass

            mode_name = "HQ Two-Stage (res_2s + CFG + distilled LoRA)"
        else:
            from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline as PipeClass

            mode_name = "Two-Stage (Euler + CFG + distilled LoRA)"

        if not args.quiet:
            print(f"Mode: {mode_name}")
            print(f"  Model: {args.model}")

        pipe = PipeClass(
            model_dir=args.model,
            gemma_model_id=args.gemma,
            low_memory=True,
            low_ram_streaming=getattr(args, "low_ram", False),
            dev_transformer=args.dev_transformer,
            distilled_lora=args.distilled_lora,
            distilled_lora_strength=args.distilled_lora_strength,
            tile_count=_build_tile_count_config(args),
        )
        pipe.verbose = not args.quiet
        if lora_paths:
            pipe._pending_loras = lora_paths
        # two-stage / HQ accept the upstream-iso multi-image conditioning list.
        # Pass the list through; the legacy single-image path is handled inside
        # the pipeline (image=None when images is set).
        kwargs: dict = dict(
            prompt=prompt,
            output_path=args.output,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            frame_rate=args.frame_rate,
            seed=args.seed,
            images=args.images,
        )
        if args.stage1_steps is not None:
            kwargs["stage1_steps"] = args.stage1_steps
        if args.stage2_steps is not None:
            kwargs["stage2_steps"] = args.stage2_steps
        if args.cfg_scale is not None:
            kwargs["cfg_scale"] = args.cfg_scale
        if args.stg_scale is not None:
            kwargs["stg_scale"] = args.stg_scale
        if args.enable_teacache:
            kwargs["enable_teacache"] = True
            if args.teacache_thresh is not None:
                kwargs["teacache_thresh"] = args.teacache_thresh
        pipe.generate_and_save(**kwargs)

    else:
        raise SystemExit(
            "generate requires one of --one-stage, --two-stage, --two-stages-hq, --distilled. "
            "Each maps to an upstream pipeline class:\n"
            "  --one-stage : TI2VidOneStagePipeline (dev + CFG, full target res)\n"
            "  --two-stage : TI2VidTwoStagesPipeline (dev + CFG, half-res + upscale, recommended)\n"
            "  --two-stages-hq        : TI2VidTwoStagesHQPipeline (res_2s + CFG, half-res + upscale)\n"
            "  --distilled : DistilledPipeline (distilled half-res + upscale, fastest)"
        )

    _print_result(args.output, t0, args.quiet)


# =============================================================================
# Audio-to-Video
# =============================================================================


def _cmd_a2v(args: argparse.Namespace) -> None:
    """Generate video from audio + text prompt."""
    t0 = time.time()

    if args.one_stage:
        from ltx_pipelines_mlx.a2vid_one_stage import A2VidPipelineOneStage as PipeClass
        pipe_kwargs: dict = {}
    else:
        from ltx_pipelines_mlx.a2vid_two_stage import A2VidPipelineTwoStage as PipeClass
        # Version-aware distilled LoRA default (2.5 → lora-450, 2.3 → lora-384).
        pipe_kwargs = {
            "distilled_lora": args.distilled_lora or _default_distilled_lora(args.model),
            "distilled_lora_strength": args.distilled_lora_strength,
        }

    if not args.quiet:
        mode_label = "one-stage experimental" if args.one_stage else "two-stage"
        print(f"Mode: Audio-to-Video ({mode_label}, Euler + CFG)")
        print(f"Audio: {args.audio}")
        print(f"  Model: {args.model}")

    pipe = PipeClass(
        model_dir=args.model,
        gemma_model_id=args.gemma,
        low_ram_streaming=getattr(args, "low_ram", False),
        **pipe_kwargs,
    )
    pipe.verbose = not args.quiet
    kwargs: dict = dict(
        prompt=args.prompt,
        output_path=args.output,
        audio_path=args.audio,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        frame_rate=args.frame_rate,
        seed=args.seed,
        images=args.images,
        audio_start_time=args.audio_start,
    )
    if args.one_stage:
        if args.steps is not None:
            kwargs["num_steps"] = args.steps
    else:
        if args.stage1_steps is not None:
            kwargs["stage1_steps"] = args.stage1_steps
        if args.stage2_steps is not None:
            kwargs["stage2_steps"] = args.stage2_steps
    if args.cfg_scale is not None:
        kwargs["cfg_scale"] = args.cfg_scale
    if args.stg_scale is not None:
        kwargs["stg_scale"] = args.stg_scale
    pipe.generate_and_save(**kwargs)

    _print_result(args.output, t0, args.quiet)


# =============================================================================
# Retake
# =============================================================================


def _cmd_retake(args: argparse.Namespace) -> None:
    """Regenerate a time segment of an existing video."""
    t0 = time.time()

    from ltx_pipelines_mlx.retake import RetakePipeline

    frame_rate = _source_video_frame_rate(args.video)
    if not args.quiet:
        print("Mode: Retake")
        print(f"Video: {args.video}, frames {args.start}-{args.end}")

    pipe = RetakePipeline(model_dir=args.model, gemma_model_id=args.gemma)
    pipe.verbose = not args.quiet
    kwargs: dict = dict(
        prompt=args.prompt,
        video_path=args.video,
        start_frame=args.start,
        end_frame=args.end,
        seed=args.seed,
        regenerate_audio=not args.no_regen_audio,
    )
    if args.steps is not None:
        kwargs["num_steps"] = args.steps
    if args.cfg_scale is not None:
        kwargs["cfg_scale"] = args.cfg_scale
    if args.stg_scale is not None:
        kwargs["stg_scale"] = args.stg_scale
    video_latent, audio_latent = pipe.retake_from_video(**kwargs)

    _decode_and_save(pipe, video_latent, audio_latent, args, frame_rate=frame_rate)
    _print_result(args.output, t0, args.quiet)


# =============================================================================
# Extend
# =============================================================================


def _cmd_extend(args: argparse.Namespace) -> None:
    """Add frames before or after an existing video."""
    t0 = time.time()

    from ltx_pipelines_mlx.retake import RetakePipeline

    frame_rate = _source_video_frame_rate(args.video)
    if not args.quiet:
        print(f"Mode: Extend ({args.direction})")
        print(f"Video: {args.video}, +{args.extend_frames} latent frames")

    pipe = RetakePipeline(model_dir=args.model, gemma_model_id=args.gemma)
    pipe.verbose = not args.quiet
    kwargs: dict = dict(
        prompt=args.prompt,
        video_path=args.video,
        extend_frames=args.extend_frames,
        direction=args.direction,
        seed=args.seed,
    )
    if args.steps is not None:
        kwargs["num_steps"] = args.steps
    if args.cfg_scale is not None:
        kwargs["cfg_scale"] = args.cfg_scale
    if args.stg_scale is not None:
        kwargs["stg_scale"] = args.stg_scale
    video_latent, audio_latent = pipe.extend_from_video(**kwargs)

    _decode_and_save(pipe, video_latent, audio_latent, args, frame_rate=frame_rate)
    _print_result(args.output, t0, args.quiet)


# =============================================================================
# Keyframe interpolation
# =============================================================================


def _cmd_keyframe(args: argparse.Namespace) -> None:
    """Interpolate between two keyframe images."""
    t0 = time.time()

    from ltx_pipelines_mlx.keyframe_interpolation import KeyframeInterpolationPipeline

    if not args.quiet:
        print("Mode: Keyframe Interpolation (two-stage)")
        print(f"Start: {args.start}, End: {args.end}")

    last_pixel_frame = args.frames - 1

    # Version-aware distilled LoRA default (2.5 → lora-450, 2.3 → lora-384).
    if args.distilled_lora is None:
        args.distilled_lora = _default_distilled_lora(args.model)

    pipe = KeyframeInterpolationPipeline(
        model_dir=args.model,
        gemma_model_id=args.gemma,
        low_ram_streaming=getattr(args, "low_ram", False),
        dev_transformer=args.dev_transformer,
        distilled_lora=args.distilled_lora,
        distilled_lora_strength=args.lora_strength,
    )
    pipe.verbose = not args.quiet
    # Build guider params (defaults match reference LTX_2_3_PARAMS)
    from ltx_core_mlx.components.guiders import MultiModalGuiderParams

    cfg = args.cfg_scale if args.cfg_scale is not None else 3.0
    stg = args.stg_scale if args.stg_scale is not None else 1.0
    video_gp = MultiModalGuiderParams(
        cfg_scale=cfg,
        stg_scale=stg,
        rescale_scale=0.7,
        modality_scale=3.0,
        stg_blocks=[28],
    )
    audio_gp = MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=stg,
        rescale_scale=0.7,
        modality_scale=3.0,
        stg_blocks=[28],
    )

    pipe.generate_and_save(
        prompt=args.prompt,
        output_path=args.output,
        keyframe_images=[args.start, args.end],
        keyframe_indices=[0, last_pixel_frame],
        keyframe_strengths=[args.start_strength, args.end_strength],
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        frame_rate=args.frame_rate,
        seed=args.seed,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        cfg_scale=cfg,
        video_guider_params=video_gp,
        audio_guider_params=audio_gp,
    )
    _print_result(args.output, t0, args.quiet)


# =============================================================================
# IC-LoRA
# =============================================================================


def _cmd_ic_lora(args: argparse.Namespace) -> None:
    """Generate video with IC-LoRA control conditioning."""
    t0 = time.time()

    from ltx_pipelines_mlx.ic_lora import ICLoraPipeline

    # Parse --lora pairs into (path, strength) tuples
    lora_paths = [(path, float(strength)) for path, strength in args.lora]

    # Parse --video-conditioning pairs
    video_conditioning = [(path, float(strength)) for path, strength in args.video_conditioning]

    if not args.quiet:
        print("Mode: IC-LoRA (two-stage)")
        for path, strength in lora_paths:
            print(f"  LoRA: {path} (strength={strength})")
        for path, strength in video_conditioning:
            print(f"  Control: {path} (strength={strength})")

    pipe = ICLoraPipeline(
        model_dir=args.model,
        lora_paths=lora_paths,
        gemma_model_id=args.gemma,
        low_memory=True,
        low_ram_streaming=getattr(args, "low_ram", False),
    )
    pipe.verbose = not args.quiet

    pipe.generate_and_save(
        prompt=args.prompt,
        output_path=args.output,
        video_conditioning=video_conditioning,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        frame_rate=args.frame_rate,
        seed=args.seed,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        images=args.images,
        conditioning_attention_strength=args.conditioning_strength,
        skip_stage_2=args.skip_stage_2,
    )
    _print_result(args.output, t0, args.quiet)


def _cmd_lipdub(args: argparse.Namespace) -> None:
    """Lip-dub a reference video: re-generate visuals matching its audio."""
    t0 = time.time()

    from ltx_pipelines_mlx.lipdub import LipDubPipeline

    lora_paths = [(path, float(strength)) for path, strength in args.lora]
    if len(lora_paths) != 1:
        raise SystemExit("lipdub: exactly one --lora is required (the lip-dub IC-LoRA).")

    if not args.quiet:
        print("Mode: LipDub (two-stage, distilled + IC-LoRA + audio reference)")
        print(f"  Reference video: {args.reference_video}")
        print(f"  IC-LoRA: {lora_paths[0][0]} (strength={lora_paths[0][1]})")

    pipe = LipDubPipeline(
        model_dir=args.model,
        lora_paths=lora_paths,
        gemma_model_id=args.gemma,
        low_ram_streaming=getattr(args, "low_ram", False),
    )
    pipe.verbose = not args.quiet
    # Build image anchors from CLI args
    images = None
    if getattr(args, "image", None):
        image_list = args.image or []
        frame_list = getattr(args, "image_frame", []) or [0] * len(image_list)
        strength_list = getattr(args, "image_strength", []) or [1.0] * len(image_list)
        images = [(str(img), int(frame), float(strength)) for img, frame, strength in zip(image_list, frame_list, strength_list)]

    pipe.generate_and_save(
        prompt=args.prompt,
        output_path=args.output,
        reference_video_path=args.reference_video,
        height=args.height,
        width=args.width,
        reference_strength=args.reference_strength,
        seed=args.seed,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        images=images,
    )
    _print_result(args.output, t0, args.quiet)


def _cmd_hdr_ic_lora(args: argparse.Namespace) -> None:
    """Generate HDR video via IC-LoRA + LogC3 inverse decompression."""
    t0 = time.time()

    from ltx_pipelines_mlx.hdr_ic_lora import HDRICLoraPipeline

    lora_paths = [(path, float(strength)) for path, strength in args.lora]
    video_conditioning = [(path, float(strength)) for path, strength in (args.video_conditioning or [])]

    if not args.quiet:
        mode_suffix = "T2V" if not video_conditioning else "V2V"
        print(f"Mode: HDR IC-LoRA (two-stage, LogC3, {mode_suffix})")
        for path, strength in lora_paths:
            print(f"  LoRA: {path} (strength={strength})")
        for path, strength in video_conditioning:
            print(f"  Control: {path} (strength={strength})")

    pipe = HDRICLoraPipeline(
        model_dir=args.model,
        lora_paths=lora_paths,
        gemma_model_id=args.gemma,
        low_memory=True,
        low_ram_streaming=getattr(args, "low_ram", False),
    )
    pipe.verbose = not args.quiet

    pipe.generate_and_save(
        prompt=args.prompt,
        output_path=args.output,
        video_conditioning=video_conditioning,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        frame_rate=args.frame_rate,
        seed=args.seed,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        images=args.images,
        conditioning_attention_strength=args.conditioning_strength,
        skip_stage_2=args.skip_stage_2,
    )
    _print_result(args.output, t0, args.quiet)


# =============================================================================
# Shared helpers
# =============================================================================


def _decode_and_save(
    pipe: object,
    video_latent: object,
    audio_latent: object,
    args: argparse.Namespace,
    *,
    frame_rate: float,
) -> None:
    """Decode latents and save to file."""
    from ltx_core_mlx.utils.memory import aggressive_cleanup

    # Free DiT + text encoder to make room for decoders
    if hasattr(pipe, "low_memory") and pipe.low_memory:
        pipe.dit = None
        pipe.text_encoder = None
        pipe.feature_extractor = None
        pipe._loaded = False
        aggressive_cleanup()

    # Load decoders on-demand and decode+save
    pipe._load_decoders()
    pipe._decode_and_save_video(video_latent, audio_latent, args.output, frame_rate=frame_rate)


def _source_video_frame_rate(video_path: str) -> float:
    """Return the source video FPS for workflows whose timing follows an input clip."""
    from ltx_core_mlx.utils.ffmpeg import probe_video_info

    return float(probe_video_info(video_path).fps)


def _maybe_enhance_prompt(args: argparse.Namespace) -> str:
    """Enhance prompt if --enhance-prompt is set."""
    prompt = args.prompt
    if not getattr(args, "enhance_prompt", False):
        return prompt

    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
    from ltx_core_mlx.utils.memory import aggressive_cleanup

    if not args.quiet:
        print("Enhancing prompt...")
    gemma = GemmaLanguageModel()
    gemma.load(args.gemma)
    if getattr(args, "images", None):
        prompt = gemma.enhance_i2v(prompt, seed=args.seed)
    else:
        prompt = gemma.enhance_t2v(prompt, seed=args.seed)
    if not args.quiet:
        print(f"Enhanced: {prompt[:200]}...")
    del gemma
    aggressive_cleanup()
    return prompt


def _print_result(output: str, t0: float, quiet: bool) -> None:
    """Print generation result."""
    elapsed = time.time() - t0
    if not quiet:
        print(f"\nSaved to: {output}")
        print(f"Time: {elapsed:.1f}s")


def _cmd_train(args: argparse.Namespace) -> None:
    """Train a LoRA or full model from a YAML config."""
    try:
        from ltx_trainer_mlx.config import LtxTrainerConfig
        from ltx_trainer_mlx.trainer import LtxvTrainer
    except ImportError:
        print("Error: ltx-trainer-mlx is not installed.")
        print("Install it with: uv pip install -e 'packages/ltx-trainer[all]'")
        sys.exit(1)

    from pathlib import Path

    import yaml

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    config = LtxTrainerConfig(**raw_config)

    if getattr(args, "low_ram", False):
        # Gradient checkpointing: recompute transformer blocks in the backward
        # pass to cap activation memory (lets the dev model backprop fit on 64 GB).
        config.optimization.enable_gradient_checkpointing = True
        print("Low-RAM training: gradient checkpointing enabled.")

    trainer = LtxvTrainer(config)
    # When stdout isn't a TTY (nohup / redirect), the rich progress bar suppresses
    # itself and shows nothing. Disable it so the per-step logger lines fire instead,
    # giving redirected/background runs a readable loss trace.
    saved_path, stats = trainer.train(disable_progress_bars=not sys.stdout.isatty())

    print("\nTraining complete!")
    print(f"  Weights saved to: {saved_path}")
    print(f"  Total time: {stats.total_time_seconds / 60:.1f} min")
    print(f"  Speed: {stats.steps_per_second:.2f} steps/s")
    print(f"  Peak memory: {stats.peak_memory_gb:.1f} GB")


def _cmd_preprocess(args: argparse.Namespace) -> None:
    """Preprocess videos into latents + conditions for training."""
    try:
        from ltx_trainer_mlx.preprocess import preprocess_dataset
    except ImportError:
        print("Error: ltx-trainer-mlx is not installed.")
        print("Install it with: uv pip install -e 'packages/ltx-trainer[all]'")
        sys.exit(1)

    preprocess_dataset(
        videos_dir=args.videos,
        output_dir=args.output,
        model_dir=args.model,
        gemma_model_id=args.gemma,
        target_height=args.height,
        target_width=args.width,
        max_frames=args.max_frames,
        captions_dir=args.captions,
        caption_ext=args.caption_ext,
        with_audio=args.with_audio,
        frame_rate=args.frame_rate,
    )


def _cmd_slice(args: argparse.Namespace) -> None:
    """Slice long videos into normalized training clips."""
    try:
        from ltx_trainer_mlx.slice_clips import slice_videos
    except ImportError:
        print("Error: ltx-trainer-mlx is not installed.")
        print("Install it with: uv pip install -e 'packages/ltx-trainer[all]'")
        sys.exit(1)

    slice_videos(
        sources=args.sources,
        out_dir=args.output,
        interval=args.interval,
        timecodes_file=args.timecodes,
        res=args.res,
        fps=args.fps,
        fit=args.fit,
        min_length=args.min_length,
        max_clips=args.max_clips,
        sample=args.sample,
        skip_start=args.skip_start,
        skip_end=args.skip_end,
        caption_template=args.caption_template,
        crf=args.crf,
    )


def _cmd_enhance(args: argparse.Namespace) -> None:
    """Enhance a prompt using Gemma."""
    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

    print("Loading Gemma...")
    gemma = GemmaLanguageModel()
    gemma.load(args.gemma)

    if args.mode == "t2v":
        enhanced = gemma.enhance_t2v(args.prompt, seed=args.seed)
    else:
        enhanced = gemma.enhance_i2v(args.prompt, seed=args.seed)

    print(f"\nOriginal: {args.prompt}")
    print(f"\nEnhanced: {enhanced}")


def _cmd_info(args: argparse.Namespace) -> None:
    """Show model info and memory estimate."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    model_dir = Path(args.model)
    if not model_dir.exists():
        try:
            model_dir = Path(snapshot_download(args.model))
        except Exception as e:
            print(f"Could not find or download model: {args.model}")
            print(f"  {e}")
            sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Path:  {model_dir}")
    print()

    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        print("  No .safetensors files found.")
        return

    total_bytes = 0
    for f in safetensor_files:
        size = f.stat().st_size
        total_bytes += size
        print(f"  {f.name:<45s} {size / 1024**2:>8.1f} MB")

    total_mb = total_bytes / 1024**2
    total_gb = total_mb / 1024
    print(f"  {'─' * 55}")
    print(f"  {'Total':<45s} {total_mb:>8.1f} MB ({total_gb:.1f} GB)")
    print(f"  Estimated RAM: ~{total_gb * 1.3:.0f} GB (model + inference overhead)")

    json_files = sorted(model_dir.glob("*.json"))
    if json_files:
        print(f"\n  Config files: {', '.join(f.name for f in json_files)}")

    upsampler_files = [f for f in safetensor_files if "upscaler" in f.name or "upsampler" in f.name]
    if upsampler_files:
        print(f"\n  Upsamplers: {', '.join(f.stem for f in upsampler_files)}")


if __name__ == "__main__":
    main()
