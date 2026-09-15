# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stability AI — vendored from github.com/Stability-AI/stable-audio-3
"""SA3 text-to-audio inference in pure MLX (no stable-audio-tools needed).

Pipeline:
    prompt → T5Gemma → conditioning (cross_attn + global) → DiT pingpong (8 steps)
           → SAME-S / SAME-L decoder → patched-pretransform unpatch → WAV

Usage:
    python sa3_mlx.py --prompt "lofi house loop" --dit small --decoder same-s --seconds 30 --out out.wav

If --dit or --decoder is omitted, the script prompts the user interactively.
"""

from __future__ import annotations
import argparse, math, os, random, subprocess, sys, termios, time, tty, wave
from pathlib import Path

import numpy as np
import mlx.core as mx

REPO = Path(__file__).resolve().parent.parent  # project root (scripts/ is one level down)
sys.path.insert(0, str(REPO))                   # so `from models.defs.*` resolves
sys.path.insert(0, str(REPO / "scripts"))       # so `from weights import *` resolves

from models.defs.sa3_pipeline import (
    SecondsTotalEmbedder,
    apply_prompt_padding,
    build_pingpong_schedule,
    sample_flow_pingpong,
    patched_decode,
    load_conditioner_from_npz,
)
from models.defs.t5gemma_mlx import T5Gemma
from weights import ensure_local, is_present

SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096  # PatchedPretransform downsample × SAME 16× expansion

# ─── Display helpers (ANSI color when stdout is a TTY) ───────────────────
_USE_COLOR = sys.stdout.isatty()
_RULE_W = 64

def _c(code: str, s: str) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if _USE_COLOR else s

def bold(s):   return _c("1", s)
def dim(s):    return _c("2", s)
def cyan(s):   return _c("36", s)
def yellow(s): return _c("33", s)
def green(s):  return _c("32", s)
def magenta(s):return _c("35", s)

def rule(char="━", color=cyan):
    print(color(char * _RULE_W))

def banner(title: str):
    rule()
    print(f"  {bold(title)}")
    rule()

def _fmt_mem(b: int) -> str:
    if b >= 1024**3: return f"{b/1024**3:5.2f} GB"
    return f"{b/1024**2:5.0f} MB"


def stage(idx_total: str, label: str, ms: float | None = None, peak_b: int | None = None):
    """Right-aligned timing column with dotted fill. `peak_b` is accepted for
    backward-compat / tracking-via-caller but is NOT rendered inline — peak RAM
    is shown only in the 'Peak RAM by stage' summary at the end."""
    head = f"  {cyan(idx_total)} {bold(label)}"
    if ms is None:
        print(head)
        return
    visible = len(f"  {idx_total} {label}")
    fill = max(2, _RULE_W - visible - 9)
    dots = dim("·" * fill)
    print(f"{head} {dots} {yellow(f'{ms:>5.0f} ms')}")

def sub(text: str):
    print(f"        {dim(text)}")

# Each DiT .npz file has the conditioner weights baked in under a "cond." key prefix.
# The codec architecture (same-s vs same-l) is fixed per DiT family.
DIT_CHOICES = {
    "sm-music": {"loader": "models.defs.dit_mlx",
                 "ckpt":    "models/mlx/dit_sm-music_f16.npz",
                 "default_decoder": "same-s"},
    "sm-sfx":   {"loader": "models.defs.dit_mlx",
                 "ckpt":    "models/mlx/dit_sm-sfx_f16.npz",
                 "default_decoder": "same-s"},
    "medium":   {"loader": "models.defs.dit_mlx_medium",
                 "ckpt":    "models/mlx/dit_medium_f16.npz",
                 "default_decoder": "same-l"},
}
DECODER_CHOICES = {
    "same-s": ("models.defs.same_s_decoder", "decode_chunked", (8, 2),
               "models/mlx/same_s_decoder_f32.npz"),
    "same-l": ("models.defs.same_l_decoder", "decode_chunked", (128, 8),
               "models/mlx/same_l_decoder_f32.npz"),
}
# Encoder paired with each decoder (used for --init-audio).
ENCODER_CHOICES = {
    "same-s": ("models.defs.same_s_encoder", 32, "models/mlx/same_s_encoder_f32.npz"),
    "same-l": ("models.defs.same_l_encoder", 16, "models/mlx/same_l_encoder_f32.npz"),
}
T5GEMMA_NPZ_REL = "models/mlx/t5gemma_f16.npz"


def _arrow_pick(prompt: str, options: list[str], default: str | None = None) -> str:
    """Tiny arrow-key picker — no external deps, posix termios only.

    Up/Down to move, Enter to select, Ctrl-C to abort. Falls back to a
    numeric prompt when stdin isn't a TTY (piped input, CI, etc.).
    """
    if not sys.stdin.isatty():
        print(prompt)
        for i, o in enumerate(options):
            mark = "*" if o == default else " "
            print(f"  {mark} [{i}] {o}")
        s = input(f"Choose [0-{len(options)-1}] (Enter for default): ").strip()
        if s == "":
            return default or options[0]
        if s.isdigit() and 0 <= int(s) < len(options):
            return options[int(s)]
        return s if s in options else (default or options[0])

    idx = options.index(default) if default in options else 0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print(prompt)
    # leave room for the options
    for _ in options:
        print()
    try:
        tty.setcbreak(fd)
        while True:
            # move up `len(options)` lines and redraw
            sys.stdout.write(f"\x1b[{len(options)}A")
            for i, o in enumerate(options):
                if i == idx:
                    sys.stdout.write(f"\x1b[2K\x1b[36m▶ {o}\x1b[0m\n")
                else:
                    sys.stdout.write(f"\x1b[2K  {o}\n")
            sys.stdout.flush()
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A":
                    idx = (idx - 1) % len(options)
                elif seq == "[B":
                    idx = (idx + 1) % len(options)
            elif ch in ("\n", "\r"):
                return options[idx]
            elif ch == "\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def prompt_user_if_missing(args):
    """Interactive arrow-key selection when --dit / --decoder aren't supplied."""
    if args.dit is None:
        args.dit = _arrow_pick("Choose DiT model:", list(DIT_CHOICES.keys()), default="medium")
        print(f"  → {args.dit}")
    if args.decoder is None:
        suggested = DIT_CHOICES[args.dit]["default_decoder"]
        args.decoder = _arrow_pick("Choose audio decoder:", list(DECODER_CHOICES.keys()), default=suggested)
        print(f"  → {args.decoder}")
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
    return args


def load_dit(dit_name: str, T_lat: int, dtype):
    cfg = DIT_CHOICES[dit_name]
    ckpt = ensure_local(cfg["ckpt"])
    import importlib, io, contextlib
    mod = importlib.import_module(cfg["loader"])
    with contextlib.redirect_stdout(io.StringIO()):
        model = mod.load_dit(str(ckpt), T_lat=T_lat, dtype=dtype, compile_=False)
    return model, str(ckpt)


def load_decoder(decoder_name: str, dtype):
    spec = DECODER_CHOICES[decoder_name]
    weights_path = ensure_local(spec[3])
    import importlib, io, contextlib
    mod = importlib.import_module(spec[0])
    with contextlib.redirect_stdout(io.StringIO()):
        model = mod.load_model(weights_path=str(weights_path), dtype=dtype, compile_=False)
    chunk_fn = getattr(mod, spec[1])
    chunk_cfg = spec[2]
    return model, chunk_fn, chunk_cfg


# Peak unified-memory tracker (works as both pre-0.34 mx.metal.* and new mx.* API)
def _reset_peak_mem():
    fn = getattr(mx, "reset_peak_memory", None) or getattr(mx.metal, "reset_peak_memory", None)
    if fn: fn()

def _get_peak_mem_bytes() -> int:
    fn = getattr(mx, "get_peak_memory", None) or getattr(mx.metal, "get_peak_memory", None)
    return int(fn()) if fn else 0


def _free_to_pool():
    """gc.collect() + flush MLX's memory cache so freed weights actually return to the OS."""
    import gc
    gc.collect()
    fn = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
    if fn: fn()


_CUMULATIVE_PEAK_B = 0
_STAGE_PEAKS: list[tuple[str, int]] = []   # [(stage_label, peak_bytes)]
def _stage_peak_b(label: str | None = None) -> int:
    """Read current peak, track cumulative max + per-stage log, reset for next stage."""
    global _CUMULATIVE_PEAK_B
    b = _get_peak_mem_bytes()
    _CUMULATIVE_PEAK_B = max(_CUMULATIVE_PEAK_B, b)
    if label is not None:
        _STAGE_PEAKS.append((label, b))
    _reset_peak_mem()
    return b


def save_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """audio: (channels, T) float32 in [-1, 1]. Writes 16-bit PCM stereo WAV."""
    if not np.isfinite(audio).all():
        n_bad = int((~np.isfinite(audio)).sum())
        raise RuntimeError(f"refusing to write WAV — audio contains {n_bad} non-finite samples (NaN/Inf)")
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16).T  # (T, channels) interleaved
    with wave.open(path, "wb") as w:
        w.setnchannels(audio.shape[0])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def read_wav(path: str) -> np.ndarray:
    """Read a WAV file. Returns (2, T) float32 in [-1, 1].

    Handles 16-bit PCM at 44.1 kHz natively. Falls back to ffmpeg for any
    other format (24-bit, 32-bit float, 48 kHz, etc.)
    Mono input is duplicated to stereo.
    """
    try:
        with wave.open(path, "rb") as w:
            nch, sw, sr, nframes = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            if sr == SAMPLE_RATE and sw == 2:
                raw = np.frombuffer(w.readframes(nframes), dtype=np.int16).astype(np.float32) / 32767.0
                if nch == 1:
                    return np.stack([raw, raw], axis=0)   # (2, T)
                return raw.reshape(-1, nch).T[:2]          # (2, T)
    except wave.Error:
        pass  # unsupported format (32-bit float, 24-bit PCM, etc.)

    # Fallback: decode via ffmpeg — handles any sample rate, bit depth, or format
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path,
             "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-"],
            capture_output=True, check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"{path}: unsupported WAV format. Install ffmpeg to handle 24-bit/32-bit/48kHz audio:\n"
            f"  brew install ffmpeg\n"
            f"Or convert manually: ffmpeg -i \"{path}\" -ar {SAMPLE_RATE} -ac 2 -sample_fmt s16 resampled.wav"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{path}: ffmpeg failed — {e.stderr.decode().strip()}")
    raw = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32767.0
    return raw.reshape(-1, 2).T  


def load_encoder(decoder_name: str, dtype):
    """Mirror load_decoder but for the matching encoder."""
    spec = ENCODER_CHOICES[decoder_name]
    weights_path = ensure_local(spec[2])
    import importlib, io, contextlib
    mod = importlib.import_module(spec[0])
    with contextlib.redirect_stdout(io.StringIO()):
        model = mod.load_model(weights_path=str(weights_path), dtype=dtype, compile_=False)
    pad_modulo = spec[1]
    return model, pad_modulo


def _preflight_download(args) -> None:
    """Resolve every weight file this run needs and download any that are
    missing — BEFORE the banner prints and BEFORE the wall-clock starts.
    Network time then isn't charged against "×realtime" and the user sees
    download progress as a clearly separate setup step."""
    needed = [
        T5GEMMA_NPZ_REL,
        DIT_CHOICES[args.dit]["ckpt"],
        DECODER_CHOICES[args.decoder][3],
    ]
    if args.init_audio:
        needed.append(ENCODER_CHOICES[args.decoder][2])
    missing = [p for p in needed if not is_present(p)]
    if not missing:
        return
    print(f"  Fetching {len(missing)} missing weight file(s) before starting:")
    for rel in missing:
        ensure_local(rel)
    print()


def patch_audio(audio: np.ndarray, patch_size: int = 256) -> np.ndarray:
    """Patched-pretransform encode: (B, 2, T_audio) → (B, 512, T_audio/256).
    Mirrors rearrange("b c (l h) -> b (c h) l", h=patch_size)."""
    B, C, T = audio.shape
    assert T % patch_size == 0, f"audio length {T} not multiple of patch_size {patch_size}"
    L = T // patch_size
    x = audio.reshape(B, C, L, patch_size).transpose(0, 1, 3, 2)   # (B, C, patch, L)
    return x.reshape(B, C * patch_size, L)                          # (B, 512, L)


class _HelpfulParser(argparse.ArgumentParser):
    """argparse that prints full help (not just usage) when a flag is unknown / invalid."""
    def error(self, message):
        sys.stderr.write(f"\nerror: {message}\n\n")
        self.print_help(sys.stderr)
        sys.exit(2)


def main():
    ap = _HelpfulParser(
        description="SA3 text-to-audio (+ audio-to-audio + inpainting) in pure MLX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "modes\n"
            "  text-to-audio    --prompt P\n"
            "  audio-to-audio   --prompt P --init-audio IN.wav [--init-noise-level σ]\n"
            "  inpainting       --prompt P --init-audio IN.wav --inpaint-range START,END\n"
            "  negative CFG     --prompt P --cfg N --negative-prompt P_NEG\n"
            "\nrun `sa3_mlx.py --help` for per-flag details."
        ),
    )
    # ── Inputs ────────────────────────────────────────────────────────────────
    ap.add_argument("--prompt", default=None,
                    help="Text prompt describing the audio to generate. "
                         "Empty string is valid (unconditional generation). "
                         "If omitted, the script asks interactively via stdin.")
    ap.add_argument("--negative-prompt", default=None,
                    help="Negative prompt for CFG's unconditional branch. "
                         "When --cfg=1.0 this flag has no effect (no uncond pass is run). "
                         "When unset and --cfg ≠ 1.0, the uncond branch uses zero embeddings.")
    ap.add_argument("--init-audio", default=None,
                    help="Path to a WAV file (44.1 kHz, 16-bit PCM, stereo or mono) to use as the "
                         "starting point. Enables audio-to-audio mode (with --init-noise-level) or "
                         "inpainting mode (with --inpaint-range). Encoder is loaded automatically. "
                         "Audio is trimmed or zero-padded to match --seconds.")
    ap.add_argument("--inpaint-range", default=None,
                    help="Inpainting time range as 'START,END' in seconds (e.g. '5,10'). "
                         "Requires --init-audio. The model regenerates the masked range while "
                         "preserving the rest of the input exactly (via paste-back).")

    # ── Models ────────────────────────────────────────────────────────────────
    ap.add_argument("--dit", choices=list(DIT_CHOICES.keys()), default=None,
                    help="DiT model. 'small' = sa3-sm-music (faster, smaller). "
                         "'medium' = sa3-medium-ARC (larger, higher quality). "
                         "If omitted, prompts interactively with arrow-key picker.")
    ap.add_argument("--decoder", choices=list(DECODER_CHOICES.keys()), default=None,
                    help="Audio decoder. 'same-s' pairs with sa3-small (50 M params). "
                         "'same-l' pairs with sa3-medium (426 M params). "
                         "If omitted, prompts interactively with arrow-key picker.")
    ap.add_argument("--dit-dtype", default="fp16", choices=["fp32", "fp16"],
                    help="DiT compute dtype. Default fp16 — validated transparent at ~50-57 dB PSNR "
                         "vs FP32. Halves DiT memory and speeds sampling ~25%%. The decoder always "
                         "runs FP32 (SAME-S catastrophically cancels at fp16 due to differential "
                         "attention) and T5Gemma is always fp16. Set to fp32 only if you need "
                         "bit-exact reproducibility against the PyTorch reference.")
    ap.add_argument("--t5gemma-npz", default=None,
                    help="Path to the bundled T5Gemma FP16 .npz (weights + tokenizer). "
                         "Default points at models/mlx/t5gemma_f16.npz next to this script; "
                         "auto-downloaded from HuggingFace if not present.")
    # ── Sampling ──────────────────────────────────────────────────────────────
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="Output audio length in seconds. T_lat (latent positions) is derived as "
                         "ceil(seconds * 44100 / 4096) — a natural ceil that depends ONLY on "
                         "--seconds (and --seed via the DiT), never on --decoder. This keeps the "
                         "sampled latent (and hence the music) identical across decoders for a "
                         "given prompt/seed. Final WAV is trimmed to exactly --seconds.")
    ap.add_argument("--steps", type=int, default=8,
                    help="Number of pingpong sampling steps. Minimum 1 (single forward pass — fastest, "
                         "lowest quality). rf_denoiser is distilled for 8 (default — sweet spot). "
                         ">8 gives diminishing returns and may overshoot. The LogSNR schedule "
                         "auto-computes for any N: steps=N produces N+1 sigma values from σmax→0. "
                         "Sample wall time scales ~linearly with steps; quality/coherence improves "
                         "noticeably from 1→4 and 4→8, less from 8→16.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed (any int). Set this for reproducible outputs. "
                         "If omitted, a random seed is chosen and printed in the final 'done' line.")
    ap.add_argument("--init-noise-level", type=float, default=1.0,
                    help="σmax — the schedule's starting noise level (always honored regardless of mode). "
                         "Valid range: [0.01, ∞). Below 0.01 the script errors because the model is "
                         "undefined at t≈0 and produces NaN. With --init-audio: 0.4-0.8 is typical for "
                         "variation; 1.0 = full regeneration (init ignored). Without --init-audio: 1.0 "
                         "is standard text-to-audio; <1.0 is a creative effect (model sees pure noise but "
                         "with t=σmax timestep — mildly OOD); >1.0 is 'guidance overshoot' (more diverse, "
                         "increasingly weird).")
    ap.add_argument("--cfg", type=float, default=1.0,
                    help="Classifier-Free Guidance scale. 1.0 = off (single forward pass, the rf_denoiser "
                         "default). >1.0 pushes toward the prompt (classic CFG). [0, 1) pulls toward the "
                         "unconditional / negative branch (less prompt-aligned, more diverse). <0 actively "
                         "pushes AWAY from the prompt. Any value ≠ 1.0 costs ~2× per step (batched cond + "
                         "uncond forward).")
    ap.add_argument("--apg", type=float, default=1.0,
                    help="Adaptive Projected Guidance scale [0..1], only matters when --cfg ≠ 1.0. "
                         "1.0 = full APG (project the cond−uncond difference orthogonal to cond_denoised; "
                         "prevents over-saturation at high CFG). 0.0 = vanilla CFG (use the full "
                         "difference). Intermediate values blend the two. rf_denoiser default is 1.0.")

    # ── Memory / runtime ──────────────────────────────────────────────────────
    ap.add_argument("--free-models", action=argparse.BooleanOptionalAction, default=True,
                    help="Free each model after its last use (T5Gemma after step 2, encoder after "
                         "step 3a, DiT after step 3b) to minimize peak RAM. Lowers decode-stage peak "
                         "by 3-5 GB on sa3-medium since the DiT (2.7 GB FP16) doesn't sit idle while "
                         "the decoder runs. Use --no-free-models to keep everything resident — only "
                         "useful if you plan to call models multiple times in one run (currently the "
                         "script doesn't). Default: on.")

    # ── Output ────────────────────────────────────────────────────────────────
    ap.add_argument("--out", default=None,
                    help="Output WAV path. Relative paths land in the project's output/ "
                         "directory (auto-created); absolute paths are used as-is. "
                         "Always written as 16-bit PCM stereo at 44.1 kHz, trimmed to "
                         "exactly --seconds. If omitted, auto-named from the prompt and seed.")
    ap.add_argument("--play", action="store_true",
                    help="After writing the WAV, play it through the default output device "
                         "via the macOS `afplay` binary. Blocking — the script exits when "
                         "playback finishes. Ctrl-C stops playback and exits the script "
                         "(SIGINT is delivered to both processes).")
    args = ap.parse_args()
    if args.steps < 1:
        ap.error(f"--steps must be ≥ 1 (got {args.steps})")

    args = prompt_user_if_missing(args)
    if args.prompt is None:
        args.prompt = input("Prompt: ").strip()

    # Resolve output path — auto-name from prompt+seed when --out is not given.
    if args.out is None:
        import re as _re
        slug = _re.sub(r'[^a-z0-9]+', '_', args.prompt.lower()).strip('_')[:48]
        args.out = f"{slug}_{args.seed}.wav" if slug else f"out_{args.seed}.wav"
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / "output" / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out = str(out_path)
    # Empty prompt is allowed — T5Gemma will produce padding-only embeddings,
    # which (with the learned padding_embedding) is the unconditional case.
    dtype = mx.float32 if args.dit_dtype == "fp32" else mx.float16
    # T_lat is a DECODER-INDEPENDENT function of --seconds only (natural ceil, no
    # even-bump). This matches the TensorRT pipeline (sa3_trt.py::resolve_T_lat) so
    # the same prompt/seed/seconds yields the SAME latent — and hence the same music —
    # regardless of --decoder. The old code bumped T_lat to even for same-s, which made
    # make_noise(T_lat, seed) draw a different noise tensor and thus sample a completely
    # different latent just by switching decoder. SAME-S's internal-length constraint
    # (T_lat*17 must align to 34, i.e. even) is satisfied by the decode dispatch below:
    # odd T_lat is routed through decode_chunked, whose every window is an even kernel.
    T_lat = max(1, math.ceil(args.seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    target_dur = T_lat * SAMPLES_PER_LATENT / SAMPLE_RATE

    # Inpaint validation + parameter mapping
    inpaint_range = None
    if args.inpaint_range is not None:
        if args.init_audio is None:
            sys.exit("error: --inpaint-range requires --init-audio (the audio to inpaint into)")
        try:
            s_str, e_str = args.inpaint_range.split(",")
            inp_start_sec = float(s_str.strip()); inp_end_sec = float(e_str.strip())
        except ValueError:
            sys.exit(f"error: --inpaint-range must be 'START,END' in seconds; got {args.inpaint_range!r}")
        if not (0 <= inp_start_sec < inp_end_sec <= args.seconds):
            sys.exit(f"error: invalid inpaint range {inp_start_sec}-{inp_end_sec}s "
                     f"(must satisfy 0 <= start < end <= {args.seconds}s)")
        inp_start_lat = max(0, int(round(inp_start_sec * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        inp_end_lat   = min(T_lat, int(round(inp_end_sec   * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        inpaint_range = (inp_start_lat, inp_end_lat)

    # σmax is always taken from --init-noise-level, regardless of mode. The mode
    # only changes how σmax is *used* (init mix vs schedule-only vs inpaint init).
    sigma_max = float(args.init_noise_level)
    mode = ("inpaint" if inpaint_range else
            "audio-to-audio" if args.init_audio else "text-to-audio")

    # Reject very low σmax in any mode — the diffusion model was never trained
    # at t≈0 and produces NaN/garbage there.
    MIN_SIGMA = 0.01
    if sigma_max < MIN_SIGMA:
        sys.exit(
            f"error: --init-noise-level={sigma_max} is too low (σmax min is {MIN_SIGMA}). "
            f"The rf_denoiser model is undefined at t≈0; below {MIN_SIGMA} the schedule "
            f"collapses and the model emits NaN. Pass --init-noise-level=1.0 for normal "
            f"text-to-audio, or use the input WAV directly if you want it unchanged."
        )

    # ── Preflight: ensure every weight file this run needs is on disk BEFORE
    # we print the banner or start the wall-time clock. ensure_local() prints
    # a "↓ downloading …" line per missing file and symlinks from the HF cache.
    _preflight_download(args)

    global _CUMULATIVE_PEAK_B
    _CUMULATIVE_PEAK_B = 0
    _STAGE_PEAKS.clear()
    _reset_peak_mem()
    t_wall_start = time.time()

    print()
    banner(f"SA3 → MLX  {mode}")
    k = lambda s: dim(f"{s:>10}")
    v = lambda s, w=10: f"{s:<{w}}"
    print(f"  {k('prompt')}  {bold(repr(args.prompt))}")
    if args.negative_prompt:
        suffix = "" if args.cfg != 1.0 else dim("  (ignored: --cfg=1.0, no uncond branch)")
        print(f"  {k('neg prompt')}  {bold(repr(args.negative_prompt))}{suffix}")
    dit_decoder_line = f"  {k('dit')}  {magenta(v(args.dit))}   {k('decoder')}  {magenta(v(args.decoder))}"
    if args.init_audio:
        dit_decoder_line += f"   {k('encoder')}  {magenta(v(args.decoder))}"
    print(dit_decoder_line)
    if args.init_audio:
        print(f"  {k('init audio')}  {bold(args.init_audio)}")
        if inpaint_range:
            s0, s1 = inpaint_range
            print(f"  {k('inpaint')}  {bold(f'{inp_start_sec:.2f}s..{inp_end_sec:.2f}s')} "
                  f"{dim(f'(latent {s0}..{s1} of {T_lat})')}")
    print(f"  {k('σmax')}  {bold(f'{sigma_max:.2f}')}")
    print(f"  {k('seconds')}  {v(f'{args.seconds}s')}   {k('steps')}  {v(args.steps)}   {k('seed')}  {args.seed}")
    cfg_label = f"{args.cfg}"
    if args.cfg != 1.0:
        cfg_label += f" (apg={args.apg})"
    print(f"  {k('DiT dtype')}  {v(args.dit_dtype)}   {k('cfg')}  {cfg_label}")
    print(f"  {k('T_lat')}  {T_lat} {dim(f'({target_dur:.2f}s → trimmed to {args.seconds}s)')}")
    print()

    # ── 1. Text encoder ──
    t0 = time.time()
    t5_path = args.t5gemma_npz if args.t5gemma_npz else str(ensure_local(T5GEMMA_NPZ_REL))
    enc = T5Gemma.from_npz(t5_path)
    embeds, mask = enc.encode([args.prompt], max_len=256)
    mx.eval(embeds, mask)
    stage("[1/5]", "T5Gemma encode", (time.time()-t0)*1000, peak_b=_stage_peak_b("T5Gemma encode"))
    sub(f"embeds {embeds.shape} {embeds.dtype}")

    # ── 2. Conditioning ──
    # Conditioner weights are baked into the DiT .npz under "cond." prefix.
    t0 = time.time()
    padding_emb, secs_embedder = load_conditioner_from_npz(
        str(ensure_local(DIT_CHOICES[args.dit]["ckpt"])), prefix="cond.")
    embeds = embeds.astype(dtype)
    embeds_padded = apply_prompt_padding(embeds, mask, padding_emb.astype(dtype))
    seconds_embed = secs_embedder(args.seconds).astype(dtype)              # (1, 1, 768)
    cross_attn = mx.concatenate([embeds_padded, seconds_embed], axis=1)    # (1, 257, 768)
    global_cond = seconds_embed[:, 0, :]                                    # (1, 768)

    # Negative / null cross_attn for CFG (used whenever cfg != 1.0)
    null_cross_attn = None
    if args.cfg != 1.0:
        if args.negative_prompt:
            neg_embeds, neg_mask = enc.encode([args.negative_prompt], max_len=256)
            mx.eval(neg_embeds, neg_mask)
            neg_embeds = neg_embeds.astype(dtype)
            neg_padded = apply_prompt_padding(neg_embeds, neg_mask, padding_emb.astype(dtype))
            null_cross_attn = mx.concatenate([neg_padded, seconds_embed], axis=1)
        else:
            # Match upstream: zeros_like(cross_attn) for the uncond branch.
            null_cross_attn = mx.zeros_like(cross_attn)
        mx.eval(null_cross_attn)

    mx.eval(cross_attn, global_cond)
    stage("[2/5]", "Conditioning", (time.time()-t0)*1000, peak_b=_stage_peak_b("Conditioning"))
    sub(f"cross_attn {cross_attn.shape}   global {global_cond.shape}"
        + (f"   neg_cross_attn ready ({'prompt' if args.negative_prompt else 'zeros'})" if null_cross_attn is not None else ""))

    # T5Gemma weights are no longer needed (we have the embeddings). Free 537 MB.
    if args.free_models:
        del enc
        _free_to_pool()

    # ── 3a. (audio-to-audio only) Encode init_audio → init_latents ──
    init_latents = None
    if args.init_audio:
        stage("[3a]", f"Encoding init audio → latents")
        t0 = time.time()
        enc_model, pad_mod = load_encoder(args.decoder, mx.float32)
        sub(f"encoder load {(time.time()-t0)*1000:.0f} ms")

        t0 = time.time()
        # The ENCODER (SAME-S) needs T_audio_patches divisible by `pad_mod` (32) → an
        # even latent length. T_lat itself stays the natural-ceil (decoder-independent)
        # value; we only round the encode grid UP to the modulo, then trim the latent
        # back to T_lat. This keeps the DiT/noise path identical across decoders.
        enc_T_lat = T_lat
        if (T_lat * 16) % pad_mod != 0:
            enc_T_lat = math.ceil((T_lat * 16) / pad_mod) * pad_mod // 16
        target_samples = enc_T_lat * SAMPLES_PER_LATENT
        audio_np = read_wav(args.init_audio)                                  # (2, T_audio_in)
        if audio_np.shape[-1] >= target_samples:
            audio_np = audio_np[:, :target_samples]
            init_action = f"trimmed to {target_samples} samples"
        else:
            pad = target_samples - audio_np.shape[-1]
            audio_np = np.pad(audio_np, ((0, 0), (0, pad)), mode="constant")
            init_action = f"zero-padded by {pad} samples"
        audio_np = audio_np[None, ...]                                        # (1, 2, T)
        sub(f"read+prep ({init_action})  {(time.time()-t0)*1000:.0f} ms")

        # Patch + encode (encoder always runs FP32 — softnorm-bottleneck-sensitive)
        t0 = time.time()
        patches_np = patch_audio(audio_np, patch_size=256)                     # (1, 512, enc_T_lat*16)
        # Sanity: T_audio_patches must be divisible by encoder's required modulo
        assert patches_np.shape[-1] % pad_mod == 0, (
            f"T_audio_patches={patches_np.shape[-1]} not divisible by {pad_mod} "
            f"(decoder={args.decoder})"
        )
        init_latents = enc_model(mx.array(patches_np))[..., :T_lat]            # trim to natural T_lat
        mx.eval(init_latents)
        _stage_peak_b('Init audio encode')
        sub(f"encode  {(time.time()-t0)*1000:.0f} ms   latents {init_latents.shape}")
        init_latents = init_latents.astype(dtype)
        del enc_model

    # ── 3b. DiT pingpong sampling ──
    stage("[3/5]", f"DiT — load + sample ({args.steps} steps, σmax={sigma_max:.2f})")
    t0 = time.time(); dit_model, _ = load_dit(args.dit, T_lat=T_lat, dtype=dtype)
    _stage_peak_b('DiT load')
    sub(f"load {time.time()-t0:.1f}s")

    sigmas = build_pingpong_schedule(args.steps, sigma_max=sigma_max, use_logsnr_shift=True)
    sched_str = " · ".join(f"{float(x):.3f}" for x in sigmas)
    sub(f"schedule  {sched_str}")

    key = mx.random.key(args.seed)
    pure_noise = mx.random.normal((1, 256, T_lat), dtype=dtype, key=key)
    if init_latents is not None and inpaint_range is None:
        # rf_denoiser init_data mix (linear interpolation, matches upstream):
        #   noise = init_data * (1 - σmax) + pure_noise * σmax
        noise = init_latents * (1.0 - sigma_max) + pure_noise * sigma_max
        sub(f"init: latent * {1-sigma_max:.2f} + noise * {sigma_max:.2f}")
    else:
        noise = pure_noise
    mx.eval(noise)

    # Build local_add_cond for inpainting: [B, T_lat, 257] = (mask, masked_input)
    local_add_cond = None
    paste_back = None
    if inpaint_range is not None:
        s0, s1 = inpaint_range
        # mask: 1 = keep (context), 0 = inpaint (regenerate); shape (1, 1, T_lat)
        mask_np = np.ones((1, 1, T_lat), dtype=np.float32)
        mask_np[:, :, s0:s1] = 0.0
        mask = mx.array(mask_np)
        # masked_input: init_latents with the inpaint region zeroed
        masked_input = init_latents.astype(mx.float32) * mask  # (1, 256, T_lat)
        # concat over channel dim → (1, 257, T_lat); DiT expects batch-last-channel → transpose
        lac = mx.concatenate([mask, masked_input], axis=1).transpose(0, 2, 1).astype(dtype)
        local_add_cond = lac
        # Paste-back keeps unchanged regions bit-exact (model preserves them well but
        # paste-back makes it guaranteed).
        paste_back = (init_latents, mask)
        sub(f"local_add_cond {tuple(lac.shape)}  inpaint mask: {s0}..{s1} of {T_lat} latents "
            f"({(s1-s0)/T_lat*100:.0f}% regenerated)")

    def model_fn(x, t):
        if args.cfg == 1.0:
            return dit_model(x, t, cross_attn, global_cond, local_add_cond=local_add_cond)

        # Batched CFG: one forward over cat([x, x]) on the batch dim.
        x2 = mx.concatenate([x, x], axis=0)
        t2 = mx.concatenate([t, t], axis=0)
        cross2 = mx.concatenate([cross_attn, null_cross_attn], axis=0)
        global2 = mx.concatenate([global_cond, global_cond], axis=0)
        lac2 = None if local_add_cond is None else mx.concatenate([local_add_cond, local_add_cond], axis=0)
        v_batched = dit_model(x2, t2, cross2, global2, local_add_cond=lac2)
        cond_v, uncond_v = mx.split(v_batched, 2, axis=0)

        # Convert to denoised space (RF: denoised = x - σ * v, with σ = t)
        sigma = t.reshape(-1, 1, 1).astype(mx.float32)
        cond_d   = x.astype(mx.float32) - cond_v.astype(mx.float32)   * sigma
        uncond_d = x.astype(mx.float32) - uncond_v.astype(mx.float32) * sigma
        diff = cond_d - uncond_d

        if args.apg <= 0.0:
            cfg_diff = diff
        else:
            # Project diff onto direction orthogonal to cond_d (per-sample, over (C, T)).
            # fp32 throughout — fp16 has catastrophic-cancellation risk in the projection.
            norm = mx.sqrt((cond_d * cond_d).sum(axis=(-2, -1), keepdims=True))
            unit = cond_d / mx.maximum(norm, 1e-8)
            parallel = (diff * unit).sum(axis=(-2, -1), keepdims=True) * unit
            diff_orth = diff - parallel
            cfg_diff = diff_orth if args.apg >= 1.0 else (args.apg * diff_orth + (1.0 - args.apg) * diff)

        cfg_d = cond_d + (args.cfg - 1.0) * cfg_diff
        cfg_v = (x.astype(mx.float32) - cfg_d) / sigma
        return cfg_v.astype(x.dtype)

    # Live in-place progress while sampling (TTY only — quiet log files).
    t_step_prev = [time.time()]
    def _on_step(i: int, total: int):
        if not _USE_COLOR:   # _USE_COLOR ≡ stdout.isatty() set at top of this file
            return
        now = time.time()
        elapsed = (now - t_step_prev[0]) * 1000
        t_step_prev[0] = now
        bar_w = 20
        filled = int(round(bar_w * i / total))
        bar = cyan("█" * filled) + dim("·" * (bar_w - filled))
        sys.stdout.write(f"\r\x1b[K        {dim('sampling')} {bar} "
                         f"{bold(f'step {i}/{total}')}  {yellow(f'{elapsed:.0f} ms')}")
        sys.stdout.flush()

    t0 = time.time()
    latents = sample_flow_pingpong(model_fn, noise, sigmas, seed=args.seed + 1,
                                    paste_back=paste_back, on_step=_on_step)
    mx.eval(latents)
    sample_ms = (time.time()-t0)*1000
    # Clear the progress line, then print the final summary in its place.
    if _USE_COLOR:
        sys.stdout.write("\r\x1b[K")
    _stage_peak_b('DiT sample')
    sub(f"sample {sample_ms:.0f} ms  ({sample_ms/max(args.steps,1):.0f} ms/step)")

    # DiT weights are no longer needed (latents are computed). Free 2.7 GB for sa3-medium.
    if args.free_models:
        del dit_model, model_fn
        _free_to_pool()

    # ── 4. Decode latents → audio patches ──
    # Decoder always fp32 — SAME-S needs it (FP16 catastrophically cancels at differential attn);
    # SAME-L works at fp16 but we use fp32 for max quality.
    stage("[4/5]", f"Decoder ({args.decoder}, FP32)")
    t0 = time.time()
    dec_dtype = mx.float32
    decoder, chunk_fn, (chunk, ovl) = load_decoder(args.decoder, dec_dtype)
    _stage_peak_b('Decoder load')
    sub(f"load {(time.time()-t0)*1000:.0f} ms")
    latents_fp32 = latents.astype(dec_dtype)
    t0 = time.time()
    kernel = chunk + 2 * ovl
    if T_lat > kernel:
        # Natural-ceil T_lat may be odd; decode_chunked windows are all even kernels,
        # so odd T_lat decodes correctly here (this is the common path, incl. 30s→323).
        patches = chunk_fn(decoder, latents_fp32, chunk, ovl)
        decode_mode = f"chunked (chunk={chunk}, ovl={ovl})"
    elif T_lat % 2 == 0:
        patches = decoder(latents_fp32)
        decode_mode = "un-chunked"
    elif T_lat > 6:
        # Odd T_lat in (6, kernel]: a chunk=2,ovl=2 kernel (=6 < T_lat) keeps every
        # window even for SAME-S while never zero-padding real latents.
        patches = chunk_fn(decoder, latents_fp32, 2, 2)
        decode_mode = "chunked (chunk=2, ovl=2)"
    else:
        # Tiny odd T_lat ≤ 6 (sub-0.5s): no even chunk kernel fits. Reflect-pad one
        # latent to make T even, un-chunked decode, then trim the extra 16 patches.
        # SAME-L has no even constraint and takes this path only for symmetry.
        latents_even = mx.concatenate([latents_fp32, latents_fp32[..., -1:]], axis=-1)
        patches = decoder(latents_even)[..., : T_lat * 16]
        decode_mode = "un-chunked (odd-pad→trim)"
    mx.eval(patches)
    _stage_peak_b('Decode')
    sub(f"decode {decode_mode}  →  {(time.time()-t0)*1000:.0f} ms   patches {patches.shape}")

    # ── 5. Unpatch → audio + save WAV ──
    t0 = time.time()
    audio = patched_decode(patches, patch_size=256, channels=2)            # (1, 2, T_lat*4096)
    mx.eval(audio)
    audio_np = np.array(audio.astype(mx.float32))[0]                       # (2, T_lat*4096)
    requested_samples = int(round(args.seconds * SAMPLE_RATE))
    if audio_np.shape[-1] > requested_samples:
        audio_np = audio_np[..., :requested_samples]
    save_wav(args.out, audio_np)
    stage("[5/5]", "Unpatch + write WAV", (time.time()-t0)*1000, peak_b=_stage_peak_b("Unpatch + WAV"))
    peak = float(np.abs(audio_np).max()); rms = float(np.sqrt((audio_np**2).mean()))
    sub(f"audio {audio_np.shape}   peak {peak:.3f}   rms {rms:.3f}")

    # ── Totals ──
    total = time.time() - t_wall_start
    audio_dur = audio_np.shape[-1] / SAMPLE_RATE
    peak_str = _fmt_mem(_CUMULATIVE_PEAK_B)

    # Per-stage peak RAM table
    print()
    print(f"  {bold('Peak RAM by stage')}")
    name_w = max(len(n) for n, _ in _STAGE_PEAKS) if _STAGE_PEAKS else 0
    sorted_for_emphasis = max((b for _, b in _STAGE_PEAKS), default=0)
    for name, b in _STAGE_PEAKS:
        bar_units = int(round(b / max(sorted_for_emphasis, 1) * 24))
        bar = cyan("█" * bar_units) + dim("·" * (24 - bar_units))
        mark = bold(" ←  max") if b == sorted_for_emphasis else ""
        print(f"    {dim(name.ljust(name_w))}  {bar}  {_fmt_mem(b)}{mark}")

    print()
    rule()
    print(f"  {bold(green('done'))}   "
          f"{bold(f'{total:.2f}s')} wall  →  "
          f"{audio_dur:.1f}s audio  →  "
          f"{bold(yellow(f'{audio_dur/total:.2f}× realtime'))}   "
          f"{dim(f'peak RAM {peak_str}')}   "
          f"{dim(f'seed {args.seed}')}")
    # Prominent path block: show relative-to-cwd if shorter (more readable when
    # the user is already in the project), absolute otherwise (more copy-pasteable).
    abs_out = os.path.abspath(args.out)
    try:
        rel_out = os.path.relpath(abs_out)
    except ValueError:
        rel_out = abs_out
    shown = rel_out if len(rel_out) <= len(abs_out) and not rel_out.startswith("..") else abs_out
    print(f"  {bold(green('▸ saved'))}  {bold(shown)}   {dim(f'({abs_out})' if shown != abs_out else '')}".rstrip())
    rule()

    if args.play:
        try:
            print(f"  {bold('▶ playing')}   {args.out}   {dim('(Ctrl-C to stop)')}")
            subprocess.run(["afplay", args.out], check=False)
        except KeyboardInterrupt:
            print()  # newline after the ^C echo


if __name__ == "__main__":
    main()
