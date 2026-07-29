"""SA3 LoRA training on Apple Silicon — pure MLX, underfit conventions.

The MLX counterpart of underfit's raw-PyTorch training loop
(github.com/dada-bots/underfit, underfit/training/loop.py), built on the
trainable adapters in models/defs/lora.py. Every training convention and
default mirrors underfit (see ../TRAINING_CONVENTIONS.md):

  - rectified-flow velocity target (noise − x), signal-only masked MSE,
    per-sample-then-mean reduction
  - uniform timestep sampler + the SA3 models' "full" distribution shift
    (min_length 256, max_length 4096)
  - CFG dropout 0.1: per-sample, the whole cross-attention conditioning
    (prompt + seconds token) is zeroed; global_cond is kept
  - AdamW, single param group, torch defaults (betas 0.9/0.999, eps 1e-8,
    weight decay 0.0), no schedule, no clipping; LR has NO default and must
    be supplied (--lr)
  - adapters train in fp32 over a frozen fp16 base
  - step-driven loop; checkpoints every --checkpoint-every (1000) steps and
    at the end, saved BEFORE anything else can fail, named
    {run_label}-step={step}-epoch={epoch}.safetensors with lora_config
    metadata {rank, alpha, adapter_type, include, exclude, step, epoch,
    base_model}
  - resume offsets: explicit flags > checkpoint metadata > filename tokens
  - per-step loss_by_timestep.bin telemetry (struct "Iff": step, t_mean,
    loss_mean), flushed every 10 steps

Dataset: pre-encoded latents from scripts/pre_encode_mlx.py (npy + json
pairs), cropped to --latent-crop-length with random crop, tiny datasets
oversampled with replacement (batch_size*100 draws/epoch) — the underfit
overfitting workflow. Prompts are built per-sample with underfit's tag
augmentation. seconds_total stays the FULL source duration after cropping
(deliberate underfit convention).

Performance:
  - the train step (loss + grad + optimizer update) is mx.compile'd by
    default (MLX docs "Compile > Training graphs" pattern; --no-compile for
    the eager path). Batch shapes are constant, so it compiles once.
  - T5Gemma prompt conditioning is cached per exact prompt string (per-run,
    in-memory; --no-t5-cache to disable) — the tiny-dataset workflow repeats
    a handful of prompts, so the hit rate is ≈100% after the first epoch.
  - --grad-checkpoint recomputes each transformer block's activations in the
    backward pass (mlx-lm style mx.checkpoint) — memory for big crops on
    small Macs, at the cost of roughly one extra forward.
  - wired memory limit is raised to the device's recommended working-set
    size at startup (mx.set_wired_limit).

Usage:
    python scripts/lora_train_mlx.py --dit sm-music \
        --latents-dir output/latents/my-set --lr 1e-4 \
        --name my-lora --save-dir output/runs --max-steps 2000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
import time
import uuid
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS_DIR))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from sa3_mlx import DIT_CHOICES, load_dit  # noqa: E402
from weights import ensure_local  # noqa: E402
from sa3_mlx import T5GEMMA_NPZ_REL  # noqa: E402
from models.defs.sa3_pipeline import (  # noqa: E402
    apply_prompt_padding, load_conditioner_from_npz,
)
from models.defs.t5gemma_mlx import T5Gemma  # noqa: E402
from models.defs.training import (  # noqa: E402
    rectified_flow_loss, sample_training_timesteps, shift_training_timesteps,
)
from models.defs.lora import (  # noqa: E402
    TrainableSecondsEmbedder, inject_from_lora_config, iter_trainable_lora_layers,
    load_lora_checkpoint, load_trainable_lora_state, save_lora_checkpoint,
    underfit_lora_config,
)
from models.defs import demo_mlx as demo  # noqa: E402
from models.defs.latent_dataset import (  # noqa: E402
    PreEncodedLatentDataset, iterate_batches,
)

# underfit registry conventions per model
BASE_MODEL_NAMES = {"sm-music": "sa3-sm-music", "sm-sfx": "sa3-sm-sfx",
                    "medium": "sa3-medium"}
LATENT_CROP_DEFAULTS = {"sm-music": 1300, "sm-sfx": 1300, "medium": 4096}
DIST_SHIFT_DEFAULT = {"shift_type": "full",
                      "options": {"min_length": 256, "max_length": 4096}}
T5_CACHE_CAP = 512  # prompt-conditioning cache entries (oldest evicted first)


def _rm(path):
    """Remove a file, ignoring a missing one (temp-ckpt cleanup)."""
    try:
        os.remove(path)
    except OSError:
        pass


def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.

    Verbatim port of mlx-lm's mlx_lm/tuner/trainer.py:grad_checkpoint —
    activations of every instance of the layer's class are recomputed during
    the backward pass instead of being kept alive, trading ~one extra forward
    for a much smaller peak working set.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


class TrainBundle(nn.Module):
    """DiT + (optional) trainable seconds conditioner under one module root so
    a single value_and_grad covers every adapter parameter. The bundle's
    trainable_parameters() are the LoRA params only (bases frozen at inject)."""

    def __init__(self, dit, secs):
        super().__init__()
        self.dit = dit
        self.secs = secs  # TrainableSecondsEmbedder or None


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit", required=True, choices=list(DIT_CHOICES.keys()))
    ap.add_argument("--dit-weights", default=None,
                    help="Override the DiT weight file — REQUIRED in practice for "
                         "training parity with underfit: train on the BASE "
                         "(rectified-flow) checkpoint, not the shipped ARC "
                         "(rf_denoiser) weights that inference uses. Convert the "
                         "HF *-base model.safetensors to npz first (441-key "
                         "layout; see TRAINING_CONVENTIONS.md).")
    ap.add_argument("--latents-dir", required=True,
                    help="Root of pre-encoded npy+json pairs (scripts/pre_encode_mlx.py)")
    ap.add_argument("--lr", type=float, required=True,
                    help="Learning rate (underfit has NO default — must be supplied)")
    ap.add_argument("--name", default="underfit-run", help="Run name (defaults.ini)")
    ap.add_argument("--save-dir", default=str(REPO / "output" / "runs"))
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=10_000_000_000,
                    help="Absolute global-step target (resume-aware)")
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    # adapter config — underfit dashboard defaults
    ap.add_argument("--adapter-type", default="dora-rows")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=None, help="Default: = rank")
    ap.add_argument("--include", default=None,
                    help="Comma-separated substring filters (underfit convention)")
    ap.add_argument("--exclude", default=None,
                    help="Comma-separated; default = underfit's dashboard exclude list")
    ap.add_argument("--no-conditioner-lora", action="store_true",
                    help="Skip the seconds-conditioner adapter (underfit adapts it "
                         "by default — e.g. the plini checkpoint carries its delta)")
    ap.add_argument("--bora-mode", default="speed", choices=["speed", "memory"],
                    help="bora/bora-xs forward: 'speed' (default) caches W0² per "
                         "adapted layer (+1 weight copy) for the reformulated "
                         "no-materialize forward; 'memory' keeps the original "
                         "full-weight forward with no cache. Ignored for other "
                         "adapter types.")
    # data
    ap.add_argument("--latent-crop-length", type=int, default=None,
                    help="Default per model: 1300 (sm-*) / 4096 (medium)")
    ap.add_argument("--random-crop", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prompt-config", default=None,
                    help="JSON file with underfit prompt_config (trigger, tag options…)")
    # schedule / conditioning
    ap.add_argument("--timestep-sampler", default="uniform",
                    choices=["uniform", "logit_normal", "trunc_logit_normal",
                             "log_snr", "log_snr_uniform"])
    ap.add_argument("--dist-shift", default="full",
                    choices=["none", "full", "flux", "logsnr"])
    ap.add_argument("--cfg-dropout-prob", type=float, default=0.1)
    ap.add_argument("--use-effective-length", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Shift timesteps by the EFFECTIVE latent length "
                         "ceil(int(seconds_total*44100)/4096) per sample instead "
                         "of the crop length (underfit model flag "
                         "use_effective_length_for_schedule; True in the SA3 "
                         "templates). Only affects the 'full'/'flux'/'logsnr' shift.")
    # optimizer (underfit reads these from optimizer_configs; torch AdamW defaults)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999,
                    help="AdamW β2 (torch default 0.999; SA3 templates use 0.95)")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="AdamW decoupled weight decay (torch default 0.0; SA3 "
                         "templates use 0.01)")
    ap.add_argument("--gradient-clip-val", type=float, default=0.0,
                    help="Global grad-norm clip over the adapter params before "
                         "the optimizer step (0 = off). underfit's dashboard "
                         "passes 1.0; the grad_norm metric is then post-clip, "
                         "matching underfit's torch loop.")
    # LR schedule (underfit's optimizer_configs.diffusion.scheduler; InverseLR
    # with warmup — at step 0 the SA3 template LR is ~5e-7, not the nominal lr)
    ap.add_argument("--lr-scheduler", default="none", choices=["none", "inverse"],
                    help="'inverse' = SAT-dev InverseLR: "
                         "lr(step) = (1-warmup^(step+1)) * "
                         "max(final, lr*(1+step/inv_gamma)^-power)")
    ap.add_argument("--inv-gamma", type=float, default=1e6)
    ap.add_argument("--lr-power", type=float, default=0.5)
    ap.add_argument("--lr-warmup", type=float, default=0.0,
                    help="InverseLR exponential-warmup base in [0,1) (template 0.995)")
    ap.add_argument("--lr-final", type=float, default=0.0)
    # performance
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                    help="mx.compile the train step (loss+grad+optimizer update; "
                         "MLX 'Compile > Training graphs' pattern). "
                         "--no-compile restores the eager path.")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="Gradient-checkpoint every transformer block (mlx-lm "
                         "style mx.checkpoint): activations are recomputed in "
                         "the backward pass — much lower peak memory for big "
                         "crops, ~one extra forward of compute")
    ap.add_argument("--t5-cache", action=argparse.BooleanOptionalAction, default=True,
                    help="Cache the padded T5Gemma prompt conditioning per exact "
                         f"prompt string (per-run, in-memory, max {T5_CACHE_CAP} "
                         "entries)")
    # resume
    ap.add_argument("--lora-ckpt-path", default=None)
    ap.add_argument("--step-offset", type=int, default=None)
    ap.add_argument("--epoch-offset", type=int, default=None)
    # demos (underfit convention: RF Euler inference on the base model + trained
    # LoRA, decode → mp3, baseline at step 0 then every --demo-every steps)
    ap.add_argument("--demo-every", type=int, default=0,
                    help="Generate demos every N steps (0 = off). A baseline "
                         "set is also rendered at step 0.")
    ap.add_argument("--demo-config", default=None,
                    help="JSON list of demo entries. Each: {prompt, cfg=7, "
                         "seed=0, steps=50, lora_strength?, lora_interval_max?}.")
    ap.add_argument("--demo-decoder", default=None,
                    choices=["same-s", "same-l"],
                    help="Decoder for demos (default: the DiT's default_decoder).")
    ap.add_argument("--demo-dir", default=None,
                    help="Where to write demo_<i>_<step>.mp3 (default: cwd, "
                         "which the dashboard sets to <run>/demos/).")
    ap.add_argument("--arc-weights", default=None,
                    help="DiT weights for ARC demos (entries with arc=true): the "
                         "SHIPPED rf_denoiser npz, NOT the training base. Default "
                         "auto-downloads models/mlx/dit_<dit>_f16.npz. The trained "
                         "LoRA is merged into a fresh copy of these weights and "
                         "sampled with the pingpong integrator.")
    return ap.parse_args()


def resolve_offsets(args, ckpt_config) -> tuple[int, int]:
    """Underfit's resume ladder: explicit flags > checkpoint metadata >
    filename step=/epoch= tokens > zero."""
    if args.step_offset is not None or args.epoch_offset is not None:
        return int(args.step_offset or 0), int(args.epoch_offset or 0)
    if ckpt_config:
        step = ckpt_config.get("step")
        epoch = ckpt_config.get("epoch")
        if step is not None:
            return int(step), int(epoch or 0)
    if args.lora_ckpt_path:
        name = Path(args.lora_ckpt_path).name
        ms = re.search(r"step=(\d+)", name)
        me = re.search(r"epoch=(\d+)", name)
        if ms:
            return int(ms.group(1)), int(me.group(1)) if me else 0
    return 0, 0


def build_conditioning(t5, padding_emb, prompts, max_len=256,
                       cache=None, cache_stats=None):
    """Frozen prompt-side conditioning: T5Gemma embeddings with the learned
    padding embedding applied. Returns [B, 256, 768] fp32 (the trainable
    seconds token is concatenated later, inside the grad scope).

    When ``cache`` (a dict) is given, the padded embedding is cached per
    EXACT prompt string. The cache is per-run only (in-memory, never
    persisted) and stores evaluated [1, max_len, 768] fp32 mx arrays; it is
    capped at T5_CACHE_CAP entries, evicting oldest-inserted first. The
    tiny-dataset underfit workflow repeats a handful of prompts, so after
    the first epoch the T5Gemma forward is skipped entirely (~100% hits)."""
    prompts = list(prompts)
    if cache is None:
        embeds, mask = t5.encode(prompts, max_len=max_len)
        mx.eval(embeds, mask)
        padded = apply_prompt_padding(embeds.astype(mx.float32), mask,
                                      padding_emb.astype(mx.float32))
        return mx.stop_gradient(padded)

    missing = [p for p in dict.fromkeys(prompts) if p not in cache]
    if cache_stats is not None:
        n_miss = sum(1 for p in prompts if p in set(missing))
        cache_stats["misses"] += n_miss
        cache_stats["hits"] += len(prompts) - n_miss
    if missing:
        embeds, mask = t5.encode(missing, max_len=max_len)
        padded = apply_prompt_padding(embeds.astype(mx.float32), mask,
                                      padding_emb.astype(mx.float32))
        mx.eval(padded)
        for i, p in enumerate(missing):
            while len(cache) >= T5_CACHE_CAP:
                cache.pop(next(iter(cache)))  # evict oldest
            row = padded[i:i + 1]
            mx.eval(row)
            cache[p] = row
        print(f"  t5-cache: encoded {len(missing)} new prompt(s) "
              f"({len(cache)} cached)")
    return mx.stop_gradient(mx.concatenate([cache[p] for p in prompts],
                                           axis=0))


def main():
    args = parse_args()
    np_rng = np.random.default_rng(args.seed)
    mx.random.seed(args.seed)

    # Let Metal wire (pin) up to the recommended working set — avoids paging
    # stalls when the training working set approaches physical memory.
    if hasattr(mx, "set_wired_limit") and hasattr(mx, "device_info"):
        wired = mx.device_info().get("max_recommended_working_set_size")
        if wired:
            mx.set_wired_limit(int(wired))

    alpha = float(args.alpha) if args.alpha is not None else float(args.rank)
    lora_config = underfit_lora_config()
    lora_config.update({"adapter_type": args.adapter_type, "rank": args.rank,
                        "alpha": alpha})
    if args.include is not None:
        lora_config["include"] = [s.strip() for s in args.include.split(",") if s.strip()]
    if args.exclude is not None:
        lora_config["exclude"] = [s.strip() for s in args.exclude.split(",") if s.strip()]

    crop_len = args.latent_crop_length or LATENT_CROP_DEFAULTS[args.dit]
    base_model = BASE_MODEL_NAMES[args.dit]

    # ── run dirs (underfit layout: <save>/<name>/<uuid8>/checkpoints) ────────
    run_label = re.sub(r"-\d{14}$", "", args.name) if args.name else None
    session = uuid.uuid4().hex[:8]
    ckpt_dir = Path(args.save_dir) / args.name / session / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Engine: MLX  ·  device: {mx.default_device()}")
    print(f"run: {args.name}  session {session}")
    print(f"checkpoints → {ckpt_dir}")

    # ── dataset ──────────────────────────────────────────────────────────────
    prompt_config = None
    if args.prompt_config:
        prompt_config = json.loads(Path(args.prompt_config).read_text())
    dataset = PreEncodedLatentDataset(
        args.latents_dir, crop_len, random_crop=args.random_crop,
        prompt_config=prompt_config, seed=args.seed)
    print(f"dataset: {len(dataset)} latent file(s), crop {crop_len} latents"
          + ("  (oversampling with replacement — tiny dataset)"
             if len(dataset) < args.batch_size else ""))
    # Batches per epoch — mirrors iterate_batches: tiny sets (< batch) oversample
    # to batch_size*100 draws (→ 100 batches); else ceil(n / batch). Used as the
    # tqdm total so the progress bar (and the dashboard's collapse signature)
    # matches underfit's torch dataloader.
    _n_idx = (args.batch_size * 100 if len(dataset) < args.batch_size
              else len(dataset))
    steps_per_epoch = max(1, (_n_idx + args.batch_size - 1) // args.batch_size)

    # ── models ───────────────────────────────────────────────────────────────
    # LoRA training uses the BASE checkpoint (rectified_flow), NOT the shipped
    # ARC weights inference uses. Default to the base-model npz (auto-downloaded
    # from HF via weights.TRAINING_BASE); --dit-weights overrides with a path.
    t0 = time.time()
    import importlib
    if args.dit_weights:
        base_path = args.dit_weights
    else:
        base_path = str(ensure_local(f"models/mlx/dit_{args.dit}-base_f16.npz"))
    mod = importlib.import_module(DIT_CHOICES[args.dit]["loader"])
    dit_model = mod.load_dit(base_path, T_lat=crop_len, dtype=mx.float16,
                             compile_=False)
    cond_src = base_path
    print(f"DiT loaded ({time.time()-t0:.1f}s, fp16 base, T_lat={crop_len})")
    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))  # frozen
    padding_emb, secs_embedder = load_conditioner_from_npz(cond_src, prefix="cond.")

    # ── inject adapters ─────────────────────────────────────────────────────
    report, saved_config = inject_from_lora_config(
        dit_model, lora_config, checkpoint_prefix="model.",
        bora_mode=args.bora_mode)
    print(f"lora: {report.layer_count} DiT layer(s), {report.adapter_type}, "
          f"rank {args.rank}, alpha {alpha:g} "
          f"({report.trainable_parameters/1e6:.2f}M trainable)")

    if args.grad_checkpoint:
        # Patch the TransformerBlock class (both DiT defs expose the blocks at
        # dit.transformer.layers) so every block recomputes in backward.
        grad_checkpoint(dit_model.transformer.layers[0])
        print(f"grad-checkpoint: on "
              f"({len(dit_model.transformer.layers)} transformer blocks)")

    secs_module = None
    if not args.no_conditioner_lora:
        secs_module = TrainableSecondsEmbedder(secs_embedder.W, secs_embedder.b)
        try:
            cond_report, _ = inject_from_lora_config(
                secs_module, lora_config,
                checkpoint_prefix="conditioners.seconds_total.",
                bora_mode=args.bora_mode)
            print(f"lora: conditioner seconds embedder adapted "
                  f"({cond_report.layer_count} layer)")
        except ValueError:
            secs_module = None  # filtered out by include/exclude
    bundle = TrainBundle(dit_model, secs_module)

    # ── resume ───────────────────────────────────────────────────────────────
    step_offset = epoch_offset = 0
    if args.lora_ckpt_path:
        sd, ckpt_cfg = load_lora_checkpoint(args.lora_ckpt_path)
        restored = load_trainable_lora_state(bundle, sd)
        step_offset, epoch_offset = resolve_offsets(args, ckpt_cfg)
        print(f"resumed {restored} adapter layer(s) from "
              f"{Path(args.lora_ckpt_path).name} (step {step_offset}, "
              f"epoch {epoch_offset})")
    else:
        step_offset, epoch_offset = resolve_offsets(args, None)

    max_new_steps = max(0, args.max_steps - step_offset)

    # ── optimizer: AdamW, single group; betas/eps/wd from config (torch
    # AdamW's decoupled weight decay — matches MLX's AdamW update exactly) ─────
    def scheduled_lr(step):
        """SAT-dev InverseLR at 0-based step (== torch last_epoch on a fresh run,
        whose scheduler restarts even on resume). --lr-scheduler none → flat."""
        if args.lr_scheduler != "inverse":
            return args.lr
        warmup_factor = 1.0 - args.lr_warmup ** (step + 1)
        lr_mult = (1.0 + step / args.inv_gamma) ** (-args.lr_power)
        return warmup_factor * max(args.lr_final, args.lr * lr_mult)

    optimizer = optim.AdamW(learning_rate=scheduled_lr(0),
                            betas=[args.beta1, args.beta2],
                            eps=args.eps, weight_decay=args.weight_decay)

    # ── training-time local conditioning (underfit/upstream convention) ──────
    # The SA3 models are the diffusion_cond_inpaint variant: the trainer feeds
    # pure generation as an all-ONES inpaint mask + zero context (verified at
    # 80.7 dB forward parity vs the torch loop; the inference path's
    # local_add_cond=None ≡ zeros is an inference-only convention and trains
    # against the wrong conditioning regime — 4x loss difference).
    lac_const = mx.array(
        np.concatenate([np.ones((1, 1, crop_len), np.float32),
                        np.zeros((1, 256, crop_len), np.float32)],
                       axis=1).transpose(0, 2, 1)).astype(mx.float16)

    # ── loss (adapters in the bundle are the only trainable params) ──────────
    # `seconds_in` is the raw seconds batch [B] fp32 when the trainable
    # seconds conditioner is active; with --no-conditioner-lora it is the
    # PRE-computed frozen seconds token [B, 1, 768] (built outside the step so
    # the compiled step stays pure mx — no numpy inside the traced graph).
    def loss_fn(bundle, latents, timesteps, loss_mask, prompt_cond,
                seconds_in, drop_mask):
        if bundle.secs is not None:
            sec_tok = bundle.secs(seconds_in)                 # [B, 1, 768] fp32
        else:
            sec_tok = seconds_in                              # frozen, precomputed
        cross = mx.concatenate([prompt_cond, sec_tok.astype(mx.float32)], axis=1)
        # CFG dropout (dit.py:441): whole cross_attn → zeros per sample;
        # global_cond (the seconds token) is kept.
        cross = mx.where(drop_mask[:, None, None], mx.zeros_like(cross), cross)
        global_cond = sec_tok[:, 0, :]
        cross16 = cross.astype(mx.float16)
        global16 = global_cond.astype(mx.float16)

        def model_fn(noised, t):
            return bundle.dit(noised, t.astype(noised.dtype), cross16, global16,
                              local_add_cond=lac_const)

        return rectified_flow_loss(model_fn, latents, timesteps,
                                   loss_mask=loss_mask)

    value_and_grad = nn.value_and_grad(bundle, loss_fn)

    # ── train step (compiled by default — MLX "Compile > Training graphs") ──
    # Captured state: model params (the LoRA adapters the optimizer mutates),
    # optimizer moments, and the global PRNG key — rectified_flow_loss draws
    # its noise INSIDE the step. Everything np/python (timestep sampling, CFG
    # dropout draw, T5 encode, telemetry) stays OUTSIDE the step; batch
    # shapes/dtypes are constant (fixed crop + batch), so this traces once.
    # optimizer.init() materializes the moment arrays up front so the captured
    # state structure never changes (no recompile on step 2).
    optimizer.init(bundle.trainable_parameters())
    state = [bundle.state, optimizer.state, mx.random.state]
    clip_val = float(args.gradient_clip_val or 0.0)

    def _global_l2(tree):
        """sqrt(Σ ‖leaf‖²) over a param/grad tree in fp32 — the single global L2
        across all adapter leaves that underfit's _compute_grad_and_lora_norms
        reports (loop.py:243)."""
        total = mx.zeros((), dtype=mx.float32)
        for _, v in tree_flatten(tree):
            total = total + (v.astype(mx.float32) ** 2).sum()
        return mx.sqrt(total)

    def train_step(latents, timesteps, loss_mask, prompt_cond, seconds_in,
                   drop):
        loss, grads = value_and_grad(bundle, latents, timesteps, loss_mask,
                                     prompt_cond, seconds_in, drop)
        if clip_val > 0.0:
            grads, _ = optim.clip_grad_norm(grads, clip_val)
        # grad_norm on the POST-clip grads, lora_magnitude on the PRE-update
        # params — the order + definition underfit's torch loop uses (clip →
        # compute norms → optimizer.step, loop.py:628-637).
        grad_norm = _global_l2(grads)
        lora_mag = _global_l2(bundle.trainable_parameters())
        optimizer.update(bundle, grads)
        return loss, grad_norm, lora_mag

    if args.compile:
        train_step = partial(mx.compile, inputs=state, outputs=state)(train_step)

    # ── telemetry (underfit's loss_by_timestep.bin, struct "Iff") ────────────
    tele = open("loss_by_timestep.bin", "ab")

    def checkpoint(step, epoch):
        fname = (f"{run_label}-step={step}-epoch={epoch}.safetensors"
                 if run_label else f"step={step}-epoch={epoch}.safetensors")
        path = ckpt_dir / fname
        save_lora_checkpoint(
            bundle, path,
            include=lora_config.get("include"),
            exclude=lora_config.get("exclude"),
            extra_config={"step": int(step), "epoch": int(epoch),
                          "base_model": base_model})
        print(f"\n  ✓ checkpoint {path.name}")
        return path

    # ── train loop (step-driven; epoch = one dataloader pass) ────────────────
    raw_step = 0
    epoch = epoch_offset
    last_saved_step = None
    t5_cache = {} if args.t5_cache else None
    t5_stats = {"hits": 0, "misses": 0}

    # ── demos (underfit RF + ARC paths) ──────────────────────────────────────
    # RF/base entries: Euler inference on the BASE model + trained LoRA. ARC
    # entries (arc=true): the trained LoRA merged into the SHIPPED rf_denoiser
    # weights, sampled with the pingpong integrator — underfit's "train on base,
    # demo on ARC" story (its torch loop swaps in the full ARC model; here we
    # load the ARC npz with the LoRA merged at load). Both decode to mp3.
    # Baseline at the start, then every --demo-every steps + a final render.
    # Idempotent (a resume never re-renders an existing demo).
    demo_entries = None
    if args.demo_every > 0:
        if args.demo_config:
            demo_entries = json.loads(Path(args.demo_config).read_text())
            n_arc = sum(1 for e in demo_entries if e.get("arc"))
            print(f"demos: {len(demo_entries)} prompt(s) "
                  f"({n_arc} ARC) every {args.demo_every} "
                  f"step(s) → {args.demo_dir or '.'}")
        else:
            print("WARNING: --demo-every set but no --demo-config — demos disabled")
    demo_dir = args.demo_dir or "."
    demo_decoder_name = args.demo_decoder or DIT_CHOICES[args.dit]["default_decoder"]
    _model_default_decoder = DIT_CHOICES[args.dit]["default_decoder"]
    if demo_entries and demo_decoder_name != _model_default_decoder:
        print(f"demos: decoding with {demo_decoder_name.upper()} instead of the "
              f"model's {_model_default_decoder.upper()} (faster demos)")
    _demo_dec = {}  # lazy: (decoder, chunk_fn, chunk_cfg)

    def _demo_T_lat(entry):
        """Per-demo latent length. The dashboard sets `duration` (seconds) on a
        demo entry only when it isn't at the crop length (see index.html
        demo_cond serialization) — full-length demos carry it, crop-length ones
        omit it. The DiT is length-agnostic (dit_mlx*), so one loaded model
        serves both without reloading."""
        dur = entry.get("duration")
        if dur:
            return max(2, int(round(float(dur) * demo.SAMPLE_RATE
                                    / demo.SAMPLES_PER_LATENT)))
        return crop_len

    def _demo_conditioning(prompt, seconds_val):
        """cross_attn [1,257,768] fp16 + global_cond [1,768] fp16 for one demo
        prompt. Model-independent (T5Gemma + the trained seconds conditioner),
        so RF (base) and ARC demos share it."""
        prompt_cond = build_conditioning(t5, padding_emb, [prompt],
                                         cache=t5_cache, cache_stats=t5_stats)
        if bundle.secs is not None:
            sec_tok = bundle.secs(mx.array([seconds_val], dtype=mx.float32))
        else:
            sec_tok = secs_embedder([seconds_val])
        sec_tok = mx.stop_gradient(sec_tok.astype(mx.float32))
        cross = mx.concatenate([prompt_cond, sec_tok], axis=1).astype(mx.float16)
        gcond = sec_tok[:, 0, :].astype(mx.float16)
        return cross, gcond

    def _run_arc_demos(step, arc_pending, decoder, chunk_fn, chunk_cfg):
        """Merge the trained LoRA into a fresh copy of the SHIPPED rf_denoiser
        weights and sample with the pingpong integrator. Loaded once for the
        whole ARC set (length-agnostic), freed after (peak = base + ARC DiT)."""
        arc_path = args.arc_weights or str(
            ensure_local(f"models/mlx/dit_{args.dit}_f16.npz"))
        # Scratch LoRA for merging into the ARC DiT — write it to the SESSION
        # dir (ckpt_dir.parent), NOT the scanned checkpoints/ dir, so it never
        # shows up as a checkpoint (dotfile too, belt + suspenders).
        tmp_ckpt = ckpt_dir.parent / f".arc_demo_tmp_{step:08d}.safetensors"
        save_lora_checkpoint(
            bundle, tmp_ckpt, include=lora_config.get("include"),
            exclude=lora_config.get("exclude"),
            extra_config={"step": int(step), "base_model": base_model})
        t_a = time.time()
        try:
            arc_dit = mod.load_dit(arc_path, T_lat=crop_len, dtype=mx.float16,
                                   compile_=False, lora_paths=[str(tmp_ckpt)],
                                   lora_strength=1.0, lora_log=lambda *a, **k: None)
            print(f"  ARC model ({args.dit}) + LoRA merged "
                  f"({time.time()-t_a:.1f}s)")
        except Exception as e:
            print(f"  ARC demos skipped: could not load "
                  f"{Path(arc_path).name} with LoRA merged: {e}", flush=True)
            _rm(tmp_ckpt)
            return
        try:
            for i, entry in arc_pending:
                prompt = entry.get("prompt", "")
                cfg = float(entry.get("cfg", 1))
                seed = int(entry.get("seed", 0))
                steps = int(entry.get("steps", 8))
                T_lat = _demo_T_lat(entry)
                seconds_val = T_lat * demo.SAMPLES_PER_LATENT / demo.SAMPLE_RATE
                cross, gcond = _demo_conditioning(prompt, seconds_val)
                null = mx.zeros_like(cross) if cfg != 1.0 else None
                model_fn = demo.make_rf_model_fn(arc_dit, cross, gcond, cfg=cfg,
                                                 null_cross_attn=null)
                sigmas = demo.build_sigmas(steps)
                noise = mx.random.normal((1, 256, T_lat), dtype=mx.float16,
                                         key=mx.random.key(seed))
                print(f"  ▸ ARC demo {i}: {T_lat} latents (~{seconds_val:.0f}s audio), "
                      f"{steps} steps", flush=True)
                try:
                    t_gen = time.time()
                    latent = demo.pingpong_sample(model_fn, noise, sigmas, seed=seed,
                                                  desc=f"ARC demo {i}")
                    audio = demo.decode_latents(decoder, chunk_fn, chunk_cfg,
                                                latent, T_lat)
                    meta = {"prompt": prompt, "cfg": cfg, "seed": seed,
                            "steps": steps, "step": int(step), "arc": True}
                    demo.save_demo_mp3(audio, i, step, demo.SAMPLE_RATE,
                                       demo_dir, meta)
                    print(f"  ♪ ARC demo {i} @ step {step}: {prompt[:44]!r} "
                          f"→ demo_{i}_{step:08d}.mp3 ({time.time()-t_gen:.0f}s)", flush=True)
                except Exception as e:
                    print(f"  ARC demo {i} failed: {e}", flush=True)
        finally:
            del arc_dit
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            _rm(tmp_ckpt)

    def run_demos(step):
        if not demo_entries:
            return
        os.makedirs(demo_dir, exist_ok=True)
        pending = [(i, e) for i, e in enumerate(demo_entries)
                   if not os.path.exists(
                       os.path.join(demo_dir, f"demo_{i}_{step:08d}.mp3"))]
        if not pending:
            return
        if "dec" not in _demo_dec:
            from sa3_mlx import load_decoder
            t_d = time.time()
            _demo_dec["dec"] = load_decoder(demo_decoder_name, dtype=mx.float32)
            print(f"  demo decoder {demo_decoder_name} loaded ({time.time()-t_d:.1f}s)")
        decoder, chunk_fn, chunk_cfg = _demo_dec["dec"]
        adapters = list(iter_trainable_lora_layers(bundle))
        rf_pending = [(i, e) for i, e in pending if not e.get("arc")]
        arc_pending = [(i, e) for i, e in pending if e.get("arc")]

        # ── RF/base demos (Euler on the base model + trained LoRA) ──
        for i, entry in rf_pending:
            prompt = entry.get("prompt", "")
            cfg = float(entry.get("cfg", 7))
            seed = int(entry.get("seed", 0))
            steps = int(entry.get("steps", 50))
            strength = entry.get("lora_strength")
            interval = entry.get("lora_interval_max")
            T_lat = _demo_T_lat(entry)
            seconds_val = T_lat * demo.SAMPLES_PER_LATENT / demo.SAMPLE_RATE
            cross, gcond = _demo_conditioning(prompt, seconds_val)
            null = mx.zeros_like(cross) if cfg != 1.0 else None
            model_fn = demo.make_rf_model_fn(bundle.dit, cross, gcond, cfg=cfg,
                                             null_cross_attn=null)
            sigmas = demo.build_sigmas(steps)
            noise = mx.random.normal((1, 256, T_lat), dtype=mx.float16,
                                     key=mx.random.key(seed))
            snap = before = None
            if interval is not None:
                snap = demo.snapshot_adapters(adapters)
                s_on = float(strength) if strength is not None else 1.0

                def before(_i, sigma, s_on=s_on, interval=interval, snap=snap):
                    demo.scale_adapters(snap, s_on if sigma <= interval else 0.0)
            elif strength is not None and float(strength) != 1.0:
                snap = demo.snapshot_adapters(adapters)
                demo.scale_adapters(snap, float(strength))
            print(f"  ▸ demo {i}: {T_lat} latents (~{seconds_val:.0f}s audio), "
                  f"{steps} steps", flush=True)
            try:
                t_gen = time.time()
                latent = demo.rf_euler_sample(model_fn, noise, sigmas,
                                              before_step=before, desc=f"demo {i}")
                audio = demo.decode_latents(decoder, chunk_fn, chunk_cfg, latent, T_lat)
                meta = {"prompt": prompt, "cfg": cfg, "seed": seed,
                        "steps": steps, "step": int(step)}
                if strength is not None:
                    meta["lora_strength"] = strength
                if interval is not None:
                    meta["lora_interval_max"] = interval
                demo.save_demo_mp3(audio, i, step, demo.SAMPLE_RATE, demo_dir, meta)
                print(f"  ♪ demo {i} @ step {step}: {prompt[:48]!r} "
                      f"→ demo_{i}_{step:08d}.mp3 ({time.time()-t_gen:.0f}s)", flush=True)
            except Exception as e:
                print(f"  demo {i} failed: {e}", flush=True)
            finally:
                if snap is not None:
                    demo.scale_adapters(snap, 1.0)  # restore trained factors

        # ── ARC demos (pingpong on the shipped ARC weights + trained LoRA) ──
        if arc_pending:
            _run_arc_demos(step, arc_pending, decoder, chunk_fn, chunk_cfg)

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()  # measure the training loop, not weight loading
    t_start = time.time()
    print(f"training: max {max_new_steps} new step(s) "
          f"(global target {args.max_steps}), lr {args.lr:g}, "
          f"batch {args.batch_size}, cfg-dropout {args.cfg_dropout_prob}, "
          f"compile {'on' if args.compile else 'off'}")
    run_demos(step_offset)  # baseline (untrained / resumed state)
    try:
        while raw_step < max_new_steps:
            # tqdm per epoch — byte-identical to underfit's torch loop
            # (loop.py:516) so the dashboard's progress collapse, history parser,
            # and grad-norm / lora-magnitude charts work unchanged. desc
            # "Step N, Epoch E"; postfix train/loss,lr,grad_norm,lora_magnitude.
            pbar = tqdm(
                iterate_batches(dataset, args.batch_size, shuffle=True,
                                seed=args.seed + epoch),
                total=steps_per_epoch,
                desc=f"Step {raw_step + step_offset}, Epoch {epoch}",
                mininterval=0, miniters=1, file=sys.stdout,
            )
            for batch in pbar:
                if raw_step >= max_new_steps:
                    break
                global_step = raw_step + step_offset + 1

                latents = mx.array(batch["latents"]).astype(mx.float16)
                loss_mask = mx.array(batch["padding_mask"])
                seconds = mx.array(np.asarray(batch["seconds_total"],
                                              dtype=np.float32))

                # timesteps: sampler + model-config distribution shift. The shift
                # sequence length is the EFFECTIVE latent length per sample when
                # --use-effective-length (underfit use_effective_length_for_schedule,
                # True in the SA3 templates), else the crop length.
                t_np = sample_training_timesteps(
                    args.timestep_sampler, latents.shape[0], rng=np_rng)
                if args.dist_shift != "none":
                    if args.use_effective_length:
                        seq_len = np.asarray(
                            [int(math.ceil(int(float(s) * 44100) / 4096))
                             for s in batch["seconds_total"]])
                    else:
                        seq_len = latents.shape[-1]
                    t_np = shift_training_timesteps(
                        t_np, seq_len, shift_type=args.dist_shift,
                        options=DIST_SHIFT_DEFAULT["options"]
                        if args.dist_shift == "full" else None)
                timesteps = mx.array(t_np)

                prompt_cond = build_conditioning(t5, padding_emb,
                                                 batch["prompt"],
                                                 cache=t5_cache,
                                                 cache_stats=t5_stats)
                drop = mx.array((np_rng.random(latents.shape[0])
                                 < args.cfg_dropout_prob))
                if bundle.secs is not None:
                    seconds_in = seconds
                else:
                    # frozen conditioner: build the seconds token OUTSIDE the
                    # (possibly compiled) step — see loss_fn
                    seconds_in = mx.stop_gradient(secs_embedder(
                        [float(s) for s in batch["seconds_total"]]))

                # InverseLR (or flat): the scheduler steps per optimizer step,
                # last_epoch = raw_step (0-based new-step index), matching torch's
                # fresh-on-resume scheduler. Setting learning_rate updates the leaf
                # in optimizer.state, which the compiled step reads via inputs=state.
                lr_now = scheduled_lr(raw_step)
                if args.lr_scheduler != "none":
                    optimizer.learning_rate = lr_now

                loss, grad_norm, lora_mag = train_step(
                    latents, timesteps, loss_mask, prompt_cond, seconds_in, drop)
                mx.eval(state, loss, grad_norm, lora_mag)

                raw_step += 1
                loss_v = float(loss)
                tele.write(struct.pack("Iff", global_step,
                                       float(t_np.mean()), loss_v))
                if raw_step % 10 == 0:
                    tele.flush()

                # Metrics on the tqdm postfix — the dashboard parses
                # train/loss=, train/lr=, train/grad_norm=, train/lora_magnitude=
                # per step (server.py _HISTORY_RE/_LR_RE/_GN_RE/_LM_RE), and the
                # "Step N," prefix gives it the authoritative global step.
                pbar.set_postfix({
                    "train/loss": f"{loss_v:.6f}",
                    "train/lr": f"{lr_now:.3e}",
                    "train/grad_norm": f"{float(grad_norm):.6f}",
                    "train/lora_magnitude": f"{float(lora_mag):.6f}",
                })
                pbar.set_description(f"Step {global_step}, Epoch {epoch}")

                save_now = (args.checkpoint_every > 0
                            and global_step % args.checkpoint_every == 0)
                demo_now = (args.demo_every > 0
                            and global_step % args.demo_every == 0)
                if save_now or demo_now:
                    # Clear + disable the bar so checkpoint/demo prints land on
                    # their own lines (matches the torch loop, loop.py:680).
                    pbar.clear()
                    pbar.disable = True
                    try:
                        if save_now:
                            checkpoint(global_step, epoch)  # save BEFORE demos
                            last_saved_step = global_step
                        if demo_now:
                            run_demos(global_step)
                    finally:
                        pbar.disable = False
                        pbar.refresh()
            pbar.close()
            epoch += 1
    except KeyboardInterrupt:
        print("\ninterrupted — saving final checkpoint")

    global_step = raw_step + step_offset
    if raw_step > 0 and global_step != last_saved_step:
        checkpoint(global_step, epoch)
    if raw_step > 0:
        run_demos(global_step)  # final render
    tele.close()
    if t5_cache is not None:
        print(f"t5 cache: {t5_stats['hits']} hit(s) / "
              f"{t5_stats['misses']} miss(es), {len(t5_cache)} cached")
    print(f"peak memory: {mx.get_peak_memory()/2**30:.2f} GB"
          + ("  (grad-checkpoint on)" if args.grad_checkpoint else ""))
    print(f"done: {raw_step} step(s) in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
