"""Training-time demos for lora_train_mlx.py — MLX RF inference on the base model.

Mirrors the RF/V path of underfit's demo step (underfit/training/demo_step.py):
plain Euler velocity integration (x += dt·v) over the sa3 sampling schedule,
optional CFG (uncond = cross→zeros, guidance done in denoised space — identical
to sa3_mlx), decode through the SAME-S/SAME-L decoder, peak-normalized int16 →
mp3 with a json sidecar, files named ``demo_<i>_<step:08d>.mp3`` in the demo
dir, idempotent per step (a resume never re-renders an existing demo).

Why Euler and not the pingpong sampler sa3_mlx ships: the trainer finetunes the
rectified_flow BASE checkpoint, whose velocity field is integrated with a plain
ODE step. sa3_mlx's ``sample_flow_pingpong`` is the 8-step re-noising sampler for
the DISTILLED rf_denoiser (ARC) weights, which is a different model. Conditioning,
CFG math and ``local_add_cond=None`` (the inference convention — training uses the
all-ones inpaint mask, see TRAINING_CONVENTIONS §9) are otherwise identical.

Per-entry ``lora_strength`` and ``lora_interval_max`` (σ-interval gating) are
applied by scaling each adapter's ``lora_B`` (``M_xs`` for -xs) — pre-norm, which
is exactly underfit's ``lora_strength`` semantics for lora/DoRA (it scales the
low-rank delta inside V = W0 + scaling·strength·B@A before the DoRA norm). This
touches no forward code, so the parity-verified training path is untouched.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from .sa3_pipeline import build_pingpong_schedule, patched_decode

SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096


# ── sampler ────────────────────────────────────────────────────────────────

def make_rf_model_fn(dit, cross_attn, global_cond, cfg=1.0,
                     null_cross_attn=None, apg=1.0):
    """Velocity model_fn(x, t) for the base rectified_flow DiT.

    cfg == 1.0: a single conditioned forward. cfg != 1.0: the sa3_mlx dual pass —
    batch cond+uncond, convert to denoised space (denoised = x − σ·v, σ = t),
    guide there (optional APG orthogonal projection), convert back to velocity.
    local_add_cond stays None (inference convention).
    """
    def model_fn(x, t):
        if cfg == 1.0:
            return dit(x, t, cross_attn, global_cond, local_add_cond=None)
        x2 = mx.concatenate([x, x], axis=0)
        t2 = mx.concatenate([t, t], axis=0)
        cross2 = mx.concatenate([cross_attn, null_cross_attn], axis=0)
        global2 = mx.concatenate([global_cond, global_cond], axis=0)
        v_batched = dit(x2, t2, cross2, global2, local_add_cond=None)
        cond_v, uncond_v = mx.split(v_batched, 2, axis=0)
        sigma = t.reshape(-1, 1, 1).astype(mx.float32)
        cond_d = x.astype(mx.float32) - cond_v.astype(mx.float32) * sigma
        uncond_d = x.astype(mx.float32) - uncond_v.astype(mx.float32) * sigma
        diff = cond_d - uncond_d
        if apg <= 0.0:
            cfg_diff = diff
        else:
            flat_c = cond_d.reshape(cond_d.shape[0], -1)
            flat_d = diff.reshape(diff.shape[0], -1)
            unit = flat_c / (mx.linalg.norm(flat_c, axis=1, keepdims=True) + 1e-8)
            proj = mx.sum(flat_d * unit, axis=1, keepdims=True) * unit
            diff_orth = (flat_d - proj).reshape(diff.shape)
            cfg_diff = diff_orth if apg >= 1.0 else (apg * diff_orth + (1.0 - apg) * diff)
        cfg_d = cond_d + (cfg - 1.0) * cfg_diff
        cfg_v = (x.astype(mx.float32) - cfg_d) / sigma
        return cfg_v.astype(x.dtype)

    return model_fn


def rf_euler_sample(model_fn, x, sigmas, before_step=None, desc=None):
    """Euler ODE integration of the RF velocity field: x_{i+1} = x_i + Δt·v.

    ``sigmas`` is (steps+1,) descending from σ_max to 0. ``before_step(i, σ)``,
    if given, runs right before each velocity eval (per-step LoRA gating).
    ``desc``, if given, shows a tqdm step bar — a full-length demo is minutes of
    generation on MLX, so the bar makes clear it's progressing, not hung.
    """
    num_steps = sigmas.shape[0] - 1
    steps = range(num_steps)
    if desc is not None:
        steps = tqdm(steps, desc=desc, file=sys.stdout, leave=False, mininterval=0.5)
    for i in steps:
        t_curr = sigmas[i]
        t_next = sigmas[i + 1]
        if before_step is not None:
            before_step(i, float(t_curr))
        t_vec = t_curr * mx.ones((x.shape[0],), dtype=x.dtype)
        v = model_fn(x, t_vec)
        x = x + (t_next - t_curr).astype(x.dtype) * v
        mx.eval(x)
    return x


def pingpong_sample(model_fn, x, sigmas, seed=0, before_step=None, desc=None):
    """Ping-pong (rf_denoiser) sampler for ARC demos — the distilled 8-step
    re-noising integrator sa3_mlx ships (``sample_flow_pingpong``). Same
    ``model_fn(x, t)`` / ``x`` / ``sigmas`` interface as ``rf_euler_sample``.

    Base demos integrate the rectified_flow BASE model with plain Euler; ARC
    demos run the SHIPPED rf_denoiser (ARC) weights with the trained LoRA merged
    in, so they use the pingpong sampler instead (matches underfit's ARC demo,
    which overrides diffusion_objective to rf_denoiser)."""
    from .sa3_pipeline import sample_flow_pingpong
    bar = (tqdm(total=int(sigmas.shape[0]) - 1, desc=desc, file=sys.stdout,
                leave=False, mininterval=0.5) if desc is not None else None)
    try:
        return sample_flow_pingpong(
            model_fn, x, sigmas, seed=seed,
            before_step=(lambda i: before_step(i, float(sigmas[i])))
            if before_step is not None else None,
            on_step=((lambda *_: bar.update(1)) if bar is not None else None))
    finally:
        if bar is not None:
            bar.close()


# ── decode ─────────────────────────────────────────────────────────────────

def decode_latents(decoder, chunk_fn, chunk_cfg, latents, T_lat):
    """Latent [1,256,T] → audio np.float32 [2, T·4096]. Mirrors sa3_mlx's
    chunk/kernel dispatch (odd T_lat routes through even-kernel chunked decode)."""
    latents = latents.astype(mx.float32)
    chunk, ovl = chunk_cfg
    kernel = chunk + 2 * ovl
    if T_lat > kernel:
        patches = chunk_fn(decoder, latents, chunk, ovl)
    elif T_lat % 2 == 0:
        patches = decoder(latents)
    elif T_lat > 6:
        patches = chunk_fn(decoder, latents, 2, 2)
    else:
        latents_even = mx.concatenate([latents, latents[..., -1:]], axis=-1)
        patches = decoder(latents_even)[..., : T_lat * 16]
    audio = patched_decode(patches, patch_size=256, channels=2)  # (1,2,T*4096)
    mx.eval(audio)
    return np.array(audio.astype(mx.float32))[0]


# Warn once (not per demo) when ffmpeg is missing and we fall back to WAV.
_WARNED_NO_FFMPEG = False


def save_demo_mp3(audio_np, demo_index, step, sample_rate, out_dir, meta=None):
    """Peak-normalized int16 → mp3 (or WAV without ffmpeg) + json sidecar.

    underfit's on-disk demo format. Atomic (temp file → rename); idempotent
    (skips if the demo already exists, in either extension).

    MP3 needs ffmpeg. If ffmpeg isn't on PATH we save a WAV instead so a
    training run never crashes on the demo step — the dashboard plays WAV
    fine, it's just larger on disk. A single warning is printed the first
    time this happens.
    """
    global _WARNED_NO_FFMPEG
    final_mp3 = os.path.join(out_dir, f"demo_{demo_index}_{step:08d}.mp3")
    final_wav = os.path.join(out_dir, f"demo_{demo_index}_{step:08d}.wav")
    # Idempotent: a resume never re-renders an existing demo (either format).
    if os.path.exists(final_mp3):
        return final_mp3
    if os.path.exists(final_wav):
        return final_wav

    peak = float(np.abs(audio_np).max())
    pcm = (audio_np / max(peak, 1e-8) * 32767.0).astype(np.int16)  # (2, T)

    def _write_wav(path):
        import wave
        with wave.open(path, "wb") as w:
            w.setnchannels(pcm.shape[0])
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.T.tobytes())  # (T, C) interleaved

    if shutil.which("ffmpeg") is not None:
        tmp_wav = os.path.join(out_dir, f".tmp_demo_{demo_index}_{step:08d}.wav")
        tmp_mp3 = os.path.join(out_dir, f".tmp_demo_{demo_index}_{step:08d}.mp3")
        _write_wav(tmp_wav)
        subprocess.run(
            ["ffmpeg", "-i", tmp_wav, "-q:a", "0", "-y", tmp_mp3, "-loglevel", "error"],
            check=True,
        )
        os.replace(tmp_mp3, final_mp3)
        os.remove(tmp_wav)
        final = final_mp3
    else:
        if not _WARNED_NO_FFMPEG:
            print(
                "WARNING: ffmpeg not found — saving training demos as WAV instead of MP3.\n"
                "         MP3 demos need ffmpeg; install it for smaller files: brew install ffmpeg",
                file=sys.stderr, flush=True,
            )
            _WARNED_NO_FFMPEG = True
        tmp_wav = os.path.join(out_dir, f".tmp_demo_{demo_index}_{step:08d}.wav")
        _write_wav(tmp_wav)
        os.replace(tmp_wav, final_wav)  # atomic
        final = final_wav

    if meta is not None:
        with open(os.path.join(out_dir, f"demo_{demo_index}_{step:08d}.json"), "w") as f:
            json.dump(meta, f)
    return final


# ── per-entry LoRA strength / σ-interval (scale lora_B pre-norm) ─────────────

def snapshot_adapters(layers):
    """Freeze the current adapter factors so per-step scaling starts from the
    trained values each time (avoids compounding across steps)."""
    snap = []
    for layer in layers:
        if hasattr(layer, "M_xs"):
            snap.append((layer, "M_xs", layer.M_xs))
        elif hasattr(layer, "lora_B"):
            snap.append((layer, "lora_B", layer.lora_B))
    return snap


def scale_adapters(snapshot, strength):
    """Set every adapter's low-rank factor to (trained value)·strength — matches
    underfit's pre-norm lora_strength for lora/DoRA. strength 1.0 restores."""
    for layer, attr, original in snapshot:
        setattr(layer, attr, original * strength if strength != 1.0 else original)


def build_sigmas(steps):
    return build_pingpong_schedule(steps)
