# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stability AI — vendored from github.com/Stability-AI/stable-audio-3
"""End-to-end SA3 inference pieces in MLX — no stable-audio-tools dependency.

Bundles four small components ported from stable-audio-tools:
  - SecondsTotalEmbedder  (ExpoFourierFeatures + Linear, conditioner for "seconds_total")
  - apply_prompt_padding  (replace padded T5Gemma positions with the learned padding_embedding)
  - build_pingpong_schedule + sample_flow_pingpong (rf_denoiser sampler, 8 steps)
  - patched_decode  (PatchedPretransform.decode inverse: [B, 512, T*16] → [B, 2, T*4096])

All shapes assume sa3-medium / sa3-sm-music: io_channels=256, patch_size=256,
channels=2 stereo, cond_token_dim=768.
"""

from __future__ import annotations
import math
from typing import Optional

import numpy as np
import mlx.core as mx

from .checkpoint_security import load_torch_checkpoint


# ─────────────────────────────────────────────────────────────────────────
# Conditioner: seconds_total → 768-dim embedding
# ─────────────────────────────────────────────────────────────────────────

def expo_fourier_features(t: mx.array, dim: int = 256,
                           min_freq: float = 0.5, max_freq: float = 10000.0) -> mx.array:
    """Mirrors stable_audio_tools.models.blocks.ExpoFourierFeatures (used for
    seconds_total). `t` shape (B,) → output (B, dim). Computed in fp32."""
    t = t.astype(mx.float32).reshape(-1, 1)            # (B, 1)
    half = dim // 2
    ramp = mx.arange(half, dtype=mx.float32) / max(half - 1, 1)
    freqs = mx.exp(ramp * (math.log(max_freq) - math.log(min_freq)) + math.log(min_freq))
    args = t * freqs * 2 * math.pi                     # (B, half)
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)  # (B, dim)


class SecondsTotalEmbedder:
    """NumberConditioner({"min_val":0, "max_val":384, fourier_features_type:"expo"}) → 768."""

    def __init__(self, weight: mx.array, bias: mx.array,
                 min_val: float = 0.0, max_val: float = 384.0, fourier_dim: int = 256):
        self.W = weight                  # (768, 256)
        self.b = bias                    # (768,)
        self.min_val = min_val
        self.max_val = max_val
        self.fourier_dim = fourier_dim

    def __call__(self, seconds: float | list[float]) -> mx.array:
        if isinstance(seconds, (int, float)):
            seconds = [float(seconds)]
        s = mx.array(seconds, dtype=mx.float32)
        s = mx.clip(s, self.min_val, self.max_val)
        norm = (s - self.min_val) / (self.max_val - self.min_val)
        ff = expo_fourier_features(norm, dim=self.fourier_dim)     # (B, 256)
        out = ff @ self.W.T + self.b                                # (B, 768)
        return out[:, None, :]                                       # (B, 1, 768)


def apply_prompt_padding(embeds: mx.array, mask: mx.array, padding_embedding: mx.array) -> mx.array:
    """Replace padded positions in T5Gemma output with the learned padding_embedding.

    embeds  : (B, S, 768) — T5Gemma last_hidden_state, dtype likely fp16
    mask    : (B, S)      int32, 1 = real, 0 = pad
    padding_embedding : (768,) learned from sa3 conditioner
    """
    m = mask.astype(embeds.dtype)[..., None]            # (B, S, 1)
    pe = padding_embedding.astype(embeds.dtype).reshape(1, 1, -1)
    return embeds * m + pe * (1 - m)


# ─────────────────────────────────────────────────────────────────────────
# Schedule + Pingpong sampler (rectified-flow / rf_denoiser)
# ─────────────────────────────────────────────────────────────────────────

def logsnr_shift(t: mx.array, anchor_logsnr: float = -6.2, logsnr_end: float = 2.0) -> mx.array:
    """Default sampling schedule shift for sa3 (rate=0 → no seq_len dependence).

    Maps t∈[0,1] through log-SNR space: t_out = sigmoid(-logsnr) where
    logsnr = logsnr_end - t * (logsnr_end - anchor_logsnr). Preserves endpoints.
    """
    t32 = t.astype(mx.float32)
    logsnr = logsnr_end - t32 * (logsnr_end - anchor_logsnr)
    t_out = mx.sigmoid(-logsnr)
    # preserve endpoints exactly
    t_out = mx.where(t32 <= 0, mx.zeros_like(t_out), t_out)
    t_out = mx.where(t32 >= 1, mx.ones_like(t_out), t_out)
    return t_out


def build_pingpong_schedule(steps: int, sigma_max: float = 1.0,
                             use_logsnr_shift: bool = True) -> mx.array:
    """Linear t from sigma_max → 0 in (steps+1) points, optionally warped by LogSNRShift.

    Returns mx.array shape (steps+1,) of float32.
    """
    t = mx.linspace(sigma_max, 0.0, steps + 1, dtype=mx.float32)
    if use_logsnr_shift:
        t = logsnr_shift(t)
        # Re-anchor start to sigma_max
        t = mx.concatenate([mx.array([sigma_max], dtype=mx.float32), t[1:]], axis=0)
    return t


def sample_flow_pingpong(model_fn, x: mx.array, sigmas: mx.array, seed: int = 0,
                         paste_back: tuple | None = None, on_step=None,
                         before_step=None) -> mx.array:
    """Ping-pong sampler for rf_denoiser models.

    Per step i (i = 0 .. num_steps-1):
        denoised = x - t_curr * v(x, t_curr)
        x_next   = (1 - t_next) * denoised + t_next * noise

    `model_fn(x, t_array)` should return the model's velocity prediction.
    `sigmas` is the schedule of shape (steps+1,) with sigmas[-1] = 0.

    `before_step(i)`, if provided, is called with the 0-based step index right
    before each model_fn call — used for per-step model state such as LoRA
    step gating (lora_merge.LoraStepPlan.sync). `on_step(i+1, total)` fires
    after each step (progress display).

    `paste_back`, if provided, is `(init_latents, inpaint_mask)` and instructs
    the sampler to overwrite non-masked positions with the init at the END
    (mask=1 = keep, mask=0 = inpaint). Mid-loop paste-back is also commonly used
    for repaint-style sampling but isn't needed here since the model is trained
    with inpaint conditioning.
    """
    key = mx.random.key(seed)
    num_steps = sigmas.shape[0] - 1
    for i in range(num_steps):
        t_curr = sigmas[i]
        t_next = sigmas[i + 1]
        t_tensor = t_curr * mx.ones((x.shape[0],), dtype=x.dtype)
        if before_step is not None:
            before_step(i)
        v = model_fn(x, t_tensor)
        denoised = x - t_curr.astype(x.dtype) * v
        if i < num_steps - 1 and float(t_next) > 0.0:
            key, sub = mx.random.split(key)
            noise = mx.random.normal(x.shape, dtype=x.dtype, key=sub)
            x = (1.0 - t_next).astype(x.dtype) * denoised + t_next.astype(x.dtype) * noise
        else:
            x = denoised
        mx.eval(x)
        if on_step is not None:
            on_step(i + 1, num_steps)

    if paste_back is not None:
        init_latents, mask = paste_back
        # mask shape (B, 1, T_lat); 1=keep init, 0=inpaint (use generated)
        m = mask.astype(x.dtype)
        x = init_latents.astype(x.dtype) * m + x * (1.0 - m)
        mx.eval(x)
    return x


# ─────────────────────────────────────────────────────────────────────────
# Patched pretransform decode: [B, 512, T*16] → [B, 2, T*4096]
# ─────────────────────────────────────────────────────────────────────────

def patched_decode(patches: mx.array, patch_size: int = 256, channels: int = 2) -> mx.array:
    """Mirror PatchedPretransform.decode rearrange("b (c h) l -> b c (l h)", h=patch_size)."""
    B, CH, L = patches.shape
    assert CH == channels * patch_size, f"Expected {channels*patch_size} channels, got {CH}"
    x = patches.reshape(B, channels, patch_size, L)     # b c h l
    x = x.transpose(0, 1, 3, 2)                          # b c l h
    return x.reshape(B, channels, L * patch_size)        # b c (l*h)


# ─────────────────────────────────────────────────────────────────────────
# Loader: pull conditioner weights from a sa3 ckpt
# ─────────────────────────────────────────────────────────────────────────

def load_conditioner_from_npz(npz_path: str, prefix: str = "") -> tuple[mx.array, SecondsTotalEmbedder]:
    """Loader for pre-extracted conditioner tensors. `prefix` lets the conditioner
    be baked into a larger npz (e.g. the DiT weight file) under a common prefix
    like ``cond.`` so we ship one npz instead of two."""
    z = np.load(npz_path)
    pad = mx.array(z[f"{prefix}padding_embedding"].astype(np.float32))
    W   = mx.array(z[f"{prefix}seconds_total_weight"].astype(np.float32))
    b   = mx.array(z[f"{prefix}seconds_total_bias"].astype(np.float32))
    embedder = SecondsTotalEmbedder(W, b, min_val=0.0, max_val=384.0, fourier_dim=256)
    return pad, embedder


def load_conditioner_from_sa3_ckpt(ckpt_path: str) -> tuple[mx.array, SecondsTotalEmbedder]:
    """Slow path: pull conditioner weights directly from the full sa3 ckpt.
    Use load_conditioner_from_npz() when a pre-extracted .npz is available."""
    PREFIX = "conditioner.conditioners."
    PAD_KEY = PREFIX + "prompt.padding_embedding"
    LIN_W_KEY = PREFIX + "seconds_total.embedder.embedding.1.weight"
    LIN_B_KEY = PREFIX + "seconds_total.embedder.embedding.1.bias"

    if str(ckpt_path).endswith(".safetensors"):
        import safetensors.torch as st
        sd = st.load_file(str(ckpt_path))
        get = lambda k: sd[k].cpu().float().numpy()
    else:
        ck = load_torch_checkpoint(str(ckpt_path))
        sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        get = lambda k: sd[k].cpu().float().numpy()

    pad = mx.array(get(PAD_KEY).astype(np.float32))
    W   = mx.array(get(LIN_W_KEY).astype(np.float32))
    b   = mx.array(get(LIN_B_KEY).astype(np.float32))
    embedder = SecondsTotalEmbedder(W, b, min_val=0.0, max_val=384.0, fourier_dim=256)
    return pad, embedder
