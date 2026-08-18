# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stability AI — vendored from github.com/Stability-AI/stable-audio-3
"""SA3-sm-music DiT — MLX implementation.

Standard MHA (NOT differential — that's sa3-medium).
embed_dim=1024, depth=20, num_heads=16, head_dim=64, ff_inner=4096.

Forward signature matches dit_torch.py:
    forward(x, t, cross_attn_cond, global_cond) -> v
"""

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .checkpoint_security import load_torch_checkpoint


# Constants (sa3-sm-music)
IO_CHANNELS = 256
EMBED_DIM = 1024
DEPTH = 20
NUM_HEADS = 16
HEAD_DIM = 64
ROPE_DIMS = 32
COND_TOKEN_DIM = 768
GLOBAL_COND_DIM = 768
LOCAL_ADD_COND_DIM = 257
NUM_MEMORY_TOKENS = 64
FF_INNER = 4096
TIMESTEP_FEAT_DIM = 256
NORM_EPS = 1e-5
QK_NORM_EPS = 1e-6


# ---- Standard (NOT differential) SelfAttention ----

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_qkv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM, bias=False)
        self.to_out = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.q_norm = nn.RMSNorm(HEAD_DIM, eps=QK_NORM_EPS)
        self.k_norm = nn.RMSNorm(HEAD_DIM, eps=QK_NORM_EPS)
        self.scale = HEAD_DIM ** -0.5

    def __call__(self, x):
        B, T, _ = x.shape
        H, D = NUM_HEADS, HEAD_DIM
        qkv = self.to_qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        def to_heads(t): return t.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        q = to_heads(q); k = to_heads(k); v = to_heads(v)
        q = self.q_norm(q); k = self.k_norm(k)
        q = mx.fast.rope(q, ROPE_DIMS, traditional=False, base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, ROPE_DIMS, traditional=False, base=10000.0, scale=1.0, offset=0)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, EMBED_DIM)
        return self.to_out(out)


# ---- Standard CrossAttention (separate to_q, to_kv) ----

class CrossAttention(nn.Module):
    def __init__(self, context_dim=EMBED_DIM):
        super().__init__()
        self.to_q  = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.to_kv = nn.Linear(context_dim, 2 * context_dim, bias=False)
        self.to_out = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.q_norm = nn.RMSNorm(HEAD_DIM, eps=QK_NORM_EPS)
        self.k_norm = nn.RMSNorm(HEAD_DIM, eps=QK_NORM_EPS)
        self.scale = HEAD_DIM ** -0.5

    def __call__(self, x, context):
        B, Tx, _ = x.shape
        Tc = context.shape[1]
        H, D = NUM_HEADS, HEAD_DIM
        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = mx.split(kv, 2, axis=-1)
        def to_q_heads(t): return t.reshape(B, Tx, H, D).transpose(0, 2, 1, 3)
        def to_c_heads(t): return t.reshape(B, Tc, H, D).transpose(0, 2, 1, 3)
        q = to_q_heads(q); k = to_c_heads(k); v = to_c_heads(v)
        q = self.q_norm(q); k = self.k_norm(k)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, Tx, EMBED_DIM)
        return self.to_out(out)


# ---- GLU FF (same shape as medium but different inner dim) ----

class _GLUWrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(EMBED_DIM, 2 * FF_INNER, bias=True)

    def __call__(self, x):
        x = self.proj(x)
        x, gate = mx.split(x, 2, axis=-1)
        return x * nn.silu(gate)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = [_GLUWrap(), None, nn.Linear(FF_INNER, EMBED_DIM, bias=True), None]

    def __call__(self, x):
        x = self.ff[0](x)
        x = self.ff[2](x)
        return x


class LocalEmbedSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = [
            nn.Linear(LOCAL_ADD_COND_DIM, EMBED_DIM, bias=True),
            None,
            nn.Linear(EMBED_DIM, EMBED_DIM, bias=True),
        ]

    def __call__(self, x):
        x = self.seq[0](x); x = nn.silu(x); x = self.seq[2](x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = nn.RMSNorm(EMBED_DIM, eps=NORM_EPS)
        self.self_attn = SelfAttention()
        self.cross_attend_norm = nn.RMSNorm(EMBED_DIM, eps=NORM_EPS)
        self.cross_attn = CrossAttention()
        self.ff_norm = nn.RMSNorm(EMBED_DIM, eps=NORM_EPS)
        self.ff = FeedForward()
        self.to_scale_shift_gate = mx.zeros((6 * EMBED_DIM,))
        self.to_local_embed = LocalEmbedSeq()

    def __call__(self, x, context, global_cond, local_emb_padded):
        ss = (self.to_scale_shift_gate + global_cond)[:, None, :]
        scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = mx.split(ss, 6, axis=-1)

        residual = x
        h = self.pre_norm(x)
        h = h * (1 + scale_self) + shift_self
        h = self.self_attn(h)
        h = h * mx.sigmoid(1 - gate_self)
        x = h + residual

        x = x + self.cross_attn(self.cross_attend_norm(x), context)
        x = x + local_emb_padded

        residual = x
        h = self.ff_norm(x)
        h = h * (1 + scale_ff) + shift_ff
        h = self.ff(h)
        h = h * mx.sigmoid(1 - gate_ff)
        x = h + residual
        return x


class ContinuousTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.project_in = nn.Linear(IO_CHANNELS, EMBED_DIM, bias=False)
        self.project_out = nn.Linear(EMBED_DIM, IO_CHANNELS, bias=False)
        self.memory_tokens = mx.zeros((NUM_MEMORY_TOKENS, EMBED_DIM))
        self.global_cond_embedder = [
            nn.Linear(EMBED_DIM, EMBED_DIM),
            None,
            nn.Linear(EMBED_DIM, 6 * EMBED_DIM),
        ]
        self.layers = [TransformerBlock() for _ in range(DEPTH)]

    def __call__(self, x, context, global_embed, local_add_cond_zeros):
        B, T, _ = x.shape
        x = self.project_in(x)
        mem = mx.broadcast_to(self.memory_tokens[None], (B, NUM_MEMORY_TOKENS, EMBED_DIM))
        x = mx.concatenate([mem, x], axis=1)
        g = self.global_cond_embedder[0](global_embed)
        g = nn.silu(g)
        g = self.global_cond_embedder[2](g)
        for layer in self.layers:
            local_emb = layer.to_local_embed(local_add_cond_zeros)
            pad = mx.zeros((B, NUM_MEMORY_TOKENS, EMBED_DIM), dtype=local_emb.dtype)
            local_emb_padded = mx.concatenate([pad, local_emb], axis=1)
            x = layer(x, context, g, local_emb_padded)
        x = x[:, NUM_MEMORY_TOKENS:, :]
        return self.project_out(x)


class ExpoFourierFeatures(nn.Module):
    def __init__(self, dim=TIMESTEP_FEAT_DIM, min_freq=0.5, max_freq=10000.0):
        super().__init__()
        half = dim // 2
        ramp = mx.linspace(0.0, 1.0, half)
        log_min, log_max = math.log(min_freq), math.log(max_freq)
        freqs = mx.exp(ramp * (log_max - log_min) + log_min)
        self.freqs = freqs * 2 * math.pi

    def __call__(self, t):
        args = t[:, None] * self.freqs
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class DiT(nn.Module):
    def __init__(self, T_lat=320):
        super().__init__()
        self.T_lat = T_lat

        self.preprocess_conv  = nn.Conv1d(IO_CHANNELS, IO_CHANNELS, kernel_size=1, bias=False)
        self.postprocess_conv = nn.Conv1d(IO_CHANNELS, IO_CHANNELS, kernel_size=1, bias=False)

        self.to_cond_embed = [
            nn.Linear(COND_TOKEN_DIM, EMBED_DIM, bias=False),
            None,
            nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
        ]
        self.to_global_embed = [
            nn.Linear(GLOBAL_COND_DIM, EMBED_DIM, bias=False),
            None,
            nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
        ]
        self.to_timestep_embed = [
            nn.Linear(TIMESTEP_FEAT_DIM, EMBED_DIM, bias=True),
            None,
            nn.Linear(EMBED_DIM, EMBED_DIM, bias=True),
        ]
        self.timestep_features = ExpoFourierFeatures(TIMESTEP_FEAT_DIM, 0.5, 10000.0)
        self.transformer = ContinuousTransformer()
        self._local_zeros_1 = mx.zeros((1, T_lat, LOCAL_ADD_COND_DIM))

    def __call__(self, x, t, cross_attn_cond_raw, global_cond_raw, local_add_cond=None):
        """
        local_add_cond: optional [B, T_lat, 257] — concat of (inpaint_mask, inpaint_masked_input)
            in batch-last-channel layout. Pass None for vanilla text-to-audio.
        """
        B = x.shape[0]

        c = self.to_cond_embed[0](cross_attn_cond_raw)
        c = nn.silu(c)
        context = self.to_cond_embed[2](c)

        g = self.to_global_embed[0](global_cond_raw)
        g = nn.silu(g)
        global_pre = self.to_global_embed[2](g)

        tf = self.timestep_features(t)
        tf = self.to_timestep_embed[0](tf)
        tf = nn.silu(tf)
        t_embed = self.to_timestep_embed[2](tf)

        global_embed = global_pre + t_embed

        x_lc = x.transpose(0, 2, 1)
        x_pp = self.preprocess_conv(x_lc) + x_lc

        if local_add_cond is None:
            # Zero local-add-cond at the INPUT length (not the baked self.T_lat),
            # so one loaded model runs any sequence length (see dit_mlx_medium).
            local = mx.zeros((B, x.shape[-1], LOCAL_ADD_COND_DIM))
        else:
            local = local_add_cond
        h = self.transformer(x_pp, context, global_embed, local)
        out = self.postprocess_conv(h) + h
        return out.transpose(0, 2, 1)


# ---- Weight conversion (torch ckpt → MLX) ----

def convert_weights_from_torch_ckpt(ckpt_path):
    """Load sa3-sm-music ckpt (torch.load), strip 'model.model.' prefix, remap to MLX layout."""
    import numpy as np

    raw = load_torch_checkpoint(ckpt_path)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    prefix = "model.model."
    sd_up = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    out = {}
    for sk, t in sd_up.items():
        if not isinstance(t, torch.Tensor):
            continue
        arr = t.cpu().float().numpy()

        # Conv1d weights: PyTorch [out, in, k] -> MLX [out, k, in]
        if sk in ("preprocess_conv.weight", "postprocess_conv.weight"):
            arr = arr.transpose(0, 2, 1)
            out[sk] = mx.array(arr)
            continue

        # RMSNorm: .gamma -> .weight
        if sk.endswith(".gamma"):
            base = sk[: -len(".gamma")]
            out[f"{base}.weight"] = mx.array(arr)
            continue

        # to_local_embed.{0,2} -> to_local_embed.seq.{0,2}
        if ".to_local_embed.0." in sk:
            sk = sk.replace(".to_local_embed.0.", ".to_local_embed.seq.0.")
        elif ".to_local_embed.2." in sk:
            sk = sk.replace(".to_local_embed.2.", ".to_local_embed.seq.2.")

        out[sk] = mx.array(arr)
    return out


def load_dit(weights_path, T_lat=320, dtype=mx.float16, compile_=False):
    """Build MLX DiT and load weights.

    weights_path can be either:
      - the sa3-sm-music torch ckpt (slow; converts at load time), OR
      - a pre-converted MLX file (.npz or .safetensors — fast path).
    """
    p = str(weights_path)
    if p.endswith(".npz") or p.endswith(".safetensors"):
        wd = dict(mx.load(p))
    else:
        wd = convert_weights_from_torch_ckpt(p)

    model = DiT(T_lat=T_lat)
    wd_list = [(k, v.astype(dtype)) for k, v in wd.items()]
    model.load_weights(wd_list, strict=False)
    # Drop source-dict references BEFORE materialization (frees the .npz-dtype copy
    # if it differs from the target dtype — keeps peak ≈ 1× weights instead of 2-3×).
    del wd, wd_list
    import gc; gc.collect()
    mx.eval(model.parameters())
    if compile_:
        model = mx.compile(model)
    return model


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    ckpt = repo / "models" / "original_ckpts" / "sa3-sm-music" / "ckpt"
    print(f"Loading MLX SA3-small DiT from {ckpt}...")
    m = load_dit(ckpt, T_lat=320, dtype=mx.float32, compile_=False)
    x = mx.random.normal((1, 256, 320))
    t = mx.array([0.5])
    cross = mx.random.normal((1, 257, 768))
    gcond = mx.random.normal((1, 768))
    v = m(x, t, cross, gcond)
    mx.eval(v)
    print(f"v: {v.shape}, mean={float(v.mean()):.4f}, std={float(v.std()):.4f}")
