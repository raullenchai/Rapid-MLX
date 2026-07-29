"""SAME-S Decoder — MLX implementation.

Mirrors `same_s_decoder_torch.py` exactly. 6 blocks, dim=768, NUM_HEADS=12,
differential attention, chunk_midpoint_shift, single learnable new_tokens,
WNConv1d output mapping. All FF use SiLU (no sinusoidal gate).

Input:  [B, 256, T_lat]
Output: [B, 512, T_lat*16]
"""

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


# Constants (SAME-S from sa3-sm-music)
LATENT_DIM       = 256
DIM              = 768
NUM_HEADS        = 12
HEAD_DIM         = 64
ROPE_DIMS        = 32
NUM_BLOCKS       = 6
FF_INNER         = 2304            # ff_mult=3 → 2304
OUT_CHANNELS     = 512
STRIDE           = 16
SUB_CHUNK_SIZE   = STRIDE + 1      # 17
CHUNK_SIZE_LAT   = 32
EFFECTIVE_CHUNK  = CHUNK_SIZE_LAT + CHUNK_SIZE_LAT // STRIDE  # 34
SHIFT            = EFFECTIVE_CHUNK // 2                       # 17
PAD_MODULO       = CHUNK_SIZE_LAT // STRIDE                   # 2
SIN_PER_POS      = SUB_CHUNK_SIZE - 1                         # 16
QK_NORM_EPS      = 1e-3

_WEIGHT_SEARCH = [
    Path(__file__).parent.parent / "mlx" / "same_s_decoder_f32.npz",
]


class DyT(nn.Module):
    """gamma * tanh(alpha * x) + beta."""
    def __init__(self, dim):
        super().__init__()
        self.alpha = mx.array([1.0])
        self.gamma = mx.ones((dim,))
        self.beta = mx.zeros((dim,))

    def __call__(self, x):
        return self.gamma * mx.tanh(self.alpha * x) + self.beta


class DifferentialAttention(nn.Module):
    """Differential SDPA: out = SDPA(q,k,v) - SDPA(q_diff,k_diff,v).

    to_qkv chunk(5) order: q, k, v, q_diff, k_diff.
    """
    def __init__(self):
        super().__init__()
        self.to_qkv = nn.Linear(DIM, 5 * DIM, bias=False)
        self.to_out = nn.Linear(DIM, DIM, bias=False)
        self.q_norm = DyT(HEAD_DIM)
        self.k_norm = DyT(HEAD_DIM)
        self.scale = HEAD_DIM ** -0.5

    def __call__(self, x):
        B, T, _ = x.shape
        H, D = NUM_HEADS, HEAD_DIM
        qkv = self.to_qkv(x)
        q, k, v, q_diff, k_diff = mx.split(qkv, 5, axis=-1)

        def to_heads(t):
            return t.reshape(B, T, H, D).transpose(0, 2, 1, 3)

        q      = to_heads(q)
        k      = to_heads(k)
        v      = to_heads(v)
        q_diff = to_heads(q_diff)
        k_diff = to_heads(k_diff)

        q      = self.q_norm(q)
        k      = self.k_norm(k)
        q_diff = self.q_norm(q_diff)
        k_diff = self.k_norm(k_diff)

        q      = mx.fast.rope(q,      ROPE_DIMS, traditional=False, base=10000.0, scale=1.0, offset=0)
        k      = mx.fast.rope(k,      ROPE_DIMS, traditional=False, base=10000.0, scale=1.0, offset=0)
        q_diff = mx.fast.rope(q_diff, ROPE_DIMS, traditional=False, base=10000.0, scale=1.0, offset=0)
        k_diff = mx.fast.rope(k_diff, ROPE_DIMS, traditional=False, base=10000.0, scale=1.0, offset=0)

        out_main = mx.fast.scaled_dot_product_attention(q,      k,      v, scale=self.scale)
        out_diff = mx.fast.scaled_dot_product_attention(q_diff, k_diff, v, scale=self.scale)
        out = out_main - out_diff

        out = out.transpose(0, 2, 1, 3).reshape(B, T, DIM)
        return self.to_out(out)


class GLU_FF(nn.Module):
    """SwiGLU feedforward, inner=2304. SiLU only (no sinusoidal in SAME-S)."""
    def __init__(self):
        super().__init__()
        self.glu_proj = nn.Linear(DIM, FF_INNER * 2, bias=True)
        self.proj_out = nn.Linear(FF_INNER, DIM, bias=True)

    def __call__(self, x):
        x = self.glu_proj(x)
        value, gate = mx.split(x, 2, axis=-1)
        return self.proj_out(value * nn.silu(gate))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = DyT(DIM)
        self.attn = DifferentialAttention()
        self.ff_norm = DyT(DIM)
        self.ff = GLU_FF()

    def __call__(self, x):
        x = x + self.attn(self.pre_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class SAMESDecoder(nn.Module):
    """SAME-S decoder.

    Input:  [B, 256, T_lat]    channels-first latents
    Output: [B, 512, T_lat*16] channels-first audio patches
    """

    def __init__(self):
        super().__init__()
        self.running_std = mx.array([1.0])
        self.project_in = nn.Linear(LATENT_DIM, DIM, bias=True)
        self.new_tokens = mx.zeros((1, 1, DIM))
        self.blocks = [TransformerBlock() for _ in range(NUM_BLOCKS)]
        # WNConv1d → MLX nn.Conv1d (weight_norm pre-fused at load time)
        # MLX Conv1d expects [out, k, in] weight layout
        self.mapping = nn.Conv1d(DIM, OUT_CHANNELS, kernel_size=3, padding=1, bias=True)

    def __call__(self, latents):
        B, _, T_lat = latents.shape

        # Bottleneck softnorm decode (scalar)
        x = latents * self.running_std

        # [B, 256, T_lat] → [B, T_lat, 256] → [B, T_lat, 768]
        x = self.project_in(x.transpose(0, 2, 1))

        # Single new_token broadcast to 16 positions per latent slot
        # x: [B, T_lat, DIM] → [B, T_lat, 1, DIM]
        # nt: [1, 1, DIM] → [1, 1, 1, DIM] → broadcast to [B, T_lat, 16, DIM]
        x_e = x[:, :, None, :]
        nt = mx.broadcast_to(self.new_tokens[None], (B, T_lat, SIN_PER_POS, DIM))
        x = mx.concatenate([x_e, nt], axis=2)              # [B, T_lat, 17, DIM]
        x = x.reshape(B, T_lat * SUB_CHUNK_SIZE, DIM)

        # First half (blocks 0..2): chunks of 34
        internal_T = T_lat * SUB_CHUNK_SIZE
        nc1 = internal_T // EFFECTIVE_CHUNK
        x = x.reshape(B * nc1, EFFECTIVE_CHUNK, DIM)
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        x = self.blocks[2](x)
        x = x.reshape(B, internal_T, DIM)

        # Shift by 17 on both ends → second half (blocks 3..5)
        left  = x[:, :SHIFT, :]
        right = x[:, -SHIFT:, :]
        x = mx.concatenate([left, x, right], axis=1)
        nc2 = (internal_T + EFFECTIVE_CHUNK) // EFFECTIVE_CHUNK
        x = x.reshape(B * nc2, EFFECTIVE_CHUNK, DIM)
        x = self.blocks[3](x)
        x = self.blocks[4](x)
        x = self.blocks[5](x)
        x = x.reshape(B, internal_T + EFFECTIVE_CHUNK, DIM)
        x = x[:, SHIFT:-SHIFT, :]

        # Drop the latent slot, keep 16 new-token positions per latent
        x = x.reshape(B * T_lat, SUB_CHUNK_SIZE, DIM)
        x = x[:, 1:, :]
        x = x.reshape(B, T_lat * SIN_PER_POS, DIM)

        # MLX Conv1d expects [B, T, C] input. Our x is [B, T_lat*16, 768]. Apply conv.
        out = self.mapping(x)                                # [B, T_lat*16, 512]
        return out.transpose(0, 2, 1)                        # → [B, 512, T_lat*16]


# ---- Weight loading ----

def load_model(weights_path=None, dtype=mx.float16, compile_=False):
    """Load SAMESDecoder with pretrained weights from .npz.

    The .npz was produced by scripts/export_same_s_weights.py. Keys use a
    flat naming consistent with both PyTorch and MLX modules.

    Conv1d weight in MLX is [out, k, in] but the .npz has PyTorch layout
    [out, in, k]. We permute on load.
    """
    if weights_path is None:
        for p in _WEIGHT_SEARCH:
            if p.exists():
                weights_path = str(p)
                break
        else:
            raise FileNotFoundError(f"No weights found; searched {_WEIGHT_SEARCH}")

    model = SAMESDecoder()
    raw = dict(mx.load(weights_path))

    # Permute conv mapping weight: PyTorch [out, in, k] → MLX [out, k, in]
    if "mapping.weight" in raw:
        w = raw["mapping.weight"]
        if w.ndim == 3 and w.shape[1] == DIM and w.shape[2] == 3:
            # Already PyTorch layout [512, 768, 3] — permute to MLX [512, 3, 768]
            raw["mapping.weight"] = w.transpose(0, 2, 1)

    weights = [(k, v.astype(dtype)) for k, v in raw.items()]
    model.load_weights(weights, strict=False)
    mx.eval(model.parameters())

    if compile_:
        model = mx.compile(model)
    return model


def decode_chunked(model, latents, chunk_size: int, overlap: int):
    """Uniform-kernel chunked decode for SAME-S. Returns [B, 512, T_lat*16].

    Every model call sees a `kernel_size = chunk_size + 2*overlap` window of
    REAL latents — never zero-padded. Three segments:
      1. First decode: model(latents[0:kernel]); the leading (chunk+overlap)
         latents of this output have enough right-context to be valid (no left
         context needed at position 0 — matches un-chunked behavior at the
         sequence start).
      2. Interior loop: stride by chunk_size with bilateral overlap context,
         keep the middle chunk_size*16 patches per call.
      3. Last decode: model(latents[T-kernel:T]); the trailing remaining
         latents of this output are valid (no right-context needed at T-1).

    No zero-padding anywhere — the model was trained on real audio latents.
    Falls back to un-chunked decode if T ≤ kernel_size.

    SAME-S internal constraint: input length × 17 must align to 34, i.e. the
    latent window length must be even. With uniform kernels all calls have
    length `kernel_size`, so we just require `chunk_size + 2*overlap` even.
    """
    B, C, T = latents.shape
    kernel = chunk_size + 2 * overlap
    assert kernel % 2 == 0, f"SAME-S needs even kernel (chunk+2*overlap); got {kernel}"
    if T <= kernel:
        # Fall back to un-chunked — but un-chunked SAME-S requires even T
        # (internal T*17 must align to 34). For odd T smaller than the kernel,
        # the caller must use a smaller (chunk, overlap) such that kernel < T.
        if T % 2 != 0:
            raise ValueError(
                f"SAME-S un-chunked requires even T (= {T}); "
                f"and T ≤ kernel ({kernel}) prevents the chunked path. "
                f"Use a smaller chunk/overlap so kernel < T (e.g. chunk=2 ovl=2 → kernel=6)."
            )
        return model(latents)

    pieces = []
    # 1) First decode covers output positions [0, chunk+overlap)
    first_out = model(latents[..., 0:kernel])
    valid_first = chunk_size + overlap
    pieces.append(first_out[..., : valid_first * 16])
    i = valid_first

    # 2) Interior: stride by chunk_size; each call covers chunk_size output positions
    while i + chunk_size + overlap <= T:
        out = model(latents[..., i - overlap : i + chunk_size + overlap])
        pieces.append(out[..., overlap * 16 : (overlap + chunk_size) * 16])
        i += chunk_size

    # 3) Last decode covers the remaining (T - i) output positions
    remaining = T - i
    if remaining > 0:
        last_out = model(latents[..., T - kernel : T])
        pieces.append(last_out[..., -(remaining * 16) :])
    return mx.concatenate(pieces, axis=-1)


if __name__ == "__main__":
    import numpy as np
    repo = Path(__file__).resolve().parents[2]
    m = load_model(dtype=mx.float32, compile_=False)
    # Smoke
    latents = mx.random.normal((1, 256, 32)) * 0.5
    out = m(latents)
    mx.eval(out)
    print(f"out: {out.shape} mean={float(out.mean()):.4f} std={float(out.std()):.4f}")
