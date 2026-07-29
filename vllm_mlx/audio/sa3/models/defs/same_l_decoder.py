"""SAME-L Decoder — MLX implementation.

The audio decoder from sa3-medium. Architecturally similar to the original
TAAEv2 decoder (12 blocks, dim=1536, SWA, sin gate on blocks 5..11) but with
the SAME-style single learnable new_tokens (broadcast) and a 1×1 conv mapping
instead of a Linear.

Input:  [B, 256, T_lat]
Output: [B, 512, T_lat*16]

426M params, 1.7 GB FP32 / 852 MB FP16.
"""

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


# Constants (SAME-L from sa3-medium)
LATENT_DIM       = 256
DIM              = 1536
NUM_HEADS        = 24
HEAD_DIM         = 64
ROPE_DIMS        = 32
NUM_BLOCKS       = 12
FF_INNER         = 4608           # ff_mult=3
SIN_START_BLOCK  = 5              # blocks 5..11 use sin(πx) gate (sinusoidal_blocks=8)
OUT_CHANNELS     = 512
STRIDE           = 16
SUB_CHUNK_SIZE   = STRIDE + 1     # 17 (1 latent + 16 new-token positions)
BLOCK_SIZE       = SUB_CHUNK_SIZE # alias for SWA convention
SIN_PER_POS      = SUB_CHUNK_SIZE - 1  # 16

_WEIGHT_SEARCH = [
    Path(__file__).parent.parent / "mlx" / "same_l_decoder_f32.npz",
]


# ---- Static SWA mask (17×51 band) ----
def _make_swa_mask():
    """Mask for grouped SWA. Query group of 17 attends to 3 consecutive KV groups (51)."""
    q = mx.arange(BLOCK_SIZE)[:, None]
    kv = mx.arange(3 * BLOCK_SIZE)[None, :]
    valid = (kv >= q) & (kv <= q + 2 * BLOCK_SIZE)
    return mx.where(valid, mx.array(0.0, dtype=mx.float32),
                    mx.array(-1e9, dtype=mx.float32))


class DyT(nn.Module):
    """gamma * tanh(alpha * x) + beta. alpha scalar, gamma/beta dim-vector."""
    def __init__(self, dim):
        super().__init__()
        self.alpha = mx.array([1.0])
        self.gamma = mx.ones((dim,))
        self.beta = mx.zeros((dim,))

    def __call__(self, x):
        return self.gamma * mx.tanh(self.alpha * x) + self.beta


class DifferentialSWA(nn.Module):
    """Differential attention with sliding-window mask.

    Self-attn: to_qkv chunk(5) → q, k, v, q_diff, k_diff (differential).
    SWA: queries grouped into blocks of 17, each block attends to 3 consecutive
    KV groups (51 KV tokens via padding+strided view).
    """
    def __init__(self):
        super().__init__()
        self.scale = HEAD_DIM ** -0.5
        self.to_qkv = nn.Linear(DIM, 5 * DIM, bias=False)
        self.to_out = nn.Linear(DIM, DIM, bias=False)
        self.q_norm = DyT(HEAD_DIM)
        self.k_norm = DyT(HEAD_DIM)

    def __call__(self, x, mask=None, full_attention=False):
        B, T, _ = x.shape
        H, D = NUM_HEADS, HEAD_DIM

        qkv = self.to_qkv(x)
        q1, k1, v, q2, k2 = mx.split(qkv, 5, axis=-1)

        def to_heads(t):
            return t.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        q1 = to_heads(q1); k1 = to_heads(k1); v = to_heads(v)
        q2 = to_heads(q2); k2 = to_heads(k2)

        q1 = self.q_norm(q1); k1 = self.k_norm(k1)
        q2 = self.q_norm(q2); k2 = self.k_norm(k2)

        q1 = mx.fast.rope(q1, ROPE_DIMS, traditional=False, base=10000.0, scale=1.0, offset=0)
        k1 = mx.fast.rope(k1, ROPE_DIMS, traditional=False, base=10000.0, scale=1.0, offset=0)
        q2 = mx.fast.rope(q2, ROPE_DIMS, traditional=False, base=10000.0, scale=1.0, offset=0)
        k2 = mx.fast.rope(k2, ROPE_DIMS, traditional=False, base=10000.0, scale=1.0, offset=0)

        if full_attention or T <= BLOCK_SIZE:
            out = self._diff_sdpa(q1, k1, v, q2, k2)
        else:
            out = self._swa(q1, k1, v, q2, k2, mask)

        return self.to_out(out.transpose(0, 2, 1, 3).reshape(B, T, DIM))

    def _diff_sdpa(self, q1, k1, v, q2, k2, mask=None):
        """Batched differential SDPA — one kernel call."""
        Q = mx.concatenate([q1, q2], axis=1)
        K = mx.concatenate([k1, k2], axis=1)
        V = mx.concatenate([v, v], axis=1)
        out = mx.fast.scaled_dot_product_attention(Q, K, V, scale=self.scale, mask=mask)
        out1, out2 = mx.split(out, 2, axis=1)
        return out1 - out2

    def _swa(self, q1, k1, v, q2, k2, mask):
        """Grouped SWA via strided KV view + fused 4D SDPA."""
        B, H, T, D = q1.shape
        G = T // BLOCK_SIZE
        W = 3 * BLOCK_SIZE  # 51

        # Pad K/V by one BLOCK on each side
        pad = [(0, 0), (0, 0), (BLOCK_SIZE, BLOCK_SIZE), (0, 0)]
        k1p = mx.pad(k1, pad)
        k2p = mx.pad(k2, pad)
        vp = mx.pad(v, pad)

        Tp = T + 2 * BLOCK_SIZE
        win_strides = [H * Tp * D, Tp * D, BLOCK_SIZE * D, D, 1]
        win_shape = [B, H, G, W, D]
        k1w = mx.as_strided(k1p, shape=win_shape, strides=win_strides)
        k2w = mx.as_strided(k2p, shape=win_shape, strides=win_strides)
        vw = mx.as_strided(vp, shape=win_shape, strides=win_strides)

        q1g = q1.reshape(B, H, G, BLOCK_SIZE, D)
        q2g = q2.reshape(B, H, G, BLOCK_SIZE, D)

        # Boundary mask: suppress zero-padded KV at sequence edges
        g_idx = mx.arange(G)[:, None]
        w_idx = mx.arange(W)[None, :]
        padded_pos = g_idx * BLOCK_SIZE + w_idx
        boundary = mx.where(
            (padded_pos >= BLOCK_SIZE) & (padded_pos < T + BLOCK_SIZE),
            mx.array(0.0, dtype=q1.dtype),
            mx.array(-1e9, dtype=q1.dtype),
        )[:, None, :]  # [G, 1, W]

        if mask is not None:
            combined = mask + boundary  # broadcast: [17, 51] + [G, 1, 51]
        else:
            combined = boundary
        combined = mx.broadcast_to(
            combined[None], (B, G, BLOCK_SIZE, W)
        ).reshape(B * G, 1, BLOCK_SIZE, W)

        q1g = q1g.transpose(0, 2, 1, 3, 4).reshape(B * G, H, BLOCK_SIZE, D)
        q2g = q2g.transpose(0, 2, 1, 3, 4).reshape(B * G, H, BLOCK_SIZE, D)
        k1w = k1w.transpose(0, 2, 1, 3, 4).reshape(B * G, H, W, D)
        k2w = k2w.transpose(0, 2, 1, 3, 4).reshape(B * G, H, W, D)
        vw = vw.transpose(0, 2, 1, 3, 4).reshape(B * G, H, W, D)

        Q = mx.concatenate([q1g, q2g], axis=1)
        K = mx.concatenate([k1w, k2w], axis=1)
        V = mx.concatenate([vw, vw], axis=1)

        out = mx.fast.scaled_dot_product_attention(
            Q, K, V, scale=self.scale, mask=combined)
        out1, out2 = mx.split(out, 2, axis=1)
        diff = out1 - out2
        return diff.reshape(B, G, H, BLOCK_SIZE, D).transpose(0, 2, 1, 3, 4).reshape(B, H, T, D)


class FeedForward(nn.Module):
    """GLU FF, optionally with sin(πx) gate (blocks 5..11)."""
    def __init__(self, use_sin=False):
        super().__init__()
        self.use_sin = use_sin
        self.glu_proj = nn.Linear(DIM, FF_INNER * 2, bias=True)
        self.proj_out = nn.Linear(FF_INNER, DIM, bias=True)

    def __call__(self, x):
        x = self.glu_proj(x)
        value, gate = mx.split(x, 2, axis=-1)
        if self.use_sin:
            activated = value * mx.sin(gate * math.pi)
        else:
            activated = value * nn.silu(gate)
        return self.proj_out(activated)


class TransformerBlock(nn.Module):
    def __init__(self, block_idx):
        super().__init__()
        self.pre_norm = DyT(DIM)
        self.attn = DifferentialSWA()
        self.ff_norm = DyT(DIM)
        self.ff = FeedForward(use_sin=(block_idx >= SIN_START_BLOCK))

    def __call__(self, x, mask=None, full_attention=False):
        x = x + self.attn(self.pre_norm(x), mask=mask, full_attention=full_attention)
        x = x + self.ff(self.ff_norm(x))
        return x


class SAMELDecoder(nn.Module):
    """SAME-L decoder: latent tokens → audio patches.

    Input:  [B, 256, T_lat]
    Output: [B, 512, T_lat*16]
    """

    def __init__(self):
        super().__init__()
        self.running_std = mx.array([1.0])
        self.project_in = nn.Linear(LATENT_DIM, DIM, bias=True)
        self.new_tokens = mx.zeros((1, 1, DIM))           # single learnable, broadcast 16x
        self.blocks = [TransformerBlock(i) for i in range(NUM_BLOCKS)]
        # Conv1d kernel=1 ≈ Linear; we store as Linear and reshape weight on load
        self.mapping = nn.Linear(DIM, OUT_CHANNELS, bias=True)

    def __call__(self, latents, full_attention=False):
        B, _, T_lat = latents.shape

        # Softnorm bottleneck decode (scalar)
        x = latents * self.running_std

        # Project: [B, 256, T_lat] → [B, T_lat, 256] → [B, T_lat, DIM]
        x = self.project_in(x.transpose(0, 2, 1))

        # Single new_token broadcast to 16 positions per latent slot
        x_e = x[:, :, None, :]
        nt = mx.broadcast_to(self.new_tokens[None], (B, T_lat, SIN_PER_POS, DIM))
        x = mx.concatenate([x_e, nt], axis=2)              # [B, T_lat, 17, DIM]
        x = x.reshape(B, T_lat * SUB_CHUNK_SIZE, DIM)

        mask = None if full_attention else _SWA_MASK.astype(x.dtype)
        for blk in self.blocks:
            x = blk(x, mask=mask, full_attention=full_attention)

        # Drop original latent slot at index 0 of each 17-group, keep 16
        x = x.reshape(B, T_lat, SUB_CHUNK_SIZE, DIM)[:, :, 1:]
        x = x.reshape(B, T_lat * SIN_PER_POS, DIM)

        # Apply mapping (Linear) — [B, T_lat*16, DIM] → [B, T_lat*16, 512]
        # then transpose to channels-first
        return self.mapping(x).transpose(0, 2, 1)


_SWA_MASK = _make_swa_mask()


def load_model(weights_path=None, dtype=mx.float16, compile_=False):
    if weights_path is None:
        for p in _WEIGHT_SEARCH:
            if p.exists():
                weights_path = str(p)
                break
        else:
            raise FileNotFoundError(f"No weights found; searched {_WEIGHT_SEARCH}")

    model = SAMELDecoder()
    raw = dict(mx.load(weights_path))

    # mapping was extracted as PyTorch Conv1d [out, in, k=1].
    # Reshape to nn.Linear weight [out, in].
    if "mapping.weight" in raw:
        w = raw["mapping.weight"]
        if w.ndim == 3 and w.shape[-1] == 1:
            raw["mapping.weight"] = w.reshape(w.shape[0], w.shape[1])

    weights = [(k, v.astype(dtype)) for k, v in raw.items()]
    model.load_weights(weights, strict=False)
    mx.eval(model.parameters())

    if compile_:
        model = mx.compile(model)
    return model


def decode_chunked(model, latents, chunk_size: int, overlap: int):
    """Uniform-kernel chunked decode for SAME-L. Returns [B, 512, T_lat*16].

    Every model call sees a `kernel_size = chunk_size + 2*overlap` window of
    REAL latents — never zero-padded. Three segments:
      1. First decode: model(latents[0:kernel]); the leading (chunk+overlap)
         latents of this output have enough right-context to be valid (position
         0 has no left-context in the un-chunked decode either).
      2. Interior loop: stride by chunk_size with bilateral overlap context,
         keep the middle chunk_size*16 patches per call.
      3. Last decode: model(latents[T-kernel:T]); the trailing remaining
         latents of this output are valid.

    No zero-padding anywhere — the model was trained on real audio latents.
    Falls back to un-chunked decode if T ≤ kernel_size.

    SAME-L has no even-length constraint. Empirical receptive field is ~8 latents
    per side; overlap=8 → ≥80 dB (with old edge handling); overlap=12 → ≈ bit-exact.
    The new edge handling should push overlap=8 closer to bit-exact too.
    """
    B, C, T = latents.shape
    kernel = chunk_size + 2 * overlap
    if T <= kernel:
        return model(latents)

    pieces = []
    # 1) First decode covers output positions [0, chunk+overlap)
    first_out = model(latents[..., 0:kernel])
    valid_first = chunk_size + overlap
    pieces.append(first_out[..., : valid_first * 16])
    i = valid_first

    # 2) Interior: stride by chunk_size
    while i + chunk_size + overlap <= T:
        out = model(latents[..., i - overlap : i + chunk_size + overlap])
        pieces.append(out[..., overlap * 16 : (overlap + chunk_size) * 16])
        i += chunk_size

    # 3) Last decode covers remaining (T - i) output positions
    remaining = T - i
    if remaining > 0:
        last_out = model(latents[..., T - kernel : T])
        pieces.append(last_out[..., -(remaining * 16) :])
    return mx.concatenate(pieces, axis=-1)


if __name__ == "__main__":
    import time
    m = load_model(dtype=mx.float32, compile_=False)
    nparams = 0
    def _count(d):
        n = 0
        if isinstance(d, dict):
            for v in d.values(): n += _count(v)
        elif isinstance(d, list):
            for v in d: n += _count(v)
        elif isinstance(d, mx.array):
            n += d.size
        return n
    nparams = _count(m.parameters())
    print(f"Params: ~{nparams:,}")

    latents = mx.random.normal((1, 256, 32)) * 0.5
    t0 = time.perf_counter()
    out = m(latents)
    mx.eval(out)
    print(f"out: {out.shape} mean={float(out.mean()):.4f} std={float(out.std()):.4f} "
          f"({(time.perf_counter()-t0)*1000:.0f}ms)")
