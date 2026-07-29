"""SAME-S Encoder — MLX implementation.

Inverse of `same_s_decoder.py`. Same transformer body (6 blocks, dim=768,
differential attention, DyT norm, chunk_midpoint_shift) — just runs in the
encode direction:

    Input:  [B, 512, T_audio_patches]   (= [B, 2*patch_size, T_lat*16] post-patched-pretransform)
    Output: [B, 256, T_lat]              (post-softnorm-bottleneck, ready for DiT)

Constants come from sa3-sm-music encoder config:
  channels=128, c_mults=[6] → dim=768
  stride=16, chunk_size=32, chunk_midpoint_shift=True
  transformer_depths=[6], differential=True, dim_heads=64, dyt=True
  mapping=WNConv1d(in_channels=512 → dim=768, kernel=1) [conv_mapping=False]

Shared building blocks are imported from same_s_decoder.py to avoid duplicating
the transformer machinery.
"""

from __future__ import annotations
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Re-use building blocks from the decoder — identical math.
from .same_s_decoder import (
    DIM, NUM_HEADS, HEAD_DIM, ROPE_DIMS,
    NUM_BLOCKS, FF_INNER, OUT_CHANNELS, STRIDE,
    SUB_CHUNK_SIZE, CHUNK_SIZE_LAT, EFFECTIVE_CHUNK, SHIFT,
    SIN_PER_POS, QK_NORM_EPS,
    DyT, DifferentialAttention, GLU_FF, TransformerBlock,
)

# Encoder-specific
LATENT_DIM     = 256
IN_CHANNELS    = 512                          # PatchedPretransform output (channels=2, patch_size=256)
PAD_MODULO_AUD = CHUNK_SIZE_LAT               # 32 — encoder zero-pads audio side to this modulo

_WEIGHT_SEARCH = [
    Path(__file__).parent.parent / "mlx" / "same_s_encoder_f32.npz",
]


class SAMESEncoder(nn.Module):
    """SAME-S encoder.

    Input:  [B, 512, T_audio_patches]   channels-first audio patches
    Output: [B, 256, T_lat]              channels-first latents in softnorm space
    """

    def __init__(self):
        super().__init__()
        # Mapping: 512 → 768 via Conv1d(kernel=1) (no conv_mapping, so kernel=1)
        # Stored as Linear since k=1 ≡ Linear after transpose.
        self.mapping = nn.Linear(IN_CHANNELS, DIM, bias=True)
        # new_tokens at the END of each 17-sub-chunk; with variable_stride=True it's (1, 1, DIM).
        self.new_tokens = mx.zeros((1, 1, DIM))
        self.blocks = [TransformerBlock() for _ in range(NUM_BLOCKS)]
        # Final projection: 768 → 256 (Transpose+Linear+Transpose in upstream; just Linear in token-major space)
        self.project_out = nn.Linear(DIM, LATENT_DIM, bias=True)
        # Softnorm bottleneck (auto_scale=True so running_std exists)
        self.scaling_factor = mx.ones((1, LATENT_DIM, 1))
        self.bias = mx.zeros((1, LATENT_DIM, 1))
        self.running_std = mx.array([1.0])

    def __call__(self, audio_patches: mx.array) -> mx.array:
        B, C, T_aud = audio_patches.shape
        assert C == IN_CHANNELS, f"expected {IN_CHANNELS} channels, got {C}"
        # Encoder requires the audio length to be a multiple of (chunk_size=32) so the
        # internal layout (chunks of 34 after expand) divides cleanly. T_aud should
        # already be a multiple of 32 in normal use — assert rather than silently pad.
        assert T_aud % PAD_MODULO_AUD == 0, (
            f"audio_patch length {T_aud} must be a multiple of {PAD_MODULO_AUD}; pad with zeros first"
        )

        # 1. Mapping 512 → 768 (kernel=1, so equivalent to per-position Linear)
        x = self.mapping(audio_patches.transpose(0, 2, 1))     # (B, T_aud, DIM)

        # 2. Group every STRIDE=16 audio positions into a "sub-chunk" of size SUB_CHUNK_SIZE=17:
        #    take 16 real positions, append 1 new_token at the END.
        T_lat = T_aud // STRIDE
        x = x.reshape(B * T_lat, STRIDE, DIM)                  # (B*T_lat, 16, DIM)
        nt = mx.broadcast_to(self.new_tokens, (B * T_lat, 1, DIM))
        x = mx.concatenate([x, nt], axis=1)                    # (B*T_lat, 17, DIM)
        x = x.reshape(B, T_lat * SUB_CHUNK_SIZE, DIM)          # (B, T_lat*17, DIM)

        internal_T = T_lat * SUB_CHUNK_SIZE                    # = T_lat * 17

        # 3. First half (blocks 0..2): chunks of EFFECTIVE_CHUNK=34, no shift
        nc1 = internal_T // EFFECTIVE_CHUNK
        x = x.reshape(B * nc1, EFFECTIVE_CHUNK, DIM)
        for blk in self.blocks[: NUM_BLOCKS // 2]:
            x = blk(x)
        x = x.reshape(B, internal_T, DIM)

        # 4. Second half (blocks 3..5): pad with first/last SHIFT internal tokens, re-chunk, shift back.
        left  = x[:, :SHIFT, :]
        right = x[:, -SHIFT:, :]
        x = mx.concatenate([left, x, right], axis=1)           # (B, internal_T + 2*SHIFT, DIM)
        nc2 = (internal_T + EFFECTIVE_CHUNK) // EFFECTIVE_CHUNK
        x = x.reshape(B * nc2, EFFECTIVE_CHUNK, DIM)
        for blk in self.blocks[NUM_BLOCKS // 2 :]:
            x = blk(x)
        x = x.reshape(B, internal_T + EFFECTIVE_CHUNK, DIM)
        x = x[:, SHIFT:-SHIFT, :]                              # back to (B, internal_T, DIM)

        # 5. Take the LAST position of every 17-sub-chunk (the new_token's output = the latent).
        x = x.reshape(B * T_lat, SUB_CHUNK_SIZE, DIM)
        x = x[:, -1:, :]                                       # (B*T_lat, 1, DIM)
        x = x.reshape(B, T_lat, DIM)

        # 6. Final projection 768 → 256
        x = self.project_out(x)                                # (B, T_lat, 256)
        x = x.transpose(0, 2, 1)                               # (B, 256, T_lat)

        # 7. Softnorm bottleneck (encoder side):
        #    x = x * scaling_factor + bias  →  x / running_std
        x = x * self.scaling_factor + self.bias
        x = x / self.running_std
        return x


def load_model(weights_path=None, dtype=mx.float16, compile_=False):
    """Load SAME-S encoder weights from .npz."""
    if weights_path is None:
        for p in _WEIGHT_SEARCH:
            if p.exists():
                weights_path = str(p)
                break
        else:
            raise FileNotFoundError(f"No weights found; searched {_WEIGHT_SEARCH}")

    model = SAMESEncoder()
    raw = dict(mx.load(weights_path))

    # mapping is Linear here (we treat the 1×1 conv as a per-position Linear) —
    # if exported as a Conv1d weight [out, k=1, in], reshape to Linear [out, in].
    if "mapping.weight" in raw:
        w = raw["mapping.weight"]
        if w.ndim == 3:
            # Either (out, k, in) [MLX layout] or (out, in, k) [PyTorch layout]; both reduce to (out, in) at k=1.
            raw["mapping.weight"] = w.reshape(w.shape[0], -1)

    weights = [(k, v.astype(dtype)) for k, v in raw.items()]
    model.load_weights(weights, strict=False)
    mx.eval(model.parameters())
    if compile_:
        model = mx.compile(model)
    return model


if __name__ == "__main__":
    import time
    m = load_model(dtype=mx.float32, compile_=False)
    # Smoke test: 30s clip → T_lat=320 → T_aud=320*16=5120
    x = mx.random.normal((1, 512, 5120)) * 0.1
    t0 = time.time(); y = m(x); mx.eval(y); dt = (time.time()-t0)*1000
    print(f"SAME-S encoder: in {x.shape} → out {y.shape}  ({dt:.0f} ms)")
    print(f"  out mean={float(y.mean()):+.4f}  std={float(y.std()):.4f}  abs.max={float(mx.abs(y).max()):.4f}")
