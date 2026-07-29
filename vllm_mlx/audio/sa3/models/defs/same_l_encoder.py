"""SAME-L Encoder — MLX implementation.

Inverse of `same_l_decoder.py`. 12 blocks · dim=1536 · differential SWA
(sliding window of 1 group each side over the 17-token groups, no chunk-shift,
no sinusoidal gates).

    Input:  [B, 512, T_audio_patches]  (post patched-pretransform, T_aud = T_lat*16)
    Output: [B, 256, T_lat]             (post softnorm bottleneck)

Constants come from sa3-medium encoder config:
  channels=256, c_mults=[6] → DIM=1536, 12 transformer blocks
  stride=16, sliding_window=[1,1] (full attention over the local 51-token window)
  differential=True, dim_heads=64, dyt=True
  mapping=WNConv1d(512 → 1536, k=1) [conv_mapping=False]
"""

from __future__ import annotations
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Re-use SAME-L blocks from the decoder — identical math
from .same_l_decoder import (
    DIM, NUM_HEADS, HEAD_DIM, ROPE_DIMS,
    NUM_BLOCKS, FF_INNER, OUT_CHANNELS, STRIDE,
    SUB_CHUNK_SIZE, BLOCK_SIZE, SIN_PER_POS,
    DyT, DifferentialSWA, FeedForward,
    _make_swa_mask, _SWA_MASK,
)

# Encoder-specific
LATENT_DIM   = 256
IN_CHANNELS  = 512

_WEIGHT_SEARCH = [
    Path(__file__).parent.parent / "mlx" / "same_l_encoder_f32.npz",
]


class _EncoderBlock(nn.Module):
    """Same as SAME-L decoder TransformerBlock but with use_sin=False everywhere."""
    def __init__(self):
        super().__init__()
        self.pre_norm = DyT(DIM)
        self.attn = DifferentialSWA()
        self.ff_norm = DyT(DIM)
        self.ff = FeedForward(use_sin=False)

    def __call__(self, x, mask=None, full_attention=False):
        x = x + self.attn(self.pre_norm(x), mask=mask, full_attention=full_attention)
        x = x + self.ff(self.ff_norm(x))
        return x


class SAMELEncoder(nn.Module):
    """SAME-L encoder.

    Input:  [B, 512, T_audio_patches]   channels-first audio patches
    Output: [B, 256, T_lat]              latents in softnorm space
    """

    def __init__(self):
        super().__init__()
        # mapping: 512 → 1536 (Conv1d k=1 = Linear after transpose)
        self.mapping = nn.Linear(IN_CHANNELS, DIM, bias=True)
        # new_token at the END of each 17-sub-chunk (variable_stride=True → single token, broadcast)
        self.new_tokens = mx.zeros((1, 1, DIM))
        self.blocks = [_EncoderBlock() for _ in range(NUM_BLOCKS)]
        # final projection 1536 → 256
        self.project_out = nn.Linear(DIM, LATENT_DIM, bias=True)
        # softnorm bottleneck params
        self.scaling_factor = mx.ones((1, LATENT_DIM, 1))
        self.bias = mx.zeros((1, LATENT_DIM, 1))
        self.running_std = mx.array([1.0])

    def __call__(self, audio_patches: mx.array, full_attention: bool = False) -> mx.array:
        B, C, T_aud = audio_patches.shape
        assert C == IN_CHANNELS, f"expected {IN_CHANNELS} channels, got {C}"
        # SAME-L has sliding_window set, so the encoder only requires T_aud % STRIDE == 0
        # (much less restrictive than SAME-S's modulo-32 requirement).
        assert T_aud % STRIDE == 0, (
            f"audio_patch length {T_aud} must be a multiple of {STRIDE}"
        )

        # 1. Mapping 512 → 1536
        x = self.mapping(audio_patches.transpose(0, 2, 1))           # (B, T_aud, DIM)

        # 2. Group every STRIDE=16 audio positions, append 1 new_token at the END
        T_lat = T_aud // STRIDE
        x = x.reshape(B * T_lat, STRIDE, DIM)                        # (B*T_lat, 16, DIM)
        nt = mx.broadcast_to(self.new_tokens, (B * T_lat, 1, DIM))
        x = mx.concatenate([x, nt], axis=1)                          # (B*T_lat, 17, DIM)
        x = x.reshape(B, T_lat * SUB_CHUNK_SIZE, DIM)                # (B, T_lat*17, DIM)

        # 3. Run all 12 transformer blocks with SWA mask
        mask = None if full_attention else _SWA_MASK.astype(x.dtype)
        for blk in self.blocks:
            x = blk(x, mask=mask, full_attention=full_attention)

        # 4. Take the LAST position of every 17-sub-chunk (the new_token's output)
        x = x.reshape(B, T_lat, SUB_CHUNK_SIZE, DIM)[:, :, -1, :]    # (B, T_lat, DIM)

        # 5. Final projection 1536 → 256
        x = self.project_out(x)                                       # (B, T_lat, LATENT_DIM)
        x = x.transpose(0, 2, 1)                                      # (B, LATENT_DIM, T_lat)

        # 6. Softnorm bottleneck (encoder direction)
        x = x * self.scaling_factor + self.bias
        x = x / self.running_std
        return x


def load_model(weights_path=None, dtype=mx.float16, compile_=False):
    if weights_path is None:
        for p in _WEIGHT_SEARCH:
            if p.exists():
                weights_path = str(p)
                break
        else:
            raise FileNotFoundError(f"No weights found; searched {_WEIGHT_SEARCH}")

    model = SAMELEncoder()
    raw = dict(mx.load(weights_path))

    # mapping stored as Conv1d weight (out, k=1, in) or (out, in, k=1); reshape to Linear (out, in)
    if "mapping.weight" in raw:
        w = raw["mapping.weight"]
        if w.ndim == 3:
            raw["mapping.weight"] = w.reshape(w.shape[0], -1)

    weights = [(k, v.astype(dtype)) for k, v in raw.items()]
    model.load_weights(weights, strict=False)
    mx.eval(model.parameters())
    if compile_:
        model = mx.compile(model)
    return model


if __name__ == "__main__":
    import time
    m = load_model(dtype=mx.float32)
    x = mx.random.normal((1, 512, 5120)) * 0.1
    t0 = time.time(); y = m(x); mx.eval(y); dt = (time.time()-t0)*1000
    print(f"SAME-L encoder: in {x.shape} → out {y.shape}  ({dt:.0f} ms)")
    print(f"  out mean={float(y.mean()):+.4f}  std={float(y.std()):.4f}  abs.max={float(mx.abs(y).max()):.4f}")
