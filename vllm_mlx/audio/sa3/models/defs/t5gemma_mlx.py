"""T5Gemma (google/t5gemma-b-b-ul2) encoder in MLX.

Files:
    models/defs/t5gemma_mlx.py    this module (model + tokenizer wrapper + simple API)
    models/mlx/t5gemma_f16.npz    weights (FP16) + SentencePiece tokenizer bytes

Dependencies: mlx, numpy, sentencepiece.

Usage:
    from models.defs.t5gemma_mlx import T5Gemma
    enc = T5Gemma.from_npz("models/mlx/t5gemma_f16.npz")
    embeds, mask = enc.encode(["lofi house loop", "Amen break 174 BPM"])
    # embeds: (2, 256, 768) fp16; mask: (2, 256) int32  (1 = real token, 0 = pad)

Encoder details (Gemma2-style):
    12 layers · dim 768 · 12 heads · head_dim 64 · GeGLU(2048) · RMSNorm(1+w)
    RoPE (θ=10000, half-half layout) · attn logit softcap=50 · embed × √hidden_size
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class T5GemmaConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    intermediate_size: int = 2048
    vocab_size: int = 256000
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attn_logit_softcapping: float = 50.0
    query_pre_attn_scalar: int = 64
    pad_token_id: int = 0

    @classmethod
    def from_json_bytes(cls, blob: bytes) -> "T5GemmaConfig":
        d = json.loads(blob.decode("utf-8"))
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ────────────────────────────────────────────────────────────────────────────
# Model ops
# ────────────────────────────────────────────────────────────────────────────

def _rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Gemma-style RMSNorm: normalize in fp32, scale by (1 + weight), cast back."""
    dtype = x.dtype
    x32 = x.astype(mx.float32)
    var = (x32 * x32).mean(axis=-1, keepdims=True)
    n = x32 * mx.rsqrt(var + eps)
    return (n * (1.0 + weight.astype(mx.float32))).astype(dtype)


def _rope_cos_sin(seq_len: int, head_dim: int, theta: float,
                  inv_freq: mx.array | None = None) -> tuple[mx.array, mx.array]:
    if inv_freq is None:
        inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    pos = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(pos, inv_freq)                    # (S, head_dim/2)
    emb = mx.concatenate([freqs, freqs], axis=-1)      # (S, head_dim) half-half
    return mx.cos(emb), mx.sin(emb)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> tuple[mx.array, mx.array]:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = (q * cos.astype(q.dtype)) + (_rotate_half(q) * sin.astype(q.dtype))
    k = (k * cos.astype(k.dtype)) + (_rotate_half(k) * sin.astype(k.dtype))
    return q, k


class _SelfAttention(nn.Module):
    def __init__(self, cfg: T5GemmaConfig):
        super().__init__()
        H, D = cfg.num_attention_heads, cfg.head_dim
        self.num_heads = H
        self.head_dim = D
        self.scaling = cfg.query_pre_attn_scalar ** -0.5
        self.softcap = cfg.attn_logit_softcapping
        self.q_proj = nn.Linear(cfg.hidden_size, H * D, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * D, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * D, bias=False)
        self.o_proj = nn.Linear(H * D, cfg.hidden_size, bias=False)

    def __call__(self, x, cos, sin, add_mask):
        B, S, _ = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(x).reshape(B, S, H, D).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, H, D).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, H, D).transpose(0, 2, 1, 3)
        q, k = _apply_rope(q, k, cos, sin)
        qk = (q @ k.transpose(0, 1, 3, 2)) * self.scaling
        if self.softcap is not None:
            qk = mx.tanh(qk / self.softcap) * self.softcap
        if add_mask is not None:
            qk = qk + add_mask
        p = mx.softmax(qk.astype(mx.float32), axis=-1).astype(v.dtype)
        out = (p @ v).transpose(0, 2, 1, 3).reshape(B, S, H * D)
        return self.o_proj(out)


class _MLP(nn.Module):
    def __init__(self, cfg: T5GemmaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class _EncoderLayer(nn.Module):
    def __init__(self, cfg: T5GemmaConfig):
        super().__init__()
        self.cfg = cfg
        self.self_attn = _SelfAttention(cfg)
        self.mlp = _MLP(cfg)
        D = cfg.hidden_size
        # Stored as raw arrays (not nn.Linear) so we can assign post-init from .npz.
        self.pre_self_attn_layernorm = mx.zeros((D,))
        self.post_self_attn_layernorm = mx.zeros((D,))
        self.pre_feedforward_layernorm = mx.zeros((D,))
        self.post_feedforward_layernorm = mx.zeros((D,))

    def __call__(self, x, cos, sin, add_mask):
        eps = self.cfg.rms_norm_eps
        h = _rms_norm(x, self.pre_self_attn_layernorm, eps)
        h = self.self_attn(h, cos, sin, add_mask)
        h = _rms_norm(h, self.post_self_attn_layernorm, eps)
        x = x + h
        h = _rms_norm(x, self.pre_feedforward_layernorm, eps)
        h = self.mlp(h)
        h = _rms_norm(h, self.post_feedforward_layernorm, eps)
        return x + h


class _Encoder(nn.Module):
    def __init__(self, cfg: T5GemmaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [_EncoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = mx.zeros((cfg.hidden_size,))
        self._normalizer = math.sqrt(cfg.hidden_size)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        x = self.embed_tokens(input_ids).astype(mx.float16)
        x = x * mx.array(self._normalizer, dtype=x.dtype)

        cos, sin = _rope_cos_sin(x.shape[1], self.cfg.head_dim, self.cfg.rope_theta,
                                 inv_freq=getattr(self, "rope_inv_freq", None))

        add_mask = None
        if attention_mask is not None:
            keep = attention_mask.astype(mx.float32)
            add_mask = ((1.0 - keep) * -1e9)[:, None, None, :].astype(x.dtype)

        for layer in self.layers:
            x = layer(x, cos, sin, add_mask)
        return _rms_norm(x, self.norm, self.cfg.rms_norm_eps)


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

class T5Gemma:
    """High-level wrapper: tokenize raw strings → MLX embeddings."""

    def __init__(self, encoder: _Encoder, cfg: T5GemmaConfig, tokenizer):
        self.encoder = encoder
        self.cfg = cfg
        self.tokenizer = tokenizer

    @classmethod
    def from_npz(cls, path: str) -> "T5Gemma":
        import sentencepiece as spm  # local import — only needed by users who call from_npz

        arrs = np.load(path)
        if "META" not in arrs.files:
            raise ValueError(f"{path} is missing META blob; expected t5gemma_f16.npz format")
        cfg = T5GemmaConfig.from_json_bytes(arrs["META"].tobytes())
        if "TOKENIZER_MODEL" not in arrs.files:
            raise ValueError(f"{path} is missing TOKENIZER_MODEL bytes")
        tok = spm.SentencePieceProcessor()
        tok.LoadFromSerializedProto(arrs["TOKENIZER_MODEL"].tobytes())

        enc = _Encoder(cfg)
        nested: dict = {}
        for k in arrs.files:
            if k in ("META", "TOKENIZER_MODEL"):
                continue
            a = mx.array(arrs[k])
            if k == "rope_inv_freq":
                enc.rope_inv_freq = a.astype(mx.float32)
                continue
            parts = k.split(".")
            cur = nested
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = a

        NORM_FIELDS = (
            "pre_self_attn_layernorm", "post_self_attn_layernorm",
            "pre_feedforward_layernorm", "post_feedforward_layernorm",
        )
        for i, layer in enumerate(enc.layers):
            ldict = nested["layers"][str(i)]
            for f in NORM_FIELDS:
                setattr(layer, f, ldict.pop(f)["weight"])
            layer.update(ldict)
        enc.embed_tokens.weight = nested["embed_tokens"]["weight"]
        enc.norm = nested["norm"]["weight"] if isinstance(nested["norm"], dict) else nested["norm"]

        mx.eval(enc.parameters())
        return cls(enc, cfg, tok)

    def tokenize(self, prompts: list[str], max_len: int = 256) -> tuple[mx.array, mx.array]:
        """Tokenize a list of strings → (input_ids, attention_mask).

        Returns int32 (B, max_len) tensors. Padding token = cfg.pad_token_id.
        Strings longer than max_len are truncated.
        """
        B = len(prompts)
        pad = self.cfg.pad_token_id
        ids = np.full((B, max_len), pad, dtype=np.int32)
        mask = np.zeros((B, max_len), dtype=np.int32)
        for i, p in enumerate(prompts):
            toks = self.tokenizer.Encode(p)[:max_len]
            ids[i, :len(toks)] = toks
            mask[i, :len(toks)] = 1
        return mx.array(ids), mx.array(mask)

    def encode(self, prompts: list[str], max_len: int = 256) -> tuple[mx.array, mx.array]:
        """Tokenize + encode in one call → (last_hidden_state, attention_mask).

        Returns (B, max_len, 768) fp16 + (B, max_len) int32. Pad positions in the
        returned embedding still contain numbers — consumers should consult `mask`.
        """
        ids, mask = self.tokenize(prompts, max_len=max_len)
        # An all-zero mask row makes the attention softmax see all -inf and produces
        # NaN. Give every empty row one visible position for the forward pass, then
        # restore mask=0 so downstream padding logic still overwrites every position.
        mask_np = np.asarray(mask, dtype=np.int32)
        empty_rows = (mask_np.sum(axis=1) == 0)
        if empty_rows.any():
            fixed = mask_np.copy()
            fixed[empty_rows, 0] = 1
            out = self.encoder(ids, mx.array(fixed))
        else:
            out = self.encoder(ids, mask)
        return out, mask


# ────────────────────────────────────────────────────────────────────────────
# CLI demo
# ────────────────────────────────────────────────────────────────────────────

def _main():
    import argparse, os, sys, time
    ap = argparse.ArgumentParser(description="Encode prompts with T5Gemma (MLX FP16).")
    _DEFAULT_NPZ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlx", "t5gemma_f16.npz")
    ap.add_argument("--npz", default=_DEFAULT_NPZ)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("prompts", nargs="*",
                    default=["A beautiful piano arpeggio grows into a grand cinematic climax",
                             "lofi house loop", "Amen break 174 BPM"])
    args = ap.parse_args()

    print(f"loading {args.npz} …")
    t0 = time.time()
    enc = T5Gemma.from_npz(args.npz)
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"encoding {len(args.prompts)} prompt(s) …")
    # warmup — eval to force compile + run before timing
    w_embeds, w_mask = enc.encode(args.prompts, max_len=args.max_len)
    mx.eval(w_embeds, w_mask)
    t0 = time.time()
    embeds, mask = enc.encode(args.prompts, max_len=args.max_len)
    mx.eval(embeds, mask)
    dt = (time.time() - t0) * 1000
    print(f"  {dt:.1f} ms total ({dt/len(args.prompts):.1f} ms/prompt) — embeds {embeds.shape} {embeds.dtype}")

    for i, p in enumerate(args.prompts):
        nz = int(np.array(mask[i]).sum())
        v = np.array(embeds[i, :nz]).astype(np.float32)  # fp32 for stable mean/std display
        print(f"  [{i}] {p!r}  tokens={nz}  embed mean={v.mean():+.4f} std={v.std():.4f}")


if __name__ == "__main__":
    _main()
