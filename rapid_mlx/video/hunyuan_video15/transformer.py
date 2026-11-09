# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified for Rapid-MLX: native MLX inference port (decoder only for VAE).
"""HunyuanVideo 1.5 dual-stream transformer. Checkpoint names follow Diffusers."""

from __future__ import annotations

import math
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn


def timestep_embedding(t, dim=256):
    freqs = mx.exp(
        -math.log(10000) * mx.arange(dim // 2, dtype=mx.float32) / (dim // 2)
    )
    phase = t.astype(mx.float32)[:, None] * freqs[None]
    return mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)


def rotary_embedding(shape, patch_size, patch_size_t, axes_dim, theta):
    sizes = (shape[2] // patch_size_t, shape[3] // patch_size, shape[4] // patch_size)
    grids = mx.meshgrid(*(mx.arange(s, dtype=mx.float32) for s in sizes), indexing="ij")
    phases = []
    for grid, dim in zip(grids, axes_dim):
        freq = theta ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        phases.append(mx.repeat(grid.reshape(-1, 1) * freq[None], 2, axis=-1))
    phase = mx.concatenate(phases, axis=-1)
    return mx.cos(phase), mx.sin(phase)


def apply_rotary(x, rope):
    cos, sin = rope
    pairs = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    rotated = mx.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(x.shape)
    return (
        x.astype(mx.float32) * cos[None, :, None] + rotated * sin[None, :, None]
    ).astype(x.dtype)


class TimeMLP(nn.Module):
    def __init__(self, inputs, dim):
        super().__init__()
        self.linear_1 = nn.Linear(inputs, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def __call__(self, x):
        return self.linear_2(nn.silu(self.linear_1(x)))


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.timestep_embedder = TimeMLP(256, dim)

    def __call__(self, t):
        return self.timestep_embedder(
            timestep_embedding(t).astype(self.timestep_embedder.linear_1.weight.dtype)
        )


class CombinedEmbedding(nn.Module):
    def __init__(self, dim, text_dim):
        super().__init__()
        self.timestep_embedder = TimeMLP(256, dim)
        self.text_embedder = TimeMLP(text_dim, dim)

    def __call__(self, t, pooled):
        # PixArtAlphaTextProjection uses SiLU for this combined embedding.
        return self.timestep_embedder(
            timestep_embedding(t).astype(pooled.dtype)
        ) + self.text_embedder(pooled)


class FeedForward(nn.Module):
    def __init__(self, dim, ratio=4, activation="gelu-approximate"):
        super().__init__()
        self.net = [nn.Module(), nn.Identity(), nn.Linear(int(dim * ratio), dim)]
        self.net[0].proj = nn.Linear(dim, int(dim * ratio))
        self.activation = activation

    def __call__(self, x):
        x = self.net[0].proj(x)
        x = nn.silu(x) if self.activation == "linear-silu" else nn.gelu_approx(x)
        return self.net[2](x)


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, joint=False):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.to_q, self.to_k, self.to_v = (nn.Linear(dim, dim) for _ in range(3))
        self.to_out = [nn.Linear(dim, dim)]
        self.joint = joint
        if joint:
            self.norm_q, self.norm_k = (
                nn.RMSNorm(head_dim, eps=1e-6),
                nn.RMSNorm(head_dim, eps=1e-6),
            )
            self.add_q_proj, self.add_k_proj, self.add_v_proj = (
                nn.Linear(dim, dim) for _ in range(3)
            )
            self.norm_added_q, self.norm_added_k = (
                nn.RMSNorm(head_dim, eps=1e-6),
                nn.RMSNorm(head_dim, eps=1e-6),
            )
            self.to_add_out = nn.Linear(dim, dim)

    def __call__(self, x, context=None, mask=None, rope=None):
        def split(y):
            return y.reshape(y.shape[0], y.shape[1], self.heads, self.head_dim)

        q, k, v = split(self.to_q(x)), split(self.to_k(x)), split(self.to_v(x))
        if self.joint:
            q, k = self.norm_q(q), self.norm_k(k)
            if rope is not None:
                q, k = apply_rotary(q, rope), apply_rotary(k, rope)
            cq, ck, cv = (
                split(self.add_q_proj(context)),
                split(self.add_k_proj(context)),
                split(self.add_v_proj(context)),
            )
            q = mx.concatenate([q, self.norm_added_q(cq)], axis=1)
            k = mx.concatenate([k, self.norm_added_k(ck)], axis=1)
            v = mx.concatenate([v, cv], axis=1)
            mask = mx.concatenate(
                [mx.ones(x.shape[:2], dtype=mx.bool_), mask.astype(mx.bool_)], axis=1
            )
        query_valid = None
        if mask is not None and mask.ndim == 2:
            query_valid = mask[:, None, :, None].astype(mx.bool_)
            mask = mask[:, None, None, :].astype(mx.bool_)
        result = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
            scale=self.head_dim**-0.5,
            mask=mask,
        )
        if query_valid is not None:
            # PyTorch SDPA returns zero for entirely masked query rows. Some
            # MLX backends return a uniform average instead; enforce parity.
            result = mx.where(query_valid, result, 0)
        result = result.transpose(0, 2, 1, 3).reshape(
            x.shape[0], -1, self.heads * self.head_dim
        )
        if self.joint:
            return self.to_out[0](result[:, : x.shape[1]]), self.to_add_out(
                result[:, x.shape[1] :]
            )
        return self.to_out[0](result)


class AdaNorm(nn.Module):
    def __init__(self, dim, chunks):
        super().__init__()
        self.linear = nn.Linear(dim, chunks * dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.chunks = chunks

    def __call__(self, x, t):
        parts = mx.split(self.linear(nn.silu(t)), self.chunks, axis=-1)
        # Continuous output norm uses scale, shift; Zero uses shift, scale, gates.
        if self.chunks == 2:
            scale, shift = parts
            return self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        shift, scale, gate, shift_ff, scale_ff, gate_ff = parts
        return (
            self.norm(x) * (1 + scale[:, None]) + shift[:, None],
            gate[:, None],
            shift_ff[:, None],
            scale_ff[:, None],
            gate_ff[:, None],
        )


class TransformerBlock(nn.Module):
    def __init__(
        self, num_attention_heads, attention_head_dim, mlp_ratio=4, qk_norm="rms_norm"
    ):
        super().__init__()
        if qk_norm != "rms_norm":
            raise ValueError("HunyuanVideo 1.5 requires rms_norm")
        dim = num_attention_heads * attention_head_dim
        self.norm1, self.norm1_context = AdaNorm(dim, 6), AdaNorm(dim, 6)
        self.attn = Attention(dim, num_attention_heads, attention_head_dim, joint=True)
        self.norm2, self.norm2_context = (
            nn.LayerNorm(dim, eps=1e-6, affine=False) for _ in range(2)
        )
        self.ff, self.ff_context = (
            FeedForward(dim, mlp_ratio),
            FeedForward(dim, mlp_ratio),
        )

    def __call__(self, x, context, temb, attention_mask, freqs_cis=None):
        h, gate, shift, scale, gate_ff = self.norm1(x, temb)
        c, cgate, cshift, cscale, cgate_ff = self.norm1_context(context, temb)
        h, c = self.attn(h, context=c, mask=attention_mask, rope=freqs_cis)
        x, context = x + gate * h, context + cgate * c
        x = x + gate_ff * self.ff(self.norm2(x) * (1 + scale) + shift)
        context = context + cgate_ff * self.ff_context(
            self.norm2_context(context) * (1 + cscale) + cshift
        )
        return x, context


class RefinerBlock(nn.Module):
    def __init__(self, dim, heads, head_dim):
        super().__init__()
        self.norm1, self.norm2 = (nn.LayerNorm(dim, eps=1e-6) for _ in range(2))
        self.attn = Attention(dim, heads, head_dim)
        self.ff = FeedForward(dim, activation="linear-silu")
        self.norm_out = nn.Module()
        self.norm_out.linear = nn.Linear(dim, 2 * dim)

    def __call__(self, x, t, mask):
        a, b = mx.split(self.norm_out.linear(nn.silu(t)), 2, axis=-1)
        x = x + self.attn(self.norm1(x), mask=mask) * a[:, None]
        return x + self.ff(self.norm2(x)) * b[:, None]


class TokenRefiner(nn.Module):
    def __init__(self, inputs, dim, heads, head_dim, layers):
        super().__init__()
        self.time_text_embed = CombinedEmbedding(dim, inputs)
        self.proj_in = nn.Linear(inputs, dim)
        self.token_refiner = nn.Module()
        self.token_refiner.refiner_blocks = [
            RefinerBlock(dim, heads, head_dim) for _ in range(layers)
        ]

    def __call__(self, x, t, mask):
        m = mask.astype(mx.float32)[..., None]
        pooled = (mx.sum(x.astype(mx.float32) * m, axis=1) / mx.sum(m, axis=1)).astype(
            x.dtype
        )
        temb = self.time_text_embed(t, pooled)
        x = self.proj_in(x)
        for block in self.token_refiner.refiner_blocks:
            x = block(x, temb, mask)
        return x


class ByT5Projection(nn.Module):
    def __init__(self, inputs, dim):
        super().__init__()
        self.norm = nn.LayerNorm(inputs, eps=1e-5)
        self.linear_1, self.linear_2, self.linear_3 = (
            nn.Linear(inputs, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, dim),
        )

    def __call__(self, x):
        return self.linear_3(
            nn.gelu(self.linear_2(nn.gelu(self.linear_1(self.norm(x)))))
        )


class ImageProjection(nn.Module):
    def __init__(self, inputs, dim):
        super().__init__()
        self.norm_in, self.norm_out = nn.LayerNorm(inputs), nn.LayerNorm(dim)
        self.linear_1, self.linear_2 = nn.Linear(inputs, inputs), nn.Linear(inputs, dim)

    def __call__(self, x):
        return self.norm_out(self.linear_2(nn.gelu(self.linear_1(self.norm_in(x)))))


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = self.config = SimpleNamespace(**config)
        d = c.num_attention_heads * c.attention_head_dim
        if c.use_meanflow or c.task_type != "t2v":
            raise ValueError("Only non-distilled T2V is implemented")
        self.x_embedder = nn.Module()
        self.x_embedder.proj = nn.Conv3d(
            c.in_channels,
            d,
            (c.patch_size_t, c.patch_size, c.patch_size),
            stride=(c.patch_size_t, c.patch_size, c.patch_size),
        )
        self.image_embedder = ImageProjection(c.image_embed_dim, d)
        self.context_embedder = TokenRefiner(
            c.text_embed_dim,
            d,
            c.num_attention_heads,
            c.attention_head_dim,
            c.num_refiner_layers,
        )
        self.context_embedder_2 = ByT5Projection(c.text_embed_2_dim, d)
        self.time_embed, self.cond_type_embed = TimeEmbedding(d), nn.Embedding(3, d)
        self.transformer_blocks = [
            TransformerBlock(
                c.num_attention_heads, c.attention_head_dim, c.mlp_ratio, c.qk_norm
            )
            for _ in range(c.num_layers)
        ]
        self.norm_out, self.proj_out = (
            AdaNorm(d, 2),
            nn.Linear(d, c.patch_size_t * c.patch_size**2 * c.out_channels),
        )

    def __call__(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        encoder_hidden_states_2,
        encoder_attention_mask_2,
        image_embeds=None,
        cancel_check=lambda: None,
    ):
        c, shape = self.config, hidden_states.shape
        rope = rotary_embedding(
            shape, c.patch_size, c.patch_size_t, c.rope_axes_dim, c.rope_theta
        )
        t = self.time_embed(timestep)
        x = self.x_embedder.proj(hidden_states.transpose(0, 2, 3, 4, 1)).reshape(
            shape[0], -1, c.num_attention_heads * c.attention_head_dim
        )
        text = (
            self.context_embedder(
                encoder_hidden_states, timestep, encoder_attention_mask
            )
            + self.cond_type_embed.weight[0]
        )
        glyph = (
            self.context_embedder_2(encoder_hidden_states_2)
            + self.cond_type_embed.weight[1]
        )
        # T2V's 729 zero image embeddings still occupy masked conditioning slots.
        if image_embeds is None:
            image_embeds = mx.zeros((shape[0], 729, c.image_embed_dim), dtype=x.dtype)
        image = self.image_embedder(image_embeds) * 0 + self.cond_type_embed.weight[2]
        contexts, masks = [], []
        # Match upstream's stable valid-first order, including zeroed text padding.
        for b in range(shape[0]):
            m1, m2 = (
                encoder_attention_mask[b].tolist(),
                encoder_attention_mask_2[b].tolist(),
            )
            valid1, valid2 = (
                [i for i, v in enumerate(m1) if v],
                [i for i, v in enumerate(m2) if v],
            )
            invalid1, invalid2 = (
                [i for i, v in enumerate(m1) if not v],
                [i for i, v in enumerate(m2) if not v],
            )
            contexts.append(
                mx.concatenate(
                    [
                        glyph[b, mx.array(valid2, dtype=mx.int32)],
                        text[b, mx.array(valid1, dtype=mx.int32)],
                        image[b],
                        mx.zeros(
                            (len(invalid2) + len(invalid1), x.shape[-1]), dtype=x.dtype
                        ),
                    ],
                    axis=0,
                )
            )
            masks.append(mx.arange(contexts[-1].shape[0]) < len(valid1) + len(valid2))
        context, mask = mx.stack(contexts), mx.stack(masks)
        for block in self.transformer_blocks:
            cancel_check()
            x, context = block(x, context, t, mask, rope)
            mx.eval(x, context)
        x = self.proj_out(self.norm_out(x, t))
        pt, p = c.patch_size_t, c.patch_size
        x = x.reshape(
            shape[0],
            shape[2] // pt,
            shape[3] // p,
            shape[4] // p,
            c.out_channels,
            pt,
            p,
            p,
        )
        return x.transpose(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            shape[0], c.out_channels, shape[2], shape[3], shape[4]
        )
