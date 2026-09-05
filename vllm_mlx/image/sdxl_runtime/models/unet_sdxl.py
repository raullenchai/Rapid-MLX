"""SDXLUNet: a faithful MLX port of diffusers' ``UNet2DConditionModel`` (SDXL base).

Mirrors the diffusers module tree (``down_blocks.N.attentions.M.transformer_blocks.K``
…) so the official SDXL UNet weights load directly via the converter. It is a
cross-attention UNet conditioned on:

* the timestep (sinusoidal -> MLP),
* the **pooled** text embedding plus SDXL's micro-conditioning ``add_time_ids``
  (original size / crop / target size), together forming the *added* embedding, and
* the concatenated per-token CLIP-L + CLIP-bigG embeddings (2048-dim) via cross
  attention.

Tensors are channels-last ``(B, H, W, C)``.
"""

from __future__ import annotations

import dataclasses
import math

import mlx.core as mx
import mlx.nn as nn

from ..caching import DeepCache
from ..configuration import Config
from ..modeling import ModelMixin


@dataclasses.dataclass
class SDXLUNetConfig(Config):
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: tuple[int, ...] = (320, 640, 1280)
    down_block_types: tuple[str, ...] = (
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    )
    up_block_types: tuple[str, ...] = (
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
    )
    layers_per_block: int = 2
    transformer_layers_per_block: tuple[int, ...] = (1, 2, 10)
    num_attention_heads: tuple[int, ...] = (5, 10, 20)
    cross_attention_dim: int = 2048
    addition_time_embed_dim: int = 256
    projection_class_embeddings_input_dim: int = 2816
    norm_groups: int = 32

    @property
    def time_embed_dim(self) -> int:
        return self.block_out_channels[0] * 4


def _timesteps(t: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
    """diffusers ``Timesteps`` (flip_sin_to_cos, shift 0): ``(N,) -> (N, dim)``."""
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t.astype(mx.float32)[:, None] * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class _TimeMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class _Resnet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def __call__(self, x: mx.array, temb: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = h + self.time_emb_proj(nn.silu(temb))[:, None, None, :]
        h = self.conv2(nn.silu(self.norm2(h)))
        skip = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        return skip + h


class _Attention(nn.Module):
    def __init__(self, query_dim: int, heads: int, cross_dim: int | None = None):
        super().__init__()
        self.heads = heads
        self.head_dim = query_dim // heads
        self.scale = self.head_dim**-0.5
        ctx = cross_dim or query_dim
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(ctx, query_dim, bias=False)
        self.to_v = nn.Linear(ctx, query_dim, bias=False)
        self.to_out = [nn.Linear(query_dim, query_dim)]

    def _heads(self, x: mx.array) -> mx.array:
        b, t, _ = x.shape
        return x.reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array, context: mx.array | None = None) -> mx.array:
        c = x if context is None else context
        b, lq = x.shape[0], x.shape[1]
        q, k, v = (
            self._heads(self.to_q(x)),
            self._heads(self.to_k(c)),
            self._heads(self.to_v(c)),
        )
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        o = o.transpose(0, 2, 1, 3).reshape(b, lq, self.heads * self.head_dim)
        return self.to_out[0](o)


class _GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def __call__(self, x: mx.array) -> mx.array:
        x, gate = mx.split(self.proj(x), 2, axis=-1)
        return x * nn.gelu(gate)


class _Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = [_GEGLU(dim, dim * mult), _Identity(), nn.Linear(dim * mult, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.net[2](self.net[0](x))


class _BasicBlock(nn.Module):
    def __init__(self, dim: int, heads: int, cross_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn1 = _Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.attn2 = _Attention(dim, heads, cross_dim)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ff = _FeedForward(dim)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class _Transformer2D(nn.Module):
    """Spatial transformer (use_linear_projection=True, as in SDXL)."""

    def __init__(self, dim: int, heads: int, depth: int, cross_dim: int, groups: int):
        super().__init__()
        self.norm = nn.GroupNorm(groups, dim, pytorch_compatible=True)
        self.proj_in = nn.Linear(dim, dim)
        self.transformer_blocks = [
            _BasicBlock(dim, heads, cross_dim) for _ in range(depth)
        ]
        self.proj_out = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        b, h, w, c = x.shape
        y = self.norm(x).reshape(b, h * w, c)
        y = self.proj_in(y)
        for block in self.transformer_blocks:
            y = block(y, context)
        y = self.proj_out(y).reshape(b, h, w, c)
        return x + y


class _Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(self.up(x))


class _DownBlock(nn.Module):
    cross_attn = False

    def __init__(
        self, in_ch, out_ch, n, temb, groups, add_down, heads=0, depth=0, cross_dim=0
    ):
        super().__init__()
        self.resnets = []
        cur = in_ch
        for _ in range(n):
            self.resnets.append(_Resnet(cur, out_ch, temb, groups))
            cur = out_ch
        self.attentions = (
            [_Transformer2D(out_ch, heads, depth, cross_dim, groups) for _ in range(n)]
            if self.cross_attn
            else None
        )
        self.downsamplers = [_Downsample(out_ch)] if add_down else None

    def __call__(self, x, temb, context):
        res = []
        for i, r in enumerate(self.resnets):
            x = r(x, temb)
            if self.attentions is not None:
                x = self.attentions[i](x, context)
            res.append(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            res.append(x)
        return x, res


class _CrossAttnDownBlock(_DownBlock):
    cross_attn = True


class _UpBlock(nn.Module):
    cross_attn = False

    def __init__(
        self,
        in_ch,
        out_ch,
        prev_out,
        n,
        temb,
        groups,
        add_up,
        heads=0,
        depth=0,
        cross_dim=0,
    ):
        super().__init__()
        self.resnets = []
        for i in range(n):
            res_skip = in_ch if i == n - 1 else out_ch
            res_in = prev_out if i == 0 else out_ch
            self.resnets.append(_Resnet(res_in + res_skip, out_ch, temb, groups))
        self.attentions = (
            [_Transformer2D(out_ch, heads, depth, cross_dim, groups) for _ in range(n)]
            if self.cross_attn
            else None
        )
        self.upsamplers = [_Upsample(out_ch)] if add_up else None

    def __call__(self, x, skips, temb, context):
        for i, r in enumerate(self.resnets):
            x = mx.concatenate([x, skips.pop()], axis=-1)
            x = r(x, temb)
            if self.attentions is not None:
                x = self.attentions[i](x, context)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class _CrossAttnUpBlock(_UpBlock):
    cross_attn = True


class _MidBlock(nn.Module):
    def __init__(self, ch, temb, groups, heads, depth, cross_dim):
        super().__init__()
        self.resnets = [_Resnet(ch, ch, temb, groups), _Resnet(ch, ch, temb, groups)]
        self.attentions = [_Transformer2D(ch, heads, depth, cross_dim, groups)]

    def __call__(self, x, temb, context):
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        return self.resnets[1](x, temb)


class SDXLUNet(ModelMixin[SDXLUNetConfig]):
    """SDXL base UNet. Channels-last ``(B, H, W, C)``."""

    config_class = SDXLUNetConfig

    def __init__(self, config: SDXLUNetConfig):
        super().__init__()
        self.config = config
        c = config
        boc = list(c.block_out_channels)
        n = len(boc)
        temb = c.time_embed_dim
        g = c.norm_groups

        self.conv_in = nn.Conv2d(c.in_channels, boc[0], 3, padding=1)
        self.time_embedding = _TimeMLP(boc[0], temb)
        self.add_embedding = _TimeMLP(c.projection_class_embeddings_input_dim, temb)

        self.down_blocks: list = []
        out = boc[0]
        for i in range(n):
            in_ch, out = out, boc[i]
            cross = c.down_block_types[i].startswith("CrossAttn")
            down_cls = _CrossAttnDownBlock if cross else _DownBlock
            self.down_blocks.append(
                down_cls(
                    in_ch,
                    out,
                    c.layers_per_block,
                    temb,
                    g,
                    add_down=i < n - 1,
                    heads=c.num_attention_heads[i],
                    depth=c.transformer_layers_per_block[i],
                    cross_dim=c.cross_attention_dim,
                )
            )

        self.mid_block = _MidBlock(
            boc[-1],
            temb,
            g,
            c.num_attention_heads[-1],
            c.transformer_layers_per_block[-1],
            c.cross_attention_dim,
        )

        rev_boc = list(reversed(boc))
        rev_heads = list(reversed(c.num_attention_heads))
        rev_depth = list(reversed(c.transformer_layers_per_block))
        self.up_blocks: list = []
        prev_out = boc[-1]
        for i in range(n):
            out = rev_boc[i]
            in_ch = rev_boc[
                min(i + 1, n - 1)
            ]  # skip channel = symmetric down level input
            cross = c.up_block_types[i].startswith("CrossAttn")
            up_cls = _CrossAttnUpBlock if cross else _UpBlock
            self.up_blocks.append(
                up_cls(
                    in_ch,
                    out,
                    prev_out,
                    c.layers_per_block + 1,
                    temb,
                    g,
                    add_up=i < n - 1,
                    heads=rev_heads[i],
                    depth=rev_depth[i],
                    cross_dim=c.cross_attention_dim,
                )
            )
            prev_out = out

        self.conv_norm_out = nn.GroupNorm(g, boc[0], pytorch_compatible=True)
        self.conv_out = nn.Conv2d(boc[0], c.out_channels, 3, padding=1)

    def __call__(
        self,
        sample: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        text_embeds: mx.array,
        time_ids: mx.array,
        cache: DeepCache | None = None,
    ) -> mx.array:
        """Predict noise for latents ``sample`` ``(B, H, W, C)``.

        ``encoder_hidden_states`` is the ``(B, L, 2048)`` concatenated CLIP context;
        ``text_embeds`` the ``(B, 1280)`` pooled bigG embedding; ``time_ids`` the
        ``(B, 6)`` SDXL micro-conditioning (sizes / crop). With a ``DeepCache``, the
        deep blocks may be skipped on most steps.
        """
        b = sample.shape[0]
        emb = self.time_embedding(
            _timesteps(
                mx.broadcast_to(timestep, (b,)), self.config.block_out_channels[0]
            )
        )
        tid = _timesteps(
            time_ids.reshape(-1), self.config.addition_time_embed_dim
        ).reshape(b, -1)
        emb = emb + self.add_embedding(mx.concatenate([text_embeds, tid], axis=-1))

        ehs = encoder_hidden_states
        if cache is not None and cache.should_reuse():
            return self._cached_forward(sample, emb, ehs, cache)

        x = self.conv_in(sample)
        res_samples = [x]
        for block in self.down_blocks:
            x, res = block(x, emb, ehs)
            res_samples += res

        x = self.mid_block(x, emb, ehs)

        for block in self.up_blocks[:-1]:
            take = len(block.resnets)
            x = block(x, res_samples[-take:], emb, ehs)
            res_samples = res_samples[:-take]

        if cache is not None:
            cache.deep = x  # the input to the shallow up block — reused on cached steps

        last = self.up_blocks[-1]
        x = last(x, res_samples[-len(last.resnets) :], emb, ehs)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))

    def _cached_forward(self, sample, emb, ehs, cache: DeepCache) -> mx.array:
        """DeepCache step: run only conv_in + first down block + last up block."""
        x = self.conv_in(sample)
        shallow = [x]
        x, res0 = self.down_blocks[0](x, emb, ehs)
        shallow += res0
        last = self.up_blocks[-1]
        skips = shallow[: len(last.resnets)]  # conv_in + first down block's resnets
        x = last(cache.deep, skips, emb, ehs)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))
