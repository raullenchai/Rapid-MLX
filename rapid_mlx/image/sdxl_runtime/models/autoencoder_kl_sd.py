"""AutoencoderKLSD: a faithful MLX port of diffusers' ``AutoencoderKL`` (SD / SDXL).

Mirrors the diffusers module tree (``encoder.down_blocks.N.resnets.M``, ``mid_block``,
``quant_conv`` …) so the official VAE weights load directly via the converter. The
decoder supports **tiling**: large latents are decoded in overlapping spatial tiles
and feather-blended, which bounds peak memory so 1024px+ images fit on a Mac.

Tensors are channels-last ``(B, H, W, C)``.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn

from ..configuration import Config
from ..modeling import ModelMixin
from .diagonal_gaussian import DiagonalGaussian


@dataclasses.dataclass
class AutoencoderKLSDConfig(Config):
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_groups: int = 32
    scaling_factor: float = 0.13025  # SDXL
    shift_factor: float = (
        0.0  # FLUX/SD3 shift latents before scaling: (z - shift) * scale
    )
    use_quant_conv: bool = True  # SD/SDXL have quant convs; FLUX/SD3 VAEs do not


class _Resnet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        skip = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        return skip + h


class _Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])  # diffusers pads bottom/right
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(self.up(x))


class _Attention(nn.Module):
    """Single-head spatial self-attention (diffusers VAE mid-block attention)."""

    def __init__(self, ch: int, groups: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, ch, pytorch_compatible=True)
        self.to_q = nn.Linear(ch, ch)
        self.to_k = nn.Linear(ch, ch)
        self.to_v = nn.Linear(ch, ch)
        self.to_out = [nn.Linear(ch, ch)]
        self.scale = ch**-0.5

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        y = self.group_norm(x).reshape(b, h * w, c)
        q, k, v = (z[:, None] for z in (self.to_q(y), self.to_k(y), self.to_v(y)))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)[:, 0]
        return x + self.to_out[0](o).reshape(b, h, w, c)


class _MidBlock(nn.Module):
    def __init__(self, ch: int, groups: int):
        super().__init__()
        self.attentions = [_Attention(ch, groups)]
        self.resnets = [_Resnet(ch, ch, groups), _Resnet(ch, ch, groups)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


class _DownBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, n_layers: int, groups: int, add_down: bool
    ):
        super().__init__()
        self.resnets = []
        cur = in_ch
        for _ in range(n_layers):
            self.resnets.append(_Resnet(cur, out_ch, groups))
            cur = out_ch
        self.downsamplers = [_Downsample(out_ch)] if add_down else None

    def __call__(self, x: mx.array) -> mx.array:
        for r in self.resnets:
            x = r(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
        return x


class _UpBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, n_layers: int, groups: int, add_up: bool
    ):
        super().__init__()
        self.resnets = []
        cur = in_ch
        for _ in range(n_layers):
            self.resnets.append(_Resnet(cur, out_ch, groups))
            cur = out_ch
        self.upsamplers = [_Upsample(out_ch)] if add_up else None

    def __call__(self, x: mx.array) -> mx.array:
        for r in self.resnets:
            x = r(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class _Encoder(nn.Module):
    def __init__(self, c: AutoencoderKLSDConfig):
        super().__init__()
        boc = list(c.block_out_channels)
        n = len(boc)
        self.conv_in = nn.Conv2d(c.in_channels, boc[0], 3, padding=1)
        self.down_blocks = []
        out = boc[0]
        for i in range(n):
            in_ch, out = out, boc[i]
            self.down_blocks.append(
                _DownBlock(
                    in_ch, out, c.layers_per_block, c.norm_groups, add_down=i < n - 1
                )
            )
        self.mid_block = _MidBlock(boc[-1], c.norm_groups)
        self.conv_norm_out = nn.GroupNorm(
            c.norm_groups, boc[-1], pytorch_compatible=True
        )
        self.conv_out = nn.Conv2d(boc[-1], 2 * c.latent_channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for b in self.down_blocks:
            x = b(x)
        x = self.mid_block(x)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))


class _Decoder(nn.Module):
    def __init__(self, c: AutoencoderKLSDConfig):
        super().__init__()
        rev = list(reversed(c.block_out_channels))
        n = len(rev)
        self.conv_in = nn.Conv2d(c.latent_channels, rev[0], 3, padding=1)
        self.mid_block = _MidBlock(rev[0], c.norm_groups)
        self.up_blocks = []
        out = rev[0]
        for i in range(n):
            in_ch, out = out, rev[i]
            self.up_blocks.append(
                _UpBlock(
                    in_ch, out, c.layers_per_block + 1, c.norm_groups, add_up=i < n - 1
                )
            )
        self.conv_norm_out = nn.GroupNorm(
            c.norm_groups, rev[-1], pytorch_compatible=True
        )
        self.conv_out = nn.Conv2d(rev[-1], c.out_channels, 3, padding=1)

    def __call__(self, z: mx.array) -> mx.array:
        x = self.conv_in(z)
        x = self.mid_block(x)
        for b in self.up_blocks:
            x = b(x)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))


def _feather(n: int, overlap: int) -> mx.array:
    """A 1-D blend window: linear ramp up over the first ``overlap`` and down the last."""
    w = mx.ones((n,))
    if overlap > 0:
        ramp = (mx.arange(overlap) + 1).astype(mx.float32) / (overlap + 1)
        w = mx.concatenate([ramp, mx.ones((n - 2 * overlap,)), ramp[::-1]])
    return w


class AutoencoderKLSD(ModelMixin[AutoencoderKLSDConfig]):
    """SD / SDXL VAE. Channels-last ``(B, H, W, C)``."""

    config_class = AutoencoderKLSDConfig

    def __init__(self, config: AutoencoderKLSDConfig):
        super().__init__()
        self.config = config
        self.encoder = _Encoder(config)
        self.decoder = _Decoder(config)
        self.quant_conv: nn.Conv2d | None = None
        self.post_quant_conv: nn.Conv2d | None = None
        if config.use_quant_conv:  # FLUX / SD3 VAEs skip the quant convs
            self.quant_conv = nn.Conv2d(
                2 * config.latent_channels, 2 * config.latent_channels, 1
            )
            self.post_quant_conv = nn.Conv2d(
                config.latent_channels, config.latent_channels, 1
            )

    @property
    def scaling_factor(self) -> float:
        return self.config.scaling_factor

    @property
    def shift_factor(self) -> float:
        return self.config.shift_factor

    def encode(self, x: mx.array) -> DiagonalGaussian:
        moments = self.encoder(x)
        if self.quant_conv is not None:
            moments = self.quant_conv(moments)
        return DiagonalGaussian(moments)

    def decode(
        self,
        z: mx.array,
        *,
        tile: bool = False,
        tile_latent: int = 64,
        overlap_latent: int = 16,
    ) -> mx.array:
        """Decode latents to pixels. With ``tile=True``, decode in overlapping tiles.

        Tiling bounds peak memory for large latents: the latent is split into
        ``tile_latent``-sized spatial tiles (stride ``tile_latent - overlap_latent``),
        each decoded independently, then feather-blended back together.
        """
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        h, w = z.shape[1], z.shape[2]
        if not tile or (h <= tile_latent and w <= tile_latent):
            return self.decoder(z)
        return self._tiled(z, tile_latent, overlap_latent)

    def _tiled(self, z: mx.array, tile: int, overlap: int) -> mx.array:
        h, w = z.shape[1], z.shape[2]
        stride = tile - overlap
        ys = sorted({*range(0, max(h - tile, 0) + 1, stride), max(h - tile, 0)})
        xs = sorted({*range(0, max(w - tile, 0) + 1, stride), max(w - tile, 0)})
        scale = 2 ** (
            len(self.config.block_out_channels) - 1
        )  # VAE spatial upsampling factor
        out = mx.zeros((z.shape[0], h * scale, w * scale, self.config.out_channels))
        wsum = mx.zeros((1, h * scale, w * scale, 1))
        for y in ys:
            for x in xs:
                dec = self.decoder(
                    z[:, y : y + tile, x : x + tile]
                )  # (B, tile*8, tile*8, C)
                th, tw = dec.shape[1], dec.shape[2]
                win = (
                    _feather(th, overlap * scale)[:, None]
                    * _feather(tw, overlap * scale)[None]
                )[None, :, :, None]
                py, px = y * scale, x * scale
                out[:, py : py + th, px : px + tw] += dec * win
                wsum[:, py : py + th, px : px + tw] += win
        return out / wsum

    def __call__(
        self, x: mx.array, key: mx.array | None = None
    ) -> tuple[mx.array, DiagonalGaussian]:
        posterior = self.encode(x)
        return self.decode(posterior.sample(key)), posterior
