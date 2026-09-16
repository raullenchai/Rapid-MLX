"""Converters for the Stable Diffusion XL components (diffusers/transformers format).

Each converter turns one component folder of an ``stable-diffusion-xl-base`` checkpoint
into the matching MLX model. The CLIP text encoders convert nearly identity (module
names mirror transformers); the VAE and UNet need conv kernels reordered to
channels-last.
"""

from __future__ import annotations

import mlx.core as mx

from ..models.autoencoder_kl_sd import AutoencoderKLSD, AutoencoderKLSDConfig
from ..models.clip_text import CLIPTextConfig, CLIPTextModel
from ..models.unet_sdxl import SDXLUNet, SDXLUNetConfig
from .base import Converter, convert_conv_weight, register_converter


class _CLIPConverter(Converter):
    with_projection = False

    def build_config(self, hf_config: dict) -> CLIPTextConfig:
        return CLIPTextConfig(
            vocab_size=hf_config["vocab_size"],
            hidden_size=hf_config["hidden_size"],
            intermediate_size=hf_config["intermediate_size"],
            num_hidden_layers=hf_config["num_hidden_layers"],
            num_attention_heads=hf_config["num_attention_heads"],
            max_position_embeddings=hf_config["max_position_embeddings"],
            hidden_act=hf_config["hidden_act"],
            projection_dim=hf_config.get("projection_dim", hf_config["hidden_size"]),
            layer_norm_eps=hf_config.get("layer_norm_eps", 1e-5),
            with_projection=self.with_projection,
        )

    def build_model(self, hf_config: dict) -> CLIPTextModel:
        return CLIPTextModel(self.build_config(hf_config))

    def convert_weights(
        self, weights: dict[str, mx.array], hf_config: dict
    ) -> dict[str, mx.array]:
        # Identity, minus registered buffers (position_ids) that aren't parameters.
        return {k: v for k, v in weights.items() if not k.endswith("position_ids")}


@register_converter
class CLIPTextModelConverter(_CLIPConverter):
    source_class = "CLIPTextModel"
    with_projection = False


@register_converter
class CLIPTextModelWithProjectionConverter(_CLIPConverter):
    source_class = "CLIPTextModelWithProjection"
    with_projection = True


@register_converter
class SDVAEConverter(Converter):
    source_class = "AutoencoderKL"

    def build_config(self, hf_config: dict) -> AutoencoderKLSDConfig:
        return AutoencoderKLSDConfig(
            in_channels=hf_config["in_channels"],
            out_channels=hf_config["out_channels"],
            latent_channels=hf_config["latent_channels"],
            block_out_channels=tuple(hf_config["block_out_channels"]),
            layers_per_block=hf_config["layers_per_block"],
            norm_groups=hf_config.get("norm_num_groups", 32),
            scaling_factor=hf_config.get("scaling_factor", 0.18215),
            # FLUX / SD3 VAEs shift latents and drop the quant convs; SD/SDXL do not.
            shift_factor=hf_config.get("shift_factor") or 0.0,
            use_quant_conv=hf_config.get("use_quant_conv", True),
        )

    def build_model(self, hf_config: dict) -> AutoencoderKLSD:
        return AutoencoderKLSD(self.build_config(hf_config))

    def convert_weights(
        self, weights: dict[str, mx.array], hf_config: dict
    ) -> dict[str, mx.array]:
        # Conv kernels -> channels-last; Linear/GroupNorm weights already match.
        return {
            k: (convert_conv_weight(v) if v.ndim == 4 else v)
            for k, v in weights.items()
        }


def _heads_from(hf_config: dict) -> tuple[int, ...]:
    # SDXL stores the per-level head COUNT under attention_head_dim (legacy naming).
    nah = hf_config.get("num_attention_heads") or hf_config["attention_head_dim"]
    return (
        tuple(nah)
        if isinstance(nah, (list, tuple))
        else (nah,) * len(hf_config["block_out_channels"])
    )


@register_converter
class SDXLUNetConverter(Converter):
    source_class = "UNet2DConditionModel"

    def build_config(self, hf_config: dict) -> SDXLUNetConfig:
        tlpb = hf_config["transformer_layers_per_block"]
        boc = hf_config["block_out_channels"]
        return SDXLUNetConfig(
            in_channels=hf_config["in_channels"],
            out_channels=hf_config["out_channels"],
            block_out_channels=tuple(boc),
            down_block_types=tuple(hf_config["down_block_types"]),
            up_block_types=tuple(hf_config["up_block_types"]),
            layers_per_block=hf_config["layers_per_block"],
            transformer_layers_per_block=tuple(tlpb)
            if isinstance(tlpb, (list, tuple))
            else (tlpb,) * len(boc),
            num_attention_heads=_heads_from(hf_config),
            cross_attention_dim=hf_config["cross_attention_dim"],
            addition_time_embed_dim=hf_config["addition_time_embed_dim"],
            projection_class_embeddings_input_dim=hf_config[
                "projection_class_embeddings_input_dim"
            ],
            norm_groups=hf_config.get("norm_num_groups", 32),
        )

    def build_model(self, hf_config: dict) -> SDXLUNet:
        return SDXLUNet(self.build_config(hf_config))

    def convert_weights(
        self, weights: dict[str, mx.array], hf_config: dict
    ) -> dict[str, mx.array]:
        return {
            k: (convert_conv_weight(v) if v.ndim == 4 else v)
            for k, v in weights.items()
        }
