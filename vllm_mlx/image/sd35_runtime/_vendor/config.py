# reference: https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion

#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import mlx.core as mx


class PositionalEncoding(Enum):
    LearnedInputEmbedding = 1
    PreSDPARope = 2


@dataclass
class MMDiTConfig:
    """Multi-modal Diffusion Transformer Configuration"""

    # Transformer spec
    num_heads: int = 24
    depth_multimodal: int = 24  # e.g. SD3: 24 (2b) or 38 (8b), FLUX.1: 19
    depth_unified: int = 0  # e.g. SD3: 0 (2b and 8b),      FLUX.1: 38
    parallel_mlp_for_unified_blocks: bool = (
        True  # e.g. FLUX.1 unified blocks, https://arxiv.org/pdf/2302.05442
    )
    mlp_ratio: int = 4
    vae_latent_dim: int = 16  # = in_channels = out_channels
    layer_norm_eps: float = 1e-6
    pos_embed_type: PositionalEncoding = PositionalEncoding.LearnedInputEmbedding
    rope_axes_dim: Optional[Tuple[int]] = None
    # FLUX uses RMSNorm post-QK projection
    use_qk_norm: bool = False
    upcast_multimodal_blocks: Optional[List[int]] = None
    upcast_unified_blocks: Optional[List[int]] = None

    # 64 * self.depth_multimodal is the SD3 convention, but can be overridden
    hidden_size_override: Optional[int] = None

    @property
    def hidden_size(self) -> int:
        return self.hidden_size_override or (64 * self.depth_multimodal)

    # x: Latent image input spec
    max_latent_resolution: int = 192
    patch_size: int = 2
    # If true, reshapes input to enact (patch_size, patch_size) space-to-depth operation
    # If false, uses 2D convolution with kernel_size=patch_size and stride=patch_size
    patchify_via_reshape: bool = False

    # y: Text input spec
    # e.g. SD3:    768 (CLIP-L/14) + 1280 (CLIP-G/14) = 2048
    #      FLUX.1: 768 (CLIP-L/14)                    = 768
    pooled_text_embed_dim: int = 2048
    # e.g. SD3:  4096 (T5-XXL) = 768 (CLIP-L/14) + 1280 (CLIP-G/14) + 2048 (zero padding)
    #      FLUX: 4096 (T5-XXL)
    token_level_text_embed_dim: int = 4096

    # t: Timestep input spec
    frequency_embed_dim: int = 256
    max_period: int = 10000

    dtype: mx.Dtype = mx.bfloat16
    float16_dtype: mx.Dtype = mx.bfloat16

    low_memory_mode: bool = True

    guidance_embed: bool = False


SD3_8b = MMDiTConfig(
    depth_multimodal=38, num_heads=38, upcast_multimodal_blocks=[35], use_qk_norm=True
)

SD3_2b = MMDiTConfig(
    depth_multimodal=24, num_heads=24, float16_dtype=mx.float16, dtype=mx.float16
)

FLUX_SCHNELL = MMDiTConfig(
    num_heads=24,
    depth_multimodal=19,
    depth_unified=38,
    parallel_mlp_for_unified_blocks=True,
    hidden_size_override=3072,
    patchify_via_reshape=True,
    pos_embed_type=PositionalEncoding.PreSDPARope,
    rope_axes_dim=(16, 56, 56),
    pooled_text_embed_dim=768,  # CLIP-L/14 only
    use_qk_norm=True,
    float16_dtype=mx.bfloat16,
    dtype=mx.bfloat16,
)

FLUX_DEV = MMDiTConfig(
    num_heads=24,
    depth_multimodal=19,
    depth_unified=38,
    parallel_mlp_for_unified_blocks=True,
    hidden_size_override=3072,
    patchify_via_reshape=True,
    pos_embed_type=PositionalEncoding.PreSDPARope,
    rope_axes_dim=(16, 56, 56),
    pooled_text_embed_dim=768,  # CLIP-L/14 only
    use_qk_norm=True,
    float16_dtype=mx.bfloat16,
    guidance_embed=True,
    dtype=mx.bfloat16,
)


@dataclass
class AutoencoderConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels_out: int = 8
    latent_channels_in: int = 4
    block_out_channels: Tuple[int] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    scaling_factor: float = 0.18215


@dataclass
class VAEDecoderConfig:
    in_channels: int = 16
    out_channels: int = 3
    block_out_channels: Tuple[int] = (128, 256, 512, 512)
    layers_per_block: int = 3
    resnet_groups: int = 32


@dataclass
class VAEEncoderConfig:
    in_channels: int = 3
    out_channels: int = 32
    block_out_channels: Tuple[int] = (128, 256, 512, 512)
    layers_per_block: int = 2
    resnet_groups: int = 32


@dataclass
class CLIPTextModelConfig:
    num_layers: int = 23
    model_dims: int = 1024
    num_heads: int = 16
    max_length: int = 77
    vocab_size: int = 49408
    projection_dim: Optional[int] = None
    hidden_act: str = "quick_gelu"
