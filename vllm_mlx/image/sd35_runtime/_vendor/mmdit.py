#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

# fmt: off

from functools import partial
import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from mlx.utils import tree_map

from .config import MMDiTConfig, PositionalEncoding

logger = logging.getLogger(__name__)

SDPA_FLASH_ATTN_THRESHOLD = 1024


class MMDiT(nn.Module):
    """Multi-modal Diffusion Transformer Architecture
    as described in https://arxiv.org/abs/2403.03206
    """

    def __init__(self, config: MMDiTConfig):
        super().__init__()
        self.config = config

        if config.guidance_embed:
            self.guidance_in = MLPEmbedder(
                in_dim=config.frequency_embed_dim, hidden_dim=config.hidden_size
            )
        else:
            self.guidance_in = nn.Identity()

        # Input adapters and embeddings
        self.x_embedder = LatentImageAdapter(config)

        if config.pos_embed_type == PositionalEncoding.LearnedInputEmbedding:
            self.x_pos_embedder = LatentImagePositionalEmbedding(config)
            self.pre_sdpa_rope = nn.Identity()
        elif config.pos_embed_type == PositionalEncoding.PreSDPARope:
            self.pre_sdpa_rope = RoPE(
                theta=10000,
                axes_dim=config.rope_axes_dim,
            )
        else:
            raise ValueError(
                f"Unsupported positional encoding type: {config.pos_embed_type}"
            )

        self.y_embedder = PooledTextEmbeddingAdapter(config)
        self.t_embedder = TimestepAdapter(config)
        self.context_embedder = nn.Linear(
            config.token_level_text_embed_dim,
            config.hidden_size,
        )

        self.multimodal_transformer_blocks = [
            MultiModalTransformerBlock(
                config,
                skip_text_post_sdpa=(i == config.depth_multimodal - 1)
                and (config.depth_unified < 1),
            )
            for i in range(config.depth_multimodal)
        ]

        if config.depth_unified > 0:
            self.unified_transformer_blocks = [
                UnifiedTransformerBlock(config) for _ in range(config.depth_unified)
            ]

        self.final_layer = FinalLayer(config)

    def cache_modulation_params(
        self,
        pooled_text_embeddings: mx.array,
        timesteps: mx.array,
    ):
        """Compute modulation parameters ahead of time to reduce peak memory load during MMDiT inference
        by offloading all adaLN_modulation parameters
        """
        y_embed = self.y_embedder(pooled_text_embeddings)
        batch_size = pooled_text_embeddings.shape[0]

        offload_size = 0
        to_offload = []

        for timestep in timesteps:
            final_timestep = timestep.item() == timesteps[-1].item()
            timestep_key = timestep.item()
            modulation_inputs = y_embed[:, None, None, :] + self.t_embedder(
                mx.repeat(timestep[None], batch_size, axis=0)
            )

            for block in self.multimodal_transformer_blocks:
                if not hasattr(block.image_transformer_block, "_modulation_params"):
                    block.image_transformer_block._modulation_params = dict()
                    block.text_transformer_block._modulation_params = dict()

                block.image_transformer_block._modulation_params[
                    timestep_key
                ] = block.image_transformer_block.adaLN_modulation(modulation_inputs)
                block.text_transformer_block._modulation_params[
                    timestep_key
                ] = block.text_transformer_block.adaLN_modulation(modulation_inputs)
                mx.eval(block.image_transformer_block._modulation_params[timestep_key])
                mx.eval(block.text_transformer_block._modulation_params[timestep_key])

                if final_timestep:
                    offload_size += (
                        block.image_transformer_block.adaLN_modulation.layers[
                            1
                        ].weight.size
                        * block.image_transformer_block.adaLN_modulation.layers[
                            1
                        ].weight.dtype.size
                    )
                    offload_size += (
                        block.text_transformer_block.adaLN_modulation.layers[
                            1
                        ].weight.size
                        * block.text_transformer_block.adaLN_modulation.layers[
                            1
                        ].weight.dtype.size
                    )
                    to_offload.extend(
                        [
                            block.image_transformer_block.adaLN_modulation.layers[1],
                            block.text_transformer_block.adaLN_modulation.layers[1],
                        ]
                    )

            if self.config.depth_unified > 0:
                for block in self.unified_transformer_blocks:
                    if not hasattr(block.transformer_block, "_modulation_params"):
                        block.transformer_block._modulation_params = dict()
                    block.transformer_block._modulation_params[
                        timestep_key
                    ] = block.transformer_block.adaLN_modulation(modulation_inputs)
                    mx.eval(block.transformer_block._modulation_params[timestep_key])

                    if final_timestep:
                        offload_size += (
                            block.transformer_block.adaLN_modulation.layers[
                                1
                            ].weight.size
                            * block.transformer_block.adaLN_modulation.layers[
                                1
                            ].weight.dtype.size
                        )
                        to_offload.extend(
                            [block.transformer_block.adaLN_modulation.layers[1]]
                        )

            if not hasattr(self.final_layer, "_modulation_params"):
                self.final_layer._modulation_params = dict()
            self.final_layer._modulation_params[
                timestep_key
            ] = self.final_layer.adaLN_modulation(modulation_inputs)
            mx.eval(self.final_layer._modulation_params[timestep_key])

            if final_timestep:
                offload_size += (
                    self.final_layer.adaLN_modulation.layers[1].weight.size
                    * self.final_layer.adaLN_modulation.layers[1].weight.dtype.size
                )
                to_offload.extend([self.final_layer.adaLN_modulation.layers[1]])

        self.to_offload = to_offload
        for x in self.to_offload:
            x.update(tree_map(lambda _: mx.array([]), x.parameters()))
            # x.clear()

        logger.info(f"Cached modulation_params for timesteps={timesteps}")
        logger.info(
            f"Cached modulation_params will reduce peak memory by {(offload_size) / 1e9:.1f} GB"
        )

    def clear_modulation_params_cache(self):
        for name, module in self.named_modules():
            if hasattr(module, "_modulation_params"):
                delattr(module, "_modulation_params")
        logger.info("Cleared modulation_params cache")

    def __call__(
        self,
        latent_image_embeddings: mx.array,
        token_level_text_embeddings: mx.array,
        timestep: mx.array,
    ) -> mx.array:
        batch, latent_height, latent_width, _ = latent_image_embeddings.shape
        token_level_text_embeddings = self.context_embedder(token_level_text_embeddings)

        if hasattr(self, "x_pos_embedder"):
            latent_image_embeddings = self.x_embedder(
                latent_image_embeddings
            ) + self.x_pos_embedder(latent_image_embeddings)
        else:
            latent_image_embeddings = self.x_embedder(latent_image_embeddings)

        latent_image_embeddings = latent_image_embeddings.reshape(
            batch, -1, 1, self.config.hidden_size
        )

        if self.config.pos_embed_type == PositionalEncoding.PreSDPARope:
            positional_encodings = self.pre_sdpa_rope(
                text_sequence_length=token_level_text_embeddings.shape[1],
                latent_image_resolution=(
                    latent_height // self.config.patch_size,
                    latent_width // self.config.patch_size,
                ),
            )
        else:
            positional_encodings = None

        if self.config.guidance_embed:
            timestep = self.guidance_in(self.t_embedder(timestep))

        # MultiModalTransformer layers
        if self.config.depth_multimodal > 0:
            for bidx, block in enumerate(self.multimodal_transformer_blocks):
                latent_image_embeddings, token_level_text_embeddings = block(
                    latent_image_embeddings,
                    token_level_text_embeddings,
                    timestep,
                    positional_encodings=positional_encodings,
                )

        # UnifiedTransformerBlock layers
        if self.config.depth_unified > 0:
            latent_unified_embeddings = mx.concatenate(
                (token_level_text_embeddings, latent_image_embeddings), axis=1
            )

            for bidx, block in enumerate(self.unified_transformer_blocks):
                latent_unified_embeddings = block(
                    latent_unified_embeddings,
                    timestep,
                    positional_encodings=positional_encodings,
                )

            latent_image_embeddings = latent_unified_embeddings[
                :, token_level_text_embeddings.shape[1] :, ...
            ]

        latent_image_embeddings = self.final_layer(
            latent_image_embeddings,
            timestep,
        )

        if self.config.patchify_via_reshape:
            latent_image_embeddings = self.x_embedder.unpack(
                latent_image_embeddings, (latent_height, latent_width)
            )
        else:
            latent_image_embeddings = unpatchify(
                latent_image_embeddings,
                patch_size=self.config.patch_size,
                target_height=latent_height,
                target_width=latent_width,
                vae_latent_dim=self.config.vae_latent_dim,
            )
        return latent_image_embeddings


class LatentImageAdapter(nn.Module):
    """Adapts the latent image input by:
    - Patchifying to reduce sequence length by `config.patch_size ** 2`
    - Projecting to `hidden_size`
    """

    def __init__(self, config: MMDiTConfig):
        super().__init__()
        self.config = config
        in_dim = config.vae_latent_dim
        kernel_size = stride = config.patch_size

        if config.patchify_via_reshape:
            in_dim *= config.patch_size**2
            kernel_size = stride = 1

        self.proj = nn.Conv2d(
            in_dim,
            config.hidden_size,
            kernel_size,
            stride,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.config.patchify_via_reshape:
            b, h_latent, w_latent, c = x.shape
            p = self.config.patch_size
            x = (
                x.reshape(b, h_latent // p, p, w_latent // p, p, c)
                .transpose(0, 1, 3, 5, 2, 4)
                .reshape(b, h_latent // p, w_latent // p, -1)
            )

        return self.proj(x)

    def unpack(self, x: mx.array, latent_image_resolution: Tuple[int]) -> mx.array:
        """Unpacks the latent image embeddings to the original resolution
        for `config.patchify_via_reshape` models
        """
        assert self.config.patchify_via_reshape

        b = x.shape[0]
        p = self.config.patch_size
        h = latent_image_resolution[0] // p
        w = latent_image_resolution[1] // p
        x = (
            x.reshape(
                b, h, w, -1, p, p
            )  # (b, hw, 1, (c*ph*pw)) -> (b, h, w, c, ph, pw)
            .transpose(0, 1, 4, 2, 5, 3)  # (b, h, w, c, ph, pw) -> (b, h, ph, w, pw, c)
            .reshape(b, h * p, w * p, -1)  # (b, h, ph, w, pw, c) -> (b, h*ph, w*pw, c)
        )
        return x


class LatentImagePositionalEmbedding(nn.Module):
    def __init__(self, config: MMDiTConfig):
        super().__init__()
        self.pos_embed = nn.Embedding(
            num_embeddings=config.max_latent_resolution**2, dims=config.hidden_size
        )
        self.max_hw = config.max_latent_resolution
        self.patch_size = config.patch_size
        self.weight_shape = (1, self.max_hw, self.max_hw, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, _ = x.shape
        assert h <= self.max_hw and w <= self.max_hw

        h = h // self.patch_size
        w = w // self.patch_size

        # Center crop the positional embedding to match the input resolution
        y0 = (self.max_hw - h) // 2
        y1 = y0 + h
        x0 = (self.max_hw - w) // 2
        x1 = x0 + w

        w = self.pos_embed.weight.reshape(*self.weight_shape)
        w = w[:, y0:y1, x0:x1, :]
        return mx.repeat(w, repeats=b, axis=0)


class PooledTextEmbeddingAdapter(nn.Module):
    def __init__(self, config: MMDiTConfig):
        super().__init__()

        d1, d2 = config.pooled_text_embed_dim, config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(d1, d2),
            nn.SiLU(),
            nn.Linear(d2, d2),
        )

    def __call__(self, y: mx.array) -> mx.array:
        return self.mlp(y)


class TimestepAdapter(nn.Module):
    def __init__(self, config: MMDiTConfig):
        super().__init__()

        d1, d2 = config.frequency_embed_dim, config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(d1, d2),
            nn.SiLU(),
            nn.Linear(d2, d2),
        )
        self.config = config

    def timestep_embedding(self, t: mx.array) -> mx.array:
        half = self.config.frequency_embed_dim // 2

        frequencies = mx.exp(
            -mx.log(mx.array(self.config.max_period))
            * mx.arange(start=0, stop=half, dtype=self.config.dtype)
            / half
        ).astype(self.config.dtype)

        args = t[:, None].astype(self.config.dtype) * frequencies[None]
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

    def __call__(self, t: mx.array) -> mx.array:
        return self.mlp(self.timestep_embedding(t)[:, None, None, :])


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: MMDiTConfig,
        skip_post_sdpa: bool = False,
        parallel_mlp: bool = False,
        num_modulation_params: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.parallel_mlp = parallel_mlp
        self.skip_post_sdpa = skip_post_sdpa
        self.per_head_dim = config.hidden_size // config.num_heads

        self.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = Attention(config.hidden_size, config.num_heads)
        if not self.parallel_mlp:
            # If parallel, reuse norm1 across attention and mlp
            self.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if skip_post_sdpa:
            self.attn.o_proj = nn.Identity()
        else:
            self.mlp = FFN(
                embed_dim=config.hidden_size,
                expansion_factor=config.mlp_ratio,
                activation_fn=nn.GELU(),
            )

        if num_modulation_params is None:
            num_modulation_params = 6
            if skip_post_sdpa:
                num_modulation_params = 2

        self.num_modulation_params = num_modulation_params
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                config.hidden_size, self.num_modulation_params * config.hidden_size
            ),
        )

        if config.use_qk_norm:
            self.qk_norm = QKNorm(config.hidden_size // config.num_heads)

    def pre_sdpa(
        self,
        tensor: mx.array,
        timestep: mx.array,
    ) -> Dict[str, mx.array]:
        if timestep.size > 1:
            timestep = timestep[0]
        modulation_params = self._modulation_params[timestep.item()]

        modulation_params = mx.split(
            modulation_params, self.num_modulation_params, axis=-1
        )

        post_norm1_shift = modulation_params[0]
        post_norm1_residual_scale = modulation_params[1]

        # LayerNorm and modulate before SDPA
        try:
            modulated_pre_attention = affine_transform(
                tensor,
                shift=post_norm1_shift,
                residual_scale=post_norm1_residual_scale,
                norm_module=self.norm1,
            )
        except Exception as e:
            logger.error(
                f"Error in pre_sdpa: {e}",
                exc_info=True,
            )
            raise e

        q = self.attn.q_proj(modulated_pre_attention)
        k = self.attn.k_proj(modulated_pre_attention)
        v = self.attn.v_proj(modulated_pre_attention)

        batch = tensor.shape[0]

        def rearrange_for_norm(t):
            # Target data layout: (batch, head, seq_len, channel)
            return t.reshape(
                batch, -1, self.config.num_heads, self.per_head_dim
            ).transpose(0, 2, 1, 3)

        q = rearrange_for_norm(q)
        k = rearrange_for_norm(k)
        v = rearrange_for_norm(v)

        if self.config.use_qk_norm:
            q, k = self.qk_norm(q, k)

        if self.config.depth_unified == 0:
            q = q.transpose(0, 2, 1, 3).reshape(batch, -1, 1, self.config.hidden_size)
            k = k.transpose(0, 2, 1, 3).reshape(batch, -1, 1, self.config.hidden_size)
            v = v.transpose(0, 2, 1, 3).reshape(batch, -1, 1, self.config.hidden_size)

        results = {"q": q, "k": k, "v": v}

        results["modulated_pre_attention"] = modulated_pre_attention

        assert len(modulation_params) in [2, 3, 6]
        results.update(
            {
                "post_norm1_shift": post_norm1_shift,
                "post_norm1_residual_scale": post_norm1_residual_scale,
            }
        )

        if len(modulation_params) > 2:
            results.update({"post_attn_scale": modulation_params[2]})

        if len(modulation_params) > 3:
            results.update(
                {
                    "post_norm2_shift": modulation_params[3],
                    "post_norm2_residual_scale": modulation_params[4],
                    "post_mlp_scale": modulation_params[5],
                }
            )

        return results

    def post_sdpa(
        self,
        residual: mx.array,
        sdpa_output: mx.array,
        modulated_pre_attention: mx.array,
        post_attn_scale: Optional[mx.array] = None,
        post_norm2_shift: Optional[mx.array] = None,
        post_norm2_residual_scale: Optional[mx.array] = None,
        post_mlp_scale: Optional[mx.array] = None,
        **kwargs,
    ):
        attention_out = self.attn.o_proj(sdpa_output)
        if self.parallel_mlp:
            # Reuse the modulation parameters and self.norm1 across attn and mlp
            mlp_out = self.mlp(modulated_pre_attention)
            return residual + post_attn_scale * (attention_out + mlp_out)
        else:
            residual = residual + attention_out * post_attn_scale
            # Apply separate modulation parameters and LayerNorm across attn and mlp
            mlp_out = self.mlp(
                affine_transform(
                    residual,
                    shift=post_norm2_shift,
                    residual_scale=post_norm2_residual_scale,
                    norm_module=self.norm2,
                )
            )
            return residual + post_mlp_scale * mlp_out

    def __call__(self):
        raise NotImplementedError("This module is not intended to be used directly")


class MultiModalTransformerBlock(nn.Module):
    def __init__(self, config: MMDiTConfig, skip_text_post_sdpa: bool = False):
        super().__init__()
        self.image_transformer_block = TransformerBlock(config)
        self.text_transformer_block = TransformerBlock(
            config, skip_post_sdpa=skip_text_post_sdpa
        )

        sdpa_impl = mx.fast.scaled_dot_product_attention
        self.sdpa = partial(sdpa_impl)

        self.config = config
        self.per_head_dim = config.hidden_size // config.num_heads

    def __call__(
        self,
        latent_image_embeddings: mx.array,  # latent image embeddings
        token_level_text_embeddings: mx.array,  # token-level text embeddings
        timestep: mx.array,  # pooled text embeddings + timestep embeddings
        positional_encodings: mx.array = None,  # positional encodings for rope
    ):
        # Prepare multi-modal SDPA inputs
        image_intermediates = self.image_transformer_block.pre_sdpa(
            latent_image_embeddings,
            timestep=timestep,
        )

        text_intermediates = self.text_transformer_block.pre_sdpa(
            token_level_text_embeddings,
            timestep=timestep,
        )

        batch = latent_image_embeddings.shape[0]

        def rearrange_for_sdpa(t):
            # Target data layout: (batch, head, seq_len, channel)
            return t.reshape(
                batch, -1, self.config.num_heads, self.per_head_dim
            ).transpose(0, 2, 1, 3)

        if self.config.depth_unified > 0:
            multimodal_sdpa_inputs = {
                "q": mx.concatenate(
                    [text_intermediates["q"], image_intermediates["q"]], axis=2
                ),
                "k": mx.concatenate(
                    [text_intermediates["k"], image_intermediates["k"]], axis=2
                ),
                "v": mx.concatenate(
                    [text_intermediates["v"], image_intermediates["v"]], axis=2
                ),
                "scale": 1.0 / np.sqrt(self.per_head_dim),
            }
        else:
            multimodal_sdpa_inputs = {
                "q": rearrange_for_sdpa(
                    mx.concatenate(
                        [image_intermediates["q"], text_intermediates["q"]], axis=1
                    )
                ),
                "k": rearrange_for_sdpa(
                    mx.concatenate(
                        [image_intermediates["k"], text_intermediates["k"]], axis=1
                    )
                ),
                "v": rearrange_for_sdpa(
                    mx.concatenate(
                        [image_intermediates["v"], text_intermediates["v"]], axis=1
                    )
                ),
                "scale": 1.0 / np.sqrt(self.per_head_dim),
            }

        if self.config.pos_embed_type == PositionalEncoding.PreSDPARope:
            assert positional_encodings is not None
            multimodal_sdpa_inputs["q"] = RoPE.apply(
                multimodal_sdpa_inputs["q"], positional_encodings
            )
            multimodal_sdpa_inputs["k"] = RoPE.apply(
                multimodal_sdpa_inputs["k"], positional_encodings
            )

        # Compute multi-modal SDPA
        sdpa_outputs = (
            self.sdpa(**multimodal_sdpa_inputs)
            .transpose(0, 2, 1, 3)
            .reshape(batch, -1, 1, self.config.hidden_size)
        )

        # Split into image-text sequences for post-SDPA layers
        img_seq_len = latent_image_embeddings.shape[1]
        txt_seq_len = token_level_text_embeddings.shape[1]

        if self.config.depth_unified > 0:
            text_sdpa_output = sdpa_outputs[:, :txt_seq_len, :, :]
            image_sdpa_output = sdpa_outputs[:, txt_seq_len:, :, :]
        else:
            image_sdpa_output = sdpa_outputs[:, :img_seq_len, :, :]
            text_sdpa_output = sdpa_outputs[:, -txt_seq_len:, :, :]

        # Post-SDPA layers
        latent_image_embeddings = self.image_transformer_block.post_sdpa(
            residual=latent_image_embeddings,
            sdpa_output=image_sdpa_output,
            **image_intermediates,
        )
        if self.text_transformer_block.skip_post_sdpa:
            # Text token related outputs from the final layer do not impact the model output
            token_level_text_embeddings = None
        else:
            token_level_text_embeddings = self.text_transformer_block.post_sdpa(
                residual=token_level_text_embeddings,
                sdpa_output=text_sdpa_output,
                **text_intermediates,
            )

        return latent_image_embeddings, token_level_text_embeddings


class UnifiedTransformerBlock(nn.Module):
    def __init__(self, config: MMDiTConfig):
        super().__init__()
        self.transformer_block = TransformerBlock(
            config,
            num_modulation_params=3 if config.parallel_mlp_for_unified_blocks else 6,
            parallel_mlp=config.parallel_mlp_for_unified_blocks,
        )

        sdpa_impl = mx.fast.scaled_dot_product_attention
        self.sdpa = partial(sdpa_impl)

        self.config = config
        self.per_head_dim = config.hidden_size // config.num_heads

    def __call__(
        self,
        latent_unified_embeddings: mx.array,  # latent image embeddings
        timestep: mx.array,  # pooled text embeddings + timestep embeddings
        positional_encodings: mx.array = None,  # positional encodings for rope
    ):
        # Prepare multi-modal SDPA inputs
        intermediates = self.transformer_block.pre_sdpa(
            latent_unified_embeddings,
            timestep=timestep,
        )

        batch = latent_unified_embeddings.shape[0]

        def rearrange_for_sdpa(t):
            # Target data layout: (batch, head, seq_len, channel)
            return t.reshape(
                batch, -1, self.config.num_heads, self.per_head_dim
            ).transpose(0, 2, 1, 3)

        multimodal_sdpa_inputs = {
            "q": intermediates["q"],
            "k": intermediates["k"],
            "v": intermediates["v"],
            "scale": 1.0 / np.sqrt(self.per_head_dim),
        }

        if self.config.pos_embed_type == PositionalEncoding.PreSDPARope:
            assert positional_encodings is not None
            multimodal_sdpa_inputs["q"] = RoPE.apply(
                multimodal_sdpa_inputs["q"], positional_encodings
            )
            multimodal_sdpa_inputs["k"] = RoPE.apply(
                multimodal_sdpa_inputs["k"], positional_encodings
            )

        # Compute multi-modal SDPA
        sdpa_outputs = (
            self.sdpa(**multimodal_sdpa_inputs)
            .transpose(0, 2, 1, 3)
            .reshape(batch, -1, 1, self.config.hidden_size)
        )

        # o_proj and mlp.fc2 uses the same bias, remove mlp.fc2 bias
        self.transformer_block.mlp.fc2.bias = self.transformer_block.mlp.fc2.bias * 0.0

        # Post-SDPA layers
        latent_unified_embeddings = self.transformer_block.post_sdpa(
            residual=latent_unified_embeddings,
            sdpa_output=sdpa_outputs,
            **intermediates,
        )

        return latent_unified_embeddings


class QKNorm(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def __call__(self, q: mx.array, k: mx.array) -> Tuple[mx.array, mx.array]:
        # Note: mlx.nn.RMSNorm has high precision accumulation (does not require upcasting)
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k


class FinalLayer(nn.Module):
    def __init__(self, config: MMDiTConfig):
        super().__init__()
        self.norm_final = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.linear = nn.Linear(
            config.hidden_size,
            (config.patch_size**2) * config.vae_latent_dim,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 2 * config.hidden_size),
        )

    def __call__(
        self,
        latent_image_embeddings: mx.array,
        timestep: mx.array,
    ) -> mx.array:
        if timestep.size > 1:
            timestep = timestep[0]
        modulation_params = self._modulation_params[timestep.item()]

        shift, residual_scale = mx.split(modulation_params, 2, axis=-1)
        latent_image_embeddings = affine_transform(
            latent_image_embeddings,
            shift=shift,
            residual_scale=residual_scale,
            norm_module=self.norm_final,
        )
        return self.linear(latent_image_embeddings)


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
    ):
        super().__init__()

        # Configure dimensions for SDPA
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert (
            self.embed_dim % self.n_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        self._sdpa_implementation = mx.fast.scaled_dot_product_attention

        # Initialize layers
        self.per_head_dim = self.embed_dim // self.n_heads
        self.kv_proj_embed_dim = self.per_head_dim * n_heads

        # Note: key bias is redundant due to softmax invariance
        self.k_proj = nn.Linear(embed_dim, self.kv_proj_embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, self.kv_proj_embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)


class FFN(nn.Module):
    def __init__(self, embed_dim, expansion_factor, activation_fn):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * expansion_factor)
        self.act_fn = activation_fn
        self.fc2 = nn.Linear(embed_dim * expansion_factor, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act_fn(self.fc1(x)))


class LayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

    def __call__(self, inputs: mx.array) -> mx.array:
        input_rank = len(inputs.shape)
        if input_rank != 4:
            raise ValueError(f"Input tensor must have rank 4, got {input_rank}")

        return mx.fast.layer_norm(inputs, weight=None, bias=None, eps=self.eps)


class RoPE(nn.Module):
    """Custom RoPE implementation for FLUX"""

    def __init__(self, theta: int, axes_dim: List[int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

        # Cache for consecutive identical calls
        self.rope_embeddings = None
        self.last_image_resolution = None
        self.last_text_sequence_length = None

    def _get_positions(
        self, latent_image_resolution: Tuple[int], text_sequence_length: int
    ) -> mx.array:
        h, w = latent_image_resolution
        image_positions = mx.stack(
            [
                mx.zeros((h, w)),
                mx.repeat(mx.arange(h)[:, None], w, axis=1),
                mx.repeat(mx.arange(w)[None, :], h, axis=0),
            ],
            axis=-1,
        ).flatten(
            0, 1
        )  # (h * w, 3)

        text_and_image_positions = mx.concatenate(
            [
                mx.zeros((text_sequence_length, 3)),
                image_positions,
            ],
            axis=0,
        )[
            None
        ]  # (text_sequence_length + h * w, 3)

        return text_and_image_positions

    def rope(self, positions: mx.array, dim: int, theta: int = 10_000) -> mx.array:
        def _rope_per_dim(positions, dim, theta):
            scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
            omega = 1.0 / (theta**scale)
            out = (
                positions[..., None] * omega[None, None, :]
            )  # mx.einsum("bn,d->bnd", positions, omega)
            return mx.stack(
                [mx.cos(out), -mx.sin(out), mx.sin(out), mx.cos(out)], axis=-1
            ).reshape(*positions.shape, dim // 2, 2, 2)

        return mx.concatenate(
            [
                _rope_per_dim(
                    positions=positions[..., i], dim=self.axes_dim[i], theta=self.theta
                )
                for i in range(len(self.axes_dim))
            ],
            axis=-3,
        ).astype(positions.dtype)

    def __call__(
        self, latent_image_resolution: Tuple[int], text_sequence_length: int
    ) -> mx.array:
        identical_to_last_call = (
            latent_image_resolution == self.last_image_resolution
            and text_sequence_length == self.last_text_sequence_length
        )

        if self.rope_embeddings is None or not identical_to_last_call:
            self.last_image_resolution = latent_image_resolution
            self.last_text_sequence_length = text_sequence_length
            positions = self._get_positions(
                latent_image_resolution, text_sequence_length
            )
            self.rope_embeddings = self.rope(positions, self.theta)
            self.rope_embeddings = mx.expand_dims(self.rope_embeddings, axis=1)
        else:
            logger.debug("Returning cached RoPE embeddings")

        return self.rope_embeddings

    @staticmethod
    def apply(q_or_k: mx.array, rope: mx.array) -> mx.array:
        in_dtype = q_or_k.dtype
        q_or_k = q_or_k.astype(mx.float32).reshape(*q_or_k.shape[:-1], -1, 1, 2)
        return (
            (rope[..., 0] * q_or_k[..., 0] + rope[..., 1] * q_or_k[..., 1])
            .astype(in_dtype)
            .flatten(-2)
        )


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def __call__(self, x):
        return self.mlp(x)


def affine_transform(
    x: mx.array,
    shift: mx.array,
    residual_scale: mx.array,
    norm_module: nn.Module = None,
) -> mx.array:
    """Affine transformation (Used for Adaptive LayerNorm Modulation)"""
    if x.shape[0] == 1 and norm_module is not None:
        return mx.fast.layer_norm(
            x, 1.0 + residual_scale.squeeze(), shift.squeeze(), norm_module.eps
        )
    elif norm_module is not None:
        return norm_module(x) * (1.0 + residual_scale) + shift
    else:
        return x * (1.0 + residual_scale) + shift


def unpatchify(
    x: mx.array,
    patch_size: int,
    target_height: int,
    target_width: int,
    vae_latent_dim: int,
) -> mx.array:
    """Unpatchify to restore VAE latent space compatible data format"""
    h, w = target_height // patch_size, target_width // patch_size
    x = x.reshape(x.shape[0], h, w, patch_size, patch_size, vae_latent_dim)
    x = x.transpose(0, 5, 1, 3, 2, 4)  # x = mx.einsum("bhwpqc->bchpwq", x)
    return x.reshape(x.shape[0], vae_latent_dim, target_height, target_width).transpose(
        0, 2, 3, 1
    )

# fmt: on
