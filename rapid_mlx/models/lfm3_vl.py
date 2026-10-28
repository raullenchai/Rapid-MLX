# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3/LFM3-VL checkpoint reference implementations.
# Licensed under the Apache License, Version 2.0; see the repository LICENSE.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import lfm3

TextConfig = lfm3.ModelArgs


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


from mlx_lm.models.base import BaseModelArgs


@dataclass
class VisionConfig(BaseModelArgs):
    model_type: str = "lfm3_vl_vision"
    hidden_size: int = 1024
    intermediate_size: int = 2560
    num_hidden_layers: int = 14
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    conv_kernel_size: int = 3
    norm_eps: float = 1e-5
    layer_types: list[str] | None = None
    rope_parameters: dict | None = None

    def __post_init__(self):
        if (
            self.layer_types is None
            or len(self.layer_types) != self.num_hidden_layers
            or not set(self.layer_types) <= {"conv", "full_attention"}
        ):
            raise ValueError("Invalid vision layer types")
        if (
            self.hidden_size % self.num_attention_heads
            or self.hidden_size // self.num_attention_heads % 4
        ):
            raise ValueError("Vision head dimension must be divisible by four")
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "axial", "rope_theta": 1e6}
        if self.rope_parameters.get("rope_type", "axial") != "axial":
            raise ValueError("Vision requires axial rotary embeddings")


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: dict | TextConfig
    vision_config: dict | VisionConfig
    model_type: str = "lfm3_vl"
    downsample_factor: int = 2
    projector_hidden_size: int = 4096
    projector_hidden_act: str = "gelu"
    projector_bias: bool = True
    projector_layer_norm_eps: float = 1e-6
    image_token_id: int = 125030
    video_token_id: int = 125029

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig.from_dict(self.vision_config)
        if (
            self.downsample_factor != self.vision_config.spatial_merge_size
            or self.projector_hidden_act != "gelu"
        ):
            raise ValueError("Unsupported projector configuration")


def grid_positions(height, width, merge):
    coords = np.stack(
        np.meshgrid(np.arange(height), np.arange(width), indexing="ij"), axis=-1
    )
    return mx.array(
        coords.reshape(height // merge, merge, width // merge, merge, 2)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, 2)
    )


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)

    def __call__(self, x, cos, sin):
        n = x.shape[0]
        q = self.q_layernorm(self.q_proj(x).reshape(n, self.n_heads, self.head_dim))
        k = self.k_layernorm(self.k_proj(x).reshape(n, self.n_kv_heads, self.head_dim))
        v = self.v_proj(x).reshape(n, self.n_kv_heads, self.head_dim)

        def rotate(t):
            dtype = t.dtype
            t = t.astype(mx.float32)
            a, b = mx.split(t, 2, axis=-1)
            return (
                t * cos[:, None] + mx.concatenate([-b, a], axis=-1) * sin[:, None]
            ).astype(dtype)

        q, k = rotate(q), rotate(k)
        out = mx.fast.scaled_dot_product_attention(
            q.transpose(1, 0, 2)[None],
            k.transpose(1, 0, 2)[None],
            v.transpose(1, 0, 2)[None],
            scale=self.scale,
        )
        return self.out_proj(out[0].transpose(1, 0, 2).reshape(n, -1))


class VisionConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.merge = config.spatial_merge_size
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.spatial_conv = nn.Conv2d(
            config.hidden_size,
            config.hidden_size,
            config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=config.hidden_size,
            bias=False,
        )
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def __call__(self, x, height, width):
        b, c, v = mx.split(self.in_proj(x), 3, axis=-1)
        m = self.merge
        grid = (
            (b * v)
            .reshape(height // m, width // m, m, m, -1)
            .transpose(0, 2, 1, 3, 4)
            .reshape(1, height, width, -1)
        )
        grid = self.spatial_conv(grid)
        out = (
            grid.reshape(height // m, m, width // m, m, -1)
            .transpose(0, 2, 1, 3, 4)
            .reshape(height * width, -1)
        )
        return self.out_proj(c * out)


class VisionLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.is_attention_layer = config.layer_types[index] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = VisionAttention(config)
        else:
            self.conv = VisionConv(config)
        self.feed_forward = MLP(config)
        self.operator_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(self, x, height, width, cos, sin):
        h = self.operator_norm(x)
        x = x + (
            self.self_attn(h, cos, sin)
            if self.is_attention_layer
            else self.conv(h, height, width)
        )
        return x + self.feed_forward(self.ffn_norm(x))


class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.patchifier = nn.Linear(
            config.in_channels * config.temporal_patch_size * config.patch_size**2,
            config.hidden_size,
            bias=False,
        )
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.layers = [VisionLayer(config, i) for i in range(config.num_hidden_layers)]

    def __call__(self, pixel_values, grid_thw):
        frames = []
        offset = 0
        config = self.config
        spatial_dim = config.hidden_size // config.num_attention_heads // 2
        inv_freq = config.rope_parameters["rope_theta"] ** (
            -mx.arange(0, spatial_dim, 2, dtype=mx.float32) / spatial_dim
        )
        for temporal, height, width in np.asarray(grid_thw).tolist():
            coords = grid_positions(height, width, config.spatial_merge_size)
            angles = (coords[..., None] * inv_freq).reshape(height * width, -1)
            angles = mx.concatenate([angles, angles], axis=-1)
            cos, sin = mx.cos(angles), mx.sin(angles)
            for _ in range(temporal):
                x = self.embedding_norm(
                    self.patchifier(
                        pixel_values[offset : offset + height * width].astype(
                            self.embedding_norm.weight.dtype
                        )
                    )
                )
                for layer in self.layers:
                    x = layer(x, height, width, cos, sin)
                frames.append(x)
                offset += height * width
        if offset != pixel_values.shape[0]:
            raise ValueError("Pixel patch count does not match grid_thw")
        return mx.concatenate(frames, axis=0)


class MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.vision_config.hidden_size * config.downsample_factor**2
        self.layer_norm = nn.LayerNorm(
            config.vision_config.hidden_size, eps=config.projector_layer_norm_eps
        )
        self.linear_1 = nn.Linear(
            self.input_dim, config.projector_hidden_size, bias=config.projector_bias
        )
        self.linear_2 = nn.Linear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )

    def __call__(self, x):
        return self.linear_2(
            nn.gelu(self.linear_1(self.layer_norm(x).reshape(-1, self.input_dim)))
        )


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        from . import lfm3

        self.config = config
        self.model_type = "lfm3_vl"
        self.vision_tower = VisionModel(config.vision_config)
        self.multi_modal_projector = MultiModalProjector(config)
        self.language_model = lfm3.Model(config.text_config)

    def encode_images(self, pixel_values, image_grid_thw):
        return self.multi_modal_projector(
            self.vision_tower(pixel_values, image_grid_thw)
        )

    def get_input_embeddings(self, input_ids, pixel_values=None, image_grid_thw=None):
        ids = np.asarray(input_ids)
        if np.any(ids == self.config.video_token_id):
            raise ValueError("Video is not supported by LFM3-VL")
        embeddings = self.language_model.model.embed_tokens(input_ids)
        indices = np.flatnonzero(ids.reshape(-1) == self.config.image_token_id)
        if pixel_values is None:
            if len(indices):
                raise ValueError("Image tokens require image pixels")
            return embeddings
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required")
        grid = np.asarray(image_grid_thw)
        if ids.ndim != 2 or ids.shape[0] != 1 or grid.ndim != 2 or grid.shape[1] != 3:
            raise ValueError("Expected one prompt and an ordered image grid")
        mask = ids.reshape(-1) == self.config.image_token_id
        boundaries = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
        lengths = np.flatnonzero(boundaries == -1) - np.flatnonzero(boundaries == 1)
        merge = self.config.downsample_factor
        expected = []
        for temporal, height, width in grid:
            if (
                temporal != 1
                or height <= 0
                or width <= 0
                or height % merge
                or width % merge
            ):
                raise ValueError("Invalid image grid")
            expected.append(int(height * width // merge**2))
        if lengths.tolist() != expected:
            raise ValueError("Image token spans do not match ordered image grids")
        features = self.encode_images(pixel_values, image_grid_thw)
        if len(indices) != features.shape[0]:
            raise ValueError("Image token count does not match image features")
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        flat[mx.array(indices)] = features.astype(flat.dtype)
        return flat.reshape(embeddings.shape)

    def __call__(self, input_ids, pixel_values=None, cache=None, **kwargs):
        embeddings = self.get_input_embeddings(
            input_ids, pixel_values, kwargs.get("image_grid_thw")
        )
        return self.language_model(
            input_ids, cache=cache, inputs_embeds=embeddings, **kwargs
        )

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def sanitize(self, weights):
        result = {}
        text = {}
        for name, value in weights.items():
            if name.startswith("language_model."):
                text[name] = value
            elif name.startswith(("vision_tower.", "multi_modal_projector.")):
                if name.endswith("spatial_conv.weight") and value.shape[1] == 1:
                    value = value.transpose(0, 2, 3, 1)
                result[name] = value
            else:
                raise ValueError(f"Unexpected weight: {name}")
        result.update(
            {
                "language_model." + name: value
                for name, value in self.language_model.sanitize(text).items()
            }
        )
        from mlx.utils import tree_flatten

        expected = dict(tree_flatten(self.parameters()))
        extra = set(result) - set(expected)
        for name in list(extra):
            if name.endswith((".scales", ".biases")):
                base = name.rsplit(".", 1)[0] + ".weight"
                if (
                    base in expected
                    and base.startswith("language_model.")
                    and expected[base].ndim == 2
                ):
                    extra.remove(name)
        for name in result:
            if name.endswith((".scales", ".biases")):
                prefix = name.rsplit(".", 1)[0]
                if prefix + ".scales" not in result or prefix + ".biases" not in result:
                    raise ValueError(f"Incomplete quantization metadata: {prefix}")
        missing = set(expected) - set(result)
        if missing or extra:
            raise ValueError(
                f"Invalid weights: missing={sorted(missing)}, unexpected={sorted(extra)}"
            )
        for name, target in expected.items():
            value = result[name]
            scales = name.removesuffix(".weight") + ".scales"
            biases = name.removesuffix(".weight") + ".biases"
            if name.endswith(".weight") and scales in result:
                if biases not in result or target.ndim != 2 or value.dtype != mx.uint32:
                    raise ValueError(f"Invalid quantized weight: {name}")
                out_dim, in_dim = target.shape
                if result[scales].dtype not in (
                    mx.float16,
                    mx.bfloat16,
                    mx.float32,
                ) or result[biases].dtype not in (mx.float16, mx.bfloat16, mx.float32):
                    raise ValueError(f"Invalid quantization metadata dtype: {name}")
                if (
                    value.shape != (out_dim, in_dim // 4)
                    or result[scales].shape != (out_dim, in_dim // 64)
                    or result[biases].shape != result[scales].shape
                ):
                    raise ValueError(f"Invalid quantized shape: {name}")
            elif value.shape != target.shape:
                raise ValueError(
                    f"Invalid shape for {name}: {value.shape}, expected {target.shape}"
                )
            elif value.dtype not in (mx.float16, mx.bfloat16, mx.float32):
                raise ValueError(f"Invalid dtype for {name}: {value.dtype}")
        return result
