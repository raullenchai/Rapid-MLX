# SPDX-License-Identifier: MIT
# Copyright © 2025 Prince Canuma
# Vendored from mlx-vlm @ b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0; see LICENSE and provenance.json.
# VENDOR-DEVIATION(subset): copy only VisionModel to accept the 4.7 tag.
# The SigLIP layers and shared shape check stay on pinned 0.7.2.
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import check_array_shape
from mlx_vlm.models.minicpmv4_6.config import VisionConfig
from mlx_vlm.models.minicpmv4_6.vision import SiglipVisionEmbeddings, SiglipEncoder


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        if self.model_type not in [
            "siglip_vision_model",
            "siglip",
            "minicpmv4_6_vision",
            "minicpmv4_7_vision",
        ]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(
        self,
        pixel_values: mx.array,
        patch_attention_mask: Optional[mx.array] = None,
        tgt_sizes: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.embeddings(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
            tgt_sizes=tgt_sizes,
        )
        hidden_states = hidden_states.astype(
            self.embeddings.patch_embedding.weight.dtype
        )
        hidden_states = self.encoder(hidden_states=hidden_states, attention_mask=None)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states

    def sanitize(self, weights):
        sanitized_weights = {}
        for key, value in weights.items():
            if "position_ids" in key:
                continue
            if key.endswith("patch_embedding.weight"):
                if not check_array_shape(value):
                    value = value.transpose(0, 2, 3, 1)
            sanitized_weights[key] = value
        return sanitized_weights
