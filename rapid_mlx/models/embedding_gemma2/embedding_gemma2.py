# SPDX-License-Identifier: MIT
# Copyright © 2025 Prince Canuma
# Text-only subset of MLX-VLM 3d87e88402f307efbf68e568971aa887ee7d9ed0; see NOTICE.

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .language import TextModel
from .pooling import EmbeddingOutput, mean_pooling, normalize_embeddings


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = TextModel(config.text_config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        **kwargs,
    ) -> EmbeddingOutput:
        if any(
            kwargs.get(name) is not None
            for name in ("pixel_values", "pixel_values_videos", "input_features")
        ):
            raise ValueError("EmbeddingGemma 2 supports text/code only")
        if attention_mask is None:
            attention_mask = kwargs.get("mask")
        if attention_mask is None:
            attention_mask = mx.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        embeddings = self.language_model.embed_tokens(input_ids)
        embeddings = embeddings * mx.array(
            self.config.text_config.hidden_size**0.5, dtype=embeddings.dtype
        )
        hidden_states = self.language_model(embeddings, attention_mask, position_ids)
        pooled = mean_pooling(hidden_states, attention_mask)
        return EmbeddingOutput(
            last_hidden_state=hidden_states,
            text_embeds=normalize_embeddings(pooled),
        )

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            key = key.removeprefix("model.")
            # Full official checkpoints contain media towers which this adapter
            # neither constructs nor serves. Unknown text weights stay strict.
            if (
                key.startswith(
                    ("vision_tower.", "embed_vision.", "audio_tower.", "embed_audio.")
                )
                or "rotary_emb.inv_freq" in key
            ):
                continue
            sanitized[key] = value
        return sanitized

    @property
    def layers(self):
        return self.language_model.layers
