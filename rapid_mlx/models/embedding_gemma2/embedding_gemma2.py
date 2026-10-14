# SPDX-License-Identifier: MIT
# Copyright © 2025 Prince Canuma
# Adapted subset of MLX-VLM 3d87e88402f307efbf68e568971aa887ee7d9ed0; see NOTICE.

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .language import TextModel
from .pooling import EmbeddingOutput, mean_pooling, normalize_embeddings


class Model(nn.Module):
    def __init__(self, config: ModelConfig, *, vision: bool = False):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = TextModel(config.text_config)
        self.vision_enabled = vision
        if vision:
            if config.vision_config is None:
                raise ValueError("Checkpoint does not support images")
            from mlx_vlm.models.gemma4.config import VisionConfig
            from mlx_vlm.models.gemma4.gemma4 import MultimodalEmbedder
            from mlx_vlm.models.gemma4.vision import VisionModel

            vision_config = VisionConfig.from_dict(config.vision_config)
            self.vision_tower = VisionModel(vision_config)
            self.embed_vision = MultimodalEmbedder(
                vision_config.hidden_size,
                config.text_config.hidden_size,
                vision_config.rms_norm_eps,
            )

    @staticmethod
    def _scatter(embeddings, input_ids, token_id, features):
        # Adapted from the validated upstream encoder, preserving sample order.
        mask = input_ids == token_id
        count = int(mask.sum().item())
        if count != features.shape[0]:
            raise ValueError("Image token count does not match feature count")
        if count == 0:
            return embeddings
        indices = mx.maximum(mx.cumsum(mask.reshape(-1)) - 1, 0)
        aligned = features[indices].reshape(embeddings.shape)
        return mx.where(mask[..., None], aligned.astype(embeddings.dtype), embeddings)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        **kwargs,
    ) -> EmbeddingOutput:
        if any(
            kwargs.get(name) is not None
            for name in (
                ("pixel_values_videos", "input_features")
                if self.vision_enabled
                else ("pixel_values", "pixel_values_videos", "input_features")
            )
        ):
            raise ValueError("EmbeddingGemma 2 supports text/code only")
        if attention_mask is None:
            attention_mask = kwargs.get("mask")
        if attention_mask is None:
            attention_mask = mx.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        pixel_values = kwargs.get("pixel_values")
        text_ids = input_ids
        if pixel_values is not None:
            text_ids = mx.where(
                input_ids == self.config.image_token_id,
                self.config.text_config.pad_token_id,
                input_ids,
            )
        embeddings = self.language_model.embed_tokens(text_ids)
        embeddings = embeddings * mx.array(
            self.config.text_config.hidden_size**0.5, dtype=embeddings.dtype
        )
        if pixel_values is not None:
            features = self.embed_vision(
                self.vision_tower(pixel_values, kwargs.get("image_position_ids"))
            )[0]
            embeddings = self._scatter(
                embeddings, input_ids, self.config.image_token_id, features
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
            # Audio is always excluded. Vision is opt-in; unknown keys stay strict.
            if (
                key.startswith(("audio_tower.", "embed_audio."))
                or (
                    not self.vision_enabled
                    and key.startswith(("vision_tower.", "embed_vision."))
                )
                or "rotary_emb.inv_freq" in key
            ):
                continue
            sanitized[key] = value
        return sanitized

    @property
    def layers(self):
        return self.language_model.layers
