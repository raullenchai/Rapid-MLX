# SPDX-License-Identifier: MIT
# Copyright © 2025 Prince Canuma
# Vendored from mlx-vlm @ b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0; see LICENSE and provenance.json.
# VENDOR-DEVIATION(subset): only the two vision embedding methods are copied
# from minicpmv4_6.Model. Keep the rest of the model on pinned 0.7.2.
from typing import Optional

import mlx.core as mx
import numpy as np
from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.models.minicpmv4_6.minicpmv4_6 import (
    Model as PinnedModel,
    _to_mx_array,
    _to_numpy,
)


class Model(PinnedModel):
    def get_vision_embedding(self, pixel_values, tgt_sizes, downsample_mode=None):
        dtype = self.vision_tower.embeddings.patch_embedding.weight.dtype

        if pixel_values is None:
            return []

        batch_size = len(pixel_values)
        vision_hidden_states = []

        for batch_idx in range(batch_size):
            batch_pixels = (
                pixel_values[batch_idx] if batch_idx < len(pixel_values) else []
            )
            batch_tgt = tgt_sizes[batch_idx] if tgt_sizes is not None else []
            batch_tgt = _to_numpy(batch_tgt)
            if batch_tgt is None or len(batch_tgt) == 0:
                batch_tgt = np.zeros((0, 2), dtype=np.int32)
            else:
                batch_tgt = np.asarray(batch_tgt, dtype=np.int32)

            sample_embeddings = []
            for image_idx, cur_pixels in enumerate(batch_pixels):
                cur_pixels = _to_mx_array(cur_pixels, dtype=dtype)
                if cur_pixels is None:
                    continue
                if cur_pixels.ndim != 3:
                    continue

                # The vision tower convolution path expects HWC inputs.
                # Convert both CHW and patch-packed (C, patch, n * patch) layouts to HWC here.
                if cur_pixels.shape[0] == 3:
                    cur_pixels = cur_pixels.transpose(1, 2, 0)
                cur_pixels = mx.expand_dims(cur_pixels, axis=0)

                if image_idx < len(batch_tgt):
                    cur_tgt = batch_tgt[image_idx]
                else:
                    seq_len = max(int(cur_pixels.shape[2] // self.config.patch_size), 1)
                    cur_tgt = np.array([1, seq_len], dtype=np.int32)

                cur_tgt = mx.array(cur_tgt, dtype=mx.int32)[None, :]
                hidden = self.vision_tower.embeddings(
                    cur_pixels,
                    tgt_sizes=cur_tgt,
                )
                hidden = hidden.astype(
                    self.vision_tower.embeddings.patch_embedding.weight.dtype
                )
                grid_h = int(cur_tgt[0, 0].item())
                grid_w = int(cur_tgt[0, 1].item())

                insert_layer_id = self.config.insert_layer_id
                use_vit_merger = (
                    downsample_mode or self.config.downsample_mode
                ) != "4x"
                for layer_index, encoder_layer in enumerate(
                    self.vision_tower.encoder.layers
                ):
                    hidden = encoder_layer(hidden, attention_mask=None)
                    if use_vit_merger and layer_index == insert_layer_id:
                        merged_hidden, grid_h, grid_w = self.vit_merger(
                            hidden[0], grid_h, grid_w
                        )
                        hidden = mx.expand_dims(merged_hidden, axis=0)

                hidden = self.vision_tower.post_layernorm(hidden)
                hidden = hidden[0]

                merged_tokens, final_h, final_w = self.merger(hidden, grid_h, grid_w)
                merged_tokens = merged_tokens.astype(dtype)
                sample_embeddings.append(merged_tokens)

            if len(sample_embeddings) > 0:
                vision_hidden_states.append(mx.concatenate(sample_embeddings, axis=0))
            else:
                vision_hidden_states.append([])

        return vision_hidden_states

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[list] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        tgt_sizes = kwargs.get("tgt_sizes", None)
        image_bound = kwargs.get("image_bound", None)
        self._set_position_state(input_ids)
        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            vision_hidden_states = cached
        elif pixel_values is not None:
            vision_hidden_states = self.get_vision_embedding(
                pixel_values, tgt_sizes, downsample_mode=kwargs.get("downsample_mode")
            )
        else:
            vision_hidden_states = None

        updated = []
        for batch_idx in range(inputs_embeds.shape[0]):
            cur_embeds = inputs_embeds[batch_idx]

            # Vision embeddings replacement.
            if vision_hidden_states is not None and image_bound is not None:
                cur_vs_hs = vision_hidden_states[batch_idx]
                if isinstance(cur_vs_hs, mx.array) and cur_vs_hs.size > 0:
                    cur_bounds = (
                        image_bound[batch_idx]
                        if isinstance(image_bound, list)
                        else image_bound[batch_idx]
                    )
                    cur_bounds = _to_numpy(cur_bounds)
                    if cur_bounds is not None and cur_bounds.size > 0:
                        cur_bounds = np.asarray(cur_bounds, dtype=np.int32).reshape(
                            -1, 2
                        )
                        indices = []
                        for start, end in cur_bounds:
                            if end > start:
                                indices.append(np.arange(start, end, dtype=np.int32))

                        if len(indices) > 0:
                            indices = np.concatenate(indices, axis=0)
                            features = cur_vs_hs.reshape(
                                -1, cur_vs_hs.shape[-1]
                            ).astype(cur_embeds.dtype)
                            if int(features.shape[0]) != int(len(indices)):
                                raise ValueError(
                                    "MiniCPM-V vision feature token count does not match "
                                    f"image placeholder span length for sample {batch_idx}: "
                                    f"features={features.shape[0]} placeholders={len(indices)}."
                                )
                            indices_mx = mx.array(indices, dtype=mx.uint32)
                            cur_embeds = self._apply_vision_delta(
                                cur_embeds, indices_mx, features
                            )

            updated.append(cur_embeds)

        inputs_embeds = mx.stack(updated, axis=0)
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

