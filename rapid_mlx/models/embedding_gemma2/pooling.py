# SPDX-License-Identifier: MIT
# Copyright © 2025 Prince Canuma
# Vendored from MLX-VLM 3d87e88402f307efbf68e568971aa887ee7d9ed0; see NOTICE.

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class EmbeddingOutput:
    last_hidden_state: mx.array | None = None
    text_embeds: mx.array | None = None


def normalize_embeddings(embeddings, p=2, axis=-1, keepdims=True, eps=1e-9):
    return embeddings / mx.maximum(
        mx.linalg.norm(embeddings, ord=p, axis=axis, keepdims=keepdims), eps
    )


def mean_pooling(token_embeddings: mx.array, attention_mask: mx.array):
    input_mask_expanded = mx.expand_dims(attention_mask, -1)
    input_mask_expanded = mx.broadcast_to(
        input_mask_expanded, token_embeddings.shape
    ).astype(mx.float32)
    sum_embeddings = mx.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = mx.maximum(mx.sum(input_mask_expanded, axis=1), 1e-9)
    return sum_embeddings / sum_mask
