# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Apple Inc.

"""Vendored NVIDIA GPT-OSS Puzzle model from ml-explore/mlx-lm#1488.

Puzzle is a NAS-pruned GPT-OSS variant: each layer selects its own MoE expert
count and attention window from ``block_configs``.  The attention, mxfp4 MoE
layout, Harmony tokenizer, and standard generation path are otherwise the
same as GPT-OSS, so this deliberately reuses the installed ``mlx_lm`` GPT-OSS
blocks rather than maintaining a fork.

Upstream: https://github.com/ml-explore/mlx-lm/pull/1488
Vendored from: ml-explore/mlx-lm@90ce9c4b8825cf8a4e2fd4ce589574695ad9b6a4

Only package-relative imports are adapted to Rapid-MLX's installed
``mlx-lm>=0.31.3``.  Registration lives in
``vllm_mlx.utils.tokenizer._register_vendored_archs`` and automatically defers
to a future native upstream module.
"""

import copy
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

# Install before importing mlx_lm; see the existing vendored HY-V3 and
# DeepSeek-V4 modules for the M5 thread-local-stream compatibility rationale.
from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.base import BaseModelArgs, create_attention_mask  # noqa: E402
from mlx_lm.models.cache import KVCache, RotatingKVCache  # noqa: E402
from mlx_lm.models.gpt_oss import Model as GptOssModel  # noqa: E402
from mlx_lm.models.gpt_oss import TransformerBlock  # noqa: E402


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gpt_oss_puzzle"
    num_hidden_layers: int = 36
    num_experts_per_tok: int = 4
    vocab_size: int = 201088
    rms_norm_eps: float = 1e-05
    hidden_size: int = 2880
    intermediate_size: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    rope_theta: int = 150000
    rope_scaling: Any = None
    # Kept outside rope_scaling because Transformers 5.x cannot standardize
    # Puzzle's YaRN field during tokenizer loading.  Restore it before layers
    # initialize their RoPE modules.
    yarn_rope_scaling: Any = None
    # Per-layer [{num_local_experts, sliding_window}, ...]; None means full
    # attention. Uniform fields are retained as config fallbacks.
    block_configs: list[dict] | None = None
    num_local_experts: int = 128
    sliding_window: int = 128


def _layer_config(args: ModelArgs, num_local_experts: int) -> ModelArgs:
    """Return a shallow config view with the layer's MoE width."""
    config = copy.copy(args)
    config.num_local_experts = num_local_experts
    return config


class PuzzleModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        if not args.block_configs:
            raise ValueError("gpt_oss_puzzle requires 'block_configs' in config")
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.windows = [config.get("sliding_window") for config in args.block_configs]
        self.layers = [
            TransformerBlock(_layer_config(args, config["num_local_experts"]))
            for config in args.block_configs
        ]
        # One causal mask can serve all layers with the same window.
        self._mask_ref: dict[int | None, int] = {}
        for index, window in enumerate(self.windows):
            self._mask_ref.setdefault(window, index)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ):
        x = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )

        if cache is None:
            cache = [None] * len(self.layers)

        masks = {}
        for window, reference in self._mask_ref.items():
            if window is None:
                masks[window] = create_attention_mask(x, cache[reference])
            else:
                masks[window] = create_attention_mask(
                    x, cache[reference], window_size=window
                )

        for layer, layer_cache, window in zip(self.layers, cache, self.windows):
            x = layer(x, masks[window], layer_cache)
        return self.norm(x)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        if args.rope_scaling is None and args.yarn_rope_scaling is not None:
            args.rope_scaling = args.yarn_rope_scaling
        self.args = args
        self.model_type = args.model_type
        self.model = PuzzleModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        return self.lm_head(self.model(inputs, cache))

    def sanitize(self, weights):
        # These fp8-KV calibration scales are unused by the bf16 KV path (and
        # ignored by NVIDIA's own implementation). The remaining mxfp4 expert
        # layout is identical to GPT-OSS, so delegate its sanitizer.
        weights = {
            key: value
            for key, value in weights.items()
            if not (key.endswith(".k_scale") or key.endswith(".v_scale"))
        }
        return GptOssModel.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            KVCache() if window is None else RotatingKVCache(max_size=window)
            for window in self.model.windows
        ]
