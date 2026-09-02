# SPDX-License-Identifier: Apache-2.0
"""Native-first Apertus 1.5 text backbone compatibility vendor.

The official Apertus 1.5 multimodal checkpoint exposes a text decoder under
``model.language_model``.  Its input embedding covers media code tokens while
the output head contains only the generatable text vocabulary.  This wrapper
reuses mlx-lm's native ``apertus`` trunk and adapts only those boundaries.

Registered by :mod:`vllm_mlx.utils.tokenizer` only when the installed mlx-lm
lacks native ``apertus1p5_text`` support.  Derived from mlx-lm#1615,
9028108c2ac9ef54a0bb945d8af81f1fee7c2a8f.
"""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.apertus import ApertusModel  # noqa: E402
from mlx_lm.models.base import BaseModelArgs  # noqa: E402


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    post_norm: bool
    qk_norm: bool
    mlp_bias: bool = False
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    output_vocab_size: int | None = None
    rope_parameters: dict[str, Any] | None = None
    rope_theta: float = 4_000_000.0
    rope_traditional: bool = False
    rope_scaling: dict[str, float | str] | None = None

    @classmethod
    def from_dict(cls, params):
        # The official checkpoint's config.json is the multimodal wrapper:
        # ``model_type: apertus1p5`` with the decoder nested under
        # ``text_config``. mlx-lm hands that outer dict straight to the
        # module resolved for the outer model_type, so flatten here — the
        # nested text config wins over wrapper-level duplicates.
        text_config = params.get("text_config")
        if isinstance(text_config, dict):
            merged = {
                key: value
                for key, value in params.items()
                if not isinstance(value, dict)
            }
            merged.update(text_config)
            params = merged
        return super().from_dict(params)

    def __post_init__(self):
        if self.rope_parameters is None:
            return
        self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
        scaling = {
            key: value
            for key, value in self.rope_parameters.items()
            if key != "rope_theta"
        }
        scaling.setdefault("rope_type", scaling.pop("type", "llama3"))
        self.rope_scaling = scaling


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ApertusModel(args)
        head_vocab_size = args.output_vocab_size or args.vocab_size
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, head_vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    # The multimodal wrapper's non-text trunks. Their shards are globbed by
    # mlx-lm's ``model*.safetensors`` pattern, and strict loading rejects any
    # key the text backbone does not own, so they must be dropped here.
    _NON_TEXT_PREFIXES = (
        "model.audio_tokenizer.",
        "model.vision_tokenizer.",
    )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized = {}
        for key, value in weights.items():
            if key.startswith(self._NON_TEXT_PREFIXES):
                continue
            if key.startswith("model.language_model."):
                key = "model." + key[len("model.language_model.") :]
            if key.endswith(("alpha_p", "alpha_n", ".beta", ".eps")):
                value = value.squeeze()
            sanitized[key] = value
        if self.args.tie_word_embeddings:
            sanitized.pop("lm_head.weight", None)
        return sanitized

    @property
    def layers(self):
        return self.model.layers
