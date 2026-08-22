# SPDX-License-Identifier: Apache-2.0
"""Vendored Nemotron-Labs-Diffusion AR (autoregressive) decoder.

``model_type: nemotron_labs_diffusion`` — NVIDIA's Ministral3-style diffusion
language model family (3B / 8B / 14B). The architecture shares a single
decoder across three inference modes:

* **AR (autoregressive)** — plain causal next-token decoding. This is what
  we vendor (P0). It is semantically identical to a Ministral3 decoder.
* **Block-diffusion (bidirectional)** — masked non-causal denoising; NOT
  implemented here.
* **Linear self-spec** — diffusion head used as a draft head; NOT implemented
  here.

The AR-mode forward pass reduces to a standard LLM: a Ministral3 backbone
(with YaRN RoPE + Llama-4 attention scaling — identical to mlx-lm's
``ministral3.py``) followed by a *separate* ``diffusion_head`` LM projection.
The head is UNTIED from the embeddings (``tie_word_embeddings: false``) and
shares weights across all three modes, so serving AR mode requires only the
backbone + linear head.

**References — the math was verified against BOTH:**

- ``modeling_nemotron_labs_diffusion.py`` (HF Python, 5.0.0) shipped inside
  the checkpoint — authoritative
- ``NemotronLabsDiffusion.swift`` (MLX Swift reference, 887 lines) — the
  MLX port used for the byte-level math cross-check

**Weight layout (affine 4-bit, group_size 64):** every projection AND the
embedding table AND the ``diffusion_head`` are stored affine-quantized
(packed uint32 ``weight`` + bf16 ``scales`` + ``biases`` per group). The
checkpoint prefixes all weights under ``language_model.*`` (NOT
``encoder.*`` as the Swift reference assumes). ``sanitize`` strips that
prefix so the keys line up with this module's ``model.*`` / ``diffusion_head.*``
structure, then mlx-lm's standard ``nn.quantize`` predicate (``.scales`` in
weights) re-quantizes exactly the affine layers — including the embedding
table and the diffusion head. No custom quant path is needed.

**Registration:** installed as ``sys.modules["mlx_lm.models.nemotron_labs_diffusion"]``
by ``vllm_mlx.utils.tokenizer._register_vendored_archs`` so mlx-lm's
``importlib.import_module(f"mlx_lm.models.{model_type}")`` lookup finds it
transparently (same trick as ``deepseek_v4`` / ``muse_glimmer``). The
registration probes for native mlx-lm support first and defers to it if
upstream ever ships the arch.

**Why AR only:** block-diffusion and linear-self-spec require the
bidirectional/masked machinery and a draft-loop integration that are
out of scope for this PR (P1/P2/P3). AR mode stands alone as a correct,
right-fast open-weights diff-LM serving lane.
"""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

# Installing this before any ``mlx_lm`` import protects M5 single-stream
# devices from mlx-lm's module-load-time thread-local stream capture (#404).
from .. import _mlx_compat

_mlx_compat.install()

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope  # noqa: E402


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: int | None = None
    max_position_embeddings: int | None = None
    num_key_value_heads: int | None = None
    rope_parameters: dict[str, float | str] | None = None
    tie_word_embeddings: bool = False
    layer_types: list[str] | None = None
    sliding_window: int | None = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


def _get_llama_4_attn_scale(size, offset, beta: float, max_position_embeddings: int):
    if isinstance(offset, mx.array) and offset.ndim > 0:
        offset = offset[:, None]

    scaling = 1 + beta * mx.log(
        1 + mx.floor((mx.arange(size) + offset) / max_position_embeddings)
    )
    if scaling.ndim == 2:
        return scaling[:, None, :, None]
    else:
        return scaling[:, None]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_parameters["rope_theta"],
            False,
            args.rope_parameters,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        attn_scale: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = 0
        if cache is not None:
            offset = cache.offset
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        queries = queries * attn_scale
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, use_sliding: bool = False):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.use_sliding = use_sliding
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        attn_scale: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), attn_scale, mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class LanguageModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, use_sliding=layer_type == "sliding_attention")
            for layer_type in self.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx = self.layer_types.index("full_attention")
        self.swa_idx = None
        for e, layer in enumerate(self.layers):
            if layer.use_sliding:
                self.swa_idx = e
                break

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)
            offset = 0
        else:
            offset = cache[0].offset

        swa_mask = fa_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        attn_scale = _get_llama_4_attn_scale(
            inputs.shape[1],
            offset,
            self.args.rope_parameters["llama_4_scaling_beta"],
            self.args.rope_parameters["original_max_position_embeddings"],
        ).astype(h.dtype)

        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, attn_scale, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        # Separate diffusion LM head — NOT tied to embeddings.
        self.diffusion_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        return self.diffusion_head(out)

    def sanitize(self, weights):
        # The checkpoint stores all weights under ``language_model.*`` (the
        # Swift reference instead uses ``encoder.*`` / ``base_model.*``).
        # Normalize onto this module's ``model.*`` / ``diffusion_head.*``
        # structure so the standard mlx-lm weight-matching path works.
        new_weights = {}
        for k, v in weights.items():
            if k.startswith("language_model."):
                k = k[len("language_model.") :]
            # Drop any precomputed rotary freqs / base-model shims that a
            # repack might carry (not present in the 3B-4bit checkpoint, kept
            # as a safety net matching the Swift reference).
            if "self_attn.rotary_emb.inv_freq" in k:
                continue
            if k.startswith("base_model.") or k.startswith("encoder."):
                k = k.split(".", 1)[-1]
                # The Swift reference names the backbone ``encoder`` while
                # this mlx-lm module names it ``model``.  Some repacks retain
                # an intermediate ``model.`` component and some do not.
                if not k.startswith(("model.", "diffusion_head.")):
                    k = f"model.{k}"
            new_weights[k] = v
        return new_weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.model.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]
