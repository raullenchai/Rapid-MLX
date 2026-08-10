# SPDX-License-Identifier: Apache-2.0
"""Vendored Meta Muse Glimmer text backbone (``model_type: muse_glimmer``).

**Why vendor:** Meta released Muse Glimmer 30B (open weights, Apache 2.0)
with day-0 support in Ollama's in-house runtime, but neither of our model
backends knows the architecture yet:

- mlx-lm: no support (checked 0.31.3 and upstream main, 2026-08-10)
- mlx-vlm: draft PR Blaizzy/mlx-vlm#1838 (open, unstable, unreviewed)

Vendoring the text backbone serves the competitive surface — chat + ATEM
tool-calls + agents (parsers landed in PR #1791) — on the standard mlx-lm
text lane (BatchGenerator, prompt cache, quantized KV) without waiting for
either upstream. The checkpoint's 1.8B vision tower is dropped at
``sanitize`` and image inputs are rejected; see the
``resolve_serving_lane`` carve-out in ``vllm_mlx/api/utils.py``.

**References — the math was verified against BOTH:**

- transformers main ``models/muse_glimmer/modular_muse_glimmer.py``
  (5.15.0.dev0) — authoritative reference
- Blaizzy/mlx-vlm#1838 draft MLX port — cross-check (its plain-scale
  final norm looked like a draft bug but matches transformers'
  ``Gemma4RMSNorm(with_scale=True)``, ones-init plain scale)

Architecture (29.6B dense LM):

- 52 layers, pattern ``[sliding ×3, full]``; ``sliding_window`` 2048
- sliding layers use RoPE (theta 5e5); full-attention layers are NoPE
  (``layer_rope_theta[i] == 0`` → no position embedding at all)
- GQA 32 query heads / 2 KV heads, head_dim 128 (hidden 6656)
- weightless QK-norm on Q and K; Q is additionally scaled by
  ``qk_scale_factor`` (3.87) ON TOP of the standard ``1/sqrt(head_dim)``
  attention scale
- gated attention output: ``attn_out * sigmoid(gate_proj(x))``
  (Afmoe-style output gate; ``gate_proj`` maps hidden → n_heads*head_dim)
- sandwich norms per layer, all centered-RMS (effective scale ``1 + w``,
  zeros-init): input/pre-ffn with ``rms_norm_eps`` (1e-5),
  post-attn/post-ffn with ``post_norm_eps`` (1e-8)
- embeddings are RMS-normed (weightless) after lookup. The checkpoint
  ships the embedding table UNQUANTIZED (no ``.scales``), so mlx-lm's
  quantization predicate keeps it fp automatically. Do NOT merge the
  norm into the table — upstream DFlash embeds without the norm.
- final ``model.norm`` is a PLAIN-scale RMSNorm (ones-init), not
  centered; ``lm_head`` is untied; logits are scaled by
  ``output_multiplier`` (1/sqrt(hidden/256)) then tanh-softcapped at
  ``final_logit_softcapping`` (20.0)

**Registration:** installed as ``sys.modules["mlx_lm.models.muse_glimmer"]``
by ``vllm_mlx.utils.tokenizer._register_vendored_archs`` so mlx-lm's
``importlib.import_module(f"mlx_lm.models.{model_type}")`` lookup finds it
transparently (same trick as ``deepseek_v4`` / ``hy_v3``).

**Sync policy:** when mlx-lm ships native ``muse_glimmer`` support,
``_register_vendored_archs`` defers to it automatically (``find_spec``
probe, same as ``hy_v3``); delete this file after diffing for bug fixes.
"""

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope

_SLIDING = "sliding_attention"
_FULL = "full_attention"


@dataclass
class TextConfig(BaseModelArgs):
    """Inner ``text_config`` of the muse_glimmer checkpoint.

    Defaults mirror the released 30B config so a truncated config (or a
    tiny test config that only overrides shapes) still builds a
    structurally faithful model.
    """

    model_type: str = "muse_glimmer_text"
    hidden_size: int = 6656
    num_hidden_layers: int = 52
    intermediate_size: int = 19968
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    vocab_size: int = 202048
    sliding_window: int = 2048
    max_position_embeddings: int = 131072
    attention_bias: bool = False
    qk_scale_factor: float = 3.87
    output_multiplier: float = 0.19611613513818404
    final_logit_softcapping: float = 20.0
    tie_word_embeddings: bool = False
    layer_types: list | None = None
    layer_rope_theta: list | None = None
    rope_parameters: dict | None = None

    def __post_init__(self):
        n = self.num_hidden_layers
        if self.layer_types is None:
            # Released pattern: [sliding, sliding, sliding, full] repeating.
            self.layer_types = [
                _FULL if (i + 1) % 4 == 0 else _SLIDING for i in range(n)
            ]
        if len(self.layer_types) != n:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries for {n} layers"
            )
        unknown = set(self.layer_types) - {_SLIDING, _FULL}
        if unknown:
            # A typo would otherwise silently become full attention in
            # DecoderLayer while still receiving RoPE from the default-
            # theta derivation — an unintended architecture (codex r2 #2).
            raise ValueError(f"unknown layer_types entries: {sorted(unknown)}")
        default_theta = float((self.rope_parameters or {}).get("rope_theta", 500000.0))
        if self.layer_rope_theta is None:
            # Sliding layers use RoPE; full-attention layers are NoPE.
            self.layer_rope_theta = [
                0.0 if t == _FULL else default_theta for t in self.layer_types
            ]
        if len(self.layer_rope_theta) != n:
            raise ValueError(
                f"layer_rope_theta has {len(self.layer_rope_theta)} entries "
                f"for {n} layers"
            )


@dataclass
class ModelArgs(BaseModelArgs):
    """Outer checkpoint config. The LM shape lives in ``text_config``."""

    model_type: str = "muse_glimmer"
    text_config: dict = field(default_factory=dict)

    def __post_init__(self):
        self.text = TextConfig.from_dict(self.text_config or {})

    @classmethod
    def from_dict(cls, params):
        # ``BaseModelArgs.from_dict`` filters unknown keys, so a
        # flattened text-only export (LM shape at the top level, no
        # nested ``text_config``) would otherwise silently collapse to
        # the 30B defaults (codex r1 #1). Rebuild the inner config from
        # the FULL raw dict in that case.
        args = super().from_dict(params)
        if not args.text_config:
            args.text = TextConfig.from_dict(params)
        return args


class RMSNormNoScale(nn.Module):
    """Weightless RMSNorm (QK-norm and the post-lookup embedding norm)."""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class CenteredRMSNorm(nn.Module):
    """RMSNorm with a zero-centered checkpoint scale (effective ``1 + w``).

    Same idiom as mlx-lm's gemma3 RMSNorm — ``mx.fast.rms_norm``
    accumulates in fp32 internally, matching transformers'
    float()-then-cast semantics.
    """

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        # Standard attention scale; qk_scale_factor is applied to Q
        # separately (transformers: on top of self.scaling).
        self.scale = self.head_dim**-0.5
        self.qk_scale_factor = args.qk_scale_factor
        self.is_sliding = args.layer_types[layer_idx] == _SLIDING
        theta = float(args.layer_rope_theta[layer_idx])
        self.use_rope = theta != 0.0

        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        # Afmoe-style output gate — no bias in the checkpoint.
        self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias
        )
        self.qk_norm = RMSNormNoScale(args.rms_norm_eps)

        if self.use_rope:
            self.rope = initialize_rope(
                self.head_dim,
                base=theta,
                traditional=False,
                max_position_embeddings=args.max_position_embeddings,
            )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1)

        # Weightless QK-norm over head_dim; Q gets the extra scale factor.
        queries = (self.qk_norm(queries) * self.qk_scale_factor).transpose(0, 2, 1, 3)
        keys = self.qk_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.use_rope:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = output * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.is_sliding = args.layer_types[layer_idx] == _SLIDING
        self.input_layernorm = CenteredRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = CenteredRMSNorm(
            args.hidden_size, args.post_norm_eps
        )
        self.pre_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.rms_norm_eps
        )
        self.post_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.post_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + self.post_attention_layernorm(r)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        return h + self.post_feedforward_layernorm(r)


class MuseModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        # Post-lookup norm; kept OUTSIDE the embedding table (weightless,
        # so no checkpoint key) — see module docstring.
        self.embed_norm = RMSNormNoScale(args.rms_norm_eps)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        # Final norm is plain-scale (ones-init) — deliberately not
        # CenteredRMSNorm; matches Gemma4RMSNorm(with_scale=True).
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Index of a representative layer per attention kind, for mask
        # construction (masks depend on the cache's current offset).
        # Either kind may be absent in a tiny/explicit config — e.g. the
        # default [S,S,S,F] pattern yields no full layer below 4 layers
        # (codex r1 #2) — so both lookups are guarded.
        self.first_full_idx = (
            args.layer_types.index(_FULL) if _FULL in args.layer_types else None
        )
        self.first_sliding_idx = (
            args.layer_types.index(_SLIDING) if _SLIDING in args.layer_types else None
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        if input_embeddings is not None:
            # Provided embeddings are used RAW — transformers norms only
            # inside the lookup (NormedEmbedding), and multimodal callers
            # pass pre-merged embeds that must not be re-normed.
            h = input_embeddings
        else:
            h = self.embed_norm(self.embed_tokens(inputs))

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = None
        if self.first_full_idx is not None:
            full_mask = create_attention_mask(h, cache[self.first_full_idx])
        sliding_mask = None
        if self.first_sliding_idx is not None:
            sliding_mask = create_attention_mask(
                h,
                cache[self.first_sliding_idx],
                window_size=self.args.sliding_window,
            )

        for layer, c in zip(self.layers, cache):
            mask = sliding_mask if layer.is_sliding else full_mask
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.text_args = args.text
        self.model = MuseModel(args.text)
        # Honour the config's tying declaration (codex r2 #1); the
        # released 30B is untied. ``sanitize`` still flips to tied as a
        # fallback when an untied config ships no head weights.
        self.tie_word_embeddings = args.text.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.text.hidden_size, args.text.vocab_size, bias=False
            )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = self.model(inputs, cache, input_embeddings)
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(h)
        else:
            logits = self.lm_head(h)
        logits = logits * self.text_args.output_multiplier
        softcap = self.text_args.final_logit_softcapping
        if softcap:
            logits = mx.tanh(logits / softcap) * softcap
        return logits

    def sanitize(self, weights):
        """Strip the multimodal wrapper down to the text backbone.

        The released checkpoint namespaces the LM under
        ``language_model.`` and ships the vision stack under three
        sibling prefixes; only the LM is served here. A future text-only
        export (keys already bare) passes through unchanged.
        """
        vision_prefixes = (
            "vision_tower.",
            "vision_adapter.",
            "vision_projection.",
            "multi_modal_projector.",
        )
        out = {}
        for k, v in weights.items():
            if k.startswith(vision_prefixes):
                continue
            if k.startswith("language_model."):
                k = k[len("language_model.") :]
            out[k] = v
        if self.tie_word_embeddings:
            # Config-declared tying: the embedding table IS the head.
            # Drop any stray head weights so strict loading can't trip
            # on keys with no matching module (codex r2 #1).
            out = {k: v for k, v in out.items() if not k.startswith("lm_head.")}
        # No silent tie fallback for an untied config missing head
        # weights: that would convert an incomplete/incompatible export
        # into a model that loads but produces WRONG logits. Strict
        # weight loading reports the missing ``lm_head.weight`` instead
        # (codex r5 #1). Tying happens only when the config declares it.
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.text_args.sliding_window)
                if t == _SLIDING
                else KVCache()
            )
            for t in self.text_args.layer_types
        ]
