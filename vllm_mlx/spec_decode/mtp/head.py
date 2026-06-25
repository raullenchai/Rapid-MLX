# SPDX-License-Identifier: Apache-2.0
"""MTP head module (vendored from mlx-lm PR #990, commit ``50c164fb``).

This is the model-side new architecture introduced by the upstream PR.
It is normally added to ``mlx_lm/models/qwen3_5.py`` directly (the PR
adds ``MTPModule`` and ``MTPDecoderLayer`` classes inline), but to
avoid forking the whole 700-line ``qwen3_5.py`` we vendor the new
pieces here and inject them at runtime via
:func:`vllm_mlx.spec_decode.mtp.qwen3_5_inject.inject_mtp_support`.

The classes are copied verbatim from upstream, with the imports
adjusted to pull from the installed ``mlx_lm`` (we cannot reference
classes that the upstream PR ALSO modifies, like ``Attention``, since
those modifications haven't landed — so we re-import them from the
current mlx-lm and trust that PR #990's MTP head only relies on the
PUBLIC surface of those classes, which has not changed in 0.31.3).

Verbatim provenance
-------------------

* ``MTPDecoderLayer`` — PR #990 diff line 674-697
  (``mlx_lm/models/qwen3_5.py``). Single full-attention transformer
  layer (no GatedDeltaNet). Reuses upstream ``Attention``,
  ``RMSNorm``, ``MLP``, ``SparseMoeBlock``.
* ``MTPModule`` — PR #990 diff line 700-734
  (``mlx_lm/models/qwen3_5.py``). One ``MTPDecoderLayer`` per
  ``mtp_num_hidden_layers`` config field, plus the
  ``pre_fc_norm_hidden`` / ``pre_fc_norm_embedding`` /
  ``fc`` / ``norm`` head wrapping. Predicts token ``t+2`` from
  ``(hidden_state_at_t, embed(token_at_t+1))``.

Both classes preserve the upstream parameter names exactly so
checkpoints converted with PR #990's ``sanitize()`` path load via
``model.load_weights`` without any renaming.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_mtp_module(
    args: Any,
    num_layers: int,
):
    """Construct a fresh ``MTPModule`` matching the upstream PR #990 schema.

    The class is instantiated INSIDE the function so the heavy mlx imports
    only fire when the caller actually needs MTP. This keeps the package's
    top-level ``import vllm_mlx.spec_decode.mtp`` cheap.

    Args:
        args: A ``mlx_lm.models.qwen3_5.TextModelArgs``-like dataclass.
            Must carry ``hidden_size``, ``rms_norm_eps``,
            ``num_experts``, ``intermediate_size``, plus all the
            Attention args (``num_attention_heads``,
            ``num_key_value_heads``, ``rope_parameters``, etc.).
        num_layers: ``mtp_num_hidden_layers`` from ``config.json``.
            Must be ``>= 1`` — caller pre-checks via
            :func:`vllm_mlx.spec_decode.mtp.detect.detect_mtp_eligibility`.

    Returns:
        An ``nn.Module`` whose forward signature is
        ``(hidden_states, next_token_ids, embed_tokens, cache=None)
        -> hidden_states_pre_lm_head``. The caller is expected to
        apply the shared ``lm_head`` (or ``embed_tokens.as_linear``
        for tied-embedding configs) on the returned tensor — exactly
        matching PR #990's ``TextModel.mtp_forward``.
    """
    import mlx.core as mx
    import mlx.nn as nn

    # Pull upstream building blocks from the installed mlx-lm. These
    # are PR #990's PUBLIC dependencies — none of them are modified by
    # the PR, so reusing them across versions is safe.
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.qwen3_5 import (
        MLP,
        Attention,
        SparseMoeBlock,
    )

    if num_layers < 1:
        raise ValueError(f"build_mtp_module requires num_layers >= 1; got {num_layers}")

    class _MTPDecoderLayer(nn.Module):
        """Full-attention-only transformer layer for the MTP head.

        Vendored from PR #990 ``MTPDecoderLayer``. Identical to a
        single ``Qwen3_5DecoderLayer`` with ``is_linear=False`` — the
        upstream PR breaks out a dedicated class so the MTP module
        doesn't accidentally pull GatedDeltaNet in.
        """

        def __init__(self, layer_args):
            super().__init__()
            self.self_attn = Attention(layer_args)
            self.input_layernorm = nn.RMSNorm(
                layer_args.hidden_size, eps=layer_args.rms_norm_eps
            )
            self.post_attention_layernorm = nn.RMSNorm(
                layer_args.hidden_size, eps=layer_args.rms_norm_eps
            )
            if getattr(layer_args, "num_experts", 0) > 0:
                self.mlp = SparseMoeBlock(layer_args)
            else:
                self.mlp = MLP(layer_args.hidden_size, layer_args.intermediate_size)

        def __call__(self, x, mask=None, cache=None):
            r = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + r
            return h + self.mlp(self.post_attention_layernorm(h))

    class _MTPModule(nn.Module):
        """Multi-Token Prediction head (Qwen3.5 / 3.6 native spec decode).

        Vendored from PR #990 ``MTPModule`` (mlx_lm/models/qwen3_5.py).
        Predicts token ``t+2`` from the backbone pre-norm hidden state
        ``h_t`` and the embedding of sampled token ``t+1``, using the
        shared ``lm_head`` with the backbone for the projection.
        """

        def __init__(self, mod_args, n_layers):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(
                mod_args.hidden_size, eps=mod_args.rms_norm_eps
            )
            self.pre_fc_norm_embedding = nn.RMSNorm(
                mod_args.hidden_size, eps=mod_args.rms_norm_eps
            )
            # 2H -> H concat projection. ``bias=False`` matches the
            # upstream PR (and any HF Qwen3.5 release).
            self.fc = nn.Linear(
                mod_args.hidden_size * 2,
                mod_args.hidden_size,
                bias=False,
            )
            self.layers = [_MTPDecoderLayer(mod_args) for _ in range(n_layers)]
            self.norm = nn.RMSNorm(mod_args.hidden_size, eps=mod_args.rms_norm_eps)

        def __call__(
            self,
            hidden_states: mx.array,
            next_token_ids: mx.array,
            embed_tokens: nn.Embedding,
            cache: Any | None = None,
        ) -> mx.array:
            embeds = embed_tokens(next_token_ids)  # (B, N, H)
            e = self.pre_fc_norm_embedding(embeds)
            h = self.pre_fc_norm_hidden(hidden_states)
            fused = self.fc(mx.concatenate([e, h], axis=-1))  # (B, N, H)

            if cache is None:
                cache = [None] * len(self.layers)

            mask = create_attention_mask(fused, cache[0])
            for layer, c in zip(self.layers, cache):
                fused = layer(fused, mask, c)

            return self.norm(fused)  # (B, N, H)

    return _MTPModule(args, num_layers)
