# SPDX-License-Identifier: Apache-2.0
"""HY3 (Tencent Hunyuan 3) native MTP head — DeepSeek-V3-style single next-n layer.

Unlike Qwen3.5's MTP head (``rapid_mlx.spec_decode.mtp.head._MTPModule``), HY3's
MTP layer is DeepSeek-V3-shaped:

* two RMSNorms named ``enorm`` (applied to the *next-token embedding*) and
  ``hnorm`` (applied to the *previous backbone hidden state*);
* a 2H->H concat projection ``eh_proj`` (no bias) with the concat order
  ``eh_proj(cat([enorm(embed), hnorm(prev_hidden)], axis=-1))`` — **embedding
  first** (confirmed from vLLM ``deepseek_mtp.py`` and SGLang ``hunyuan_v3``
  nextn: both apply ``enorm`` to ``inputs_embeds`` and place it first in the
  cat);
* a single HY3 ``DecoderLayer`` (QK-norm attention + sigmoid-router SwitchGLU
  MoE + a shared expert) forced onto the MoE branch;
* a final RMSNorm ``norm`` (== upstream ``final_layernorm``);
* the shared backbone ``lm_head`` for the projection (HY3 has
  ``tie_word_embeddings=false``).

We reuse :class:`rapid_mlx.models.hy_v3.DecoderLayer` verbatim for the transformer
block so the sidecar's quantized param tree is byte-identical to a quantized
backbone MoE layer (same ``self_attn.q_norm``, ``mlp.router.gate``,
``mlp.router.expert_bias``, ``mlp.switch_mlp.*``, ``mlp.shared_mlp.*`` names).
The DecoderLayer is constructed with ``layer_idx=first_k_dense_replace`` (== 1)
so ``layer_idx < first_k_dense_replace`` is False and it takes the MoE branch
(matches SGLang's ``config.first_k_dense_replace = 0`` override for the nextn
decoder).
"""

from __future__ import annotations

from typing import Any


def build_hy3_mtp_module(args: Any, num_layers: int):
    """Construct a fresh ``_HY3MTPModule`` matching the HY3 layer-80 schema.

    Args:
        args: A :class:`rapid_mlx.models.hy_v3.ModelArgs` dataclass (carries
            ``hidden_size``, ``rms_norm_eps``, ``num_experts``,
            ``first_k_dense_replace``, all attention args, etc.).
        num_layers: ``num_nextn_predict_layers`` from config (== 1 for HY3).

    Returns:
        An ``nn.Module`` whose forward is
        ``(hidden_states, next_token_ids, embed_tokens, cache=None)
        -> hidden_states_pre_lm_head``. The caller applies the shared
        ``lm_head`` on the returned tensor.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.base import create_attention_mask

    from rapid_mlx.models.hy_v3 import DecoderLayer  # pragma: no cover - injection path

    if num_layers < 1:
        raise ValueError(
            f"build_hy3_mtp_module requires num_layers >= 1; got {num_layers}"
        )
    if num_layers != 1:
        # HY3 ships exactly one next-n layer. Guard so a future >1 config
        # doesn't silently under-run.
        raise ValueError(f"HY3 MTP supports exactly 1 next-n layer; got {num_layers}")

    fkdr = int(getattr(args, "first_k_dense_replace", 1))
    # Force the MoE branch: DecoderLayer picks MoE iff layer_idx >= fkdr.
    moe_layer_idx = max(fkdr, 1)

    class _HY3MTPModule(nn.Module):
        """DeepSeek-V3-style MTP head for HY3 (embedding-first eh_proj concat)."""

        def __init__(self, mod_args, n_layers):
            super().__init__()
            self.enorm = nn.RMSNorm(mod_args.hidden_size, eps=mod_args.rms_norm_eps)
            self.hnorm = nn.RMSNorm(mod_args.hidden_size, eps=mod_args.rms_norm_eps)
            self.eh_proj = nn.Linear(
                mod_args.hidden_size * 2, mod_args.hidden_size, bias=False
            )
            # Single HY3 DecoderLayer on the MoE branch.
            self.layers = [
                DecoderLayer(mod_args, moe_layer_idx) for _ in range(n_layers)
            ]
            self.norm = nn.RMSNorm(mod_args.hidden_size, eps=mod_args.rms_norm_eps)

        def __call__(
            self,
            hidden_states: mx.array,
            next_token_ids: mx.array,
            embed_tokens: nn.Embedding,
            cache: Any | None = None,
        ) -> mx.array:
            # embed next tokens, then DeepSeek-V3-style fused projection.
            embeds = embed_tokens(next_token_ids)  # (B, N, H)
            e = self.enorm(embeds)
            h = self.hnorm(hidden_states)
            # EMBEDDING FIRST — confirmed from vLLM deepseek_mtp.py /
            # SGLang hunyuan_v3 nextn.
            fused = self.eh_proj(mx.concatenate([e, h], axis=-1))  # (B, N, H)

            if cache is None:
                cache = [None] * len(self.layers)

            mask = create_attention_mask(fused, cache[0])
            for layer, c in zip(self.layers, cache):
                fused = layer(fused, mask=mask, cache=c)

            return self.norm(fused)  # (B, N, H)

    return _HY3MTPModule(args, num_layers)
