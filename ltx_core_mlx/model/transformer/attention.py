"""Multi-head attention with RoPE and per-head gating.

Ported from ltx-core/src/ltx_core/model/transformer/attention.py

Weight keys (relative to parent, e.g. ``attn1``):
    ``to_q.{weight,bias}``, ``to_k.{weight,bias}``, ``to_v.{weight,bias}``
    ``to_out.{weight,bias}``          -- single Linear (NOT ``to_out.0``)
    ``to_gate_logits.{weight,bias}``  -- per-head gate logits
    ``q_norm.weight``, ``k_norm.weight`` -- RMSNorm over full inner_dim
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.transformer.rope import apply_rope_interleaved, apply_rope_split


class Attention(nn.Module):
    """Multi-head attention with optional RoPE and per-head gating.

    The gate is computed as ``2 * sigmoid(to_gate_logits(x))`` so that
    zero-initialised logits produce an identity gate (value 1.0).

    Args:
        query_dim: Dimension of the query input.
        kv_dim: Dimension of the key/value input.  Defaults to *query_dim*.
        out_dim: Dimension of the output projection.  Defaults to *query_dim*.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        qkv_bias: Whether to use bias in Q/K/V projections.
        use_rope: Whether to apply rotary position embeddings.
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int | None = None,
        out_dim: int | None = None,
        num_heads: int = 32,
        head_dim: int = 128,
        qkv_bias: bool = True,
        use_rope: bool = True,
        norm_eps: float = 1e-6,
        apply_gated_attention: bool = True,
    ):
        super().__init__()
        if kv_dim is None:
            kv_dim = query_dim
        if out_dim is None:
            out_dim = query_dim

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_rope = use_rope
        self.scale = head_dim**-0.5

        inner_dim = num_heads * head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, out_dim, bias=True)

        # Per-head gate: gate = 2 * sigmoid(logits), zero-init -> gate = 1
        # Reference: only created when apply_gated_attention=True
        if apply_gated_attention:
            self.to_gate_logits = nn.Linear(query_dim, num_heads, bias=True)
        else:
            self.to_gate_logits = None

        # QK normalization (RMSNorm over full inner_dim, applied per-token)
        self.q_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(inner_dim, eps=norm_eps)

    def __call__(
        self,
        x: mx.array,
        encoder_hidden_states: mx.array | None = None,
        rope_freqs: mx.array | None = None,
        rope_freqs_k: mx.array | None = None,
        attention_mask: mx.array | None = None,
        perturbation_mask: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input of shape (B, N, query_dim).
            encoder_hidden_states: Cross-attention keys/values if not None.
            rope_freqs: RoPE frequencies for Q (and K if rope_freqs_k is None).
            rope_freqs_k: Separate RoPE frequencies for K (cross-attention).
            attention_mask: Optional mask broadcastable to (B, 1, Nq, Nk).

        Returns:
            Output of shape (B, N, out_dim).
        """
        B, N, _ = x.shape
        kv_input = encoder_hidden_states if encoder_hidden_states is not None else x

        q = self.to_q(x)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        # QK normalization (over full inner_dim before reshaping)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape to (B, num_heads, seq_len, head_dim)
        q = q.reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE -- freqs are (cos, sin, rope_type) tuple
        if self.use_rope and rope_freqs is not None:
            cos_f, sin_f, rtype = rope_freqs
            _apply = apply_rope_split if rtype == "split" else apply_rope_interleaved
            q = _apply(q, cos_f, sin_f)

            if rope_freqs_k is not None:
                cos_fk, sin_fk, _ = rope_freqs_k
                k = _apply(k, cos_fk, sin_fk)
            elif encoder_hidden_states is None:
                # Self-attention can reuse the query frequencies.  For
                # cross-attention, however, the key positions may differ;
                # without separate key positions there is no valid frequency
                # tensor to apply to it.
                k = _apply(k, cos_f, sin_f)

        # Scaled dot-product attention (fused Flash Attention kernel)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attention_mask)

        # STG perturbation: blend attn output with value projection
        # Reference: out = attn_out * mask + v * (1 - mask)
        if perturbation_mask is not None:
            # perturbation_mask: (B, 1, 1, 1) — 1.0=keep attn, 0.0=skip to values
            out = out * perturbation_mask + v * (1.0 - perturbation_mask)

        # Per-head gating: gate = 2 * sigmoid(logits)
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # (B, N, num_heads)
            gate = 2.0 * mx.sigmoid(gate_logits)
            out = out * gate.transpose(0, 2, 1)[:, :, :, None]  # (B, heads, N, 1)

        # Reshape back: (B, num_heads, N, head_dim) -> (B, N, inner_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, -1, self.num_heads * self.head_dim)
        return self.to_out(out)
