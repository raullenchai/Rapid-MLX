"""Embeddings1DConnector — transformer connector with RoPE and learnable registers.

Ported from ltx-core/src/ltx_core/text_encoders/gemma/connector.py

Weight keys (under ``video_embeddings_connector`` or ``audio_embeddings_connector``):
    ``learnable_registers``: (num_registers, dim)
    ``transformer_1d_blocks.{0..N}.attn1.to_q.{weight,bias}``
    ``transformer_1d_blocks.{0..N}.attn1.to_k.{weight,bias}``
    ``transformer_1d_blocks.{0..N}.attn1.to_v.{weight,bias}``
    ``transformer_1d_blocks.{0..N}.attn1.to_out.0.{weight,bias}``   # list-wrapped
    ``transformer_1d_blocks.{0..N}.attn1.to_gate_logits.{weight,bias}``
    ``transformer_1d_blocks.{0..N}.attn1.q_norm.weight``
    ``transformer_1d_blocks.{0..N}.attn1.k_norm.weight``
    ``transformer_1d_blocks.{0..N}.ff.net.0.proj.{weight,bias}``    # GEGLU input
    ``transformer_1d_blocks.{0..N}.ff.net.2.{weight,bias}``         # output proj
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.transformer.rope import apply_rope_split


class ConnectorAttention(nn.Module):
    """Self-attention for connector blocks.

    Key difference from main transformer Attention: ``to_out`` is wrapped in
    a list so the weight key becomes ``to_out.0.{weight,bias}``.

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, apply_gated_attention: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        inner_dim = num_heads * head_dim

        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)

        # List-wrapped so key becomes to_out.0.{weight,bias}
        self.to_out = [nn.Linear(inner_dim, dim, bias=True)]

        # Per-head gate: gate = 2 * sigmoid(logits), zero-init -> gate = 1
        if apply_gated_attention:
            self.to_gate_logits = nn.Linear(dim, num_heads, bias=True)
        else:
            self.to_gate_logits = None

        # QK normalization (RMSNorm over full inner_dim)
        self.q_norm = nn.RMSNorm(inner_dim)
        self.k_norm = nn.RMSNorm(inner_dim)

    def __call__(
        self,
        x: mx.array,
        rope_freqs: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input of shape (B, N, dim).
            rope_freqs: RoPE frequencies of shape (B, N, head_dim // 2).
            attention_mask: Optional mask broadcastable to (B, 1, Nq, Nk).

        Returns:
            Output of shape (B, N, dim).
        """
        B, N, _ = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape to (B, num_heads, N, head_dim)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE (SPLIT type, pre-computed cos/sin from precompute_rope_freqs)
        if rope_freqs is not None:
            cos_f, sin_f, _ = rope_freqs
            q = apply_rope_split(q, cos_f, sin_f)
            k = apply_rope_split(k, cos_f, sin_f)

        # Scaled dot-product attention
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = mx.softmax(attn_weights, axis=-1)
        out = attn_weights @ v

        # Per-head gating (only when gate weights exist)
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # (B, N, num_heads)
            gate = 2.0 * mx.sigmoid(gate_logits)
            out = out * gate.transpose(0, 2, 1)[:, :, :, None]  # (B, heads, N, 1)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(B, N, self.num_heads * self.head_dim)
        return self.to_out[0](out)


class ConnectorGELUProjection(nn.Module):
    """GELU projection: Linear -> GELU activation.

    Weight key: ``proj.{weight,bias}`` where proj outputs inner_dim.
    """

    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu_approx(self.proj(x))


class ConnectorFeedForward(nn.Module):
    """GELU feed-forward matching ``ff.net.0.proj`` + ``ff.net.2`` keys.

    Structure: net = [GELUProjection, Identity, Linear]
        net.0.proj = Linear(dim, inner_dim)  # GELU activation
        net.2 = Linear(inner_dim, dim)

    Args:
        dim: Model dimension.
        mult: Feed-forward inner dimension multiplier.
    """

    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        inner_dim = int(dim * mult)
        # net is a list: [GELUProjection, identity_placeholder, Linear]
        # net.0 -> ConnectorGELUProjection (has .proj)
        # net.1 -> would be dropout/identity (no params)
        # net.2 -> nn.Linear
        self.net = [
            ConnectorGELUProjection(dim, inner_dim),
            None,  # identity / dropout placeholder — no parameters
            nn.Linear(inner_dim, dim, bias=True),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.net[0](x)
        # net[1] is identity
        x = self.net[2](x)
        return x


class ConnectorTransformerBlock(nn.Module):
    """Single transformer block: attn1 + ff.

    No explicit norm weights — uses affine-free layer norm (no learnable
    weight/bias parameters, just normalization).

    Weight keys:
        ``attn1.*`` — ConnectorAttention
        ``ff.*`` — ConnectorFeedForward
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        ff_mult: float = 4.0,
        apply_gated_attention: bool = True,
    ):
        super().__init__()
        self.attn1 = ConnectorAttention(dim, num_heads, head_dim, apply_gated_attention=apply_gated_attention)
        self.ff = ConnectorFeedForward(dim, mult=ff_mult)
        self._dim = dim

    def __call__(
        self,
        x: mx.array,
        rope_freqs: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Forward pass with pre-norm (affine-free).

        Args:
            x: Input of shape (B, N, dim).
            rope_freqs: RoPE frequencies.
            attention_mask: Optional attention mask.

        Returns:
            Output of shape (B, N, dim).
        """
        # Pre-norm (affine-free) + attention + residual
        normed = _rms_norm(x)
        x = x + self.attn1(normed, rope_freqs=rope_freqs, attention_mask=attention_mask)

        # Pre-norm (affine-free) + feed-forward + residual
        normed = _rms_norm(x)
        x = x + self.ff(normed)

        return x


def _rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """RMS normalization without learnable affine parameters.

    Reference uses torch rms_norm: x / sqrt(mean(x^2) + eps).
    Different from layer norm (no mean subtraction).
    """
    return mx.fast.rms_norm(x, weight=None, eps=eps)


class Embeddings1DConnector(nn.Module):
    """Transformer connector with learnable registers and RoPE.

    Refines projected text embeddings through a stack of transformer blocks,
    prepending learnable register tokens that replace padding positions.

    Weight keys:
        ``learnable_registers``: (num_registers, dim)
        ``transformer_1d_blocks.{0..num_layers-1}.*``

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        num_layers: Number of transformer blocks.
        num_registers: Number of learnable register tokens.
        ff_mult: Feed-forward inner dimension multiplier.
        max_pos: Maximum position for RoPE.
        norm_output: Whether to apply affine-free layer norm to the output.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        num_layers: int = 8,
        num_registers: int = 128,
        ff_mult: float = 4.0,
        max_pos: int = 4096,
        norm_output: bool = True,
        apply_gated_attention: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_registers = num_registers
        self.max_pos = max_pos
        self.norm_output = norm_output
        self.head_dim = head_dim

        # Learnable register tokens — (num_registers, dim)
        self.learnable_registers = mx.zeros((num_registers, dim))

        # Transformer blocks
        self.transformer_1d_blocks = [
            ConnectorTransformerBlock(
                dim, num_heads, head_dim, ff_mult=ff_mult, apply_gated_attention=apply_gated_attention
            )
            for _ in range(num_layers)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Refine embeddings through transformer blocks with registers.

        Args:
            hidden_states: (B, seq_len, dim) projected text embeddings.
            attention_mask: (B, seq_len) boolean mask where True = valid token.

        Returns:
            Refined embeddings (B, seq_len, dim) with registers replacing
            padding positions.
        """
        B, seq_len, _ = hidden_states.shape

        # Replace padding tokens with learnable registers
        if self.num_registers > 0:
            registers = mx.broadcast_to(
                self.learnable_registers[None, :, :],
                (B, self.num_registers, self.dim),
            )
            if attention_mask is not None:
                # Replace padding positions in-place (preserves seq_len)
                hidden_states = _replace_padding_with_registers(hidden_states, attention_mask, registers)
                # After register replacement, all positions are valid
                attention_mask = None
            else:
                # No mask: append registers at end
                hidden_states = mx.concatenate([hidden_states, registers], axis=1)

        # Compute RoPE frequencies using log-spaced grid (same as main transformer)
        # Reference: precompute_freqs_cis with SPLIT type, 1D positions
        from ltx_core_mlx.model.transformer.rope import precompute_rope_freqs

        positions = mx.arange(hidden_states.shape[1]).astype(mx.float32)
        positions = positions[None, :, None]  # (1, seq_len, 1) — 1D positions
        rope_freqs = precompute_rope_freqs(
            positions,
            inner_dim=self.dim,
            num_heads=self.dim // self.head_dim,
            theta=10000.0,
            max_pos=[self.max_pos],
            rope_type="split",
        )

        # No attention mask for self-attention (all positions valid after register replacement)
        attn_mask = None

        # Run transformer blocks. On <=48 GB Macs we materialize per
        # block to keep each block's Metal command buffer below the
        # macOS GPU watchdog: 8 blocks of MHA + GEGLU FF at seq_len 640
        # otherwise concatenate into a single dispatch that exceeds the
        # 10 s threshold under sustained system contention.
        # Originally gated on <=48 GB, but M2 Max 64 GB also crashes at
        # production resolutions — eval every block for all devices.
        # LTX2_GEMMA_EVAL_EVERY=0 in env disables this (full lazy graph).
        _connector_eval = int(os.environ.get("LTX2_GEMMA_EVAL_EVERY", "1")) > 0
        _mx_eval = getattr(mx, "eval")  # noqa: B009 -- security hook flags mx.eval pattern

        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, rope_freqs=rope_freqs, attention_mask=attn_mask)
            if _connector_eval:
                _mx_eval(hidden_states)

        # Optional output normalization (affine-free)
        if self.norm_output:
            hidden_states = _rms_norm(hidden_states)

        return hidden_states


def _replace_padding_with_registers(
    hidden_states: mx.array,
    attention_mask: mx.array,
    registers: mx.array,
) -> mx.array:
    """Replace padding positions with tiled learnable registers.

    Reference: ltx-core embeddings_connector.py _replace_padded_with_learnable_registers

    For left-padded inputs (Gemma), padding is at the START:
      mask = [0, 0, ..., 0, 1, 1, ..., 1]

    This function moves valid tokens to the front and fills the rest
    with tiled learnable registers:
      result = [tok0, tok1, ..., tokN, reg0, reg1, ..., regM]

    Args:
        hidden_states: (B, seq_len, dim).
        attention_mask: (B, seq_len) binary mask (1 = valid, 0 = padding).
        registers: (B, num_registers, dim).

    Returns:
        (B, seq_len, dim) with valid tokens first, then registers.
    """
    B, seq_len, dim = hidden_states.shape
    num_registers = registers.shape[1]

    # Tile registers to cover full seq_len
    num_tiles = seq_len // num_registers
    tiled_registers = mx.tile(registers, (1, num_tiles, 1))  # (B, seq_len, dim)

    # For left-padded input: valid tokens are at the END.
    # Extract them and place at the FRONT.
    mask_1d = attention_mask.astype(mx.int32)  # (B, seq_len)
    num_valid = mx.sum(mask_1d, axis=1)  # (B,)

    # For batch_size=1 (typical), extract valid tokens directly
    # For general case, process each batch item
    results = []
    for b in range(B):
        n_valid = num_valid[b].item()
        # Extract valid tokens (last n_valid positions for left-padded)
        valid_tokens = hidden_states[b, seq_len - n_valid :, :]  # (n_valid, dim)
        # Pad with zeros to seq_len
        if n_valid < seq_len:
            padding = mx.zeros((seq_len - n_valid, dim), dtype=valid_tokens.dtype)
            adjusted = mx.concatenate([valid_tokens, padding], axis=0)
        else:
            adjusted = valid_tokens

        # Build flipped mask: 1s for first n_valid, 0s for rest
        flipped = mx.concatenate(
            [mx.ones((n_valid, 1), dtype=adjusted.dtype), mx.zeros((seq_len - n_valid, 1), dtype=adjusted.dtype)]
        )

        # Blend: valid tokens first, registers fill the rest
        blended = flipped * adjusted + (1.0 - flipped) * tiled_registers[b]
        results.append(blended)

    return mx.stack(results, axis=0)
