"""T5 Encoder for CogVideoX-Fun, ported to MLX.

Implements T5EncoderModel (encoder-only T5) for text encoding.
Architecture: T5-v1.1-XXL with gated-GELU feed-forward.
"""

import json
import math
import os
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class T5RMSNorm(nn.Module):
    """T5-style RMS normalization (no bias, no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class T5Attention(nn.Module):
    """T5 self-attention with relative position bias."""

    def __init__(
        self,
        d_model: int,
        d_kv: int,
        num_heads: int,
        has_relative_attention_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_kv = d_kv
        self.has_relative_attention_bias = has_relative_attention_bias

        self.q = nn.Linear(d_model, num_heads * d_kv, bias=False)
        self.k = nn.Linear(d_model, num_heads * d_kv, bias=False)
        self.v = nn.Linear(d_model, num_heads * d_kv, bias=False)
        self.o = nn.Linear(num_heads * d_kv, d_model, bias=False)

        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)
            self.relative_attention_num_buckets = relative_attention_num_buckets
            self.relative_attention_max_distance = relative_attention_max_distance

    @staticmethod
    def _relative_position_bucket(
        relative_position: mx.array,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> mx.array:
        """Compute relative position buckets (T5-style)."""
        relative_buckets = mx.zeros(relative_position.shape, dtype=mx.int32)
        num_buckets //= 2
        relative_buckets = relative_buckets + (relative_position > 0).astype(mx.int32) * num_buckets
        relative_position = mx.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            mx.log(relative_position.astype(mx.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(mx.int32)
        relative_position_if_large = mx.minimum(relative_position_if_large, mx.array(num_buckets - 1))
        relative_buckets = relative_buckets + mx.where(
            is_small, relative_position.astype(mx.int32), relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int) -> mx.array:
        """Compute relative position bias."""
        context_position = mx.arange(query_length).reshape(-1, 1)
        memory_position = mx.arange(key_length).reshape(1, -1)
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        # (query_length, key_length, num_heads) -> (1, num_heads, query_length, key_length)
        values = values.transpose(2, 0, 1).reshape(1, self.num_heads, query_length, key_length)
        return values

    def __call__(
        self,
        x: mx.array,
        position_bias: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> tuple:
        B, L, _ = x.shape

        q = self.q(x).reshape(B, L, self.num_heads, self.d_kv).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, L, self.num_heads, self.d_kv).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, L, self.num_heads, self.d_kv).transpose(0, 2, 1, 3)

        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(L, L)

        scale = self.d_kv**-0.5
        scores = (q * scale) @ k.transpose(0, 1, 3, 2)

        if position_bias is not None:
            scores = scores + position_bias

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = mx.softmax(scores, axis=-1)
        out = weights @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = self.o(out)

        return out, position_bias


class T5GatedGELU(nn.Module):
    """T5 gated-GELU feed-forward (DenseReluDense with gating)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)  # value
        self.wo = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.gelu_approx(self.wi_0(x))
        value = self.wi_1(x)
        return self.wo(gate * value)


class T5Block(nn.Module):
    """T5 encoder block: self-attention + feed-forward."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_kv: int,
        num_heads: int,
        eps: float = 1e-6,
        has_relative_attention_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
    ):
        super().__init__()
        # layer.0 = self-attention sublayer
        self.layer = [
            T5AttentionSublayer(
                d_model,
                d_kv,
                num_heads,
                eps,
                has_relative_attention_bias,
                relative_attention_num_buckets,
                relative_attention_max_distance,
            ),
            T5FFSublayer(d_model, d_ff, eps),
        ]

    def __call__(self, x: mx.array, position_bias=None, attention_mask=None):
        x, position_bias = self.layer[0](x, position_bias, attention_mask)
        x = self.layer[1](x)
        return x, position_bias


class T5AttentionSublayer(nn.Module):
    """Self-attention sublayer with pre-norm and residual."""

    def __init__(self, d_model, d_kv, num_heads, eps, has_rpb, num_buckets, max_dist):
        super().__init__()
        self.layer_norm = T5RMSNorm(d_model, eps)
        self.SelfAttention = T5Attention(
            d_model,
            d_kv,
            num_heads,
            has_rpb,
            num_buckets,
            max_dist,
        )

    def __call__(self, x, position_bias=None, attention_mask=None):
        normed = self.layer_norm(x)
        attn_out, position_bias = self.SelfAttention(normed, position_bias, attention_mask)
        return x + attn_out, position_bias


class T5FFSublayer(nn.Module):
    """Feed-forward sublayer with pre-norm and residual."""

    def __init__(self, d_model, d_ff, eps):
        super().__init__()
        self.layer_norm = T5RMSNorm(d_model, eps)
        self.DenseReluDense = T5GatedGELU(d_model, d_ff)

    def __call__(self, x):
        normed = self.layer_norm(x)
        return x + self.DenseReluDense(normed)


class T5Encoder(nn.Module):
    """T5 Encoder (encoder-only model for text embedding)."""

    def __init__(self, config: dict):
        super().__init__()
        d_model = config["d_model"]
        d_ff = config["d_ff"]
        d_kv = config["d_kv"]
        num_heads = config["num_heads"]
        num_layers = config["num_layers"]
        vocab_size = config["vocab_size"]
        eps = config.get("layer_norm_epsilon", 1e-6)
        num_buckets = config.get("relative_attention_num_buckets", 32)
        max_dist = config.get("relative_attention_max_distance", 128)

        self.shared = nn.Embedding(vocab_size, d_model)
        self.encoder = T5EncoderStack(
            d_model,
            d_ff,
            d_kv,
            num_heads,
            num_layers,
            eps,
            num_buckets,
            max_dist,
        )

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Encode input token IDs to hidden states.

        Args:
            input_ids: (B, L) integer token IDs.

        Returns:
            (B, L, d_model) hidden states.
        """
        x = self.shared(input_ids)
        # Compute in float32 for numerical stability (weights are bf16)
        x = x.astype(mx.float32)

        # Create attention mask: 0 for real tokens, -inf for padding (id=0)
        # Shape: (B, 1, 1, L) for broadcasting with (B, H, L, L) scores
        pad_mask = (input_ids == 0).astype(mx.float32)  # 1 where padding
        attention_mask = pad_mask[:, None, None, :] * -1e9  # (B, 1, 1, L)

        return self.encoder(x, attention_mask=attention_mask)

    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load T5 encoder from a directory with config + safetensors.

        Supports two layouts:
        - mlx-forge flat: text_encoder_config.json + text_encoder.safetensors
        - HuggingFace nested: text_encoder/config.json + text_encoder/*.safetensors
        """
        # Find config
        config_file = os.path.join(model_path, "text_encoder_config.json")
        if not os.path.exists(config_file):
            config_file = os.path.join(model_path, "text_encoder", "config.json")
        if not os.path.exists(config_file):
            config_file = os.path.join(model_path, "config.json")
        with open(config_file) as f:
            config = json.load(f)

        model = cls(config)

        # Will quantize after loading weights dict but before load_weights

        # Find weights — flat or nested, single or sharded
        weights_file = os.path.join(model_path, "text_encoder.safetensors")
        if not os.path.exists(weights_file):
            from pathlib import Path

            te_dir = Path(model_path) / "text_encoder"
            if te_dir.is_dir():
                shard_files = sorted(te_dir.glob("*.safetensors"))
                if shard_files:
                    weights = {}
                    for sf in shard_files:
                        weights.update(mx.load(str(sf)))
                    cleaned = {k.removeprefix("text_encoder."): v for k, v in weights.items()}
                    from videox_fun_mlx.utils import quantize_model_from_weights

                    quantize_model_from_weights(model, cleaned, model_path, "text_encoder")
                    model.load_weights(list(cleaned.items()))
                    return model
            raise FileNotFoundError(f"No T5 weights found in {model_path}")

        weights = mx.load(weights_file)
        cleaned = {k.removeprefix("text_encoder."): v for k, v in weights.items()}
        from videox_fun_mlx.utils import quantize_model_from_weights

        quantize_model_from_weights(model, cleaned, model_path, "text_encoder")
        model.load_weights(list(cleaned.items()))
        return model


class T5EncoderStack(nn.Module):
    """Stack of T5 encoder blocks."""

    def __init__(self, d_model, d_ff, d_kv, num_heads, num_layers, eps, num_buckets, max_dist):
        super().__init__()
        self.block = []
        for i in range(num_layers):
            self.block.append(
                T5Block(
                    d_model,
                    d_ff,
                    d_kv,
                    num_heads,
                    eps,
                    has_relative_attention_bias=(i == 0),
                    relative_attention_num_buckets=num_buckets,
                    relative_attention_max_distance=max_dist,
                )
            )
        self.final_layer_norm = T5RMSNorm(d_model, eps)

    def __call__(self, x: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        position_bias = None
        for block in self.block:
            x, position_bias = block(x, position_bias, attention_mask)
        return self.final_layer_norm(x)
