"""CLIPTextModel: MLX port of the CLIP text encoders SDXL conditions on.

SDXL uses two of these — CLIP ViT-L/14 (``text_encoder``) and OpenCLIP ViT-bigG/14
(``text_encoder_2``). For each it takes the **penultimate** hidden state as the
per-token sequence (the two are concatenated → 2048-dim cross-attention context),
and from the bigG encoder it also takes the **projected pooled** embedding (the
hidden state at the EOS token, run through ``text_projection``) for SDXL's added
time/text conditioning.

Module names mirror transformers' ``CLIPTextModel`` so weight conversion is nearly
identity. The text transformer is causal (autoregressive masking) with pre-LayerNorm
blocks; the two encoders differ only in size and activation (``quick_gelu`` for L,
``gelu`` for bigG).
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn

from ..configuration import Config
from ..modeling import ModelMixin


@dataclasses.dataclass
class CLIPTextConfig(Config):
    vocab_size: int = 49408
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"  # "gelu" for the bigG encoder
    projection_dim: int = 768
    layer_norm_eps: float = 1e-5
    with_projection: bool = (
        False  # True for the bigG encoder (CLIPTextModelWithProjection)
    )


def _act(name: str):
    if name == "quick_gelu":
        return lambda x: x * mx.sigmoid(1.702 * x)
    if name in ("gelu", "gelu_new"):
        return nn.gelu
    raise ValueError(f"Unsupported CLIP activation {name!r}.")


class _CLIPEmbeddings(nn.Module):
    def __init__(self, c: CLIPTextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(c.vocab_size, c.hidden_size)
        self.position_embedding = nn.Embedding(c.max_position_embeddings, c.hidden_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        positions = mx.arange(input_ids.shape[-1])
        return self.token_embedding(input_ids) + self.position_embedding(positions)


class _CLIPAttention(nn.Module):
    def __init__(self, c: CLIPTextConfig):
        super().__init__()
        self.num_heads = c.num_attention_heads
        self.head_dim = c.hidden_size // c.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(c.hidden_size, c.hidden_size)
        self.k_proj = nn.Linear(c.hidden_size, c.hidden_size)
        self.v_proj = nn.Linear(c.hidden_size, c.hidden_size)
        self.out_proj = nn.Linear(c.hidden_size, c.hidden_size)

    def _heads(self, x: mx.array) -> mx.array:
        b, t, _ = x.shape
        return x.reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        q, k, v = (
            self._heads(self.q_proj(x)),
            self._heads(self.k_proj(x)),
            self._heads(self.v_proj(x)),
        )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        b, _, t, _ = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(b, t, self.num_heads * self.head_dim)
        return self.out_proj(out)


class _CLIPMLP(nn.Module):
    def __init__(self, c: CLIPTextConfig):
        super().__init__()
        self.fc1 = nn.Linear(c.hidden_size, c.intermediate_size)
        self.fc2 = nn.Linear(c.intermediate_size, c.hidden_size)
        self.act = _act(c.hidden_act)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class _CLIPEncoderLayer(nn.Module):
    def __init__(self, c: CLIPTextConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.self_attn = _CLIPAttention(c)
        self.layer_norm2 = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.mlp = _CLIPMLP(c)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class _CLIPEncoder(nn.Module):
    def __init__(self, c: CLIPTextConfig):
        super().__init__()
        self.layers = [_CLIPEncoderLayer(c) for _ in range(c.num_hidden_layers)]


class _CLIPTextTransformer(nn.Module):
    def __init__(self, c: CLIPTextConfig):
        super().__init__()
        self.embeddings = _CLIPEmbeddings(c)
        self.encoder = _CLIPEncoder(c)
        self.final_layer_norm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)


class CLIPTextModel(ModelMixin[CLIPTextConfig]):
    """CLIP text encoder. Returns all hidden states + the projected pooled embedding."""

    config_class = CLIPTextConfig

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        self.text_model = _CLIPTextTransformer(config)
        self.text_projection = (
            nn.Linear(config.hidden_size, config.projection_dim, bias=False)
            if config.with_projection
            else None
        )

    def __call__(self, input_ids: mx.array) -> tuple[list[mx.array], mx.array]:
        """``(B, L) int -> (hidden_states, pooled)``.

        ``hidden_states`` has length ``num_layers + 1`` (embeddings then each layer
        output); SDXL reads ``[-2]``. ``pooled`` is the final-norm hidden state at the
        EOS token (the highest token id), projected by ``text_projection`` when present
        (the bigG encoder). SDXL uses the pooled output only from that encoder.
        """
        x = self.text_model.embeddings(input_ids)
        seq = input_ids.shape[-1]
        mask = mx.triu(mx.full((seq, seq), -mx.inf, dtype=x.dtype), k=1)

        hidden_states = [x]
        for layer in self.text_model.encoder.layers:
            x = layer(x, mask)
            hidden_states.append(x)

        last = self.text_model.final_layer_norm(x)
        eos = mx.argmax(input_ids, axis=-1)
        pooled = last[mx.arange(input_ids.shape[0]), eos]
        if self.text_projection is not None:
            pooled = self.text_projection(pooled)
        return hidden_states, pooled
