# SPDX-License-Identifier: Apache-2.0
"""Vendored AI9Stars G9v3 MoE text backbone (``model_type: g9v3``).

**Why vendor:** ``ai9stars/G9v3-39A5B`` (open weights, 39B total / ~5B
active per token) ships as a ``trust_remote_code`` transformers checkpoint
with no MLX conversion on the Hub and no mlx-lm support (checked 0.31.3 and
upstream main, 2026-09-05). The architecture is a DeepSeek-V3-style
sigmoid-routed MoE under a Qwen3-Next-style *gated* GQA attention, so this
port reuses mlx-lm's tested kernels for both halves and only owns the glue:
config parsing, the gated ``q_proj`` split, the dense first layer, and the
checkpoint → stacked-expert weight sanitize.

**References — the math was verified against BOTH:**

- ``ai9stars/G9v3-39A5B`` remote code ``modeling_g9v3.py`` +
  ``configuration_g9v3.py`` (transformers 4.57 lineage) — authoritative
  reference; random-weight logit parity and released-weight top-k parity
  are recorded in the PR body of the vendoring change (#3046)
- mlx-lm 0.31.3 ``models/deepseek_v3.py`` (``group_expert_select`` router,
  ``SwitchGLU`` expert MLP, per-expert → stacked weight layout) and
  ``models/qwen3_next.py`` (gated ``q_proj`` split) — the reused kernels

Architecture (39B-A5B, 38 layers, hidden 2048, vocab 130560):

- pre-norm decoder layers; ``RMSNorm`` eps 1e-6 on the layer inputs, the
  post-attention residual and the final ``model.norm``; no embedding
  scaling, no logit softcap; ``lm_head`` is untied
- GQA: 32 query heads / 2 KV heads, head_dim 128, no QK-norm, no biases
  (``attention_bias: false``)
- gated attention (``use_gated_attention: true``): ``q_proj`` emits
  ``2 * n_heads * head_dim``; the output is viewed as
  ``(…, n_heads, 2 * head_dim)`` and chunked on the last axis into
  ``[query | gate]`` per head; the attention output (before ``o_proj``) is
  multiplied by ``sigmoid(gate)``. transformers' ``torch.chunk(…, 2,
  dim=-1)`` on that view is exactly ``mx.split(…, 2, axis=-1)`` here.
- standard NeoX RoPE (rotate-half, ``traditional=False``), theta 5e6, no
  scaling (``rope_scaling: null``)
- layer 0 is a dense SwiGLU MLP (``first_k_dense_replace: 1``,
  intermediate 8192); layers 1..37 are MoE blocks with 320 routed experts
  (intermediate 512), 32 active per token, plus one always-on shared
  expert (intermediate ``512 * n_shared_experts``) added to the routed sum
- router (``G9v3TopkRouter``): logits computed in float32 → sigmoid;
  ``e_score_correction_bias`` is added for expert *selection only*;
  ``n_group = topk_group = 1`` so the group stage is a no-op (plain top-32
  of 320); the mixing weights are the raw sigmoid scores of the selected
  experts, normalised to sum 1 (``norm_topk_prob: true``) and scaled by
  ``routed_scaling_factor`` 3.66. That is precisely mlx-lm's
  ``group_expert_select`` contract, so the router is *imported*, not
  re-derived.

Checkpoint layout: keys are the plain transformers names
(``model.layers.N.self_attn.{q,k,v,o}_proj``, ``model.layers.N.mlp.{gate_proj,
up_proj,down_proj}`` on the dense layer, ``…mlp.gate.{weight,
e_score_correction_bias}``, ``…mlp.experts.E.*``, ``…mlp.shared_experts.*``).
``sanitize`` stacks the 320 per-expert matrices into ``SwitchGLU``'s
``switch_mlp.{gate,up,down}_proj`` tensors (bf16 ``weight`` or quantized
``weight/scales/biases``) and passes an already-stacked MLX export through
unchanged. The router ``weight`` is a raw array (not ``nn.Linear``) so
mlx-lm's quantizer leaves it in full precision — the same choice as
``deepseek_v3`` — ``Model.quant_predicate`` pins the conversion recipe
(experts at the requested bits, everything else 8-bit; see its docstring
for the measurements) and ``Model.cast_predicate`` keeps the F32 router
bias out of the bf16 cast (rounding it re-picks experts).

**Registration:** installed as ``sys.modules["mlx_lm.models.g9v3"]`` by
``vllm_mlx.utils.tokenizer._register_vendored_archs`` so mlx-lm's
``importlib.import_module(f"mlx_lm.models.{model_type}")`` lookup finds it
transparently (same trick as ``deepseek_v4`` / ``hy_v3`` /
``muse_glimmer``). The tokenizer is a stock ``tokenizer.json`` fast
tokenizer; ``_VENDORED_MODEL_TYPES`` membership keeps the tokenizer
fallback off transformers' ``AutoConfig`` for the unknown ``model_type``.

**Sync policy:** when mlx-lm ships native ``g9v3`` support,
``_register_vendored_archs`` defers to it automatically (``find_spec``
probe, same as ``hy_v3``); delete this file after diffing for bug fixes.
"""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

# Install the MLX hardware compatibility shim before importing any mlx-lm
# module. mlx-lm captures its default stream during package import, which is
# unsafe on the M5 single-stream path; every vendored model follows this
# ordering contract.
from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.deepseek_v3 import group_expert_select
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    """``config.json`` of a g9v3 checkpoint.

    Defaults mirror the released ``ai9stars/G9v3-39A5B`` config (and the
    remote ``G9v3Config`` defaults for keys it omits) so a truncated
    config, or a tiny test config that only overrides shapes, still builds
    a structurally faithful model.
    """

    model_type: str = "g9v3"
    hidden_size: int = 2048
    num_hidden_layers: int = 38
    intermediate_size: int = 8192
    moe_intermediate_size: int = 512
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int | None = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 130560
    max_position_embeddings: int = 131072
    rope_theta: float = 5000000.0
    rope_scaling: dict | None = None
    attention_bias: bool = False
    use_gated_attention: bool = True
    hidden_act: str = "silu"
    n_routed_experts: int = 320
    n_shared_experts: int = 1
    num_experts_per_tok: int = 32
    first_k_dense_replace: int = 1
    routed_scaling_factor: float = 3.66
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.head_dim is None:
            # Remote ``G9v3Config``: ``hidden_size // num_attention_heads``.
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.hidden_act != "silu":
            # The remote code routes ``hidden_act`` through ACT2FN; only the
            # released SwiGLU variant is ported, so refuse anything else
            # instead of silently running the wrong non-linearity.
            raise ValueError(
                f"unsupported hidden_act {self.hidden_act!r} (expected 'silu')"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a "
                f"multiple of num_key_value_heads ({self.num_key_value_heads})"
            )
        if not 0 <= self.first_k_dense_replace <= self.num_hidden_layers:
            raise ValueError(
                f"first_k_dense_replace ({self.first_k_dense_replace}) must be "
                f"within [0, num_hidden_layers={self.num_hidden_layers}]"
            )
        if self.first_k_dense_replace < self.num_hidden_layers:
            if not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
                raise ValueError(
                    f"num_experts_per_tok ({self.num_experts_per_tok}) must be "
                    f"within [1, n_routed_experts={self.n_routed_experts}]"
                )
            if self.n_group < 1 or self.n_routed_experts % self.n_group != 0:
                raise ValueError(
                    f"n_group ({self.n_group}) must divide n_routed_experts "
                    f"({self.n_routed_experts})"
                )
            if not 1 <= self.topk_group <= self.n_group:
                raise ValueError(
                    f"topk_group ({self.topk_group}) must be within [1, n_group={self.n_group}]"
                )


class Attention(nn.Module):
    """GQA attention with the optional per-head output gate.

    Mirrors ``G9v3Attention``: when ``use_gated_attention`` is set the
    ``q_proj`` width doubles and each head's ``2 * head_dim`` slice is
    ``[query | gate]``; the attention output is scaled by ``sigmoid(gate)``
    right before ``o_proj``.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        assert args.head_dim is not None  # resolved in ModelArgs.__post_init__
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.gated = args.use_gated_attention

        q_width = self.n_heads * self.head_dim * (2 if self.gated else 1)
        self.q_proj = nn.Linear(args.hidden_size, q_width, bias=args.attention_bias)
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=args.attention_bias
        )
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Any = None,
        cache: Any = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x)
        gate = None
        if self.gated:
            # (B, L, n_heads, 2 * head_dim) -> per-head [query | gate].
            q, gate = mx.split(q.reshape(B, L, self.n_heads, -1), 2, axis=-1)
            gate = gate.reshape(B, L, -1)
        queries = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if gate is not None:
            output = output * mx.sigmoid(gate)
        return self.o_proj(output)


class MLP(nn.Module):
    """Dense SwiGLU MLP (``G9v3MLP``); also the shared expert."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    """``G9v3TopkRouter`` on top of mlx-lm's ``group_expert_select``.

    ``weight`` and ``e_score_correction_bias`` are raw arrays (matching the
    checkpoint keys ``mlp.gate.weight`` / ``mlp.gate.e_score_correction_bias``)
    so the quantizer never touches the router.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.n_routed_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        # The reference computes the router logits in float32
        # (``F.linear(x.float(), weight.float())``); match it so expert
        # selection at the top-k margin agrees with transformers.
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        inds, scores = group_expert_select(
            logits,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )
        return inds, scores


class MoE(nn.Module):
    """``G9v3MoE``: routed experts (``SwitchGLU``) + always-on shared expert."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts
        )
        self.gate = MoEGate(args)
        self.n_shared_experts = args.n_shared_experts or 0
        if self.n_shared_experts:
            self.shared_experts = MLP(
                args.hidden_size, args.moe_intermediate_size * self.n_shared_experts
            )

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        # float32 accumulation over the selected experts, cast back to the
        # activation dtype — same as the reference ``moe()`` loop.
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.n_shared_experts:
            y = y + self.shared_experts(x)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp: MLP | MoE
        if layer_idx < args.first_k_dense_replace:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = MoE(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Any = None,
        cache: Any = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class G9v3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Any = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = G9v3Model(args)
        self.tie_word_embeddings = args.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Any = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = self.model(inputs, cache, input_embeddings)
        if self.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    def sanitize(self, weights):
        """Stack per-expert checkpoint tensors into the ``SwitchGLU`` layout.

        ``model.layers.N.mlp.experts.E.{gate,up,down}_proj.{weight|scales|biases}``
        (E = 0..n_routed_experts-1) becomes
        ``model.layers.N.mlp.switch_mlp.{gate,up,down}_proj.<k>`` with the
        expert axis first. An export that is already stacked (an MLX
        conversion produced by this module) passes through unchanged.
        """
        n_experts = self.args.n_routed_experts
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for kind in ("weight", "scales", "biases"):
                    if f"{prefix}.experts.0.{proj}.{kind}" not in weights:
                        continue
                    names = [
                        f"{prefix}.experts.{e}.{proj}.{kind}" for e in range(n_experts)
                    ]
                    missing = [name for name in names if name not in weights]
                    if missing:
                        raise ValueError(
                            f"g9v3 checkpoint is missing {len(missing)} expert "
                            f"tensor(s) for layer {layer_idx} {proj}.{kind}, "
                            f"first: {missing[0]}"
                        )
                    weights[f"{prefix}.switch_mlp.{proj}.{kind}"] = mx.stack(
                        [weights.pop(name) for name in names]
                    )
        if self.tie_word_embeddings:
            weights = {k: v for k, v in weights.items() if not k.startswith("lm_head.")}
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        """Default ``mlx_lm.convert`` recipe: routed experts at the requested
        bits, everything else (attention, dense + shared MLP, embeddings,
        ``lm_head``) at 8-bit / group 64.

        Measured on the released weights against the bf16 transformers
        reference (mean top-1 agreement over all prompt positions / mean KL
        on five fixed prompts; bf16 MLX itself scores 0.955 / 0.020):
        4-bit everywhere 0.801 / 0.312, 4-bit experts + 8-bit rest
        0.899 / 0.077, 8-bit everywhere 0.958 / 0.025. The attention
        projections (2 KV heads, gated ``q_proj``) are the
        quantization-sensitive part and, together with the other
        non-expert weights, are ~5% of the parameters, so the 8-bit tail
        costs ~2 GB on the 4-bit export. Same mechanism as ``gpt_oss`` /
        ``qwen3_next`` in mlx-lm (``quantize_model`` picks the model's
        predicate up when the caller passes none).
        """

        def predicate(path: str, _module: nn.Module) -> bool | dict[str, int]:
            if ".switch_mlp." in path:
                return True
            return {"group_size": 64, "bits": 8}

        return predicate

    @property
    def cast_predicate(self):
        """Keep the router's ``e_score_correction_bias`` in float32 through
        ``mlx_lm.convert``'s ``--dtype`` pass (which otherwise casts every
        floating parameter to the config's ``torch_dtype``, bf16 here).

        The checkpoint ships that bias as F32 on purpose: it only shifts the
        top-32 selection, so rounding it to bf16 changes which experts fire —
        measured as a 4-point drop in top-1 agreement on the exported model
        before this hook existed. Same hook and same rule as ``deepseek_v3``.
        """

        def predicate(path: str) -> bool:
            return "e_score_correction_bias" not in path

        return predicate
