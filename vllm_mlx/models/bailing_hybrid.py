# SPDX-License-Identifier: Apache-2.0
"""Vendored inclusionAI Ling 3.0 backbone (``model_type: bailing_hybrid``).

**Why vendor:** Ling-3.0-tiny (7.9B total / 1.3B active, MIT) landed with
day-0 SGLang/vLLM support and no MLX story beyond an unmerged mlx-lm PR:

- mlx-lm: no ``bailing_hybrid`` (checked 0.31.3 + upstream main,
  2026-08-10); PR ml-explore/mlx-lm#1227 (Ling-2.6-era shape) open and
  unreviewed since 2026-07-06 — the same situation ``hy_v3`` was vendored
  out of.
- transformers: no native port; the HF repo ships remote code that
  imports ``fla`` (triton) at module level, so it cannot even run on
  macOS.

One module serves the whole family: Ling-3.0-tiny, Ling-3.0-flash and
Ling-2.6-flash all declare ``bailing_hybrid``.

**References (math verified against ALL of):**

- HF remote code ``modeling_bailing_moe_v3.py`` — authoritative
- fla kernels (``fla/ops/kda``) — exact ``safe_gate``/``lower_bound``
  semantics (see ``_kda_gate``)
- mlx-lm ``kimi_linear.py`` — MLX KDA idiom (KDA is Kimi's design;
  Ling 3.0 reuses it with a safe-gate variant); ``ShortConv1d`` and the
  group expert select are adapted from it
- mlx-lm PR #1227 — Ling-2.6 MLX port (cross-check)

Architecture (Ling-3.0-tiny values):

- 24 layers in ``[KDA ×3, MLA]`` groups (``layer_group_size`` 4; the
  trailing remainder layers are MLA); layer 0 is a dense MLP
  (``first_k_dense_replace`` 1), all later layers are sparse MoE
- KDA (Kimi Delta Attention, linear): q/k/v projections through causal
  short convolutions (kernel 4, silu), per-head decay gate with the
  SAFE-GATE law ``g = lower_bound * sigmoid(exp(A_log) * (f(x) +
  dt_bias))`` (fla ``USE_LOWER_BOUND`` branch — NOT the softplus law
  mlx-lm's ``compute_g`` implements), delta-rule state update, gated
  RMSNorm output
- MLA (DeepSeek-style): q LoRA 256 / kv LoRA 512, 128 nope + 64 rope
  dims, interleaved RoPE theta 6e6, plus a V3-only HEAD-WISE output
  gate (``g_proj``: hidden → n_heads, output scaled by sigmoid); the
  output projection is named ``dense`` in the checkpoint
- MoE: 128 routed experts top-8 + 1 shared, sigmoid scores with
  ``noaux_tc`` expert-bias selection, 8 groups pick 4
  (``topk_group``), normalized top-k probs scaled by 2.5
- untied ``lm_head``; embeddings under ``model.word_embeddings``

**Registration:** installed as ``sys.modules["mlx_lm.models.bailing_hybrid"]``
by ``vllm_mlx.utils.tokenizer._register_vendored_archs`` (same trick as
``deepseek_v4`` / ``hy_v3`` / ``muse_glimmer``), with a ``find_spec``
probe that defers to native mlx-lm support the moment upstream ships it.

**Sync policy:** when mlx-lm merges #1227's successor with V3 support,
diff and delete this file.
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
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "bailing_hybrid"
    hidden_size: int = 1536
    num_hidden_layers: int = 24
    intermediate_size: int = 4608
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 157184
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = False
    # ---- layer schedule -------------------------------------------------
    layer_group_size: int = 4
    first_k_dense_replace: int = 1
    # ---- KDA (linear attention) ----------------------------------------
    short_conv_kernel_size: int = 4
    no_kda_lora: bool = True
    kda_safe_gate: bool = True
    kda_lower_bound: float = -5.0
    # ---- MLA -----------------------------------------------------------
    q_lora_rank: int | None = 256
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 6000000.0
    rope_interleave: bool = True
    rope_scaling: dict | None = None
    use_qkv_bias: bool = False
    gated_attention_proj_granularity_type: str | None = "head_wise"
    # ---- MoE -----------------------------------------------------------
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    moe_intermediate_size: int = 512
    moe_shared_expert_intermediate_size: int = 512
    n_group: int = 8
    topk_group: int = 4
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    moe_router_enable_expert_bias: bool = True

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def is_mla_layer(self, idx: int) -> bool:
        """Softmax (MLA) layers sit last in each group; any remainder
        layers past the final full group are MLA too (torch ref
        ``BailingMoeV3DecoderLayer.__init__``)."""
        g = self.layer_group_size
        full = self.num_hidden_layers // g * g
        return (idx + 1) % g == 0 or idx >= full


class ShortConv1d(nn.Module):
    """Causal depthwise conv with silu and a rolling cache.

    Adapted from mlx-lm ``kimi_linear.ShortConv1d`` (same wire: the
    checkpoint stores ``*_conv1d.weight`` of shape [channels, ksize, 1]).
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, bias=False
        )

    def __call__(
        self,
        x: mx.array,
        state: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        B, T, C = x.shape
        if state is None:
            state = mx.zeros((B, self.kernel_size - 1, C), dtype=x.dtype)
        conv_input = mx.concatenate([state, x], axis=1)
        out = nn.silu(self.conv(conv_input))
        if self.kernel_size == 1:
            # ``-(k-1)`` would be ``-0`` and retain the WHOLE input,
            # growing the cache every step (codex r3 #3).
            new_state = conv_input[:, :0, :]
        else:
            new_state = conv_input[:, -(self.kernel_size - 1) :, :]
        return out, new_state


def _kda_gate(
    f: mx.array,
    A_log: mx.array,  # noqa: N803 — the checkpoint's parameter name
    dt_bias: mx.array,
    *,
    safe_gate: bool,
    lower_bound: float,
) -> mx.array:
    """Per-channel log-decay gate.

    fla ``fused_recurrent_kda_fwd_kernel`` (L166-171):

        g_pre = f + dt_bias
        USE_LOWER_BOUND: g = lower_bound * sigmoid(exp(A_log) * g_pre)
        else:            g = -exp(A_log) * softplus(g_pre)

    Ling 3.0 ships ``kda_safe_gate=True`` / ``kda_lower_bound=-5`` — the
    sigmoid law, clamping the log-decay into (lower_bound, 0). Computed
    in float32 like the kernel.
    """
    f = f.astype(mx.float32) + dt_bias.astype(mx.float32)
    a = mx.exp(A_log.astype(mx.float32))
    if safe_gate:
        return lower_bound * mx.sigmoid(a[..., None] * f)
    return -a[..., None] * nn.softplus(f)


def _kda_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g_log: mx.array,
    beta: mx.array,
    state: mx.array | None,
) -> tuple[mx.array, mx.array]:
    """Run the KDA delta-rule recurrence via mlx-lm's fused gated-delta
    path (the same Metal kernel / ops pair qwen3-next and kimi_linear
    ship on), with our precomputed safe-gate decay.

    ``gated_delta_ops``/``gated_delta_kernel`` take the decay as a
    MULTIPLIER (``compute_g`` returns ``exp(...)``), vectorized per
    channel ``[B, T, H, Dk]`` — so the log-space safe gate is
    exponentiated here. A per-token Python loop over the full prompt
    (the naive fla reference) is prohibitively slow at 131K context
    (codex r1 #1); the kernel path chunks the recurrence on Metal.
    """
    from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops

    g = mx.exp(g_log)
    if state is None:
        B = q.shape[0]
        state = mx.zeros((B, v.shape[-2], v.shape[-1], k.shape[-1]), dtype=mx.float32)
    # The Metal kernel tiles Dk in 32-lane strips (``n_per_t = Dk/32``);
    # head dims below 32 would generate a zero-length array and fail the
    # metallib build. Production checkpoints use head_dim 128; tiny test
    # configs take the ops path.
    use_kernel = (
        gated_delta_kernel is not None
        and k.shape[-1] % 32 == 0
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )
    fn = gated_delta_kernel if use_kernel else gated_delta_ops
    return fn(q, k, v, g, beta, state, None)


class BailingKDA(nn.Module):
    """Kimi Delta Attention with the Ling V3 safe gate."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.proj_dim = self.num_heads * self.head_dim
        self.conv_kernel = args.short_conv_kernel_size
        self.safe_gate = args.kda_safe_gate
        self.lower_bound = float(args.kda_lower_bound)
        self.no_kda_lora = args.no_kda_lora
        self.scale = float(self.head_dim) ** -0.5

        hidden = args.hidden_size
        # q/k/v (and, on no_kda_lora checkpoints, f/g) are independent
        # full-rank projections in the checkpoint; decode is dominated
        # by kernel-launch count at B=T=1, so they are served from
        # row-concatenated fused weights (5 GEMVs + 3 depthwise convs
        # -> 2 GEMVs + 1 conv, measured ~0.8 ms/token across the 18 KDA
        # layers). Row concatenation is mathematically identical and —
        # because every quantization scheme here packs per row — the
        # fused quantized bytes are the source rows verbatim.
        # ``sanitize`` performs the concat at load time.
        self.qkv_proj = nn.Linear(hidden, 3 * self.proj_dim, bias=False)
        self.qkv_conv1d = ShortConv1d(3 * self.proj_dim, self.conv_kernel)

        if self.no_kda_lora:
            self.fg_proj = nn.Linear(hidden, 2 * self.proj_dim, bias=False)
        else:
            # Ling-2.6 / flash variants use LoRA pairs. The bottleneck is
            # head_dim BY DESIGN — the authoritative reference hard-codes
            # it the same way (modeling_bailing_moe_v3.BailingMoeV3
            # KimiDeltaAttention: ``f_a_proj = Linear(hidden_size,
            # head_dim)``) and no bailing config revision publishes a
            # separate f/g LoRA-rank field to wire.
            self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.f_b_proj = nn.Linear(self.head_dim, self.proj_dim, bias=False)
            self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, self.proj_dim, bias=False)

        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.A_log = mx.zeros((self.num_heads,))
        self.dt_bias = mx.zeros((self.proj_dim,))
        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.proj_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, T, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            qkv_state, ssm_state = cache[0], cache[1]
        else:
            qkv_state = ssm_state = None

        qkv_conv, qkv_state = self.qkv_conv1d(self.qkv_proj(x), qkv_state)
        if cache is not None:
            cache[0] = qkv_state

        q, k, v = (
            t.reshape(B, T, self.num_heads, self.head_dim)
            for t in mx.split(qkv_conv, 3, axis=-1)
        )

        # fla applies L2-norm to q/k in-kernel plus scale=d^-0.5. Use the
        # exact l2norm form (x / (||x|| + eps)) so the eps placement
        # matches the kernel bit-for-bit; the kimi_linear rms_norm
        # folding trick puts eps under the sqrt and costs ~2e-4 parity.
        qf = q.astype(mx.float32)
        kf = k.astype(mx.float32)
        q = self.scale * qf / (mx.linalg.norm(qf, axis=-1, keepdims=True) + 1e-6)
        k = kf / (mx.linalg.norm(kf, axis=-1, keepdims=True) + 1e-6)

        if self.no_kda_lora:
            f, gate = mx.split(self.fg_proj(x), 2, axis=-1)
        else:
            f = self.f_b_proj(self.f_a_proj(x))
            gate = self.g_b_proj(self.g_a_proj(x))
        f = f.reshape(B, T, self.num_heads, self.head_dim)
        g = _kda_gate(
            f,
            self.A_log,
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
        )
        beta = mx.sigmoid(self.b_proj(x).astype(mx.float32))

        out, ssm_state = _kda_update(q, k, v, g, beta, ssm_state)
        if cache is not None:
            cache[1] = ssm_state

        gate = gate.reshape(B, T, self.num_heads, self.head_dim)
        out = self.o_norm(out.astype(dtype)) * mx.sigmoid(gate)
        return self.o_proj(out.reshape(B, T, -1))


class BailingMLA(nn.Module):
    """DeepSeek-style MLA plus the V3 head-wise output gate."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.q_lora_rank = args.q_lora_rank
        self.scale = self.qk_head_dim**-0.5
        self.gate_kind = args.gated_attention_proj_granularity_type

        hidden = args.hidden_size
        bias = args.use_qkv_bias
        if self.q_lora_rank is None:
            # bias=False BY REFERENCE: the authoritative torch modeling
            # hard-codes the direct q_proj without bias (use_qkv_bias
            # applies only to the LoRA a-projections and kv_a/dense) —
            # mirroring it exactly, not an omission.
            self.q_proj = nn.Linear(
                hidden, self.num_heads * self.qk_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(hidden, self.q_lora_rank, bias=bias)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, self.kv_lora_rank + self.qk_rope_head_dim, bias=bias
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        if self.gate_kind == "head_wise":
            self.g_proj = nn.Linear(hidden, self.num_heads, bias=False)
        elif self.gate_kind == "element_wise":
            self.g_proj = nn.Linear(
                hidden, self.num_heads * self.v_head_dim, bias=False
            )
        # Checkpoint names the output projection ``dense``.
        self.dense = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=bias)
        # rope_interleave=True (Ling 3.0) is MLX's ``traditional`` layout
        # (consecutive-pair rotation); wire the flag instead of pinning
        # it, and refuse silently-wrong serving for scaled-rope exports
        # we have not verified (codex r1 #3).
        if args.rope_scaling:
            raise NotImplementedError(
                "bailing_hybrid: rope_scaling is not supported by the "
                f"vendored backbone yet (got {args.rope_scaling!r})"
            )
        self.rope = nn.RoPE(
            self.qk_rope_head_dim,
            traditional=bool(args.rope_interleave),
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, L, self.num_heads, self.qk_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed = self.kv_a_proj_with_mqa(x)
        kv_latent, k_pe = mx.split(compressed, [self.kv_lora_rank], axis=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)

        kv = self.kv_b_proj(kv_latent)
        kv = kv.reshape(
            B, L, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(0, 2, 1, 3)
        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset=offset)
        k_pe = self.rope(k_pe, offset=offset)
        k_pe = mx.broadcast_to(k_pe, (B, self.num_heads, L, self.qk_rope_head_dim))

        queries = mx.concatenate([q_nope, q_pe], axis=-1)
        keys = mx.concatenate([k_nope, k_pe], axis=-1)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        out = scaled_dot_product_attention(
            queries,
            values=values,
            keys=keys,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.gate_kind == "head_wise":
            gate = mx.sigmoid(self.g_proj(x))  # [B, L, H]
            out = out.reshape(B, L, self.num_heads, self.v_head_dim)
            out = out * gate[..., None]
            out = out.reshape(B, L, -1)
        elif self.gate_kind == "element_wise":
            out = out * mx.sigmoid(self.g_proj(x))
        return self.dense(out)


class BailingMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class BailingGate(nn.Module):
    """Sigmoid-score router with noaux_tc expert-bias group selection.

    DeepSeek-V3 routing law (torch ref ``BailingMoeV3Gate``): selection
    scores = sigmoid(logits) + expert_bias, grouped into ``n_group``;
    the ``topk_group`` best groups (by sum of each group's top-2
    selection scores) survive; the top-k experts within the surviving
    groups are chosen by selection score; the WEIGHTS are the raw
    sigmoid scores of the chosen experts (bias excluded), normalized if
    ``norm_topk_prob`` and scaled by ``routed_scaling_factor``.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        if args.moe_router_enable_expert_bias:
            self.expert_bias = mx.zeros((args.num_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        a = self.args
        scores = mx.sigmoid((x @ self.weight.T).astype(mx.float32))
        select = scores
        if a.moe_router_enable_expert_bias:
            select = scores + self.expert_bias

        B = select.shape[:-1]
        k_drop = a.n_group - a.topk_group
        if k_drop > 0:
            grouped = select.reshape(*B, a.n_group, a.num_experts // a.n_group)
            # Group score = sum of the top-2 selection scores in the group.
            top2 = mx.topk(grouped, 2, axis=-1)
            group_scores = top2.sum(axis=-1)
            drop = mx.argpartition(group_scores, kth=k_drop - 1, axis=-1)[..., :k_drop]
            # ``put_along_axis`` broadcasts the trailing size-1 index dim
            # across the experts-per-group axis (numpy semantics), so the
            # single write masks EVERY expert slot of each dropped group —
            # unlike torch scatter, no expand is needed. Pinned by
            # ``test_gate_group_drop_masks_whole_group`` and by the
            # reference parity run (1.5e-6 on a drop-path config).
            masked = mx.put_along_axis(
                grouped,
                mx.expand_dims(drop, -1),
                mx.array(-float("inf"), grouped.dtype),
                axis=-2,
            )
            select = masked.reshape(*B, a.num_experts)
        # topk_group == n_group keeps every group — nothing to drop
        # (codex r3 #1: argpartition(kth=-1) would fault).

        k = a.num_experts_per_tok
        idx = mx.argpartition(-select, kth=k - 1, axis=-1)[..., :k]
        w = mx.take_along_axis(scores, idx, axis=-1)
        if a.norm_topk_prob:
            w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
        w = w * a.routed_scaling_factor
        return idx, w


class BailingSparseMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = BailingGate(args)
        self.experts = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_experts = BailingMLP(
            args,
            args.moe_shared_expert_intermediate_size * args.num_shared_experts,
        )

    def __call__(self, x: mx.array) -> mx.array:
        idx, w = self.gate(x)
        routed = self.experts(x, idx)
        out = (routed * w[..., None].astype(routed.dtype)).sum(axis=-2)
        return out + self.shared_experts(x)


class BailingDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_mla = args.is_mla_layer(layer_idx)
        if self.is_mla:
            self.attention = BailingMLA(args)
        else:
            self.attention = BailingKDA(args, layer_idx)
        if layer_idx >= args.first_k_dense_replace:
            self.mlp = BailingSparseMoE(args)
        else:
            self.mlp = BailingMLP(args, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        h = x + self.attention(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class BailingModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            BailingDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.first_mla_idx = next(
            i for i in range(args.num_hidden_layers) if args.is_mla_layer(i)
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = (
            input_embeddings
            if input_embeddings is not None
            else self.word_embeddings(inputs)
        )
        if cache is None:
            cache = [None] * len(self.layers)
        mla_mask = create_attention_mask(h, cache[self.first_mla_idx])
        for layer, c in zip(self.layers, cache):
            mask = mla_mask if layer.is_mla else None
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = BailingModel(args)
        self.tie_word_embeddings = args.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = self.model(inputs, cache, input_embeddings)
        if self.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(h)
        return self.lm_head(h)

    def sanitize(self, weights):
        """Stack per-expert weights into SwitchGLU layout and drop any
        MTP head (flash/2.6 ship ``num_nextn_predict_layers`` layers we
        do not serve)."""
        weights = {
            k: v for k, v in weights.items() if ".mtp_" not in k and "mtp." not in k
        }
        n_layers = self.args.num_hidden_layers
        for li in range(n_layers):
            prefix = f"model.layers.{li}"
            if f"{prefix}.mlp.experts.0.gate_proj.weight" not in weights and (
                f"{prefix}.mlp.experts.0.gate_proj.scales" not in weights
            ):
                continue
            for m in ("gate_proj", "up_proj", "down_proj"):
                for part in ("weight", "scales", "biases"):
                    key0 = f"{prefix}.mlp.experts.0.{m}.{part}"
                    if key0 not in weights:
                        continue
                    joined = mx.stack(
                        [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{part}")
                            for e in range(self.args.num_experts)
                        ]
                    )
                    weights[f"{prefix}.mlp.experts.{m}.{part}"] = joined
        # Conv weights: the checkpoint stores them at ``*_conv1d.weight``
        # in torch depthwise layout [C, 1, ksize] (or flat [C, ksize]);
        # our ShortConv1d nests an nn.Conv1d at ``.conv`` expecting
        # [C, ksize, 1].
        for k in list(weights):
            if k.endswith("_conv1d.weight"):
                w = weights.pop(k)
                if w.ndim == 2:
                    w = w[..., None]
                elif w.ndim == 3 and w.shape[1] == 1:
                    w = w.swapaxes(1, 2)
                weights[k[: -len(".weight")] + ".conv.weight"] = w

        # KDA fused serving layout: concatenate the checkpoint's separate
        # q/k/v projections (+ their convs) and, on no_kda_lora
        # checkpoints, f/g into single row-fused tensors. Row concat is
        # exact for plain tensors AND for every per-row-packed quantized
        # part (weight/scales/biases), so this is a pure relayout.
        # Idempotent: already-fused checkpoints have no ``q_proj`` keys.
        def _concat(dst: str, srcs: list[str]) -> None:
            for part in ("weight", "scales", "biases"):
                keys = [f"{s}.{part}" for s in srcs]
                if keys[0] not in weights:
                    continue
                weights[f"{dst}.{part}"] = mx.concatenate(
                    [weights.pop(k) for k in keys], axis=0
                )

        for li in range(n_layers):
            attn = f"model.layers.{li}.attention"
            # KDA layers ship all three of q/k/v; an MLA layer with
            # q_lora_rank=None ships a lone q_proj (kv via kv_a/kv_b)
            # and must not be touched.
            if not all(
                f"{attn}.{t}_proj.weight" in weights
                or f"{attn}.{t}_proj.scales" in weights
                for t in "qkv"
            ):
                continue
            _concat(f"{attn}.qkv_proj", [f"{attn}.{t}_proj" for t in "qkv"])
            _concat(
                f"{attn}.qkv_conv1d.conv",
                [f"{attn}.{t}_conv1d.conv" for t in "qkv"],
            )
            # f/g are full-rank (and fusable) only on no_kda_lora
            # checkpoints; LoRA-pair variants keep f_a/f_b/g_a/g_b.
            if f"{attn}.f_proj.weight" in weights or f"{attn}.f_proj.scales" in weights:
                _concat(f"{attn}.fg_proj", [f"{attn}.f_proj", f"{attn}.g_proj"])
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # KDA layers hold [fused qkv conv state, KDA ssm state].
        return [
            KVCache() if layer.is_mla else ArraysCache(size=2)
            for layer in self.model.layers
        ]

    @property
    def quant_predicate(self):
        def predicate(path, module):
            # Short-conv weights stay fp — tiny tensors, and quantized
            # depthwise conv has no parity coverage (codex r3 #2).
            if "_conv1d" in path:
                return False
            # Router quality is precision-critical (deepseek-v3
            # precedent): keep the gate at 8-bit.
            if "mlp.gate" in path and "gate_proj" not in path:
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
