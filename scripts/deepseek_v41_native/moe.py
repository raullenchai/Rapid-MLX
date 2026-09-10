"""V4.1 MoE: sqrt-softplus scored routing over 384 experts, top-6, one shared expert.

Hash routing is gone (a V4 feature); every MoE layer scores. What remains
non-standard:

* ``sqrt(softplus(x))`` scoring;
* **the selection bias does not reach the routing weights** — scores are read
  before the bias is added; the bias only reorders the top-k. The checkpoint
  carries a second bias, ``gate.bias_vl``, selected for tokens inside image
  spans (training's ``noaux_tc_for_vl``). This text-only runtime always uses
  ``gate.bias`` but keeps ``bias_vl`` loaded so the checkpoint round-trips;
* top-k weights are normalized by ``sum + 1e-20`` (not ``norm_eps``) and scaled
  by ``routed_scaling_factor`` 1.5;
* clamped SwiGLU (limit 10): ``up`` clamped two-sided, ``gate`` upper-only.

Routing weights are applied after ``down_proj`` instead of before (one scalar per
token — linear, so identical), which lets the batched SwitchGLU gather-matmul
replace the reference's per-expert Python loop.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from vllm_mlx import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402

from .config import ModelArgs


class ClampedSwiGLU(nn.Module):
    """Called by SwitchGLU as activation(x_up, x_gate)."""

    def __init__(self, limit: float = 0.0):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        x = x.astype(mx.float32)
        gate = gate.astype(mx.float32)
        if self.limit > 0:
            x = mx.clip(x, -self.limit, self.limit)
            gate = mx.minimum(gate, self.limit)
        return nn.silu(gate) * x


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = mx.zeros((args.n_routed_experts, args.dim))
        self.bias = mx.zeros((args.n_routed_experts,), dtype=mx.float32)
        self.bias_vl = mx.zeros((args.n_routed_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        scores = (
            x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        ) / self.gate_temp
        if self.score_func == "softmax":
            scores = mx.softmax(scores, axis=-1)
        elif self.score_func == "sigmoid":
            scores = mx.sigmoid(scores)
        else:  # sqrtsoftplus
            scores = mx.sqrt(nn.softplus(scores))

        # the bias picks experts but does not scale them
        biased = scores + self.bias
        indices = mx.argpartition(-biased, self.topk - 1, axis=-1)[..., : self.topk]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices


class SharedExpert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.limit = limit

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        gate = self.w1(x).astype(mx.float32)
        up = self.w3(x).astype(mx.float32)
        if self.limit > 0:
            up = mx.clip(up, -self.limit, self.limit)
            gate = mx.minimum(gate, self.limit)
        h = nn.silu(gate) * up
        return self.w2(h.astype(dtype))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.gate = Gate(args)
        self.experts = SwitchGLU(
            args.dim,
            args.moe_inter_dim,
            args.n_routed_experts,
            activation=ClampedSwiGLU(args.swiglu_limit),
            bias=False,
        )
        self.shared_experts = SharedExpert(
            args.dim, args.moe_inter_dim, args.swiglu_limit
        )

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        xf = x.reshape(-1, self.dim)
        weights, indices = self.gate(xf)
        y = self.experts(xf, indices)  # [tokens, topk, dim]
        y = mx.sum(y.astype(mx.float32) * weights[..., None], axis=-2)
        y = y + self.shared_experts(xf).astype(mx.float32)
        return y.reshape(shape).astype(x.dtype)
