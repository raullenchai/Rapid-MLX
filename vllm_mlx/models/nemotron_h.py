# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Apple Inc.
"""Compatibility vendor for heterogeneous Nemotron-H Puzzle checkpoints.

The installed mlx-lm 0.31.x ``nemotron_h`` implementation assumes every MoE
block has the same intermediate width and top-k.  NVIDIA Nemotron Puzzle
instead supplies those values in per-layer ``block_configs``.  This module
keeps the upstream Nemotron-H implementation intact for uniform checkpoints
and adds only the Puzzle plumbing from ml-explore/mlx-lm#1536.

It is registered only while the installed mlx-lm lacks ``block_configs``;
newer native implementations take precedence.  The implementation deliberately
does not include MTP: the commonly available 6-bit Puzzle artifact omits its
``mtp.*`` tensors, and self-speculative rollback for the hybrid SSM backbone is
a separate follow-up.

Upstream provenance: ml-explore/mlx-lm#1536, commits
aec9d8ecf90f3fc600be4b8f76b816a52c6a3944 and
77c06c84592b60015ac5ba17dd636b8fd2887746.
"""

import copy
from dataclasses import dataclass

import mlx.nn as nn

from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models import nemotron_h as _native  # noqa: E402


@dataclass
class ModelArgs(_native.ModelArgs):
    """Nemotron-H arguments plus Puzzle's per-layer MoE descriptions."""

    block_configs: list[dict] | None = None
    # Parse-only compatibility for Puzzle configs whose MTP tensors were
    # stripped during quantization. This vendor deliberately does not build
    # an MTP module; inherited sanitize() drops any stray ``mtp.*`` weights.
    num_nextn_predict_layers: int = 0
    mtp_layers_block_type: list[str] | None = None
    mtp_hybrid_override_pattern: list[str] | None = None
    mtp_block_configs: list[dict] | None = None


def _moe_layer_args(args: ModelArgs, block_cfg: dict | None) -> ModelArgs:
    """Return a shallow per-layer view with Puzzle's MoE dimensions."""
    if not block_cfg:
        return args
    layer_args = copy.copy(args)
    if block_cfg.get("moe_intermediate_size") is not None:
        layer_args.moe_intermediate_size = block_cfg["moe_intermediate_size"]
    if block_cfg.get("num_experts_per_tok") is not None:
        layer_args.num_experts_per_tok = block_cfg["num_experts_per_tok"]
    return layer_args


class MoEGate(_native.MoEGate):
    """Puzzle omits optional group-routing values; use identity defaults."""

    def __init__(self, config: ModelArgs):
        nn.Module.__init__(self)
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor or 1.0
        self.n_group = config.n_group or 1
        self.topk_group = config.topk_group or 1
        self.weight = _native.mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = _native.mx.zeros((self.n_routed_experts,))


class NemotronHMoE(nn.Module):
    """Native MoE with the Puzzle-safe gate; all other behavior is unchanged."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.moe_latent_size = config.moe_latent_size
        expert_input_dim = config.moe_latent_size or config.hidden_size
        self.switch_mlp = _native.SwitchMLP(
            expert_input_dim,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=nn.ReLU2(),
        )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = _native.NemotronHMLP(
                config, intermediate_size=config.moe_shared_expert_intermediate_size
            )
        if config.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(
                config.hidden_size, config.moe_latent_size, bias=config.mlp_bias
            )
            self.fc2_latent_proj = nn.Linear(
                config.moe_latent_size, config.hidden_size, bias=config.mlp_bias
            )

    def __call__(self, x):
        residuals = x
        inds, scores = self.gate(x)
        if self.moe_latent_size is not None:
            x = self.fc1_latent_proj(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.moe_latent_size is not None:
            y = self.fc2_latent_proj(y)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(residuals)
        return y


class NemotronHBlock(_native.NemotronHBlock):
    """Native block constructor with Puzzle's MoE implementation for E blocks."""

    def __init__(self, args: ModelArgs, block_type: str):
        nn.Module.__init__(self)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.block_type = block_type
        if block_type == "M":
            self.mixer = _native.NemotronHMamba2Mixer(args)
        elif block_type == "*":
            self.mixer = _native.NemotronHAttention(args)
        elif block_type == "-":
            self.mixer = _native.NemotronHMLP(args)
        elif block_type == "E":
            self.mixer = NemotronHMoE(args)


class NemotronHModel(_native.NemotronHModel):
    """Native backbone with heterogeneous construction for MoE layers."""

    def __init__(self, args: ModelArgs):
        nn.Module.__init__(self)
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        pattern = args.hybrid_override_pattern
        block_configs = args.block_configs or [None] * len(pattern)
        if len(block_configs) != len(pattern):
            raise ValueError("block_configs must align with hybrid_override_pattern")
        self.layers = [
            NemotronHBlock(_moe_layer_args(args, cfg) if kind == "E" else args, kind)
            for kind, cfg in zip(pattern, block_configs)
        ]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.fa_idx = 0
        self.ssm_idx = 0
        for kind in pattern:
            if kind == "*":
                break
            if kind == "M":
                self.fa_idx += 1
        for kind in pattern:
            if kind == "*":
                self.ssm_idx += 1
            elif kind == "M":
                break


class Model(_native.Model):
    """Puzzle-aware model wrapper retaining native cache and sanitize paths."""

    def __init__(self, args: ModelArgs):
        nn.Module.__init__(self)
        self.args = args
        self.backbone = NemotronHModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.model_type = args.model_type

    @property
    def quant_predicate(self):
        if self.model_type != "nemotron_h_puzzle":
            return lambda _path, _module: True
        # Upstream #1536: Puzzle's output projection is too sensitive to
        # affine low-bit quantization; retain its checkpoint precision.
        return lambda path, _module: path != "lm_head"
