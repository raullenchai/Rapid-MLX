# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Apple Inc.
"""Sarvam-105B MLA-MoE compatibility model.

Sarvam-105B uses the ``sarvam_mla`` configuration schema but its attention,
MoE routing, latent-cache layout, and checkpoint sanitization follow
DeepSeek-V3.  This intentionally thin adapter remaps only its differing
configuration names, leaving the mature DeepSeek-V3 implementation to own the
forward, cache, and weight-sanitization paths.
"""

from dataclasses import dataclass

# MUST run before any mlx_lm import: mlx_lm's package initializer captures a
# thread-local MLX stream, which is unusable on M5 single-stream GPUs (#404).
from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.deepseek_v3 import Model as DeepseekV3ForCausalLM
from mlx_lm.models.deepseek_v3 import ModelArgs as _V3Args


@dataclass
class ModelArgs(_V3Args):
    """Map Sarvam's config keys onto MLX's DeepSeek-V3 model arguments."""

    model_type: str = "sarvam_mla"
    num_experts: int | None = None
    num_shared_experts: int | None = None
    # Sarvam has direct q_proj rather than DeepSeek-V3's q_a/q_b LoRA path.
    q_lora_rank: int | None = None
    # None distinguishes an absent key from an explicit ``1`` (no group limit).
    n_group: int | None = None
    topk_group: int | None = None
    # Retain Sarvam-only config fields so BaseModelArgs.from_dict accepts them.
    q_head_dim: int = 192
    head_dim: int = 576
    use_qk_norm: bool = False
    moe_router_enable_expert_bias: bool = True
    default_theta: float = 10000.0

    def __post_init__(self):
        if self.num_experts is not None:
            self.n_routed_experts = self.num_experts
        if self.num_shared_experts is not None:
            self.n_shared_experts = self.num_shared_experts

        # Sarvam's published config omits these group-routing controls. Its
        # reference gate defaults to routed_experts / 8 groups and top-2 groups.
        # Do not replace an explicit ``1``: it means no group restriction.
        n_routed = self.n_routed_experts or self.num_experts or 128
        if self.n_group is None:
            self.n_group = n_routed // 8
        if self.topk_group is None:
            self.topk_group = 2


class Model(DeepseekV3ForCausalLM):
    """Sarvam configuration wrapper over MLX's DeepSeek-V3 model."""

    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.model_type = config.model_type
