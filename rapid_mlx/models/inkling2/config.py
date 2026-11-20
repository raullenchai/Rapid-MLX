# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
    "TextArgs",
]


@dataclass
class TextArgs(BaseModelArgs):
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    unpadded_vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    swa_head_dim: int
    sliding_window_size: int
    local_layer_ids: list[int]
    d_rel: int
    rel_extent: int
    log_scaling_n_floor: int
    log_scaling_alpha: float
    rms_norm_eps: float
    qk_l2_clip_threshold: float
    rel_l2_clip_threshold: float
    sconv_kernel_size: int
    n_layers_per_attn_res_block: int
    n_routed_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    mlp_gate_softcap: float
    mlp_up_softcap: float
    route_scale: float
    final_logit_softcapping: float
    loop_config: dict | None = None

    def __post_init__(self):
        loop = self.loop_config or {}
        self.loop_core = range(loop.get("core_start", 0), loop.get("core_end", 0))


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    vision_config: dict | None = None
    image_token_id: int = 200054

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config)
        self.vision = None
