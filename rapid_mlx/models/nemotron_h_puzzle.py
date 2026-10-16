# SPDX-License-Identifier: MIT
# ruff: noqa: N803, E741
# MIT License
#
# Copyright © 2023 Apple Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Copyright © 2025 Apple Inc.
# SSM attention attribution: https://github.com/cartesia-ai/edge
# Adapted from ml-explore/mlx-lm PR #1535, commit
# 0f88e167aaa6267ca3ec36e024da4e08c249d92f (nemotron_h.py and ssm.py).

"""Text-only heterogeneous Puzzle backbone.

Conversion must write ``text_config(config)`` as the output config. This keeps
vision routing out of the runtime and leaves already converted text artifacts
loadable through the ordinary model loader. Sanitization accepts both original
omni weights and converted text weights.
"""

from copy import copy
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .. import _mlx_compat

_mlx_compat.install()

from mlx_lm.models import nemotron_h as base  # noqa: E402
from mlx_lm.models.base import (  # noqa: E402
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
)
from mlx_lm.models.ssm import _ssm_kernel, segsum  # noqa: E402
from mlx_lm.models.switch_layers import SwitchMLP  # noqa: E402


def text_config(config):
    """Extract a standalone language config without executing remote code."""
    config = dict(config)
    if config.get("model_type") == "nemotron_h_omni":
        config = dict(config["llm_config"])
    if config.get("model_type") != "nemotron_h_puzzle":
        raise ValueError("Expected nemotron_h_puzzle language config")
    config.pop("auto_map", None)
    return config


@dataclass()
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_bias: bool
    mamba_num_heads: int
    mamba_head_dim: int
    mamba_proj_bias: bool
    ssm_state_size: int
    conv_kernel: int
    n_groups: int
    mlp_bias: bool
    layer_norm_epsilon: float
    use_bias: bool
    use_conv_bias: bool
    hybrid_override_pattern: list[str] | None = None
    layers_block_type: list[str] | None = None
    head_dim: int | None = None
    moe_intermediate_size: int | None = None
    moe_shared_expert_intermediate_size: int | None = None
    moe_latent_size: int | None = None
    n_group: int | None = None
    n_routed_experts: int | None = None
    n_shared_experts: int | None = None
    topk_group: int | None = None
    num_experts_per_tok: int | None = None
    norm_topk_prob: bool | None = None
    routed_scaling_factor: float | None = None
    time_step_limit: tuple[float, float] | None = None
    time_step_min: float | None = None
    time_step_max: float | None = None
    block_configs: list[dict[str, Any]] | None = None

    # Map from layers_block_type names to single-char pattern codes
    _block_type_to_char = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}

    @classmethod
    def from_dict(cls, params):
        params = text_config(params)
        if params.get("num_hidden_layers") is None:
            blocks = params.get("block_configs") or params.get("layers_block_type")
            if blocks is not None:
                params["num_hidden_layers"] = len(blocks)
        return super().from_dict(params)

    def __post_init__(self):
        if self.time_step_limit is not None:
            self.time_step_limit = tuple(
                float(value["__float__"]) if isinstance(value, dict) else float(value)
                for value in self.time_step_limit
            )
        if self.time_step_limit is None:
            # The reference Nemotron-H Puzzle implementation clamps dt only from
            # below (its upper clamp to time_step_max is disabled), and its
            # default limit is (0.0, inf). time_step_max is an initialization
            # hyperparameter for dt_bias, not an inference bound, so the
            # fallback upper bound is intentionally infinite.
            lower_limit = (
                self.time_step_min
                if self.model_type == "nemotron_h_puzzle"
                and self.time_step_min is not None
                else 0.0
            )
            self.time_step_limit = (lower_limit, float("inf"))

        if self.block_configs is not None:
            if len(self.block_configs) != self.num_hidden_layers:
                raise ValueError(
                    "block_configs must contain one entry per hidden layer: "
                    f"got {len(self.block_configs)} entries for "
                    f"{self.num_hidden_layers} layers"
                )
            block_types = []
            for layer_idx, config in enumerate(self.block_configs):
                block_type = config.get("block_type")
                if block_type not in self._block_type_to_char:
                    raise ValueError(
                        f"Puzzle block {layer_idx} has unsupported block_type "
                        f"{block_type!r}; expected one of "
                        f"{sorted(self._block_type_to_char)}"
                    )
                block_types.append(block_type)
                if block_type == "moe":
                    missing = {
                        "moe_intermediate_size",
                        "num_experts_per_tok",
                    } - config.keys()
                    if missing:
                        raise ValueError(
                            f"Puzzle MoE block {layer_idx} is missing {sorted(missing)}"
                        )
            pattern = [self._block_type_to_char[t] for t in block_types]
            if (
                self.hybrid_override_pattern is not None
                and list(self.hybrid_override_pattern) != pattern
            ):
                raise ValueError("block_configs conflicts with hybrid_override_pattern")
            self.hybrid_override_pattern = pattern
            self.layers_block_type = block_types

        # Normalize to hybrid_override_pattern (single-char list)
        if self.hybrid_override_pattern is None and self.layers_block_type is not None:
            self.hybrid_override_pattern = [
                self._block_type_to_char[t] for t in self.layers_block_type
            ]
        if self.hybrid_override_pattern is not None:
            self.num_hidden_layers = len(self.hybrid_override_pattern)

    def for_layer(self, layer_idx: int):
        """Return a shallow config copy with Puzzle blockwise values applied."""
        if self.block_configs is None:
            return self

        layer_args = copy(self)
        block_config = self.block_configs[layer_idx]
        for name in (
            "moe_intermediate_size",
            "num_experts_per_tok",
            "n_routed_experts",
        ):
            if name in block_config:
                setattr(layer_args, name, block_config[name])
        if block_config["block_type"] == "moe":
            if (
                not layer_args.n_routed_experts
                or layer_args.num_experts_per_tok is None
                or not 0 < layer_args.num_experts_per_tok <= layer_args.n_routed_experts
            ):
                raise ValueError(
                    "MoE top-k must be positive and no greater than expert count"
                )
        return layer_args


@mx.compile
def rms_norm_fp32(x, weight, eps):
    input_dtype = x.dtype
    x = x.astype(mx.float32)
    variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
    x = x * mx.rsqrt(variance + eps)
    return (weight.astype(mx.float32) * x).astype(input_dtype)


class NemotronHRMSNorm(nn.Module):
    """Reference-compatible RMSNorm with float32 reduction and scaling."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm_fp32(x, self.weight, self.eps)


class NemotronHMamba2Mixer(base.NemotronHMamba2Mixer):
    def __init__(self, args):
        layer_args = copy(args)
        layer_args.mamba_proj_bias = args.use_bias
        super().__init__(layer_args)

    def _ssm(
        self,
        hidden_states: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        cache: Any,
        mask: mx.array | None,
    ) -> mx.array:
        (batch_size, seq_len, _) = hidden_states.shape
        output_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        B = B.astype(mx.float32)
        C = C.astype(mx.float32)
        A_log = self.A_log.astype(mx.float32)
        D = self.D.astype(mx.float32)
        dt_bias = self.dt_bias.astype(dt.dtype)
        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
        if cache:
            state = cache[1]
            lengths = cache.lengths
        else:
            (state, lengths) = (None, None)
        (y, state) = ssm_update(
            hidden_states,
            A_log,
            B,
            C,
            D,
            dt,
            dt_bias,
            state,
            self.time_step_limit,
            mask=mask,
            lengths=lengths,
        )
        if cache:
            cache[1] = state
        y = y.astype(output_dtype)
        return y.reshape(batch_size, seq_len, self.intermediate_size)


class MoEGate(base.MoEGate):
    def __call__(self, x):
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        normalize_top_one = self.top_k == 1 and self.norm_topk_prob
        inds, scores = base.group_expert_select(
            logits,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            1.0 if normalize_top_one else self.routed_scaling_factor,
            self.norm_topk_prob,
        )
        # The shared selector normalizes only when top_k > 1. The reference
        # normalizes every selected set before applying routed scaling.
        if normalize_top_one:
            scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
            scores = scores * self.routed_scaling_factor
        return inds, scores


class NemotronHMoE(base.NemotronHMoE):
    def __init__(self, config: ModelArgs):
        nn.Module.__init__(self)
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.moe_latent_size = config.moe_latent_size

        # When latent projection is used, experts operate on the latent dim
        expert_input_dim = (
            config.moe_latent_size
            if config.moe_latent_size is not None
            else config.hidden_size
        )
        self.switch_mlp = SwitchMLP(
            expert_input_dim,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=nn.ReLU2(),
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_shared_expert_intermediate_size
            self.shared_experts = base.NemotronHMLP(
                config, intermediate_size=intermediate_size
            )

        # Latent projection layers for dimensionality reduction before/after experts
        if config.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(
                config.hidden_size, config.moe_latent_size, bias=config.mlp_bias
            )
            self.fc2_latent_proj = nn.Linear(
                config.moe_latent_size, config.hidden_size, bias=config.mlp_bias
            )


class NemotronHBlock(base.NemotronHBlock):
    def __init__(self, args: ModelArgs, block_type: str):
        nn.Module.__init__(self)
        norm_cls = NemotronHRMSNorm
        self.norm = norm_cls(args.hidden_size, eps=args.layer_norm_epsilon)

        self.block_type = block_type

        if self.block_type == "M":
            self.mixer = NemotronHMamba2Mixer(args)
        elif self.block_type == "*":
            self.mixer = base.NemotronHAttention(args)
        elif self.block_type == "-":
            self.mixer = base.NemotronHMLP(args)
        elif self.block_type == "E":
            self.mixer = NemotronHMoE(args)


class NemotronHModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            NemotronHBlock(args.for_layer(i), kind)
            for i, kind in enumerate(args.hybrid_override_pattern)
        ]
        self.norm_f = NemotronHRMSNorm(args.hidden_size, args.layer_norm_epsilon)
        stateful = [
            layer.block_type for layer in self.layers if layer.block_type in ("M", "*")
        ]
        self.fa_idx = stateful.index("*") if "*" in stateful else None
        self.ssm_idx = stateful.index("M") if "M" in stateful else None

    def __call__(self, inputs, cache=None):
        x = self.embeddings(inputs)
        if cache is None:
            cache = [None] * sum(
                layer.block_type in ("M", "*") for layer in self.layers
            )
        attn_mask = (
            create_attention_mask(x, cache[self.fa_idx])
            if self.fa_idx is not None
            else None
        )
        ssm_mask = (
            create_ssm_mask(x, cache[self.ssm_idx])
            if self.ssm_idx is not None
            else None
        )
        caches = iter(cache)
        for layer in self.layers:
            layer_cache = next(caches) if layer.block_type in ("M", "*") else None
            mask = attn_mask if layer.block_type == "*" else ssm_mask
            x = layer(x, mask=mask, cache=layer_cache)
        return self.norm_f(x)


class Model(base.Model):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.args = args
        self.model_type = args.model_type
        self.backbone = NemotronHModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def quant_predicate(self):
        # Preserve the output head, router, and latent projections in full precision.
        return lambda path, _: (
            path != "lm_head"
            and not any(
                name in path.split(".")
                for name in ("gate", "fc1_latent_proj", "fc2_latent_proj")
            )
        )

    def sanitize(self, weights):
        # Official Hugging Face Puzzle checkpoints use ``model.*`` while MLX's
        # Nemotron-H implementation names the same module ``backbone.*``.
        weights = {
            k.removeprefix("language_model."): v
            for k, v in weights.items()
            if not k.startswith(("vision_model.", "mlp1."))
        }
        weights = {
            (f"backbone.{k[6:]}" if k.startswith("model.") else k): v
            for k, v in weights.items()
        }
        weights = {k: v for (k, v) in weights.items() if not k.startswith("mtp.")}
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)

        # Stack experts
        for l in range(self.args.num_hidden_layers):
            prefix = f"backbone.layers.{l}.mixer"
            for m, n in [("down_proj", "fc2"), ("up_proj", "fc1")]:
                if f"{prefix}.experts.0.{m}.weight" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.experts.{e}.{m}.weight")
                        for e in range(self.args.for_layer(l).n_routed_experts)
                    ]
                    weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

        return weights


@mx.compile
def compute_dt_native(dt, dt_bias, time_step_limit):
    dt = nn.softplus(dt + dt_bias)
    return mx.clip(dt, time_step_limit[0], time_step_limit[1])


def ssm_update_kernel(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    time_step_limit: tuple[float, float],
):
    n, _, h, d = hidden_states.shape
    input_type = hidden_states.dtype
    state_type = state.dtype
    hb, ds = B.shape[-2:]
    dt = compute_dt_native(dt, dt_bias, time_step_limit)
    return _ssm_kernel(
        inputs=[hidden_states, A_log, B, C, D, dt, state],
        template=[
            ("T", input_type),
            ("U", state_type),
            ("Dh", d),
            ("Ds", ds),
            ("H", h),
            ("G", h // hb),
        ],
        grid=(32, d, h * n),
        threadgroup=(32, 8, 1),
        output_shapes=[(n, 1, h, d), state.shape],
        output_dtypes=[input_type, state_type],
    )


def ssm_attn(
    x: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array | None = None,
    time_step_limit: tuple[float, float] = (0.001, 100.0),
    mask: mx.array | None = None,
    lengths: mx.array | None = None,
    step: int = 256,
) -> tuple[mx.array, mx.array]:
    """SSD-SSM forward pass.

    Args:
        x: Input of shape (batch_size, seq_len, num_heads, head_dim).
        dt: Time deltas of shape (seq_len, num_heads,).
        A_log: State transition of shape (num_heads,).
        B: Input mixing of shape (batch_size, seq_len, num_groups, n).
        C: Output mixing of shape (batch_size, seq_len, num_groups, n).
        D: Residual connection.
        dt_bias: Bias for time deltas of shape (num_heads,).
        time_step_limit: Minimum and maximum value for time deltas.
        mask: Optional multiplicative mask.
        lengths: Optional lenghts of sequences, assumed to be the full length if unspecified.
        step: Step size for processing x.

    """
    b, l, h, dh = x.shape
    _, _, g, d = B.shape

    dt = compute_dt_native(dt, dt_bias, time_step_limit)
    repeats = h // g
    # In native-dt mode the state transition follows A_log's dtype. Callers
    # that require an FP32 recurrence must therefore pass FP32 A_log.
    A_dtype = A_log.dtype
    A = -mx.exp(A_log).astype(A_dtype)
    dtA = dt * A.reshape(1, 1, -1)
    dtx = dt.reshape(b, l, h, 1) * x

    def _step(dtx, dtA, B, C, state, mask):
        s = dtx.shape[1]
        if lengths is not None:
            valid = mx.arange(s)[None, :] < lengths[:, None]
            mask = valid if mask is None else mask & valid
        B = mx.transpose(B, (0, 2, 3, 1))

        CB = mx.swapaxes(C, 1, 2) @ B
        CB = mx.repeat(CB, repeats, axis=1)

        decay = mx.exp(segsum(dtA.swapaxes(1, 2), mask=mask))

        surrogate_attention_matrix = mx.tril(CB * decay, 0)

        y = surrogate_attention_matrix @ dtx.swapaxes(1, 2)
        y = mx.swapaxes(y, 1, 2)

        if lengths is not None:
            # Remaining cache lengths can exceed this call's final chunk.
            pos = mx.maximum(mx.minimum(lengths, s) - 1, 0)
            pos = mx.expand_dims(pos, (1, 2, 3))
            decay = mx.take_along_axis(decay, pos, axis=2)
        else:
            decay = decay[:, :, -1:, :]

        decay = decay.transpose(0, 3, 1, 2)
        B = mx.repeat(B, h // g, axis=1).swapaxes(2, 3)
        dtxdecay = dtx * decay
        dtxdecay = dtxdecay.swapaxes(1, 2).swapaxes(2, 3)

        next_state = dtxdecay @ B

        if state is not None:
            # Incoming state advances only through valid tokens, including in
            # the partial final chunk of a right-padded batch.
            if mask is not None:
                dtA = mx.where(mask[..., None], dtA, 0)
            exp_dtA_cumsum = mx.exp(mx.cumsum(dtA, axis=-2))
            next_state += exp_dtA_cumsum[:, -1, :, None, None] * state
            C = C.reshape(b, s, g, 1, d, 1)
            y_prev = (
                (state.reshape((b, 1, g, repeats, dh, d)) @ C).squeeze(-1).flatten(2, 3)
            )
            y += exp_dtA_cumsum[..., None] * y_prev
        if lengths is not None and state is not None:
            next_state = mx.where(
                mx.expand_dims(lengths <= 0, (1, 2, 3)), state, next_state
            )

        return y.astype(x.dtype), next_state

    ys = []
    for i in range(0, l, step):
        y, state = _step(
            dtx[:, i : i + step],
            dtA[:, i : i + step],
            B[:, i : i + step],
            C[:, i : i + step],
            state,
            None if mask is None else mask[..., i : i + step],
        )
        if lengths is not None:
            lengths = lengths - step
        ys.append(y)
    y = mx.concatenate(ys, axis=1) + x * D.reshape(1, 1, h, 1)
    return y, state


def ssm_update(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array | None = None,
    time_step_limit: tuple[float, float] = (0.001, 100.0),
    mask: mx.array | None = None,
    lengths: mx.array | None = None,
):
    seq_len = hidden_states.shape[1]
    if (
        seq_len > 1
        or state is None
        or mask is not None
        or lengths is not None
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return ssm_attn(
            hidden_states,
            A_log,
            B,
            C,
            D,
            dt,
            dt_bias,
            state,
            time_step_limit,
            mask=mask,
            lengths=lengths,
        )
    else:
        return ssm_update_kernel(
            hidden_states,
            A_log,
            B,
            C,
            D,
            dt,
            dt_bias,
            state,
            time_step_limit,
        )
