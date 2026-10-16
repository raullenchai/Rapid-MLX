# SPDX-License-Identifier: Apache-2.0
"""Precision adapter for standalone Nemotron-H text extracted from Super VL.

Artifacts retain the native ``nemotron_h`` config and tensor layout. The
``rapid_mlx_nemotron_h`` opt-in selects the reviewed local math in Rapid's
checked loader, without replacing the installed architecture globally.
"""

import mlx.core as mx
import mlx.nn as nn

from .. import _mlx_compat

_mlx_compat.install()

from mlx_lm.models import nemotron_h as base  # noqa: E402

from . import nemotron_h_puzzle as shared  # noqa: E402


def text_config(config):
    """Extract text metadata without executing checkpoint Python files."""
    config = dict(config)
    if config.get("model_type") == "nemotron_h_omni":
        config = dict(config["llm_config"])
    if config.get("model_type") != "nemotron_h":
        raise ValueError("Expected nemotron_h language config")
    for key in ("auto_map", "model_file"):
        config.pop(key, None)
    config["num_hidden_layers"] = len(config["layers_block_type"])
    # JSON's encoded infinity is not a numeric argument to mx.clip. Omitting
    # the default bound is valid JSON and uses the native 0..inf default.
    limit = config.get("time_step_limit")
    if limit == [0.0, {"__float__": "Infinity"}]:
        config.pop("time_step_limit")
    config["rapid_mlx_nemotron_h"] = True
    # No Rapid consumer implements Nemotron-H MTP; its tensors are dropped.
    config.pop("num_nextn_predict_layers", None)
    config.pop("mtp_layers_block_type", None)
    return config


class ModelArgs(base.ModelArgs):
    @classmethod
    def from_dict(cls, params):
        return super().from_dict(text_config(params))

    def for_layer(self, layer_idx):
        return self


@mx.compile
def rms_norm(x, weight, eps):
    # Round the normalized activation before applying its weight.
    value = x.astype(mx.float32)
    variance = mx.mean(mx.square(value), axis=-1, keepdims=True)
    return weight * (value * mx.rsqrt(variance + eps)).astype(x.dtype)


@mx.compile
def gated_rms_norm(x, weight, gate, eps, group_size):
    value = x.astype(mx.float32)
    if gate is not None:
        value = value * nn.silu(gate.astype(mx.float32))
    grouped = mx.unflatten(value, axis=-1, shape=(-1, group_size))
    variance = mx.mean(mx.square(grouped), axis=-1, keepdims=True)
    grouped = grouped * mx.rsqrt(variance + eps)
    return weight * grouped.flatten(-2).astype(x.dtype)


class RMSNorm(shared.NemotronHRMSNorm):
    def __call__(self, x):
        return rms_norm(x, self.weight, self.eps)


class GatedRMSNorm(base.MambaRMSNormGated):
    def __call__(self, x, gate=None):
        return gated_rms_norm(x, self.weight, gate, self.eps, self.group_size)


class Model(shared.Model):
    def __init__(self, args):
        super().__init__(args)
        for layer in self.layers:
            layer.norm = RMSNorm(args.hidden_size, args.layer_norm_epsilon)
            if layer.block_type == "M":
                mixer = layer.mixer
                mixer.norm = GatedRMSNorm(
                    mixer.intermediate_size,
                    args.layer_norm_epsilon,
                    mixer.intermediate_size // mixer.n_groups,
                )
        self.backbone.norm_f = RMSNorm(args.hidden_size, args.layer_norm_epsilon)

    def sanitize(self, weights):
        return super().sanitize(
            {k: v for k, v in weights.items() if not k.startswith("vision_projector.")}
        )


def get_model_classes(config):
    return Model, ModelArgs
