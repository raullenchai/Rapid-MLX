# SPDX-License-Identifier: Apache-2.0
"""Native-MTP attachment for the vendored Qwen4 decoder.

The checkpoint stores its one-layer predictor under ``mtp.*`` while the
ordinary target loader intentionally ignores that subtree.  This module builds
the matching MLX layer, validates every packed tensor, and attaches the generic
MTP generation protocol only after a complete load succeeds.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_inner(model: Any) -> Any | None:
    inner = getattr(model, "language_model", None)
    if inner is not None and getattr(inner, "model_type", None) == "qwen4_exp_text":
        return inner
    if getattr(model, "model_type", None) == "qwen4_exp_text":
        return model
    return None


def _mtp_weight_files(source: str | Path) -> list[Path]:
    path = Path(source).expanduser()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"MTP checkpoint path does not exist: {path}")
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map", {})
        names = sorted(
            {filename for key, filename in weight_map.items() if key.startswith("mtp.")}
        )
        if names:
            return [path / name for name in names]
    single = path / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"No safetensors containing mtp.* found under {path}")


def _sanitize_mtp_weights(raw: dict[str, Any]) -> dict[str, Any]:
    """Map checkpoint expert packing into the vendored MLX module tree."""

    sanitized: dict[str, Any] = {}
    for original_key, value in raw.items():
        if not original_key.startswith("mtp."):
            continue
        key = original_key.removeprefix("mtp.")
        if ".mlp.experts.gate_up_proj" in key:
            midpoint = value.shape[-2] // 2
            prefix, suffix = key.split("experts.gate_up_proj", 1)
            leaf = {"": "weight", ".scales": "scales", ".biases": "biases"}.get(suffix)
            if leaf is None:
                sanitized[key] = value
                continue
            base = f"{prefix}switch_mlp"
            sanitized[f"{base}.gate_proj.{leaf}"] = value[..., :midpoint, :]
            sanitized[f"{base}.up_proj.{leaf}"] = value[..., midpoint:, :]
            continue
        if ".mlp.experts.down_proj" in key:
            prefix, suffix = key.split("experts.down_proj", 1)
            leaf = {"": "weight", ".scales": "scales", ".biases": "biases"}.get(suffix)
            if leaf is not None:
                key = f"{prefix}switch_mlp.down_proj.{leaf}"
        sanitized[key] = value
    return sanitized


def _build_mtp(inner: Any):
    import mlx.nn as nn

    from vllm_mlx.models.qwen4_exp import (
        DecoderLayer,
        GatedResidual,
        TextModelArgs,
        ZeroCenteredRMSNorm,
    )

    args = inner.args
    params = dict(vars(args))
    params.update(
        num_hidden_layers=1,
        layer_types=["full_attention"],
        ple_layer_ids=[],
    )
    mtp_args = TextModelArgs.from_dict(params)

    class Qwen4ExpMTP(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = mtp_args.hidden_size
            self.hc_count = mtp_args.hc_count
            self.hidden_size = hidden
            self.pre_fc_norm_embedding = ZeroCenteredRMSNorm(
                hidden, eps=mtp_args.rms_norm_eps
            )
            self.pre_fc_norm_hidden = ZeroCenteredRMSNorm(
                hidden * mtp_args.hc_count,
                group_size=hidden,
                eps=mtp_args.rms_norm_eps,
            )
            self.fc_embedding = nn.Linear(hidden, hidden, bias=False)
            self.fc_hidden = nn.Linear(hidden, hidden, bias=False)
            self.layers = [DecoderLayer(mtp_args, 0)]
            self.hyper_connection_mixer = GatedResidual(mtp_args, use_combine=False)

        def __call__(self, hidden_states, token_embeddings, cache):
            embedding = self.fc_embedding(self.pre_fc_norm_embedding(token_embeddings))
            shape = hidden_states.shape
            branches = self.pre_fc_norm_hidden(hidden_states).reshape(
                *shape[:-1], self.hc_count, self.hidden_size
            )
            branches = self.fc_hidden(branches)
            multi_hidden = (branches + embedding[..., None, :]).flatten(-2)
            multi_hidden = self.layers[0](
                multi_hidden,
                input_ids=mx.zeros(token_embeddings.shape[:-1], dtype=mx.uint32),
                mask=None,
                cache=cache[0],
            )
            sample_hidden = self.hyper_connection_mixer(multi_hidden)
            return sample_hidden, multi_hidden

    import mlx.core as mx

    mtp = Qwen4ExpMTP()

    def predicate(path: str, module: Any):
        if not hasattr(module, "to_quantized"):
            return False
        return inner.quant_predicate(path, module)

    nn.quantize(
        mtp,
        group_size=64,
        bits=4,
        class_predicate=predicate,
    )
    return mtp


def inject_qwen4_exp_mtp_support(
    model: Any,
    *,
    mtp_sidecar: str | Path | None = None,
    allow_random_init: bool = False,
) -> bool:
    """Attach native Qwen4 MTP surfaces after complete tensor validation."""

    import mlx.core as mx
    from mlx.utils import tree_flatten

    inner = _resolve_inner(model)
    if inner is None:
        return False
    if int(getattr(inner.args, "mtp_num_hidden_layers", 0) or 0) != 1:
        logger.warning("[mtp.qwen4] only one-layer checkpoints are supported")
        return False
    if mtp_sidecar is None and not allow_random_init:
        logger.warning("[mtp.qwen4] a local base-checkpoint path is required")
        return False

    try:
        mtp = _build_mtp(inner)
        weights: dict[str, Any] = {}
        if mtp_sidecar is not None:
            for file in _mtp_weight_files(mtp_sidecar):
                weights.update(_sanitize_mtp_weights(mx.load(str(file))))

            expected = dict(tree_flatten(mtp.parameters()))
            missing = set(expected) - set(weights)
            unexpected = set(weights) - set(expected)
            mismatched = {
                key: (tuple(weights[key].shape), tuple(expected[key].shape))
                for key in set(expected) & set(weights)
                if tuple(weights[key].shape) != tuple(expected[key].shape)
            }
            if missing or unexpected or mismatched:
                logger.error(
                    "[mtp.qwen4] tensor contract mismatch: missing=%s "
                    "unexpected=%s shape=%s",
                    sorted(missing)[:8],
                    sorted(unexpected)[:8],
                    sorted(mismatched.items())[:8],
                )
                return False
            mtp.load_weights(list(weights.items()), strict=True)
        else:
            mx.eval(mtp.parameters())

        original_class = type(inner)

        class _Qwen4ExpWithMTP(original_class):  # type: ignore[valid-type, misc]
            mtp_prompt_lookup_supported = False

            def mtp_forward(
                self,
                hidden_states,
                next_token_ids,
                mtp_cache,
                return_hidden: bool = False,
            ):
                token_embeddings = self.model.embed_tokens(next_token_ids)
                sample_hidden, multi_hidden = self.mtp(
                    hidden_states, token_embeddings, mtp_cache
                )
                if self.args.tie_word_embeddings:
                    logits = self.model.embed_tokens.as_linear(sample_hidden)
                else:
                    logits = self.lm_head(sample_hidden)
                return (logits, multi_hidden) if return_hidden else logits

            def make_mtp_cache(self):
                from mlx_lm.models.cache import CacheList, KVCache

                from vllm_mlx.models.qwen4_exp_cache import QSAIndexCache

                ratio = self.mtp.layers[0].self_attn.indexer.compress_ratio
                return [CacheList(KVCache(), QSAIndexCache(ratio))]

        inner.mtp = mtp
        inner.mtp_max_speculative_tokens = 1
        model.mtp_max_speculative_tokens = 1
        inner.__class__ = _Qwen4ExpWithMTP
        mx.eval(mtp.parameters())
        return True
    except Exception:
        logger.exception("[mtp.qwen4] native MTP attachment failed")
        return False


def validate_qwen4_exp_mtp_support(model: Any) -> bool:
    inner = _resolve_inner(model)
    if inner is None or not hasattr(inner, "mtp"):
        return False
    try:
        signature = inspect.signature(inner.__call__)
    except (TypeError, ValueError):
        return False
    return (
        callable(getattr(inner, "mtp_forward", None))
        and callable(getattr(inner, "make_mtp_cache", None))
        and "return_hidden" in signature.parameters
        and "n_confirmed" in signature.parameters
    )
