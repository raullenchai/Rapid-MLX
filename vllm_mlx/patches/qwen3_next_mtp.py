# SPDX-License-Identifier: Apache-2.0
"""
Runtime MTP (Multi-Token Prediction) support for Qwen3-Next models.

Qwen3-Next models may include a built-in MTP head that predicts token n+2
from hidden states + token n+1.  MTP weights are added to the quantized
MLX model via scripts/add_mtp_weights.py.

Since mlx_lm's qwen3_next.py does NOT define MTP module/methods, this
module provides:
  - inject_mtp_support(): dynamically creates MTP module, loads weights,
    and monkey-patches the model class with return_hidden, mtp_forward,
    and make_mtp_cache
  - validate_mtp_support(): checks whether a loaded model has working MTP

The actual MTP scheduling logic lives in:
  - vllm_mlx/scheduler.py  (_install_mtp, _mtp_step, _mtp_next)
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def inject_mtp_support(model: Any, model_path, config: dict) -> bool:
    """Inject MTP module into a loaded Qwen3-Next model.

    mlx_lm's qwen3_next.py does not define MTP layers, so we:
    1. Create MTP module matching the weight structure
    2. Quantize it to match the base model
    3. Load MTP weights from model-mtp.safetensors
    4. Monkey-patch Model with return_hidden, mtp_forward, make_mtp_cache

    Args:
        model: A model loaded via mlx_lm (strict=False, MTP weights ignored)
        model_path: Path to model directory (contains model-mtp.safetensors)
        config: Parsed config.json dict

    Returns:
        True if MTP was successfully injected, False otherwise.
    """
    import mlx.core as mx
    import mlx.nn as nn

    num_mtp_layers = config.get("num_nextn_predict_layers", 0)
    if num_mtp_layers == 0:
        logger.info("[MTP inject] num_nextn_predict_layers=0, skipping")
        return False

    model_path = Path(model_path)
    mtp_file = model_path / "model-mtp.safetensors"
    if not mtp_file.exists():
        logger.warning(f"[MTP inject] model-mtp.safetensors not found in {model_path}")
        return False

    args = model.args

    # Import model components
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.qwen3_next import Qwen3NextDecoderLayer

    # --- Step 1: Create MTP module ---
    logger.info(f"[MTP inject] Creating MTP module ({num_mtp_layers} layers)")

    class _MTPModule(nn.Module):
        def __init__(self, args, n_layers):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.pre_fc_norm_embedding = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            # MTP decoder uses full attention (not linear/delta-net)
            fa_idx = args.full_attention_interval - 1
            self.layers = [
                Qwen3NextDecoderLayer(args, layer_idx=fa_idx) for _ in range(n_layers)
            ]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    mtp = _MTPModule(args, num_mtp_layers)

    # --- Step 2: Quantize MTP module to match base model ---
    # nn.quantize handles Linear → QuantizedLinear but NOT SwitchLinear →
    # QuantizedSwitchLinear.  We must replace SwitchLinear BEFORE load_weights
    # so the parameter names (weight, scales, biases) match the saved file.
    quant_config = config.get("quantization", {})
    if quant_config:
        bits = quant_config.get("bits", 6)
        group_size = quant_config.get("group_size", 64)
        mode = quant_config.get("mode", "affine")

        # 2a: Replace SwitchLinear → QuantizedSwitchLinear in MoE blocks
        try:
            from mlx_lm.models.switch_layers import (
                QuantizedSwitchLinear,
                SwitchLinear,
            )

            for layer in mtp.layers:
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
                    sm = layer.mlp.switch_mlp
                    for proj_name in ["up_proj", "down_proj", "gate_proj"]:
                        proj = getattr(sm, proj_name, None)
                        if proj is not None and isinstance(proj, SwitchLinear):
                            ne = proj.weight.shape[0]   # num_experts
                            od = proj.weight.shape[1]    # output_dims
                            id_ = proj.weight.shape[2]   # input_dims
                            q = QuantizedSwitchLinear(
                                id_, od, ne,
                                bias=False,
                                group_size=group_size,
                                bits=bits,
                                mode=mode,
                            )
                            setattr(sm, proj_name, q)
                    logger.info("[MTP inject] Replaced SwitchLinear → QuantizedSwitchLinear")
        except ImportError:
            logger.warning("[MTP inject] Could not import QuantizedSwitchLinear")

        # 2b: Quantize remaining Linear layers (attention, shared_expert, gate)
        def _mtp_quant_pred(path, module):
            if not isinstance(module, nn.Linear):
                return False
            # fc kept as FP (concat projection)
            if path == "fc":
                return False
            # shared_expert_gate kept as FP (small, stored unquantized)
            if path.endswith("shared_expert_gate"):
                return False
            return True

        nn.quantize(
            mtp, group_size=group_size, bits=bits, class_predicate=_mtp_quant_pred
        )
        logger.info(f"[MTP inject] Quantized MTP: {bits}-bit, group_size={group_size}")

    # --- Step 3: Load MTP weights ---
    logger.info(f"[MTP inject] Loading weights from {mtp_file.name}")
    raw = mx.load(str(mtp_file))
    mtp_weights = {
        k.removeprefix("mtp."): v for k, v in raw.items() if k.startswith("mtp.")
    }
    mtp.load_weights(list(mtp_weights.items()), strict=False)
    mx.eval(mtp.parameters())
    logger.info(f"[MTP inject] Loaded {len(mtp_weights)} MTP weight tensors")

    # --- Step 4: Attach MTP and monkey-patch model class ---
    model.mtp = mtp

    original_class = model.__class__

    class _Qwen3NextMTP(original_class):
        """Qwen3-Next with MTP support (injected at runtime)."""

        def __call__(
            self,
            inputs,
            cache=None,
            return_hidden: bool = False,
        ):
            inner = self.model
            hidden_states = inner.embed_tokens(inputs)
            if cache is None:
                cache = [None] * len(inner.layers)
            fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
            ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])
            for layer, c in zip(inner.layers, cache):
                mask = ssm_mask if layer.is_linear else fa_mask
                hidden_states = layer(hidden_states, mask=mask, cache=c)
            normed = inner.norm(hidden_states)
            if self.args.tie_word_embeddings:
                out = inner.embed_tokens.as_linear(normed)
            else:
                out = self.lm_head(normed)
            if return_hidden:
                return out, hidden_states  # pre-norm hidden states
            return out

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            cache=None,
            mtp_cache=None,
        ):
            """Run MTP head: predict token n+2 from hidden states + token n+1."""
            input_embeds = self.model.embed_tokens(next_token_ids)
            h = self.mtp.pre_fc_norm_hidden(hidden_states)
            e = self.mtp.pre_fc_norm_embedding(input_embeds)
            x = self.mtp.fc(mx.concatenate([h, e], axis=-1))
            layer = self.mtp.layers[0]
            c = mtp_cache[0] if mtp_cache else None
            mask = create_attention_mask(x, c)
            x = layer(x, mask=mask, cache=c)
            x = self.mtp.norm(x)
            if self.args.tie_word_embeddings:
                return self.model.embed_tokens.as_linear(x)
            return self.lm_head(x)

        def make_mtp_cache(self):
            """Create KV cache for MTP layers."""
            if self.mtp is None:
                return None
            return [KVCache() for _ in self.mtp.layers]

    model.__class__ = _Qwen3NextMTP
    logger.info("[MTP inject] Model class patched with MTP support")
    return True


def validate_mtp_support(model: Any) -> bool:
    """Validate that a loaded model has working MTP support.

    Checks:
    1. model.mtp exists and is not None (MTP module instantiated)
    2. model.mtp has layers with loaded weights
    3. model has return_hidden support in __call__
    4. model has mtp_forward method
    5. model has make_mtp_cache method

    Args:
        model: A model loaded via mlx_lm.load()

    Returns:
        True if MTP is fully functional, False otherwise.
    """
    # Check 1: MTP module exists
    mtp = getattr(model, "mtp", None)
    if mtp is None:
        num_mtp = 0
        # Try model.args (Qwen3-Next) and model.language_model.args (Qwen3.5)
        args = getattr(model, "args", None)
        if args is None:
            lm = getattr(model, "language_model", None)
            if lm is not None:
                args = getattr(lm, "args", None)
        if args is not None:
            num_mtp = getattr(args, "num_nextn_predict_layers", 0)
        if num_mtp > 0:
            logger.warning(
                "[MTP] Model config has num_nextn_predict_layers=%d but "
                "model.mtp is None. MTP weights may not be in the model files. "
                "Run scripts/add_mtp_weights.py to add them.",
                num_mtp,
            )
        else:
            logger.info(
                "[MTP] Model does not have MTP config (num_nextn_predict_layers=0)."
            )
        return False

    # Check 2: MTP layers have weights
    mtp_layers = getattr(mtp, "layers", [])
    if not mtp_layers:
        logger.warning("[MTP] model.mtp exists but has no layers.")
        return False

    # Check 3: return_hidden support
    import inspect

    call_sig = inspect.signature(type(model).__call__)
    if "return_hidden" not in call_sig.parameters:
        logger.warning(
            "[MTP] Model.__call__ does not accept return_hidden parameter. "
            "The mlx_lm model implementation may be outdated."
        )
        return False

    # Check 4: mtp_forward method
    if not hasattr(model, "mtp_forward") or not callable(model.mtp_forward):
        logger.warning("[MTP] Model does not have mtp_forward() method.")
        return False

    # Check 5: make_mtp_cache method
    if not hasattr(model, "make_mtp_cache") or not callable(model.make_mtp_cache):
        logger.warning("[MTP] Model does not have make_mtp_cache() method.")
        return False

    # All checks passed
    args = getattr(model, "args", None)
    num_layers = getattr(args, "num_nextn_predict_layers", 0) if args else 0
    logger.info(
        "[MTP] Model has working MTP support: "
        "%s MTP layer(s), %d predictor decoder layer(s)",
        num_layers,
        len(mtp_layers),
    )
    return True
