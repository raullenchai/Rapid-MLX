# SPDX-License-Identifier: Apache-2.0
"""Request scheduling policy shared by text and MLLM text-only consumers."""

from dataclasses import replace
from typing import cast

from . import model_profile, resolve, runtime_context
from .mtp import _expert_quantization


def scheduling_value(name: str, model, *, mode: str = "plain", batch: int = 1) -> float:
    """Resolve with the loaded model and device facts, outside token loops."""
    import mlx.core as mx

    model = getattr(model, "language_model", model)
    args = getattr(model, "args", None)
    config = vars(args) if hasattr(args, "__dict__") else {}
    profile = model_profile(config)
    if profile.family == "qwen3_5_text" and not profile.num_experts:
        # Live text args omit quantization; declarations cannot qualify mixed weights.
        quant = _expert_quantization(model, dense=True)
        profile = replace(profile, bits=None, group_size=None, quant_mode="unknown")
        if quant is not None:
            profile = replace(
                profile, bits=quant[0], group_size=quant[1], quant_mode=quant[2]
            )
    context = runtime_context(mx.device_info(), model=profile, mode=mode, batch=batch)
    return cast(float, resolve(name, context).value)


def skip_prefix_boundary(model, uncached_tokens: int, *, mode: str = "plain") -> bool:
    """Strictly short tails trade a boundary snapshot for fewer forwards."""
    return uncached_tokens < scheduling_value(
        "prefix_boundary_min_tokens", model, mode=mode
    )
