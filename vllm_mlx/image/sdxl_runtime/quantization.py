"""Weight-only quantization helpers built on ``mlx.nn.quantize``.

Quantization is a *load-time* choice (``from_pretrained(..., quantize=4)``), not a
new class hierarchy. We expose one function plus a sensible default predicate that
skips layers too small to quantize cleanly.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.nn as nn

__all__ = ["quantize_module", "default_quant_predicate"]

VALID_BITS = (2, 3, 4, 6, 8)


def default_quant_predicate(group_size: int) -> Callable[[str, nn.Module], bool]:
    """Predicate that quantizes Linear/Embedding layers whose dims divide cleanly.

    Layers whose feature dimension is not a multiple of ``group_size`` (e.g. tiny
    projection heads) are left in full precision to avoid shape errors and quality
    cliffs.
    """

    def predicate(path: str, module: nn.Module) -> bool:
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        weight = module.weight
        return bool(weight.shape[-1] % group_size == 0)

    return predicate


def quantize_module(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 64,
    predicate: Callable[[str, nn.Module], bool] | None = None,
) -> nn.Module:
    """Quantize ``model`` in place to ``bits`` and return it.

    Args:
        model: module to quantize (modified in place).
        bits: one of ``2, 3, 4, 6, 8``.
        group_size: quantization group size (must divide quantized dims).
        predicate: ``(path, module) -> bool`` selecting layers; defaults to
            :func:`default_quant_predicate`.
    """
    if bits not in VALID_BITS:
        raise ValueError(f"bits must be one of {VALID_BITS}, got {bits}.")
    predicate = predicate or default_quant_predicate(group_size)
    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
    return model
