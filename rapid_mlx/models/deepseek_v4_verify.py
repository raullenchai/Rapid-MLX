"""Decode-equivalent multi-row projections for DSpark target verify."""

from __future__ import annotations


def install() -> None:
    import mlx.nn as nn

    from .deepseek_v4_rollback import is_armed
    from .deepseek_v4_verify_qmv import (
        dense_eligible,
        eligible,
        exact_verify_gemv,
        exact_verify_qmv,
    )

    linear = nn.Linear
    if not getattr(linear, "_rapid_dspark_verify", False):
        original = linear.__call__

        def linear_call(self, x):
            if is_armed() and dense_eligible(self, x):
                return exact_verify_gemv(self, x)
            return original(self, x)

        linear.__call__ = linear_call
        linear._rapid_dspark_verify = True

    quantized = nn.QuantizedLinear
    if not getattr(quantized, "_rapid_dspark_verify", False):
        original_q = quantized.__call__

        def quantized_call(self, x):
            if is_armed() and eligible(self, x):
                return exact_verify_qmv(self, x)
            return original_q(self, x)

        quantized.__call__ = quantized_call
        quantized._rapid_dspark_verify = True
