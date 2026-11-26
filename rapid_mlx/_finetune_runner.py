# SPDX-License-Identifier: Apache-2.0
"""Install Qwen3.5 compatibility patches inside the mlx-lm subprocess."""

import inspect
import sys


def _install_gated_delta():
    import mlx.core as mx

    if not hasattr(mx.fast, "gated_delta_update"):
        return False
    try:
        from mlx_lm.models import gated_delta, qwen3_5

        stock = gated_delta.gated_delta_update
        compute_g = gated_delta.compute_g
        signature = inspect.signature(stock)
    except (ImportError, AttributeError, TypeError, ValueError):
        return False

    def update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):  # noqa: N803
        # MLX and mlx-lm differ on the output at masked positions.
        if use_kernel or mask is not None:
            return stock(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
        g = compute_g(A_log, a, dt_bias)
        return mx.fast.gated_delta_update(q, k, v, g, mx.sigmoid(b), state, mask)

    parameters = [
        p.replace(annotation=inspect.Parameter.empty)
        for p in signature.parameters.values()
    ]
    if (
        inspect.Signature(parameters) != inspect.signature(update)
        or getattr(qwen3_5, "gated_delta_update", None) is not stock
    ):
        return False
    gated_delta.gated_delta_update = qwen3_5.gated_delta_update = update
    return True


def main():
    if sys.argv[1:2] == ["--qwen35"]:
        sys.argv.pop(1)
        from .patches.qwen3_5_norm_shift import install_qwen3_5_norm_shift_fix

        install_qwen3_5_norm_shift_fix()
        if not _install_gated_delta():
            print(
                "Warning: Qwen3.5 training will be slow and memory-hungry with "
                "this MLX/MLX-LM combination. The memory estimate assumes the native kernel.",
                file=sys.stderr,
            )

    from mlx_lm.lora import main as lora_main

    lora_main()


if __name__ == "__main__":
    main()
