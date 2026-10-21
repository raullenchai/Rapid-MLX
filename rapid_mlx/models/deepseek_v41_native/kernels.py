# SPDX-License-Identifier: Apache-2.0
"""Process-wide switch for the native DeepSeek V4.1 custom Metal kernels.

``RAPID_MLX_DSV41_FUSED=0`` (or ``false``/``no``/``off``/``disable``/
``disabled``) sends every custom kernel of this backend (head and router GEMV,
fused MoE and down-route QMV, hyper-connection kernels, rope + fake-quant and
fused decode attention) to its stock MLX path. It is read at each kernel's
existing eligibility check, next to the module's ``ENABLED`` constant, so it
takes effect without reloading weights. It selects no model, parser, engine or
serving lane; it only trades speed for the stock numerics.
"""

from __future__ import annotations

import os

from rapid_mlx._env import is_falsey

FUSED_ENV = "RAPID_MLX_DSV41_FUSED"


def custom_kernels_enabled() -> bool:
    """False when the operator turned the V4.1 custom kernels off."""
    return not is_falsey(os.environ.get(FUSED_ENV))
