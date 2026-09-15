# Copyright © 2026 Apple Inc.
# FusedSwitchGLU vendored from spicyneuron/mlx-lm@df50b7a3 for DeepSeek V4.

import mlx.core as mx
import mlx.nn as nn

from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

try:
    # oMLX's audited Apple-Silicon extension supplies MXFP4 block-list MoE
    # kernels.  Keep this optional: stock Rapid-MLX continues to use mlx-lm
    # when the native bundle is not installed.
    from omlx.patches.deepseek_v4.switch_layers import (
        SwiGLU,
        SwitchLinear,
        _gather_sort,
        _scatter_unsort,
    )
except (ImportError, OSError):
    from mlx_lm.models.switch_layers import (
        SwiGLU,
        SwitchLinear,
        _gather_sort,
        _scatter_unsort,
    )


class FusedSwitchGLU(nn.Module):
    """Fused gate/up sparse expert projection used by DeepSeek V4."""

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = SwitchLinear(
            input_dims, 2 * hidden_dims, num_experts, bias=bias
        )
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation
        self.hidden_dims = hidden_dims

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        # DSpark verifies at most six rows; with eight routed experts that is
        # 48 routes.  The native block-list kernel is already profitable at
        # this size on M3 Ultra, while mlx-lm's generic path keeps its original
        # 64-route sorting threshold.
        native_blocks = SwitchLinear.__module__.startswith("omlx.")
        do_sort = indices.size >= (32 if native_blocks else 64)
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        x_gate, x_up = mx.split(
            self.gate_proj(x, idx, sorted_indices=do_sort),
            [self.hidden_dims],
            axis=-1,
        )
        x = self.down_proj(
            self.activation(x_up, x_gate), idx, sorted_indices=do_sort
        )
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)
