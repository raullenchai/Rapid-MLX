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
# Adapted from mlx-lm 0.31.3 (nemotron_h.py and ssm.py).

"""Correct shared SSM right-padding and forward Nemotron-H cache lengths.

Installed from the canonical model loader. The shared function is patched in
its defining module so existing imported ssm_update aliases also see the fix.
The Nemotron-H mixer additionally needs to forward its cache lengths and avoid
using the unmasked single-token kernel during padded prefill. Ordinary decode
continues through the dependency's dispatcher and Metal kernel.
"""

from __future__ import annotations

import inspect
import logging
import threading
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mlx.core as mx

_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)


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
    """Shared SSD recurrence with padding excluded from incoming-state decay.

    Retains upstream compute_dt, segsum, accumulation dtype and chunking.
    Remove this shim once the dependency supplies these padding corrections.
    """
    import mlx.core as mx
    from mlx_lm.models.ssm import compute_dt, segsum

    b, l, h, dh = x.shape
    _, _, g, d = B.shape

    dt = compute_dt(dt, dt_bias, time_step_limit)
    repeats = h // g
    A = -mx.exp(A_log).astype(dt.dtype)
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


def _mixer_ssm(
    self,
    hidden_states: mx.array,
    B: mx.array,
    C: mx.array,
    dt: mx.array,
    cache: Any,
    mask: mx.array | None,
) -> mx.array:
    from mlx_lm.models.ssm import ssm_update

    batch_size, seq_len, _ = hidden_states.shape

    hidden_states = hidden_states.reshape(
        batch_size, seq_len, self.num_heads, self.head_dim
    )
    B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
    C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size)
    if cache:
        state = cache[1]
        lengths = cache.lengths
    else:
        state, lengths = None, None

    # The Metal decode kernel has no mask/length arguments.
    update = ssm_attn if mask is not None or lengths is not None else ssm_update
    y, state = update(
        hidden_states,
        self.A_log,
        B,
        C,
        self.D.astype(hidden_states.dtype),
        dt,
        self.dt_bias,
        state,
        self.time_step_limit,
        mask=mask,
        lengths=lengths,
    )
    if cache:
        cache[1] = state

    return y.reshape(batch_size, seq_len, self.intermediate_size)


def install_nemotron_h_ssm_fix() -> None:
    """Install once, including when model modules were imported before loading."""
    with _LOCK:
        try:
            from mlx_lm.models import nemotron_h, ssm
        except ImportError:
            return
        mixer = getattr(nemotron_h, "NemotronHMamba2Mixer", None)
        if (
            getattr(ssm, "ssm_attn", None) is ssm_attn
            and getattr(mixer, "_ssm", None) is _mixer_ssm
        ):
            return

        def contract(function):
            # Annotations vary across Python releases; calling conventions do not.
            return tuple(
                (p.name, p.kind, p.default)
                for p in inspect.signature(function).parameters.values()
            )

        try:
            update_contract = contract(ssm_attn)[:-1]
            update_contract = (
                ("hidden_states", *update_contract[0][1:]),
            ) + update_contract[1:]
            supported = (
                version("mlx-lm") == "0.31.3"
                and contract(ssm.ssm_attn) == contract(ssm_attn)
                and contract(ssm.ssm_update) == update_contract
                and contract(nemotron_h.NemotronHMamba2Mixer._ssm)
                == contract(_mixer_ssm)
            )
        except (AttributeError, TypeError, ValueError, PackageNotFoundError):
            supported = False
        if not supported:
            _LOGGER.warning(
                "Skipping Nemotron-H padding shim: unsupported mlx-lm version or "
                "SSM calling convention; no runtime functions were changed"
            )
            return
        # Install both only after validation. Loader callers share this lock;
        # the shared recurrence is compatible with the original mixer too.
        ssm.ssm_attn = ssm_attn
        nemotron_h.NemotronHMamba2Mixer._ssm = _mixer_ssm
