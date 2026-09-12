# SPDX-License-Identifier: Apache-2.0
"""Fuse MoE routed gate/up expert projections into one ``gather_qmm``.

At single-token decode, the stock ``SwitchGLU`` implementations in mlx-lm
and mlx-vlm issue three tiny
``gather_qmm`` launches per MoE layer (gate, up, down). Affine
quantization packs each output row independently, so concatenating the
gate and up expert weights along the output axis and issuing ONE
``gather_qmm`` over ``[E, 2*inter, hidden]`` is bit-identical to the two
separate calls while removing one GPU launch per MoE layer per token.
The same fused weights serve prefill and batched decode unchanged.

Measured on Qwen3.6-35B-A3B-8bit (M3 Ultra): decode 86.4 -> 92.5 tok/s
(+7.0%) with byte-identical greedy output across the fused 40 layers.

Fusion design adapted from the oMLX project
(https://github.com/jundot/omlx, Apache-2.0), which ships it default-on
for the same layer type.

``fuse_gate_up(model)`` runs once post-load: it patches each loaded
``SwitchGLU.__call__`` family with a fused-aware branch (class-level, once per
process, stock path preserved for unfused instances) and rewrites each
eligible instance in place — the gate module becomes the fused container
so quantization parameters carry over, and the original gate/up buffers
are freed layer-by-layer to bound the transient. Instances whose
gate/up pair fails the structural checks are left untouched. Set
``RAPID_MLX_MOE_GATE_UP_FUSION=0`` to disable.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


def _supports_fused_call_contract(switch_glu_cls) -> bool:
    """Fail closed if an upstream SwitchGLU changes its calling convention."""
    try:
        parameters = tuple(inspect.signature(switch_glu_cls.__call__).parameters)
    except (TypeError, ValueError):
        return False
    return parameters == ("self", "x", "indices")


def _switch_layer_families():
    """Return loaded compatible SwitchGLU families without importing mlx-vlm.

    mlx-vlm carries a private copy of the switch layers rather than re-exporting
    mlx-lm's classes. Importing mlx-vlm from a text-only process would defeat
    Rapid's optional vision dependency, so enroll that family only after its
    module is already resident (which is necessarily true for a loaded VLM).
    """
    families = []
    try:
        from mlx_lm.models.switch_layers import (
            QuantizedSwitchLinear,
            SwitchGLU,
            SwitchLinear,
            _gather_sort,
            _scatter_unsort,
        )
    except ImportError:
        pass
    else:
        if _supports_fused_call_contract(SwitchGLU):
            families.append(
                (
                    QuantizedSwitchLinear,
                    SwitchGLU,
                    SwitchLinear,
                    _gather_sort,
                    _scatter_unsort,
                )
            )

    vlm_switch = sys.modules.get("mlx_vlm.models.switch_layers")
    if vlm_switch is not None:
        try:
            family = (
                vlm_switch.QuantizedSwitchLinear,
                vlm_switch.SwitchGLU,
                vlm_switch.SwitchLinear,
                vlm_switch._gather_sort,
                vlm_switch._scatter_unsort,
            )
        except AttributeError:
            pass
        else:
            if _supports_fused_call_contract(family[1]) and all(
                family[1] is not existing[1] for existing in families
            ):
                families.append(family)
    return tuple(families)


def _can_fuse(switch_mlp: Any, quantized_cls, plain_cls) -> bool:
    """Structural gate: only fuse a gate/up pair the concat math is exact for."""
    if hasattr(switch_mlp, "gate_up_proj"):
        return False
    if not (
        hasattr(switch_mlp, "gate_proj")
        and hasattr(switch_mlp, "up_proj")
        and hasattr(switch_mlp, "down_proj")
    ):
        return False
    gate, up = switch_mlp.gate_proj, switch_mlp.up_proj
    if type(gate) is not type(up):
        return False
    if isinstance(gate, quantized_cls):
        if (gate.group_size, gate.bits, gate.mode) != (
            up.group_size,
            up.bits,
            up.mode,
        ):
            return False
        if (gate.get("biases") is None) != (up.get("biases") is None):
            return False
    elif not isinstance(gate, plain_cls):
        return False
    if ("bias" in gate) != ("bias" in up):
        return False
    gate_w, up_w = gate["weight"], up["weight"]
    return gate_w.shape == up_w.shape and gate_w.dtype == up_w.dtype


def _fuse_one(switch_mlp: Any, quantized_cls) -> None:
    gate, up = switch_mlp.gate_proj, switch_mlp.up_proj
    # Concat order is [gate, up] along the output axis (the HF
    # gate_up_proj checkpoint convention); the fused call splits in the
    # same order.
    fused = {"weight": mx.concatenate([gate["weight"], up["weight"]], axis=1)}
    if isinstance(gate, quantized_cls):
        fused["scales"] = mx.concatenate([gate["scales"], up["scales"]], axis=1)
        if gate.get("biases") is not None:
            fused["biases"] = mx.concatenate([gate["biases"], up["biases"]], axis=1)
    if "bias" in gate:
        fused["bias"] = mx.concatenate([gate["bias"], up["bias"]], axis=-1)
    mx.eval(list(fused.values()))

    # Reuse the gate module as the fused container so quantization params
    # and frozen state carry over; deleting gate_proj/up_proj drops the
    # original buffers.
    for name, array in fused.items():
        setattr(gate, name, array)
    switch_mlp.gate_up_proj = gate
    del switch_mlp.gate_proj
    del switch_mlp.up_proj


def _make_fused_call(orig_call, gather_sort, scatter_unsort):
    def fused_call(self, x: mx.array, indices: mx.array) -> mx.array:
        gate_up = getattr(self, "gate_up_proj", None)
        if gate_up is None:
            return orig_call(self, x, indices)

        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_gate_up = gate_up(x, idx, sorted_indices=do_sort)
        x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )
        if do_sort:
            x = scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)

    return fused_call


def _ensure_call_patch(switch_glu_cls, gather_sort, scatter_unsort) -> None:
    if getattr(switch_glu_cls, "_rapid_gate_up_fused_call", False):
        return
    orig = switch_glu_cls.__call__
    switch_glu_cls.__call__ = _make_fused_call(orig, gather_sort, scatter_unsort)
    switch_glu_cls._rapid_gate_up_fused_call = True
    switch_glu_cls._rapid_gate_up_original_call = orig


def fuse_gate_up(model: Any) -> int:
    """Fuse gate+up expert projections on a loaded MoE model, in place.

    Returns the number of fused ``SwitchGLU`` instances (0 when disabled
    via ``RAPID_MLX_MOE_GATE_UP_FUSION=0``, mlx-lm is unavailable, or the
    model has nothing eligible). Dense models are a cheap no-op — the
    module scan finds no ``SwitchGLU``. Idempotent: already-fused
    instances carry ``gate_up_proj`` and are skipped by the gate.
    """
    if os.environ.get("RAPID_MLX_MOE_GATE_UP_FUSION", "1") == "0":
        logger.info("[moe_fusion] disabled via RAPID_MLX_MOE_GATE_UP_FUSION=0")
        return 0
    families = _switch_layer_families()
    if not families:
        return 0

    try:
        modules = [m for _, m in model.named_modules()]
    except Exception:  # noqa: BLE001 — unusual model containers: no fusion
        return 0
    # Exact type only: a subclass may override __call__ and never consult
    # gate_up_proj, so fusing it would silently waste memory (or worse if
    # it reads gate_proj directly).
    fused_count = 0
    for (
        quantized_cls,
        switch_glu_cls,
        plain_cls,
        gather_sort,
        scatter_unsort,
    ) in families:
        targets = [
            m
            for m in modules
            if type(m) is switch_glu_cls and _can_fuse(m, quantized_cls, plain_cls)
        ]
        if not targets:
            continue
        _ensure_call_patch(switch_glu_cls, gather_sort, scatter_unsort)
        for switch_mlp in targets:
            _fuse_one(switch_mlp, quantized_cls)
            fused_count += 1
            # The freed gate/up buffers land in the MLX buffer pool. Drain per
            # fused layer so the load-time transient stays bounded to a single
            # layer's worth instead of ~2/3 of the whole routed expert set.
            mx.clear_cache()
    if fused_count:
        logger.info("[moe_fusion] gate+up fusion applied: %d MoE layers", fused_count)
    return fused_count
