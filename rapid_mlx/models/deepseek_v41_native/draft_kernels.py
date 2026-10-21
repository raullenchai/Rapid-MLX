"""Acceptance-qualified draft kernels with deferred per-component finite checks."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import mlx.core as mx

from . import head_gemv, moe_decode

logger = logging.getLogger(__name__)
ENABLED = True


class _Projection(SimpleNamespace):
    def __contains__(self, name):
        return hasattr(self, name)


def _checked(owner, label, x, bits, candidate, stock):
    """Dispatch supported kernels without probes or host reads.

    Only finite-input failures belong to this component; propagated NaNs must
    not disable otherwise healthy downstream layers or the head.
    """
    disabled = getattr(owner, "_rapid_draft_disabled", ())
    if candidate is None or label in disabled:
        return stock(x)
    output = candidate(x)
    pending = getattr(owner, "_rapid_draft_pending", None)
    if pending is not None:
        healthy = mx.all(mx.isfinite(output)) | ~mx.all(mx.isfinite(x))
        pending.append((owner, label, bits, healthy))
    return output


def begin_round(draft):
    pending = []
    draft.adapter._rapid_draft_pending = pending
    draft.w._rapid_draft_pending = pending
    draft._draft_kernel_pending = pending


def round_status(draft):
    """Device vector explicitly evaluated at the existing readback boundary."""
    return (
        mx.array([True])
        if not draft._draft_kernel_pending
        else mx.stack([entry[3] for entry in draft._draft_kernel_pending])
    )


def disable_nonfinite(draft):
    """Consume already evaluated flags; permanently disable only failed kernels."""
    pending = getattr(draft, "_draft_kernel_pending", ())
    if not pending:
        return False
    changed = False
    for (owner, label, bits, _), healthy in zip(
        pending, draft._draft_kernel_status.tolist(), strict=True
    ):
        if healthy:
            continue
        disabled = set(getattr(owner, "_rapid_draft_disabled", ()))
        if label not in disabled:
            disabled.add(label)
            owner._rapid_draft_disabled = disabled
            logger.warning(
                "draft %s kernel non-finite (bits=%s); permanently using stock draft path",
                label,
                bits,
            )
            changed = True
    return changed


def head(weights, x):
    """Reuse the head GEMV's shared weight read for multi-row draft logits."""
    stock = lambda value: weights.linear("head", value)
    if not ENABLED:
        return stock(x)
    weight = weights.read("head.weight")
    if not head_gemv._supports_weight(weight, x):
        return stock(x)
    # Quantized/bias-bearing heads must retain their original projection.
    if "head.scales" in weights.entries or "head.bias" in weights.entries:
        return stock(x)
    # _launch selects the existing shared-read vocabulary kernel for multiple
    # rows. Keep draft finite checks here instead of target bitwise qualification.
    return _checked(
        weights,
        "head",
        x,
        str(weight.dtype),
        lambda value: head_gemv._launch(weight, value),
        stock,
    )


def moe(adapter, base, x, packed, route, stock):
    """Keep draft routing; reuse the target's fused expert up/down kernels."""
    if not ENABLED or not moe_decode.supports(x):
        return stock(x)
    config, weights = adapter.c, adapter.w

    def projection(values):
        return _Projection(
            **{k: v for k, v in values.items() if k != "config"}, **values["config"]
        )

    routed = [projection(packed[name]) for name in ("w1", "w3", "w2")]
    shared = []
    for name in ("w1", "w3", "w2"):
        prefix = f"{base}.shared_experts.{name}"
        values = {
            suffix: weights.read(f"{prefix}.{suffix}")
            for suffix in ("weight", "scales", "biases")
            if f"{prefix}.{suffix}" in weights.entries
        }
        values["config"] = weights.quant_config(prefix) if "scales" in values else {}
        if f"{prefix}.bias" in weights.entries:
            values["bias"] = weights.read(f"{prefix}.bias")
        shared.append(projection(values))
    limit = config["swiglu_limit"]
    layer = SimpleNamespace(
        experts=SimpleNamespace(
            gate_proj=routed[0],
            up_proj=routed[1],
            down_proj=routed[2],
            activation=SimpleNamespace(limit=limit),
        ),
        shared_experts=SimpleNamespace(
            w1=shared[0], w3=shared[1], w2=shared[2], limit=limit
        ),
        gate=SimpleNamespace(
            topk=config["dspark_num_experts_per_tok"],
            score_func="sqrtsoftplus",
            weight=weights.read(base + ".gate.weight"),
            bias=weights.read(base + ".gate.bias"),
        ),
    )
    layout = moe_decode._layout(layer, external_routing=True)
    bits = tuple(getattr(p, "bits", None) for p in routed + shared)

    def candidate(value):
        selected, picks = route(value)
        return moe_decode._experts(
            layer, layout, value, selected, picks.astype(mx.uint32)
        )

    return _checked(adapter, base, x, bits, candidate if layout else None, stock)
