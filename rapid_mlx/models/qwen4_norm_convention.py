# SPDX-License-Identifier: Apache-2.0
"""Admit Qwen4 RMSNorm storage using unanimous producer-level anchors.

Non-anchor learned gains can cross the midpoint in either convention, so they
are validated for tensor safety but do not provide independent convention
evidence.
"""

from __future__ import annotations

import mlx.core as mx

_FP32_ABSOLUTE_ROUNDING_BUDGET = 4 * 2**-23


def _norm_targets(model, norm_type, prefix=""):
    return {
        f"{prefix}{path}.weight": module
        for path, module in model.named_modules()
        if type(module) is norm_type
    }


def _validated_mean(key, module, value):
    if (
        not isinstance(value, mx.array)
        or not mx.issubdtype(value.dtype, mx.floating)
        or value.shape != module.weight.shape
        or not bool(mx.all(mx.isfinite(value)).item())
    ):
        raise ValueError(f"invalid Qwen4 RMSNorm tensor: {key}")
    return float(mx.mean(value.astype(mx.float32)).item())


def apply_qwen4_norm_convention(model, weights, norm_type, convention, *, prefix=""):
    """Map only the exact zero-centered norm class into its residual ABI.

    The checkpoint's attention hyper-connection anchors establish the source
    convention before this function maps the affected norms. Gated GDN norms
    are a different class, already direct-gamma, and remain unchanged.
    """
    if convention not in ("zero_centered", "direct_gamma"):
        raise ValueError("Qwen4 RMSNorm requires an admitted backbone convention")
    replacements = {}
    for key, module in _norm_targets(model, norm_type, prefix).items():
        if key not in weights:
            continue  # Strict checkpoint loading owns missing-key errors.
        value = weights[key]
        if (
            not isinstance(value, mx.array)
            or not mx.issubdtype(value.dtype, mx.floating)
            or value.shape != module.weight.shape
            or not bool(mx.all(mx.isfinite(value)).item())
        ):
            raise ValueError(f"invalid Qwen4 RMSNorm tensor: {key}")
        if convention == "direct_gamma":
            # BF16 subtraction loses small trained MTP gains. Keep the
            # residual in FP32 before the runtime's 1 + weight operation.
            gamma = value.astype(mx.float32)
            residual = gamma - 1.0
            restored = 1.0 + residual
            # Subtracting and re-adding one can move a gain by a few FP32 ULPs.
            # The absolute term is required near zero, where relative error is
            # not a meaningful representability test for the residual ABI.
            # Still reject total cancellation of a nonzero trained gain.
            lost_nonzero = mx.any((gamma != 0) & (restored == 0))
            if bool(lost_nonzero.item()) or not bool(
                mx.allclose(
                    restored,
                    gamma,
                    rtol=5e-7,
                    atol=_FP32_ABSOLUTE_ROUNDING_BUDGET,
                ).item()
            ):
                raise ValueError(
                    "Qwen4 RMSNorm gain cannot be represented faithfully "
                    f"as an FP32 residual: {key}"
                )
            replacements[key] = residual
    weights.update(replacements)
    return len(replacements)


def normalize_qwen4_checkpoint(model, weights, norm_type):
    """Detect complete backbone anchor evidence, then canonicalize once."""
    prefix = (
        "language_model."
        if any(key.startswith("language_model.") for key in weights)
        else ""
    )
    targets = _norm_targets(model, norm_type, prefix)
    if not (targets.keys() & weights.keys()):
        return None
    anchors = {
        key: module
        for key, module in targets.items()
        if key.endswith(".attn_hyper_connection.hc_norm.weight")
    }
    if not anchors or not anchors.keys() <= weights.keys():
        raise ValueError("Qwen4 RMSNorm convention requires complete backbone anchors")
    target_means = {
        key: _validated_mean(key, module, weights[key])
        for key, module in targets.items()
        if key in weights
    }
    means = [target_means[key] for key in anchors]
    ordered = sorted(means)
    n = len(ordered)
    median = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2
    ones_vote = sum(value > 0.5 for value in means) / n
    if ones_vote == 1.0 and 0.75 <= median <= 1.5:
        convention = "direct_gamma"
    elif (
        ones_vote == 0.0
        and all(value < 0.5 for value in means)
        and -0.5 <= median <= 0.25
    ):
        convention = "zero_centered"
    else:
        raise ValueError(
            "ambiguous or mixed Qwen4 RMSNorm checkpoint convention: "
            f"{n} anchors, median={median:.6g}, direct-gamma vote={ones_vote:.3f}"
        )
    previous = getattr(model, "norm_convention_receipt", None)
    if previous and previous["source_convention"] != convention:
        raise ValueError(
            "Qwen4 RMSNorm source convention changed on an admitted model; "
            "load a different checkpoint into a fresh model"
        )
    converted = apply_qwen4_norm_convention(
        model, weights, norm_type, convention, prefix=prefix
    )
    return {
        "source_convention": convention,
        "runtime_convention": "zero_centered",
        "anchor_count": n,
        "anchor_median": median,
        "direct_gamma_vote": ones_vote,
        "recentered_tensors": converted,
        "detection": "complete_unanimous_attention_hc_anchor_band_v2",
    }
