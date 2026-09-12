# SPDX-License-Identifier: Apache-2.0
"""Admit raw or converted Qwen4 RMSNorm storage before loading parameters.

The anchor vote and median bands follow oMLX's Qwen4 compatibility loader.
Unified MLX likewise distinguishes raw residuals from converted direct gains.
This is evidence of a checkpoint-wide convention, not a way to prove that
arbitrary mixtures in individual trained norm tensors are impossible.
"""

from __future__ import annotations

import mlx.core as mx


def _norm_targets(model, norm_type, prefix=""):
    return {
        f"{prefix}{path}.weight": module
        for path, module in model.named_modules()
        if type(module) is norm_type
    }


def apply_qwen4_norm_convention(model, weights, norm_type, convention, *, prefix=""):
    """Map only the exact zero-centered norm class into its residual ABI.

    MTP must inherit the backbone decision: its learned gains cannot reliably
    classify the source convention independently. Gated GDN norms are a
    different class, already direct-gamma, and are never changed here.
    """
    if convention not in ("zero_centered", "direct_gamma"):
        raise ValueError("Qwen4 RMSNorm requires an admitted backbone convention")
    replacements = {}
    for key, module in _norm_targets(model, norm_type, prefix).items():
        if key not in weights:
            continue  # The strict checkpoint loader owns missing-key errors.
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
            # residual in FP32, as in the established oMLX loader.
            gamma = value.astype(mx.float32)
            residual = gamma - 1.0
            restored = 1.0 + residual
            if not bool(mx.all(restored.view(mx.uint32) == gamma.view(mx.uint32)).item()):
                raise ValueError(
                    f"Qwen4 RMSNorm gain cannot be represented exactly as an FP32 residual: {key}"
                )
            replacements[key] = residual
    weights.update(replacements)
    return len(replacements)


def normalize_qwen4_checkpoint(model, weights, norm_type):
    """Detect complete backbone anchor evidence, then canonicalize once.

    Partial mappings containing no actual norm parameters remain usable by
    conversion/key-remapping tools. A norm-bearing mapping must include every
    instantiated attention HC anchor; strict model loading checks other keys.
    """
    prefix = "language_model." if any(
        key.startswith("language_model.") for key in weights
    ) else ""
    targets = _norm_targets(model, norm_type, prefix)
    if not (targets.keys() & weights.keys()):
        return None
    anchors = {
        key: module for key, module in targets.items()
        if key.endswith(".attn_hyper_connection.hc_norm.weight")
    }
    if not anchors or not anchors.keys() <= weights.keys():
        raise ValueError("Qwen4 RMSNorm convention requires complete backbone anchors")
    means = []
    for key, module in anchors.items():
        value = weights[key]
        if (
            not isinstance(value, mx.array)
            or not mx.issubdtype(value.dtype, mx.floating)
            or value.shape != module.weight.shape
            or not bool(mx.all(mx.isfinite(value)).item())
        ):
            raise ValueError(f"invalid Qwen4 RMSNorm anchor: {key}")
        means.append(float(mx.mean(value.astype(mx.float32)).item()))
    ordered = sorted(means)
    n = len(ordered)
    median = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2
    ones_vote = sum(value > 0.5 for value in means) / n
    if ones_vote >= 0.9 and 0.75 <= median <= 1.5:
        convention = "direct_gamma"
    elif ones_vote <= 0.1 and -0.5 <= median <= 0.25:
        convention = "zero_centered"
    else:
        raise ValueError(
            "ambiguous or mixed Qwen4 RMSNorm checkpoint convention: "
            f"{n} anchors, median={median:.6g}, direct-gamma vote={ones_vote:.3f}"
        )
    previous = getattr(model, "norm_convention_receipt", None)
    if previous and previous["source_convention"] != convention:
        # Re-sanitizing already canonicalized weights must not overwrite the
        # source convention subsequently used for the original MTP sidecar.
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
        "detection": "complete_attention_hc_anchor_vote_v1",
    }
