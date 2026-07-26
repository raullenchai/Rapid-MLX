# SPDX-License-Identifier: Apache-2.0
"""Correct mlx-lm's spurious RMSNorm-weight +1.0 shift on Qwen3.5/3.6 VLM checkpoints.

Background
----------
``mlx_lm.models.qwen3_5.TextModel.sanitize`` decides whether to add ``+1.0``
to every RMSNorm gain via a *proxy*::

    has_mtp_weights      = any("mtp." in k for k in weights)
    has_unsanitized_conv1d = any("conv1d.weight" in k and v.shape[-1] != 1 ...)
    should_shift_norm_weights = has_mtp_weights or has_unsanitized_conv1d

The shift exists to convert checkpoints whose norm gains are stored in the
"``1 + w`` zero-centered" convention (gains stored centered at 0) into the
standard convention mlx-lm's ``nn.RMSNorm`` expects (gains centered at 1). The
``mtp.``-presence / unsanitized-``conv1d`` signals were chosen as a proxy for
"this checkpoint uses the zero-centered convention".

The proxy misfires on the ``mlx-community/Qwen3.6-35B-A3B-*`` MLX quants (and
any sibling VLM checkpoint that bundles an MTP head). These checkpoints:

* bundle an MTP sidecar head (``mtp.norm.weight``, ``mtp.layers.0.*``) →
  ``has_mtp_weights`` is True, so the proxy fires; **but**
* store their norm gains in the **standard** convention already (per-tensor
  means ~1, aggregate ~1.2, never ~0).

So mlx-lm adds ``+1.0`` to gains that were already ~1, yielding ~2 — a doubled
RMSNorm scale that corrupts generation into garbage from the first token.
``mlx-vlm`` loads the identical weights **without** shifting the
``language_model.*`` norms and generates coherently, confirming no shift is the
correct behavior. Reported upstream: ml-explore/mlx-lm#1197.

Verified: loading ``mlx-community/Qwen3.6-35B-A3B-4bit`` with this patch active
produces coherent output ("The capital of Japan is Tokyo", "17 × 23 = 391 …");
without it, garbage. The 105 norm tensors' loaded means drop from ~2.0 to ~1.0.

What this module patches
------------------------
Wraps ``mlx_lm.models.qwen3_5.TextModel.sanitize`` (the single site that owns
the shift — the MoE class ``qwen3_5_moe.Model`` and the VLM wrapper
``qwen3_5.Model`` both delegate norm handling to it). The wrapper:

1. Inspects the **raw** ``weights`` BEFORE the original runs and computes
   whether the original will *spuriously* shift: the proxy fires
   (``has_mtp``/unsanitized ``conv1d``) **and** the norm gains are already in
   standard form (aggregate mean ``>= _ZERO_CENTERED_THRESHOLD``, i.e. NOT
   zero-centered).
2. Calls the original ``sanitize`` verbatim (so all its other behavior —
   ``mtp.`` stripping, tied-embedding drop, ``conv1d`` moveaxis — is inherited).
3. Only if step 1 flagged a spurious shift **and** the original actually
   applied it (a representative norm gain rose by ~1.0 across the call) does it
   subtract the erroneous ``+1.0`` back from the shifted norm tensors.

The step-3 "actually applied" re-check keys the correction off observed
behavior, not off a copy of upstream's gate — so if upstream later fixes the
misfire (stops shifting standard-form checkpoints), this wrapper detects no
shift happened and does nothing. Delete this module + the import/call in
``vllm_mlx.model_runner`` once mlx-lm ships the fix; nothing else changes.

What it does NOT change
-----------------------
* Checkpoints whose norms are genuinely zero-centered (aggregate mean ~0) keep
  the shift — the correction only fires when gains are already standard-form,
  so a currently-working checkpoint (shift correct, or proxy False) is never
  altered. Adding +1.0 to standard-form gains is always wrong (it doubles the
  RMSNorm scale); undoing it is therefore always safe.
* Non-Qwen3.5 architectures are untouched (only ``qwen3_5.TextModel``).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_INSTALLED = False

# The exact norm-weight suffixes mlx-lm's qwen3_5 sanitize shifts (kept in sync
# with ``mlx_lm.models.qwen3_5.TextModel.sanitize``'s ``norm_keys``). The
# ``linear_attn.norm.weight`` gains are deliberately absent — mlx-lm does not
# shift them, and neither must the correction.
_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)

# Aggregate norm-gain mean below this => "zero-centered" storage (shift is
# legitimate). At/above => "standard" storage (a proxy-triggered shift is
# spurious). 0.5 is the natural midpoint between gains centered at 0 and gains
# centered at 1; observed values are ~0 (zero-centered) vs ~1.2 (Qwen3.6),
# leaving a wide margin.
_ZERO_CENTERED_THRESHOLD = 0.5

# Delta above which a representative norm gain is considered to have been
# shifted by the original sanitize (the shift adds exactly 1.0).
_SHIFT_DETECT_MIN = 0.5


def _is_norm_key(key: str) -> bool:
    return any(key.endswith(sfx) for sfx in _NORM_SUFFIXES)


def _norm_gain_mean(value: Any) -> float | None:
    """Mean of a 1-D norm gain tensor, or ``None`` if it isn't one."""
    if getattr(value, "ndim", None) != 1:
        return None
    try:
        return float(value.mean())
    except Exception:  # pragma: no cover - defensive; quantized 1-D shouldn't hit
        return None


def _would_spuriously_shift(weights: dict) -> bool:
    """True when the original sanitize would shift, but the norms are already
    standard-form (so the shift corrupts them)."""
    has_mtp = any("mtp." in k for k in weights)
    has_unsanitized_conv1d = any(
        "conv1d.weight" in k and getattr(v, "shape", (1,))[-1] != 1
        for k, v in weights.items()
    )
    if not (has_mtp or has_unsanitized_conv1d):
        return False
    means = [
        m
        for k, v in weights.items()
        if _is_norm_key(k) and (m := _norm_gain_mean(v)) is not None
    ]
    if not means:
        return False
    aggregate = sum(means) / len(means)
    return aggregate >= _ZERO_CENTERED_THRESHOLD


def _sanitize_applied_shift(raw: dict, result: dict) -> bool:
    """True when the original sanitize actually added ~1.0 to the norm gains.

    Keys the correction off observed behavior rather than a copy of upstream's
    gate, so an eventual upstream fix (no shift) is auto-detected as "no undo".
    ``TextModel.sanitize`` does not rename keys, so a norm key present in the
    input is present unchanged in the output.
    """
    for k, rv in raw.items():
        if not _is_norm_key(k) or k not in result:
            continue
        rm = _norm_gain_mean(rv)
        om = _norm_gain_mean(result[k])
        if rm is None or om is None:
            continue
        return (om - rm) >= _SHIFT_DETECT_MIN
    return False


def install_qwen3_5_norm_shift_fix() -> None:
    """Install the Qwen3.5/3.6 norm-shift correction. Idempotent + thread-safe."""
    global _INSTALLED

    with _LOCK:
        if _INSTALLED:
            return

        try:
            from mlx_lm.models import qwen3_5 as q
        except ImportError:  # pragma: no cover - mlx_lm always present in our env
            logger.debug("[qwen3_5_norm_shift] mlx_lm not importable; skipping install")
            return

        # Stash the upstream original on the module the first time so a later
        # install after a module reload retrieves the true original instead of
        # re-wrapping the already-patched callable (would infinite-loop).
        if not getattr(q, "_RAPID_MLX_NORM_SHIFT_INSTALLED", False):
            q._RAPID_MLX_ORIG_TEXTMODEL_SANITIZE = q.TextModel.sanitize

        orig_sanitize = q._RAPID_MLX_ORIG_TEXTMODEL_SANITIZE

        # Re-entry guard: another module instance already patched the class.
        if getattr(q, "_RAPID_MLX_NORM_SHIFT_INSTALLED", False):
            _INSTALLED = True
            return

        def _patched_sanitize(self, weights):
            spurious = _would_spuriously_shift(weights)
            result = orig_sanitize(self, weights)
            if spurious and _sanitize_applied_shift(weights, result):
                shifted = 0
                for k in list(result):
                    v = result[k]
                    if _is_norm_key(k) and getattr(v, "ndim", None) == 1:
                        result[k] = v - 1.0
                        shifted += 1
                logger.info(
                    "[qwen3_5_norm_shift] undid spurious +1.0 norm shift on "
                    "%d standard-form gains (mtp-bundled Qwen3.5/3.6 checkpoint)",
                    shifted,
                )
            return result

        q.TextModel.sanitize = _patched_sanitize
        q._RAPID_MLX_NORM_SHIFT_INSTALLED = True
        _INSTALLED = True
        logger.debug("[qwen3_5_norm_shift] installed")


def uninstall_qwen3_5_norm_shift_fix() -> None:
    """Undo the patch. Test-only; production code does not call this."""
    global _INSTALLED

    with _LOCK:
        if not _INSTALLED:
            return
        try:
            from mlx_lm.models import qwen3_5 as q
        except ImportError:  # pragma: no cover
            return
        orig = getattr(q, "_RAPID_MLX_ORIG_TEXTMODEL_SANITIZE", None)
        if orig is not None:
            q.TextModel.sanitize = orig
        q._RAPID_MLX_NORM_SHIFT_INSTALLED = False
        _INSTALLED = False


def is_installed() -> bool:
    return _INSTALLED
