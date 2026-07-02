# SPDX-License-Identifier: Apache-2.0
"""``model_type`` → MTP inject router.

Historically, callers of the vendored MTP path
(:mod:`vllm_mlx.spec_decode.mtp`) reached directly into
:mod:`vllm_mlx.spec_decode.mtp.qwen3_5_inject`. Growing the allowlist
to ``gemma4`` / ``gemma4_unified`` (paired with Google's official
``google/gemma-4-*-it-assistant`` drafter checkpoints, Apache 2.0)
means callers now need to pick the correct family-specific inject at
runtime. The Gemma 4 drafter is a 4-layer transformer that reuses
target K/V, structurally different from the Qwen3.5 MTP head. Instead
of a giant ``if model_type in {...}`` inside a hot module, we keep the
family implementations as separate modules and land the routing here.

This module is intentionally the smallest possible dispatcher — no
config mutation, no monkey-patching. It resolves the family, forwards
the call, and returns the bool the caller expects (see the
``bench/bench_spec_decode_mtp.py`` and ``vllm_mlx.utils.tokenizer``
caller shape).

Adding a new architecture
-------------------------

1. Write the family-specific inject module.
2. Add the ``model_type`` string(s) to :data:`_MTP_INJECT_DISPATCH`
   below, mapping to the module path + entry function name.
3. Add the ``model_type`` to ``detect._SUPPORTED_MODEL_TYPES`` so the
   CLI can advertise ``--spec-decode mtp`` for that architecture at
   parse time.

All three steps are strictly additive — existing architectures keep
their current call sites.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Map ``config.json::model_type`` → ``(module_dotted_path,
# entry_function_name)``. The module is imported lazily on first
# dispatch so this file stays cheap to import from top-level MTP.
_MTP_INJECT_DISPATCH: dict[str, tuple[str, str]] = {
    # Qwen3.5 dense + MoE (vendored mlx-lm PR #990 MTP head).
    "qwen3_5": (
        "vllm_mlx.spec_decode.mtp.qwen3_5_inject",
        "inject_mtp_support",
    ),
    "qwen3_5_moe": (
        "vllm_mlx.spec_decode.mtp.qwen3_5_inject",
        "inject_mtp_support",
    ),
    # Gemma 4 — paired with Google's official
    # ``google/gemma-4-*-it-assistant`` drafter checkpoints. Both the
    # outer wrapper model_types (``gemma4`` / ``gemma4_unified``) and
    # the inner text model_types (``gemma4_text`` /
    # ``gemma4_unified_text``) route to the same ``gemma4_inject``
    # module so callers that resolve model_type on the inner
    # ``language_model.args`` still land correctly.
    "gemma4": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "inject_mtp_support",
    ),
    "gemma4_unified": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "inject_mtp_support",
    ),
    "gemma4_text": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "inject_mtp_support",
    ),
    "gemma4_unified_text": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "inject_mtp_support",
    ),
}


# Same schema, but for ``validate_mtp_support``. Kept as a separate
# table so a family with a bespoke validator can register it
# independently of the inject entry.
_MTP_VALIDATE_DISPATCH: dict[str, tuple[str, str]] = {
    "qwen3_5": (
        "vllm_mlx.spec_decode.mtp.qwen3_5_inject",
        "validate_mtp_support",
    ),
    "qwen3_5_moe": (
        "vllm_mlx.spec_decode.mtp.qwen3_5_inject",
        "validate_mtp_support",
    ),
    "gemma4": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "validate_mtp_support",
    ),
    "gemma4_unified": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "validate_mtp_support",
    ),
    "gemma4_text": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "validate_mtp_support",
    ),
    "gemma4_unified_text": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "validate_mtp_support",
    ),
}


def dispatch_mtp_inject(
    model: Any,
    model_type: str,
    *,
    mtp_sidecar: str | Path | None = None,
    allow_random_init: bool = False,
) -> bool:
    """Route an inject call to the family-specific implementation.

    Args:
        model: Loaded model instance (from ``mlx_lm.load()``).
        model_type: The ``config.json::model_type`` string.
        mtp_sidecar: Optional sidecar reference — forwarded verbatim.
        allow_random_init: Test-only escape hatch — forwarded verbatim.

    Returns:
        ``True`` when the family inject succeeded and the model now
        exposes the four MTP contract surfaces. ``False`` on any
        refusal — unknown ``model_type``, family-level fail-closed
        default, missing sidecar, or import failure.

    Never raises — an unknown ``model_type`` is treated as "no MTP
    support for this arch" (log-and-return False), matching the
    fail-closed default the qwen3_5 side installed.
    """
    key = _MTP_INJECT_DISPATCH.get(model_type)
    if key is None:
        logger.info(
            "[mtp.dispatch] model_type=%r has no registered MTP inject; "
            "skipping. Registered: %s",
            model_type,
            sorted(_MTP_INJECT_DISPATCH),
        )
        return False

    module_path, func_name = key
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        logger.warning(
            "[mtp.dispatch] could not import %s for model_type=%r: %s",
            module_path,
            model_type,
            exc,
        )
        return False

    fn = getattr(module, func_name, None)
    if fn is None:
        logger.warning(
            "[mtp.dispatch] %s has no %s(); skipping.", module_path, func_name
        )
        return False

    return bool(
        fn(
            model,
            mtp_sidecar=mtp_sidecar,
            allow_random_init=allow_random_init,
        )
    )


def dispatch_mtp_validate(model: Any, model_type: str) -> bool:
    """Route a ``validate_mtp_support`` call to the family validator.

    Returns ``False`` for any unknown ``model_type``.
    """
    key = _MTP_VALIDATE_DISPATCH.get(model_type)
    if key is None:
        logger.info(
            "[mtp.dispatch] model_type=%r has no registered validator; "
            "returning False.",
            model_type,
        )
        return False

    module_path, func_name = key
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        logger.warning(
            "[mtp.dispatch] could not import %s for validator: %s",
            module_path,
            exc,
        )
        return False

    fn = getattr(module, func_name, None)
    if fn is None:
        return False
    return bool(fn(model))
