# SPDX-License-Identifier: Apache-2.0
"""``model_type`` → MTP inject router.

Historically, callers of the vendored MTP path
(:mod:`vllm_mlx.spec_decode.mtp`) reached directly into
:mod:`vllm_mlx.spec_decode.mtp.qwen3_5_inject`. That was safe when
Qwen3.5 / Qwen3.6 were the only architectures on the allowlist
(``detect._SUPPORTED_MODEL_TYPES``). Growing the allowlist to
``gemma4`` / ``gemma4_unified`` (PR-2, the sibling PR that adds
detection support) means callers now need to pick the correct
family-specific inject at runtime — Gemma 4's MTP sidecar (the
``gemma4-assistant`` architecture shipped by Mia-AiLab's GGUF) is
structurally different from Qwen3.5's MTP head, so a single unified
inject would end up with a giant ``if model_type in {...}`` branch
inside a hot module. Instead, we keep the family-specific inject
helpers as-is and land the routing here.

This is intentionally the SMALLEST possible dispatcher — no config
mutation, no ``inject_mtp_support``-level side effects, no monkey
patching. It resolves the family, forwards the call, and returns the
bool the caller expects (see the ``bench/bench_spec_decode_mtp.py``
and ``vllm_mlx.utils.tokenizer`` caller shape).

Adding a new architecture
-------------------------

1. Write the family-specific inject module (see ``qwen3_5_inject.py``
   for the reference implementation, ``gemma4_inject.py`` for the
   safe-refusal template).
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
    # Qwen3.5 dense + MoE — historical path (R15 task #302 / mlx-lm PR #990).
    "qwen3_5": (
        "vllm_mlx.spec_decode.mtp.qwen3_5_inject",
        "inject_mtp_support",
    ),
    "qwen3_5_moe": (
        "vllm_mlx.spec_decode.mtp.qwen3_5_inject",
        "inject_mtp_support",
    ),
    # Gemma 4 — 12B unified (dense / 8bit) and the multimodal / MoE
    # variants (e2b, e4b, 26B-A4B). Both text_config model_types
    # (``gemma4_text``, ``gemma4_unified_text``) live under the same
    # outer ``gemma4`` / ``gemma4_unified`` wrapper, so the routing
    # happens on the OUTER type only.
    "gemma4": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "inject_mtp_support",
    ),
    "gemma4_unified": (
        "vllm_mlx.spec_decode.mtp.gemma4_inject",
        "inject_mtp_support",
    ),
}


# Same schema, but for ``validate_mtp_support``. Kept as a separate
# table so a family with a bespoke validator can register it
# independently of the inject entry. All entries currently share the
# ``validate_mtp_support`` symbol name.
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
        model_type: The ``config.json::model_type`` string. This is
            the OUTER wrapper's type (``gemma4``, not
            ``gemma4_text``) — that's the level our alias / detect
            paths already work at.
        mtp_sidecar: Optional sidecar reference — forwarded verbatim
            to the family-specific inject.
        allow_random_init: Test-only escape hatch — forwarded
            verbatim.

    Returns:
        ``True`` when the family-specific inject succeeded and the
        model now exposes the four MTP contract surfaces. ``False``
        when the model_type has no registered inject, when the
        family-specific inject refused (sidecar unresolvable, config
        missing ``mtp_num_hidden_layers``, architectural mismatch,
        or the family's scaffold-only production gate), or when the
        module import failed.

    Never raises — an unknown ``model_type`` is treated as "no MTP
    support for this arch" (log-and-return False), matching the
    fail-closed default the codex round-5 review installed on
    ``qwen3_5_inject.inject_mtp_support``.

    Scaffold vs production gating (codex round-7 blocking fix)
    ----------------------------------------------------------

    The dispatcher used to carry a table
    (``_PRODUCTION_SCAFFOLD_REFUSE``) shortcutting ``gemma4`` /
    ``gemma4_unified`` to ``False`` before reaching the family
    module. Codex round-7 flagged this as a two-place inconsistency
    — the family module could then never actually run through the
    dispatcher, which contradicted the reason we registered the
    entries in the first place.

    The dispatcher is now architecture-agnostic. Scaffold-only
    families (``gemma4_inject`` today) must gate themselves in
    their own ``inject_mtp_support``. The Gemma 4 module does this
    on FOUR redundant surfaces:

    1. ``mtp_sidecar is None and not allow_random_init`` → False
       (the fail-closed default matches the Qwen3.5 side).
    2. Sidecar fingerprints as ``gemma4-assistant`` → False.
    3. Non-assistant sidecar without the
       ``_accept_scaffold_sidecar_for_tests`` private opt-in → False
       (scaffold cannot drive multi-token KV cache).
    4. Successful inject stamps ``_mtp_is_scaffold`` so
       :func:`dispatch_mtp_validate` returns False even when
       :func:`dispatch_mtp_inject` returned True (the wiring-probe
       path with ``allow_random_init=True``).

    Points (1) + (4) together prevent any production caller from
    accidentally driving the Gemma 4 scaffold, and (2) + (3) keep
    even a well-meaning operator-supplied sidecar from mis-loading.
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
            "[mtp.dispatch] %s has no %s(); skipping.",
            module_path,
            func_name,
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
    """Route a ``validate_mtp_support`` call to the family-specific validator.

    Codex round-2 flagged that PR-3's bench call site injects Gemma 4
    via the dispatcher but validates via ``qwen3_5_inject.validate_mtp_support``,
    which reads the Qwen3.5 surface and misjudges the Gemma 4 patch as
    invalid. This helper routes the validator by the same
    ``model_type`` table.

    Returns ``False`` for any unknown model_type. Family-side scaffold
    gating (codex round-7): the family's own ``validate_mtp_support``
    is responsible for returning False on scaffold-only injects —
    ``gemma4_inject.validate_mtp_support`` checks the
    ``_mtp_is_scaffold`` marker set by the paired inject and returns
    False even when the surfaces attach.
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
