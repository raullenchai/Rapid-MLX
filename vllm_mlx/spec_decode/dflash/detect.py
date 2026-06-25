# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 / Qwen3.6 DFlash architecture detection (R15 task #313).

Detection lives off the loaded ``config.json`` dict PLUS the
:mod:`vllm_mlx.spec_decode.dflash.drafter_registry` side-table.
Reasons (mirror the MTP detect.py rationale; both backends share the
same closed-key schema constraint on ``aliases.json``):

* The closed-key schema in ``aliases.json`` only accepts the field set
  documented in :data:`vllm_mlx.model_aliases._ALLOWED_PROFILE_KEYS`.
  Adding ``dflash_drafter`` would silently fail at load — and the
  existing ``dflash_draft_model`` field is reserved by the original
  mlx-vlm DFlash bridge (:mod:`vllm_mlx.speculative.dflash`), which
  carries its own eligibility gates we don't want to disturb. So the
  spec-decode drafter binding lives in a side-registry instead.
* ``model_type`` is already populated on every HF config and is the
  canonical anchor mlx-lm itself uses to route to a model class — we
  piggyback on it for the architecture allowlist.

Eligibility ladder
------------------

* :attr:`DFlashEligibility.NONE` — model architecture not supported, or
  no drafter is bound for this alias. CLI must reject
  ``--spec-decode dflash`` here.
* :attr:`DFlashEligibility.READY` — Qwen3.5 / Qwen3.6 + a drafter is
  bound. The CLI proceeds; the generator loads the drafter lazily.

The contract is deliberately binary today. A future ``EAGER_FALLBACK``
state could land for cases where the drafter binding is missing but
the operator passed ``--dflash-drafter-path`` on the CLI to override —
the runtime check sits at CLI parse time (cli.py) so this module's
binary contract stays clean.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any


class DFlashEligibility(str, enum.Enum):
    """Result of :func:`detect_dflash_eligibility`.

    * ``NONE`` — model architecture is not on the Qwen3.5 / 3.6
      allowlist, the config dict is malformed, or no drafter is bound
      for the alias. The CLI rejects ``--spec-decode dflash`` in this
      case.
    * ``READY`` — model is Qwen3.5 / 3.6 AND a drafter HF path is
      bound for the alias (either via the side-registry or via the
      CLI override). The generator can load the drafter.
    """

    NONE = "none"
    READY = "ready"


# Architectures (``config.json::model_type``) that the DFlash paper /
# z-lab reference drafter covers. Same allowlist as the MTP detect
# module — every drafter checkpoint published so far targets one of
# the dense or MoE Qwen3.5 / 3.6 model_types.
_SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "qwen3_5",
        "qwen3_5_moe",
    }
)


@dataclass(frozen=True)
class _DetectionResult:
    """Internal — surfaced through :func:`detect_dflash_eligibility`.

    Tests assert on ``reason`` to lock the contract — keep the
    short-strings stable across versions.
    """

    eligibility: DFlashEligibility
    model_type: str | None
    drafter_path: str | None
    reason: str


def detect_dflash_eligibility(
    config: dict[str, Any] | None,
    *,
    alias: str | None = None,
    drafter_override: str | None = None,
) -> DFlashEligibility:
    """Return the DFlash eligibility class for a parsed ``config.json``.

    Args:
        config: The parsed ``config.json`` dict. ``None`` (or a non-dict)
            returns :attr:`DFlashEligibility.NONE`.
        alias: Optional alias name; the side-registry is consulted for
            a default drafter binding. Pass ``None`` when the model was
            loaded by raw HF path (no alias resolved).
        drafter_override: Optional CLI ``--dflash-drafter-path`` value
            that bypasses the side-registry lookup. A non-empty string
            here ALWAYS satisfies the drafter-binding gate, even if the
            alias has no registered drafter.

    Returns:
        :class:`DFlashEligibility`. Detection is conservative — any
        ambiguity (unsupported ``model_type``, missing drafter binding,
        structurally-broken config) collapses to ``NONE`` so
        ``--spec-decode dflash`` on an ineligible config rejects at
        boot rather than silently emitting wrong tokens.
    """
    result = _detect_dflash_eligibility_verbose(
        config, alias=alias, drafter_override=drafter_override
    )
    return result.eligibility


def _detect_dflash_eligibility_verbose(
    config: dict[str, Any] | None,
    *,
    alias: str | None = None,
    drafter_override: str | None = None,
) -> _DetectionResult:
    """Detection helper that returns the full reason string."""
    if not isinstance(config, dict):
        return _DetectionResult(
            DFlashEligibility.NONE, None, None, "config is not a dict"
        )

    model_type = config.get("model_type")
    if not isinstance(model_type, str):
        return _DetectionResult(
            DFlashEligibility.NONE,
            None,
            None,
            "model_type missing or not a string",
        )

    if model_type not in _SUPPORTED_MODEL_TYPES:
        return _DetectionResult(
            DFlashEligibility.NONE,
            model_type,
            None,
            f"model_type {model_type!r} not in DFlash allowlist",
        )

    # Drafter binding: CLI override wins over the side-registry lookup
    # so an operator can experiment with a custom drafter checkpoint
    # without editing the registry. An empty-string override falls back
    # to the registry (matches argparse's default-empty-on-omit behaviour).
    drafter_path: str | None = None
    if drafter_override:
        drafter_path = drafter_override
    elif alias:
        # Local import to avoid a top-level cycle: the registry imports
        # are cheap (pure dict ops) but threading them through
        # ``__init__`` would force the dataclass / enum import here.
        from .drafter_registry import get_dflash_drafter_path

        drafter_path = get_dflash_drafter_path(alias)

    if not drafter_path:
        return _DetectionResult(
            DFlashEligibility.NONE,
            model_type,
            None,
            (
                "no drafter bound for alias "
                f"{alias!r} (register via "
                "vllm_mlx.spec_decode.dflash.register_dflash_drafter or "
                "pass --dflash-drafter-path on the CLI)"
            ),
        )

    return _DetectionResult(
        DFlashEligibility.READY,
        model_type,
        drafter_path,
        f"Qwen3.5/3.6 + drafter {drafter_path!r} bound",
    )
