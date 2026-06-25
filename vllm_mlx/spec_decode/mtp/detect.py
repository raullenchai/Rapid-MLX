# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 / Qwen3.6 MTP architecture detection (R15 task #302).

Detection lives off the loaded ``config.json`` dict rather than the
``aliases.json`` schema. Reasons:

* The closed-key schema in ``aliases.json`` only accepts the field set
  used by the existing alias profile (no ``architecture``, no
  ``family``, no ``quantization``, no ``notes`` — see
  ``knowledge/gotchas.md``). Adding an ``mtp_num_hidden_layers`` /
  ``mtp_capable`` field would silently fail at load.
* The ``mtp_num_hidden_layers`` value is an intrinsic property of the
  checkpoint, not of the alias. A user passing a raw HF path like
  ``Qwen/Qwen3.5-27B`` should still get MTP eligibility without us
  having to ship an alias for every Qwen3.5 / Qwen3.6 quant.
* ``model_type`` is already populated on every HF config and is the
  canonical anchor mlx-lm itself uses to route to a model class — we
  just piggyback on it.

Eligibility is binary right now (``MTP`` or ``NONE``). A future ``TREE``
variant would land here once upstream ships a ``mtp_num_hidden_layers
>= 2`` checkpoint, but as of vendoring date every released Qwen3.5 /
Qwen3.6 checkpoint ships ``mtp_num_hidden_layers: 1`` — chain MTP only.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any


class MTPEligibility(str, enum.Enum):
    """Result of :func:`detect_mtp_eligibility` on a loaded config dict.

    ``NONE`` — model architecture does not have an MTP head, or the
    config explicitly sets ``mtp_num_hidden_layers: 0`` (Qwen3.5 / 3.6
    checkpoints that have been re-converted with MTP stripped). The CLI
    must reject ``--spec-decode mtp`` in this case.

    ``CHAIN`` — model has a single MTP layer (``mtp_num_hidden_layers
    == 1``). One draft token per backbone step. This is what every
    upstream Qwen3.5 / Qwen3.6 release ships today.

    ``TREE`` — reserved for ``mtp_num_hidden_layers >= 2``. Not in use
    yet; emits a runtime warning and is treated as ``CHAIN`` until the
    tree-MTP code path lands.
    """

    NONE = "none"
    CHAIN = "chain"
    TREE = "tree"


# Architectures (``config.json::model_type``) that the upstream PR #990
# touches. Qwen3.5 dense / MoE share ``qwen3_5`` / ``qwen3_5_moe`` (the
# MoE subclasses the dense model and routes through the same MTP path).
# Qwen3.6 reuses the same code paths upstream; ``qwen3_5`` is the
# canonical ``model_type`` for the dense Qwen3.6 release too, while the
# MoE variant carries ``qwen3_5_moe``.
_SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "qwen3_5",
        "qwen3_5_moe",
    }
)


@dataclass(frozen=True)
class _DetectionResult:
    """Internal — surfaced through :func:`detect_mtp_eligibility`."""

    eligibility: MTPEligibility
    model_type: str | None
    num_mtp_layers: int
    reason: str


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce ``value`` to ``int``, returning ``default`` on bad input.

    ``config.json`` is operator-supplied. Some hand-edited configs ship
    string values (``"1"`` instead of ``1``); raw HF re-uploads have
    been known to ship floats (``1.0``). Both are silent-OK here.
    Anything we can't coerce — ``None``, ``"foo"``, lists — falls back
    to ``default`` so we degrade to ``NONE`` rather than crash boot.
    """
    if value is None:
        return default
    try:
        # ``int(True)`` returns ``1``; ``int(1.0)`` returns ``1``. Both
        # are acceptable.
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def detect_mtp_eligibility(config: dict[str, Any] | None) -> MTPEligibility:
    """Return the MTP eligibility class for a parsed ``config.json``.

    Args:
        config: The parsed ``config.json`` dict. ``None`` (or a
            non-dict) returns ``MTPEligibility.NONE`` — used by the CLI
            so callers can pass ``model_auto_config.get_config(path)``
            output unguarded.

    Returns:
        :class:`MTPEligibility` value. Detection is conservative — any
        ambiguity (unsupported ``model_type``, ``mtp_num_hidden_layers``
        absent or zero, structurally-broken config) collapses to
        ``NONE`` so ``--spec-decode mtp`` on an ineligible model is
        rejected at boot rather than silently emitting wrong tokens.
    """
    result = _detect_mtp_eligibility_verbose(config)
    return result.eligibility


def _detect_mtp_eligibility_verbose(
    config: dict[str, Any] | None,
) -> _DetectionResult:
    """Detection helper that returns the full reason string.

    Tests assert on ``reason`` to lock the contract — keep the
    short-strings stable across versions.
    """
    if not isinstance(config, dict):
        return _DetectionResult(
            MTPEligibility.NONE, None, 0, "config is not a dict"
        )

    model_type = config.get("model_type")
    if not isinstance(model_type, str):
        return _DetectionResult(
            MTPEligibility.NONE, None, 0, "model_type missing or not a string"
        )

    if model_type not in _SUPPORTED_MODEL_TYPES:
        return _DetectionResult(
            MTPEligibility.NONE,
            model_type,
            0,
            f"model_type {model_type!r} not in MTP allowlist",
        )

    num_mtp_layers = _safe_int(config.get("mtp_num_hidden_layers"), 0)
    if num_mtp_layers <= 0:
        # Qwen3.5 / Qwen3.6 model_type but MTP stripped from this checkpoint —
        # operator must re-convert from HF with the PR #990 sanitize() path
        # that preserves ``mtp.*`` weights. See task report for the conversion
        # SOP.
        return _DetectionResult(
            MTPEligibility.NONE,
            model_type,
            num_mtp_layers,
            "mtp_num_hidden_layers <= 0 (MTP weights stripped at convert time)",
        )

    if num_mtp_layers == 1:
        return _DetectionResult(
            MTPEligibility.CHAIN,
            model_type,
            num_mtp_layers,
            "single MTP layer (chain mode, 1 draft / verify)",
        )

    # num_mtp_layers >= 2 — reserved for future tree MTP. Treat as
    # CHAIN for now; the generator only consumes the first layer until
    # the tree code path lands.
    return _DetectionResult(
        MTPEligibility.TREE,
        model_type,
        num_mtp_layers,
        f"{num_mtp_layers} MTP layers (tree variant — not yet implemented; "
        "running as chain on the first layer)",
    )
