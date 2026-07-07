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
  having to ship an alias for every Qwen3.5 / Qwen3.6 quant. MLX
  community Qwen3.5 / Qwen3.6 configs currently carry the value under
  ``text_config.mtp_num_hidden_layers``; hand-converted configs may put
  it at the root. Detection accepts both shapes.
* ``model_type`` is already populated on every HF config and is the
  canonical anchor mlx-lm itself uses to route to a model class — we
  just piggyback on it.

Eligibility is binary right now (``CHAIN`` or ``NONE``). A future
``TREE`` variant would land here once upstream ships a
``mtp_num_hidden_layers >= 2`` checkpoint, but as of vendoring date
every released Qwen3.5 / Qwen3.6 checkpoint ships
``mtp_num_hidden_layers: 1`` — chain MTP only.
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
    must reject ``--speculative-config '{"method":"mtp"}'`` in this case.

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


# Architectures (``config.json::model_type``) whose model class ships an
# MTP head that this engine knows how to drive.
#
# Upstream mlx-lm PR #990 Qwen3.5 / Qwen3.6:
# - ``qwen3_5``     — dense (also the canonical model_type for the dense
#                     Qwen3.6 release).
# - ``qwen3_5_moe`` — MoE variant; subclasses the dense model and routes
#                     through the same MTP path.
#
_SUPPORTED_MODEL_TYPES: frozenset[str] = frozenset(
    {
        # Qwen3.5 / Qwen3.6 (upstream PR #990)
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


def _mtp_num_hidden_layers(config: dict[str, Any]) -> int:
    """Return MTP layer count from root or nested text_config metadata."""

    root_layers = _safe_int(config.get("mtp_num_hidden_layers"), 0)
    if root_layers > 0:
        return root_layers
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return _safe_int(text_config.get("mtp_num_hidden_layers"), 0)
    return root_layers


def detect_mtp_eligibility(
    config: dict[str, Any] | None,
    *,
    has_external_sidecar: bool = False,
) -> MTPEligibility:
    """Return the MTP eligibility class for a parsed ``config.json``.

    Args:
        config: The parsed ``config.json`` dict. ``None`` (or a
            non-dict) returns ``MTPEligibility.NONE`` — used by the CLI
            so callers can pass ``model_auto_config.get_config(path)``
            output unguarded.
        has_external_sidecar: Accepted for compatibility with the
            legacy CLI flag surface, but currently does not promote any
            architecture. Qwen3.5 / Qwen3.6 eligibility requires
            ``mtp_num_hidden_layers >= 1`` in the base config
            (root or ``text_config``) because their MTP head is part of
            the target checkpoint. Gemma 4 sidecar promotion remains
            disabled until it passes end-to-end greedy-lossless validation.

    Returns:
        :class:`MTPEligibility` value. Detection is conservative — any
        ambiguity (unsupported ``model_type``, ``mtp_num_hidden_layers``
        absent or zero, structurally-broken config) collapses to
        ``NONE`` so MTP speculative config on an ineligible model is
        rejected at boot rather than silently emitting wrong tokens.
    """
    result = _detect_mtp_eligibility_verbose(
        config, has_external_sidecar=has_external_sidecar
    )
    return result.eligibility


def _detect_mtp_eligibility_verbose(
    config: dict[str, Any] | None,
    *,
    has_external_sidecar: bool = False,
) -> _DetectionResult:
    """Detection helper that returns the full reason string.

    Tests assert on ``reason`` to lock the contract — keep the
    short-strings stable across versions.
    """
    if not isinstance(config, dict):
        return _DetectionResult(MTPEligibility.NONE, None, 0, "config is not a dict")

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

    _ = has_external_sidecar
    num_mtp_layers = _mtp_num_hidden_layers(config)
    if num_mtp_layers <= 0:
        # MTP-capable model_type but MTP weights not present on this
        # checkpoint. For Qwen3.5 / Qwen3.6 this is a stripped convert —
        # operator must re-convert from HF with the PR #990 sanitize()
        # path that preserves ``mtp.*`` weights. Detection collapses to
        # NONE so MTP speculative config is rejected loudly at boot.
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
