# SPDX-License-Identifier: Apache-2.0
"""Side-registry mapping alias name → DFlash drafter HF path.

Rapid-mlx's ``aliases.json`` enforces a CLOSED-KEY schema
(:data:`vllm_mlx.model_aliases._ALLOWED_PROFILE_KEYS`) — any unknown
field on an alias entry raises at JSON-load time. The existing
``dflash_draft_model`` field on :class:`AliasProfile` is already
reserved by the original mlx-vlm DFlash bridge
(:mod:`vllm_mlx.speculative.dflash`), which carries its OWN eligibility
gates (``supports_dflash`` + 4-bit/MoE rejection) that we don't want to
disturb for the new spec-decode path.

So the spec-decode drafter binding lives in this side-registry instead.
Two operator surfaces fill the table:

1. **Default registrations** at module-import time — the known z-lab
   drafters for the validated Qwen3.5 / 3.6 8-bit aliases. These are
   the same drafter checkpoints the mlx-vlm bridge uses (single source
   of truth for the drafter URL even though the runtime path is
   different).
2. **Operator-supplied registrations** via :func:`register_dflash_drafter`
   — used by tests and by future plugin loaders. Idempotent — registering
   the same alias twice with the same path is a no-op; with a DIFFERENT
   path is a warning + overwrite (last-wins so a plugin can override the
   default).

Lookup
------

:func:`get_dflash_drafter_path` returns the bound drafter HF path for
an alias, or ``None`` if no binding exists. The detect module hits
this on every eligibility check; the lookup is a single dict read
(no I/O, no MLX evals) so it stays cheap on the CLI boot path.

The registry is process-local. Multi-model serving (#387) routes every
DFlash request through the same lookup, and the registry is read-only
after boot so no per-request lock is needed (the underlying dict is
guarded by a module-level lock only for the register/clear sides).
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# The canonical z-lab drafter checkpoints for the validated 8-bit
# Qwen3.5 / 3.6 aliases. Same drafter URLs the existing mlx-vlm bridge
# uses — keep them aligned so an alias is "DFlash-eligible" in BOTH
# code paths or neither, never half-supported. When new aliases are
# validated (post-PoC bench), add them here AND set ``supports_dflash``
# + ``dflash_draft_model`` on the matching alias entry in
# ``aliases.json``.
_DEFAULT_REGISTRY: dict[str, str] = {
    # 0.9.0: only `qwen3.5-27b-8bit` survived the chat non-code-floor
    # gate (`scripts/bench_dflash.py --runs 3`, M-series, code median
    # 1.85×, chat 0.74×). The 9B-8bit and 3.6-27B-8bit aliases were
    # benched + flipped to `supports_dflash=false` in the same
    # release (chat regression 0.41-0.57×). The 9B-4bit entry was
    # never validated and the 4-bit cliff documented in
    # `eligibility.py` gates it anyway. Drafter scaffolding survives
    # in this module so `register_dflash_drafter()` keeps working at
    # runtime; we just don't pre-populate aliases the registry can't
    # honestly ship today. Add an entry back after a future bench
    # verifies the alias passes SOP §6 with the non-code-floor rule.
    "qwen3.5-27b-8bit": "z-lab/Qwen3.5-27B-DFlash",
}

# Working copy populated from ``_DEFAULT_REGISTRY`` at module import.
# Mutations route through :func:`register_dflash_drafter` /
# :func:`clear_drafter_registry_for_tests` so the lock is held on
# every mutation path.
_registry: dict[str, str] = dict(_DEFAULT_REGISTRY)
_registry_lock = threading.Lock()


def get_dflash_drafter_path(alias: str | None) -> str | None:
    """Return the bound DFlash drafter HF path for ``alias``.

    Args:
        alias: The alias name (e.g. ``"qwen3.5-9b-4bit"``). ``None``
            or empty returns ``None`` (a model loaded by raw HF path
            without an alias has no registry entry to look up).

    Returns:
        The bound drafter HF path, or ``None`` if no binding exists
        for this alias. Callers that need to fail loud on missing
        bindings should layer that on top — this lookup is the soft
        path used by detection.
    """
    if not alias:
        return None
    # No lock on the read side — Python dict reads are atomic and the
    # registry mutates rarely (boot + tests). Avoiding the lock keeps
    # the CLI boot path fast.
    return _registry.get(alias)


def register_dflash_drafter(alias: str, drafter_hf_path: str) -> None:
    """Register a DFlash drafter for an alias.

    Args:
        alias: The alias name to bind. Must be non-empty.
        drafter_hf_path: The drafter HF path (e.g.
            ``"z-lab/Qwen3.5-9B-DFlash"``). Must be non-empty.

    Re-registering the same alias with the SAME path is a silent no-op.
    Re-registering with a DIFFERENT path logs a warning and overwrites
    (last-wins) — used by plugin loaders that need to swap the default
    drafter for a fine-tuned variant.
    """
    if not alias:
        raise ValueError("alias must be a non-empty string")
    if not drafter_hf_path:
        raise ValueError("drafter_hf_path must be a non-empty string")

    with _registry_lock:
        existing = _registry.get(alias)
        if existing == drafter_hf_path:
            return
        if existing is not None and existing != drafter_hf_path:
            logger.warning(
                "[dflash.registry] Overwriting DFlash drafter binding for "
                "alias %r: %r -> %r",
                alias,
                existing,
                drafter_hf_path,
            )
        _registry[alias] = drafter_hf_path


def list_registered_aliases() -> list[str]:
    """Return all alias names with a bound DFlash drafter (sorted).

    Used by diagnostic / ``rapid-mlx info`` surfaces to enumerate
    DFlash-eligible aliases without round-tripping every alias through
    the detect helper.
    """
    return sorted(_registry.keys())


# ---------------------------------------------------------------------------
# Test-only helpers
# ---------------------------------------------------------------------------


def clear_drafter_registry_for_tests() -> None:
    """Reset the registry to the default set. **TEST-ONLY** hook.

    Pytest cases that mutate the registry via
    :func:`register_dflash_drafter` MUST call this in their teardown
    so the mutation does not leak into sibling tests. Production code
    never calls this — the registry is read-only after boot.
    """
    global _registry

    with _registry_lock:
        _registry = dict(_DEFAULT_REGISTRY)
