# SPDX-License-Identifier: Apache-2.0
"""Patch mlx-lm's ``ArraysCache`` to carry a ``rollback_state`` slot.

mlx-lm PR #990 adds a ``rollback_state: Optional[tuple] = None`` class
attribute to ``mlx_lm.models.cache.ArraysCache``. It is set by the
GatedDeltaNet layer's ``_process_chunk`` split (saves the
``(conv_state, ssm_state)`` snapshot at position ``n_confirmed``) and
read by the MTP generator's ``_rollback_draft`` (restores the snapshot
on draft rejection). Both writers and readers run under
``mx.stream(generation_stream)`` so the lock-free attribute access is
safe.

Until upstream merges, our installed ``mlx_lm 0.31.3`` does not have
the attribute. Setting it on a per-instance basis from the patched
model's ``_process_chunk`` would work, but Python's attribute lookup
falls back to the class only after the instance miss, so the FIRST
write succeeds — but the ``hasattr(cache, "rollback_state")`` guard in
the generator's ``_clear_rollback`` runs against the CLASS first and
would return ``False`` on a fresh cache, skipping the clear. That's
fine in isolation (nothing to clear) but the same guard is used to
gate ``rollback_state is not None`` checks; without the class slot we
have to fall back to ``getattr(c, "rollback_state", None)`` everywhere
which is fragile.

Patching the class once at import time is the simple fix. The patch is:

* Idempotent — calling :func:`patch_arrays_cache_rollback_state` twice
  is a no-op.
* Reversible only via process restart — the patch is intentionally
  one-way. There is no test path that needs to un-patch (mlx-lm's
  ``ArraysCache`` is a behaviorally-pure attribute slot; adding it
  doesn't change anything for callers that don't touch it).
* Safe under future mlx-lm versions that add the slot themselves —
  the guard checks ``"rollback_state" in cls.__dict__`` before
  patching, so once upstream lands the change this becomes a no-op.

The patch is applied automatically the first time
:func:`vllm_mlx.spec_decode.mtp.generator.mtp_generate_step` is
imported (the import in the generator module forces the side-effect).
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# Module-level guard so concurrent threads importing the generator
# don't race on the class attribute install. Without the lock, two
# threads could both see ``"rollback_state" not in cls.__dict__`` and
# both setattr — harmless for an attribute set to ``None`` (the writes
# are identical) but conceptually racy. The lock keeps the install
# atomic.
_install_lock = threading.Lock()
_patched = False


def patch_arrays_cache_rollback_state() -> bool:
    """Install ``rollback_state = None`` on ``mlx_lm.models.cache.ArraysCache``.

    Returns ``True`` if the patch was applied, ``False`` if the slot
    was already present (either from a previous call or from a future
    mlx-lm version that lands the change upstream).

    Raises:
        ImportError: If ``mlx_lm.models.cache`` cannot be imported.
            The MTP path is fundamentally unusable without mlx-lm so
            we let the import error propagate rather than silently
            falling back.
    """
    global _patched

    with _install_lock:
        if _patched:
            return False

        # Defer the import so a static analyzer can't trip on the
        # mlx_lm dependency before the package is installed (the
        # generator module itself imports mlx_lm at the top, so by the
        # time this patch fires, the import must already work — but
        # we still keep it lazy for symmetry with the rest of the MTP
        # package).
        from mlx_lm.models.cache import ArraysCache

        # ``cls.__dict__`` check (not ``hasattr``) so a future mlx-lm
        # that ships the slot wins over our patch — we don't want to
        # shadow an upstream rename or type change.
        if "rollback_state" in ArraysCache.__dict__:
            _patched = True
            logger.debug(
                "[mtp.cache_patch] ArraysCache.rollback_state already present "
                "(upstream version or prior patch); skipping install."
            )
            return False

        # The class attribute default is ``None``; instance writes
        # shadow it transparently. This mirrors the upstream PR #990
        # patch verbatim (``ArraysCache`` is a ``_BaseCache`` subclass
        # built via ``__new__``, so class-level defaults are the right
        # shape — there is no ``__init__`` that would otherwise
        # initialize the slot).
        ArraysCache.rollback_state = None  # type: ignore[attr-defined]
        _patched = True
        logger.info(
            "[mtp.cache_patch] Installed rollback_state slot on "
            "ArraysCache (vendored from mlx-lm PR #990)."
        )
        return True


def _is_patched_for_tests() -> bool:
    """Test-only — inspect the install flag."""
    return _patched


def _unpatch_for_tests() -> None:
    """Test-only — clear the install flag and remove the class attr.

    Allows tests to verify the install side-effect by toggling the
    install state. Never called from production.
    """
    global _patched

    with _install_lock:
        try:
            from mlx_lm.models.cache import ArraysCache

            if "rollback_state" in ArraysCache.__dict__:
                delattr(ArraysCache, "rollback_state")
        except ImportError:
            pass
        _patched = False
