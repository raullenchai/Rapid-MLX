# SPDX-License-Identifier: Apache-2.0
"""DFlash runtime — lazy bridge into mlx-vlm's spec-decode machinery.

mlx-vlm 0.5.0+ implements DFlash: drafter loading
(``load_drafter``), the per-step draft-verify-walk loop
(``_dflash_rounds``), and hidden-state capture on Qwen3.5/3.6 language
models. We don't vendor any of that — we *call into* it from
``BatchedEngine._step``. This module is the import boundary so the
mlx-vlm dependency stays optional (``pip install rapid-mlx[dflash]``).

Public surface:
  - ``DFlashRuntime`` — handle owning the drafter + the call adapter
  - ``load_runtime(drafter_repo)`` — lazy import + drafter load
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .eligibility import have_runtime

logger = logging.getLogger(__name__)


@dataclass
class DFlashRuntime:
    """Handle around an mlx-vlm DFlash drafter + its call adapter.

    ``drafter`` is the loaded model object (mlx-vlm's ``DFlashDraftModel``);
    ``kind`` is the resolved drafter family (e.g. ``"dflash"``) — kept
    so log lines and metric names stay aligned with what mlx-vlm reports
    internally.
    """

    drafter: Any
    kind: str
    drafter_repo: str

    def reset_accept_lens(self) -> None:
        """Clear the per-round acceptance counters between requests so
        metric reports don't pool acceptance across sessions. Tolerant
        of mlx-vlm versions that might rename / change the type of the
        attribute — silently no-ops if it isn't a list (the public
        contract of mlx-vlm 0.5.0's drafter has it as ``list[int]``,
        but the upstream API is not yet declared stable)."""
        accept_lens = getattr(self.drafter, "accept_lens", None)
        if isinstance(accept_lens, list):
            accept_lens.clear()
        elif accept_lens is not None:
            logger.warning(
                "DFlash drafter.accept_lens has unexpected type %s; "
                "metrics may pool across requests",
                type(accept_lens).__name__,
            )

    def accept_lens_snapshot(self) -> list[int]:
        """Return a copy of the current accept-len list. Cheap; used by
        the metrics endpoint to compute mean accept per request without
        racing with the in-progress generator."""
        accept_lens = getattr(self.drafter, "accept_lens", None)
        if not isinstance(accept_lens, list):
            return []
        return list(accept_lens)


def load_runtime(drafter_repo: str, kind: str = "dflash") -> DFlashRuntime:
    """Lazy-import mlx-vlm's drafter loader and return a ``DFlashRuntime``.

    The mlx-vlm import is deferred to call time so installing rapid-mlx
    without the ``[dflash]`` extras leaves the CLI / unit tests working;
    only users who actually pass ``--enable-dflash`` ever touch the
    mlx-vlm code path.
    """
    if not have_runtime():
        raise RuntimeError(
            "DFlash runtime not available — mlx-vlm 0.5.0+ is required. "
            "Install with: pip install 'rapid-mlx[dflash]'"
        )
    # Import here, not at module top, so the optional dep stays optional.
    from mlx_vlm.speculative.drafters import load_drafter

    logger.info("Loading DFlash drafter: %s (kind=%s)", drafter_repo, kind)
    drafter, resolved_kind = load_drafter(drafter_repo, kind=kind)
    return DFlashRuntime(drafter=drafter, kind=resolved_kind, drafter_repo=drafter_repo)
