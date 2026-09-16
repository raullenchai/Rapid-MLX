# SPDX-License-Identifier: Apache-2.0
"""DFlash runtime — lazy bridge into mlx-vlm's spec-decode machinery.

mlx-vlm implements DFlash drafter loading
(``load_drafter``), the per-step draft-verify-walk loop
(``_dflash_rounds``), and hidden-state capture on Qwen3.5/3.6 language
models. DFlash2 is exposed under the same dispatch kind. We don't
vendor any of that — the dedicated DFlash server calls into it. This module is
the import boundary so the dependency stays optional
(``pip install rapid-mlx[dflash]``).

Public surface:
  - ``DFlashRuntime`` — handle owning the drafter + the call adapter
  - ``load_runtime(drafter_repo)`` — lazy import + drafter load
"""

from __future__ import annotations

import logging
import os
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
    target_revision: str | None = None
    drafter_revision: str | None = None
    algorithm: str = "dflash"

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

    def speculative_stats_snapshot(self) -> dict[str, float | int]:
        """Return the current request's draft acceptance diagnostics."""
        accepted = self.accept_lens_snapshot()
        drafted = getattr(self.drafter, "draft_lens", None)
        drafted = list(drafted) if isinstance(drafted, list) else []
        rounds = len(accepted)
        accepted_total = float(sum(accepted))
        drafted_total = int(sum(drafted)) if len(drafted) == rounds else 0
        return {
            "rounds": rounds,
            "accepted_drafts": int(accepted_total),
            "drafted_tokens": drafted_total,
            "accepted_drafts_per_round": (accepted_total / rounds if rounds else 0.0),
            "acceptance_rate": (
                accepted_total / drafted_total if drafted_total else 0.0
            ),
        }


def _runtime_algorithm(drafter: Any) -> str:
    """Return the concrete DFlash architecture loaded by mlx-vlm.

    ``draft_kind`` cannot distinguish DFlash2 because both generations use
    kind="dflash".  mlx-vlm normalizes DFlash2's config.model_type to
    ``dflash2``; the class-name check is a compatibility fallback for config
    wrappers that do not expose attributes directly.
    """
    config = getattr(drafter, "config", None)
    model_type = (
        config.get("model_type")
        if isinstance(config, dict)
        else getattr(config, "model_type", None)
    )
    if model_type == "dflash2" or type(drafter).__name__ == "DFlash2DraftModel":
        return "dflash2"
    return "dflash"


def load_runtime(
    drafter_repo: str,
    kind: str = "dflash",
    *,
    target_revision: str | None = None,
    drafter_revision: str | None = None,
    expected_algorithm: str | None = None,
    block_size: int | None = None,
) -> DFlashRuntime:
    """Lazy-import mlx-vlm's drafter loader and return a ``DFlashRuntime``.

    The mlx-vlm import is deferred to call time so installing rapid-mlx
    without the ``[dflash]`` extras leaves the CLI / unit tests working;
    only users who actually enable DFlash ever touch the
    mlx-vlm code path.
    """
    if not have_runtime():
        raise RuntimeError(
            "DFlash runtime not available — mlx-vlm 0.5.0+ is required. "
            "Install with: pip install 'rapid-mlx[dflash]'"
        )
    # Import here, not at module top, so the optional dep stays optional.
    from mlx_vlm.speculative.drafters import load_drafter

    load_source = drafter_repo
    if drafter_revision is not None:
        from mlx_vlm.utils import get_model_path

        load_source = str(get_model_path(drafter_repo, revision=drafter_revision))
    logger.info(
        "Loading DFlash drafter: %s revision=%s (kind=%s)",
        drafter_repo,
        drafter_revision or "default",
        kind,
    )
    drafter, resolved_kind = load_drafter(load_source, kind=kind)
    qualification_block_size = os.environ.get(
        "RAPID_MLX_DFLASH_QUALIFICATION_BLOCK_SIZE"
    )
    requested_block_size = (
        block_size if block_size is not None else qualification_block_size
    )
    if requested_block_size is not None:
        requested = int(requested_block_size)
        trained = int(drafter.config.block_size)
        if not 2 <= requested <= trained:
            raise ValueError(
                "qualification DFlash block size must be between 2 and "
                f"the trained width {trained}, got {requested}"
            )
        drafter.config.runtime_block_size = requested
        # A qualification override is an explicit fixed-width experiment.
        # DFlash2 otherwise starts at three tokens and adapts downward/upward,
        # which makes an advertised width=5 run measure a different policy.
        drafter.prefer_requested_block_size = True
        log = logger.info if block_size is not None else logger.warning
        log("Fixed DFlash runtime block size=%d", requested)
    algorithm = _runtime_algorithm(drafter)
    if expected_algorithm is not None and algorithm != expected_algorithm:
        raise RuntimeError(
            "DFlash drafter architecture mismatch: "
            f"expected {expected_algorithm!r}, loaded {algorithm!r} from "
            f"{drafter_repo!r}. Refusing to serve with a misleading or "
            "unqualified speculative-decoding route."
        )
    logger.info("Loaded DFlash runtime algorithm=%s", algorithm)
    return DFlashRuntime(
        drafter=drafter,
        kind=resolved_kind,
        drafter_repo=drafter_repo,
        target_revision=target_revision,
        drafter_revision=drafter_revision,
        algorithm=algorithm,
    )
