# SPDX-License-Identifier: Apache-2.0
"""Process-local DFlash accept-rate counter (R15 task #313).

Mirrors :class:`vllm_mlx.spec_decode.mtp.accept_counter.MTPAcceptCounter`
field-for-field so the Prometheus surface stays symmetric across the
two model-side speculative backends. The only semantic difference: a
single DFlash attempt corresponds to a BLOCK of draft tokens (default 16)
rather than the single draft token MTP proposes — so ``tokens_saved /
attempts`` measures the per-block win, not the per-step win.

Counters
--------

* ``attempts`` — Number of DFlash blocks the drafter proposed. Bumped
  once per ``dflash_generate_step`` outer-loop verify iteration. The
  block size is fixed for a given run (default 16) so
  ``attempts * BLOCK_SIZE`` is the theoretical max ``tokens_saved``.
* ``accepts`` — Subset of ``attempts`` where at least one draft token in
  the block was accepted (i.e. ``accepted_len >= 1``). Always
  ``accepts <= attempts``. Pure ``attempts - accepts == 0`` cases
  collapse to a single corrective verify token at the divergence
  position (still lossless).
* ``tokens_saved`` — Cumulative bonus tokens emitted from draft
  acceptance. For DFlash this is the sum of ``accepted_len`` across
  every accept event, NOT just ``accepts`` (block size > 1 means a
  single accept can save up to ``BLOCK_SIZE`` tokens). The bench
  harness reads ``tokens_saved / attempts`` for the per-block bonus and
  ``tokens_saved / accepts`` for the per-accept bonus.

Lossless contract
-----------------

The ``method="dflash"`` accept ratio is the per-block speedup signal:

* ``accept_ratio >= 0.80`` for Qwen3.5-9B-w4 at temp=0 on the paper
  bench (paper claims 4.37x, M5 Max projects ~135 tok/s decode lossless).
* A block accepted in full (all 16 tokens match the verifier) saves
  ``BLOCK_SIZE - 1`` extra tokens beyond the always-accepted verify
  pred at position 0. ``tokens_saved`` per such attempt is therefore
  ``BLOCK_SIZE - 1``. A partial accept of ``k`` tokens
  (``1 <= k < BLOCK_SIZE``) saves ``k - 1`` extra; ``k == 0`` saves 0
  (only the single corrective token is emitted).

All bookkeeping is guarded by ``self._lock`` for thread safety —
identical pattern to :mod:`vllm_mlx.spec_decode.mtp.accept_counter`.

Reset semantics
---------------

Production Prometheus counters never reset. :meth:`reset` is provided
ONLY for test isolation — see :func:`reset_global_counter_for_tests`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class DFlashAcceptSnapshot:
    """Causally-consistent snapshot of the DFlash counter state.

    Returned by :meth:`DFlashAcceptCounter.snapshot`. The fields are
    raw counter values; the renderer in :mod:`vllm_mlx.routes.metrics`
    converts ``attempts`` / ``accepts`` into the
    ``rapid_mlx_spec_decode_dflash_accept_ratio`` gauge.
    """

    attempts: int
    accepts: int
    tokens_saved: int

    @property
    def accept_ratio(self) -> float:
        """Accepts / attempts. Returns 0.0 when no attempts recorded.

        The gauge starts at 0.0 (Prometheus convention: "no data means
        0") rather than NaN so dashboards don't flip to "no-data" state
        during the cold-start window before the first DFlash attempt.
        """
        if self.attempts == 0:
            return 0.0
        return self.accepts / self.attempts

    @property
    def mean_tokens_per_attempt(self) -> float:
        """Average bonus tokens emitted per attempt — the headline speedup signal.

        ``0.0`` when no attempts. The bench harness pairs this with the
        block size to derive the per-block acceptance length (the paper's
        "Accepted tokens per block" axis on the bench plot).
        """
        if self.attempts == 0:
            return 0.0
        return self.tokens_saved / self.attempts


class DFlashAcceptCounter:
    """Thread-safe accept-rate counter for DFlash speculative decoding.

    Three counters, all monotonically non-decreasing for the process
    lifetime. See module docstring for field semantics.

    All bookkeeping is guarded by ``self._lock``. The lock is held for
    O(1) integer addition; no allocation, no MLX evals.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempts = 0
        self._accepts = 0
        self._tokens_saved = 0

    # ---- recording side --------------------------------------------------

    def record_attempt(self) -> None:
        """Record one DFlash block proposal. Bump ``attempts`` by 1."""
        with self._lock:
            self._attempts += 1

    def record_accept(self, tokens_saved: int = 1) -> None:
        """Record one accept event.

        Bumps BOTH ``accepts`` (by 1) AND ``tokens_saved`` (by
        ``tokens_saved``) atomically.

        Args:
            tokens_saved: Bonus tokens this accept emitted — the
                accepted draft length minus the always-emitted verify
                pred at position 0. Range ``[0, BLOCK_SIZE - 1]``.
                ``0`` is allowed: it represents "the block had a
                divergence at position 0 but we still want to count
                the verify step as an accept event for accounting"
                — the generator never actually uses this branch
                (``accepted_len == 0`` calls :meth:`record_reject`
                instead) but we keep ``0`` valid for symmetry with
                MTP's :meth:`MTPAcceptCounter.record_accept` which
                also accepts ``tokens_saved=0`` for the same reason.
        """
        if tokens_saved < 0:
            raise ValueError(f"tokens_saved must be non-negative; got {tokens_saved}")
        with self._lock:
            self._accepts += 1
            self._tokens_saved += tokens_saved

    def record_reject(self) -> None:
        """No-op kept for symmetry with :meth:`record_accept`.

        Rejections don't bump any counter — ``attempts - accepts`` is
        the rejection count, derivable on the Prometheus side. The
        hook is here so the generator can emit a single explicit call
        per outcome (mirrors the MTP code structure).
        """
        return None

    # ---- read side -------------------------------------------------------

    def snapshot(self) -> DFlashAcceptSnapshot:
        """Take a causally-consistent snapshot of all three counters."""
        with self._lock:
            return DFlashAcceptSnapshot(
                attempts=self._attempts,
                accepts=self._accepts,
                tokens_saved=self._tokens_saved,
            )

    # ---- test-only -------------------------------------------------------

    def reset(self) -> None:
        """Reset all counters to zero. **TEST-ONLY** hook.

        Production scrape paths never call this — Prometheus counters
        MUST be monotonic. The :func:`reset_global_counter_for_tests`
        helper at module level uses this to reset between pytest cases.
        """
        with self._lock:
            self._attempts = 0
            self._accepts = 0
            self._tokens_saved = 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_counter = DFlashAcceptCounter()


def get_global_counter() -> DFlashAcceptCounter:
    """Return the process-global DFlash accept counter.

    Used by:

    * :func:`vllm_mlx.spec_decode.dflash.generator.dflash_generate_step`
      to record each attempt / accept on the hot path.
    * :mod:`vllm_mlx.routes.metrics` to render the Prometheus surface.
    """
    return _global_counter


def reset_global_counter_for_tests() -> None:
    """Test-only — reset the singleton counter between pytest cases."""
    _global_counter.reset()
