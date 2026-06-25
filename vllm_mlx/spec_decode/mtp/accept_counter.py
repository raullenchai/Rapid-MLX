# SPDX-License-Identifier: Apache-2.0
"""Process-local MTP accept-rate counter (R15 task #302).

The Prometheus surface (``rapid_mlx_spec_decode_*``) is the canonical
external observability for whether the lossless contract is actually
holding in production. ``MTPAcceptCounter`` is the in-process backing
state — both the chain MTP generator and any future tree MTP variant
write into the SAME counter, with the ``method`` / ``family`` labels
distinguishing the source.

Design choices
--------------

* Counters are process-global. There is exactly one MTP path per loaded
  model — multi-model serving (#387) still routes each MTP request
  through the same loop, so a single counter is sufficient. The
  module-level :func:`get_global_counter` returns the singleton.
* Counters are ``int``-typed and updated under a single ``threading.Lock``.
  The MTP generator runs from the scheduler thread but the metrics
  reader runs from the FastAPI worker pool — without the lock, a
  scrape could observe an attempts-without-accepts race window and
  paint a transient 0% accept ratio on dashboards.
* Snapshots are taken under the same lock, so the snapshot is causally
  consistent: ``accepts <= attempts`` and ``tokens_saved >= accepts``
  always hold across scrapes.
* The counter only tracks ``method="mtp"`` for now — the label is on
  the metric-render side, not on the counter struct, so adding a new
  method (suffix / dflash already have their own counters; this
  ``spec_decode_*`` family is for the model-side speculative variants)
  doesn't require any schema change here. Counters never reset on
  ``record_*`` calls.

Lossless contract
-----------------

The ``method="mtp"`` accept ratio is the lossless contract surface:

* ``accept_ratio >= 0.80`` for Qwen3.5-9B-w4 at temp=0 on the bench
  workload (PR #990 reports ~85%).
* ``accept_ratio`` only equals 1.0 when EVERY draft was accepted; a
  ratio below 1.0 means at least one rejection fired and the verify
  step took the corrective path. Both states still produce
  byte-identical token output to non-spec-decode generation; the
  acceptance rate is the *speedup* signal, not a correctness one.
* ``tokens_saved`` counts the cumulative "bonus tokens emitted from
  draft acceptance" — when a draft is accepted, the generator emits
  both the verified primary token and the accepted draft token in the
  SAME backbone step, saving one full forward pass. ``tokens_saved /
  attempts`` is therefore the per-attempt token win.

The bench harness reads ``snapshot()`` to compute the headline 1.57×
decode tok/s win, so the snapshot format is part of the public API.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class MTPAcceptSnapshot:
    """Causally-consistent snapshot of the counter state.

    Returned by :meth:`MTPAcceptCounter.snapshot`. The fields are the
    raw counter values; the renderer in :mod:`vllm_mlx.routes.metrics`
    converts ``attempts`` / ``accepts`` into the
    ``rapid_mlx_spec_decode_accept_ratio`` gauge.
    """

    attempts: int
    accepts: int
    tokens_saved: int

    @property
    def accept_ratio(self) -> float:
        """Accepts / attempts. Returns 0.0 when no attempts recorded.

        The gauge starts at 0.0 (Prometheus convention: "no data
        means 0") rather than NaN so dashboards don't flip to
        "no-data" state during the cold-start window before the first
        MTP attempt. Some dashboards alert on the gauge dropping
        below a threshold, and "no data" would silently mask a stuck
        loop.
        """
        if self.attempts == 0:
            return 0.0
        return self.accepts / self.attempts


class MTPAcceptCounter:
    """Thread-safe accept-rate counter for MTP speculative decoding.

    Three counters, all monotonically non-decreasing for the process
    lifetime:

    * ``attempts`` — Number of times the MTP head proposed a draft
      token. Bumped once per ``mtp_generate_step`` outer-loop verify
      iteration. Does not count the first cold-start primary-only
      step.
    * ``accepts`` — Subset of ``attempts`` where the verify backbone
      pass accepted the proposed draft. Bumped after the
      ``min(1, p_target/p_draft)`` probabilistic test (or exact-match
      test at temp=0). Always satisfies ``accepts <= attempts``.
    * ``tokens_saved`` — Bonus tokens emitted because a draft was
      accepted. Bumped by 1 per accept. ``tokens_saved == accepts``
      under the chain MTP variant — separate field because a future
      tree MTP could accept a multi-token branch.

    All bookkeeping is guarded by ``self._lock``. The lock is held for
    O(1) integer addition; no allocation, no MLX evals.

    Reset semantics
    ---------------

    The counter never resets on its own. ``reset()`` is provided ONLY
    for tests — the production Prometheus surface relies on monotonic
    counters, and resetting would surface as a counter decrement that
    would either spike ``rate()`` to +Inf or go negative for one
    scrape. The route-side rendering in
    :mod:`vllm_mlx.routes.metrics` therefore does NOT wrap this
    counter in a sticky accumulator — there is no underlying state
    that ever decrements.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempts = 0
        self._accepts = 0
        self._tokens_saved = 0

    # ---- recording side --------------------------------------------------

    def record_attempt(self) -> None:
        """Record one MTP draft proposal. Bump ``attempts`` by 1."""
        with self._lock:
            self._attempts += 1

    def record_accept(self, tokens_saved: int = 1) -> None:
        """Record one accepted draft.

        Bumps BOTH ``accepts`` (by 1) AND ``tokens_saved`` (by
        ``tokens_saved``) atomically. Callers MUST also call
        :meth:`record_attempt` for the same draft — they are separate
        because the attempt is recorded at draft-time and the accept
        only fires after the verify backbone pass, and a midway
        exception would otherwise wedge the counter at an
        attempts > accepts state where the rejection actually fired.

        Args:
            tokens_saved: Bonus tokens this accept emitted. Defaults
                to 1 (chain MTP — one draft accepted = one bonus
                token).
        """
        if tokens_saved < 0:
            raise ValueError(
                f"tokens_saved must be non-negative; got {tokens_saved}"
            )
        with self._lock:
            self._accepts += 1
            self._tokens_saved += tokens_saved

    def record_reject(self) -> None:
        """No-op kept for symmetry. Rejections don't bump any counter —
        ``attempts - accepts`` is the rejection count, derivable at the
        Prometheus side.

        The hook is here so the generator can emit a single explicit
        call per outcome rather than a conditional branch around the
        accept path; otherwise readers grep the codebase, see only
        ``record_accept`` calls, and wonder how rejections enter the
        counter.
        """
        return None

    # ---- read side -------------------------------------------------------

    def snapshot(self) -> MTPAcceptSnapshot:
        """Take a causally-consistent snapshot of all three counters.

        The three values are read in one lock acquisition, so a
        concurrent ``record_accept`` either lands fully before or
        fully after the snapshot — never mid-tuple.
        """
        with self._lock:
            return MTPAcceptSnapshot(
                attempts=self._attempts,
                accepts=self._accepts,
                tokens_saved=self._tokens_saved,
            )

    # ---- test-only -------------------------------------------------------

    def reset(self) -> None:
        """Reset all counters to zero. **TEST-ONLY** hook.

        Production scrape paths never call this — Prometheus counters
        MUST be monotonic. The :func:`reset_global_counter_for_tests`
        helper at module level uses this to reset between
        ``pytest`` cases.
        """
        with self._lock:
            self._attempts = 0
            self._accepts = 0
            self._tokens_saved = 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_counter = MTPAcceptCounter()


def get_global_counter() -> MTPAcceptCounter:
    """Return the process-global MTP accept counter.

    Used by:

    * :func:`vllm_mlx.spec_decode.mtp.generator.mtp_generate_step` to
      record each attempt / accept on the hot path.
    * :mod:`vllm_mlx.routes.metrics` to render the Prometheus surface.

    There is intentionally only one counter per process — multi-model
    serving (#387) routes every MTP request through the same loop, so
    the counter spans all models for ``method="mtp"``.
    """
    return _global_counter


def reset_global_counter_for_tests() -> None:
    """Test-only — reset the singleton counter between pytest cases."""
    _global_counter.reset()
