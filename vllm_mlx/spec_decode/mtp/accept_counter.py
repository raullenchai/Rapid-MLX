# SPDX-License-Identifier: Apache-2.0
"""Process-local MTP accept-rate counter (R15 task #302).

The Prometheus surface (``rapid_mlx_spec_decode_*``) is the canonical
external observability for whether speculation is paying for itself in
production. ``MTPAcceptCounter`` is the in-process backing
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

Verification and performance contract
-------------------------------------

The ``method="mtp"`` accept ratio is the performance surface:

* ``accept_ratio >= 0.80`` for Qwen3.5-9B-w4 at temp=0 on the bench
  workload (PR #990 reports ~85%).
* ``accept_ratio`` only equals 1.0 when EVERY draft was accepted; a
  ratio below 1.0 means at least one rejection fired and the target
  verify step took the corrective path. The target therefore remains
  authoritative, but byte identity with ordinary decode depends on the
  family-specific numerical verify path and is tested separately.
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
    # #3155 per-verify-call breakdown (all zero-defaulted so older
    # constructor call sites keep working).  ``drafted_by_depth[d]`` counts
    # verify calls that carried a draft at depth ``d`` (1-indexed);
    # ``accepted_by_depth[d]`` those where depth ``d`` was accepted.  Both are
    # sorted ``(depth, count)`` tuples so the snapshot stays hashable.
    verify_calls: int = 0
    correction_tokens: int = 0
    bonus_tokens: int = 0
    drafted_by_depth: tuple[tuple[int, int], ...] = ()
    accepted_by_depth: tuple[tuple[int, int], ...] = ()

    @property
    def mean_accepted_per_verify(self) -> float:
        """Committed draft tokens per verify call (MTPLX's headline number).

        ``sum(accepted_by_depth) / verify_calls``; 0.0 with no verify calls.
        """
        if self.verify_calls == 0:
            return 0.0
        return sum(count for _, count in self.accepted_by_depth) / self.verify_calls

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
        self._verify_calls = 0
        self._correction_tokens = 0
        self._bonus_tokens = 0
        self._drafted_by_depth: dict[int, int] = {}
        self._accepted_by_depth: dict[int, int] = {}

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
            raise ValueError(f"tokens_saved must be non-negative; got {tokens_saved}")
        with self._lock:
            self._accepts += 1
            self._tokens_saved += tokens_saved

    def record_verify(self, depth: int, accepted: int) -> None:
        """Record one verify call of a chain-of-K draft (#3155).

        ``depth`` drafts were proposed, the first ``accepted`` of them were
        accepted (chain semantics: acceptance is a prefix).  The target's
        own token from the same forward is a *bonus* token when every
        draft was accepted and a *correction* token otherwise — the
        split MTPLX reports as ``bonus_tokens`` / ``correction_tokens``.
        Verify-only bookkeeping; callers on the single-request path keep
        their existing ``record_attempt`` / ``record_accept`` calls.
        ``depth == 0`` (no draft proposed, nothing verified) records
        nothing, matching :meth:`record_round`.
        """
        self._check_outcome(depth, accepted)
        if depth == 0:
            return
        with self._lock:
            self._record_verify_locked(depth, accepted)

    @staticmethod
    def _check_outcome(depth: int, accepted: int) -> None:
        if depth < 0 or accepted < 0 or accepted > depth:
            raise ValueError(
                f"invalid verify outcome depth={depth} accepted={accepted}"
            )

    def _record_verify_locked(self, depth: int, accepted: int) -> None:
        """Body of :meth:`record_verify`; caller holds ``_lock``."""
        self._verify_calls += 1
        for d in range(1, depth + 1):
            self._drafted_by_depth[d] = self._drafted_by_depth.get(d, 0) + 1
        for d in range(1, accepted + 1):
            self._accepted_by_depth[d] = self._accepted_by_depth.get(d, 0) + 1
        if accepted < depth:
            self._correction_tokens += 1
        else:
            self._bonus_tokens += 1

    def record_round(self, depth: int, accepted: int) -> None:
        """Record a whole verify round at once (continuous-batching path).

        Equivalent to ``depth`` × :meth:`record_attempt`, ``accepted`` ×
        :meth:`record_accept` and one :meth:`record_verify`, under one
        lock acquisition.  ``depth == 0`` (a target-only cycle) records
        nothing: no draft was proposed, so there is nothing to verify.
        """
        self._check_outcome(depth, accepted)
        if depth == 0:
            return
        with self._lock:
            self._attempts += depth
            self._accepts += accepted
            self._tokens_saved += accepted
            self._record_verify_locked(depth, accepted)

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
                verify_calls=self._verify_calls,
                correction_tokens=self._correction_tokens,
                bonus_tokens=self._bonus_tokens,
                drafted_by_depth=tuple(sorted(self._drafted_by_depth.items())),
                accepted_by_depth=tuple(sorted(self._accepted_by_depth.items())),
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
            self._verify_calls = 0
            self._correction_tokens = 0
            self._bonus_tokens = 0
            self._drafted_by_depth = {}
            self._accepted_by_depth = {}


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
