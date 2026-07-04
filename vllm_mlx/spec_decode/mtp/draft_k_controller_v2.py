# SPDX-License-Identifier: Apache-2.0
"""EV-based draft-K controller — port of Ollama's ``speculate_depth.go``.

This is a near-verbatim Python translation of
``github.com/ollama/ollama`` ``x/mlxrunner/speculate_depth.go`` (up to
tag ``v0.31.1``). It picks ``K = argmax_N EV(N)`` where
``EV(N) = expected_committed(N) / expected_cost(N)`` over depths from
``0`` (plain decode, "park") up to one past the frontier. The depth-0
floor lets the controller stop speculating when no draft depth pays;
the frontier ceiling keeps it from scoring a depth on the optimistic
inherited rate, so it climbs outward one position at a time.

State persists at speculation scope (per-model, across requests) via
:func:`get_or_create_controller`. The v1 controller in
``vllm_mlx.spec_decode.mtp.draft_k_controller`` used per-request state
and an ``accept×K`` signal that maximizes accepted-tokens-per-step
rather than throughput — Ollama's commit ``505e35f`` documents that
exact failure mode. This v2 fixes both: EV signal + cross-request state.

Alphas match Ollama exactly:

* ``costEWMA_alpha = 0.3`` (cost EWMA weight)
* ``costClampFraction = 0.25`` (per-innovation clamp ±25%)
* ``acceptanceEWMA_alpha = 0.1`` (acceptance EWMA weight)
* ``acceptanceMinSamples = 10`` (under-sampled positions inherit deepest trusted rate)
* ``depthProbeInterval = 4`` (base cadence for probing frontier+1)
* ``depthProbeIntervalMax = 512`` (exponential backoff cap)

The module is thread-safe: the global registry is guarded by a mutex,
and each :class:`DepthController` instance is safe to call from a
single generator thread (the vendored ``mtp_generate_step`` is
single-request, single-thread). Multi-request batching would need
per-uid controllers or a queue guard around ``pick_k`` / ``record``.
"""

from __future__ import annotations

import bisect
import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ollama's tuned constants — do NOT change without a bench-verified reason.
# ---------------------------------------------------------------------------

# Cost EWMA weight. Fixed-depth target-forward cost is low-variance so a
# responsive alpha converges in a few visits while smoothing scheduler jitter.
COST_EWMA_ALPHA = 0.3

# Per-innovation clamp on cost EWMA. A host stall (cache trim, backpressure)
# can inflate a sample severalfold; unclamped, one bad sample can flip the EV
# comparison against plain decode, and once the controller stops parking it
# stops resampling depth 0, so the error never heals.
COST_CLAMP_FRACTION = 0.25

# Acceptance EWMA weight. Acceptance drifts with content, so a smaller alpha
# follows the drift instead of being anchored by early tokens.
ACCEPTANCE_EWMA_ALPHA = 0.1

# How many times a position needs to be reached before its rate is trusted.
# Also gates how fast the frontier advances (since the search reaches one
# past the frontier). Set near the EWMA's memory (~1/alpha).
ACCEPTANCE_MIN_SAMPLES = 10

# Base cadence at which the controller drafts one past its selection to
# refresh the next position up.
DEPTH_PROBE_INTERVAL = 4

# Cap the probe backoff. Probes that keep changing nothing double the
# interval up to this cap.
DEPTH_PROBE_INTERVAL_MAX = 512

# Default cap on the depth the controller may select. HACK-mode note: the
# generator currently only implements K ∈ {0, 1}, so callers pass
# ``max_k=1``; when chain-of-K verify lands (K≥2), bump this default to 3
# to match Ollama's tuned band.
DEFAULT_MAX_K = 3

# Minimum cost samples per depth before the controller stops force-
# seeding it. Under the pre-0.9.13-fix cold-start, ONE sample per depth
# passed ``sampled()``, but a single cost measurement is dominated by
# first-touch host jitter (cold cache line, first kv-cache write). A
# single-sample K=0 read of ~22ms (instead of the steady-state ~16ms)
# was enough to flip EV(0) vs EV(1) permanently once the frontier's
# optimistic 1.0 acceptance fallback lifted EV(1). Once the controller
# started picking K=1, the periodic probe brought it back to K=0 only
# every DEPTH_PROBE_INTERVAL=4 rounds — so prose spent 3/4 of its
# budget paying drafter cost for near-zero accept: the operator's
# observed "0 park rounds / 81.7% attempts but tok/s tanks" signature.
#
# 4 samples per depth folds two clamped innovations at
# COST_CLAMP_FRACTION=0.25 into the EWMA, which walks a first-touch
# 22ms → ~17ms and lets the true EV(0)>EV(1) comparison come out
# right on prose. On coding/JSON where K=1 wins, the extra 3 rounds
# per depth cost <100ms of setup — negligible against the tok/s gain.
COST_SEED_MIN_SAMPLES = 4

# Base cadence for the 0.9.13 starvation probe. Distinct from
# ``DEPTH_PROBE_INTERVAL`` (which paces the outward probe over
# [sel, sel+1]) because the starvation probe covers the FULL
# [0, min(frontier+1, max_k)] band. Kept at the same base as the
# outward probe (``4``) to match Ollama's tested cadence for cost-
# refresh probes; doubles up to ``DEPTH_PROBE_INTERVAL_MAX`` on
# every fire (fast throughput recovery once the cost EWMAs
# converge), resets to base on any EV-pick shift.
STARVATION_PROBE_INTERVAL = 4


# ---------------------------------------------------------------------------
# CostModel — per-K target-forward wall-time EWMA.
# ---------------------------------------------------------------------------


class CostModel:
    """Target-forward cost for validating N drafts as an EWMA per visited
    draft depth, read by piecewise-linear interpolation between samples
    (flat beyond the extremes).

    Interpolation assumes no curve shape, so a steep compute-bound or a
    flat bandwidth-bound forward is represented as measured. Cost is
    static within a run, learned from decode steps that already sync the
    forward, so there is no startup probe.

    Mirrors Go ``costModel`` (`speculate_depth.go:130-205`).
    """

    __slots__ = ("_ewma", "_depths", "_visits")

    def __init__(self) -> None:
        self._ewma: dict[int, float] = {}
        # Sorted list of sampled depths, maintained via bisect.
        self._depths: list[int] = []
        # Per-depth observation count. Read by the bootstrap gate so
        # ``_cost_seed_depth`` can force multiple samples at each depth
        # before the EV comparator runs on noisy first-touch numbers.
        self._visits: dict[int, int] = {}

    def observe(self, drafts: int, wall_ms: float) -> None:
        """Fold one forward's wall time into the draft depth's EWMA,
        clamping the innovation so one stall-inflated sample cannot
        move it far.

        Args:
            drafts: draft depth (K) for the observed forward.
            wall_ms: measured wall-clock milliseconds of the target forward.
        """
        if drafts < 0 or wall_ms <= 0.0:
            return
        prev = self._ewma.get(drafts)
        if prev is None:
            self._ewma[drafts] = wall_ms
            bisect.insort(self._depths, drafts)
        else:
            limit = COST_CLAMP_FRACTION * prev
            innovation = wall_ms - prev
            if innovation > limit:
                innovation = limit
            elif innovation < -limit:
                innovation = -limit
            self._ewma[drafts] = prev + COST_EWMA_ALPHA * innovation
        self._visits[drafts] = self._visits.get(drafts, 0) + 1

    def ready(self) -> bool:
        """True when two distinct depths have been sampled, so a slope
        exists and EV comparison is possible."""
        return len(self._ewma) >= 2

    def sampled(self, drafts: int) -> bool:
        return drafts in self._ewma

    def visits(self, drafts: int) -> int:
        """Number of times ``observe`` has folded a sample for this depth.

        Read by the bootstrap gate; distinct from ``sampled`` (which is
        a single-bit "ever seen") to enable a fixed-sample-count warmup
        that survives first-touch jitter.
        """
        return self._visits.get(drafts, 0)

    def cost(self, drafts: int) -> float:
        """Estimated target-forward wall time (ms) for validating
        ``drafts`` tokens: a piecewise-linear interpolation of the
        per-depth EWMAs, clamping to the nearest sample outside the
        sampled range (the curve beyond is unknown).
        """
        ds = self._depths
        if not ds:
            return 0.0
        if drafts <= ds[0]:
            return self._ewma[ds[0]]
        if drafts >= ds[-1]:
            return self._ewma[ds[-1]]
        # Linear interp between the enclosing samples.
        for i in range(1, len(ds)):
            hi = ds[i]
            if drafts <= hi:
                lo = ds[i - 1]
                t = (drafts - lo) / (hi - lo)
                return self._ewma[lo] + t * (self._ewma[hi] - self._ewma[lo])
        return self._ewma[ds[-1]]

    def sample_string(self) -> str:
        """Diagnostics: ``"0:12ms 1:18ms 2:24ms"``."""
        return " ".join(f"{d}:{self._ewma[d]:.0f}ms" for d in self._depths)


# ---------------------------------------------------------------------------
# AcceptanceModel — per-position conditional acceptance-rate EWMA.
# ---------------------------------------------------------------------------


class AcceptanceModel:
    """Per-position conditional acceptance rate (probability position i
    is accepted given the whole prefix before it was accepted) as an
    EWMA, shared across requests so a fresh request keeps the
    proven-out frontier.

    Drift is handled by EWMA forgetting, not by discarding the estimate.

    Mirrors Go ``acceptanceModel`` (`speculate_depth.go:220-302`).
    """

    __slots__ = ("_rate", "_seen")

    def __init__(self) -> None:
        # Index 0 is unused (position 1 is the first draft position).
        # We keep index 0 in the list so ``i`` is a direct index; this
        # matches Go's initialization ``[]float64{0}`` / ``[]int{0}``.
        self._rate: list[float] = [0.0]
        self._seen: list[int] = [0]

    def _grow(self, i: int) -> None:
        while len(self._seen) <= i:
            self._rate.append(0.0)
            self._seen.append(0)

    def observe(self, drafted: int, accepted: int) -> None:
        """Fold a step's outcome into each reached position's EWMA.

        A position ``i`` is reached only when the prefix before it
        survived (``accepted >= i - 1``), and is accepted iff
        ``accepted >= i``; updating only the surviving prefix avoids
        diluting deeper positions the step never reached.

        Args:
            drafted: K, the number of draft tokens the target forward validated.
            accepted: how many of those drafts were accepted (0..drafted).
        """
        for i in range(1, drafted + 1):
            if accepted < i - 1:
                break  # prefix did not survive to position i
            self._grow(i)
            outcome = 1.0 if accepted >= i else 0.0
            if self._seen[i] == 0:
                self._rate[i] = outcome
            else:
                self._rate[i] += ACCEPTANCE_EWMA_ALPHA * (outcome - self._rate[i])
            self._seen[i] += 1

    def acceptance(self, i: int) -> float:
        """Rate that position ``i`` is accepted given its prefix
        survived. Under-sampled positions inherit the deepest trusted
        rate rather than zero, so the controller keeps exploring
        deeper instead of locking shallow on noise. Falls back to
        optimistic ``1.0`` if no trusted rate exists yet.
        """
        if 1 <= i < len(self._seen) and self._seen[i] >= ACCEPTANCE_MIN_SAMPLES:
            return self._rate[i]
        # Inherit deepest trusted rate.
        for j in range(i - 1, 0, -1):
            if j < len(self._seen) and self._seen[j] >= ACCEPTANCE_MIN_SAMPLES:
                return self._rate[j]
        return 1.0

    def expected_committed(self, n: int) -> float:
        """Expected committed tokens at depth N: the current token
        (which always commits) plus the expected number of accepted
        drafts — each draft position contributes the probability its
        whole prefix was accepted, the running product of the
        per-position rates summed over positions.
        """
        total = 1.0
        prod = 1.0
        for k in range(1, n + 1):
            prod *= self.acceptance(k)
            total += prod
        return total

    def frontier(self) -> int:
        """Deepest position with a trusted acceptance rate. The
        controller never selects beyond ``frontier + 1`` so the
        selection grows outward one position at a time instead of
        jumping deep on inherited optimism.
        """
        f = 0
        for i in range(1, len(self._seen)):
            if self._seen[i] >= ACCEPTANCE_MIN_SAMPLES:
                f = i
            else:
                break
        return f


# ---------------------------------------------------------------------------
# DepthController — argmax-EV depth picker with periodic frontier probing.
# ---------------------------------------------------------------------------


class DepthController:
    """Drafts ``K = argmax_N EV(N)`` where ``EV(N) = committed(N) / cost(N)``,
    over depths from 0 (plain decode) up to one past the frontier.

    Holds the depth-selection state learned across requests — the
    target forward's per-depth cost curve, the drafts' per-position
    acceptance rates, the probe cadence, and the depth scheduled for
    the next round — persisted on the speculation scope so a fresh
    request starts at the proven-out depth instead of re-ramping from
    shallow.

    Mirrors Go ``depthController`` (`speculate_depth.go:30-119`).
    """

    def __init__(self, max_k: int = DEFAULT_MAX_K) -> None:
        self.cost = CostModel()
        self.acc = AcceptanceModel()
        # The previous round drafted a probe (used to trigger the
        # exponential-backoff branch on the NEXT round).
        self._probed = False
        # Depth ``next()`` chose for the upcoming round, carried
        # across requests so a new request's first round consumes it
        # instead of recomputing at the boundary.
        self.scheduled = 0
        # Probe cadence, persisted so a backed-off request need not
        # restart at the base.
        self._probe_interval = DEPTH_PROBE_INTERVAL
        self._probe_since = 0
        self._last_selected = 0
        # Ceiling on the controller's decision (hard cap). Ollama has
        # no explicit cap; we use it to match the generator's
        # implemented range (HACK-mode K∈{0,1}) and to enforce
        # ``--mtp-max-k``.
        self.max_k = max(0, max_k)
        # ------------------------------------------------------------------
        # 0.9.13 starvation-probe schedule (fix for K-lock at the max_k cap).
        # ------------------------------------------------------------------
        # Ollama's built-in outward probe (``_probe_since`` / ``_probe_interval``)
        # picks ``min(sel + 1, frontier + 1, max_k)`` — when ``sel == max_k``
        # and ``frontier >= max_k`` the probe clamps to ``sel`` and never
        # fires. That is exactly the pathology parent measured on Gemma 4
        # 12B 4bit: 92.7% of rounds locked at K=3 after bootstrap, with
        # stale K=1 / K=2 cost estimates. Because cost is only refreshed
        # for the K actually drafted, once the controller stops picking
        # K∈{1,2} their cost EWMAs freeze at bootstrap values while K=3's
        # keeps updating — the EV comparator's cost slope goes stale and
        # can never flip back even if K=1 is now genuinely faster.
        #
        # Fix: a second, unconditional periodic probe that force-samples
        # the least-recently-visited K in ``[0, min(frontier+1, max_k)]``.
        # Interval starts at ``STARVATION_PROBE_INTERVAL=4`` (same base
        # as the outward probe — Ollama's tested cadence) and doubles
        # up to ``DEPTH_PROBE_INTERVAL_MAX=512`` on every probe (the
        # exponential backoff for cost-refresh probes); it resets to
        # the base whenever the underlying EV pick changes, giving the
        # new selection a full interval to settle before we perturb it.
        self._round_probe_counter = 0
        self._round_probe_interval = STARVATION_PROBE_INTERVAL
        # Last EV pick observed by the starvation probe (separate from
        # ``_last_selected`` above, which tracks the outward-probe cadence).
        self._round_probe_last_sel = 0
        # Rolling window of the K's actually consumed by the target
        # forward. Bounded at ``DEPTH_PROBE_INTERVAL_MAX`` so the memory
        # footprint stays fixed regardless of run length; the argmin
        # scan reads only the last ``_round_probe_interval`` entries.
        self._recent_k_used: deque[int] = deque(maxlen=DEPTH_PROBE_INTERVAL_MAX)
        # Diagnostics on starvation probes.
        self.starvation_probe_count = 0
        # Diagnostics.
        self.park_count = 0
        self.round_count = 0
        # Per-K round histogram, keyed by the K actually consumed by the
        # target forward. Populated by ``record`` and read by the
        # Prometheus renderer (``rapid_mlx_spec_decode_k_chosen_*``).
        # Kept as a dict rather than a list so a future max_k lift needs
        # no schema migration on the metrics side.
        self.k_histogram: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def frontier(self) -> int:
        return self.acc.frontier()

    def pick_k(self) -> int:
        """Return the draft depth for the upcoming step.

        The EV-optimal depth (capped at frontier+1 and at ``self.max_k``),
        except periodically the controller probes one past the
        selection to refresh the next position up. The probe stays
        within the frontier window. The cadence doubles toward its cap
        while probes change nothing and resets on any selection change,
        giving the new selection a full interval to settle. The chosen
        depth is recorded in ``self.scheduled`` for the next request's
        open to consume.

        Mirrors Go ``depthController.next`` (`speculate_depth.go:58-88`)
        with one 0.9.13 addition: an unconditional starvation-probe pass
        after the outward probe / EV pick that periodically force-samples
        the least-recently-visited K, so a max_k-clamped controller can
        still refresh stale K∈[0, frontier] cost estimates. See
        ``_apply_starvation_probe`` and the ``_round_probe_*`` state
        docstring in ``__init__`` for the rationale.
        """
        sel = self._selected()
        if sel != self._last_selected:
            self._probe_interval = DEPTH_PROBE_INTERVAL
            self._probe_since = 0
            self._last_selected = sel
        elif self._probed:
            self._probe_interval = min(
                self._probe_interval * 2, DEPTH_PROBE_INTERVAL_MAX
            )
        self._probed = False

        self._probe_since += 1
        depth: int
        outward_probe_fired = False
        if self._probe_since >= self._probe_interval:
            self._probe_since = 0
            probe = min(sel + 1, self.frontier() + 1, self.max_k)
            if probe != sel:
                self._probed = True
                depth = probe
                outward_probe_fired = True

        if not outward_probe_fired:
            # Seed a clean cost sample for every depth in [0, frontier+1]
            # before judging by EV. Without this the controller stays at
            # the one depth it can sit at without a transition (depth 0)
            # and never learns that drafting pays on a deep-optimum model.
            seed = self._cost_seed_depth()
            if seed >= 0:
                depth = seed
            else:
                depth = sel
            # Enforce the hard cap.
            depth = min(depth, self.max_k)

        # 0.9.13 starvation probe (applies after the outward probe / EV
        # pick / bootstrap seed). See ``_apply_starvation_probe`` for
        # rationale; safe to run every round because it is a no-op until
        # its own counter overflows the interval.
        depth = self._apply_starvation_probe(sel, depth)

        self.scheduled = depth
        return depth

    def _apply_starvation_probe(self, sel: int, depth: int) -> int:
        """Periodic override that force-samples the least-recently-visited
        K in ``[0, min(frontier+1, max_k)]``.

        This is the 0.9.13 K-lock fix. Ollama's outward probe reaches
        ``sel + 1``, which is a no-op at the ``max_k`` ceiling — leaving
        the cost EWMAs for shallower K's frozen at bootstrap values
        while K=max_k's EWMA keeps refreshing. The EV comparator then
        never re-elects a shallower K even if it would now genuinely
        win. This pass fires on a periodic cadence (base 4 rounds,
        doubles up to 512, resets to base whenever the EV pick changes)
        and overrides ``depth`` with the K having fewest recent
        observations in the frontier-bounded window.

        Ties break shallow (prefer K=0 then K=1 then K=2 ...) so the
        cheaper K's get their cost refreshed first.

        Args:
            sel: the current EV pick (``self._selected()``) — used to
                reset the interval when the EV optimum shifts.
            depth: the depth already chosen by the outward probe / EV
                pick / bootstrap seed. Returned unchanged when the
                starvation probe is not firing this round.

        Returns:
            Overridden depth if the starvation probe fires and picks a
            different K than ``depth``; else ``depth`` unchanged.
        """
        # Reset the cadence whenever the EV pick shifts — the new
        # selection deserves a full interval of undisturbed operation
        # before we perturb it with another probe.
        if sel != self._round_probe_last_sel:
            self._round_probe_interval = STARVATION_PROBE_INTERVAL
            self._round_probe_counter = 0
            self._round_probe_last_sel = sel

        self._round_probe_counter += 1
        if self._round_probe_counter < self._round_probe_interval:
            return depth

        # Probe cadence has elapsed — reset counter and pick the K in
        # ``[0, min(frontier+1, max_k)]`` with the fewest samples in the
        # most recent ``_round_probe_interval`` rounds.
        self._round_probe_counter = 0
        limit = min(self.frontier() + 1, self.max_k)
        if limit <= 0:
            # No range to explore beyond K=0; nothing to probe.
            self._round_probe_interval = min(
                self._round_probe_interval * 2, DEPTH_PROBE_INTERVAL_MAX
            )
            return depth

        window_size = self._round_probe_interval
        # Slice the last ``window_size`` entries. ``deque`` doesn't
        # slice, so materialize the tail into a list.
        if len(self._recent_k_used) > 0:
            if len(self._recent_k_used) <= window_size:
                window = list(self._recent_k_used)
            else:
                window = list(self._recent_k_used)[-window_size:]
        else:
            window = []

        counts: dict[int, int] = {n: 0 for n in range(0, limit + 1)}
        for k in window:
            if k in counts:
                counts[k] += 1

        # Argmin over counts, tie-break by lowest K (cheap first).
        probe_k = min(counts.keys(), key=lambda n: (counts[n], n))
        # Only override if the probe would pick a genuinely different K.
        # Even when probe_k == depth we still consume this probe slot
        # (interval doubles) so the cadence doesn't wedge on a no-op.
        self._round_probe_interval = min(
            self._round_probe_interval * 2, DEPTH_PROBE_INTERVAL_MAX
        )
        if probe_k != depth:
            self.starvation_probe_count += 1
            return probe_k
        return depth

    def record(
        self,
        k_used: int,
        wall_ms: float,
        accepts: list[bool] | None = None,
    ) -> None:
        """Fold one round's outcome into the cost + acceptance models.

        Args:
            k_used: draft depth K actually consumed by the target
                forward for this round.
            wall_ms: measured wall-clock milliseconds of that target forward.
            accepts: list of per-position accept outcomes, length K.
                Position ``i`` (1-indexed here as ``accepts[i-1]``) is
                the draft at that depth. When ``k_used == 0``, pass an
                empty list (or ``None``) — no acceptance to record.

        Mirrors the ``observe`` half of Go ``speculativeDecoder.observe``
        (which calls both models' ``observe``).
        """
        self.round_count += 1
        if k_used == 0:
            self.park_count += 1
        self.k_histogram[k_used] = self.k_histogram.get(k_used, 0) + 1
        # Track the K just consumed in the starvation-probe rolling window.
        # The deque is bounded so no eviction bookkeeping is required.
        self._recent_k_used.append(k_used)
        if wall_ms > 0.0:
            self.cost.observe(k_used, wall_ms)
        if accepts:
            # Count consecutive True from position 1 up.
            accepted = 0
            for outcome in accepts:
                if outcome:
                    accepted += 1
                else:
                    break
            self.acc.observe(len(accepts), accepted)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _selected(self) -> int:
        """EV-optimal draft depth without mutating probe state, the
        argmax over ``[0, min(frontier+1, max_k)]``. Returns 0 until
        the cost model can compare depths.
        """
        if not self.cost.ready():
            return 0
        limit = min(self.frontier() + 1, self.max_k)
        best = 0
        best_ev = self.acc.expected_committed(0) / self.cost.cost(0)
        for n in range(1, limit + 1):
            cost_n = self.cost.cost(n)
            if cost_n <= 0.0:
                continue
            ev = self.acc.expected_committed(n) / cost_n
            if ev > best_ev:
                best = n
                best_ev = ev
        return best

    def _cost_seed_depth(self) -> int:
        """Shallowest depth in ``[0, min(frontier+1, max_k)]`` with fewer
        than ``COST_SEED_MIN_SAMPLES`` observations, or -1 if every depth
        is sufficiently sampled.

        0.9.13 fix: was ``sampled(n)`` (1-bit gate). A single cost
        sample was noisy enough — first-touch cold-cache jitter — to
        wedge EV comparison in a wrong-signed steady state (see
        ``COST_SEED_MIN_SAMPLES`` docstring). The threshold read is
        symmetric across depths, so bootstrap rotates through
        ``[0]*N + [1]*N`` (then ``[2]*N`` once chain-of-K lifts the
        max) before releasing to ``_selected()``.
        """
        limit = min(self.frontier() + 1, self.max_k)
        for n in range(0, limit + 1):
            if self.cost.visits(n) < COST_SEED_MIN_SAMPLES:
                return n
        return -1

    def diagnostics(self) -> str:
        """Human-readable snapshot for INFO logs."""
        return (
            f"K_scheduled={self.scheduled} frontier={self.frontier()} "
            f"max_k={self.max_k} rounds={self.round_count} "
            f"parks={self.park_count} "
            f"cost=[{self.cost.sample_string()}] "
            f"probe_interval={self._probe_interval} "
            f"starve_interval={self._round_probe_interval} "
            f"starve_probes={self.starvation_probe_count}"
        )


# ---------------------------------------------------------------------------
# Global registry — per-model, cross-request state.
# ---------------------------------------------------------------------------

_controllers: dict[str, DepthController] = {}
_lock = threading.Lock()


def get_or_create_controller(
    model_id: str,
    max_k: int = DEFAULT_MAX_K,
) -> DepthController:
    """Return the process-global :class:`DepthController` for ``model_id``,
    creating it on first access. Thread-safe.

    Args:
        model_id: cache key identifying the target model + drafter
            combination. Different sidecars against the same target
            should get different keys.
        max_k: hard ceiling on the controller's depth selection,
            passed on FIRST access only. Subsequent calls with a
            different ``max_k`` are ignored (a warning is logged) —
            reconfiguring an in-flight controller mid-request would
            invalidate its learned frontier.
    """
    with _lock:
        ctrl = _controllers.get(model_id)
        if ctrl is None:
            ctrl = DepthController(max_k=max_k)
            _controllers[model_id] = ctrl
            logger.info(
                "[MTP-controller] created DepthController for model_id=%r max_k=%d",
                model_id,
                max_k,
            )
        elif ctrl.max_k != max_k:
            logger.warning(
                "[MTP-controller] ignoring max_k=%d for existing controller "
                "model_id=%r (already max_k=%d); reset via reset_controllers()"
                " to change",
                max_k,
                model_id,
                ctrl.max_k,
            )
        return ctrl


def reset_controllers() -> None:
    """Drop all cached controllers. Test-only; not called from prod code."""
    with _lock:
        _controllers.clear()


def get_controller_snapshot() -> dict[str, str]:
    """Snapshot of ``model_id -> diagnostics`` for logging / metrics."""
    with _lock:
        return {k: v.diagnostics() for k, v in _controllers.items()}


def sum_across_controllers() -> tuple[int, int, dict[int, int]]:
    """Aggregate (round_count, park_count, k_histogram) across all
    registered controllers. Called by the /metrics renderer so a
    multi-model process reports one park counter per family label.

    Returns a fresh ``k_histogram`` dict rather than a shared reference
    so metric rendering can iterate it outside the registry lock. The
    controller instances themselves stay in the registry; only their
    scalar counters are copied out.
    """
    with _lock:
        round_total = 0
        park_total = 0
        hist: dict[int, int] = {}
        for ctrl in _controllers.values():
            round_total += ctrl.round_count
            park_total += ctrl.park_count
            for k, count in ctrl.k_histogram.items():
                hist[k] = hist.get(k, 0) + count
        return round_total, park_total, hist
