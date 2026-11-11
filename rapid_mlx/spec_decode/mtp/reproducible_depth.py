# SPDX-License-Identifier: Apache-2.0
"""Draft schedule for MTP requests whose output must be a function of the request.

Why a separate schedule at all
------------------------------

A verify forward over ``K + 1`` rows does not reproduce a one-row step's
logits bit for bit -- the matmul, attention and recurrent kernels differ by
width -- and in bf16 the top two logits are often equal or one ulp apart. So
the width that verified a position can flip a greedy argmax. The adaptive
depth controller (:mod:`.draft_k_controller_v2`) picks every round's width
from wall-clock costs and from acceptance pooled over all earlier requests,
and ``CopyDraftGate`` admits copy-drafts on tokens per millisecond, so two
identical greedy requests verified different widths and returned different
text, on a fresh server and on a warm one.

What a greedy request may use instead
-------------------------------------

Its own tokens, plus a host profile (:class:`GreedySchedule`) supplied by the
tuning cell or calibrated once per process, model and ``max_k``. Each round's
depth maximizes expected committed tokens / cost on the measured curve and
this request's own acceptance. No wall times or cross-request acceptance
enter a greedy request's decisions after calibration.

Unmeasured cells fall back to calibration using scratch forwards at one row
and each verify width, plus the greedy draft head. Paired round-robin ratios
suppress common clock drift, and medians reject isolated stalls. The measured curve is continuous: crossing a
verify/step ratio of 1.33 must not switch the K=1 price from 1.272 to 1.87.
Shipped curves preserve depth decisions across restarts for the same cell.
Fallback calibration freezes its curve within the process. Timing variation
in that fallback can still change greedy output across restarts.

Seeded requests retain their configured depth and clock-free copy curve.
An operator's ``disable_auto_k`` depth is used as configured.

"""

from __future__ import annotations

import logging
import math
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .draft_k_controller_v2 import COST_SEED_MIN_SAMPLES, DepthController

logger = logging.getLogger(__name__)

# Legacy class helpers below remain for explicitly constructed profiles.
# Calibrated profiles use measured costs and never select a class curve.
# Extra verified row's cost, as a fraction of a one-row step, above which the
# host's verify is "steep" (the quantized-matmul vector path; M2 Pro:
# ~0.45-0.7 depending on how it is timed) rather than cheap (M4 Pro: ~0.2).
# Retained for diagnostics only on measured profiles.
STEEP_VERIFY_ROW_COST = 0.33

# Per-class cost of one extra verified row and of one draft step, in one-row
# steps, as measured round-robin (Qwen3.5-9B-4bit on an M4 Pro: 3-row verify
# 1.39x, draft 0.14x; Qwen3.5-4B-4bit on an M2 Pro: 1.93x, 0.16x). With
# :data:`ROUND_OVERHEAD` charged to every round they reproduce the round
# costs the adaptive controller learned from the clock on those hosts while
# parked rounds were unpipelined (M2 Pro: 1.53 / 2.05); the steep request
# curve now charges it to drafting rounds only (see
# :data:`CLASS_PARK_PAYS_OVERHEAD`).
CLASS_ROW_COST = {False: 0.2, True: 0.46}
CLASS_DRAFT_COST = {False: 0.14, True: 0.16}

# The generator's own per-round host work, as a fraction of a one-row step
# (parked rounds ran at ~22 ms against a 17.5 ms forward on an M2 Pro before
# parked rounds were pipelined). Added to every measured round alike, so it
# dampens the relative cost of wider rounds.
ROUND_OVERHEAD = 0.25

# Whether a class's parked round still pays :data:`ROUND_OVERHEAD`. Parked
# rounds now launch their step ahead of delivery like plain decode, so on a
# steep host (M2 Pro, 4B) they run at plain-decode speed and only drafting
# rounds keep that host work: a parked round costs 1.0, a depth-``k`` round
# ``1 + k * per_depth + ROUND_OVERHEAD``. Charging the overhead to every
# round there made drafting look ~20% cheaper than it is, and greedy prose
# drafted where plain steps were faster. The default class keeps its curve:
# on an M4 Pro the same change costs prose (9B story and summary drafted
# less and decoded 4-5% slower), because its verify rows are cheap enough
# that the dampened curve is the better fit.
CLASS_PARK_PAYS_OVERHEAD = {False: True, True: False}

# Round-robin timing passes: the first compile kernels and are discarded.
TIMING_WARMUP_ROUNDS = 2
TIMING_ROUNDS = 21


@dataclass(frozen=True)
class GreedySchedule:
    """Fixed host costs from a tuning cell or process-local calibration.

    Attributes:
        depth: the deepest round a request may draft (``max_k``), or ``0``
            where even a fully accepted round cannot pay.
        steep_verify: diagnostic classification; selects legacy proxy costs
            only when ``verify_costs`` is empty.
        round_costs: ``round_costs[k]`` is a depth-``k`` round's cost in
            plain rounds on the frozen curve (``round_costs[0] == 1``).
            Empty when the host could not be timed: the request then drafts
            at ``depth`` every round.
    """

    depth: int
    steep_verify: bool = False
    round_costs: tuple[float, ...] = ()
    # Normalized measured target-forward costs, starting with width one.
    # Empty for an unmeasured/legacy profile; no class boundary prices a
    # measured profile's copy-drafts or changes its parked-round policy.
    verify_costs: tuple[float, ...] = ()


# Guards the two dicts only; measuring holds the per-key lock, so a first
# greedy request for one model never waits on another model's timing.
_lock = threading.Lock()
_key_locks: dict[tuple[str, int], threading.Lock] = {}
_schedules: dict[tuple[str, int], GreedySchedule] = {}

# ``(step_ms, verify_ms, draft_ms)`` with ``verify_ms[k - 1]`` the
# ``k + 1``-row verify, for ``k`` in ``1..max_k``.
Timings = tuple[float, tuple[float, ...], float]


def _check(step_ms: float, verify_ms: tuple[float, ...], draft_ms: float) -> None:
    if (
        not math.isfinite(step_ms)
        or step_ms <= 0.0
        or not verify_ms
        or any(not math.isfinite(v) or v <= 0.0 for v in verify_ms)
        or not math.isfinite(draft_ms)
        or draft_ms < 0.0
    ):
        raise ValueError(
            f"forward costs must be positive (step={step_ms}, "
            f"verify={verify_ms}, draft={draft_ms})"
        )


def round_costs(
    step_ms: float, verify_ms: tuple[float, ...], draft_ms: float
) -> tuple[float, ...]:
    """Each depth's measured round cost in plain rounds."""
    _check(step_ms, verify_ms, draft_ms)
    costs = [1.0]
    for depth, verify in enumerate(verify_ms, start=1):
        raw = verify / step_ms + depth * draft_ms / step_ms + ROUND_OVERHEAD
        costs.append(raw / (1.0 + ROUND_OVERHEAD))
    return tuple(costs)


def class_round_costs(
    steep: bool, max_k: int, *, parks_pipelined: bool = True
) -> tuple[float, ...]:
    """The class curve a request decides on: depth ``k`` costs ``k`` extra
    verified rows, ``k`` drafts and the round overhead, in plain rounds.

    Where a parked round does not pay the overhead
    (:data:`CLASS_PARK_PAYS_OVERHEAD`), a plain round is one step and only
    drafting rounds carry it. ``parks_pipelined=False`` (a request whose
    parked rounds cannot run ahead, e.g. with logits processors) charges it
    to every round again.
    """
    per_depth = CLASS_ROW_COST[steep] + CLASS_DRAFT_COST[steep]
    if CLASS_PARK_PAYS_OVERHEAD[steep] or not parks_pipelined:
        return tuple(
            round(
                (1.0 + depth * per_depth + ROUND_OVERHEAD) / (1.0 + ROUND_OVERHEAD), 4
            )
            for depth in range(max_k + 1)
        )
    return (1.0,) + tuple(
        round(1.0 + depth * per_depth + ROUND_OVERHEAD, 4)
        for depth in range(1, max_k + 1)
    )


def depth_that_pays(costs: tuple[float, ...], max_k: int) -> int:
    """``max_k`` unless no depth pays even with every draft accepted, else 0.

    ``costs`` is :func:`round_costs`' curve; a depth-``k`` round commits at
    most ``k + 1`` tokens for ``costs[k]`` plain rounds.
    """
    if max_k <= 0 or len(costs) < 2:
        return 0
    best = max(
        (depth + 1) / costs[depth] for depth in range(1, min(max_k, len(costs) - 1) + 1)
    )
    return max_k if best > 1.0 else 0


def verify_is_steep(step_ms: float, verify_ms: tuple[float, ...], max_k: int) -> bool:
    """Whether each extra verified row costs more than
    :data:`STEEP_VERIFY_ROW_COST` of a step on this host."""
    if max_k <= 0 or not verify_ms:
        return False
    widest = min(max_k, len(verify_ms))
    return (verify_ms[widest - 1] / step_ms - 1.0) / widest >= STEEP_VERIFY_ROW_COST


def reproducible_schedule(
    key: str,
    max_k: int,
    measure: Callable[[int], Timings],
) -> GreedySchedule:
    """The schedule every greedy request for ``key`` runs on.

    Decided once per process, ``key`` and ``max_k`` and then never
    revisited, so all such requests in this process share it.
    ``measure(max_k)`` returns :data:`Timings`; if it raises, the request
    drafts at the configured ``max_k`` every round rather than failing.
    """
    if max_k <= 0:
        return GreedySchedule(depth=0)
    with _lock:
        key_lock = _key_locks.setdefault((key, max_k), threading.Lock())
    with key_lock:
        with _lock:
            cached = _schedules.get((key, max_k))
        if cached is not None:
            return cached
        try:
            step_ms, verify_ms, draft_ms = measure(max_k)
            schedule = schedule_from_timings(step_ms, verify_ms, draft_ms, max_k)
        except Exception:  # noqa: BLE001 -- measurement must never fail a request
            logger.warning(
                "[MTP] could not time the target forward for reproducible "
                "requests on %r; using the configured depth %d",
                key,
                max_k,
                exc_info=True,
            )
            schedule = GreedySchedule(depth=max_k)
        with _lock:
            _schedules[(key, max_k)] = schedule
        logger.info(
            "[MTP] greedy requests on %r: depth <= %d, %s verify curve, round costs %s",
            key,
            schedule.depth,
            "measured" if schedule.verify_costs else "unmeasured",
            list(schedule.round_costs),
        )
        return schedule


def schedule_from_timings(
    step_ms: float, verify_ms: tuple[float, ...], draft_ms: float, max_k: int
) -> GreedySchedule:
    """Freeze one measured curve for request-local depth and copy decisions."""
    costs = round_costs(step_ms, verify_ms, draft_ms)
    if len(verify_ms) < max_k:
        raise ValueError("calibration omitted a configured verify depth")
    return GreedySchedule(
        depth=depth_that_pays(costs, max_k),
        steep_verify=verify_is_steep(step_ms, verify_ms, max_k),
        round_costs=costs,
        verify_costs=(1.0,) + tuple(v / step_ms for v in verify_ms),
    )


def time_target_forwards(model: Any, max_k: int) -> Timings:
    """Median ms of a one-row step, each ``k + 1``-row verify (``k`` in
    ``1..max_k``) and a draft.

    Uses the same surfaces as ``mtp_generate_step`` (``mtp_target_forward``
    when the model delegates it, ``mtp_forward``, ``make_mtp_cache``) on
    scratch caches made fresh for every call, so no request's state is read
    or written and every target shape is timed at the same (empty) context.
    Shapes are timed round-robin, :data:`TIMING_WARMUP_ROUNDS` untimed passes
    then :data:`TIMING_ROUNDS` timed ones, and each keeps its median.
    """
    import mlx.core as mx
    from mlx_lm.models import cache as cache_module

    target = getattr(model, "mtp_target_forward", None) or model
    last_hidden: list = []

    def forward(rows: int, cache):
        logits, hidden = target(
            mx.zeros((1, rows), dtype=mx.uint32),
            cache=cache,
            return_hidden=True,
            # As the generator calls it: a verify of ``rows`` rows confirms
            # ``rows - 1`` drafts (hybrid targets snapshot per position).
            n_confirmed=rows - 1,
        )
        last_hidden[:] = [hidden[:, -1:, :]]
        return logits, hidden

    fused_greedy = getattr(model, "mtp_greedy", None)

    def draft(cache):
        # The draft a greedy round actually runs: the fused argmax head when
        # the family has one (it may decline a call), else the full head.
        token = mx.zeros((1, 1), dtype=mx.uint32)
        if callable(fused_greedy):
            fused = fused_greedy(last_hidden[0], token, cache)
            if fused is not None:
                return fused
        return model.mtp_forward(last_hidden[0], token, cache)

    def target_cache():
        return cache_module.make_prompt_cache(model)

    def width(rows: int):
        return lambda cache: forward(rows, cache)

    # Shapes are timed round-robin rather than one after another, so a clock
    # or thermal drift during measurement moves every shape alike instead of
    # skewing their ratios -- the ratios are what the profile keeps.
    shapes = [(width(rows), target_cache) for rows in range(1, max_k + 2)]
    shapes.append((draft, model.make_mtp_cache))
    samples: list[list[float]] = [[] for _ in shapes]
    with mx.stream(mx.default_stream(mx.default_device())):
        for round_index in range(TIMING_WARMUP_ROUNDS + TIMING_ROUNDS):
            for index, (run, fresh_cache) in enumerate(shapes):
                # A fresh cache per sample: every width is timed at the same
                # (empty) context length.
                cache = fresh_cache()
                started = time.perf_counter()
                mx.eval(run(cache))
                if round_index >= TIMING_WARMUP_ROUNDS:
                    samples[index].append((time.perf_counter() - started) * 1000.0)
    logger.debug("[MTP] calibration raw ms by width and draft: %s", samples)
    return summarize_forward_samples(samples)


def summarize_forward_samples(samples: list[list[float]]) -> Timings:
    """Robust paired ratios from complete passes (step, verify widths, draft)."""
    if len(samples) < 3 or not samples[0]:
        raise ValueError("calibration requires step, verify and draft samples")
    if any(len(values) != len(samples[0]) for values in samples):
        raise ValueError("calibration requires complete timing passes")
    for values in samples:
        if any(not math.isfinite(v) or v <= 0.0 for v in values):
            raise ValueError("calibration samples must be finite and positive")
    # Pair each shape with its pass's one-row forward before taking the
    # median. A common frequency shift then cancels instead of moving two
    # independent medians onto different portions of the clock ramp.
    step_ms = statistics.median(samples[0])
    relative = [
        statistics.median(value / step for value, step in zip(values, samples[0]))
        for values in samples[1:]
    ]
    verify_ms = tuple(step_ms * ratio for ratio in relative[:-1])
    draft_ms = step_ms * relative[-1]
    return step_ms, verify_ms, draft_ms


def greedy_schedule(model: Any, key: str, max_k: int) -> GreedySchedule:
    """Use the tuning cell's measured costs, falling back to process calibration."""
    from ...tuning.mtp import forward_costs

    values = forward_costs(model, max_k)
    if values:
        return schedule_from_timings(values[0], values[1:-1], values[-1], max_k)
    return reproducible_schedule(
        key, max_k, lambda depth: time_target_forwards(model, depth)
    )


def request_depth_controller(costs: tuple[float, ...], max_k: int) -> DepthController:
    """A depth controller for one greedy request.

    The same EV rule as the adaptive controller -- ``argmax_K
    committed(K) / cost(K)`` with its probes -- but its cost curve is the
    host's fixed ``costs`` (fed the same constant after every round, see
    :func:`request_round_cost`) and its acceptance is this request's own.
    Every choice is therefore a function of the request's tokens and the
    process's host profile, never of the clock or of another request.
    """
    controller = DepthController(max_k=max_k)
    for depth in range(max_k + 1):
        for _ in range(COST_SEED_MIN_SAMPLES):
            controller.cost.observe(depth, request_round_cost(costs, depth))
    return controller


def request_round_costs(
    schedule: GreedySchedule, *, parks_pipelined: bool
) -> tuple[float, ...]:
    """The round-cost curve one request decides on.

    Measured profiles use the same normalized forward/draft/host-overhead
    estimate for every request. Legacy class profiles retain their existing
    adjustment when request configuration prevents parked run-ahead.
    """
    if (
        schedule.verify_costs
        or parks_pipelined
        or not schedule.round_costs
        or CLASS_PARK_PAYS_OVERHEAD[schedule.steep_verify]
    ):
        return schedule.round_costs
    return class_round_costs(
        schedule.steep_verify,
        len(schedule.round_costs) - 1,
        parks_pipelined=False,
    )


def measured_verify_cost(schedule: GreedySchedule, rows: int) -> float:
    """Clock-free copy verification price on the frozen measured curve.

    Beyond the timed widths scale the existing lookup proxy to the last
    measured width. Its tile charges still apply to large lookup blocks;
    a noisy final slope must not make an unmeasured wide block free.
    """
    if rows < 1 or not schedule.verify_costs:
        raise ValueError("a measured verify curve and positive row count are required")
    costs = schedule.verify_costs
    if rows <= len(costs):
        return costs[rows - 1]
    from .prompt_lookup import verify_cost_estimate

    return costs[-1] * verify_cost_estimate(rows) / verify_cost_estimate(len(costs))


def request_round_cost(costs: tuple[float, ...], depth: int) -> float:
    """``costs[depth]``, clamped to the deepest measured depth."""
    return costs[min(depth, len(costs) - 1)]


def reset_for_tests() -> None:
    """Forget every decided schedule. Test-only."""
    with _lock:
        _schedules.clear()
        _key_locks.clear()
