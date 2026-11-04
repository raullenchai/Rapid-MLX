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

Its own tokens, plus a host profile (:class:`GreedySchedule`) fixed once per
process, model and ``max_k`` and shared by every greedy request. Each round's
depth comes from :func:`request_depth_controller` -- the adaptive controller's
EV rule (``argmax_K committed(K) / cost(K)``) on a frozen measured round-cost
curve and this request's own acceptance. Marginal costs are bounded below
by 0.5% of a plain round and above by the host's old class increments.
Invalid or unstable measurements fall back to the class curve.
The profile's ``depth`` is 0 where even fully accepted drafts cannot pay.

The copy-draft gate prices blocks on the measured verify curve of the host's
class (``steep_verify``; ``prompt_lookup.verify_cost_estimate``). A seeded
request keeps the ``max_k`` its route already pinned and the default curve;
an operator's ``disable_auto_k`` depth is used as configured. Neither is
measured.

How the profile is measured
---------------------------

Before the first greedy request decodes, run the real generator at each
fixed K on private prompt/head caches and observers. Time whole delivery
rounds, including rollback, head alignment and the next async draft launch.
Switch depths round-robin, draining and discarding a re-entry round before
each sample so another depth's queued work cannot contaminate the interval.
Separate target-width timings choose only the fallback/copy-draft class.
No synthetic host overhead is added to already complete wall timings.
The measured profile never changes within a process; identical output across
restarts is best effort.
"""

from __future__ import annotations

import logging
import math
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

from .draft_k_controller_v2 import COST_SEED_MIN_SAMPLES, DepthController

logger = logging.getLogger(__name__)

# Extra verified row's cost, as a fraction of a one-row step, above which the
# host's verify is "steep" (the quantized-matmul vector path; M2 Pro:
# ~0.45-0.7 depending on how it is timed) rather than cheap (M4 Pro: ~0.2).
# Midway, so neither measured host is near it.
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
TIMING_ROUNDS = 5
TIMING_BUDGET_SECONDS = 5.0
MIN_DEPTH_COST = 0.005


@dataclass(frozen=True)
class GreedySchedule:
    """What a greedy request may know about its host, fixed for the process.

    Attributes:
        depth: the deepest round a request may draft (``max_k``), or ``0``
            where even a fully accepted round cannot pay.
        steep_verify: the host's class (see :data:`STEEP_VERIFY_ROW_COST`);
            it picks fallback costs, park overhead and the copy-draft curve.
        round_costs: ``round_costs[k]`` is a depth-``k`` round's cost in
            plain rounds (``round_costs[0] == 1``), measured or class fallback.
            Empty for manually constructed profiles without a cost curve.
    """

    depth: int
    steep_verify: bool = False
    round_costs: tuple[float, ...] = ()


# Guards the two dicts only; measuring holds the per-key lock, so a first
# greedy request for one model never waits on another model's timing.
_lock = threading.Lock()
_key_locks: dict[tuple[str, int], threading.Lock] = {}
_schedules: dict[tuple[str, int], GreedySchedule] = {}


class Timings(NamedTuple):
    step_ms: float
    verify_ms: tuple[float, ...]
    draft_ms: float
    # Real steady-state generator rounds, including K=0 (no added overhead).
    round_ms: tuple[float, ...] = ()


def _check(step_ms: float, verify_ms: tuple[float, ...], draft_ms: float) -> None:
    if (
        not all(math.isfinite(v) for v in (step_ms, *verify_ms, draft_ms))
        or step_ms <= 0.0
        or not verify_ms
        or min(verify_ms) <= 0.0
        or draft_ms < 0.0
    ):
        raise ValueError(
            f"forward costs must be positive (step={step_ms}, "
            f"verify={verify_ms}, draft={draft_ms})"
        )


def round_costs(
    step_ms: float,
    verify_ms: tuple[float, ...],
    draft_ms: float,
    *,
    steep: bool = False,
) -> tuple[float, ...]:
    """Each depth's measured round cost in plain rounds."""
    _check(step_ms, verify_ms, draft_ms)
    costs = [1.0]
    for depth, verify in enumerate(verify_ms, start=1):
        raw = verify / step_ms + depth * draft_ms / step_ms + ROUND_OVERHEAD
        costs.append(raw / (1.0 + ROUND_OVERHEAD) if not steep else raw)
    return tuple(costs)


def bounded_round_costs(
    costs: tuple[float, ...], steep: bool, max_k: int
) -> tuple[float, ...]:
    """Freeze a normalized curve; small timing inversions cannot make K free.

    A >10% inversion indicates an unreliable profile, rather than a cheap
    extra draft. Invalid/incomplete profiles use the entire class fallback.
    """
    fallback = class_round_costs(steep, max_k)
    if len(costs) != max_k + 1 or any(not math.isfinite(c) or c <= 0 for c in costs):
        return fallback
    normalized = tuple(c / costs[0] for c in costs)
    increments = [b - a for a, b in zip(normalized, normalized[1:])]
    if any(delta < -0.1 for delta in increments):
        return fallback
    bounded = [1.0]
    for k, delta in enumerate(increments, start=1):
        ceiling = fallback[k] - fallback[k - 1]
        bounded.append(bounded[-1] + min(ceiling, max(MIN_DEPTH_COST, delta)))
    return tuple(bounded)


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
    measure: Callable[[int], Timings | tuple[float, tuple[float, ...], float]],
) -> GreedySchedule:
    """The schedule every greedy request for ``key`` runs on.

    Decided once per process, ``key`` and ``max_k`` and then never
    revisited, so all such requests in this process share it.
    ``measure(max_k)`` returns :class:`Timings` (or legacy three-tuples).
    If it raises, the request uses class costs rather than failing.
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
        steep = False
        try:
            timings = measure(max_k)
            step_ms, verify_ms, draft_ms = timings[:3]
            _check(step_ms, verify_ms, draft_ms)
            if len(verify_ms) != max_k:
                raise ValueError("incomplete verify timings")
            steep = verify_is_steep(step_ms, verify_ms, max_k)
            measured_rounds = getattr(timings, "round_ms", ())
            if measured_rounds:
                if len(measured_rounds) != max_k + 1 or any(
                    not math.isfinite(c) or c <= 0 for c in measured_rounds
                ):
                    raise ValueError("incomplete real-round timings")
                costs = tuple(c / measured_rounds[0] for c in measured_rounds)
            else:
                # Legacy measurement providers keep their isolated-cost contract.
                costs = round_costs(step_ms, verify_ms, draft_ms, steep=steep)
            schedule = GreedySchedule(
                depth=depth_that_pays(
                    costs
                    if measured_rounds
                    else round_costs(step_ms, verify_ms, draft_ms),
                    max_k,
                ),
                steep_verify=steep,
                round_costs=bounded_round_costs(costs, steep, max_k),
            )
        except Exception:  # noqa: BLE001 -- measurement must never fail a request
            logger.warning(
                "[MTP] could not time the target forward for reproducible "
                "requests on %r; using class costs at depth <= %d",
                key,
                max_k,
                exc_info=True,
            )
            schedule = GreedySchedule(
                depth=max_k,
                steep_verify=steep,
                round_costs=class_round_costs(steep, max_k),
            )
        with _lock:
            _schedules[(key, max_k)] = schedule
        logger.info(
            "[MTP] greedy requests on %r: depth <= %d, %s verify curve, round costs %s",
            key,
            schedule.depth,
            "steep" if schedule.steep_verify else "default",
            list(schedule.round_costs),
        )
        return schedule


def advance_greedy_round(generator: Any, stats: dict[str, float]) -> int:
    """Drain one real round, preserving its launch-before-delivery pipeline.

    The first yield publishes the number of emissions already prepared by
    that round. Draining exactly those emissions avoids running any of the
    following round or synchronizing its asynchronously launched drafts.
    Also useful for standalone fixed-depth diagnostics.
    """
    next(generator)
    count = int(stats["round_tokens"])
    for _ in range(count - 1):
        next(generator)
    return count


def calibration_generator(model: Any, prompt: Any, depth: int, stats: dict[str, float]):
    """Real fixed-K decode with private caches, acceptance and cost observers."""
    from ..._seeded_sampler import RequestSeededRNG
    from .accept_counter import MTPAcceptCounter
    from .generator import mtp_generate_step

    return mtp_generate_step(
        prompt,
        model,
        max_tokens=1_000_000,
        temp=0.0,
        max_k=depth,
        disable_auto_k=True,
        prompt_lookup_enabled=False,
        accept_counter=MTPAcceptCounter(),
        timing_stats=stats,
        _cost_controller=DepthController(max_k=depth),
        lane_rng=RequestSeededRNG(0),
    )


def time_target_forwards(model: Any, max_k: int) -> Timings:
    """Measure real steady-state wall ms per K, with a five-second soft budget.

    Each depth retains its own real generator/cache across round-robin visits.
    On re-entry, drain outstanding GPU work outside timing, then discard one
    whole round to restore the pipeline before timing the next whole round.
    No per-token eval or artificial consumer delay is inserted. Acceptance,
    rollback, head-history rebuild/commit and async launches are production
    code, not a reconstruction. The scratch prompt uses only token 0 and has
    16 tokens so it needs neither a tokenizer nor any request-owned state.
    The budget is checked between timing sweeps (one sweep can overrun);
    at least three samples per depth are required or the class fallback wins.
    Greedy verification still draws uniforms; a private RNG keeps calibration
    from advancing the process sampling stream.
    """
    import mlx.core as mx
    from mlx_lm.models import cache as cache_module

    started = time.perf_counter()
    target = getattr(model, "mtp_target_forward", None) or model
    samples: list[list[float]] = [[] for _ in range(max_k + 1)]
    generators = []
    stats = [{} for _ in samples]
    prompt = mx.zeros((16,), dtype=mx.uint32)

    def check_budget():
        if time.perf_counter() - started > TIMING_BUDGET_SECONDS:
            raise TimeoutError("startup round calibration exceeded time budget")

    with mx.stream(mx.default_stream(mx.default_device())):
        try:
            # Isolated widths are ONLY for host classification. Allocate once,
            # not a new prefill for every shape and every measurement pass.
            cache = cache_module.make_prompt_cache(model)
            _, hidden = target(
                prompt[None], cache=cache, return_hidden=True, n_confirmed=0
            )
            mx.eval(hidden)
            widths = [[] for _ in range(max_k + 1)]
            for turn in range(3):
                for rows in range(1, max_k + 2):
                    check_budget()
                    t0 = time.perf_counter()
                    mx.eval(
                        target(
                            mx.zeros((1, rows), dtype=mx.uint32),
                            cache=cache,
                            return_hidden=True,
                            n_confirmed=rows - 1,
                        )
                    )
                    if turn:
                        widths[rows - 1].append((time.perf_counter() - t0) * 1000)
            # Retain legacy draft_ms diagnostics, not used to price real rounds.
            head_cache = model.make_mtp_cache()
            mx.eval(model.mtp_forward(hidden[:, -1:, :], prompt[:1][None], head_cache))
            t0 = time.perf_counter()
            fused = getattr(model, "mtp_greedy", None)
            result = (
                fused(hidden[:, -1:, :], prompt[:1][None], head_cache)
                if callable(fused)
                else None
            )
            if result is None:
                result = model.mtp_forward(
                    hidden[:, -1:, :], prompt[:1][None], head_cache
                )
            mx.eval(result)
            draft_ms = (time.perf_counter() - t0) * 1000
            for depth in range(max_k + 1):
                check_budget()
                gen = calibration_generator(model, prompt, depth, stats[depth])
                generators.append(gen)
                advance_greedy_round(gen, stats[depth])  # prefill/bootstrap
            for turn in range(TIMING_WARMUP_ROUNDS + TIMING_ROUNDS):
                if time.perf_counter() - started > TIMING_BUDGET_SECONDS:
                    break
                for depth, gen in enumerate(generators):
                    mx.synchronize()
                    advance_greedy_round(gen, stats[depth])  # restore pipeline
                    t0 = time.perf_counter()
                    advance_greedy_round(gen, stats[depth])
                    elapsed = (time.perf_counter() - t0) * 1000
                    if int(stats[depth]["round_depth"]) != depth:
                        raise ValueError(
                            "scratch cache cannot exercise requested depth"
                        )
                    if turn >= TIMING_WARMUP_ROUNDS:
                        samples[depth].append(elapsed)
            if min(map(len, samples)) < 3:
                raise TimeoutError("insufficient startup round samples within budget")
        finally:
            # Flush pending async work before releasing private caches, even
            # if calibration fails. No serving controller/counter was touched.
            mx.synchronize()
            for gen in generators:
                gen.close()
    medians = tuple(statistics.median(values) for values in samples)
    for values, median in zip(samples, medians):
        if (
            not math.isfinite(median)
            or median <= 0
            or statistics.median(abs(v - median) for v in values) > 0.2 * median
        ):
            raise ValueError("unstable startup round timings")
    width_ms = tuple(statistics.median(values) for values in widths)
    return Timings(width_ms[0], width_ms[1:], draft_ms, medians)


def greedy_schedule(model: Any, key: str, max_k: int) -> GreedySchedule:
    """:func:`reproducible_schedule` measured with :func:`time_target_forwards`."""
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

    ``schedule.round_costs`` assumes parked rounds run ahead of delivery. A
    request whose parked rounds cannot (fixed by its own configuration, never
    by timing) uses the same frozen curve with overhead on every round.
    """
    if (
        parks_pipelined
        or not schedule.round_costs
        or CLASS_PARK_PAYS_OVERHEAD[schedule.steep_verify]
    ):
        return schedule.round_costs
    # Preserve the measured increments. Only the parked baseline changes:
    # drafting rounds already pay overhead, parks now pay it as well.
    adjusted = [1.0]
    for cost in schedule.round_costs[1:]:
        adjusted.append(
            round(max(adjusted[-1] + MIN_DEPTH_COST, cost / (1.0 + ROUND_OVERHEAD)), 12)
        )
    return tuple(adjusted)


def request_round_cost(costs: tuple[float, ...], depth: int) -> float:
    """``costs[depth]``, clamped to the deepest measured depth."""
    return costs[min(depth, len(costs) - 1)]


def reset_for_tests() -> None:
    """Forget every decided schedule. Test-only."""
    with _lock:
        _schedules.clear()
        _key_locks.clear()
