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
EV rule (``argmax_K committed(K) / cost(K)``) on the round costs of the
host's class (:data:`CLASS_ROW_COST`) and this request's own acceptance. So a host where a 2-row verify is
cheap drafts K=1 on prose and K=2 on structured output, as the adaptive
controller does, and a host where every verified row costs most of a step
(the quantized-matmul vector path) drafts only when nearly every draft lands.
The profile's ``depth`` is 0 where even fully accepted drafts cannot pay.

The copy-draft gate prices blocks on the measured verify curve of the host's
class (``steep_verify``; ``prompt_lookup.verify_cost_estimate``). A seeded
request keeps the ``max_k`` its route already pinned and the default curve;
an operator's ``disable_auto_k`` depth is used as configured. Neither is
measured.

How the profile is measured
---------------------------

Before the first greedy request decodes, the scheduler times the target
forward at one row and at every ``k + 1`` rows up to ``max_k + 1`` plus one
greedy draft step (:func:`time_target_forwards`: a few dozen scratch forwards
on caches nobody else reads, timed round-robin). The timings choose a class,
not a curve: the request decides on its class's fixed curve, so timing noise
between boots changes nothing unless it moves the host across the class
threshold, which sits far from the hosts measured (per extra verified row,
~0.2 of a step on an M4 Pro, ~0.45-0.7 on an M2 Pro; threshold 0.33).
Within one process the profile never changes, which is the guarantee;
identical output across restarts is best effort.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
# :data:`ROUND_OVERHEAD` they reproduce the round costs the adaptive
# controller learns from the clock on those hosts (M2 Pro: 1.53 / 2.05).
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
TIMING_ROUNDS = 7


@dataclass(frozen=True)
class GreedySchedule:
    """What a greedy request may know about its host, fixed for the process.

    Attributes:
        depth: the deepest round a request may draft (``max_k``), or ``0``
            where even a fully accepted round cannot pay.
        steep_verify: the host's class (see :data:`STEEP_VERIFY_ROW_COST`);
            it picks the round-cost curve and the copy-draft gate's curve.
        round_costs: ``round_costs[k]`` is a depth-``k`` round's cost in
            plain rounds on the class curve (``round_costs[0] == 1``).
            Empty when the host could not be timed: the request then drafts
            at ``depth`` every round.
    """

    depth: int
    steep_verify: bool = False
    round_costs: tuple[float, ...] = ()


# Guards the two dicts only; measuring holds the per-key lock, so a first
# greedy request for one model never waits on another model's timing.
_lock = threading.Lock()
_key_locks: dict[tuple[str, int], threading.Lock] = {}
_schedules: dict[tuple[str, int], GreedySchedule] = {}

# ``(step_ms, verify_ms, draft_ms)`` with ``verify_ms[k - 1]`` the
# ``k + 1``-row verify, for ``k`` in ``1..max_k``.
Timings = tuple[float, tuple[float, ...], float]


def _check(step_ms: float, verify_ms: tuple[float, ...], draft_ms: float) -> None:
    if step_ms <= 0.0 or not verify_ms or min(verify_ms) <= 0.0 or draft_ms < 0.0:
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


def class_round_costs(steep: bool, max_k: int) -> tuple[float, ...]:
    """The class curve a request decides on: depth ``k`` costs ``k`` extra
    verified rows, ``k`` drafts and the round overhead, in plain rounds.

    Where a parked round does not pay the overhead
    (:data:`CLASS_PARK_PAYS_OVERHEAD`), a plain round is one step and only
    drafting rounds carry it.
    """
    per_depth = CLASS_ROW_COST[steep] + CLASS_DRAFT_COST[steep]
    if CLASS_PARK_PAYS_OVERHEAD[steep]:
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
            steep = verify_is_steep(step_ms, verify_ms, max_k)
            schedule = GreedySchedule(
                depth=depth_that_pays(round_costs(step_ms, verify_ms, draft_ms), max_k),
                steep_verify=steep,
                round_costs=class_round_costs(steep, max_k),
            )
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
            "steep" if schedule.steep_verify else "default",
            list(schedule.round_costs),
        )
        return schedule


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
    medians = [statistics.median(shape_samples) for shape_samples in samples]
    step_ms, verify_ms, draft_ms = medians[0], tuple(medians[1:-1]), medians[-1]
    return step_ms, verify_ms, draft_ms


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


def request_round_cost(costs: tuple[float, ...], depth: int) -> float:
    """``costs[depth]``, clamped to the deepest measured depth."""
    return costs[min(depth, len(costs) - 1)]


def reset_for_tests() -> None:
    """Forget every decided schedule. Test-only."""
    with _lock:
        _schedules.clear()
        _key_locks.clear()
