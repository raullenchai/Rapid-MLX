"""Per-model Metal memory budgeting (#2858).

The engine historically exposed one global knob — ``--gpu-memory-utilization``,
default ``0.90`` — that had to be tuned per model by hand: a 12B checkpoint
runs fine at ``0.75`` while a 20B MoE with an ~11.2 GB Metal footprint needs
``0.95`` on the same 16 GB Mac.  Worse, when the cap was too low the model
still loaded and reported healthy while the D-METAL-CAP admission gate
deterministically rejected every request with HTTP 503.

This module closes that loop with two pieces, both pure and unit-testable:

* :func:`plan_metal_limit` — resolve the effective utilization for one loaded
  model.  When the operator passed an explicit ``--gpu-memory-utilization``
  the value is honored verbatim (advanced override).  In auto mode (the new
  default) the limit is sized to the model actually loaded: the MEASURED
  weight footprint (``mx.get_active_memory()`` right after load — no disk
  heuristics) plus a runtime headroom, clamped to
  ``[AUTO_UTILIZATION_FLOOR, AUTO_UTILIZATION_CEILING]`` of the device's
  recommended working-set budget.  The floor keeps small models byte-identical
  to the historical ``0.90`` default; the ceiling leaves the OS a margin the
  same way vLLM's ``gpu_memory_utilization`` never goes to 1.0.

* :func:`format_preflight_error` — the actionable admission-impossible
  message required by #2858: required vs available memory plus concrete
  remediations.  Raised (wrapped in :class:`MetalPreflightError`) by
  ``Scheduler.preflight_metal_admission`` when even a modest request could
  never be admitted under the resolved cap, so startup fails BEFORE the
  server reports the model ready instead of serving deterministic 503s.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

# Auto-mode bounds. The floor matches the historical global default so any
# model that fit comfortably before resolves to a byte-identical limit; the
# ceiling reserves ~3% of the device budget for allocator slack so auto mode
# never plans right up to the working-set edge.
AUTO_UTILIZATION_FLOOR = 0.90
AUTO_UTILIZATION_CEILING = 0.97

# The knob's absolute maximum: an explicit --gpu-memory-utilization may go
# all the way to 1.0, past the auto ceiling. Advice to raise the knob is
# only impossible (and therefore suppressed) at or beyond this value; a
# tiny slack absorbs float representation of "1.0".
MAX_UTILIZATION = 1.0 - 1e-9

# Runtime headroom charged on top of the measured weight footprint in auto
# mode: KV cache for in-flight requests, activation workspace, Metal heap
# fragmentation. Fractional so big models reserve proportionally more, with an
# absolute floor so tiny models still get a workable slice.
_AUTO_HEADROOM_FRACTION = 0.08
_AUTO_HEADROOM_MIN_BYTES = 512 * 1024**2


# ── Process-wide utilization ratchet (codex round 2 BLOCKING #2) ──
# In multi-model mode each BatchedEngine resolves its own budget, but the
# Metal allocator and ``mx.get_active_memory()`` are process-wide: when a
# second model auto-raises the limit, the FIRST model's scheduler must not
# keep enforcing its earlier, lower cap against an ``active`` figure that
# now includes the new model — that would reject every request to the
# already-resident model. Every resolution notes the utilization here;
# schedulers read the floor (with a generation counter so their cached cap
# invalidates on each upward ratchet) and enforce
# ``max(own configured utilization, process floor)``. The floor only ever
# rises, mirroring the allocation limit itself.
_process_floor_lock = threading.Lock()
_process_utilization_floor: float = 0.0
_process_floor_generation: int = 0


def note_resolved_utilization(utilization: float) -> None:
    """Record a resolved per-engine utilization into the process floor."""
    global _process_utilization_floor, _process_floor_generation
    with _process_floor_lock:
        if utilization > _process_utilization_floor:
            _process_utilization_floor = utilization
            _process_floor_generation += 1


def process_utilization_floor() -> tuple[float, int]:
    """Return ``(floor, generation)`` for cache-invalidating readers."""
    with _process_floor_lock:
        return _process_utilization_floor, _process_floor_generation


def ratchet_utilization_and_apply(
    utilization: float,
    apply: Callable[[float], None] | None = None,
) -> tuple[float, int]:
    """Ratchet the floor and apply the effective value atomically.

    Concurrent model loads each publish a resolved utilization and then
    install the corresponding Metal allocation limit. Doing those as two
    separate lock acquisitions leaves a window (codex round 4 BLOCKING #1)
    where a loader holding an older, lower floor applies its limit LAST —
    schedulers would then enforce the newer higher cap while Metal holds
    the stale lower allocation limit. Holding the lock across both the
    ratchet and the ``apply`` callback serializes the setter calls in
    floor order, so the last limit installed always reflects the highest
    published utilization.

    ``apply`` receives the effective (post-ratchet) utilization —
    ``max(utilization, floor)`` — and must contain its own failures if a
    setter error should not propagate (the callers do; see codex round 1
    BLOCKING #2). Returns ``(effective_utilization, generation)``.
    """
    global _process_utilization_floor, _process_floor_generation
    with _process_floor_lock:
        if utilization > _process_utilization_floor:
            _process_utilization_floor = utilization
            _process_floor_generation += 1
        effective = max(utilization, _process_utilization_floor)
        if apply is not None:
            apply(effective)
        return effective, _process_floor_generation


class MetalPreflightError(RuntimeError):
    """The resolved Metal cap can never admit a request for this model.

    Raised during engine startup — before the server reports the model
    ready — so the operator sees one actionable message instead of a
    healthy-looking model whose every request returns HTTP 503.
    """


@dataclass(frozen=True)
class MetalBudgetPlan:
    """The resolved Metal allocation budget for one loaded model."""

    weights_bytes: int
    device_budget_bytes: int
    requested_utilization: float | None
    resolved_utilization: float
    limit_bytes: int
    mode: str  # "manual" (operator override) or "auto"


def plan_metal_limit(
    *,
    weights_bytes: int,
    device_budget_bytes: int,
    requested_utilization: float | None = None,
) -> MetalBudgetPlan:
    """Resolve the Metal allocation limit for one loaded model.

    Args:
        weights_bytes: Measured Metal footprint of the loaded weights
            (``mx.get_active_memory()`` right after load). ``<= 0`` means
            "no measurement available" and auto mode falls back to the
            historical floor.
        device_budget_bytes: The device's recommended working-set size
            (``max_recommended_working_set_size``). Must be positive.
        requested_utilization: The operator's explicit
            ``--gpu-memory-utilization``, or ``None`` for auto.

    Returns:
        A :class:`MetalBudgetPlan`. ``resolved_utilization`` is the value the
        engine must feed to BOTH ``mx.set_memory_limit`` and the scheduler's
        D-METAL-CAP admission gate — the two enforcement points must always
        agree on the same cap.
    """
    if device_budget_bytes <= 0:
        raise ValueError("device_budget_bytes must be positive")

    if requested_utilization is not None:
        resolved = float(requested_utilization)
        mode = "manual"
    elif weights_bytes <= 0:
        resolved = AUTO_UTILIZATION_FLOOR
        mode = "auto"
    else:
        headroom = max(
            int(weights_bytes * _AUTO_HEADROOM_FRACTION), _AUTO_HEADROOM_MIN_BYTES
        )
        needed = (weights_bytes + headroom) / device_budget_bytes
        resolved = min(max(needed, AUTO_UTILIZATION_FLOOR), AUTO_UTILIZATION_CEILING)
        mode = "auto"

    return MetalBudgetPlan(
        weights_bytes=max(0, int(weights_bytes)),
        device_budget_bytes=int(device_budget_bytes),
        requested_utilization=requested_utilization,
        resolved_utilization=resolved,
        limit_bytes=int(device_budget_bytes * resolved),
        mode=mode,
    )


def format_preflight_error(
    *,
    required_bytes: int,
    active_bytes: int,
    min_kv_bytes: int,
    cap_bytes: int,
    utilization: float,
    device_budget_bytes: int,
) -> str:
    """Build the actionable admission-impossible startup message (#2858).

    The remediation list is tailored to what can still help (codex rounds
    1 and 3): "increase --gpu-memory-utilization" is suggested while the
    enforced utilization sits below 1.0 — an explicit override can legally
    go all the way there, even past the auto ceiling. Only at a full 1.0
    is the knob truly exhausted, and the honest advice becomes that this
    Mac does not have the memory for this configuration.
    """
    if utilization < MAX_UTILIZATION:
        remediation = (
            "Increase --gpu-memory-utilization, reduce context length or "
            "concurrency, close memory-heavy apps, or choose a smaller model."
        )
    else:
        remediation = (
            "This Mac does not have enough unified memory for this "
            "configuration even at the maximum Metal budget — reduce "
            "context length or concurrency, close memory-heavy apps, "
            "retry after in-flight requests on other models finish, or "
            "choose a smaller model."
        )
    return (
        f"This model needs approximately {required_bytes / 1e9:.1f} GB of "
        f"Metal memory for the current configuration (weights and runtime "
        f"{active_bytes / 1e9:.1f} GB + minimum KV cache "
        f"{min_kv_bytes / 1e9:.1f} GB), but the current limit is "
        f"{cap_bytes / 1e9:.1f} GB "
        f"(gpu_memory_utilization={utilization:g} of the "
        f"{device_budget_bytes / 1e9:.1f} GB Metal working-set budget). "
        f"{remediation}"
    )
