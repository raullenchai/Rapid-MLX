# SPDX-License-Identifier: Apache-2.0
"""Universal MTP draft-``k`` auto-tune controller (0.9.11, PR-1 of Gemma 4 MTP).

Background
----------

The chain-MTP generator in :mod:`vllm_mlx.spec_decode.mtp.generator`
proposes a single draft token per verify backbone pass. A future
Gemma-4 sidecar (PR-2/3) will let callers propose ``k`` draft tokens
per pass, but choosing the right ``k`` is workload-dependent:

* On highly-predictable content (JSON, tool calls, code
  boilerplate) the accept-rate stays close to 1.0 and larger ``k``
  amortizes the verify cost across more accepted tokens.
* On free-form chat / creative prose the accept-rate drops and a
  larger ``k`` inflates the verify forward cost without a matching
  accept rate — a net regression.
* The optimum drifts within a single session — a code-heavy request
  followed by a chat request wants a different ``k``.

This module implements the *runtime feedback loop* that adjusts
``k`` based on the rolling accept rate. It is intentionally pure
Python (no MLX imports) so it can be unit-tested without spawning
a server, and it uses hysteresis + a cooldown so a noisy window
around the thresholds doesn't oscillate ``k`` up-and-down every
attempt.

Design contract
---------------

* **Rolling window.** The controller maintains a bounded deque of
  the last ``window`` attempt outcomes (``True`` / ``False``). The
  accept-rate is ``sum(True) / len(window)``.
* **Hysteresis thresholds.** If the rate is ``>= upshift_threshold``
  and ``current_k < k_max``, ``k`` is bumped by 1. If the rate is
  ``<= downshift_threshold`` and ``current_k > k_min``, ``k`` is
  cut by 1. In between → hold. The gap between the two thresholds
  is the hysteresis band that prevents the ``k = 3 ↔ k = 4``
  oscillation the naïve single-threshold controller would exhibit
  when the rate hovers right at the boundary.
* **Cooldown.** Adjustments are only considered every ``cooldown``
  recorded attempts. Without cooldown, a lucky streak of accepts
  right after startup would bump ``k`` up before the window even
  has ``k_max - k_min`` samples of steady-state behavior.
* **Bounds.** ``k`` is clamped to ``[k_min, k_max]`` at all times.
  ``k_min == k_max`` is the degenerate case (auto-tune disabled by
  the operator via a tight bound) and MUST NOT crash — the
  controller still records attempts but never adjusts.
* **Validation at construction time.** NaN, ``+inf``, ``-inf``, and
  values outside ``(0, 1)`` on either threshold raise
  :class:`ValueError`. This is deliberate — the memory gotcha
  ``pydantic-field-does-not-reject-nan`` notes that Pydantic
  ``Field(ge=, le=)`` accepts NaN silently, so any float that
  reaches a math kernel needs an explicit ``math.isfinite`` gate.
  The controller is upstream of any math kernel, so we gate here.
* **``upshift_threshold`` must be strictly greater than
  ``downshift_threshold``.** An operator passing them the other
  way around would produce a controller that ratchets ``k`` down
  on high accept rates — a silent inversion that would be very
  hard to notice at scale. Fail loud at construction time.

Thread-safety
-------------

The MTP generator runs from the scheduler thread; the metrics
endpoint reads :meth:`snapshot` from the FastAPI worker pool. All
state changes and reads acquire ``self._lock`` — the lock is held
for O(1) work (a deque push, a small sum over the deque, one or two
int assignments) so contention is negligible in practice.

Public surface
--------------

* :class:`DraftKController` — the controller class.
* :func:`get_global_controller` — module-level singleton getter.
  Returns ``None`` when auto-tune is disabled (the default). The
  generator uses ``get_global_controller() or None`` to preserve
  the static-``k`` fast path when auto-tune is off — no extra
  branches in the hot loop except the ``is None`` check.
* :func:`install_global_controller` — CLI-boot-time installer.
  Called from ``vllm_mlx/cli.py::serve_command`` when the operator
  passes ``--mtp-draft-k-auto-tune``. Idempotent — a subsequent
  install replaces the singleton (used by the test suite).
* :func:`clear_global_controller` — teardown hook, primarily used
  by the test fixture to isolate cases.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Any


class DraftKController:
    """Runtime feedback loop that adjusts MTP draft-``k`` from accept rate.

    See the module docstring for the design contract. Instances are
    thread-safe.

    Args:
        k_min: Lower bound on ``k`` (inclusive). Must be ``>= 1``.
        k_max: Upper bound on ``k`` (inclusive). Must be ``>= k_min``.
        k_start: Starting value for ``k``. Defaults to ``k_min``.
            Must satisfy ``k_min <= k_start <= k_max``.
        window: Rolling window size (in attempts) used to compute
            the accept rate. Defaults to 64.
        upshift_threshold: Accept-rate at or above which ``k`` is
            bumped by 1 (when ``current_k < k_max``). Defaults to
            ``0.80``. Must be a finite float in ``(0, 1]``.
        downshift_threshold: Accept-rate at or below which ``k`` is
            cut by 1 (when ``current_k > k_min``). Defaults to
            ``0.40``. Must be a finite float in ``[0, 1)`` and
            strictly less than ``upshift_threshold``.
        cooldown: Minimum number of recorded attempts between
            adjustment decisions. Defaults to 128. Must be ``>= 1``.
    """

    def __init__(
        self,
        *,
        k_min: int = 1,
        k_max: int = 4,
        k_start: int | None = None,
        window: int = 64,
        upshift_threshold: float = 0.80,
        downshift_threshold: float = 0.40,
        cooldown: int = 128,
    ) -> None:
        # --- integer bounds ------------------------------------------------
        # ``bool`` is a subclass of ``int`` in Python; reject it explicitly
        # so a ``True``/``False`` mistake in a caller (e.g. from a
        # ``argparse`` boolean flag routed to the wrong kwarg) fails loud
        # rather than silently ratcheting ``k`` to 0.
        for name, val in (
            ("k_min", k_min),
            ("k_max", k_max),
            ("window", window),
            ("cooldown", cooldown),
        ):
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"{name} must be int, got {type(val).__name__}")
        if k_min < 1:
            raise ValueError(f"k_min must be >= 1, got {k_min}")
        if k_max < k_min:
            raise ValueError(f"k_max ({k_max}) must be >= k_min ({k_min})")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if cooldown < 1:
            raise ValueError(f"cooldown must be >= 1, got {cooldown}")

        # --- threshold validation -----------------------------------------
        # Pydantic ``Field(ge=, le=)`` does not reject NaN (per the
        # ``pydantic-field-does-not-reject-nan`` gotcha) — even though
        # we're not going through Pydantic here, any operator-supplied
        # float needs an explicit ``math.isfinite`` guard because
        # comparisons against NaN silently return False.
        for name, val in (
            ("upshift_threshold", upshift_threshold),
            ("downshift_threshold", downshift_threshold),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be float, got {type(val).__name__}")
            if not math.isfinite(float(val)):
                raise ValueError(f"{name} must be finite, got {val!r}")

        upshift_threshold = float(upshift_threshold)
        downshift_threshold = float(downshift_threshold)

        if not (0.0 < upshift_threshold <= 1.0):
            raise ValueError(
                f"upshift_threshold must be in (0.0, 1.0], got {upshift_threshold}"
            )
        if not (0.0 <= downshift_threshold < 1.0):
            raise ValueError(
                f"downshift_threshold must be in [0.0, 1.0), got {downshift_threshold}"
            )
        if upshift_threshold <= downshift_threshold:
            raise ValueError(
                "upshift_threshold must be strictly greater than "
                f"downshift_threshold; got upshift={upshift_threshold}, "
                f"downshift={downshift_threshold}. A controller with the "
                "thresholds inverted would ratchet k DOWN on high accept "
                "rates — almost certainly a config error."
            )

        # --- starting-k validation ----------------------------------------
        if k_start is None:
            k_start = k_min
        if isinstance(k_start, bool) or not isinstance(k_start, int):
            raise TypeError(f"k_start must be int, got {type(k_start).__name__}")
        if not (k_min <= k_start <= k_max):
            raise ValueError(
                f"k_start ({k_start}) must satisfy k_min ({k_min}) "
                f"<= k_start <= k_max ({k_max})"
            )

        self._k_min = k_min
        self._k_max = k_max
        self._current_k = k_start
        self._window_size = window
        self._upshift = upshift_threshold
        self._downshift = downshift_threshold
        self._cooldown = cooldown

        # ``deque(maxlen=window)`` is O(1) push + auto-eviction, which
        # is the cheapest ring buffer Python ships.
        self._buf: deque[bool] = deque(maxlen=window)
        # Attempts recorded since the last adjustment/hold decision.
        # Resets to zero on every re-evaluation (whether we changed
        # ``k`` or held). This is what enforces the ``cooldown`` gate.
        self._steps_since_eval = 0
        self._adjust_count = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # recording side
    # ------------------------------------------------------------------

    def record_attempt(self, accepted: bool) -> None:
        """Record one MTP verify outcome and maybe adjust ``k``.

        Called once per verify backbone pass in
        :func:`vllm_mlx.spec_decode.mtp.generator.mtp_generate_step`
        AFTER the accept/reject decision has been made. When the
        rolling window has enough samples and the cooldown has
        elapsed, the accept rate is compared against the thresholds
        and ``k`` is bumped, cut, or held.

        Args:
            accepted: ``True`` when the verify pass accepted the
                draft token(s); ``False`` when it rejected. Anything
                truthy is coerced to ``True`` for defensive symmetry
                with ``bool()`` in call sites.
        """
        accepted_bool = bool(accepted)
        with self._lock:
            self._buf.append(accepted_bool)
            self._steps_since_eval += 1

            # Fast path: not enough attempts since the last decision
            # OR the window isn't fully populated yet. Waiting for a
            # full window prevents an "all-accept first 5 samples"
            # cold start from ratcheting ``k`` all the way up before
            # steady state kicks in.
            if self._steps_since_eval < self._cooldown:
                return
            if len(self._buf) < self._window_size:
                return

            rate = sum(1 for a in self._buf if a) / len(self._buf)

            # Every re-evaluation resets the cooldown counter,
            # whether we adjusted or held — the invariant is
            # "one decision per cooldown window", not "one
            # adjustment per cooldown window". Otherwise a
            # persistently-in-band accept rate would recompute
            # ``rate`` every single attempt after the first
            # cooldown, which is wasted work and also a subtle
            # source of oscillation (the deque contents shift by
            # one every attempt, and edge cases at the
            # thresholds could produce alternating decisions).
            self._steps_since_eval = 0

            if rate >= self._upshift and self._current_k < self._k_max:
                self._current_k += 1
                self._adjust_count += 1
            elif rate <= self._downshift and self._current_k > self._k_min:
                self._current_k -= 1
                self._adjust_count += 1
            # else: hold. No state change beyond the cooldown reset.

    # ------------------------------------------------------------------
    # read side
    # ------------------------------------------------------------------

    def current_k(self) -> int:
        """Return the currently-in-use ``k``.

        Callers reading this from the MTP generator hot loop hold
        onto the returned value for the next verify pass; the next
        :meth:`record_attempt` may adjust it, but the reader-side
        value is used for the in-flight step, not the just-completed
        one. This is intentional — swapping ``k`` mid-verify would
        require additional cache bookkeeping that PR-1 does not
        implement.
        """
        with self._lock:
            return self._current_k

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of controller state for observability.

        Consumed by :mod:`vllm_mlx.routes.metrics` to render the
        ``rapid_mlx_spec_decode_mtp_current_draft_k`` gauge and by
        the ``/logs`` debugging surface. The snapshot is causally
        consistent — all fields are read under a single lock so a
        concurrent ``record_attempt`` either lands fully before or
        fully after the snapshot, never mid-tuple.

        Returns:
            A dict with the following stable keys:

            * ``current_k`` — current draft-``k``.
            * ``k_min``, ``k_max`` — configured bounds.
            * ``window_size`` — configured window size.
            * ``window_fill`` — number of samples currently in the
              rolling window (``<= window_size``; ramps up during
              cold start).
            * ``window_accept_rate`` — accept rate over the current
              window contents. Returns ``0.0`` when the window is
              empty (Prometheus convention: no data → 0, not NaN).
            * ``cooldown_steps`` — configured cooldown.
            * ``steps_since_eval`` — attempts since the last
              adjustment/hold decision.
            * ``adjust_count`` — cumulative number of adjustments
              (up + down) since the controller was constructed.
            * ``upshift_threshold``, ``downshift_threshold`` —
              configured hysteresis band.
        """
        with self._lock:
            n = len(self._buf)
            accepts = sum(1 for a in self._buf if a)
            return {
                "current_k": self._current_k,
                "k_min": self._k_min,
                "k_max": self._k_max,
                "window_size": self._window_size,
                "window_fill": n,
                "window_accept_rate": (accepts / n) if n else 0.0,
                "cooldown_steps": self._cooldown,
                "steps_since_eval": self._steps_since_eval,
                "adjust_count": self._adjust_count,
                "upshift_threshold": self._upshift,
                "downshift_threshold": self._downshift,
            }


# ---------------------------------------------------------------------------
# Module-level singleton — installed by CLI when --mtp-draft-k-auto-tune=True.
# ---------------------------------------------------------------------------

_global_controller: DraftKController | None = None
_global_controller_lock = threading.Lock()


def get_global_controller() -> DraftKController | None:
    """Return the process-global :class:`DraftKController`, if installed.

    Returns ``None`` when auto-tune has not been enabled (the
    default). The MTP generator uses this to preserve the static-
    ``k`` fast path — only an ``is None`` check separates the two
    modes in the hot loop.
    """
    return _global_controller


def install_global_controller(controller: DraftKController) -> None:
    """Install (or replace) the process-global controller.

    Called from ``vllm_mlx/cli.py::serve_command`` at boot when the
    operator passes ``--mtp-draft-k-auto-tune``. Idempotent: a
    subsequent install replaces the singleton. The test suite uses
    the replacement behavior to isolate cases without relying on
    module-reload tricks.

    Args:
        controller: A pre-constructed :class:`DraftKController`.
            Passing the constructed instance rather than raw kwargs
            keeps the CLI-side validation and error-formatting in
            one place.
    """
    global _global_controller
    if not isinstance(controller, DraftKController):
        raise TypeError(
            f"controller must be a DraftKController, got {type(controller).__name__}"
        )
    with _global_controller_lock:
        _global_controller = controller


def clear_global_controller() -> None:
    """Reset the process-global controller to ``None``.

    Primarily a test hook — production code never uninstalls the
    controller once boot has completed. Fixtures call this in
    teardown to isolate cases.
    """
    global _global_controller
    with _global_controller_lock:
        _global_controller = None
