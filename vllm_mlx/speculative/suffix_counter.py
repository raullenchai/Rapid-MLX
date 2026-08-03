# SPDX-License-Identifier: Apache-2.0
"""Process-local telemetry for SuffixDecoding.

The scheduler already tracked everything an operator needs to answer
"I enabled suffix decoding and nothing got faster — why?": verify count,
acceptance, and a seven-way breakdown of *why* a step fell through to a
plain forward. None of it had an exit. ``_suffix_stats`` was written and
never read, so the only way to see acceptance was to patch a log line in
and rebuild.

That mattered in practice. The same build, flags and prompt gave ~2x on
one Mac and ~1.0x on another; without acceptance numbers the obvious
guess is "the drafter isn't proposing on that machine". It was — both
machines accepted 82.3% of drafts, byte-identical. The difference was
entirely in how the two chips scale a K-wide verify forward, which is
only diagnosable once you can see that acceptance is fine.

Mirrors :class:`vllm_mlx.spec_decode.mtp.MTPAcceptCounter`: monotonic
counters, O(1) locked integer adds, no allocation and no MLX evals on
the hot path. ``reset()`` exists for tests only — the Prometheus surface
relies on counters never decrementing.
"""

from __future__ import annotations

import threading

__all__ = ["SuffixAcceptCounter", "get_global_counter", "reset_global_counter"]


class SuffixAcceptCounter:
    """Thread-safe counters for the SuffixDecoding verify loop.

    Monotonic for the process lifetime:

    * ``verify_steps`` — steps that ran a verify forward.
    * ``fallthrough_steps`` — steps that took the plain forward instead.
      The ``ft_*`` breakdown sums to this.
    * ``draft_tokens_proposed`` — sum of K over verify steps (draft
      *tokens*, not proposals).
    * ``draft_tokens_accepted`` — subset that matched the target's greedy
      prediction. ``accepted / proposed`` is the acceptance ratio; the
      speedup ceiling is ``(1 + accepted_per_verify) / cost_ratio(K)``.
    * ``cooldown_trips`` — times the backoff window re-armed. Healthy
      high-overlap traffic shows zero; low-overlap shows a handful and
      then goes quiet.

    ``current_k`` and ``backoff_level`` are gauges (last-write-wins), not
    counters — they describe the drafter's present state.
    """

    _FT_REASONS = (
        "batch_size",
        "uids_size",
        "non_greedy",
        "logits_processors",
        "no_draft",
        "cooldown",
        "non_trimmable_cache",
        # Drafter or verify-forward raised; the step still took a plain
        # forward, so it must appear in the breakdown or verify +
        # fallthrough stops reconciling with actual decode steps exactly
        # when something is going wrong.
        "error",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._verify_steps = 0
        self._fallthrough_steps = 0
        self._proposed = 0
        self._accepted = 0
        self._cooldown_trips = 0
        self._errors = 0
        self._ft = dict.fromkeys(self._FT_REASONS, 0)
        # Gauges.
        self._current_k = 0
        self._backoff_level = 0

    def record_verify(self, proposed: int, accepted: int) -> None:
        with self._lock:
            self._verify_steps += 1
            self._proposed += proposed
            self._accepted += accepted

    def record_fallthrough(self, reason: str) -> None:
        with self._lock:
            self._fallthrough_steps += 1
            if reason in self._ft:
                self._ft[reason] += 1

    def record_error(self) -> None:
        """An error fallback IS a fallthrough step — count it as both."""
        with self._lock:
            self._errors += 1
            self._fallthrough_steps += 1
            self._ft["error"] += 1

    def record_cooldown_trip(self, level: int) -> None:
        with self._lock:
            self._cooldown_trips += 1
            self._backoff_level = level

    def set_state(self, current_k: int, backoff_level: int) -> None:
        with self._lock:
            self._current_k = current_k
            self._backoff_level = backoff_level

    def snapshot(self) -> dict[str, float | int]:
        """Consistent read of every counter under one lock acquisition."""
        with self._lock:
            proposed = self._proposed
            accepted = self._accepted
            snap: dict[str, float | int] = {
                "verify_steps": self._verify_steps,
                "fallthrough_steps": self._fallthrough_steps,
                "draft_tokens_proposed": proposed,
                "draft_tokens_accepted": accepted,
                # 0.0 rather than NaN when nothing has been proposed, so
                # the series stays scrapeable from the first scrape.
                "accept_ratio": (accepted / proposed) if proposed else 0.0,
                "cooldown_trips": self._cooldown_trips,
                "errors": self._errors,
                "current_k": self._current_k,
                "backoff_level": self._backoff_level,
            }
            for reason, count in self._ft.items():
                snap[f"ft_{reason}"] = count
            return snap

    def reset(self) -> None:
        """Tests only — see the module docstring on monotonicity."""
        with self._lock:
            self._verify_steps = 0
            self._fallthrough_steps = 0
            self._proposed = 0
            self._accepted = 0
            self._cooldown_trips = 0
            self._errors = 0
            self._ft = dict.fromkeys(self._FT_REASONS, 0)
            self._current_k = 0
            self._backoff_level = 0


_GLOBAL = SuffixAcceptCounter()


def get_global_counter() -> SuffixAcceptCounter:
    """The process-wide counter the scheduler writes and /metrics reads."""
    return _GLOBAL


def reset_global_counter() -> None:
    """Tests only."""
    _GLOBAL.reset()
