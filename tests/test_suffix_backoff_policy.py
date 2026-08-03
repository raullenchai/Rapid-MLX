# SPDX-License-Identifier: Apache-2.0
"""The SuffixDecoding backoff / adaptive-width policy, in isolation.

The policy lives inside a closure in ``_install_suffix_decoding`` and is
only reachable through a live BatchGenerator, so these tests re-implement
the exact state machine and assert on its behaviour. That keeps the
intent pinned even though the production copy is not directly importable:
if someone changes the constants or the update rules, the numbers below
are what they have to re-justify.

Why the policy exists at all (measured on gemma-4-12b-4bit):

* Fixed K=8 with the old fixed-10 cooldown cost **-32%** on free-form
  generation — acceptance there is 0.044, and a verify forward is ~3.4x a
  plain one on M3 Pro, so the drafter burned budget it never recovered.
* The same configuration is worth **+119%** on code-edit traffic, where
  acceptance is 0.823.

The policy has to keep the second without paying the first.
"""

import pytest

# Mirrors the production constants in vllm_mlx/scheduler.py.
COOLDOWN_TRIGGER = 3
COOLDOWN_BASE = 10
COOLDOWN_MAX = 320
BACKOFF_DECAY_MIN_ACCEPT = 2
K_MIN = 2


class Policy:
    """Faithful re-implementation of the closure's state machine."""

    def __init__(self, max_draft=8):
        self.max_draft = max_draft
        self.k = K_MIN
        self.level = 0
        self.zeros = 0
        self.cooldown = 0
        self.trips = 0

    def step(self, accepted_fn):
        """One decode step. ``accepted_fn(k)`` returns tokens accepted.

        Returns ``"verify"`` or ``"skip"``.
        """
        if self.cooldown > 0:
            self.cooldown -= 1
            return "skip"
        k = self.k
        accepted = accepted_fn(k)

        # Adaptive width.
        if accepted >= k:
            self.k = min(self.k * 2, self.max_draft)
        else:
            self.k = max(K_MIN, min(accepted + 1, self.max_draft))

        # Backoff.
        if accepted == 0:
            self.zeros += 1
            trigger = COOLDOWN_TRIGGER if self.level == 0 else 1
            if self.zeros >= trigger:
                self.level += 1
                self.cooldown = min(
                    COOLDOWN_BASE * (2 ** (self.level - 1)), COOLDOWN_MAX
                )
                self.zeros = 0
                self.trips += 1
        else:
            self.zeros = 0
            if self.level:
                if accepted * 2 >= k:
                    self.level = 0
                elif accepted >= BACKOFF_DECAY_MIN_ACCEPT:
                    self.level -= 1
        return "verify"


# ── high-overlap: must never back off ─────────────────────────────────


def test_high_overlap_never_backs_off_and_reaches_max_width():
    """Traffic that accepts everything must draft on every step and widen
    to the cap — that is where the +119% comes from."""
    p = Policy()
    outcomes = [p.step(lambda k: k) for _ in range(200)]
    assert set(outcomes) == {"verify"}
    assert p.trips == 0
    assert p.level == 0
    assert p.k == 8


def test_width_doubles_toward_the_cap_not_past_it():
    p = Policy()
    widths = []
    for _ in range(6):
        widths.append(p.k)
        p.step(lambda k: k)
    assert widths == [2, 4, 8, 8, 8, 8]


# ── low-overlap: must go quiet, and fast ──────────────────────────────


def test_low_overlap_converges_to_almost_no_drafting():
    """MUTATION-KILL for the exponential growth: with the old fixed-10
    window this ratio sits near 0.23 (3 wasted verifies per 13 steps),
    which is the -32% regression."""
    p = Policy()
    outcomes = [p.step(lambda k: 0) for _ in range(2000)]
    verifies = outcomes.count("verify")
    assert verifies / len(outcomes) < 0.02, (
        f"{verifies} verifies in 2000 steps — backoff is not converging"
    )


def test_backoff_window_doubles_then_caps():
    p = Policy()
    windows = []
    for _ in range(12):
        # Drive to the next trip, recording the window it set.
        while True:
            if p.step(lambda k: 0) == "verify" and p.cooldown:
                windows.append(p.cooldown + 1)  # +1: this step consumed none yet
                break
    assert windows[0] == COOLDOWN_BASE + 1
    assert windows[1] == COOLDOWN_BASE * 2 + 1
    assert windows[2] == COOLDOWN_BASE * 4 + 1
    assert max(windows) <= COOLDOWN_MAX + 1


def test_first_trip_tolerates_a_brief_stumble():
    """One or two misses inside accepting traffic must NOT arm a window —
    otherwise a momentary miss costs a high-overlap request 10 steps."""
    p = Policy()
    p.step(lambda k: k)  # establish signal
    p.step(lambda k: 0)
    p.step(lambda k: 0)
    assert p.cooldown == 0, "armed a window after only two misses"


# ── recovery: the case that broke first ───────────────────────────────


def test_strong_signal_resets_backoff_immediately():
    """A deep window plus decay-by-one could not recover: measured 22.8
    tok/s against 33.2 for identical work, because the request never got
    enough drafting chances to walk the level back down. A half-or-better
    accept therefore resets outright."""
    p = Policy()
    for _ in range(2000):
        p.step(lambda k: 0)
    assert p.level > 0
    p.cooldown = 0  # window expires, we get one probe
    p.step(lambda k: k)  # full accept
    assert p.level == 0, "strong signal must return to eager drafting"


def test_weak_signal_relative_to_width_only_decays_one_level():
    """ "Strong" is measured against the CURRENT width, not an absolute
    count. At k=8, two accepted tokens is weak — it decays one level
    rather than resetting. Crediting weak accepts fully kept low-overlap
    traffic oscillating back to eager (measured: 8 trips, level back to
    0, still -24%).
    """
    p = Policy()
    for _ in range(2000):
        p.step(lambda k: 0)
    p.level = 5
    p.k = 8  # a width where 2 accepted is genuinely weak
    p.cooldown = 0
    p.step(lambda k: 2)
    assert p.level == 4, "weak-but-real signal should decay exactly one level"


def test_single_token_accept_is_noise_and_credits_nothing():
    """Below ``BACKOFF_DECAY_MIN_ACCEPT`` and below half-width: neither
    reset nor decay."""
    p = Policy()
    p.level = 5
    p.k = 8
    p.cooldown = 0
    p.step(lambda k: 1)
    assert p.level == 5, "a 1-of-8 accept must not credit the traffic"


def test_mixed_traffic_recovers_within_one_probe():
    """A chat that starts emitting a repeated code block: parked deep,
    then the next probe lands fully. It must be drafting again right
    away, not after another climb."""
    p = Policy()
    for _ in range(2000):
        p.step(lambda k: 0)
    p.cooldown = 0
    p.step(lambda k: k)
    outcomes = [p.step(lambda k: k) for _ in range(50)]
    assert set(outcomes) == {"verify"}


@pytest.mark.parametrize("one_in", [0, 50, 20])
def test_weak_traffic_of_any_shape_stays_cheap(one_in):
    """Not just the all-zero case: anything materially below break-even
    acceptance must end up mostly skipping. ``one_in=0`` means never
    accept; otherwise one verify in ``one_in`` lands fully."""
    p = Policy()
    calls = {"n": 0}

    def accepted(k):
        calls["n"] += 1
        if one_in and calls["n"] % one_in == 0:
            return k
        return 0

    outcomes = [p.step(accepted) for _ in range(2000)]
    verify_share = outcomes.count("verify") / len(outcomes)
    assert verify_share < 0.25, f"drafting on {verify_share:.0%} of steps"
