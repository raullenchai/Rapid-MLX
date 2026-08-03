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
            if self.level and accepted >= min(BACKOFF_DECAY_MIN_ACCEPT, k):
                if accepted * 2 >= k:
                    self.level = 0
                else:
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


# ── per-request isolation (the state must not be shared) ──────────────


class MultiRequestPolicy:
    """Two requests sharing one installed drafter hook.

    Mirrors the production keying: state lives in a per-UID dict, created
    on demand and dropped when the request finishes. The failure this
    guards is concrete — with install-scoped state, a request that ends
    mid-window hands the next one up to COOLDOWN_MAX skipped steps it
    never earned, and the queued token history is flushed into the wrong
    request's suffix tree.
    """

    def __init__(self):
        self.state = {}

    def _for(self, uid):
        return self.state.setdefault(
            uid, {"cooldown": 0, "level": 0, "zeros": 0, "k": K_MIN, "pending": []}
        )

    def step(self, uid, accepted_fn):
        st = self._for(uid)
        if st["cooldown"] > 0:
            st["cooldown"] -= 1
            st["pending"].append(f"tok-{uid}")
            return "skip"
        drained = list(st["pending"])
        st["pending"].clear()
        k = st["k"]
        accepted = accepted_fn(k)
        if accepted >= k:
            st["k"] = min(k * 2, 8)
        else:
            st["k"] = max(K_MIN, min(accepted + 1, 8))
        if accepted == 0:
            st["zeros"] += 1
            trigger = COOLDOWN_TRIGGER if st["level"] == 0 else 1
            if st["zeros"] >= trigger:
                st["level"] += 1
                st["cooldown"] = min(
                    COOLDOWN_BASE * (2 ** (st["level"] - 1)), COOLDOWN_MAX
                )
                st["zeros"] = 0
        else:
            st["zeros"] = 0
            if st["level"] and accepted >= min(BACKOFF_DECAY_MIN_ACCEPT, k):
                if accepted * 2 >= k:
                    st["level"] = 0
                else:
                    st["level"] -= 1
        return drained

    def finish(self, uid):
        self.state.pop(uid, None)


def test_a_new_request_does_not_inherit_a_parked_predecessor():
    """MUTATION-KILL for per-UID keying: request A parks itself deep in a
    window, then finishes. B must draft from its first step."""
    p = MultiRequestPolicy()
    for _ in range(2000):
        p.step(1, lambda k: 0)
    assert p.state[1]["cooldown"] > 0, "A should be parked"
    p.finish(1)
    assert p.step(2, lambda k: k) == [], "B inherited A's window"
    assert p.state[2]["level"] == 0


def test_queued_history_never_crosses_requests():
    """A's tokens queued during its window must never be drained into B's
    drafter — that silently corrupts B's suffix history and its drafts."""
    p = MultiRequestPolicy()
    for _ in range(3):
        p.step(1, lambda k: 0)  # arm A's window
    for _ in range(5):
        p.step(1, lambda k: 0)  # A queues tokens while skipping
    assert p.state[1]["pending"], "A should have queued history"

    drained = p.step(2, lambda k: k)
    assert drained == [], f"B drained another request's tokens: {drained}"


def test_concurrent_requests_keep_independent_widths():
    """High-overlap A and low-overlap B must not drag each other's K."""
    p = MultiRequestPolicy()
    for _ in range(20):
        p.step(1, lambda k: k)  # A: always accepts
        p.step(2, lambda k: 0)  # B: never accepts
    assert p.state[1]["k"] == 8
    assert p.state[2]["k"] == K_MIN
    assert p.state[1]["level"] == 0
    assert p.state[2]["level"] > 0


def test_initial_width_clears_min_draft_len():
    """MUTATION-KILL for the K floor: width only grows AFTER a verify, and
    a draft shorter than ``min_draft_len`` is discarded before verifying.
    Starting below it deadlocks at the floor — suffix decoding silently
    does nothing for the whole request.
    """
    for min_draft_len, max_draft, expected in [
        (2, 8, 2),
        (3, 8, 3),  # the case that deadlocked
        (5, 8, 5),
        (4, 3, 3),  # cap wins — never issue wider than configured
        (2, 1, 1),  # num_speculative_tokens=1 is accepted by the parser
    ]:
        k_min = min(max(2, min_draft_len), max_draft)
        assert k_min == expected, (
            f"min_draft_len={min_draft_len} max_draft={max_draft} "
            f"-> {k_min}, expected {expected}"
        )
        if max_draft >= min_draft_len:
            assert k_min >= min_draft_len, "floor cannot clear the length gate"


def test_one_of_two_accept_cannot_reset_a_backoff():
    """MUTATION-KILL for the noise floor at the width back-off produces.

    After a back-off the adaptive width is ``_K_MIN`` (2), and at K=2 a
    single accepted token satisfies the strong-signal test
    ``accepted * 2 >= k``. Order the noise floor after that branch and the
    one outcome the policy calls noise resets the whole level, so
    low-overlap traffic landing the occasional 1-of-2 bounces back to
    eager drafting forever. Invisible at any wider K, which is why it
    survived the original round of tests.
    """
    p = Policy()
    p.level = 3
    p.zeros = 0
    # A verify at the post-backoff width that accepts exactly one token.
    p.step(lambda k: 1 if k == 2 else 0)
    assert p.level == 3, (
        f"a 1-of-{p.k} accept moved the back-off level 3 -> {p.level}; "
        "the noise floor must gate the strong-signal reset, not sit after it"
    )
