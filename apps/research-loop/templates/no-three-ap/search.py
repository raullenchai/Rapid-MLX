"""Find a large set of integers in 0..N-1 that contains no three-term arithmetic progression (a, a+d, a+2d)."""

import json
import os
import random
import signal
import time

N = 243
BUDGET_S = 6.0


def search(rng, deadline):
    # Baseline: greedy, scanning numbers in order and keeping each one that creates no progression.
    chosen = []
    members = set()
    for x in range(N):
        if all((2 * y - x) not in members for y in chosen):
            chosen.append(x)
            members.add(x)
        if time.monotonic() > deadline:
            break
    report(chosen)
    return chosen


START = time.monotonic()
_best = 0
_best_candidate = []


def report(candidate):
    """Call whenever you have a valid candidate; it records progress for the live curve."""
    global _best, _best_candidate
    if len(candidate) > _best and is_valid(candidate):
        _best = len(candidate)
        _best_candidate = list(candidate)
        print(
            json.dumps(
                {"elapsed_s": round(time.monotonic() - START, 3), "size": _best}
            ),
            flush=True,
        )


def is_valid(s):
    s = sorted(set(s))
    m = set(s)
    if any(not (0 <= x < N) for x in s):
        return False
    return not any(
        2 * s[j] - s[i] in m for i in range(len(s)) for j in range(i + 1, len(s))
    )


rng = random.Random(int(os.environ.get("SEED", "0")))


def budget_expired(signum, frame):
    raise TimeoutError("search budget exhausted")


signal.signal(signal.SIGALRM, budget_expired)
signal.setitimer(signal.ITIMER_REAL, BUDGET_S)
try:
    result = search(rng, START + BUDGET_S)
except TimeoutError:
    result = _best_candidate
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
assert is_valid(result), "the returned set contains a three-term progression"
print(
    "RESULT "
    + json.dumps(
        {"size": len(set(result)), "seconds": round(time.monotonic() - START, 2)}
    ),
    flush=True,
)
