"""Find a large set of integers in 0..N-1 that contains no three-term arithmetic progression (a, a+d, a+2d)."""
import json
import os
import random
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
        if time.time() > deadline:
            break
    report(chosen)
    return chosen


START = time.time()
_best = 0


def report(candidate):
    """Call whenever you have a valid candidate; it records progress for the live curve."""
    global _best
    if len(candidate) > _best and is_valid(candidate):
        _best = len(candidate)
        print(json.dumps({"t": round(time.time() - START, 3), "size": _best}), flush=True)


def is_valid(s):
    s = sorted(set(s))
    m = set(s)
    if any(not (0 <= x < N) for x in s):
        return False
    return not any(2 * s[j] - s[i] in m for i in range(len(s)) for j in range(i + 1, len(s)))


rng = random.Random(int(os.environ.get("SEED", "0")))
result = search(rng, START + BUDGET_S)
assert is_valid(result), "the returned set contains a three-term progression"
print("RESULT " + json.dumps({"size": len(set(result)), "seconds": round(time.time() - START, 2)}), flush=True)
