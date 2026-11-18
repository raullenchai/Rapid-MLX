"""Tiny CPU-only template. Requires only Python's standard library."""

import json
import os
import random
import time

SIZE = 2000
REPEATS = 20


def sort_values(values):
    result = []
    for value in values:
        index = 0
        while index < len(result) and result[index] < value:
            index += 1
        result.insert(index, value)
    return result


rng = random.Random(int(os.environ.get("SEED", "0")))
values = tuple(rng.randrange(100000) for _ in range(SIZE))
start = time.perf_counter()
for _ in range(REPEATS):
    result = sort_values(list(values))
runtime_ms = (time.perf_counter() - start) * 1000 / REPEATS
assert list(result) == sorted(values)
print("RESULT " + json.dumps({"runtime_ms": runtime_ms}), flush=True)
