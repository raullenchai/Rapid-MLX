"""Simulate a planet on an eccentric orbit for 10 revolutions with a fixed budget of force evaluations."""
import json
import math

import numpy as np

ECC = 0.6
ORBITS = 10
STEPS = 4000
FORCE_BUDGET = 2 * STEPS
_calls = 0


def force(q):
    """Gravitational acceleration toward the origin (GM = 1). Each call counts against FORCE_BUDGET."""
    global _calls
    _calls += 1
    if _calls > FORCE_BUDGET:
        raise RuntimeError("force budget exceeded")
    r = np.linalg.norm(q)
    return -q / r**3


def step(q, p, dt):
    """Advance position q and velocity p by dt. Baseline: explicit Euler."""
    a = force(q)
    return q + dt * p, p + dt * a


def energy(q, p):
    return 0.5 * p @ p - 1 / np.linalg.norm(q)


# Start at perihelion of an orbit with semi-major axis 1; the exact period is 2*pi.
q = np.array([1 - ECC, 0.0])
p = np.array([0.0, math.sqrt((1 + ECC) / (1 - ECC))])
E0 = energy(q, p)
dt = ORBITS * 2 * math.pi / STEPS
for i in range(1, STEPS + 1):
    q, p = step(q, p, dt)
    if i % 40 == 0:
        print(json.dumps({"orbit": round(i / STEPS * ORBITS, 2), "drift": round(math.log10(abs((energy(q, p) - E0) / E0) + 1e-16), 3)}), flush=True)
# After a whole number of periods the exact solution is back at the starting point.
position_error = float(np.linalg.norm(q - np.array([1 - ECC, 0.0])))
print("RESULT " + json.dumps({"log_error": round(math.log10(position_error + 1e-16), 3), "energy_drift": float(abs((energy(q, p) - E0) / E0)), "force_calls": _calls}), flush=True)
