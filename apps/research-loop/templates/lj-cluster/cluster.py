"""Find the lowest-energy arrangement of 13 argon-like atoms (Lennard-Jones cluster) with a fixed evaluation budget."""
import json
import os

import numpy as np

ATOMS = 13
EVAL_BUDGET = 30000
_evals = 0
_best = float("inf")


def energy_and_grad(x):
    """Lennard-Jones energy (epsilon = sigma = 1) and its gradient for flat coordinates x. Each call counts once."""
    global _evals, _best
    _evals += 1
    if _evals > EVAL_BUDGET:
        raise RuntimeError("evaluation budget exceeded")
    pos = x.reshape(ATOMS, 3)
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = (diff**2).sum(-1) + np.eye(ATOMS)
    inv6 = (1 / r2) ** 3
    e = float(np.sum(np.triu(4 * (inv6**2 - inv6), 1)))
    coef = (-48 * inv6**2 + 24 * inv6) / r2
    np.fill_diagonal(coef, 0)
    grad = (coef[:, :, None] * diff).sum(1)
    if e < _best:
        _best = e
        if _evals % 50 == 0 or e < -40:
            print(json.dumps({"evals": _evals, "energy": round(e, 4)}), flush=True)
    return e, grad.ravel()


def optimize(rng):
    """Return flat coordinates. Baseline: random start, plain gradient descent with a fixed step."""
    x = rng.uniform(-1.2, 1.2, ATOMS * 3)
    while _evals < EVAL_BUDGET - 1:
        e, g = energy_and_grad(x)
        x = x - 0.0005 * np.clip(g, -100, 100)
    return x


rng = np.random.default_rng(int(os.environ.get("SEED", "0")))
x = optimize(rng)
_evals -= 1  # the final check below is free
final, _ = energy_and_grad(np.asarray(x, dtype=float))
print("RESULT " + json.dumps({"energy": round(final, 4), "evals": _evals}), flush=True)
