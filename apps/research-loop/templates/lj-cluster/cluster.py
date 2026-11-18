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


# Optional protocol-1 outputs; ordinary command-line use still prints RESULT.
if artifact_dir := os.environ.get("ARTIFACT_DIR"):
    import csv
    from pathlib import Path

    destination = Path(artifact_dir)
    positions = np.asarray(x, dtype=float).reshape(ATOMS, 3).tolist()
    with (destination / "coordinates.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x", "y", "z"])
        writer.writerows(positions)
    # Orthographic projection, stdlib only. Preserve aspect ratio and label units.
    xs, ys = [p[0] for p in positions], [p[1] for p in positions]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    scale = 300 / span
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    circles = "".join(
        f'<circle cx="{180 + (a - cx) * scale:.3f}" cy="{180 - (b - cy) * scale:.3f}" r="9" fill="#285e89"/>'
        for a, b, _ in positions
    )
    (destination / "cluster.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 390"><rect width="360" height="390" fill="white"/>'
        + circles
        + '<text x="180" y="380" text-anchor="middle" font-size="13" fill="#172b3b">Atom positions · x/y projection (sigma)</text></svg>'
    )
    (destination / "summary.md").write_text(
        f"Final cluster: {ATOMS} atoms.\nFinal energy: {final:.4f} (reduced units).\nEnergy evaluations: {_evals} of {EVAL_BUDGET}.\n"
    )
    declarations = [
        dict(
            kind="data",
            path="coordinates.csv",
            title="Final atom coordinates",
            role="coordinates",
            columns={
                axis: dict(type="number", unit="sigma") for axis in ("x", "y", "z")
            },
        ),
        dict(
            kind="image",
            path="cluster.svg",
            title="Cluster projection",
            role="structure",
        ),
        dict(kind="report", path="summary.md", title="Cluster summary", role="summary"),
    ]
    (destination / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in declarations)
    )
