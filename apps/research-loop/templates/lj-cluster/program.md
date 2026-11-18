# Find the most stable shape of a 13-atom cluster

Thirteen atoms interact through the Lennard-Jones potential, the standard model of noble-gas atoms. The arrangement with the
lowest energy is the most stable one. The landscape has many local minima, so finding the global one is a classic test for
optimisation methods. The baseline starts from random positions and runs plain gradient descent.

Goal: minimise `energy` of the returned arrangement (lower is better), using at most 30,000 calls of `energy_and_grad`.

Rules:
- Edit only `optimize(rng)` (and helpers). Keep ATOMS, EVAL_BUDGET, `energy_and_grad`, and the final check. numpy and scipy are available.
- Every energy or gradient must come from `energy_and_grad`; scipy optimizers are fine if they call it.
- One idea per experiment. Each experiment runs 3 seeds; results are averaged. Differences under 0.3 are noise.
