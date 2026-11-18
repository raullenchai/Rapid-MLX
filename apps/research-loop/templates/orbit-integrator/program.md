# Keep a simulated planet on its orbit

A planet orbits a star on an ellipse with eccentricity 0.6. We simulate 10 full revolutions with 4,000 time steps and at most
8,000 force evaluations. After exactly 10 periods the true planet is back where it started; numerical integrators drift away.
The baseline uses explicit Euler, the simplest method.

Goal: minimise `log_error` = log10 of the distance between the simulated and true final position (lower is better; -2 means
an error of 1%). The live curve shows log10 relative energy drift over time.

Rules:
- Edit only `step(q, p, dt)` (and helpers). Keep ECC, ORBITS, STEPS, FORCE_BUDGET, `force`, the start state, the loop and the scoring.
- Use `force(q)` for accelerations; do not compute gravity yourself or exceed the budget (2 calls per step on average).
- One concrete change per experiment to test a hypothesis. The simulation is deterministic; differences under 0.1 are noise.
