# Large sets with no three-term arithmetic progression

Find as large a set as possible of integers in 0..242 that contains no three numbers a, a+d, a+2d (d > 0).
This is a classical problem in additive combinatorics (Salem–Spencer sets). The greedy baseline produces the
numbers whose base-3 digits avoid 2, which is valid but not the largest possible.

Goal: maximise `size`, the number of elements in the returned set (higher is better).

Rules:
- Edit only `search(rng, deadline)` and helpers you add. Keep N, BUDGET_S, `report`, `is_valid`, and the final checks unchanged.
- Return a list of integers; it is verified from scratch. Stop by the deadline (6 seconds).
- Call `report(candidate)` whenever you find a better valid set so progress can be plotted.
- One concrete change per experiment to test a hypothesis. Each experiment runs with 2 seeds; results are averaged. Differences under 0.5 are noise.
