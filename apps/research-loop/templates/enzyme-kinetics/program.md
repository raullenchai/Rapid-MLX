# Fit enzyme kinetics so the model predicts unseen measurements

An enzyme's reaction rate follows the Michaelis-Menten law v = Vmax * S / (Km + S). We have 12 classic measurements
(Treloar 1974, puromycin-treated). The baseline estimates Vmax and Km with the textbook Lineweaver-Burk double-reciprocal line.

Goal: minimise `loo_rmse`, the leave-one-out prediction error in rate units (lower is better): each point is predicted by a fit
that never saw it.

Rules:
- Edit only `fit(conc, rate)` (and helpers). Keep the data, `predict`, and the evaluation unchanged. numpy and scipy are available.
- `fit` must return (vmax, km) estimated from its inputs only.
- One concrete change per experiment to test a hypothesis. The evaluation is deterministic; differences under 0.5 are noise.
