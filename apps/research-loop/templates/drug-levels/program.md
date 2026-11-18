# Model drug levels in the blood after a pill

After an oral dose of theophylline (an asthma drug), its concentration in the blood rises as the pill is absorbed and then
falls as the body eliminates it. We have 11 blood samples from one patient over 24 hours (a classic pharmacokinetics dataset).
The baseline fits a single exponential decay, which ignores absorption.

Goal: minimise `loo_rmse`, the error in mg/L when predicting each sample (after time 0) from a model fitted without it
(lower is better).

Rules:
- Edit only `fit(time, conc, dose)` (and helpers): it returns a function of time. numpy and scipy are available.
- Keep the data, the leave-one-out evaluation and the scoring unchanged. Use only the data passed in.
- One concrete change per experiment to test a hypothesis. The evaluation is deterministic; differences under 0.05 are noise.
