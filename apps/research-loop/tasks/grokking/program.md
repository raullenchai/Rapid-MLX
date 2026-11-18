# Research task: make a small network generalise (grok) on modular addition

A small MLP learns (a + b) mod 97 from 40% of all pairs. With the baseline settings it memorises the training set (train accuracy 100%) but test accuracy stays near 0% within the step budget: it never generalises.

Goal: raise **test accuracy at the end of a fixed 8,000-step budget** (higher is better). Secondary: `grok_step`, the first step at which test accuracy reaches 95% (lower is better).

Rules:
- You may edit anything in `train.py` except the task constants `P = 97` and `TRAIN_FRAC = 0.4`, the data split, the step budget, and the RESULT line.
- One idea per experiment: change one thing (or one tightly related group of things), so the result says something about the idea.
- Each experiment runs with 2 random seeds; results are averaged. Differences under 0.02 are noise.
