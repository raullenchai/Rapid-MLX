# OOD abstention — first measurements (2026-09-21)

**Question (posed by external review):** does low in-distribution ECE (0.031)
actually buy reliable out-of-distribution refusal? Answer: **yes, with the
caveat that our probe is far-OOD.**

## Setup

- Adapter: marvins-garden-v15c (routing/tool_gate/injection_guard), single
  forward, letter-probability readout, think-mode enabled.
- In-dist: 192 held-out items (the same-ruler set). Acc 95.31%, ECE 0.029.
- OOD probe: 600 Slay-the-Spire card-choice prompts (a game domain the
  adapter never trained on; different menu shape, 4–8 candidates). The probe
  is deliberately FAR from the business distribution.
- Tool: `scripts/ood_abstention_analysis.py`; dumps
  `results/v15c_indist_dump.jsonl`, `results/v15c_spire_ood_dump.jsonl`.

## Results

| | in-dist (n=192) | OOD-spire (n=600) |
| --- | --- | --- |
| confidence p50 | 0.994 | 0.582 |
| confidence p10 | 0.897 | 0.429 |
| accuracy | 0.953 | 0.293 |

**AUROC (confidence separates OOD from in-dist): 0.971.**

Risk–coverage on in-dist (abstain lowest-confidence first):

| coverage | residual error |
| --- | --- |
| 100% | 4.7% |
| 95% | 3.3% |
| 90% | 1.7% |
| 80% | 0.7% |
| 70% | 0.7% |
| 60% | 0.9% |
| 50% | 0.0% |

Abstaining the least-confident 10% cuts the error rate 4.7% → 1.7% (−64%).

Operating thresholds:

| threshold | OOD refused | in-dist refused |
| --- | --- | --- |
| conf < 0.70 | 71.3% | 3.1% |
| conf < 0.80 | 84.8% | 5.2% |
| conf < 0.90 | 94.2% | 10.4% |

## Caveats (carried from the external review)

1. The probe is FAR-OOD (a game). **Near-OOD probes are not yet minted**:
   ambiguous intent, missing candidates, tool-argument anomalies, injection
   embedded in normal content — the cases that look like normal traffic but
   should not be auto-decided. Until those exist, treat 0.971 as an upper
   bound on what far-OOD detection buys.
2. Spire menus have more candidates (4–8) than routing (2–6), so part of the
   confidence gap may reflect menu-size entropy, not OOD-ness per se. A
   menu-size-matched OOD probe would disentangle this.
3. Thresholds must be re-fit on a validation split and reported on a frozen
   test split with per-family coverage + dangerous-error rates (esp.
   injection_guard mis-allows).

## Where this lands in the plan

This was the "OOD abstention bet". First evidence is positive: the same
single forward that produces the decision also produces a confidence that
rank-orders OOD inputs at 0.971 AUROC — no extra model, no extra pass. Next:
mint near-OOD probes, fit per-family thresholds, add "none of the above"
candidate tests (candidate-relative softmax blind spot).

## Same-day addendum: eval scale doubled (external review action ①)

Fresh-seed eval set (384 items, seed 20260921, disjoint states):
**95.31% accuracy, ECE 0.029, 1.21 s/decision** — identical to the 192-item
result (`results/eval_v15c_large.json` + dump). Per-family:
injection_guard 1.00 · tool_gate 0.948 · model_routing 0.932 (n=128/192/64).
The headline number is now reproduced on two independent draws. Remaining
review actions: paired significance test vs Jev on per-item outcomes, cheap
baselines (rules / small model / base without LoRA), near-OOD minting.
