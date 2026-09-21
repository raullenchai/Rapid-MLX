# Marvin vs Jev vs Nimble — a fair comparison

Numbers below are either **measured by us on our ruler** (192-sample
held-out set, committed and byte-deterministic) or **published self-reports
by each model's owners**, and the table says which is which. Everything
comes from `bench/marvins_garden/results/*.json` with reproducible
commands in the perf log.

## The table

| Dimension | **Marvin v15c** (ours) | **Jev** (`jev-latest`) | **Nimble** |
| --- | --- | --- | --- |
| Accuracy, same ruler (measured by us) | **94.44% ± 1.08%** (n=3 full replications; best run 95.31%) | 93.23% | *not measurable — no access* |
| Published self-report | — | 93.21% ✓ agrees with our measurement | 90.12% ⚠ different ruler, never verified by us |
| Calibration (ECE 15-bin, same ruler) | **0.019–0.031** | 0.102 | unknown |
| Latency per decision | ~1.1 s on M3 Ultra (local); est. 50–150 ms on a cloud GPU (single forward, not yet measured) | **0.19 s p50** (hosted API incl. RTT) | unknown |
| Deployment | **runs anywhere the weights fit** — 13.5 GB 2-bit base + 58 MB adapter; local, on-prem, air-gapped | hosted API only | hosted API only |
| Decision cost | one forward pass, zero generated tokens; batchable | same shape (SystemOne), metered tokens | unknown |
| Interface | 3 deep lanes; `/v1/classify` spec exists (endpoint pending review) | **productized API**: named questions, choice/noul/score, multi-question batching, usage metering | unknown |
| Task breadth | deep in its lanes; new families are cheap to mint with the contrastive generator, but out-of-distribution tasks degrade to base (40% zero-shot on our set) | **broad classification face** (tone, billing, spam …) | unknown |
| Openness | **full pipeline, data, evals, decision logs, RC adapter committed; 14/14 tests** | closed API | closed API |
| Per-family (same ruler) | routing 91.7–94.8% · tool gate 89.6–93.8% · guard 100% | routing 87.5% · tool gate **97.9%** · guard 100% | — |

## How to read this fairly

1. **Same-ruler holds only for Marvin vs Jev.** We ran `jev-latest` on our
   192 items with identical information and identical scoring
   (`demos/../bench` cross-eval script). Nimble has no public eval we could
   obtain; its 90.12% is a self-report on its own ruler and is **not**
   comparable to the same-ruler column.
2. **The ruler was built by us** — home advantage is possible. Two
   mitigations: held-out items were never trained on, and Jev's score on
   our ruler (93.23%) matches its published 93.21% almost exactly, so the
   ruler does not systematically favor or punish Jev. Even so, treat
   sub-2-point gaps with humility.
3. **Latency is not hardware-comparable.** Marvin's 1.1 s is local inference
   on one M3 Ultra; Jev's 0.19 s is their hosted edge. The product decision
   (owner, 2026-09-20) is ~1 s intelligence-first — we do not race hosted
   few-hundred-ms products on latency.
4. **Where each wins.** Marvin: accuracy under policy constraints
   (routing +7.3 pts), calibration 3–5× better (real thresholds and
   abstention), deployment freedom (air-gapped, on-device, per-seat cost ≈
   electricity). Jev: latency as a service, API maturity, and breadth —
   if your task is generic classification outside our lanes, Jev today is
   the safer call. Nimble: unverified on our ruler; on published numbers
   only it likely sits below both, but we claim nothing we did not measure.
5. **License gate.** Marvin's base is prism-ml's Ternary-Bonsai-27B; any
   hosted commercial offering requires a license review before launch
   (release-owner decision).

## Reproduce

```bash
# same-ruler Jev cross-eval (key via env only)
JEVAI_KEY=... python bench/marvins_garden/cross_eval_jev.py

# our numbers (see perf doc for the full matrix and training recipe)
python bench/marvins_garden/eval_label_readout.py --model <snapshot> \
  --adapter adapters/release/marvins-garden-v15c --temperature 1.0 --think-mode enabled
```
