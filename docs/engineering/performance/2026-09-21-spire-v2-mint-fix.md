# Spire v2 — mint-pipeline fix + state-probe recipe (2026-09-21, in progress)

## What broke in v1 (recap)

v1's 600 labels contained **zero Defend-optimal rows**: the margin≥1.0 filter
plus a uniform-HP sampler silently removed the entire defensive dimension,
so the model "correctly" learned a never-defend prior (38.8% final vs 33.7%
letter prior). Candidate letters were also correlated with hand-slot order
(`semantic_actions` preserves enumeration order).

## v2 mint changes (bench/marvins_garden/generate_spire.py, --v2)

1. **Defend-optimal scenario injection** (`--defend-pressure 0.30`): 30% of
   states are forced into the low-HP-vs-incoming-damage region
   (`bc.player.cur_hp ≈ incoming + [-3,+5]`), where blocking is the live
   strategic axis and rollout margins are naturally large.
2. **Candidate-order shuffling**: menu letters are decoupled from hand-slot
   order (candidates and option_lines shuffled together).
3. **Label-distribution gate**: hard-fails unless Defend-family ≥10% and no
   letter position holds >40% of labels (v2: Defend 34.1%, gate PASS).
4. **State pairs** (`spire_lethal` family): engine-oracle yes/no survival
   questions ("if you end your turn now, do the incoming attacks kill
   you?"), 300 pairs, balanced 150/150, letters shuffled. Usable as a
   diagnostic probe set and as mixing data.

Dataset: 900 train / 300 heldout action pairs (seed 20260922) + 300 state
pairs. Arm B = arm A + 225 state pairs mixed into training (75 held out).

## Results so far (same ruler: 300-item v2 heldout, think enabled)

| arm | action acc | ECE 15 | state probe (75) |
| --- | ---: | ---: | ---: |
| v15c release (baseline, same distribution) | 35.7% | — | — |
| **spire_v2a** (pipeline fix only) | **54.0%** | 0.082 | 53.3% (≈coin flip) |
| spire_v2b (+ state pairs mixed) | pending | — | — |

- Paired improvement vs baseline on identical items: **+18.3 pts** — the
  pipeline fix is confirmed, not a distribution artifact.
- State-probe accuracy 53.3% shows v2a never learned an explicit state
  representation; its gains come from better action-prior coverage.
  Arm B tests whether mixing oracle state pairs teaches it.

## Notes

- Training: v15c continuation, 300 it @ 1e-5, batch 2 (the only recipe that
  reached 48.2% in v1), via train_supervisor.sh — zero hangs this round.
- eval loads + 256-sample temperature calibration cost ~6 min per run;
  probe evals reuse the fitted T (--temperature) to skip recalibration.
- pair_id is now emitted by the generator (v1 needed a retrofit patch).
