# Trio-Flash plays Tetris (zero-shot OOD probe)

`play_tetris.py` drives a full Tetris loop through the production API
(`/v1/classify`): orientation decision (2-way) + landing-half decision (2-way)
per piece, game engine computes each option's outcome and the model makes the
semantic trade-off. Renders a replay GIF.

## Results (Studio Mac, mlx backend, 2026-09-22)

| Variant | Decisions | Latency p50 | Result |
|---|---|---|---|
| raw board state, 10-way column | 36 | 368ms | stacks one column, 0 lines |
| + strategy rules in prompt | 60 | 476ms | same prior, ignores rules |
| + per-option outcome features | 60 | 378ms | same prior, ignores features |
| training-shaped 2-way decisions | 18 | 294ms | conf 0.82–0.97 auto_decide, still fixed side |

**A/B flip probe** (identical situation, order swapped): the model picked
"A." both times (conf 0.51 = chance). Candidate text does not influence the
letter readout — zero-shot policy is position prior.

## Conclusions

1. **Mechanical layer is production-grade**: hierarchical decisions, rate-limit
   backoff, dispositions, ~300ms/decision — the API drives arbitrary discrete
   decision loops as-is.
2. **Policy layer does not transfer zero-shot.** v15c's readout is a letter
   classifier over trained task shapes; out-of-domain it collapses to a
   position prior while remaining *confident* (0.9+ auto_decide).
3. **Calibration does not catch task-level OOD.** ECE 0.031 holds in-domain;
   a novel task type reads as just another routing ticket. Near-OOD task
   probes (astra P1) are therefore required at the product layer, not just
   numeric confidence thresholds.
4. Making it *play well* requires an in-domain "game lane" LoRA (the training
   pipeline exists: auto-generate (board, best-move) pairs, ~2h GPU) — a
   separate task, not part of the v15c release scope.

## Run

```bash
python3 play_tetris.py --blocks 24 --gif replay.gif   # vs localhost:8123
```
