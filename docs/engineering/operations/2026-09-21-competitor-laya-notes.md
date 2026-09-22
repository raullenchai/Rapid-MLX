# Competitor FYI — Laya (brainfunctioncollapse.com/laya), 2026-09-21

Owner-flagged. **Laya: "open-source alternative to TypeSafe Jev, runs
locally" — 322M params, 650 MB, 21 ms/decision on M1 Max, Apache-2.0, zero
generated tokens, typed answers with probabilities.** Playground + Flappy/
Runner/Tetris live demos + "Laya vs Jev, measured" benchmark + a SKILL.md
for coding agents (curl-one-line install). Same six use-case patterns as our
families (route/guard/score/filter/watch/cascade). Author: Nandakishor M,
Convai Innovations; playground/benchmark/skill by the site author.

## Honest read

- **This is the "cheap baseline" the external review demanded — already
  productized.** It concedes "Jev is more accurate out of the box" and sells
  speed × locality × fine-tunability × privacy. If its accuracy on OUR eval
  is close to v15c, the 27B value proposition needs the paired proof.
- **Demo design is ahead of ours in one specific way**: the games ask for the
  STATE ("where is the bird") not the ACTION ("which way to move") — "asked
  which way to move, every checkpoint answered backwards". They also route
  arithmetic to code and hand the model conclusions in words, test wording
  sensitivity explicitly (0.75 vs 0.45 across phrasings), and show death/failure
  honestly (Tetris tops out ~28 rows/s; a life lasts 95–120 s).
- **Speed class is incomparable by design**: 322M vs 27B. Their 21 ms and our
  1.2 s are different weight classes; the comparison that matters is accuracy
  on complex, multi-candidate, long-fact decisions — exactly where a 322M
  model should fall off. That must be measured, not assumed.

## Immediate actions (all free)

1. **Same-ruler eval of Laya on our 384-item held-out set** (open weights;
   CPU-fast). This closes the "cheap baseline" gap in COMPARISON.md.
2. Adopt "ask for the state, not the action" + wording-robustness checks in
   the spire v2 mint (alongside candidate shuffling + defend scenarios).
3. Steal the honest-failure pattern for our playground (a "where it fails"
   section) and the SKILL.md distribution channel for agents.
4. Positioning: NOT "open-source Jev alternative" (taken) but "complex
   decisions with real constraints, calibration, and overnight custom lanes —
   the 322M class cannot read long facts". Prove with the eval above.

## Same-ruler measurement (2026-09-21, `convaiinnovations/laya` 0.3.5)

384-item held-out set, literal mapping (state = our full prompt, criteria =
candidate texts). Machine: M3 Ultra, CPU/MPS, 14 s total.

| family | Laya covered-acc | Marvin v15c |
| --- | --- | --- |
| injection_guard (96) | 0.562 | 1.000 |
| tool_gate (96) | 0.552 | 0.948 |
| model_routing (192) | 0.062 | 0.932 |
| **macro** | **0.310** | **0.9531** |

Coverage 384/384 (its choice head returns probabilities over a CLOSED option
vocabulary — invented candidates matched or hallucinated in). Routing at 6.2%
is worse than the 33.9% letter prior: it defaults to model names from its own
training distribution regardless of the policy text in the state.

**Caveat (honesty):** the literal mapping is not Laya-favorable — its
criteria interface expects short phrase options; hand-written semantic
criteria per family could lift guard/gate. Routing semantics (policy → model)
cannot be saved by rewording: the model ignored the policy. Conclusion
stands in the owner's framing: Laya is faster and simpler, we are slower and
(same-ruler) much more accurate. Its sellable regime — fixed short options at
21 ms — is a different product; ours is arbitrary menus + policy reasoning.

Dump: bench/marvins_garden/results/laya_large_dump.jsonl
