# Decision: Marvin's Garden — Bonsai 27B as the decision-model base

Date: 2026-09-19
Owner: Atlas
Status: MVP signal in hand; product path gated on `/v1/classify` review

## Decision

Build the Marvin's Garden decision model (routing / tool gating / injection
guarding, high-frequency small decisions) on **`bonsai-27b-2bit`**
(Ternary-Bonsai-27B, 2-bit, 13 GB resident) — not on a mid-size general
model. The base is already smarter than the class of models others
post-train for this job (Qwen3.5-9B, the Bespoke-Nimble base); our job is
to remove latency, not add intelligence: contrastive-SFT the base into a
single-forward label readout with **zero generated tokens**.

MVP signal (see
`docs/engineering/performance/2026-09-19-marvins-garden-mvp-signal.md`):
92.19% held-out accuracy / flip both-correct 85.4% / ECE 0.093 /
1.06 s/decision — above Nimble's published 90.12% under the same
methodology caveat (self-built eval, no standard benchmark in this class).

## Recipe (v1)

- Contrastive data curation: synthetic pairs differing in ONE audited
  `flip_key` whose mutation flips the label; labels derivable from stored
  scenario fields (unit-tested). No distillation, no probabilities in the
  labels, no RL. Deterministic generator, `schema.json` fail-closed
  validation, byte-reproducible from the committed seed.
- Training: `mlx_lm.lora` LoRA on the quantized base, `--mask-prompt`
  (load-bearing — see perf doc), LR 3e-5, 400 iters, 1088 train / 192
  held-out samples across 3 families.
- Serving: one prefill + letter-token softmax, program-side structured
  output. The evaluator (`eval_label_readout.py`) is the executable spec
  for the future `/v1/classify` endpoint.

## Alternatives considered

- **Qwen3.5-9B base (exact Nimble replica)** — rejected: smaller base,
  same recipe already beats its published number with the 27B; keeping the
  family in-house also keeps the alias-menu task self-consistent.
- **Smaller Bonsai (1.7B) for speed** — deferred: MVP first proves the
  recipe on the smartest base; a distilled 1.7B Marvin is a follow-up lane
  once `/v1/classify` exists.
- **CoT/rubric two-step decisions** — deferred to v2 ("look-then-commit");
  v1 deliberately commits in one forward pass.

## Consequences / risks

- 13 GB resident (vs 8.7 GB for a 9B): acceptable on the targeted tier,
  and the readout lane shares the resident-model machinery.
- ~1.06 s/decision is prefill-bound; the fixed menu prefix is
  prefix-cache-friendly. Nimble-style 100 ms class needs the serving work
  below, not a smaller base.
- **API change required**: `/v1/classify` is a public-server surface —
  needs review sign-off before implementation; interim path is
  `max_tokens=1` + `logprobs` with the known top-k failure mode.
- Self-built eval caveat (same as Nimble's published caveat): cross-family
  judge agreement is on the roadmap before any external claim.
- Adapter distribution (58 MB) is a release decision; nothing published
  without release-owner authorization.

## Next actions

1. Review: approve `/v1/classify` seam (public API).
2. Vector: serving-path speed work (prefix-cache the menu block, prompt
   tightening, resident decision lane) targeting sub-300 ms warm.
3. Data v2: grow routing family (81.3% is the weak family), add a 4th
   family (memory triage) + cross-family judge agreement.
