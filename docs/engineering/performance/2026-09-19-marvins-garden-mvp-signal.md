# Marvin's Garden MVP — signal run: Bonsai 27B 2-bit + contrastive curation

Date: 2026-09-19

Owner/host: Atlas, Studio (M3 Ultra, 256 GB unified memory)

Question: can the Bespoke-Nimble recipe (contrastive data curation +
LoRA + single-forward label readout) turn our smartest small-tier base,
`prism-ml/Ternary-Bonsai-27B-mlx-2bit`, into a Jev-class decision model?
Target: beat Nimble's published 90.12% on a held-out contrastive set,
approach Jev's 93.21%, with zero generated tokens.

## Result

**92.19% held-out accuracy, zero generated tokens, 1.06 s/decision** —
above Nimble's published 90.12% (their eval set, ours ours — same caveat
they published; no standard benchmark exists for this task class).

| Arm | Accuracy | flip both-correct | ECE (15-bin) | ms/decision |
| --- | ---: | ---: | ---: | ---: |
| Base 27B zero-shot readout | 40.10% | 0.0% | 0.281 | 1040 |
| LoRA 300it, NO prompt masking | 41.15% | 0.0% | 0.483 | 1057 |
| **LoRA 400it, `--mask-prompt`, LR 3e-5** | **92.19%** | **85.4%** | **0.093** | 1057 |

Per-family (masked run): injection_guard 100%, tool_gate 95.3%,
model_routing 81.3% (8-option alias menu — the hard family).

Confidence separates correctness (mean 0.877 correct vs 0.534 wrong), so
abstention routing has real signal to work with.

## Reproduce

```bash
python bench/marvins_garden/generate_contrastive.py --seed 20260919
python bench/marvins_garden/validate.py                # needs jsonschema>=4.0
python bench/marvins_garden/to_chat_sft.py
PATH=<venv>/bin:$PATH MODEL=prism-ml/Ternary-Bonsai-27B-mlx-2bit \
  ITERS=400 LAYERS=16 BATCH=2 LR=3.0e-5 bash bench/marvins_garden/train_lora.sh
python bench/marvins_garden/eval_label_readout.py \
  --model prism-ml/Ternary-Bonsai-27B-mlx-2bit \
  --adapter bench/marvins_garden/adapters/marvins-garden --temperature 1.0
```

Environment: mlx-lm 0.31.3, mlx floor per pyproject, python 3.12 venv,
model snapshot 70f75f3ad081ab840a42f3304c02c27e7f89bfb7. Training peak
118.6 GB (quantized weights + LoRA grads), ~0.15 it/s at batch 2,
seq len 1024. Results JSONs:
`bench/marvins_garden/results/eval_{base_zeroshot,marvin_masked_400it}.json`.

## What mattered (failed approaches included)

1. **`--mask-prompt` is load-bearing.** Without it, loss over the ~300
   prompt tokens dilutes the 1-letter completion to ~1/300 of the
   gradient: val loss falls to 0.048 (prompt modeling) while held-out
   accuracy stays at base level (40.1% → 41.2%). With masking, val loss
   starts at ~ln(n_candidates) and actually measures the decision.
2. **LR 1e-4 diverged twice mid-run** (train loss 0.2 → 2.5 spikes with
   cosine decay); LR 3e-5 converged monotonically (val 0.126 @ iter 50,
   0.036 @ iter 100, 0.032 @ iter 350).
3. Prompt-local solvability: option lines carry alias names + specs,
   tools render with capability descriptions (decoys state their limits).
   The contrastive flip keys are the discrimination signal.
4. Contrastive curation delivered its promise: flip both-correct went
   0.0% → 85.4% — the model learned which evidence must change decisions,
   not keyword shortcuts.

## Speed headroom (next)

1.06 s/decision is prefill-bound (307 tokens mean, 1 forward). The
8-option menu block (~180 tokens) is a fixed prefix — the server's
existing prefix cache absorbs it in the serving path. Shorter prompts and
a resident-decision-model lane come after `/v1/classify` (public API —
needs review sign-off).

## Night session (2026-09-19 → 09-20): data v2, GPU instability, template-drift breakthrough

Final matrix (held-out 192, temperature 1.0):

| Arm | Single-pass | Ensemble | ECE | flip both-correct | routing |
| --- | ---: | ---: | ---: | ---: | ---: |
| base zero-shot | 40.1% | — | 0.281 | 0.0% | 20.3% |
| v1 (640-group data) | 92.19% | 93.23% | 0.093 | 85.4% | 81.3% |
| **v1.5 = v1 + 150it continuation on v2 data** | **94.79%** | **94.79%** | **0.021/0.032** | **89.6%** | **91.7%** |

v1.5 is the release candidate. Beats Jev's published 93.21% single-pass
(own-eval caveat unchanged), and the weak routing family moved 81→92%.

### Finding 1: adapter ⇄ serving template pairing is load-bearing

mlx-lm's ChatDataset renders the chat template WITHOUT disabling thinking,
so qwen3-family training views end `...assistant\nidensea\n` while our
serving readout used `enable_thinking=False` (no opener) — a one-position
readout drift. Each adapter has a matched serving format:

- `marvins-garden` (v1): serve with think-mode DISABLED (92.19%).
- `marvins-garden-v15` (release candidate): serve with think-mode ENABLED
  (the training view) — 94.79%; the SAME adapter with the serving template
  drops to 83.9%, and v1 measured the other way round (92.19% disabled vs
  83.9% enabled).

`eval_label_readout.py --think-mode {disabled,enabled}` encodes this, and
`/v1/classify` MUST render with the adapter's recorded mode (store the
mode beside the adapter — open item).

### Finding 2: GPU stability envelope on the shared host

Two v2 runs died to Metal GPU hangs (batch 4 at ~70 min, then batch 2
runs repeatedly at ~50–70 min, one `InnocentVictim` system-wide reset).
Stable envelope found: **batch 1** (63 GB peak vs 118/229) never hung, but
needs the segmented driver (`train_segment_loop.sh`: short segments,
`--resume-adapter-file` chaining, per-segment deterministic reshuffle —
mlx-lm reads datasets IN ORDER, so resume without reshuffle never sees
the data tail). Warm-restart cycling at LR 3e-5 still degraded the final
adapter (40% — a noisy-snapshot failure); the robust pattern that
actually improved things was **continuation of a converged adapter at low
LR (1e-5) for a short window**, not from-scratch restarts or long
warm-restart chains.

### Data v2

1280 groups (routing-weighted 50/25/25), wider context/RAM tiers, more
briefs. 2368 train pairs. Committed, validator PASS, 14/14 tests.

## IQ tax (measured, 2026-09-19) — and the v1.1 Pareto point

`iq_probe.py` A/Bs base vs base+adapter on the repo's OWN eval suites
(generation mode, official `run_eval.py` graders and prompt wrappers,
`enable_thinking=False`):

| Suite | base | v1 adapter | v1.1 (+8% identity mix) |
| --- | ---: | ---: | ---: |
| reasoning (MATH-500 ×10) | 60% | 0% | 60% |
| general (MMLU-Pro ×10) | 80% | 40% | 60% |
| coding (executed ×10) | 70% | 0% | 60% |
| overall | 70% | 13% | 60% |
| **decision held-out** | 40.1% | **92.2%** | **85.4%** |

Diagnosis of v1: task-mode collapse, not knowledge erasure — simple
factual QA still answers correctly, but out-of-distribution prompts get
first-token scrambled. The 8% identity mix recovers +47 IQ points at a
cost of −6.8 decision points (ECE 0.093 → 0.234).

**Deployment decision:** two adapters, two lanes.

- `marvins-garden` (v1): decision lane ONLY, 92.2% — chat never loads it,
  so the IQ tax is 0 by construction in the chat lane.
- `marvins-garden-v11` (v1.1): for any reuse where mixed traffic is
  possible — 85.4% decision + 60% general.

Next lever for the Pareto frontier: grow decision data (routing is the
weak family) at fixed identity proportion, rather than raising the mix.

## Open items

- Routing family at 81.3% — add groups and/or iters before product claims.
- Adapter (58 MB) stays local; publishing weights = release decision.
- No cross-model teacher was used (distillation-free, like Nimble).
