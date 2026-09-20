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

## Open items

- Routing family at 81.3% — add groups and/or iters before product claims.
- Adapter (58 MB) stays local; publishing weights = release decision.
- No cross-model teacher was used (distillation-free, like Nimble).
